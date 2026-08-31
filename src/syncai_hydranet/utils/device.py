"""Device selection: CUDA (Jetson / servers) -> MPS (Apple Silicon) -> CPU.

On Apple Silicon, MPS is roughly 7x faster than CPU for this model (measured on an
M4 Pro: RegNetX-800MF at 384x512, batch 4, 705 ms/step on CPU vs 97 ms/step on MPS),
so local development must not silently fall back to CPU.

AMP and pinned memory are CUDA-only today, so those checks live here too rather than
being scattered across the trainer.

`pick_device` also refuses a CUDA build that has no kernels for the card it found, which
is a failure this project can otherwise only discover at the first kernel launch --
`torch.cuda.is_available()` and `get_device_capability()` both answer correctly on a build
with nothing to run. Nothing in the dependency set states which CUDA build arrives:
`uv.lock` pins torch by hash, so a `uv sync --locked` is reproducible, but the wheel PyPI
serves for a given torch version carries whatever architectures that release was built
for, and a version bump can change the set without changing anything a lockfile diff
shows. Pinning an index would not close this -- the index is per-project, the build set is
per-release -- so the check is on the installed artefact instead.
"""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the fastest available device, or ``prefer`` ('cuda' / 'mps' / 'cpu')."""
    if prefer:
        device = torch.device(prefer)
        if device.type == "cuda" and torch.cuda.is_available():
            _refuse_a_build_with_no_kernels()
        return device
    if torch.cuda.is_available():
        _refuse_a_build_with_no_kernels()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _refuse_a_build_with_no_kernels() -> None:
    """Raise, rather than fall back to CPU, when the installed torch cannot run here.

    A silent fall-back is the failure this exists to prevent: on this project's box the
    difference between a build with `sm_120` and one without is the difference between a
    training run and a run that is 40x slower and reports nothing unusual about itself.
    """
    arch_list = torch.cuda.get_arch_list()
    capability = torch.cuda.get_device_capability()
    if cuda_arch_is_supported(arch_list, capability):
        return
    raise RuntimeError(
        f"torch {torch.__version__} was built for {arch_list or '(nothing)'} and this is a "
        f"sm_{capability[0]}{capability[1]} card ({torch.cuda.get_device_name()}). CUDA "
        "reports the device fine and the first kernel launch is where this would otherwise "
        "fail. Install a build that lists this architecture, then re-run; pass "
        "pick_device('cpu') to work without the GPU."
    )


def cuda_arch_is_supported(arch_list: list[str], capability: tuple[int, int]) -> bool:
    """Can a build advertising `arch_list` launch a kernel on a card of `capability`?

    Split out from `pick_device` and taking both as arguments so it is testable on a box
    with no GPU, which is every box CI has.

    Two ways to be supported. A matching `sm_NM` is a compiled cubin and the normal case.
    A `compute_XY` entry is embedded PTX, which the driver JIT-compiles forward onto any
    newer architecture -- slow on first launch, but it runs, so treating it as unsupported
    would refuse a build that works.

    The trailing letter in `sm_90a` and `sm_120a` is dropped rather than parsed. Those
    are architecture-specific feature sets, not a different card: a build listing `sm_120a`
    targets the same sm_120 hardware `get_device_capability` reports as (12, 0), and the
    letter never reaches the numeric form the capability comes in.

    Returns True when the list says nothing this function understands (empty on a CPU-only
    build, `gfx*` on ROCm): the question then is not "does this fail" but "was it asked",
    and a guard that answers a question it cannot see is worse than no guard.
    """
    sms, ptx = set(), []
    for entry in arch_list:
        kind, _, digits = entry.partition("_")
        digits = digits.rstrip("abcdefghijklmnopqrstuvwxyz")
        if kind not in ("sm", "compute") or len(digits) < 2 or not digits.isdigit():
            continue
        arch = (int(digits[:-1]), int(digits[-1]))
        if kind == "sm":
            sms.add(arch)
        else:
            ptx.append(arch)
    if not sms and not ptx:
        return True
    return capability in sms or any(p <= capability for p in ptx)


def supports_amp(device: torch.device) -> bool:
    """GradScaler and autocast are only fully supported on CUDA."""
    return device.type == "cuda"


def supports_pinned_memory(device: torch.device) -> bool:
    """MPS does not support pinned memory; setting it only emits a warning."""
    return device.type == "cuda"


def supports_bfloat16(device: torch.device) -> bool:
    """bf16 needs Ampere or newer; pre-Ampere CUDA cards emulate it, slowly.

    Asked separately from `supports_amp` because the answer decides whether the
    configured `amp_dtype` is honoured or refused, and a silent downgrade to fp16 is
    exactly the kind of thing that would explain a run's numbers months later.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.is_bf16_supported()
