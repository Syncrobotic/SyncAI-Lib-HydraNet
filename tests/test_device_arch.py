"""A torch build that cannot launch a kernel on this card is caught before it launches one.

The failure this covers is quiet at every point a person would look. `torch.cuda
.is_available()` returns True, `get_device_capability()` returns the right card, and
`nvidia-smi` is fine; the build simply carries no cubin for the architecture, and the
first kernel launch is where that surfaces. On this project's Blackwell box that is
several minutes into a training run.

Nothing pins which CUDA build arrives. `uv.lock` pins torch by hash, so a
`uv sync --locked` is reproducible today, but the wheel PyPI serves for a torch release
carries whatever architectures that release was built for -- and a version bump can
change the set while the lockfile diff shows only a version and a hash.

**The arithmetic is tested here, not the box.** `cuda_arch_is_supported` takes the arch
list and the capability as arguments precisely so this file runs on CI, which has no GPU:
a guard whose only test skips everywhere it is run is the shape `test_figures_are_audited`
was written to avoid. The one test below that does need a GPU asserts the live box agrees
with the arithmetic, and skips honestly when there is nothing to ask.
"""

from __future__ import annotations

import pytest
import torch

from syncai_hydranet.utils.device import cuda_arch_is_supported

BLACKWELL = (12, 0)
# What `torch.cuda.get_arch_list()` returns for torch 2.13.0+cu130, the build this
# project runs on, read off the box on 2026-08-31.
CU130 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
# The same list one release earlier, before sm_120 was added -- the regression shape.
NO_BLACKWELL = ["sm_75", "sm_80", "sm_86", "sm_90"]


def test_the_shipped_build_covers_the_card_this_project_runs_on():
    assert cuda_arch_is_supported(CU130, BLACKWELL)


def test_a_build_without_the_card_is_refused():
    """The whole point: this is the list that runs, reports itself healthy, and dies."""
    assert not cuda_arch_is_supported(NO_BLACKWELL, BLACKWELL)


def test_embedded_ptx_for_an_older_arch_is_forward_compatible():
    """`compute_90` JIT-compiles onto sm_120. Slow, but it runs, so refusing it is wrong."""
    assert cuda_arch_is_supported([*NO_BLACKWELL, "compute_90"], BLACKWELL)


def test_ptx_newer_than_the_card_does_not_help():
    """JIT goes forward only. PTX for sm_120 cannot be run by an sm_86 card."""
    assert not cuda_arch_is_supported(["sm_120", "compute_120"], (8, 6))


@pytest.mark.parametrize(
    ("entry", "capability"),
    [("sm_90a", (9, 0)), ("sm_120a", (12, 0)), ("sm_100f", (10, 0))],
)
def test_an_architecture_specific_suffix_names_the_same_card(entry, capability):
    """`sm_120a` is a feature set on sm_120 hardware, not a thirteenth architecture.

    Parsing the letter as part of the number is how a first draft of this crashed:
    `int("a")` on a list torch is entitled to return. The capabilities are written out
    rather than derived from the string, which is how the first draft of *this* test was
    wrong in turn -- it read `sm_90a` as card (90, 0) and failed a correct function.
    """
    assert cuda_arch_is_supported([entry], capability)


@pytest.mark.parametrize("arch_list", [[], ["gfx90a", "gfx942"], ["nonsense"]])
def test_a_list_this_cannot_read_is_not_a_refusal(arch_list):
    """CPU-only builds report `[]` and ROCm reports `gfx*`.

    Answering "unsupported" there would refuse builds this function has no opinion about
    -- a guard firing on a question it cannot see is worse than no guard.
    """
    assert cuda_arch_is_supported(arch_list, BLACKWELL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device to ask")
def test_the_installed_build_can_run_on_the_card_it_found():
    """The one test that needs the hardware, and the reason the rest of this file exists.

    It is not asserting a fixed list: a different box legitimately has a different card
    and a different build. It asserts they match, which is the property that stops being
    true silently.
    """
    arch_list = torch.cuda.get_arch_list()
    capability = torch.cuda.get_device_capability()
    assert cuda_arch_is_supported(arch_list, capability), (
        f"installed torch {torch.__version__} lists {arch_list} and this box has a "
        f"sm_{capability[0]}{capability[1]} card. A kernel launch will fail."
    )
