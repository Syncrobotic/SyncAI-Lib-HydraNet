"""Where this box stops being GPU-bound: a batch-size and worker-count sweep.

    uv run python scripts/bench_batch_size.py --config configs/hydranet_retail.yaml

Committed for the same reason as ``bench_channels_last.py``: the shipped
``batch_size: 8`` was chosen on a much smaller card, and "8 is too small for a Blackwell"
is a guess until something measures it. It was measured, and the answer was not the one
the guess implied -- see the config comments this produced.

**Two independent limits, measured separately, because tuning against the wrong one is
the whole trap.** A training step can be slow because the GPU is busy, or because the
loader cannot feed it. They look identical from the outside -- low img/s -- and they
have opposite fixes. Raising the batch on a loader-bound run buys nothing at all.

  phase 1  model only, synthetic tensors, no loader.  The compute ceiling.
  phase 2  loader only, real dataset, no model.       The supply ceiling.

The useful batch size is the smallest one that reaches the compute ceiling, and the
useful worker count is the smallest that clears it. Anything past either is memory and
CPU spent on nothing.

Note the loss is a synthetic scalar over the outputs rather than the real multi-task
loss, for the reason ``bench_channels_last.py`` gives: the question is which kernels run.
Absolute milliseconds here are therefore not comparable to a real training step.

Nothing on the card but this. If a training run is in flight, pause it first --
``kill -STOP <pid>`` and ``kill -CONT <pid>`` -- rather than reading contended numbers.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.data.datasets import build_dataset  # noqa: E402
from syncai_hydranet.data.multitask import collate  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.seeding import apply_channels_last  # noqa: E402

WARMUP, ITERS = 5, 12


def loss_of(out) -> torch.Tensor:
    total = 0.0
    for v in out.values():
        for t in v if isinstance(v, list | tuple) else [v]:
            total = total + t.float().mean()
    return total


def bench_compute(cfg, batch: int, channels_last: bool, compile_mode: str | None) -> dict:
    """Steady-state ms/step for one batch size, loader excluded."""
    h, w = cfg["data"]["input_size"]
    model = build_model(cfg).cuda().train()
    apply_channels_last(model, channels_last)
    fwd = torch.compile(model, mode=compile_mode) if compile_mode else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, 3, h, w, device="cuda")
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    def step():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_of(fwd(x))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(WARMUP):  # includes graph capture when compiling
        step()
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    peak = torch.cuda.max_memory_allocated() / 2**30
    ms = statistics.median(times)
    # Drop the references so `empty_cache` has something to reclaim before the next
    # batch size is built -- rebinding rather than `del`, because `step` closes over
    # these names and deleting them leaves the closure referring to unbound locals.
    # It happens to work today only because nothing calls `step` after this point.
    model = fwd = opt = x = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"ms": ms, "ips": 1000 * batch / ms, "gib": peak}


def bench_loader(cfg, batch: int, workers: int, batches: int = 30) -> float:
    """Images per second the input pipeline alone can deliver."""
    from torch.utils.data import DataLoader

    ds = build_dataset(
        cfg["data"]["datasets"][0],
        cfg["data"]["input_size"],
        "train",
        letterbox=bool(cfg["data"].get("letterbox", False)),
        augment=cfg["data"].get("augment"),
    )
    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    it = iter(dl)
    for _ in range(3):  # let the worker pool spin up
        next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in range(batches):
        next(it)
        n += batch
    dt = time.perf_counter() - t0
    del it, dl
    return n / dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/hydranet_retail.yaml")
    ap.add_argument("--batches", type=int, nargs="*", default=[8, 16, 32, 48, 64])
    ap.add_argument("--workers", type=int, nargs="*", default=[4, 8, 12, 16])
    ap.add_argument("--loader-batch", type=int, default=32)
    ap.add_argument("--compile", action="store_true", help="also time each batch compiled")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = load_config(args.config, ["model.backbone.pretrained=false"])
    h, w = cfg["data"]["input_size"]
    print(f"{torch.cuda.get_device_name(0)}  {h}x{w}  bf16 autocast\n")

    print("phase 1: compute ceiling (model only, no loader)")
    print(
        f"  {'batch':>5} {'NCHW img/s':>11} {'NHWC img/s':>11} {'NHWC ms':>9} {'peak GiB':>9}"
    )
    compute = {}
    for b in args.batches:
        try:
            a = bench_compute(cfg, b, channels_last=False, compile_mode=None)
            c = bench_compute(cfg, b, channels_last=True, compile_mode=None)
        except torch.OutOfMemoryError:
            print(f"  {b:>5}   out of memory")
            torch.cuda.empty_cache()
            continue
        compute[b] = c
        print(f"  {b:>5} {a['ips']:>11.1f} {c['ips']:>11.1f} {c['ms']:>9.1f} {c['gib']:>9.1f}")

    if args.compile:
        print("\n  compiled (NHWC, mode=default)")
        for b in args.batches:
            if b not in compute:
                continue
            try:
                r = bench_compute(cfg, b, channels_last=True, compile_mode="default")
            except (torch.OutOfMemoryError, RuntimeError) as e:
                print(f"  {b:>5}   {type(e).__name__}")
                continue
            gain = 100 * (r["ips"] / compute[b]["ips"] - 1)
            print(f"  {b:>5} {r['ips']:>11.1f} img/s  ({gain:+.0f}% vs eager)")

    print(f"\nphase 2: supply ceiling (loader only, batch {args.loader_batch})")
    print(f"  {'workers':>7} {'img/s':>9}")
    for nw in args.workers:
        print(f"  {nw:>7} {bench_loader(cfg, args.loader_batch, nw):>9.1f}", flush=True)

    print(
        "\nRead them together. The batch to use is the smallest that reaches the compute\n"
        "plateau; the worker count is the smallest whose supply exceeds it. A loader below\n"
        "the compute line means the GPU idles and a larger batch changes nothing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
