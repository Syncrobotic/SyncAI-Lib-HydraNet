"""Interleaved A/B for train.channels_last.

    uv run python scripts/bench_channels_last.py

Committed because the first measurement of this flag was not. A -33% figure went into
the notes, the script that produced it was never kept, and the number could not be
reproduced or explained afterwards -- the same failure as a config default nobody ever
ran, one directory over.

Alternates the two arms rather than running A then B, so that anything else on the card
lands on both equally instead of on whichever went second. Read the median and the min
together: when they diverge, something else is using the GPU and the min is the closer
estimate; when they agree, the card is yours.

The loss is a synthetic scalar over the outputs rather than the real multi-task loss.
The question is which convolution kernels run, and fabricating targets would only add a
term both arms pay. It does mean the absolute milliseconds here are not comparable to a
timing taken around a real training step.
"""

import statistics
import sys
import time

import torch

from syncai_hydranet.config import load_config
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.seeding import apply_channels_last

CFG = "configs/hydranet_indoor.yaml"
BATCH, H, W = 48, 512, 640
WARMUP, ITERS, ROUNDS = 8, 15, 3


def build(channels_last: bool):
    cfg = load_config(CFG, ["model.backbone.pretrained=false"])
    model = build_model(cfg).cuda().train()
    apply_channels_last(model, channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, opt


def loss_of(out):
    total = 0.0
    for v in out.values():
        for t in v if isinstance(v, (list, tuple)) else [v]:
            total = total + t.float().mean()
    return total


def timed(model, opt, x, n):
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_of(model(x))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    arms = {}
    for channels_last in (False, True):
        model, opt = build(channels_last)
        fmt = torch.channels_last if channels_last else torch.contiguous_format
        x = torch.randn(BATCH, 3, H, W, device="cuda").contiguous(memory_format=fmt)
        timed(model, opt, x, WARMUP)  # cuDNN autotuning happens here
        arms[channels_last] = (model, opt, x, [])

    for r in range(ROUNDS):
        for channels_last in (False, True):
            model, opt, x, samples = arms[channels_last]
            samples.extend(timed(model, opt, x, ITERS))
        print(f"  round {r + 1}/{ROUNDS} done", file=sys.stderr)

    nchw = statistics.median(arms[False][3])
    nhwc = statistics.median(arms[True][3])
    print(f"\nbatch {BATCH} at {H}x{W}, bf16 autocast, {ROUNDS * ITERS} steps each")
    print(f"  NCHW  {nchw:7.1f} ms   (min {min(arms[False][3]):.1f})")
    print(f"  NHWC  {nhwc:7.1f} ms   (min {min(arms[True][3]):.1f})")
    print(f"  delta {(nhwc / nchw - 1) * 100:+6.1f}%")


if __name__ == "__main__":
    main()
