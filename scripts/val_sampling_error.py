#!/usr/bin/env python3
"""How much of a val score is the val set, and how many images would fix it.

    python3 scripts/val_sampling_error.py runs/hydranet_retail_security_b03_cw

`ARCHITECTURE.md` section 4 rule 4 records that three seeds of the same config
land 0.0196 apart in `terrain_mIoU/site_seg`, and that pairing does not shrink it. That
number is measured with **every seed scored on the same 48 images**, so it is model-side
variance by construction and a larger val set cannot touch it -- which is the argument
against annotating more, and it is only half true.

The other half is what this measures. A per-class IoU over 48 images is a ratio of pixel
sums, and `column` contributes 1.5% of those pixels. Two models that differ slightly can
land far apart on such a ratio simply because the few objects carrying the class fall one
way or the other. That amplification *is* a property of the val set, and it is what shrinks
when the set grows.

So: resample the val images with replacement, recompute the metric, and report the spread.
If it lands near the seed spread, the instrument is the limiting factor and annotation buys
real resolution. If it lands far below, the variance is genuinely in the models and more
images would be an expensive way to measure nothing.

`--target-sd` then sizes the job, on `sd ~ 1/sqrt(n)`. That extrapolation assumes the new
images resemble the old ones, which is exactly what adding *cameras* rather than frames
breaks -- read the printed per-camera spread before trusting it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from site_confusion import per_image_confusions  # noqa: E402

from syncai_hydranet.utils.device import pick_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run")
    ap.add_argument("--split", default="val")
    ap.add_argument("--dataset", default="datasets/retail_objects_batch03")
    ap.add_argument("--label-map", default="retail_surfaces_from_objects")
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--target-sd",
        type=float,
        default=0.005,
        help="sd of mIoU the val set should reach; sizes the annotation job",
    )
    ap.add_argument(
        "--unit",
        default="image",
        choices=["image", "camera"],
        help="resample images, or whole cameras -- the second is the honest unit if the "
        "claim is about cameras rather than about these cameras",
    )
    return ap


def miou(mats: np.ndarray) -> tuple[float, np.ndarray]:
    """mIoU over the non-void classes, matching what training logs.

    Training reports `terrain_mIoU/<set>` as the unweighted mean of the per-class IoUs with
    class 0 (`void`) dropped, each IoU computed on summed pixel counts rather than averaged
    per image. Reproduced here rather than imported so the bootstrap cannot silently diverge
    from the number it is explaining.
    """
    c = mats.sum(0) if mats.ndim == 3 else mats
    ious = []
    for i in range(1, c.shape[0]):
        denom = c[i].sum() + c[:, i].sum() - c[i, i]
        ious.append(c[i, i] / denom if denom else np.nan)
    per = np.array(ious, dtype=float)
    return float(np.nanmean(per)), per


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = pick_device(None)
    mats, classes, paths = per_image_confusions(Path(args.run), args, device)
    names = classes[1:]
    cams = np.array([p.parts[-2].split("__")[0] for p in paths])

    point, per = miou(mats)
    print(f"\n{Path(args.run).name} on {args.dataset}/{args.split}")
    print(f"  {len(paths)} images, {len(set(cams))} cameras, {mats.sum():,} labelled pixels")
    cols = "  ".join(f"{n[:7]} {v:.3f}" for n, v in zip(names, per, strict=True))
    print(f"  mIoU {point:.4f}   " + cols)

    share = mats.sum(0).sum(1) / mats.sum()
    print(
        "\n  pixel share: "
        + "  ".join(f"{n[:7]} {share[i + 1]:.2%}" for i, n in enumerate(names))
    )

    rng = np.random.default_rng(args.seed)
    groups = (
        [np.flatnonzero(cams == c) for c in sorted(set(cams))]
        if args.unit == "camera"
        else None
    )
    draws = np.empty((args.bootstrap, 1 + len(names)))
    for b in range(args.bootstrap):
        if groups is None:
            idx = rng.integers(0, len(mats), len(mats))
        else:
            picked = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[g] for g in picked])
        m, p = miou(mats[idx])
        draws[b] = [m, *p]

    sd = np.nanstd(draws, axis=0, ddof=1)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    unit = "images" if args.unit == "image" else "cameras"
    print(f"\n  resampling {unit}, {args.bootstrap} draws")
    print(f"    {'':10s}{'sd':>8s}{'2.5%':>9s}{'97.5%':>9s}")
    for i, n in enumerate(["mIoU", *names]):
        print(f"    {n[:10]:10s}{sd[i]:8.4f}{lo[i]:9.4f}{hi[i]:9.4f}")

    n_now = len(groups) if groups is not None else len(mats)
    need = n_now * (sd[0] / args.target_sd) ** 2
    print(
        f"\n  mIoU sd {sd[0]:.4f} at {n_now} {unit}"
        f" -> {args.target_sd} needs about {need:.0f} {unit}"
        f" ({need / n_now:.1f}x)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
