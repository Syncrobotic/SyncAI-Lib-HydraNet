#!/usr/bin/env python3
"""Where a terrain head's pixels actually go, on the site val split, as a confusion matrix.

    python3 scripts/site_confusion.py runs/hydranet_retail_security_b03 \\
        runs/hydranet_retail_security_b03_cw

A per-class IoU says a class is bad. It does not say what the model said instead, and on
this project the difference decided the fix. `column` at IoU 0.094 reads as "needs more
data"; the matrix says **86.2% of it comes back as `wall`**, and `wall` recalls 97.9% of
itself while scoring 0.566 -- one class eating its neighbours, which is frequency bias and
is answered by `SegLoss.class_weights` rather than by annotation. batch03 had already
given `column` 3.24% of pixels across 105 of 224 masks; supervision was not what it
lacked.

`SegLoss`'s own docstring records the same signature one class earlier: "One class
absorbing and the rest under-firing is the signature of frequency bias, and it survived
dice_weight 1.5." That was `fixture` in the batch02 run. This is the tool for reading it.

Two runs may be given, and then the second is printed as deltas against the first, which
is the form a decision gets made from.

**The labels are SAM 3's**, so this measures agreement with a pre-label and not accuracy;
`retail_objects_batch02/split.json` says the same of its own boxes. What it is good for is
*where* two models differ, which is a comparison between two things measured the same way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))


from syncai_hydranet.engine.confusion import (  # noqa: E402
    per_image_confusions,
    run_config,
)
from syncai_hydranet.utils.device import pick_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="run directories; the first is the reference")
    ap.add_argument("--split", default="val")
    ap.add_argument("--dataset", default="datasets/retail_objects_batch02")
    ap.add_argument("--label-map", default="retail_surfaces_from_objects")
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--checkpoint", default="best.pt")
    return ap


def confusion(run: Path, args, device) -> tuple[np.ndarray, list[str]]:
    return per_image_confusions(
        run,
        checkpoint=args.checkpoint,
        weights=args.weights,
        label_map=args.label_map,
        dataset=args.dataset,
        split=args.split,
        device=device,
    )[0].sum(0), run_config(run)["data"]["terrain_classes"]


def iou(c: np.ndarray, i: int) -> float:
    denom = c[i].sum() + c[:, i].sum() - c[i, i]
    return float(c[i, i] / denom) if denom else float("nan")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = pick_device(None)
    mats = []
    for r in args.runs:
        c, classes = confusion(Path(r), args, device)
        mats.append((Path(r).name, c, classes))

    for name, c, classes in mats:
        print(f"\n{name}   rows = label, columns = prediction")
        print(
            f"{'':10s}" + "".join(f"{x[:8]:>9s}" for x in classes) + f"{'IoU':>8s}{'share':>8s}"
        )
        for i, cls in enumerate(classes):
            if not c[i].sum():
                continue
            row = "".join(f"{v:8.1%} " for v in c[i] / c[i].sum())
            print(f"{cls:10s}{row}{iou(c, i):8.3f}{c[i].sum() / c.sum():8.1%}")

    if len(mats) > 1:
        base_name, base, classes = mats[0]
        for name, c, _ in mats[1:]:
            print(f"\n{name} minus {base_name}, IoU per class")
            for i, cls in enumerate(classes):
                if not base[i].sum():
                    continue
                a, b = iou(base, i), iou(c, i)
                print(f"  {cls:10s} {a:6.3f} -> {b:6.3f}   {b - a:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
