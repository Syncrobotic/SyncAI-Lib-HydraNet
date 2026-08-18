#!/usr/bin/env python3
"""Turn site clips into an annotation batch the model has already had a go at.

    python3 scripts/annotation_batch.py --config configs/hydranet_retail.yaml \\
        --checkpoint runs/hydranet_retail_base/best.pt \\
        --out datasets/retail_batch01 --frames 40 assets/archive_*.mp4

Writes the `seg_folder` layout `hydranet-annotation check --scheme retail` expects: one
session directory per clip, images beside pre-labelled masks. Annotators correct the mask
instead of drawing it, which is several times faster on the classes the model already
half-knows and no slower on the ones it does not.

Two things this does that a naive "dump every frame" does not, and both change how much
annotation a euro buys.

**Frames are chosen by how different they are, not by a clock.** A fixed CCTV camera
produces 900 nearly identical frames in five minutes. Labelling 40 of those at random
gets you perhaps three distinct scenes, and a train/val split over them measures
memorisation. Farthest-point sampling over a coarse image descriptor picks frames that
disagree with each other, which is what the model has not seen yet.

**Where the model is unsure, the pre-label is `ignore`, not a guess.** A confident wrong
pre-label is worse than a blank one: it is the class an annotator is most likely to accept
without looking. Holes are visible; a plausible mistake is not. `--confidence` sets the
line.

The pre-labels are a starting point and nothing more. Nothing here should be trained on
until a human has been over it -- which is exactly what the annotation contract in
docs/ANNOTATION_SETUP.md is for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.data.video import frames, probe  # noqa: E402  # isort: skip
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.data.frame_selection import describe, farthest_first  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.preprocessing import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402

IGNORE = 255


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--weights",
        choices=["ema", "model"],
        default="ema",
        help="EMA weights need enough training steps to be meaningful; see docs/TRAIN_MACOS.md",
    )
    ap.add_argument("--out", required=True, help="dataset root to write")
    ap.add_argument("--split", default="train")
    ap.add_argument("--frames", type=int, default=40, help="frames to keep per clip")
    ap.add_argument("--sample-fps", type=float, default=1.0, help="rate to consider frames at")
    ap.add_argument(
        "--confidence",
        type=float,
        default=0.75,
        help="softmax below this becomes ignore in the pre-label, so an annotator sees a "
        "hole rather than a confident guess",
    )
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(select_weights(ckpt, args.weights))
    size = cfg["data"]["input_size"]
    classes = cfg["data"].get("terrain_classes", [])

    root = Path(args.out)
    summary = {"clips": [], "checkpoint": args.checkpoint, "confidence": args.confidence}
    totals = np.zeros(256, dtype=np.int64)

    for clip in args.clips:
        session = Path(clip).stem
        img_dir = root / "images" / args.split / session
        ann_dir = root / "annotations" / args.split / session
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        w, h, _ = probe(clip)
        kept, descs = [], []
        for frame in frames(clip, w, h, args.sample_fps):
            kept.append(frame.copy())
            descs.append(describe(frame))
        picks = farthest_first(descs, args.frames)
        print(f"{session}: {len(kept)} candidates -> {len(picks)} kept")

        for n, idx in enumerate(picks):
            frame = kept[idx]
            img = Image.fromarray(frame)
            small = img.resize((size[1], size[0]), Image.Resampling.BILINEAR)
            arr = (np.asarray(small, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(arr.transpose(2, 0, 1))[None].to(device)
            with torch.no_grad():
                logits = model(x)["terrain"]
                prob = logits.softmax(1)
                conf, pred = prob.max(1)
            mask = pred[0].to(torch.uint8).cpu().numpy()
            # Unsure means unsure. A confident wrong pre-label is the one an annotator
            # accepts without looking; a hole is the one they fill.
            mask[conf[0].cpu().numpy() < args.confidence] = IGNORE
            mask_img = Image.fromarray(mask).resize(img.size, Image.Resampling.NEAREST)

            stem = f"{n:04d}"
            img.save(img_dir / f"{stem}.jpg", quality=92)
            mask_img.save(ann_dir / f"{stem}.png")
            totals += np.bincount(np.asarray(mask_img).reshape(-1), minlength=256)

        summary["clips"].append({"session": session, "source": clip, "frames": len(picks)})

    labelled = totals[: len(classes)].sum()
    print(f"\nwrote {root}")
    print("pre-label composition (share of labelled pixels; ignore is what needs drawing)")
    for i, name in enumerate(classes):
        if totals[i]:
            print(f"  {i:>3} {name:<20} {100 * totals[i] / max(labelled, 1):6.2f}%")
    unsure = 100 * totals[IGNORE] / max(totals.sum(), 1)
    print(f"  ignore (unsure or unlabelled): {unsure:.1f}% of all pixels")
    summary["pre_label_share"] = {
        classes[i]: round(float(totals[i] / max(labelled, 1)), 4)
        for i in range(len(classes))
        if totals[i]
    }
    (root / "batch.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\nNext:")
    print("  hydranet-annotation labels --scheme retail --out cvat_labels_retail.json")
    print(f"  hydranet-annotation check {root} --scheme retail")
    print("Correct the masks; do not train on them until a human has been over every frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
