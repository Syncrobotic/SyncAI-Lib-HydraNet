"""Overlay predictions on a single image or a folder of images.

    hydranet-infer-image --config configs/hydranet_indoor.yaml \
        --checkpoint runs/hydranet_indoor/best.pt --input photo.jpg --output out/

Writes ``<stem>_pred.jpg``: traversability overlay with detection boxes on the left,
terrain overlay on the right.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ..config import load_config
from ..data.coco_subsets import COCO_NAMES, retail_box_label
from ..models.heads.detection import SCORE_THR_VIEW
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint, select_weights
from ..utils.device import pick_device
from ..utils.visualize import (
    TRAV_COLORS,
    crop_box,
    overlay,
    preprocess,
    terrain_palette,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-infer-image", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--output", default="out")
    ap.add_argument(
        "--weights",
        choices=["ema", "model"],
        default="ema",
        help="EMA weights need enough training steps to be meaningful; see docs/TRAIN_MACOS.md",
    )
    ap.add_argument("--score-thr", type=float, default=SCORE_THR_VIEW)
    ap.add_argument(
        "--nms-thr",
        type=float,
        default=0.6,
        help="IoU above which two boxes of one class are treated as one detection. "
        "Raise it for crowded scenes, where shoppers overlap and the tighter default "
        "merges two people into one. Note the audit's evidence for missed people points "
        "at the SCORE threshold, not at this: on Taichung-cam01 person detections went "
        "62 -> 72 -> 112 as the score cut fell 0.35 -> 0.25 -> 0.15. This knob is "
        "exposed because it was hard-coded, not because it was measured",
    )
    ap.add_argument(
        "--vocab",
        choices=["coco", "retail"],
        default="coco",
        help="how to name a box. retail reads COCO's answer as a shop noun -- "
        "product/book, fixture/refrigerator -- which is a rename of the same head's "
        "same output, not extra knowledge; see data/coco_subsets.py",
    )
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(select_weights(ckpt, args.weights))

    in_path = Path(args.input)
    paths = sorted(in_path.glob("*")) if in_path.is_dir() else [in_path]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg["data"]["input_size"]
    use_lb = bool(cfg["data"].get("letterbox", False))
    terrain_colors = terrain_palette(cfg["data"].get("terrain_classes"))
    # Boxes used to be drawn as `73:0.42`. The id is the least useful of the three
    # things known about a detection, and scene.py already resolved it to a name.
    name_of = (
        retail_box_label
        if args.vocab == "retail"
        else (lambda i: COCO_NAMES[i] if i < len(COCO_NAMES) else str(i))
    )

    for p in paths:
        if p.suffix.lower() not in IMG_EXTS:
            continue
        x, canvas, region = preprocess(Image.open(p), size, use_lb)
        with torch.no_grad():
            result = model.predict(x.to(device), score_thr=args.score_thr, nms_thr=args.nms_thr)

        x0, y0, cw, ch = region
        terr = crop_box(result["terrain"][0].cpu().numpy(), region)
        vis_img = canvas.crop((x0, y0, x0 + cw, y0 + ch))
        vis_terr = overlay(vis_img, terr, terrain_colors)
        # A model may legitimately have no traversability head: it is a lookup on
        # terrain, and the object-segmentation configs drop it. Ask the result rather
        # than assuming, so this CLI keeps working on both -- it used to index
        # `result["traversability"]` outright and would raise KeyError with nothing in
        # the message about heads or configs.
        vis_trav = (
            overlay(
                vis_img,
                crop_box(result["traversability"][0].cpu().numpy(), region),
                TRAV_COLORS,
            )
            if "traversability" in result
            else None
        )
        # Boxes go on whichever panel is leftmost, so they are never drawn twice or
        # into a panel that is not written.
        box_panel = vis_trav if vis_trav is not None else vis_terr

        det = result.get("detection", [{}])[0]
        if det:
            draw = ImageDraw.Draw(box_panel)
            for box, score, label in zip(
                det["boxes"].cpu().numpy(),
                det["scores"].cpu().numpy(),
                det["labels"].cpu().numpy(),
                strict=True,
            ):
                b = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
                if b[2] <= 0 or b[3] <= 0 or b[0] >= cw or b[1] >= ch:
                    continue  # entirely inside the letterbox padding
                draw.rectangle(b, outline=(255, 255, 255), width=2)
                draw.text(
                    (b[0] + 2, b[1] + 2),
                    f"{name_of(int(label))} {score:.2f}",
                    fill=(255, 255, 255),
                )

        if vis_trav is not None:
            combined = Image.new("RGB", (vis_trav.width * 2, vis_trav.height))
            combined.paste(vis_trav, (0, 0))
            combined.paste(vis_terr, (vis_trav.width, 0))
            legend = "left: traversability + boxes, right: terrain"
        else:
            combined = vis_terr
            legend = "terrain + boxes"
        out_file = out_dir / f"{p.stem}_pred.jpg"
        combined.save(out_file, quality=92)
        print(f"{p.name} -> {out_file}  ({legend})")


if __name__ == "__main__":
    main()
