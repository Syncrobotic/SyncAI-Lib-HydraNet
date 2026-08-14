"""Overlay predictions on a single image or a folder of images.

    hydranet-infer-image --config configs/hydranet_indoor.yaml \
        --checkpoint runs/hydranet_indoor/best.pt --input photo.jpg --output out/

Writes ``<stem>_pred.jpg``: traversability overlay with detection boxes on the left,
terrain overlay on the right.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..config import load_config
from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD
from ..models.heads.detection import SCORE_THR_VIEW
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint
from ..utils.device import pick_device
from ..utils.visualize import (
    TRAV_COLORS,
    crop_box,
    letterbox,
    overlay,
    terrain_palette,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def preprocess(img: Image.Image, size, use_letterbox: bool = False):
    """Return ``(tensor, canvas, region)``.

    ``region`` locates the real content inside the canvas so predictions can be cropped
    back and the output keeps the source aspect ratio instead of carrying grey bars.
    """
    h, w = size
    img = img.convert("RGB")
    if use_letterbox:
        img, region = letterbox(img, size)
    else:
        img, region = img.resize((w, h), Image.BILINEAR), (0, 0, w, h)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))[None], img, region


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-infer-image", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--output", default="out")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR_VIEW)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])

    in_path = Path(args.input)
    paths = sorted(in_path.glob("*")) if in_path.is_dir() else [in_path]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg["data"]["input_size"]
    use_lb = bool(cfg["data"].get("letterbox", False))
    terrain_colors = terrain_palette(cfg["data"].get("terrain_classes"))

    for p in paths:
        if p.suffix.lower() not in IMG_EXTS:
            continue
        x, canvas, region = preprocess(Image.open(p), size, use_lb)
        with torch.no_grad():
            result = model.predict(x.to(device), score_thr=args.score_thr)

        x0, y0, cw, ch = region
        trav = crop_box(result["traversability"][0].cpu().numpy(), region)
        terr = crop_box(result["terrain"][0].cpu().numpy(), region)
        vis_img = canvas.crop((x0, y0, x0 + cw, y0 + ch))
        vis_trav = overlay(vis_img, trav, TRAV_COLORS)
        vis_terr = overlay(vis_img, terr, terrain_colors)

        det = result.get("detection", [{}])[0]
        if det:
            draw = ImageDraw.Draw(vis_trav)
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
                    (b[0] + 2, b[1] + 2), f"{int(label)}:{score:.2f}", fill=(255, 255, 255)
                )

        combined = Image.new("RGB", (vis_trav.width * 2, vis_trav.height))
        combined.paste(vis_trav, (0, 0))
        combined.paste(vis_terr, (vis_trav.width, 0))
        out_file = out_dir / f"{p.stem}_pred.jpg"
        combined.save(out_file, quality=92)
        print(f"{p.name} -> {out_file}  (left: traversability + boxes, right: terrain)")


if __name__ == "__main__":
    main()
