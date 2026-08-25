#!/usr/bin/env python3
"""Shelf coverage: what fraction of a fixture is stocked, measured rather than counted.

    python3 scripts/sam3_product_coverage.py --out runs/coverage01 \\
        --frames 6 datasets/studioa_clips/Taichung-cam*/archive_*.mp4

Counting merchandise and measuring how much of a shelf it covers are different products,
and `hydranet_retail_products.yaml` already separated them in one sentence: `product` as a
region is "still the right answer to how much shelf is stocked even when it is the wrong
answer to how many items". The detection head answers the count. This answers the area,
and it needs no per-item edges, which is exactly why the stride-8 objection that moved
`product` out of the dense head does not apply to it.

**Native resolution is the whole trick.** `product` prompts were once reported as finding
0 instances on accessory walls; that figure was taken at 352x240 and with one prompt of
eight. At 1920x1080 with the swept prompt set the same walls come back as a third of their
pixels. Nothing here downscales.

---------------------------------------------------------------------------
WHAT THE NUMBER IS, AND WHAT IT IS NOT

Coverage is `product pixels inside the fixture region / fixture region pixels`, in the
image. Two consequences that decide how it may be read:

* **It is comparable over time on one camera and not between cameras.** A shelf is
  vertical, so no ground-plane homography rectifies it, and a camera looking along an
  aisle sees a different projected area of the same shelf than one facing it. Restocking
  and depletion on a fixed camera are visible; "Taichung stocks better than Kaohsiung" is
  not a claim this supports.
* **The denominator is the terrain head's `fixture`, so it inherits that head's errors.**
  Where the model calls a wall a fixture the coverage is diluted, and where it misses a
  gondola the coverage is over-stated. `fixture` is the head's strongest class at 0.72 IoU,
  which is why this is worth doing at all -- and it is still a model output rather than a
  measurement of a shelf.

The masks are written in `seg_folder` layout as well, so a dense `product` class can be
trained from them later and the coverage can then be computed by the network alone, at
6 ms rather than at SAM 3's seconds per frame.
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
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
sys.path.insert(0, str(HERE))

from syncai_bev3d.teachers.photometry import is_daylight  # noqa: E402
from syncai_bev3d.teachers.sam3 import load_sam3, segment  # noqa: E402
from syncai_hydranet.cli.infer_video import frames, probe  # noqa: E402
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.data.sam3_prompts_objects import CONCEPTS  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--seg-config", default="runs/hydranet_retail_surfaces/config.yaml")
    ap.add_argument("--seg-checkpoint", default="runs/hydranet_retail_surfaces/best.pt")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--save-masks", action="store_true", help="write seg_folder masks too")
    return ap


def product_concept():
    for c in CONCEPTS:
        if c.name == "product":
            return c
    raise SystemExit("no `product` concept in sam3_prompts_objects.CONCEPTS")


def fixture_region(seg, cfg, device, frame: np.ndarray) -> np.ndarray:
    """The terrain head's `fixture` pixels at the frame's own resolution."""
    classes = list(cfg["data"]["terrain_classes"])
    fid = classes.index("fixture")
    img = Image.fromarray(frame)
    x, _, region = preprocess(img, cfg["data"]["input_size"])
    with torch.no_grad():
        terrain = seg.predict(x.to(device))["terrain"][0].cpu().numpy().astype(np.uint8)
    x0, y0, cw, ch = region
    terrain = terrain[y0 : y0 + ch, x0 : x0 + cw]
    terrain = np.asarray(
        Image.fromarray(terrain).resize(
            (frame.shape[1], frame.shape[0]), Image.Resampling.NEAREST
        )
    )
    return terrain == fid


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = pick_device(None)
    proc, sam = load_sam3(args.model_id, "cuda" if device.type == "cuda" else "cpu")
    cfg = load_config(args.seg_config)
    seg = build_model(cfg).to(device).eval()
    seg.load_state_dict(select_weights(load_checkpoint(args.seg_checkpoint), "ema"))
    concept = product_concept()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for clip in args.clips:
        camera = Path(clip).parent.name
        w, h, _ = probe(clip)
        every = max(int(300 * 5 / max(args.frames, 1)), 1)
        seen = 0
        for i, frame in enumerate(frames(clip, w, h, 5.0)):
            if i % every or not is_daylight(frame):
                continue
            img = Image.fromarray(frame)  # native, nothing downscaled
            product = np.zeros((h, w), dtype=bool)
            per_prompt = {}
            for prompt in concept.prompts:
                pairs = segment(proc, sam, img, prompt, concept.min_score, device.type)
                m = np.zeros((h, w), dtype=bool)
                for mask, _s in pairs:
                    m |= mask
                per_prompt[prompt] = round(float(m.mean()), 4)
                product |= m
            fixture = fixture_region(seg, cfg, device, frame)
            inter = int((product & fixture).sum())
            rows.append(
                {
                    "camera": camera,
                    "session": Path(clip).stem,
                    "frame": i,
                    "resolution": f"{w}x{h}",
                    "product_share_of_frame": round(float(product.mean()), 4),
                    "fixture_share_of_frame": round(float(fixture.mean()), 4),
                    "coverage_of_fixture": round(inter / max(int(fixture.sum()), 1), 4),
                    "per_prompt_share": per_prompt,
                }
            )
            if args.save_masks:
                mdir = out_root / "masks"
                mdir.mkdir(exist_ok=True)
                lab = np.zeros((h, w), dtype=np.uint8)
                lab[product] = 1
                Image.fromarray(lab).save(mdir / f"{camera}__{i:05d}.png")
                Image.fromarray(frame).save(mdir / f"{camera}__{i:05d}.jpg", quality=92)
            seen += 1
            if seen >= args.frames:
                break
        cam_rows = [r for r in rows if r["camera"] == camera]
        if cam_rows:
            cov = np.array([r["coverage_of_fixture"] for r in cam_rows])
            print(
                f"{camera:18s} {len(cam_rows)} frames  fixture "
                f"{np.mean([r['fixture_share_of_frame'] for r in cam_rows]):.1%} of frame  "
                f"coverage {cov.mean():.1%}  (min {cov.min():.1%} max {cov.max():.1%})"
            )
    (out_root / "coverage.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_root / 'coverage.json'} -- {len(rows)} frames")
    print("Comparable over time on one camera; not comparable between cameras.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
