"""Known-false-positive polygons, derived from the box population (PLAN 2.1g).

The 2026-08-24 tiering pass measured that 43.4% of the Gray `person` boxes (score < 0.35
after NMS) come from 37 fixed positions -- hanging blister-pack walls and printed people.
A fixed position is a per-camera constant, so it belongs in `camera.json`, and per the
no-manual-annotation policy it is *derived*, not drawn: this tool re-runs the journal's
recipe (64 px cells, a cell that fires on more than half of the camera's frames -- a bar
no moving person can meet), merges adjacent hotspot cells, and writes the proposed
polygons into the camera's `camera.json`. The human's job is accept/reject on the
rendered proposal sheet, nothing more.

Usage: uv run python tools/commissioning/fp_polygons.py [camera ...]   (default: all 8)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from syncai_bev3d.teachers.boxes import nms
from syncai_hydranet.geometry.camera_json import CameraFile

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
ANN_DIR = ROOT / "datasets/site30k_v1/annotations"
GRAY_MAX = 0.35
CELL_PX = 64
STRONG_SHARE = 0.5
CANDIDATE_SHARE = 0.35
COMMISSIONED = [
    "Taichung-cam01",
    "Taichung-cam04",
    "Taichung-cam07",
    "Taichung-cam10",
    "Taichung-cam11",
    "Tao-Hsin-cam03",
    "Tao-Hsin-cam04",
    "Kaohsiung-cam04",
]


def gray_centres_by_camera():
    """(camera -> frame_id -> Nx2 gray centres at 1920x1080), from all three splits."""
    per_cam_frames: dict[str, set] = defaultdict(set)
    per_frame_boxes: dict[tuple, list] = defaultdict(list)
    frame_cam: dict[int, tuple] = {}
    for split_file in sorted(ANN_DIR.glob("instances_all_*.json")):
        d = json.loads(split_file.read_text())
        for im in d["images"]:
            cam = im["file_name"].split("__")[0]
            key = (split_file.stem, im["id"])
            frame_cam[key] = cam
            per_cam_frames[cam].add(key)
        for a in d["annotations"]:
            key = (split_file.stem, a["image_id"])
            x, y, w, h = a["bbox"]
            per_frame_boxes[key].append((x, y, x + w, y + h, a.get("score", 1.0)))
    out: dict[str, dict] = defaultdict(dict)
    for key, boxes in per_frame_boxes.items():
        arr = nms(np.asarray(boxes, float), 0.5)
        gray = arr[arr[:, 4] < GRAY_MAX]
        cx = (gray[:, 0] + gray[:, 2]) / 2
        cy = (gray[:, 1] + gray[:, 3]) / 2
        out[frame_cam[key]][key] = np.stack([cx, cy], 1)
    return out, per_cam_frames


def hotspot_polygons(frames: dict, n_frames: int, size=(1920, 1080)):
    """Cells firing on more than half the frames, merged into padded rectangles."""
    gw, gh = size[0] // CELL_PX + 1, size[1] // CELL_PX + 1
    seen = np.zeros((gh, gw), np.int32)
    for pts in frames.values():
        if not len(pts):
            continue
        cells = np.unique(
            (pts[:, 1] // CELL_PX).astype(int) * gw + (pts[:, 0] // CELL_PX).astype(int)
        )
        seen.ravel()[cells] += 1
    share_grid = seen / max(n_frames, 1)
    polys = []
    for tier, lo, hi in (
        ("strong", STRONG_SHARE, 10.0),
        ("candidate", CANDIDATE_SHARE, STRONG_SHARE),
    ):
        hot = (share_grid > lo) & (share_grid <= hi)
        lab, _n = ndimage.label(hot, structure=np.ones((3, 3)))
        for sl in ndimage.find_objects(lab):
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            pad = CELL_PX // 2
            x0 = max(0, c0 * CELL_PX - pad)
            y0 = max(0, r0 * CELL_PX - pad)
            x1 = min(size[0], c1 * CELL_PX + pad)
            y1 = min(size[1], r1 * CELL_PX + pad)
            share = float(share_grid[sl].max())
            polys.append(((x0, y0, x1, y1), share, tier))
    return polys, len(polys)


def main():
    cameras = sys.argv[1:] or COMMISSIONED
    centres, frames_per_cam = gray_centres_by_camera()
    for camera in cameras:
        frames = centres.get(camera, {})
        n_frames = len(frames_per_cam.get(camera, ()))
        if not n_frames:
            print(f"{camera}: no frames in site30k_v1, skipped")
            continue
        polys, n_cells = hotspot_polygons(frames, n_frames)
        total_gray = sum(len(p) for p in frames.values())
        covered = 0
        for pts in frames.values():
            for x, y in pts:
                if any(x0 <= x < x1 and y0 <= y < y1 for (x0, y0, x1, y1), *_ in polys):
                    covered += 1
        cf_path = ROOT / f"runs/commission01/{camera}.camera.json"
        if not cf_path.exists():
            print(f"{camera}: {len(polys)} hotspots (no camera.json -- proposal only)")
            continue
        cf = CameraFile.load(cf_path)
        w, h = cf.image_size_px
        sx, sy = w / 1920.0, h / 1080.0
        scaled = tuple(
            (
                (round(x0 * sx, 1), round(y0 * sy, 1)),
                (round(x1 * sx, 1), round(y0 * sy, 1)),
                (round(x1 * sx, 1), round(y1 * sy, 1)),
                (round(x0 * sx, 1), round(y1 * sy, 1)),
            )
            for (x0, y0, x1, y1), _, tier in polys
            if tier == "strong"
        )
        import dataclasses

        cf = dataclasses.replace(cf, false_positive_polygons_px=scaled)
        cf.validate()
        cf.save(cf_path)

        # Proposal sheet on the cleanest plate, for the accept/reject pass.
        cache = np.load(ROOT / f"runs/commission01/{camera}/structure_cache.npz")
        plate = ROOT / "datasets/studioa_static" / camera / f"plate_{cache['cleanest']}.png"
        img = Image.open(plate).convert("RGB").resize((960, 540))
        d = ImageDraw.Draw(img, "RGBA")
        for i, ((x0, y0, x1, y1), share, tier) in enumerate(polys):
            bx = [x0 / 2, y0 / 2, x1 / 2, y1 / 2]
            col = (255, 60, 40) if tier == "strong" else (255, 170, 40)
            d.rectangle(bx, outline=col, width=2, fill=(*col, 40))
            d.text((bx[0] + 3, bx[1] + 2), f"FP{i} {share:.0%}", fill=(255, 235, 215))
        d.rectangle([0, 524, 960, 540], fill=(0, 0, 0))
        d.text(
            (6, 526),
            f"{camera}  FP proposal: red>=50% auto-written, orange 35-50% awaiting verdict "
            f"(covers {covered / max(total_gray, 1):.0%} of Gray; "
            f"label = share of frames the hotspot fires in)",
            fill=(255, 255, 255),
        )
        img.save(ROOT / f"assets/commission_fp_{camera}.png")
        print(
            f"{camera}: {len(polys)} hotspots ({n_cells} cells), cover "
            f"{covered / max(total_gray, 1):.1%} of {total_gray} gray over {n_frames} frames"
        )


if __name__ == "__main__":
    main()
