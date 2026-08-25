"""The three-minute demo: detections and tracks on the left, the mesh scene walked
through on the right.

The right panel is the standing instruction: the solid-furniture scene
(`scene_mesh.build_scene_regular`), not the flat ribbon panel -- with a mesh figure
placed at every confirmed track's floor position, so a shopper's motion is visible in
metres in the same picture that shows the furniture they walk between. The camera.json
false-positive polygons are applied to the detections before tracking, which is those
polygons doing their production job for the first time.

Usage:
  uv run python tools/commissioning/demo_video.py <camera> [--clip PATH]
      [--frames 900] [--fps 5]

Writes assets/demo_<camera>.mp4 (gitignored -- customer footage) and three sample
frames for the frame-check.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import scene_mesh

from syncai_bev3d.meshes import Placement, human, place
from syncai_hydranet.analytics.tracker import Tracker
from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
RUN = ROOT / "runs/hydranet_retail_security_b03_cw_xl"
TRACK_COLORS = [
    (255, 99, 71), (65, 180, 255), (255, 200, 60), (120, 220, 120),
    (220, 120, 255), (255, 150, 100), (100, 230, 210), (250, 100, 160),
]  # fmt: skip


def in_fp_zone(cf: CameraFile, cx: float, cy: float) -> bool:
    for poly in cf.false_positive_polygons_px:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        if min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args()
    camera = args.camera

    clip = (
        Path(args.clip)
        if args.clip
        else sorted((ROOT / "datasets/studioa_clips" / camera).glob("archive_*11*.mp4"))[0]
    )  # a late-morning clip: the store is open and populated

    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    cfg = load_config(str(RUN / "config.yaml"), validate=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(RUN / "best.pt"), "ema"))
    size = cfg["data"]["input_size"]
    person_label = list(cfg["model"]["heads"]["detection"]["classes"]).index("person")

    # the static scene, built once; the view frozen so the room does not swim
    scene_mesh.SS = 1
    _cf2, items, heights = scene_mesh.build_scene_regular(camera)
    # furniture and floor only: the product slabs read as clutter at video scale, and
    # the demo's subject is people moving through the room
    items = [it for it in items if not str(it[1]).startswith("product")]
    xs = np.concatenate([m[0][:, 0] for m, *_ in items])
    zs = np.concatenate([m[0][:, 2] for m, *_ in items])
    cx_m, cz_m = float(np.median(xs)), float(np.median(zs))
    eye = [cx_m + 7.5, 5.6, cz_m - 6.5]
    target = [cx_m, 0.6, cz_m]

    panel_w, panel_h = 890, 540
    out_w, out_h = 960 + 8 + panel_w, 540
    out_path = ROOT / f"assets/demo_{camera}.mp4"
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-framerate", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    tracker = Tracker()
    checks = {0, args.frames // 2, args.frames - 1}
    n = 0
    for frame in decode_frames(str(clip), 1920, 1080, args.fps):
        if n >= args.frames:
            break
        img = Image.fromarray(frame)
        x, _canvas, region = preprocess(img, size)
        with torch.no_grad():
            out = model.predict(x.to(device), score_thr=0.5)
        det = out.get("detection", [{}])[0]
        boxes_src = np.zeros((0, 4), np.float32)
        if det and len(det.get("boxes", [])):
            b = det["boxes"].cpu().numpy()
            lab = det["labels"].cpu().numpy()
            x0, y0, cw, _ch = region
            b = (b - np.array([x0, y0, x0, y0])) * (1920.0 / cw)
            b = b[lab == person_label]
            keep = [
                i
                for i, bb in enumerate(b)
                if not in_fp_zone(cf, (bb[0] + bb[2]) / 4, (bb[1] + bb[3]) / 4)
            ]
            boxes_src = b[keep]
        tracks = [t for t in tracker.update(boxes_src, n) if t.hits >= 3]

        # left: source view with boxes and ids
        view = img.resize((960, 540))
        d = ImageDraw.Draw(view)
        figures = []
        for t in tracks:
            col = TRACK_COLORS[t.track_id % len(TRACK_COLORS)]
            bx = np.asarray(t.box, float) / 2.0
            d.rectangle(list(bx), outline=col, width=2)
            d.text((bx[0] + 3, bx[1] + 2), f"#{t.track_id}", fill=col)
            foot = np.array([[(t.box[0] + t.box[2]) / 2 / 2.0, t.box[3] / 2.0]])
            if cf.lens is not None:
                foot = undistort_points(foot, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
            fx, fz = pixel_to_ground(foot[:, 0], foot[:, 1], cf.camera, cf.plane)
            if np.isfinite(fx[0]) and 0 < fz[0] < 14 and abs(fx[0]) < 12:
                figures.append(
                    (
                        place(human(1.70), Placement(float(fx[0]), float(fz[0]))),
                        "person",
                        255,
                        True,
                    )
                )

        scene_mesh.PALETTE.setdefault("person", (168, 208, 250))
        tmp = ROOT / "assets/_demo_panel.png"
        scene_mesh.render(camera, items + figures, heights, tmp, eye=eye, target=target)
        panel = Image.open(tmp).resize((panel_w, panel_h))

        composite = Image.new("RGB", (out_w, out_h), (7, 9, 13))
        composite.paste(view, (0, 0))
        composite.paste(panel, (968, 0))
        dd = ImageDraw.Draw(composite)
        dd.rectangle([0, 524, 960, 540], fill=(0, 0, 0))
        dd.text(
            (6, 526),
            f"{camera}  detections+tracks (FP zones applied)  |  right: commissioning "
            f"mesh scene, figures at tracked floor positions  frame {n}",
            fill=(255, 255, 255),
        )
        enc.stdin.write(np.asarray(composite, np.uint8).tobytes())
        if n in checks:
            composite.save(ROOT / f"assets/demo_{camera}_check{n:03d}.png")
        if n % 100 == 0:
            print(f"  {n}/{args.frames}", flush=True)
        n += 1

    enc.stdin.close()
    enc.wait()
    (ROOT / "assets/_demo_panel.png").unlink(missing_ok=True)
    print(f"wrote {out_path} ({n} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
