"""Per-camera `appearance_thr` for `camera.json`, or None where it cannot be had.

    uv run python tools/commissioning/appearance_calibrate.py Taichung-cam04
    uv run python tools/commissioning/appearance_calibrate.py --all --write

WHY THIS IS PER CAMERA AND NOT A CONSTANT
`Tracker(appearance_thr=...)` refuses a re-association whose appearance disagrees with the
track's last observation. The two errors are not symmetric:

  * below a camera's p99 of SAME-person distances, the gate splits one shopper into two,
    and the visit count -- a headline number -- rises for no reason;
  * above its p10 of DIFFERENT-people distances, the gate catches nothing and behaviour
    reverts to what every published figure was measured under.

Both bounds are properties of the camera: its lighting, how uniform the clothing is, how
close people stand. Scanned across the eight commissioned cameras on 2026-09-09, the
fleet-wide constraints are "> 0.323" and "< 0.175" -- **empty**. There is no single number,
and the one that looked robust on the camera it was found on (0.30 on Taichung-cam04) is
below Tao-Hsin-cam04's safe floor and above Tao-Hsin-cam03's useful ceiling.

WHAT IT REFUSES TO ANSWER
`None` is written, and gating is off for that camera, when the two distributions overlap
or when there are fewer than `MIN_PAIRS` of either. Both are honest states: on
Tao-Hsin-cam04 same-person distances ran LARGER than different-people ones, and on
Taichung-cam07 no two people were in frame at once in the clip, so there was no control to
compare against. Not gating is the safe half of the asymmetry.

NOT MEASURED HERE: the night. The fleet's fourth sample is 23:58-00:00 with the shops shut
(0 tracks on Taichung-cam04, 1 on Kaohsiung-cam04). A monochrome IR frame should collapse
the descriptor, every distance shrink, and the gate fall silent -- benign, and benign by
argument rather than by measurement. A camera that will run at night should be re-scanned
on night footage before its number is trusted there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(os.environ.get("SYNCAI_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))

from syncai_hydranet.analytics.appearance import torso_histogram  # noqa: E402
from syncai_hydranet.analytics.clip_tracks import (  # noqa: E402
    PERSON,
    to_source_pixels,
    undistort_boxes,
)
from syncai_hydranet.analytics.tracker import Tracker, appearance_distance  # noqa: E402
from syncai_hydranet.data.video import frames, probe  # noqa: E402
from syncai_hydranet.geometry.camera_json import CameraFile  # noqa: E402
from syncai_hydranet.shipped import load_model  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402

FPS, BIRTH, LOW, IOU, MAXAGE, MINHITS = 5.0, 0.35, 0.20, 0.3, 5, 3
MAX_FRAMES = 750  # 150 s: this is a distribution, not a total
MIN_PAIRS = 100


def scan(cam: str, clip: Path, model, cfg, device) -> dict:
    cf = CameraFile.load(ROOT / f"runs/commission01/{cam}.camera.json")
    k1 = float(cf.lens.k1) if cf.lens else None
    w, h, _ = probe(str(clip))
    tr = Tracker(iou_threshold=IOU, max_age=MAXAGE, min_hits=MINHITS, birth_thr=BIRTH)
    by_track: dict[int, list] = {}
    by_frame: dict[int, list] = {}
    for n, fr in enumerate(frames(str(clip), w, h, FPS)):
        if n >= MAX_FRAMES:
            break
        x, _, region = preprocess(Image.fromarray(fr), cfg["data"]["input_size"])
        with torch.no_grad():
            det = model.predict(x.to(device), score_thr=LOW)["detection"][0]
        keep = det["labels"].cpu().numpy() == PERSON
        raw = to_source_pixels(det["boxes"].cpu().numpy()[keep], region, w, h)
        sc = det["scores"].cpu().numpy()[keep]
        # The descriptor is cut from the RAW frame at RAW coordinates; the tracker is fed
        # undistorted boxes. Mixing the two displaces the crop by over a body height at
        # the frame edge.
        hists = [torso_histogram(fr, b) for b in raw]
        boxes = undistort_boxes(raw, k1, w, h) if k1 is not None else raw
        for t in tr.update(boxes, n, scores=sc):
            if not t.frames or t.frames[-1] != n or not len(boxes):
                continue
            j = int(np.argmin(np.abs(boxes[:, 0] - t.boxes[-1][0])))
            if hists[j] is None:
                continue
            by_track.setdefault(t.track_id, []).append(hists[j])
            by_frame.setdefault(n, []).append((t.track_id, hists[j]))
    within = [appearance_distance(a, b) for v in by_track.values() for a, b in pairwise(v)]
    rng = np.random.default_rng(0)
    between = []
    for lst in by_frame.values():
        if len(lst) < 2:
            continue
        for _ in range(min(6, len(lst))):
            i, j = rng.choice(len(lst), 2, replace=False)
            if lst[i][0] != lst[j][0]:
                between.append(appearance_distance(lst[i][1], lst[j][1]))
    out = {"camera": cam, "clip": clip.stem, "within_n": len(within), "between_n": len(between)}
    if len(within) < MIN_PAIRS or len(between) < MIN_PAIRS:
        return {**out, "appearance_thr": None, "why": "too few pairs to compare"}
    w_arr, b_arr = np.asarray(within), np.asarray(between)
    floor = float(np.percentile(w_arr, 99))
    ceiling = float(np.percentile(b_arr, 10))
    out |= {
        "safe_floor": round(floor, 4),
        "useful_ceiling": round(ceiling, 4),
        "within_p50": round(float(np.median(w_arr)), 4),
        "between_p50": round(float(np.median(b_arr)), 4),
    }
    if floor >= ceiling:
        return {
            **out,
            "appearance_thr": None,
            "why": "the two distributions overlap: no value both splits nobody real "
            "and catches anything",
        }
    # The midpoint. The asymmetry says a site seeing inflated visit counts should raise
    # this rather than lower it -- too high is a no-op, too low is a defect.
    return {
        **out,
        "appearance_thr": round((floor + ceiling) / 2, 4),
        "why": "midpoint of the gap",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cameras", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--clip-index",
        type=int,
        default=2,
        help="which of the camera's sorted clips to scan; 2 is the evening "
        "window the fleet was compared on",
    )
    ap.add_argument(
        "--write", action="store_true", help="write appearance_thr into each camera.json"
    )
    args = ap.parse_args(argv)
    cams = args.cameras
    if args.all:
        cams = sorted(
            p.stem.replace(".camera", "")
            for p in (ROOT / "runs/commission01").glob("*.camera.json")
            if (ROOT / "datasets/studioa_clips" / p.stem.replace(".camera", "")).is_dir()
        )
    if not cams:
        ap.error("name at least one camera, or pass --all")
    model, cfg, device = load_model(
        ROOT / "runs/hydranet_retail_person01/config.yaml",
        ROOT / "runs/hydranet_retail_person01/last.pt",
        validate=False,
    )
    rows = []
    for cam in cams:
        clips = sorted((ROOT / "datasets/studioa_clips" / cam).glob("*.mp4"))
        r = scan(cam, clips[args.clip_index], model, cfg, device)
        rows.append(r)
        print(
            f"{cam:18} thr={r['appearance_thr']!s:>7}  "
            f"pairs {r['within_n']}/{r['between_n']}  {r['why']}",
            flush=True,
        )
        if args.write:
            path = ROOT / f"runs/commission01/{cam}.camera.json"
            raw = json.loads(path.read_text())
            raw["appearance_thr"] = r["appearance_thr"]
            path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
            CameraFile.load(path)  # refuse to leave a file this reader cannot read back
    out = ROOT / "runs/commission01/appearance_calibration.json"
    out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
