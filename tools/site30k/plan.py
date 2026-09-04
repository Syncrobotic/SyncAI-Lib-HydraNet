#!/usr/bin/env python3
"""Build the site30k work plan: which clips get annotated, how many frames each.

The unit of work is a (camera, DATE), not a clip and not a camera: the floor and the
immovable objects are decided once over the day plates a camera has ON THAT DATE. Per
camera is what makes a pillar keep its class between tranches (the batch measured that);
per date is what stops a fixture that moved in three weeks from being painted where it
used to stand.

Scope, and the reason for each bound:

  cameras   only those whose calibration is METRIC and whose zones carry a
            `walkable_floor` polygon in metres -- 9 of the 23 that have a calib file at
            all. The floor recipe needs a ground plane AND a scale: without them there is
            no on-plane test, no metre-space BEV vote and no tol(r), and the camera would
            fall back to b03 alone, which is the teacher this whole recipe exists to
            correct. The other 14 have `dav2_raw` zones (relative depth, no scale) or no
            zones file, and belong to a calibration pass, not to this one.
  clips     day only, and 1920x1080 only. Night is a person-only tranche by decision
            (all-IGNORE seg masks) and the plate teachers have no measurement on IR. The
            resolution check is not paranoia: the phase-2 pull is MIXED -- the store's
            archive serves 352x240 for the first dates of the window and 1920x1080 after
            them (73 of 883 clips on these cameras are the small one). SAM 3 returns
            nothing useful at 352x240 -- that is the measured `product` failure this
            campaign already has on record -- and the whole recipe was reviewed at
            1080p, so a small clip is left out rather than quietly upscaled.
  frames    the target divided over the day clips, capped so no clip contributes more
            than a fraction of the set -- 30k frames drawn from 200 clips would be 150
            near-duplicate seconds each and the split would leak between them.

Writes campaign_plan.json: one entry per unit, ready for run_campaign.sh.
"""

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Derived from this file rather than written out: an absolute path here breaks on any
# machine but the one it was typed on, and `sys.path` is the one place where that
# fails before anything else can report it. `parents[2]` is the repo root --
# tools/site30k/<file>.py.
from syncai_hydranet.data.video import probe

# The repo root, derived rather than written out: every one of these 26 tools had it
# as an absolute path, so a second checkout ran against the first one's `runs/` and
# any machine but this one failed at import with a path and no reason. Two levels up
# from `tools/<group>/<tool>.py`, and `tests/test_no_absolute_sys_path.py` keeps it so.
ROOT = Path(__file__).resolve().parents[2]
PULL = ROOT / "datasets/studioa_pull_site30k"
CALIB = ROOT / "runs/onboard01"
# Filenames are UTC; the store is UTC+8. 23:00-06:00 local is the night tranche.
NIGHT_UTC_HOURS = {15, 16, 17, 18, 19, 20, 21, 22}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=30000, help="frames to annotate")
    ap.add_argument("--max-per-clip", type=int, default=40)
    ap.add_argument(
        "--min-width",
        type=int,
        default=1920,
        help="clips narrower than this are left out (see the docstring)",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "runs/site30k_qa/campaign_plan.json")
    ap.add_argument("--split-json", type=Path, default=ROOT / "datasets/site30k/split.json")
    args = ap.parse_args()

    split = json.loads(args.split_json.read_text())["assign"]
    cameras = []
    rejected = []
    for calib_path in sorted(CALIB.glob("*.calib.json")):
        cam = calib_path.name.replace(".calib.json", "")
        zones_path = ROOT / f"runs/zones01/{cam}.zones.json"
        if not zones_path.exists():
            rejected.append((cam, "no zones file"))
            continue
        calib = json.loads(calib_path.read_text())
        zones = json.loads(zones_path.read_text())
        if zones.get("units") != "m" or calib.get("scale") is None:
            rejected.append(
                (cam, f"zones units={zones.get('units')} scale={calib.get('scale')}")
            )
            continue
        if not any(pr["name_suggestion"] == "walkable_floor" for pr in zones["proposals"]):
            rejected.append((cam, "no walkable_floor polygon"))
            continue
        cameras.append(cam)
    units: dict[tuple[str, str], list[str]] = defaultdict(list)
    skipped_night = 0
    candidates = []
    for cam in cameras:
        for clip in sorted((PULL / cam).glob("*.mp4")):
            m = re.search(r"archive_(\d{8})-(\d{6})_", clip.name)
            if not m:
                continue
            date, hhmmss = m.group(1), m.group(2)
            if int(hhmmss[:2]) in NIGHT_UTC_HOURS:
                skipped_night += 1
                continue
            candidates.append((cam, date, clip))

    def sized(item):
        cam, date, clip = item
        try:
            w, h, _ = probe(str(clip))
        except Exception:
            return cam, date, clip, 0, 0
        return cam, date, clip, w, h

    small = 0
    unreadable = 0
    with ThreadPoolExecutor(12) as pool:
        for cam, date, clip, w, _h in pool.map(sized, candidates):
            if w == 0:
                unreadable += 1
                continue
            if w < args.min_width:
                small += 1
                continue
            units[(cam, date)].append(clip.stem)

    n_clips = sum(len(v) for v in units.values())
    per_clip = min(args.max_per_clip, max(1, round(args.target / max(n_clips, 1))))
    plan = [
        {
            "camera": cam,
            "date": date,
            "split": split.get(cam, "train"),
            "clips": sorted(stems),
            "frames_per_clip": per_clip,
            "frames": per_clip * len(stems),
        }
        for (cam, date), stems in sorted(units.items())
    ]
    total = sum(u["frames"] for u in plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "target": args.target,
                "frames_per_clip": per_clip,
                "units": len(plan),
                "cameras_rejected": dict(rejected),
                "clips": n_clips,
                "planned_frames": total,
                "night_clips_skipped": skipped_night,
                "clips_below_min_width": small,
                "clips_unreadable": unreadable,
                "cameras": len(cameras),
                "plan": plan,
            },
            indent=1,
        )
    )

    by_split: dict[str, int] = defaultdict(int)
    for u in plan:
        by_split[u["split"]] += u["frames"]
    print(f"{len(rejected)} cameras left out for want of a metric calibration:")
    for cam, why in rejected:
        print(f"   {cam:20s} {why}")
    print(
        f"{len(cameras)} usable cameras, {len(plan)} camera-date units, "
        f"{n_clips} day clips at {args.min_width}px or wider"
    )
    print(
        f"   left out: {skipped_night} night clips, {small} sub-resolution clips, "
        f"{unreadable} unreadable"
    )
    print(f"{per_clip} frames per clip -> {total} frames planned (target {args.target})")
    for s, n in sorted(by_split.items()):
        print(f"   {s:6s} {n:6d} frames  ({100 * n / max(total, 1):.1f}%)")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
