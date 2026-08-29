#!/usr/bin/env python3
"""Do zone visits survive the tracker? PLAN step 5's other half.

    python3 scripts/zone_dwell.py --gt runs/gt_cam01

Step 5's gate is two sentences and only one of them has ever been measured. The first --
"position error small enough that zone events land in the right zone" -- has §7.13's
7.3 cm, and the second has nothing: *"dwell/loiter durations survive -- an ID switch
mid-loiter resets the clock, so switches on the watched clips must be rare enough not
to"*. §7.18's IDF1 answers which tracker keeps identities better; it does not answer what
that costs the thing the layer above actually reports, which is a duration in a zone.

So this runs the same chain the product runs -- `world_frames` -> `journeys` -> visits in
metres, through the camera's own commissioned geometry -- three times over the same clip:
once over the **labelled identities**, and once over each tracker's output. The first is
what the store did. The other two are what would have been reported.

Two numbers, because the gate names two failures:

  **survival**   for each true visit, the longest SINGLE reported visit overlapping it,
                 as a fraction of its duration. A loiter chopped in three by two ID
                 switches survives at ~0.33 however good the total is, and it is the
                 fraction rather than the total that a dwell alarm fires on.
  **purity**     for each reported visit, the share of its observations that belong to
                 the true identity it mostly is. A visit stitched from two shoppers reads
                 as one long dwell and is the failure the other direction.

Both are needed and neither implies the other: a tracker can hold one identity perfectly
and still split its visits (the geometry jitters across a boundary), and it can hold a
visit intact by walking it onto the wrong person.

**What this inherits.** The labels cover the detector's boxes only, so a shopper the
detector never saw is missing from the truth as well: survival is measured against what
was seen, not against what happened. And `runs/gt_*/provenance.json` records that a model
read those labels, not a person.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from syncai_hydranet.analytics import events as ev  # noqa: E402
from syncai_hydranet.analytics.journey import journeys  # noqa: E402
from syncai_hydranet.analytics.tracker import Track  # noqa: E402
from syncai_hydranet.analytics.world import world_frames  # noqa: E402
from syncai_hydranet.data.video import probe  # noqa: E402
from syncai_hydranet.geometry.camera_json import CameraFile  # noqa: E402

COMMISSIONED = ROOT / "runs/commission01"


def as_tracks(path: Path) -> list[Track]:
    """A `{id: {frame: box}}` file as `Track` objects the vector space already consumes.

    `confirmed=True` on every one: these are either labelled identities or an arm's
    finished tracks, both of which have already been through whatever confirmation they
    get. Re-deciding it here would silently drop rows the metrics upstream counted.
    """
    raw = json.loads(path.read_text())
    out = []
    for i, (_tid, byframe) in enumerate(raw.items()):
        fr = sorted(byframe, key=int)
        boxes = [np.asarray(byframe[f], dtype=float) for f in fr]
        out.append(
            Track(
                track_id=i,
                box=boxes[-1],
                frames=[int(f) for f in fr],
                boxes=boxes,
                confirmed=True,
                hits=len(fr),
            )
        )
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def visits_of(tracks: list[Track], cam_file, zones, fps, src, min_seconds):
    wf = world_frames(tracks, cam_file, name="person", fps=fps, source_size_px=src)
    js = journeys(wf, zones, fps=fps, min_seconds=min_seconds)
    rows = []
    for j in js:
        for v in j.visits:
            rows.append(
                {
                    "track": j.track_id,
                    "zone": v.zone,
                    "seconds": float(v.seconds or 0.0),
                    "first": int(v.frame_start),
                    "last": int(v.frame_end),
                }
            )
    return rows, js


def survival(true_rows: list[dict], pred_rows: list[dict]) -> list[dict]:
    """Longest single predicted visit overlapping each true visit, as a fraction of it."""
    out = []
    for t in true_rows:
        best = 0.0
        for p in pred_rows:
            if p["zone"] != t["zone"]:
                continue
            lo, hi = max(t["first"], p["first"]), min(t["last"], p["last"])
            if lo > hi:
                continue
            best = max(best, p["seconds"])
        out.append({**t, "longest_single_reported_s": best,
                    "survived": best / t["seconds"] if t["seconds"] else 0.0})  # fmt: skip
    return out


def purity(
    pred_rows: list[dict], pred_tracks: list[Track], gt_tracks: list[Track]
) -> list[dict]:
    """For each reported visit, the share of its frames on the identity it mostly is."""
    gt_by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for t in gt_tracks:
        for f, b in zip(t.frames, t.boxes, strict=True):
            gt_by_frame.setdefault(f, []).append((t.track_id, b))
    by_id = {t.track_id: t for t in pred_tracks}
    out = []
    for r in pred_rows:
        tr = by_id[r["track"]]
        seen: dict[int, int] = {}
        n = 0
        for f, b in zip(tr.frames, tr.boxes, strict=True):
            if not (r["first"] <= f <= r["last"]):
                continue
            n += 1
            best, who = 0.5, None
            for gid, gb in gt_by_frame.get(f, []):
                v = iou(b, gb)
                if v > best:
                    best, who = v, gid
            if who is not None:
                seen[who] = seen.get(who, 0) + 1
        top = max(seen.values()) if seen else 0
        out.append({**r, "frames": n, "matched": sum(seen.values()),
                    "identities": len(seen), "purity": top / n if n else 0.0})  # fmt: skip
    return out


def summarise(rows: list[dict], key: str) -> dict:
    v = sorted(r[key] for r in rows)
    if not v:
        return {"n": 0}
    q = lambda p: round(float(np.percentile(v, p)), 3)  # noqa: E731
    return {"n": len(v), "p10": q(10), "median": q(50), "p90": q(90),
            "mean": round(float(np.mean(v)), 3)}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument(
        "--loiter-seconds",
        type=float,
        default=10.0,
        help="a visit at least this long is the one a dwell alarm fires on",
    )
    a = ap.parse_args()

    prov = json.loads((a.gt / "provenance.json").read_text())
    camera = Path(prov["clip"]).parent.name
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    zones = ev.zones_from_camera(cam_file, kinds=["display", "till", "premium_shelf"])
    src_w, src_h, _ = probe(prov["clip"])
    fps = float(prov["fps"])

    gt_tracks = as_tracks(a.gt / "ground_truth.json")
    arms = {}
    wanted = (("arm_single", "single-stage (shipped)"), ("arm_two_stage", "two-stage"))
    for stem, label in wanted:
        p = a.gt / f"{stem}.json"
        if p.exists():
            arms[label] = as_tracks(p)
    if not arms:
        raise SystemExit(
            f"{a.gt} has no arm_*.json -- run scripts/track_idf1.py --gt {a.gt} first, "
            "which is what writes the two trackers' output over the same detections."
        )

    src = (src_w, src_h)
    true_rows, _ = visits_of(gt_tracks, cam_file, zones, fps, src, a.min_seconds)
    print(f"{camera}: {len(zones)} zones, {len(gt_tracks)} labelled identities")
    print(
        f"  truth: {len(true_rows)} visits, {sum(r['seconds'] for r in true_rows):.1f} s "
        f"total, {sum(1 for r in true_rows if r['seconds'] >= a.loiter_seconds)} of them "
        f">= {a.loiter_seconds:.0f} s"
    )
    report = {
        "camera": camera,
        "clip": prov["clip"],
        "zones": len(zones),
        "identities": len(gt_tracks),
        "truth_visits": true_rows,
        "arms": {},
    }
    for label, tracks in arms.items():
        rows, _ = visits_of(tracks, cam_file, zones, fps, src, a.min_seconds)
        sur = survival(true_rows, rows)
        pur = purity(rows, tracks, gt_tracks)
        long_true = [r for r in sur if r["seconds"] >= a.loiter_seconds]
        kept = sum(1 for r in long_true if r["survived"] >= 0.9)
        split = [r for r in pur if r["identities"] > 1]
        report["arms"][label] = {
            "visits": len(rows),
            "total_seconds": round(sum(r["seconds"] for r in rows), 1),
            "survival": summarise(sur, "survived"),
            "long_visits": len(long_true),
            "long_visits_kept_whole": kept,
            "purity": summarise(pur, "purity"),
            "visits_spanning_two_identities": len(split),
            "rows": pur,
        }
        print(
            f"  {label:24s} {len(rows):3d} visits  "
            f"survival p50 {report['arms'][label]['survival'].get('median', 0):.2f}  "
            f"loiters kept whole {kept}/{len(long_true)}  "
            f"purity p50 {report['arms'][label]['purity'].get('median', 0):.2f}  "
            f"visits spanning 2+ identities {len(split)}"
        )
    (a.gt / "zone_dwell.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"-> {a.gt / 'zone_dwell.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
