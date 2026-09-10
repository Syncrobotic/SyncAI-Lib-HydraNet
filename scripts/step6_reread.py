#!/usr/bin/env python3
"""Re-read a step 6 fleet log at the loiter thresholds a store would actually set.

    python3 scripts/step6_reread.py runs/step6_fleet01 runs/step6_fleet02

docs/PLAN.md step 6 quotes four numbers from one run -- 1,478 / 427 / 30 / 6 alerts per
camera-day at 8 / 30 / 120 / 240 s -- and until this file existed the arithmetic behind
them lived in a session transcript. A number with no instrument that reproduces it rots
the first time the run is repeated with a different tracker, which is what happened on
2026-09-10 when the survival band went in. So: the arithmetic, once, and the same
arithmetic over every log handed to it, so two runs are compared by one rule.

The rule, stated because every part of it is a choice:

* A **loitering** event's `value` is the dwell in seconds, so the log at policy T=8 s can
  be re-read at any T >= 8 by counting the events whose value clears it. Not below 8: an
  event under the policy was never filed, so that population is unobserved.
* **occupancy_exceeded** has no duration to re-read against and is counted whole at every
  threshold. Its docstring says it counts tracks and so over-counts by the fragmentation
  rate; that is the number this re-read exists to watch move.
* **Open hours** are local 10:00-22:00, twelve trading hours, and a clip is open or closed
  by its start. The corpus stamps names in UTC and the stores are UTC+8
  (`events.clip_start_from_name` carries the cost of forgetting that). Closed-hours
  events are reported separately, not folded in: an after-hours person is a different
  alert with a different threshold, and 1 event in 40 camera-minutes says nothing about
  the trading day.
* **Per camera-day** = open-hours alerts / open-hours camera-minutes x 720. Every clip is
  one camera, so footage minutes are camera-minutes.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta, timezone
from pathlib import Path

from syncai_hydranet.analytics.events import clip_start_from_name

STORE_TZ = timezone(timedelta(hours=8))
OPEN_HOURS = (10, 22)
THRESHOLDS = (8.0, 30.0, 120.0, 240.0)
TRADING_MINUTES = (OPEN_HOURS[1] - OPEN_HOURS[0]) * 60


def is_open(clip_name: str) -> bool:
    hour = clip_start_from_name(clip_name, tz=STORE_TZ).hour
    return OPEN_HOURS[0] <= hour < OPEN_HOURS[1]


def reread(fleet: dict, cameras: set[str] | None = None) -> dict:
    """The table for one log. `cameras` restricts to a common set for a comparison."""
    reports = [r for r in fleet["clip_reports"] if cameras is None or r["camera"] in cameras]
    fps = float(fleet["fps"])
    open_min = closed_min = 0.0
    open_ev: list[dict] = []
    closed_ev: list[dict] = []
    per_cam: dict[str, dict] = {}
    for r in reports:
        minutes = r["frames_read"] / fps / 60
        c = per_cam.setdefault(r["camera"], {"clips": 0, "tracks": 0, "events": 0})
        c["clips"] += 1
        c["tracks"] += r["tracks"]
        c["events"] += len(r["events"])
        if is_open(r["clip"]):
            open_min += minutes
            open_ev += r["events"]
        else:
            closed_min += minutes
            closed_ev += r["events"]
    occupancy = sum(e["type"] == "occupancy_exceeded" for e in open_ev)
    rows = []
    for t in THRESHOLDS:
        loiter = sum(e["type"] == "loitering" and e["value"] >= t for e in open_ev)
        rows.append(
            {
                "threshold_s": t,
                "loitering": loiter,
                "occupancy_exceeded": occupancy,
                # The headline is loitering only, which is what the plan's four numbers
                # are; occupancy is beside it so the over-count is visible, not hidden.
                "loitering_per_camera_day": loiter / open_min * TRADING_MINUTES,
                "all_per_camera_day": (loiter + occupancy) / open_min * TRADING_MINUTES,
            }
        )
    return {
        "cameras": sorted(per_cam),
        "clips": len(reports),
        "open_camera_minutes": round(open_min, 1),
        "closed_camera_minutes": round(closed_min, 1),
        "closed_events": len(closed_ev),
        "tracks": sum(c["tracks"] for c in per_cam.values()),
        "events": sum(c["events"] for c in per_cam.values()),
        "per_camera": per_cam,
        "by_threshold": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="step 6 output directories")
    ap.add_argument(
        "--common",
        action="store_true",
        help="restrict every log to the cameras present in all of them, so a run that "
        "commissioned one more camera is compared on the same footage",
    )
    ap.add_argument("--json", type=Path, default=None, help="write the tables here too")
    args = ap.parse_args()

    fleets = {str(p): json.loads((p / "fleet.json").read_text()) for p in args.runs}
    common = None
    if args.common:
        sets = [{r["camera"] for r in f["clip_reports"]} for f in fleets.values()]
        common = set.intersection(*sets)
    tables = {name: reread(f, common) for name, f in fleets.items()}

    for name, t in tables.items():
        print(
            f"\n{name}: {t['clips']} clips over {len(t['cameras'])} cameras, "
            f"{t['tracks']} tracks, {t['events']} events"
        )
        print(
            f"  open {t['open_camera_minutes']} camera-min, closed "
            f"{t['closed_camera_minutes']} camera-min with {t['closed_events']} events"
        )
        head = ("T (s)", "loiter", "occup", "loiter/cam-day", "all/cam-day")
        print(f"  {head[0]:>6} {head[1]:>7} {head[2]:>6} {head[3]:>15} {head[4]:>12}")
        for r in t["by_threshold"]:
            print(
                f"  {r['threshold_s']:6.0f} {r['loitering']:7d} {r['occupancy_exceeded']:6d} "
                f"{r['loitering_per_camera_day']:15.1f} {r['all_per_camera_day']:12.1f}"
            )
    if len(tables) > 1:
        print(f"\n{'camera':18s}" + "".join(f"{Path(n).name:>22s}" for n in tables))
        cams = sorted(set().union(*(t["per_camera"] for t in tables.values())))
        for cam in cams:
            cells = []
            for t in tables.values():
                c = t["per_camera"].get(cam)
                cells.append(f"{c['tracks']:5d} trk {c['events']:4d} ev" if c else " " * 16)
            print(f"{cam:18s}" + "".join(f"{x:>22s}" for x in cells))
    if args.json:
        args.json.write_text(json.dumps(tables, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
