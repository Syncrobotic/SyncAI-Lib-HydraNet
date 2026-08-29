#!/usr/bin/env python3
"""Pull a bounded sample of site clips from gs://studioa-recording.

    python3 scripts/pull_studioa.py --date 2026-08-16 --times 11:30 16:00 \\
        --out datasets/studioa_clips

The bucket holds 48 cameras x 30 days x ~189 clips of ~11 MB -- on the order of **3 TB**.
Nothing here should ever pull all of it, and the reason is not only egress cost: a
fixed camera's 189th clip of a day tells you almost nothing its 1st did not. Camera
coverage is the axis with information in it, so the sample is wide across cameras and
thin within each one.

**Two times, for two different jobs.** The static classes -- `column`, `fixture`,
`floor` -- are best drawn when the fewest people stand in front of them, so an
early trading-hour clip is worth more than a busy one. `person` and `product`-in-use
want the opposite. One clip each, per camera, is cheaper than arguing about it.

The clip whose *start* is nearest the target is chosen. Clips run about 5 minutes, so
"nearest" is within a few minutes of the intent and the exact offset is recorded in the
manifest rather than assumed.

**THE FILENAMES ARE UTC AND THE STORES ARE UTC+8.** Caught by looking at the frames
rather than the paths: a pull targeting `16:00` returned footage whose burnt-in
timestamp reads `2026/08/17 00:02` -- greyscale IR, shutters down, not a shopper in it.
Nothing about the filename says which zone it is in, and a request for "16:00, the busy
hour" that silently returns a closed store is exactly the failure this project keeps
naming: it does not error, it produces a plausible-looking sample of the wrong thing.

So `--times` is **local store time** by default and converted here, with `--utc-offset`
naming the zone. The manifest records both, so a reader never has to guess which one a
number was measured in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

BUCKET = "gs://studioa-recording"
PROJECT = "syncrobotic-aisw"


def cameras() -> list[str]:
    out = subprocess.run(
        ["gcloud", "storage", "ls", f"{BUCKET}/", f"--project={PROJECT}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout
    return sorted(line.rstrip("/").rsplit("/", 1)[-1] for line in out.split() if line.strip())


def day_clips(cam: str, date: str) -> list[str]:
    """Every mp4 that camera recorded that day. Empty is a fact, not an error --
    a camera can be offline for a day and a crash here would lose the other 47."""
    y, m, d = date.split("-")
    try:
        out = subprocess.run(
            ["gcloud", "storage", "ls", f"{BUCKET}/{cam}/{y}/{m}/{d}/*.mp4"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        ).stdout
    except subprocess.CalledProcessError:
        return []
    return [line for line in out.split() if line.endswith(".mp4")]


def start_of(uri: str) -> datetime:
    """archive_20260816-113012_20260816-113518.mp4 -> the first timestamp."""
    stamp = uri.rsplit("/", 1)[-1].removeprefix("archive_").split("_", 1)[0]
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S")


def nearest(
    clips: list[str], date: str, hhmm: str, utc_offset: float
) -> tuple[str, int] | None:
    """The clip starting closest to a **local** target, and the gap in seconds (signed).

    ``hhmm`` is store-local; the object names are UTC. The subtraction is the whole fix
    and it is worth one line of comment rather than a silent arithmetic: local 16:00 in
    a UTC+8 store is 08:00 UTC, and asking for the UTC 16:00 instead lands at midnight
    with the shutters down.
    """
    if not clips:
        return None
    local = datetime.strptime(f"{date} {hhmm}", "%Y-%m-%d %H:%M")
    target = local - timedelta(hours=utc_offset)
    best = min(clips, key=lambda c: abs((start_of(c) - target).total_seconds()))
    return best, int((start_of(best) - target).total_seconds())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD, in UTC as the paths are")
    ap.add_argument(
        "--times", nargs="+", default=["11:00", "19:30"], help="HH:MM, STORE-LOCAL time"
    )
    ap.add_argument(
        "--utc-offset",
        type=float,
        default=8.0,
        help="hours the store is ahead of UTC; Taiwan is +8 and the object names are UTC",
    )
    ap.add_argument("--out", default="datasets/studioa_clips")
    ap.add_argument("--cameras", nargs="*", help="default: every camera in the bucket")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cams = args.cameras or cameras()
    out_root = Path(args.out)
    print(
        f"{len(cams)} cameras x {len(args.times)} times = {len(cams) * len(args.times)} clips"
    )

    wanted, manifest = [], []
    for cam in cams:
        clips = day_clips(cam, args.date)
        if not clips:
            print(f"  {cam}: NOTHING on {args.date}")
            manifest.append({"camera": cam, "date": args.date, "clips": []})
            continue
        picks = []
        for hhmm in args.times:
            hit = nearest(clips, args.date, hhmm, args.utc_offset)
            if hit is None:
                continue
            uri, gap = hit
            dest = out_root / cam / uri.rsplit("/", 1)[-1]
            picks.append(
                {
                    "target_local": hhmm,
                    "utc_offset": args.utc_offset,
                    "uri": uri,
                    "offset_s": gap,
                    "path": str(dest),
                }
            )
            wanted.append((uri, dest))
        manifest.append(
            {"camera": cam, "date": args.date, "day_clips": len(clips), "clips": picks}
        )
        gaps = ", ".join(f"{p['target_local']}L{p['offset_s']:+d}s" for p in picks)
        print(f"  {cam}: {len(clips)} clips that day -> {gaps}")

    if args.dry_run:
        print("\ndry run, nothing downloaded")
        return 0

    for uri, dest in wanted:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        subprocess.run(["gcloud", "storage", "cp", uri, str(dest)], check=True, timeout=600)

    out_root.mkdir(parents=True, exist_ok=True)
    # One manifest per (date, times) pull rather than one per directory. A second pull
    # into the same tree is the normal case -- a first choice of hours is often wrong,
    # as it was here -- and overwriting the manifest would erase the record of which
    # clips the earlier numbers were measured on.
    tag = f"{args.date}_{'-'.join(t.replace(':', '') for t in args.times)}L"
    dest_manifest = out_root / f"manifest_{tag}.json"
    dest_manifest.write_text(
        json.dumps(
            {
                "bucket": BUCKET,
                "date_utc": args.date,
                "times_local": args.times,
                "utc_offset": args.utc_offset,
                "pulled": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    total = sum(p.stat().st_size for p in out_root.rglob("*.mp4"))
    print(f"\n{len(wanted)} clips, {total / 1e9:.2f} GB total in {out_root}")
    print(f"manifest: {dest_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
