#!/usr/bin/env python3
"""Record synchronised robot video + telemetry off the live dashboard, for E-prep.

    python3 scripts/robot_capture.py --seconds 60 --out datasets/robot_eprep/stand01

`docs/RESEARCH_OCCUPANCY.md` says E-prep "consumes footage we already have". It does not:
every video in this repo is a fixed store camera, product B. The quadruped has never
recorded anything. This script is what makes robot footage exist, and it records the
telemetry alongside it because footage without the ultrasound and odometry that frame is
worth nothing to E-prep -- the whole point is measuring a depth teacher *against* them.

---------------------------------------------------------------------------
THE CLOCKS, WHICH ARE THE ONLY HARD PART

The robot's clock runs about **107 seconds ahead of pro6000's** (measured 2026-08-18; it
has no NTP and drifts on its own). So the one thing this script must never do is stamp a
frame or a sample with the local `time.time()` -- that silently shears video from
telemetry by nearly two minutes and every residual computed afterwards is noise.

It does not have to. Both timestamps available here are *already* on the robot's clock:

  * `/api/telemetry`'s `t` is `time.time()` inside `hydra_dash.py`, on the robot.
  * `#EXT-X-PROGRAM-DATE-TIME` in the HLS playlist is written by mediamtx, on the robot.

So they are directly comparable and no skew correction is needed or wanted. Local time is
recorded once in the manifest, as the measured offset, and used for nothing else.

The gap between them is not skew, it is **pipeline latency**: the live-edge segment starts
about 2.2 s before the telemetry sample you get by asking at the same moment. That is
encode + segment + proxy, it is real, and it is why frames are timed from PDT rather than
from when the fetch returned.

---------------------------------------------------------------------------
WHY IT SPEAKS HLS BY HAND

ffmpeg segfaults on this machine the instant it opens any http:// input (7.0.2 static,
exit 139 before the first read; local files decode fine). Rather than debug someone
else's build, the playlist is parsed here and segments are pulled with plain HTTP, then
concatenated onto the fMP4 init segment. The result is an ordinary mp4 that local ffmpeg
reads happily.

Segments live at the live edge for only a few seconds before mediamtx rotates them out --
fetching a URI seen ten seconds ago returns an empty body, not an error. The polling
interval is therefore well under the ~1 s segment duration, and any segment that comes
back short is recorded as a miss in the manifest instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from datetime import datetime
from itertools import takewhile
from pathlib import Path

DASH = "http://localhost:8080"


def _get(url: str, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return fh.read()


def _robot_clock(stamp: str) -> float:
    """`EXT-X-PROGRAM-DATE-TIME` as a robot-clock epoch.

    mediamtx trims trailing zeros off the fractional second, so it emits `.43` where
    Python 3.10's `fromisoformat` accepts only 3 or 6 digits -- it raised on roughly one
    poll in three, which cost 9 seconds of the first capture. 3.11 parses it fine; this
    runs on 3.10, so the fraction is padded rather than the runtime argued with.
    """
    head, _, tail = stamp.partition(".")
    if not tail:
        return datetime.fromisoformat(stamp).timestamp()
    frac = "".join(takewhile(str.isdigit, tail))
    return datetime.fromisoformat(f"{head}.{frac:0<6}{tail[len(frac) :]}").timestamp()


def parse_playlist(text: str) -> tuple[str | None, list[dict]]:
    """Return (init URI, segments) with each segment timed on the robot's clock.

    `EXT-X-PROGRAM-DATE-TIME` is emitted only for the last couple of segments -- the live
    edge -- so a segment's time is that tag carried forward by the accumulated `EXTINF`
    durations. Anything before the first tag gets `pdt=None` and is dropped by the caller
    rather than guessed at. `EXT-X-GAP` segments are placeholders with no media; they still
    advance the clock, so they are skipped but their duration is not.
    """
    init: str | None = None
    segs: list[dict] = []
    clock: float | None = None
    dur = 0.0
    gap = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-MAP:URI="):
            init = line.split('URI="', 1)[1].split('"', 1)[0]
        elif line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            clock = _robot_clock(line.split(":", 1)[1])
        elif line.startswith("#EXT-X-GAP"):
            gap = True
        elif line.startswith("#EXTINF:"):
            dur = float(line.split(":", 1)[1].rstrip(",").split(",")[0])
        elif not line.startswith("#"):
            if not gap and "_part" not in line:
                segs.append({"uri": line, "pdt": clock, "extinf": dur})
            if clock is not None:
                clock += dur
            gap = False
    return init, segs


def poll_telemetry(url: str, out: Path, stop: threading.Event, hz: float) -> None:
    """Append every snapshot verbatim. `t` inside it is the robot's clock; nothing here
    adds a local timestamp, because a local one would only ever be used by mistake."""
    period = 1.0 / hz
    with out.open("w") as fh:
        while not stop.is_set():
            began = time.monotonic()
            try:
                fh.write(_get(url, timeout=5.0).decode() + "\n")
                fh.flush()
            except Exception as exc:  # a dropped sample is not worth losing the run over
                fh.write(json.dumps({"error": repr(exc)}) + "\n")
            stop.wait(max(0.0, period - (time.monotonic() - began)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dash", default=DASH, help="dashboard base URL (pro6000 forward)")
    ap.add_argument("--out", type=Path, required=True, help="capture directory")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--telemetry-hz", type=float, default=10.0)
    ap.add_argument("--poll", type=float, default=0.4, help="playlist poll interval (s)")
    ap.add_argument("--note", default="", help="what the robot was doing, in words")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    video = out / "raw.mp4"
    stop = threading.Event()

    local_before = time.time()
    snap = json.loads(_get(f"{args.dash}/api/telemetry"))
    local_after = time.time()
    robot_offset = snap["t"] - (local_before + local_after) / 2

    tele = threading.Thread(
        target=poll_telemetry,
        args=(f"{args.dash}/api/telemetry", out / "telemetry.jsonl", stop, args.telemetry_hz),
        daemon=True,
    )
    tele.start()

    seen: set[str] = set()
    records: list[dict] = []
    misses: list[str] = []
    init_written = False
    deadline = time.monotonic() + args.seconds
    print(f"capturing {args.seconds:.0f}s -> {out}  (robot clock is {robot_offset:+.1f}s)")

    with video.open("wb") as vh:
        while time.monotonic() < deadline:
            began = time.monotonic()
            try:
                init, segs = parse_playlist(_get(f"{args.dash}/cam/stream.m3u8").decode())
            except Exception as exc:
                print(f"  playlist: {exc!r}")
                stop.wait(args.poll)
                continue
            if init and not init_written:
                vh.write(_get(f"{args.dash}/cam/{init}"))
                init_written = True
            for seg in segs:
                if seg["uri"] in seen or seg["pdt"] is None:
                    continue
                seen.add(seg["uri"])
                body = _get(f"{args.dash}/cam/{seg['uri']}")
                if len(body) < 1024:  # rotated out before we asked
                    misses.append(seg["uri"])
                    continue
                vh.write(body)
                vh.flush()
                records.append({**seg, "bytes": len(body)})
            stop.wait(max(0.0, args.poll - (time.monotonic() - began)))

    stop.set()
    tele.join(timeout=5.0)

    media = sum(r["extinf"] for r in records)
    manifest = {
        "note": args.note,
        "dash": args.dash,
        "seconds_requested": args.seconds,
        "telemetry_hz": args.telemetry_hz,
        # Every timestamp in segments.jsonl and telemetry.jsonl is on the robot's clock.
        # This is the only number that relates it to this machine's, and it is here so a
        # reader can tell the two apart -- not so anything downstream corrects by it.
        "robot_clock_offset_s": round(robot_offset, 3),
        "local_clock_at_start": local_before,
        "segments": len(records),
        "segments_missed": misses,
        "media_seconds": round(media, 2),
        "pdt_first": records[0]["pdt"] if records else None,
        "pdt_last": records[-1]["pdt"] if records else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (out / "segments.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    lines = (out / "telemetry.jsonl").read_text().count("\n")
    print(f"  {len(records)} segments, {media:.1f}s media, {lines} telemetry samples")
    if misses:
        print(f"  MISSED {len(misses)} segments (rotated out): {misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
