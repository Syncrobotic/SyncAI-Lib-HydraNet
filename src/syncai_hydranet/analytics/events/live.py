"""The event layer one frame at a time, for the serving path.

`zones.zone_events` and `zones.occupancy_events` read **finished tracks**: a clip has been
tracked, every run of frames inside a polygon is known end to end, and the event carries
the whole duration. That is the right instrument for a log that is re-read at four
thresholds (`scripts/step6_reread.py`), and it cannot raise an alert, because an alert
has to be raised while the shopper is still standing there. This is the same rules over
`world.WorldFrame`s as they arrive, and it fires **at the crossing**: the frame in which a
run's duration first clears the threshold.

---------------------------------------------------------------------------
THE CONTRACT WITH THE OFFLINE RULES, STATED SO IT CAN BE TESTED

Over the same tracks, the offline functions and this monitor produce the **same set of
events** -- same type, zone, track ids and `frame_start` -- and differ in two fields, both
deliberately:

* `frame_end` here is the crossing frame; offline it is the run's last frame.
* `value` here is the duration (or the count) *at the crossing*; offline it is the run's
  total (or peak). A live loitering alert reads "8.0 s of 8.0" and the offline row for the
  same shopper reads "46.8 s of 8.0", and both are true when they were written.
* A per-track event's `extra` carries the shopper's floor position at the crossing
  (`x_m`, `z_m`), which the offline row does not have and a review sheet needs: the
  alert row carries no box, and the position is what lets a reviewer's frame be drawn
  -- and the person on it blurred -- from the calibration alone.

`tests/test_live_events.py` holds the monitor to exactly that. The semantics that make it
hold, copied from the offline code rather than re-derived:

* A run is broken only by an **observed** foot point outside the polygon (or one the
  geometry refused, which `Zone.contains` reads as outside). Frames where the track was
  not observed -- coasting, or absent from the frame -- neither extend nor break a run:
  `Track.frames` records observed frames only, and `zones._runs` splits on consecutive
  *observations*, so a gap in observation with the shopper still inside afterwards is one
  run, gap included. `observed=False` objects are therefore skipped here.
* Duration is `(frame - run_start + 1) / fps`, frames being the unit of record
  (`stage.StageFrame`), and `time_s` is not used for it: the offline rule does not, and a
  PTS that rebases on reconnect would make the two disagree in a way nobody could audit.
* Occupancy counts observed objects inside per frame and a span is every consecutive
  frame with the count over the limit, so **the caller must `step` every consumed frame,
  including frames with no objects at all** -- a frame not stepped is a frame the monitor
  cannot see the count drop in. Offline, `occupancy_events` reads missing frames as zero
  for the same reason.

---------------------------------------------------------------------------
WHAT IT DOES NOT DO

No `support` on the events: the `TrackSupport` a row carries offline is computed over the
track's whole score series, which a live frame does not have. `None` means "not
recorded", which is what `SecurityEvent.support` documents.

No forgetting of a track's runs until `forget_after_frames` of absence, and that number
has to be at least the tracker's `max_age` or a coasting shopper's run would be cut where
the offline rule keeps it. It is a memory bound, not a rule: track ids are never reused,
so a forgotten run can only belong to a track that is gone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from syncai_hydranet.analytics.events._types import SecurityEvent, Zone
from syncai_hydranet.analytics.world import WorldFrame

#: Frames of absence after which a track's runs are dropped. Above `serving/camera.py`'s
#: `TIME_BASE_FRAMES` (64), so a track that is still on the tracker's time base cannot
#: have been forgotten here.
FORGET_AFTER_FRAMES = 96


@dataclass
class _Run:
    start: int
    last: int
    fired: set[str] = field(default_factory=set)


@dataclass
class _Span:
    start: int
    peak: int
    fired: bool = False


class ZoneMonitor:
    """One camera's zones, fed one `WorldFrame` at a time. See the module docstring."""

    def __init__(
        self,
        zones: list[Zone],
        *,
        fps: float,
        camera: str,
        min_seconds: float = 1.0,
        occupancy_min_seconds: float = 2.0,
        forget_after_frames: int = FORGET_AFTER_FRAMES,
    ) -> None:
        if not fps > 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.zones = list(zones)
        self.fps = float(fps)
        self.camera = camera
        self.min_seconds = float(min_seconds)
        self.occupancy_min_seconds = float(occupancy_min_seconds)
        self.forget_after_frames = int(forget_after_frames)
        self._runs: dict[tuple[int, str], _Run] = {}
        self._spans: dict[str, _Span] = {}
        self._last_seen: dict[int, int] = {}

    def _seconds(self, start: int, frame: int) -> float:
        return (frame - start + 1) / self.fps

    def step(self, frame: WorldFrame) -> list[SecurityEvent]:
        """Advance one frame. Returns the events that crossed their threshold on it."""
        fi = int(frame["frame_index"])
        observed = [o for o in frame["objects"] if o["observed"]]
        events: list[SecurityEvent] = []
        if observed:
            pts = np.array([[o["x_m"], o["z_m"]] for o in observed], dtype=float)
        else:
            pts = np.zeros((0, 2))
        for o in observed:
            self._last_seen[o["track_id"]] = fi

        for zone in self.zones:
            inside = zone.contains(pts) if len(pts) else np.zeros(0, dtype=bool)
            count = 0
            for o, is_in in zip(observed, inside, strict=True):
                key = (o["track_id"], zone.name)
                if not is_in:
                    self._runs.pop(key, None)
                    continue
                count += 1
                run = self._runs.get(key)
                if run is None:
                    run = self._runs[key] = _Run(start=fi, last=fi)
                run.last = fi
                seconds = self._seconds(run.start, fi)
                if (
                    zone.restricted
                    and "zone_intrusion" not in run.fired
                    and seconds >= self.min_seconds
                ):
                    run.fired.add("zone_intrusion")
                    events.append(
                        SecurityEvent(
                            type="zone_intrusion",
                            camera=self.camera,
                            frame_start=run.start,
                            frame_end=fi,
                            fps=self.fps,
                            track_ids=(o["track_id"],),
                            zone=zone.name,
                            value=seconds,
                            threshold=self.min_seconds,
                            basis="seconds a tracked foot point stayed inside the polygon",
                            extra={"filed_at": "crossing", "x_m": o["x_m"], "z_m": o["z_m"]},
                        )
                    )
                if (
                    zone.loiter_seconds is not None
                    and "loitering" not in run.fired
                    and seconds >= zone.loiter_seconds
                ):
                    run.fired.add("loitering")
                    events.append(
                        SecurityEvent(
                            type="loitering",
                            camera=self.camera,
                            frame_start=run.start,
                            frame_end=fi,
                            fps=self.fps,
                            track_ids=(o["track_id"],),
                            zone=zone.name,
                            value=seconds,
                            threshold=zone.loiter_seconds,
                            basis="seconds a single track stayed inside the polygon",
                            extra={"filed_at": "crossing", "x_m": o["x_m"], "z_m": o["z_m"]},
                        )
                    )

            if zone.max_occupancy is None:
                continue
            if count > zone.max_occupancy:
                span = self._spans.get(zone.name)
                if span is None:
                    span = self._spans[zone.name] = _Span(start=fi, peak=count)
                span.peak = max(span.peak, count)
                if (
                    not span.fired
                    and self._seconds(span.start, fi) >= self.occupancy_min_seconds
                ):
                    span.fired = True
                    events.append(
                        SecurityEvent(
                            type="occupancy_exceeded",
                            camera=self.camera,
                            frame_start=span.start,
                            frame_end=fi,
                            fps=self.fps,
                            zone=zone.name,
                            value=float(span.peak),
                            threshold=float(zone.max_occupancy),
                            basis="distinct track ids whose foot point was inside the polygon",
                            extra={"filed_at": "crossing"},
                        )
                    )
            else:
                self._spans.pop(zone.name, None)

        # The memory bound. Track ids are never reused, so a run dropped here belongs to
        # a track the tracker retired long ago.
        gone = [
            t for t, last in self._last_seen.items() if fi - last > self.forget_after_frames
        ]
        for t in gone:
            del self._last_seen[t]
            for key in [k for k in self._runs if k[0] == t]:
                del self._runs[key]
        return events
