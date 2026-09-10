"""L3 on the serving path: the last consumed frame's tracks in, alert rows filed.

docs/PLAN.md section 9.6 found, on 2026-09-09, that `world_frame`, `zone_events` and
`record_alert` appeared zero times under `serving/` -- the chain had run end to end only
as an offline script, and the serving path stopped at L0 plus a tracker. This is the
missing rung, and it is deliberately thin: three calls the offline runner already makes,
in the order it makes them, so that a recorded clip pushed through the pilot files the
same kind of row `scripts/step6_events.py` files.

    CameraState.world_frame  ->  events.live.ZoneMonitor.step  ->  dispositions.record_alert
             L1                          L3                              the store

One `CameraAlerts` per camera, because everything in it is per camera: the zones with
the store's policy on them, the calibration hash every row carries, the clip or stream
the row's `frame_ref` points into. A camera whose `CameraState` cannot measure metres is
refused at construction rather than filing nothing quietly -- the caller decides whether
an uncommissioned camera serving detection only is acceptable, and says so.

Durations reach the monitor as frames over ``fps``, which is the rate frames are
*consumed* at, not decoded at. `scripts/serve_pilot.py` paces each stream at
`--stream-fps` and consumes every frame when the ticks keep up, so the two agree there;
a deployment that drops frames under load has to pass the consumption rate it measures,
or the seconds on its alerts will be short by the drop rate. Reading PTS instead is the
right fix and is not done here, for the reason `events/live.py` gives: the offline rule
the log is compared against counts frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from syncai_hydranet.analytics.events._types import with_clip_start
from syncai_hydranet.analytics.events.live import ZoneMonitor
from syncai_hydranet.analytics.policy import StorePolicy
from syncai_hydranet.analytics.tracker import Track
from syncai_hydranet.serving.camera import CameraState
from syncai_hydranet.serving.dispositions import AlertRecord, file_hash, record_alert


class CameraAlerts:
    """One camera's L3: its zones under the store's policy, filing into one store."""

    def __init__(
        self,
        state: CameraState,
        policy: StorePolicy,
        *,
        fps: float,
        root: str | Path,
        model: Mapping[str, Any],
        calib_path: str | Path,
        clip: str | Path | None = None,
        clip_start: datetime | None = None,
    ) -> None:
        if not state.measures_metres:
            raise ValueError(
                f"{state.camera}: this CameraState has no commissioned geometry, so it "
                "cannot place a shopper on the floor and no metric alert can be filed "
                "for it. Serve it without L3, and say so in the report."
            )
        if clip is None and clip_start is None:
            raise ValueError(
                f"{state.camera}: an alert needs a clip path or a stream start time, or "
                "the footage behind it cannot be pulled for review (record_alert refuses "
                "the row; refused here so the first alert is not the first error)"
            )
        self.state = state
        self.policy = policy
        assert state.camera_file is not None  # measures_metres
        self.monitor = ZoneMonitor(
            policy.zones_for(state.camera_file),
            fps=fps,
            camera=state.camera,
            min_seconds=policy.min_seconds,
        )
        self.root = Path(root)
        self.model = dict(model)
        self.calib_version = file_hash(calib_path)
        self.clip = None if clip is None else str(clip)
        self.clip_start = clip_start
        self.filed: list[str] = []

    @property
    def zones(self) -> int:
        return len(self.monitor.zones)

    def on_frame(self, tracks: Sequence[Track]) -> list[AlertRecord]:
        """The camera's tracks after one `CameraState.update`; the rows that fired."""
        frame = self.state.world_frame(tracks, name="person")
        if frame is None:
            return []
        events = with_clip_start(self.monitor.step(frame), self.clip_start)
        records = [
            record_alert(
                self.root, e, model=self.model, calib=self.calib_version, clip=self.clip
            )
            for e in events
        ]
        self.filed += [r.alert_id for r in records]
        return records
