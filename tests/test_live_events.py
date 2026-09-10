"""The live event monitor produces the offline rules' events, at the crossing.

The contract is stated in `analytics/events/live.py`: same type, zone, track ids and
`frame_start` as `zones.zone_events` / `zones.occupancy_events` over the same tracks;
`frame_end` and `value` are the crossing's rather than the run's. Every case below builds
finished tracks, runs both, and compares.

pytest tests/test_live_events.py -v
"""

from __future__ import annotations

import numpy as np

from _cameras import HALF_RES_CAM, HALF_RES_PLANE, HALF_RES_SIZE
from syncai_hydranet.analytics import events as ev
from syncai_hydranet.analytics.events.live import ZoneMonitor
from syncai_hydranet.analytics.tracker import Track
from syncai_hydranet.analytics.world import WorldFrame, world_frames
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import ground_to_pixel

FPS = 5.0
CAM_FILE = CameraFile(
    camera_id="Taichung-cam01",
    image_size_px=HALF_RES_SIZE,
    camera=HALF_RES_CAM,
    plane=HALF_RES_PLANE,
)


def square(cx: float, cz: float, half: float = 0.5) -> np.ndarray:
    return np.array([[cx - half, cz - half], [cx + half, cz - half],
                     [cx + half, cz + half], [cx - half, cz + half]])  # fmt: skip


def box_at(x_m: float, z_m: float) -> np.ndarray:
    u, v, _ = ground_to_pixel(np.array([x_m]), np.array([z_m]), HALF_RES_CAM, HALF_RES_PLANE)
    return np.array([float(u[0]) - 20.0, float(v[0]) - 120.0, float(u[0]) + 20.0, float(v[0])])


def track(track_id: int, positions: dict[int, tuple[float, float]]) -> Track:
    """A finished track observed at exactly the given frames, standing at those metres."""
    frames = sorted(positions)
    boxes = [box_at(*positions[f]) for f in frames]
    return Track(track_id=track_id, box=boxes[-1], frames=frames, boxes=boxes, confirmed=True)


def replay(tracks: list[Track]) -> list[WorldFrame]:
    """Every frame index in range, including the empty ones the monitor must see."""
    frames = world_frames(tracks, CAM_FILE, name="person", fps=FPS)
    by_index = {f["frame_index"]: f for f in frames}
    lo, hi = min(by_index), max(by_index)
    empty = {"time_s": None, "camera_id": CAM_FILE.camera_id, "space": frames[0]["space"]}
    return [
        by_index.get(i, WorldFrame(frame_index=i, objects=[], **empty))
        for i in range(lo, hi + 1 + 1)  # one past the end, so a final span can close
    ]


def offline(tracks: list[Track], zones: list[ev.Zone]) -> list[ev.SecurityEvent]:
    out = ev.zone_events(tracks, zones, HALF_RES_CAM, HALF_RES_PLANE, FPS, "Taichung-cam01")
    for z in zones:
        if z.max_occupancy is not None:
            out += ev.occupancy_events(
                tracks, z, HALF_RES_CAM, HALF_RES_PLANE, FPS, "Taichung-cam01"
            )
    return out


def live(tracks: list[Track], zones: list[ev.Zone]) -> list[ev.SecurityEvent]:
    mon = ZoneMonitor(zones, fps=FPS, camera="Taichung-cam01")
    out: list[ev.SecurityEvent] = []
    for frame in replay(tracks):
        out += mon.step(frame)
    return out


def key(e: ev.SecurityEvent) -> tuple:
    return (e.type, e.zone, e.track_ids, e.frame_start)


def assert_same_events(tracks, zones):
    off, lv = offline(tracks, zones), live(tracks, zones)
    assert sorted(map(key, lv)) == sorted(map(key, off))
    by_key = {key(e): e for e in off}
    for e in lv:
        o = by_key[key(e)]
        assert e.frame_end <= o.frame_end, "live fires at the crossing, never after the run"
        assert e.threshold == o.threshold and e.basis == o.basis
        assert e.value is not None and o.value is not None
        assert e.value >= e.threshold, "a crossing has cleared its threshold"
        assert e.value <= o.value, "the crossing's value never exceeds the run's total"
        assert e.extra["filed_at"] == "crossing"
    return off, lv


A = ev.Zone(name="table", polygon=square(0.0, 3.0), loiter_seconds=8.0)
B = ev.Zone(name="stockroom", polygon=square(1.5, 3.0), restricted=True)
FLOOR = ev.Zone(name="floor", polygon=square(0.0, 3.0, half=2.0), max_occupancy=2)


def test_a_shopper_standing_at_a_table_loiters_once_and_the_alert_fires_at_eight_seconds():
    t = track(1, dict.fromkeys(range(60), (0.0, 3.0)))  # 12 s
    off, lv = assert_same_events([t], [A])
    (e,) = lv
    assert e.type == "loitering" and e.frame_start == 0
    assert e.frame_end == 39 and e.value == 8.0  # 40 frames at 5 fps
    assert off[0].value == 12.0


def test_walking_through_a_zone_fires_nothing_in_either():
    t = track(2, {0: (-3.0, 3.0), 1: (0.0, 3.0), 2: (0.0, 3.0), 3: (3.0, 3.0)})
    off, lv = assert_same_events([t], [A, B])
    assert off == [] and lv == []


def test_a_gap_in_observation_does_not_break_a_run_in_either():
    """Coasting frames are not in `Track.frames`; the run continues across them."""
    positions = dict.fromkeys(range(20), (0.0, 3.0)) | dict.fromkeys(range(30, 60), (0.0, 3.0))
    t = track(3, positions)
    off, lv = assert_same_events([t], [A])
    assert len(lv) == 1 and lv[0].frame_start == 0
    assert off[0].value == 12.0  # 60 frames, gap included, exactly as offline counts it


def test_stepping_out_and_back_in_is_two_runs_in_both():
    positions = dict.fromkeys(range(45), (0.0, 3.0)) | {45: (5.0, 3.0)}
    positions |= dict.fromkeys(range(46, 100), (0.0, 3.0))
    t = track(4, positions)
    _, lv = assert_same_events([t], [A])
    assert sorted(e.frame_start for e in lv) == [0, 46]


def test_a_restricted_zone_fires_intrusion_after_the_jitter_floor():
    t = track(5, dict.fromkeys(range(10), (1.5, 3.0)))  # 2 s in the stockroom
    _, lv = assert_same_events([t], [B])
    (e,) = lv
    assert e.type == "zone_intrusion" and e.frame_end == 4 and e.value == 1.0


def test_occupancy_fires_once_per_span_with_the_count_so_far():
    three = [track(i, dict.fromkeys(range(5, 20), (0.2 * i, 3.0))) for i in (6, 7, 8)]
    off, lv = assert_same_events(three, [FLOOR])
    (e,) = lv
    assert e.type == "occupancy_exceeded"
    assert e.frame_start == 5 and e.frame_end == 14 and e.value == 3.0  # 10 frames = 2 s
    assert off[0].frame_end == 19


def test_an_unobserved_object_neither_extends_nor_breaks_a_run():
    mon = ZoneMonitor([A], fps=FPS, camera="c")
    base = {"time_s": None, "camera_id": "c", "space": "camera_floor(c)"}

    def frame(i, observed):
        obj = {"track_id": 9, "name": "person", "x_m": 0.0, "z_m": 3.0, "vx_ms": None,
               "vz_ms": None, "yaw_rad": None, "height_m": None, "observed": observed,
               "basis": "foot_point", "score": None}  # fmt: skip
        return WorldFrame(frame_index=i, objects=[obj], **base)

    fired = []
    for i in range(39):
        fired += mon.step(frame(i, observed=True))
    fired += mon.step(frame(39, observed=False))  # a coasted frame: no crossing yet
    assert fired == []
    fired += mon.step(frame(40, observed=True))
    assert [e.type for e in fired] == ["loitering"] and fired[0].frame_start == 0


def test_a_forgotten_track_starts_a_fresh_run_when_it_reappears():
    mon = ZoneMonitor([A], fps=FPS, camera="c", forget_after_frames=10)
    t = track(
        10, dict.fromkeys(range(5), (0.0, 3.0)) | dict.fromkeys(range(50, 100), (0.0, 3.0))
    )
    starts = [e.frame_start for f in replay([t]) for e in mon.step(f)]
    assert starts == [50]  # the first five frames were forgotten, so the run restarts
