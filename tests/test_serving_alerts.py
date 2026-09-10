"""The serving path files an alert: canvas boxes in, a disposition row on disk out.

`tests/test_serving_world.py` stops at metres. This continues the chain one rung --
`CameraState.world_frame` -> `events.live.ZoneMonitor` -> `dispositions.record_alert` --
which is the rung docs/PLAN.md section 9.6 found absent from `serving/`.

pytest tests/test_serving_alerts.py -v
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from _cameras import HALF_RES_CAM, HALF_RES_PLANE, HALF_RES_SIZE
from syncai_hydranet.analytics.bytetrack import OfflineForward, as_track
from syncai_hydranet.analytics.policy import load_policy
from syncai_hydranet.geometry.camera_json import CameraFile, Zone
from syncai_hydranet.geometry.ground import ground_to_pixel
from syncai_hydranet.preprocessing import letterbox_region
from syncai_hydranet.serving.alerts import CameraAlerts
from syncai_hydranet.serving.camera import CameraState
from syncai_hydranet.serving.dispositions import file_hash, iter_records

ROOT = Path(__file__).resolve().parents[1]
CANVAS_HW = (640, 1120)
STREAM = (1920, 1080)
FPS = 5.0
MODEL = {"checkpoint": "exports/x.plan", "git": {"commit": "abc"}, "config_hash": "h"}


def square(cx, cz, half=0.5):
    return (
        (cx - half, cz - half),
        (cx + half, cz - half),
        (cx + half, cz + half),
        (cx - half, cz + half),
    )


def a_camera_file(*zones: Zone) -> CameraFile:
    return CameraFile(
        camera_id="Taichung-cam01",
        image_size_px=HALF_RES_SIZE,
        camera=HALF_RES_CAM,
        plane=HALF_RES_PLANE,
        zones=zones,
    )


def canvas_box_for(x_m: float, z_m: float, height_px: float = 178.0) -> np.ndarray:
    """metres -> calibrated px -> stream px -> canvas px; the same path as test_serving_world."""
    u, v, _ = ground_to_pixel(np.array([x_m]), np.array([z_m]), HALF_RES_CAM, HALF_RES_PLANE)
    sx, sy = float(u[0]) * 2.0, float(v[0]) * 2.0
    x0, y0, cw, ch = letterbox_region(*STREAM, CANVAS_HW)
    cu, cv = sx * (cw / STREAM[0]) + x0, sy * (ch / STREAM[1]) + y0
    return np.array([cu - 30.0, cv - height_px, cu + 30.0, cv])


def a_state(camera_file: CameraFile | None) -> CameraState:
    return CameraState(
        camera="Taichung-cam01",
        num_terrain_classes=4,
        canvas_hw=CANVAS_HW,
        det_classes=["person", "bag", "device", "boxed_stock"],
        tracker_factory=lambda: OfflineForward(0.35, 0.20, 0.3, 0.4, 5, 1, 25.0 / FPS),
        camera_file=camera_file,
        source_size_px=None if camera_file is None else STREAM,
    )


def alerts_for(state: CameraState, tmp_path: Path) -> CameraAlerts:
    calib = tmp_path / "cam.json"
    calib.write_text("{}")
    return CameraAlerts(
        state,
        load_policy(ROOT / "configs/policy/demo.yaml"),
        fps=FPS,
        root=tmp_path / "dispositions",
        model=MODEL,
        calib_path=calib,
        clip="datasets/studioa_clips/Taichung-cam01/archive_20260816-030041_20260816-030545.mp4",
        clip_start=datetime(2026, 8, 16, 3, 0, 41, tzinfo=UTC),
    )


def step(state: CameraState, alerts: CameraAlerts, seq: int, positions):
    terrain = np.zeros(CANVAS_HW, dtype=np.uint8)
    boxes = (
        np.stack([canvas_box_for(x, z) for x, z in positions])
        if positions
        else np.zeros((0, 4))
    )
    state.update(
        seq,
        terrain,
        boxes,
        np.full(len(positions), 0.9),
        np.zeros(len(positions), dtype=np.int64),
    )
    return alerts.on_frame([as_track(f) for f in state.tracker.tracks])


def test_a_shopper_standing_at_a_display_table_files_one_loitering_row(tmp_path):
    state = a_state(a_camera_file(Zone("table_01", "display", square(0.0, 3.0))))
    alerts = alerts_for(state, tmp_path)
    assert alerts.zones == 1
    filed = []
    for i in range(60):  # 12 s at 5 fps against the demo policy's 8 s
        filed += step(state, alerts, i, [(0.0, 3.0)])
    assert [r.event["type"] for r in filed] == ["loitering"]
    (rec,) = filed
    assert rec.event["zone"] == "table_01" and rec.event["frame_start"] == 0
    assert rec.event["frame_end"] == 39 and rec.event["value"] == 8.0  # the crossing
    assert rec.event["started_at"] is not None  # stamped from the clip start
    # The row on disk carries what a reviewer months later needs.
    (row,) = list(iter_records(tmp_path / "dispositions", kinds=("alert",)))
    assert row.alert_id == rec.alert_id == alerts.filed[0]
    assert row.frame_ref["clip"].endswith("archive_20260816-030041_20260816-030545.mp4")
    assert row.calib_version == file_hash(tmp_path / "cam.json")
    assert row.model == MODEL


def test_walking_past_files_nothing_and_the_store_stays_empty(tmp_path):
    state = a_state(a_camera_file(Zone("table_01", "display", square(0.0, 3.0))))
    alerts = alerts_for(state, tmp_path)
    filed = []
    for i, x in enumerate(np.linspace(-3.0, 3.0, 8)):
        filed += step(state, alerts, i, [(float(x), 3.0)])
    assert filed == [] and alerts.filed == []
    assert not (tmp_path / "dispositions").exists()


def test_a_camera_without_geometry_is_refused_not_silently_served(tmp_path):
    with pytest.raises(ValueError, match="no commissioned geometry"):
        alerts_for(a_state(None), tmp_path)


def test_an_alert_with_no_footage_to_pull_is_refused_at_construction(tmp_path):
    state = a_state(a_camera_file(Zone("t", "display", square(0.0, 3.0))))
    calib = tmp_path / "cam.json"
    calib.write_text("{}")
    with pytest.raises(ValueError, match="clip path or a stream start"):
        CameraAlerts(state, load_policy(ROOT / "configs/policy/demo.yaml"), fps=FPS,
                     root=tmp_path, model=MODEL, calib_path=calib)  # fmt: skip


def test_the_row_is_one_json_line_a_person_with_grep_can_read(tmp_path):
    state = a_state(a_camera_file(Zone("table_01", "display", square(0.0, 3.0))))
    alerts = alerts_for(state, tmp_path)
    for i in range(45):
        step(state, alerts, i, [(0.0, 3.0)])
    (day,) = sorted((tmp_path / "dispositions").glob("*.jsonl"))
    lines = [json.loads(line) for line in day.read_text().splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0]["kind"] == "alert"
