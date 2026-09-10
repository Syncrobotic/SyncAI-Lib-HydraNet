"""The grading surface's pure parts: where the frame is, what is drawn, what a verdict does.

The HTTP layer is a stdlib server around these; ffmpeg is not run here.

pytest tests/test_review_server.py -v
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from _cameras import HALF_RES_CAM, HALF_RES_PLANE, HALF_RES_SIZE
from syncai_hydranet.analytics.events._types import SecurityEvent
from syncai_hydranet.geometry.camera_json import CameraFile, Zone
from syncai_hydranet.serving.dispositions import (
    current_dispositions,
    iter_records,
    record_alert,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "review_server", ROOT / "deploy/retail-security/review_server.py"
)
assert _spec is not None and _spec.loader is not None
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

MODEL = {"checkpoint": "exports/x.plan", "git": {"commit": "abc"}, "config_hash": "h"}
SQUARE = ((-0.5, 2.5), (0.5, 2.5), (0.5, 3.5), (-0.5, 3.5))
CAM = CameraFile(
    camera_id="Taichung-cam01",
    image_size_px=HALF_RES_SIZE,
    camera=HALF_RES_CAM,
    plane=HALF_RES_PLANE,
    zones=(Zone("table_01", "display", SQUARE),),
)


def an_event(**over) -> dict:
    e = SecurityEvent(
        type="loitering", camera="Taichung-cam01", frame_start=0, frame_end=39, fps=5.0,
        track_ids=(1,), zone="table_01", value=8.0, threshold=8.0,
        basis="seconds a single track stayed inside the polygon",
        extra={"filed_at": "crossing", "x_m": 0.0, "z_m": 3.0},
        clip_start=datetime(2026, 8, 16, 3, 0, 41, tzinfo=UTC),
    )  # fmt: skip
    row = e.as_row()
    row.update(over)
    return row


def test_the_frame_is_the_crossing_frame_and_a_looped_clip_wraps():
    ev = an_event()
    assert rs.event_fps(ev) == 5.0
    assert rs.frame_time(ev, 304.0) == (7.8, False)
    assert rs.frame_time(an_event(frame_end=1600, frame_start=1561), 304.0) == (
        pytest.approx(1600 / 5.0 - 304.0),
        True,
    )
    assert rs.frame_time(ev, None) == (7.8, False)


def test_the_person_box_stands_on_the_recorded_floor_point_and_is_a_person_tall():
    frame_wh = (1920, 1080)  # the clip is twice the calibrated frame
    box = rs.person_box(an_event(), CAM, frame_wh)
    assert box is not None
    x0, y0, x1, y1 = box
    assert y0 < y1 and x0 < x1
    assert 100 < (y1 - y0) <= 1080, "a 1.8 m person at 3 m is a few hundred rows tall"
    assert y0 >= 0.0  # a head above the frame is clipped to its top row, not negative
    # The foot marker is where (0, 3) projects, in the frame's own pixels.
    feet, _ = rs.to_frame_pixels(np.array([[0.0, 3.0]]), CAM, frame_wh)
    assert abs((x0 + x1) / 2 - feet[0, 0]) < 1.0 and abs(y1 - feet[0, 1]) < 1.0


def test_an_occupancy_row_has_no_position_and_so_no_box():
    ev = an_event(type="occupancy_exceeded", track_ids=(), x_m=None, z_m=None)
    assert rs.person_box(ev, CAM, (1920, 1080)) is None
    assert rs.zone_outline(ev, CAM, (1920, 1080)) is not None  # the zone still draws


def test_drawing_blurs_the_head_before_anything_is_drawn_on_top():
    frame = Image.effect_noise((1920, 1080), 64).convert("RGB")  # every pixel distinct
    before = np.asarray(frame).astype(float)
    out = np.asarray(rs.draw_review_frame(frame, an_event(), CAM)).astype(float)
    x0, y0, x1, y1 = rs.person_box(an_event(), CAM, (1920, 1080))
    head = (slice(int(y0) + 4, int(y0 + (y1 - y0) * 0.3)), slice(int(x0) + 4, int(x1) - 4))
    # Local variance of noise collapses under a Gaussian blur; away from the box it does not.
    assert out[head].std() < before[head].std() * 0.5
    far = (slice(20, 120), slice(20, 120))
    assert abs(out[far].std() - before[far].std()) < 2.0


def test_a_verdict_lands_in_the_store_as_a_disposition_row(tmp_path):
    root = tmp_path / "d"
    rec = record_alert(root, an_event(), model=MODEL, clip="x.mp4")
    store = rs.Store(root, tmp_path, by="tester")
    assert [(r.alert_id, v) for r, v in store.alerts()] == [(rec.alert_id, "unreviewed")]
    store.file_verdict(rec.alert_id, "rejected", "staff")
    assert store.alerts()[0][1] == "rejected"
    d = current_dispositions(list(iter_records(root)))[rec.alert_id]
    assert (d.by, d.reason) == ("tester", "staff")
    with pytest.raises(ValueError, match="verdict"):
        store.file_verdict(rec.alert_id, "maybe", None)
    page = rs.render_index(store, "all")
    assert rec.alert_id in page and "rejected" in page and "tester" in page
