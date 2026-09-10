"""A store's policy as a file: what it puts on a zone, and what it refuses.

pytest tests/test_policy.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from _cameras import HALF_RES_CAM, HALF_RES_PLANE, HALF_RES_SIZE
from syncai_hydranet.analytics.policy import POLICY_SCHEMA, ZoneRule, load_policy
from syncai_hydranet.geometry.camera_json import CameraFile, Zone

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "configs/policy/demo.yaml"
SQUARE = ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))


def a_camera_file(*zones: Zone) -> CameraFile:
    return CameraFile(
        camera_id="Taichung-cam01",
        image_size_px=HALF_RES_SIZE,
        camera=HALF_RES_CAM,
        plane=HALF_RES_PLANE,
        zones=zones,
    )


def write_policy(tmp_path: Path, **over) -> Path:
    doc = {
        "schema": POLICY_SCHEMA,
        "store": "t",
        "provenance": "a test",
        "timezone_hours": 8,
        "open_hours": [10, 22],
        "min_seconds": 1.0,
        "zones": {"display": {"loiter_seconds": 8}, "walkable": {"max_occupancy": 4}},
    }
    doc.update(over)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_the_demo_policy_is_the_step6_defaults_it_replaced():
    """The numbers runs/step6_fleet01 and 02 were produced under, so they re-run unchanged."""
    pol = load_policy(DEMO)
    assert pol.rules["display"].loiter_seconds == 8
    assert pol.rules["walkable"].max_occupancy == 4
    assert pol.min_seconds == 1.0
    assert pol.open_hours == (10, 22)
    assert pol.timezone_hours == 8
    assert pol.provenance  # a policy states where its numbers came from


def test_a_rule_lands_on_every_zone_of_its_kind_and_nowhere_else(tmp_path):
    pol = load_policy(write_policy(tmp_path))
    cf = a_camera_file(
        Zone("t1", "display", SQUARE),
        Zone("t2", "display", SQUARE),
        Zone("floor", "walkable", SQUARE),
        Zone("door", "stockroom_door", SQUARE),
        Zone("in", "entrance_line", ((0.0, 0.0), (1.0, 0.0))),
    )
    zones = {z.name: z for z in pol.zones_for(cf)}
    assert set(zones) == {"t1", "t2", "floor", "door"}  # the line is not a polygon
    assert zones["t1"].loiter_seconds == zones["t2"].loiter_seconds == 8
    assert zones["floor"].max_occupancy == 4 and zones["floor"].loiter_seconds is None
    # No rule for the door: defaults, so it never fires, which is the camera-only state.
    assert (
        zones["door"].loiter_seconds,
        zones["door"].max_occupancy,
        zones["door"].restricted,
    ) == (None, None, False)
    assert zones["t1"].polygon.shape == (4, 2)  # geometry untouched


def test_is_open_reads_the_store_clock_not_utc(tmp_path):
    pol = load_policy(write_policy(tmp_path))
    # 16:02 UTC is 00:02 in the store -- the closed-hours clip step 6's re-read found.
    assert not pol.is_open(datetime(2026, 8, 16, 16, 2, tzinfo=UTC))
    assert pol.is_open(datetime(2026, 8, 16, 3, 0, tzinfo=UTC))  # 11:00 local
    with pytest.raises(ValueError, match="aware"):
        pol.is_open(datetime(2026, 8, 16, 3, 0))


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"schema": "store_policy/0"}, "schema"),
        ({"provenance": ""}, "provenance"),
        ({"store": ""}, "store"),
        ({"open_hours": [22, 10]}, "open_hours"),
        ({"open_hours": [10]}, "open_hours"),
        ({"zones": {"lobby": {"loiter_seconds": 8}}}, "not one of"),
        ({"zones": {"entrance_line": {"restricted": True}}}, "not one of"),
        ({"zones": {"display": {}}}, "sets nothing"),
        ({"zones": {"display": {"dwell_seconds": 8}}}, "does not read"),
        ({"zones": {"display": {"loiter_seconds": 0}}}, "loiter_seconds"),
        ({"zones": {"walkable": {"max_occupancy": 0}}}, "max_occupancy"),
        ({"min_seconds": -1}, "min_seconds"),
    ],
)
def test_what_a_policy_file_refuses(tmp_path, over, match):
    with pytest.raises(ValueError, match=match):
        load_policy(write_policy(tmp_path, **over))


def test_a_rule_that_names_no_threshold_is_refused_directly():
    with pytest.raises(ValueError, match="sets nothing"):
        ZoneRule(kind="till")
