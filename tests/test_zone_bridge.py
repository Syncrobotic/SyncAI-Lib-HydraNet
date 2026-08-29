"""camera.json's zones, as the event layer's zones. The bridge and what it refuses.

`camera_json.Zone` and `events.Zone` both mean "a named region on the floor" and are not
the same object: one is what the camera is, the other carries thresholds a store manager
changes on a Tuesday. `analytics/journey.py` needs the second and every commissioned
camera holds the first, so the conversion is real code with real refusals rather than a
comprehension at each call site.

pytest tests/test_zone_bridge.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from syncai_hydranet.analytics import events as ev
from syncai_hydranet.geometry.camera_json import CameraFile, Zone
from syncai_hydranet.geometry.ground import Camera, GroundPlane

SQUARE = ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))


def a_camera_file(*zones: Zone) -> CameraFile:
    return CameraFile(
        camera_id="Taichung-cam01",
        image_size_px=(960, 540),
        camera=Camera(fx=382.7, fy=382.7, cx=480.0, cy=270.0),
        plane=GroundPlane(height=2.49, pitch=math.radians(49.5)),
        zones=zones,
    )


def test_a_polygon_zone_crosses_with_its_geometry_and_no_policy():
    """Policy stays out: `camera.json` is what the camera is, not what the store rules are."""
    cf = a_camera_file(Zone("front_till", "till", SQUARE))
    (z,) = ev.zones_from_camera(cf)
    assert z.name == "front_till"
    assert z.polygon.shape == (4, 2)
    assert (z.max_occupancy, z.loiter_seconds, z.restricted) == (None, None, False)
    assert bool(z.contains(np.array([[0.5, 1.5]]))[0])
    assert not bool(z.contains(np.array([[5.0, 5.0]]))[0])


def test_an_entrance_line_is_not_returned_as_a_region():
    """A two-point polyline read as a polygon is a zone that never fires and never errors."""
    cf = a_camera_file(
        Zone("front_till", "till", SQUARE),
        Zone("door", "entrance_line", ((-1.0, 2.0), (1.0, 2.0))),
    )
    assert [z.name for z in ev.zones_from_camera(cf)] == ["front_till"]
    (line,) = ev.counting_lines(cf)
    assert line.name == "door"
    assert line.a.tolist() == [-1.0, 2.0] and line.b.tolist() == [1.0, 2.0]


def test_kinds_can_be_narrowed_by_the_caller():
    cf = a_camera_file(
        Zone("front_till", "till", SQUARE),
        Zone("walkable_floor", "walkable", SQUARE),
    )
    assert [z.name for z in ev.zones_from_camera(cf)] == ["front_till", "walkable_floor"]
    narrowed = ev.zones_from_camera(cf, kinds=["till"])
    assert [z.name for z in narrowed] == ["front_till"]


def test_a_three_point_entrance_line_is_refused_rather_than_truncated():
    cf = a_camera_file(Zone("door", "entrance_line", ((-1.0, 2.0), (0.0, 2.0), (1.0, 2.0))))
    with pytest.raises(ValueError, match="one directed segment"):
        ev.counting_lines(cf)


def test_a_camera_with_no_zones_yields_nothing_rather_than_failing():
    cf = a_camera_file()
    assert ev.zones_from_camera(cf) == []
    assert ev.counting_lines(cf) == []


SHIPPED = Path(__file__).resolve().parents[1] / "runs/commission01/Taichung-cam01.camera.json"


@pytest.mark.skipif(not SHIPPED.exists(), reason="needs the commissioned camera.json")
def test_the_shipped_camera_bridges():
    """The real commissioned file, not a fixture -- it is the input this exists for.

    Two things were wrong with the first version of this test and both are the same
    mistake, which is why it is worth a paragraph rather than a diff.

    It read `runs/commission01/...` as a bare relative path. `runs/` is gitignored, so a
    fresh checkout does not have it and CI would have raised `FileNotFoundError` here --
    green only because a direct push to `dev` runs no workflow. `test_indoor25_baseline`
    already carries the pattern for an artefact the repository does not ship, and this
    now uses it: anchored to the repo root, skipped when the artefact is absent.

    And it asserted `== ["walkable_floor"]` against a file that commissioning **rewrites**.
    `tools/commissioning/service_zones.py --apply` added nine `fixture_NN` zones to that
    exact camera on 2026-08-26 and this test went red -- for doing its job, on a file whose
    zone list is an output. What the bridge promises is that every zone in the file crosses
    as a region with geometry and no policy; the *contents* of one camera's zone list are a
    fact about a store on a Wednesday. So the count is not pinned and the property is.
    """
    cf = CameraFile.load(SHIPPED)
    zones = ev.zones_from_camera(cf)
    regions = [z.name for z in cf.zones if z.kind != "entrance_line"]
    assert [z.name for z in zones] == regions, (
        "every region in the file bridges; a kind this camera holds and the bridge drops "
        "would be a zone that silently stops existing at the event layer. `entrance_line` "
        "is the one deliberate omission and `counting_lines` is where it goes instead"
    )
    assert "walkable_floor" in {z.name for z in zones}, (
        "every commissioned camera carries the walkable outline -- if this one does not, "
        "the file was written by something other than the commissioning pipeline"
    )
    for z in zones:
        assert z.polygon.ndim == 2 and z.polygon.shape[1] == 2 and len(z.polygon) >= 3
        assert (z.max_occupancy, z.loiter_seconds, z.restricted) == (None, None, False)
