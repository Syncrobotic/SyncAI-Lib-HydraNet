"""Metric geometry, checked against scenes whose answer is known by construction.

Every test builds a synthetic depth image where the right answer can be written down --
a floor at a known height, a box at a known distance -- so a failure points at the
arithmetic rather than at the model. The two properties worth protecting are that a
metre in equals a metre out, and that missing depth stays missing instead of being
filled in.

pytest tests/test_bev_scene.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bev_scene import (
    BLOCKED,
    CELL_BLOCKED,
    CELL_EMPTY,
    CELL_GO,
    GO,
    backproject,
    bev_grid,
    build_scene,
    object_from_box,
)

H, W = 60, 80
K = np.array([[50.0, 0.0, W / 2], [0.0, 50.0, H / 2], [0.0, 0.0, 1.0]])


def flat_depth(z: float = 3.0) -> np.ndarray:
    return np.full((H, W), z, dtype=np.float32)


# ------------------------------------------------------------------- back-projection


def test_the_principal_point_maps_to_the_optical_axis():
    pts = backproject(flat_depth(2.5), K)
    centre = pts[H // 2, W // 2]
    assert centre[2] == pytest.approx(2.5)
    assert centre[0] == pytest.approx(0.0)
    assert centre[1] == pytest.approx(0.0)


def test_a_pixel_offset_becomes_the_metre_offset_the_intrinsics_imply():
    """One focal length to the right of centre is one depth-unit to the right."""
    pts = backproject(flat_depth(2.0), K)
    u = W // 2 + int(K[0, 0]) // 2  # half a focal length
    assert pts[H // 2, u][0] == pytest.approx(2.0 * 0.5, abs=1e-3)


def test_using_the_wrong_ks_scale_moves_the_whole_scene():
    """The stride trap: K for full resolution applied to a subsampled image is
    self-consistent and wrong by exactly that factor -- no error, just a scene at the
    wrong size. Halving fx and fy doubles every lateral offset."""
    half_k = K.copy()
    half_k[0, 0] /= 2
    half_k[1, 1] /= 2
    full = backproject(flat_depth(3.0), K)
    half = backproject(flat_depth(3.0), half_k)
    assert abs(half[0, 0][0]) == pytest.approx(2 * abs(full[0, 0][0]), rel=1e-5)


# -------------------------------------------------------------------------- the grid


def test_a_wall_of_floor_lands_at_the_right_distance():
    trav = np.full((H, W), GO, dtype=np.uint8)
    grid = bev_grid(backproject(flat_depth(3.0), K), trav, cell=0.5)
    cells = np.array(grid["cells"]).reshape(grid["nz"], grid["nx"])
    occupied_rows = np.unique(np.nonzero(cells == CELL_GO)[0])
    assert occupied_rows.tolist() == [int(3.0 / 0.5)]


def test_blocked_wins_over_walkable_in_a_shared_cell():
    """Under-reporting an obstacle is the expensive direction of this error, so a cell
    holding any blocked point is blocked even if most of its points are floor."""
    pts = backproject(flat_depth(2.0), K)
    trav = np.full((H, W), GO, dtype=np.uint8)
    centre = np.abs(pts[..., 0]) < 0.05  # a thin column inside one cell
    trav[centre] = BLOCKED
    grid = bev_grid(pts, trav, cell=0.5)
    cells = np.array(grid["cells"]).reshape(grid["nz"], grid["nx"])
    row = cells[int(2.0 / 0.5)]
    middle = grid["nx"] // 2
    assert row[middle] == CELL_BLOCKED, "the mixed cell must not report walkable"
    assert CELL_GO in row.tolist(), "the cells either side are still floor"


def test_pixels_without_depth_leave_a_hole_rather_than_floor():
    """The failure this exists to prevent: a confident segmentation head plus a blind
    sensor drawing certainty over a polished floor."""
    depth = flat_depth(3.0)
    depth[:, : W // 2] = 0.0  # specular half returns nothing
    trav = np.full((H, W), GO, dtype=np.uint8)
    grid = bev_grid(backproject(depth, K), trav, cell=0.5)
    cells = np.array(grid["cells"]).reshape(grid["nz"], grid["nx"])
    left = cells[:, : grid["nx"] // 2]
    assert (left == CELL_EMPTY).all(), "blind pixels must not become floor"
    assert grid["blind_fraction"] == pytest.approx(0.5, abs=0.02)


def test_beyond_the_extent_is_dropped_not_clamped():
    grid = bev_grid(backproject(flat_depth(50.0), K), np.full((H, W), GO, np.uint8))
    assert set(grid["cells"]) == {CELL_EMPTY}


# ------------------------------------------------------------------------- objects


def _scene_with_slab(z=2.0, half_w=0.3, half_h=0.45):
    """A fronto-parallel slab: known centre, known width, known height."""
    depth = np.zeros((H, W), np.float32)
    pts = backproject(np.full((H, W), z, np.float32), K)
    inside = (np.abs(pts[..., 0]) < half_w) & (np.abs(pts[..., 1]) < half_h)
    depth[inside] = z
    ys, xs = np.nonzero(inside)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return depth, box


def test_an_object_reports_the_distance_it_was_built_at():
    depth, box = _scene_with_slab(z=2.0)
    obj = object_from_box(box, backproject(depth, K))
    assert obj is not None
    assert obj["z"] == pytest.approx(2.0, abs=0.01)
    assert obj["range_m"] == pytest.approx(2.0, abs=0.05)


def test_extent_comes_back_in_metres_slightly_under():
    """Extent is a 5-95 percentile span, so a 0.9 m slab reports about 0.81. That
    under-reporting is deliberate: one stray return at the far wall would otherwise set
    the size, and an object drawn slightly small is a smaller lie than one drawn large."""
    depth, box = _scene_with_slab(z=2.0, half_w=0.3, half_h=0.45)
    obj = object_from_box(box, backproject(depth, K))
    assert obj["w"] == pytest.approx(0.6 * 0.9, abs=0.06)
    assert obj["h"] == pytest.approx(0.9 * 0.9, abs=0.06)


def test_a_box_with_too_few_returns_is_dropped_not_guessed():
    """Glass, and anything past a D435's useful range, land here. A missing object is
    honest; one placed from six pixels is not."""
    depth = np.zeros((H, W), np.float32)
    depth[10, 10] = 2.0
    assert object_from_box((5, 5, 20, 20), backproject(depth, K)) is None


def _points_with_footprint(x, z, y=None):
    """Build a points array directly, so a footprint's shape is exactly what is meant."""
    n = len(x)
    pts = np.zeros((n, 1, 3), dtype=np.float32)
    pts[:, 0, 0] = x
    pts[:, 0, 1] = 0.0 if y is None else y
    pts[:, 0, 2] = z
    return pts


def test_a_round_footprint_reports_no_confidence_in_its_heading():
    """A chair seen from the front is roughly as wide as it is deep, and its point cloud
    does not say which way it faces. The renderer needs to know that rather than draw a
    confident arrow."""
    a = np.linspace(0, 2 * np.pi, 400)
    obj = object_from_box(
        (0, 0, 1, 400), _points_with_footprint(0.3 * np.cos(a), 2.0 + 0.3 * np.sin(a))
    )
    assert obj is not None
    assert obj["yaw_confidence"] < 0.3


def test_an_elongated_footprint_does_report_a_heading():
    """A sofa or a shelving run has a real principal axis; that is where yaw means
    something -- modulo the 180 degrees no point cloud can resolve."""
    t = np.linspace(-1.0, 1.0, 400)
    obj = object_from_box(
        (0, 0, 1, 400),
        _points_with_footprint(t, 2.0 + 0.02 * np.random.default_rng(0).normal(size=400)),
    )
    assert obj is not None
    assert obj["yaw_confidence"] > 0.8
    assert abs(np.sin(obj["yaw"])) < 0.2, "the axis lies along x"


# --------------------------------------------------------------------------- payload


def test_build_scene_is_serialisable_and_metric():
    import json

    depth, box = _scene_with_slab(z=2.0)
    depth[depth == 0] = 3.0  # floor behind the slab
    trav = np.full((H, W), GO, np.uint8)
    scene = build_scene(trav, depth, K, [{"box": box, "cls": "chair", "score": 0.8}])
    json.dumps(scene)  # must round-trip: it is served as JSON every frame
    assert scene["objects"][0]["cls"] == "chair"
    assert scene["objects"][0]["z"] == pytest.approx(2.0, abs=0.05)
    assert scene["grid"]["cell"] > 0


def test_background_inside_the_box_does_not_set_the_extent():
    """A detection box is a rectangle around something that is not one, so it always
    contains floor in front and wall behind. Measured naively the wall sets the size: a
    chair 0.6 m wide came back 9.65 m in the demo scene before this was handled."""
    depth = np.full((H, W), 6.0, np.float32)  # a wall right across the frame
    pts_at = backproject(np.full((H, W), 2.0, np.float32), K)
    obj = (np.abs(pts_at[..., 0]) < 0.3) & (np.abs(pts_at[..., 1]) < 0.4)
    depth[obj] = 2.0  # the object, well in front of it
    ys, xs = np.nonzero(obj)
    box = (xs.min() - 6, ys.min() - 6, xs.max() + 7, ys.max() + 7)  # a loose box

    got = object_from_box(box, backproject(depth, K))
    assert got is not None
    assert got["z"] == pytest.approx(2.0, abs=0.05), "the wall must not drag the distance"
    assert got["w"] < 0.8, f"the wall must not set the width, got {got['w']:.2f} m"


def test_a_deep_object_is_not_trimmed_to_its_front_face():
    """The rejection window scales with the object's own spread, so a table extending
    away from the camera keeps its depth rather than being flattened."""
    zs = np.linspace(2.0, 3.0, H)
    depth = np.repeat(zs[:, None], W, axis=1).astype(np.float32)
    obj = object_from_box((0, 0, W, H), backproject(depth, K))
    assert obj is not None
    assert obj["d"] > 0.6, f"a 1 m deep surface should not collapse, got {obj['d']:.2f} m"
