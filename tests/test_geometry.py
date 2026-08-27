"""Geometry has to be checked against something other than itself.

Every test here either round-trips a transform, or builds a scene whose answer is known
by construction. A projection that is confidently wrong looks exactly like one that is
right until someone measures a distance in the real world.
"""

import math

import numpy as np
import pytest

from syncai_bev3d import (
    BevGrid,
    free_space_map,
    place_boxes,
    project_mask,
    ray_reach,
    scene,
)
from syncai_hydranet.geometry import (
    Camera,
    GroundPlane,
    fit_ground_plane,
    ground_to_pixel,
    pixel_to_ground,
    unproject,
)
from syncai_hydranet.labels import IGNORE

CAM = Camera.from_vfov(512, 288, 65.0)
PLANE = GroundPlane(height=1.2, pitch=math.radians(18))


# ----------------------------------------------------------------- round trips


@pytest.mark.parametrize(
    "plane",
    [
        GroundPlane(1.2, math.radians(18)),
        GroundPlane(0.4, math.radians(35)),
        GroundPlane(1.5, math.radians(22), math.radians(7)),  # a quadruped mid-stride
        # The shipped CCTV fleet, from `runs/commission01/*.camera.json`. The three above
        # are the robot line's geometry and none of them reaches this range: the fleet
        # sits at 2.17-2.91 m and 38.8-52.3 deg, and Kaohsiung-cam04 banks -12.9 deg,
        # which is the largest roll in the fleet and the opposite sign to the case above.
        # PLAN step 5 is about to charge metre errors to the calibration, so the
        # round-trip has to be known-good at the poses that are actually deployed.
        GroundPlane(2.17, math.radians(52.3), math.radians(-12.9)),  # Kaohsiung-cam04
        GroundPlane(2.91, math.radians(42.7), math.radians(0.5)),  # Tao-Hsin-cam04
        GroundPlane(2.77, math.radians(38.8), math.radians(-8.3)),  # Taichung-cam07
    ],
)
def test_pixel_and_ground_are_inverses(plane):
    xs, zs = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(1.0, 8.0, 8))
    u, v, depth = ground_to_pixel(xs, zs, CAM, plane)
    assert (depth > 0).all(), "floor points must be in front of the camera"
    x2, z2 = pixel_to_ground(u, v, CAM, plane)
    assert np.allclose(x2, xs, atol=1e-6)
    assert np.allclose(z2, zs, atol=1e-6)


def test_optical_axis_hits_the_floor_where_trigonometry_says():
    """The centre pixel looks along the axis; its range is height / tan(pitch)."""
    x, z = pixel_to_ground(CAM.cx, CAM.cy, CAM, PLANE)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(PLANE.height / math.tan(PLANE.pitch), rel=1e-6)


def test_rays_at_or_above_the_horizon_are_nan_not_huge():
    """A number there would be read as a measurement. The horizon sits fy*tan(pitch) above centre.

    A pixel a hair below the horizon is also rejected, and should be: it lands kilometres
    away, which is a rounding artefact rather than a floor position. The offsets here are
    in whole pixels for that reason.
    """
    horizon_v = CAM.cy - CAM.fy * math.tan(PLANE.pitch)
    _, z = pixel_to_ground(
        np.full(3, CAM.cx), np.array([horizon_v - 5, horizon_v, horizon_v + 5]), CAM, PLANE
    )
    assert np.isnan(z[0]) and np.isnan(z[1])
    assert np.isfinite(z[2]) and z[2] > 20, "five pixels below the horizon is still far away"


def test_further_down_the_image_is_nearer_the_camera():
    v = np.array([CAM.cy + 10, CAM.cy + 100, CAM.cy + 200], dtype=float)
    _, z = pixel_to_ground(np.full(3, CAM.cx), v, CAM, PLANE)
    assert z[0] > z[1] > z[2] > 0


# ----------------------------------------------------------------- plane fitting


def synthetic_depth(cam: Camera, plane: GroundPlane, shape=(512, 288)) -> np.ndarray:
    """Depth of a perfectly flat floor filling the frame, as the camera would see it."""
    h, w = shape
    vv, uu = np.mgrid[0:h, 0:w]
    rays = np.stack(
        [(uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fy, np.ones_like(uu, float)], -1
    )
    level = rays @ plane.rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        t = plane.height / level[..., 1]
    z = np.where(level[..., 1] > 1e-9, t, 0.0)
    return np.nan_to_num(z, nan=0.0, posinf=0.0)


@pytest.mark.parametrize(
    "truth",
    [
        GroundPlane(1.20, math.radians(18)),
        GroundPlane(0.55, math.radians(30)),
        GroundPlane(1.45, math.radians(24), math.radians(6)),
    ],
)
def test_fit_recovers_the_plane_it_was_given(truth):
    depth = synthetic_depth(CAM, truth)
    got, residual = fit_ground_plane(depth, CAM, seed=1)
    assert got is not None
    assert got.height == pytest.approx(truth.height, abs=0.01)
    assert math.degrees(got.pitch) == pytest.approx(math.degrees(truth.pitch), abs=0.5)
    assert math.degrees(got.roll) == pytest.approx(math.degrees(truth.roll), abs=0.5)
    finite = residual[np.isfinite(residual)]
    assert np.abs(finite).max() < 0.02, "a flat floor should have no residual"


def test_a_step_shows_up_in_the_residual():
    """The residual is an output, not a diagnostic: geometry the plane does not explain."""
    depth = synthetic_depth(CAM, PLANE)
    depth[400:, :] *= 0.85  # a raised slab across the near field
    _, residual = fit_ground_plane(depth, CAM, seed=1)
    near = residual[420:, :]
    assert np.nanmax(np.abs(near)) > 0.05


def test_no_depth_returns_no_plane_rather_than_a_confident_one():
    plane, residual = fit_ground_plane(np.zeros((512, 288)), CAM, seed=1)
    assert plane is None
    assert np.isnan(residual).all()


def test_unproject_puts_the_floor_at_the_camera_height():
    depth = synthetic_depth(CAM, PLANE)
    pts = unproject(depth, CAM)
    level = pts @ PLANE.rotation  # back to the frame where y is straight down
    y = level[..., 1][np.isfinite(level[..., 1])]
    assert np.allclose(y[y != 0], PLANE.height, atol=1e-6)


# ----------------------------------------------------------------- projection


def test_projected_mask_keeps_left_on_the_left():
    """A mask painted on one side must not come back mirrored."""
    mask = np.full((512, 288), 2, np.uint8)
    mask[:, :144] = 0  # blocked down the left half of the image
    bev = project_mask(mask, CAM, PLANE, BevGrid(cell=0.05))
    seen = bev[bev != IGNORE]
    assert len(seen) > 0
    left, right = bev[:, : bev.shape[1] // 2], bev[:, bev.shape[1] // 2 :]
    assert (left == 0).sum() > (right == 0).sum()


def test_near_field_is_at_the_bottom_of_the_map():
    mask = np.full((512, 288), 2, np.uint8)
    bev = project_mask(mask, CAM, PLANE, BevGrid(z_min=0.5, z_max=9.0, cell=0.05))
    rows = np.where((bev != IGNORE).any(axis=1))[0]
    # the camera sees a wedge: it is narrow near the robot, so the bottom row is thin
    assert (bev[rows[-1]] != IGNORE).sum() < (bev[rows[0]] != IGNORE).sum()


def test_cells_no_pixel_sees_are_ignore_not_a_class():
    mask = np.full((512, 288), 2, np.uint8)
    bev = project_mask(mask, CAM, PLANE, BevGrid(x_min=-8, x_max=8, cell=0.05))
    assert (bev == IGNORE).any(), "a wide grid must have corners outside the field of view"


def test_a_box_standing_at_a_known_distance_comes_back_at_that_distance():
    x_true, z_true = 0.8, 3.5
    u, v, _ = ground_to_pixel(np.array([x_true]), np.array([z_true]), CAM, PLANE)
    box = np.array([[u[0] - 20, v[0] - 90, u[0] + 20, v[0]]])  # bottom edge on the floor
    ((x, z),) = place_boxes(box, CAM, PLANE)
    assert x == pytest.approx(x_true, abs=1e-6)
    assert z == pytest.approx(z_true, abs=1e-6)


def test_a_box_floating_above_the_horizon_is_not_given_a_position():
    horizon_v = CAM.cy - CAM.fy * math.tan(PLANE.pitch)
    ((x, z),) = place_boxes(np.array([[100, 10, 140, horizon_v - 5]]), CAM, PLANE)
    assert np.isnan(x) and np.isnan(z)


def test_scene_reports_metres_and_omits_unplaceable_objects():
    mask = np.full((512, 288), 2, np.uint8)
    u, v, _ = ground_to_pixel(np.array([0.5]), np.array([2.0]), CAM, PLANE)
    boxes = np.array(
        [
            [u[0] - 15, v[0] - 60, u[0] + 15, v[0]],  # on the floor
            [10, 5, 40, 20],
        ]
    )  # up near the ceiling
    doc, bev = scene(
        mask,
        CAM,
        PLANE,
        boxes=boxes,
        labels=[56, 62],
        scores=[0.7, 0.5],
        names={56: "chair", 62: "tv"},
    )
    assert len(doc["objects"]) == 1, "the ceiling box has no floor position"
    obj = doc["objects"][0]
    assert obj["name"] == "chair"
    assert obj["z_m"] == pytest.approx(2.0, abs=1e-3)
    assert obj["range_m"] == pytest.approx(math.hypot(0.5, 2.0), abs=1e-3)
    assert doc["plane"]["height_m"] == pytest.approx(PLANE.height)
    assert bev.shape == tuple(doc["grid"]["shape"])


# ----------------------------------------------------------------- free space

# A grid whose rows are easy to name: 40 rows of 0.1 m from 0.5 m out to 4.5 m. Row 0 is
# the far edge, because that is the order `project_mask` returns and every consumer of a
# BEV array in this package has to agree with it.
FS_GRID = BevGrid(x_min=-2.0, x_max=2.0, z_min=0.5, z_max=4.5, cell=0.1)
FS_ROWS, FS_COLS = FS_GRID.shape
FS_LAST_WALKABLE_ROW = 20  # rows 20..39 are floor, i.e. everything nearer than 2.5 m


def floor_out_to_2m5() -> np.ndarray:
    """A BEV with clean floor across the near half and something blocking it at 2.5 m."""
    bev = np.zeros((FS_ROWS, FS_COLS), np.uint8)  # blocked
    bev[FS_LAST_WALKABLE_ROW:] = 2  # go
    return bev


def test_free_space_reads_rows_the_way_project_mask_writes_them():
    """The boundary belongs just beyond the last floor cell, on the far side of it.

    This is the whole row-order contract in one assertion. A consumer that takes row 0 for
    the near edge computes every range mirrored: the map it draws is still smooth, still
    plausible, and puts the edge of the free space at the wrong distance -- which is the
    one number a navigation stack reads off it.
    """
    out = free_space_map(floor_out_to_2m5(), FS_GRID)
    edge_rows = np.unique(np.where(out == 0)[0])
    assert edge_rows.size, "the floor ended somewhere and that has to be marked"
    assert edge_rows.max() < FS_LAST_WALKABLE_ROW, "the boundary is on the far side"
    assert edge_rows.min() >= FS_LAST_WALKABLE_ROW - 6, (
        f"and adjacent to the floor, not out at the far edge: rows {edge_rows}"
    )


def test_nothing_is_asserted_behind_the_boundary():
    """Beyond the boundary a single camera knows nothing, and must not say `blocked`.

    The mask calls the far field blocked because a wall pixel's ray meets the plane well
    past the wall. Copying that through would paint an obstacle over ground the camera
    never saw, and it would be trusted in the direction that costs a robot a route.
    """
    out = free_space_map(floor_out_to_2m5(), FS_GRID)
    beyond = out[: FS_LAST_WALKABLE_ROW - 6]
    assert (beyond == IGNORE).all(), "the far field is unknown, not blocked"


def test_a_ray_that_never_saw_floor_gets_no_boundary():
    bev = floor_out_to_2m5()
    bev[:, :10] = 0  # a whole column of bearings with no floor at any range
    out = free_space_map(bev, FS_GRID)
    assert (out[:, :6] == IGNORE).all(), "nothing was established about those bearings"


def test_the_perspective_renderer_shares_the_ray_reduction():
    """`bev3d` draws the boundary the flat map filters on. Two implementations of "how
    far did the floor reach" would drift, and the picture would stop agreeing with the
    map it is drawn from."""
    from syncai_bev3d import bev3d

    bev = floor_out_to_2m5()
    angles, reach, _, _ = ray_reach(bev, FS_GRID, n_rays=64)
    a3, r3 = bev3d.boundary_rays(bev, FS_GRID, n_rays=64)
    assert np.allclose(angles, a3)
    assert np.allclose(reach, r3)


def test_every_walkable_cell_survives_the_filter():
    """The floor is kept whole; only the far field is dropped.

    Worth pinning because it is easy to write this filter as a range test against the
    reach -- and a range test can never remove a walkable cell, since the reach *is* that
    cell's own bearing's maximum. Anything that looks like it is filtering the floor here
    is either doing nothing or is a bug; this asserts the first.
    """
    bev = floor_out_to_2m5()
    bev[30:, 5:9] = 1  # a caution patch, which counts as floor for reachability
    out = free_space_map(bev, FS_GRID)
    walkable = (bev == 2) | (bev == 1)
    assert np.array_equal(out[walkable], bev[walkable])
    assert (out[walkable] != IGNORE).all()


def test_the_reach_is_the_distance_to_the_floor_edge_not_the_grid_edge():
    """Straight ahead the floor stops at 2.5 m, so that is what the reduction reports."""
    _, reach, _, _ = ray_reach(floor_out_to_2m5(), FS_GRID, n_rays=64)
    ahead = reach[len(reach) // 2]
    assert ahead == pytest.approx(2.45, abs=0.1), "the last floor cell is centred at 2.45 m"


def test_intrinsics_rescale_with_the_image():
    half = CAM.scaled_to(256, 144, from_shape=(512, 288))
    assert half.fx == pytest.approx(CAM.fx / 2)
    assert half.cx == pytest.approx(CAM.cx / 2)
    # and the same floor point still lands on the same spot, in the smaller frame
    u0, v0, _ = ground_to_pixel(np.array([1.0]), np.array([4.0]), CAM, PLANE)
    u1, v1, _ = ground_to_pixel(np.array([1.0]), np.array([4.0]), half, PLANE)
    assert u1[0] == pytest.approx(u0[0] / 2, abs=1e-6)
    assert v1[0] == pytest.approx(v0[0] / 2, abs=1e-6)
