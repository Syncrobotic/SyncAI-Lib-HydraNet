"""Calibration from floor tiles: recovering a known lens, and refusing an unknown one.

Every camera fit this project has made before ran to its search boundary and was reported
as a result anyway. `fit_camera_from_people.py` records the shape: the residual fell
*monotonically* with the assumed vfov -- 4.9% at 55 degrees, 0.7% at 150 -- so "lowest
residual" chose the most extreme lens available rather than the real one, and the fitted
camera height slid from 1.87 m to 2.98 m underneath it.

So the tests that matter here are not "does it return a number". They are:

* it recovers a `k1` that was put in, to the resolution of the sweep;
* it says whether the answer is an interior maximum or the edge of the search;
* and the acceptance test refuses a pose whose horizon falls inside the frame, which is
  the one condition under which a ground projection manufactures floor that is not there.

Two bugs were found by these tests and both produced confident wrong answers rather than
errors. They are pinned below by name.

pytest tests/test_calibrate.py -v
"""

import math

import numpy as np
import pytest

from syncai_bev3d.calibrate import (
    Pose,
    concentration,
    fit_k1,
    floor_edge_points,
    height_from_known_length,
    horizon_row,
    hough,
    is_interior_maximum,
    pose_from_vanishing_points,
    vanishing_point,
)
from syncai_hydranet.geometry.ground import undistort_points

H, W = 720, 1280
CENTRE = np.array([W / 2.0, H / 2.0])
RADIUS = math.hypot(H, W) / 2.0


def _grid():
    """A straight tile grid: two orthogonal families, as a shop floor gives."""
    pts = []
    for x0 in np.linspace(120, W - 120, 9):
        pts += [(x0, y) for y in np.linspace(80, H - 80, 200)]
    for y0 in np.linspace(120, H - 120, 6):
        pts += [(x, y0) for x in np.linspace(60, W - 60, 260)]
    return np.array(pts, float)


def _distort(points, k):
    """Exact inverse of the division model, so the test puts in a known lens.

    Solves ``k * ru * rd^2 - rd + ru = 0`` rather than approximating, because an
    approximate forward model would make a recovery error indistinguishable from the
    test's own sloppiness.
    """
    d = points - CENTRE
    ru = np.linalg.norm(d, axis=1) / RADIUS
    if abs(k) < 1e-12:
        return points.copy()
    disc = 1 - 4 * k * ru**2
    rd = np.where(
        disc >= 0, (1 - np.sqrt(np.clip(disc, 0, None))) / (2 * k * np.maximum(ru, 1e-12)), ru
    )
    scale = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
    return CENTRE + d * scale[:, None]


# --- the property the module exists for ------------------------------------


@pytest.mark.parametrize("true_k1", [-0.30, -0.225, -0.12, 0.0, 0.15])
def test_a_known_lens_is_recovered(true_k1):
    """-0.225 is the value the cam01 fit found, so it is in the list on purpose."""
    recovered, sweep = fit_k1(_distort(_grid(), true_k1), (H, W))
    step = 0.9 / 60  # the sweep's own resolution
    assert abs(recovered - true_k1) <= step + 1e-9
    assert is_interior_maximum(sweep)


def test_the_sweep_is_returned_not_just_the_argmax():
    """The shape of the curve is the result. An argmax alone cannot say whether the
    objective was still improving where the search stopped."""
    _, sweep = fit_k1(_distort(_grid(), -0.2), (H, W))
    assert sweep.shape[0] == 2
    assert sweep.shape[1] == 61


def test_a_maximum_at_the_edge_of_the_search_is_not_an_interior_one():
    """The condition every earlier camera fit in this project failed silently."""
    rising = np.stack([np.linspace(-0.6, 0.3, 61), np.linspace(0.0, 1.0, 61)])
    assert not is_interior_maximum(rising)
    peaked = np.stack([np.linspace(-0.6, 0.3, 61), -((np.arange(61) - 30.0) ** 2)])
    assert is_interior_maximum(peaked)


# --- the two bugs, both of which returned confident wrong answers ----------


def test_the_objective_does_not_reward_shrinking_everything():
    """**Bug one.** A positive `k1` pulls every point toward the centre, which clusters the
    Hough votes and raises a naive concentration no matter how curved the lines are. The
    sweep ran to +0.300 on a grid with *zero* distortion. Fixing the scale before scoring
    removes a degeneracy the pinhole focal length absorbs anyway."""
    recovered, _ = fit_k1(_grid(), (H, W))
    assert abs(recovered) < 0.05, "an undistorted grid must not fit a large k1"


def test_concentration_is_measured_per_orientation():
    """**Bug two.** A floor grid has two line directions, so only 2 of 180 theta rows carry
    any straightness signal; the other 178 are smeared whatever the lens does. Scoring the
    accumulator as a block let those 178 dominate and the fit was wrong for every input."""
    straight = hough(_grid(), (H, W))
    curved = hough(_distort(_grid(), -0.3), (H, W))
    assert concentration(straight) > concentration(curved)


def test_a_scaled_copy_of_the_same_lines_fits_the_same_lens():
    """The invariance the first bug violated, asserted where it is actually implemented.

    `hough` is *not* scale-invariant and does not need to be -- its rho bins are fixed, so
    a shrunken point set lands in fewer of them and scores higher. That is exactly the
    degeneracy `fit_k1` removes by renormalising before it scores, and asserting the
    property on `concentration` instead would be testing a guarantee nothing makes.
    """
    step = 0.9 / 60
    pts = _grid()
    shrunk = (pts - CENTRE) * 0.6 + CENTRE
    full, small = fit_k1(pts, (H, W))[0], fit_k1(shrunk, (H, W))[0]
    # Both land within one sweep step of zero -- -0.015 and +0.015, the bins either side
    # of it. Asserting they are *equal* would be asserting sub-step precision the sweep
    # does not have, which is a tighter claim than the invariance being tested.
    assert abs(full) <= step + 1e-9
    assert abs(small) <= step + 1e-9


# --- undistortion itself ---------------------------------------------------


def test_zero_k1_is_the_identity():
    pts = _grid()
    assert np.allclose(undistort_points(pts, 0.0, CENTRE, RADIUS), pts)


def test_the_centre_never_moves():
    out = undistort_points(np.array([CENTRE]), -0.4, CENTRE, RADIUS)
    assert np.allclose(out[0], CENTRE)


def test_barrel_correction_pushes_points_outward():
    """Negative k1 is barrel; correcting it must move a corner further from the centre."""
    corner = np.array([[W - 10.0, H - 10.0]])
    moved = undistort_points(corner, -0.25, CENTRE, RADIUS)
    assert np.linalg.norm(moved[0] - CENTRE) > np.linalg.norm(corner[0] - CENTRE)


# --- pose, and the acceptance test -----------------------------------------


def test_a_bucket_of_lines_that_do_not_concur_reports_a_large_residual():
    """The guard for the step most likely to be subtly wrong. A mis-grouping produces a
    plausible focal length rather than an error, because `pose_from_vanishing_points` only
    refuses the gross case. The residual is what separates "these lines meet at a point"
    from "these lines were put in the same bucket"."""
    concurrent = []
    target = np.array([1500.0, 150.0])
    for angle in (0.25, 0.55, 0.85, 1.15):
        n = np.array([-math.sin(angle), math.cos(angle)])
        concurrent.append([math.atan2(n[1], n[0]) % math.pi, float(n @ target)])
    tight = vanishing_point(np.array(concurrent))

    scattered = np.array(concurrent, float)
    scattered[:, 1] += np.array([0.0, 140.0, -160.0, 190.0])  # same angles, shifted apart
    loose = vanishing_point(scattered)

    assert tight.residual_px < 1.0
    assert loose.residual_px > 20.0


def test_vanishing_point_of_converging_lines():
    """Three lines through (900, 200) must intersect there."""
    target = np.array([900.0, 200.0])
    lines = []
    for angle in (0.3, 0.7, 1.1):
        d = np.array([math.cos(angle), math.sin(angle)])
        n = np.array([-d[1], d[0]])
        lines.append([math.atan2(n[1], n[0]) % math.pi, float(n @ target)])
    vp = vanishing_point(np.array(lines))
    assert np.allclose(vp.point, target, atol=1e-6)
    assert vp.residual_px < 1e-6, "lines that truly concur have no residual"
    assert np.allclose(np.asarray(vp), target, atol=1e-6), "still usable as a bare point"


def test_lines_that_stay_parallel_have_no_finite_vanishing_point():
    parallel = np.array([[0.0, 100.0], [0.0, 300.0], [0.0, 500.0]])
    with pytest.raises(ValueError, match="no finite vanishing point"):
        vanishing_point(parallel)


def test_non_orthogonal_families_are_refused_rather_than_fitted():
    """The formula only holds for perpendicular floor directions. Fitting it anyway would
    return a focal length from a square root of the wrong sign's magnitude."""
    with pytest.raises(ValueError, match="orthogonal"):
        pose_from_vanishing_points([W + 500.0, H / 2], [W + 900.0, H / 2 + 10], (H, W))


def test_a_horizon_inside_the_frame_is_rejected():
    """The binary acceptance test. At 43 degrees the cam01 fit put the horizon at row 36 of
    1080 and manufactured a floor spanning 0.07 to 3543 metres."""
    bad = Pose(k1=-0.2, vfov_deg=95, pitch_deg=43.0, height_m=1.96,
               horizon_row=36.0, image_h=1080, image_w=1920)  # fmt: skip
    good = Pose(k1=-0.225, vfov_deg=70.4, pitch_deg=50.2, height_m=2.38,
                horizon_row=-379.0, image_h=1080, image_w=1920)  # fmt: skip
    assert bad.rejection is not None and "grazing" in bad.rejection
    assert good.rejection is None
    assert "REJECT" in bad.summary()
    assert "pass" in good.summary()


def test_a_horizon_below_the_frame_is_a_camera_looking_up_and_is_rejected():
    """**The bug this class shipped with.** `horizon_is_inside` was the acceptance test, so
    a horizon at row 1425 of 1080 read as "outside the frame" and passed -- but outside
    *below* means tan(pitch) < 0 means the camera is pitched up. That was a ceiling camera
    pointed at a floor, and `[pass]` is what a fleet report would have printed for it.

    `GroundPlane` fixes the sign: pitch positive is looking down. So the only acceptable
    horizon is above the frame, not merely outside it."""
    up = Pose(k1=-0.21, vfov_deg=29.6, pitch_deg=-23.4, height_m=None,
              horizon_row=1425.0, image_h=1080, image_w=1920)  # fmt: skip
    assert not up.horizon_is_inside, "it really is outside the frame -- that was the trap"
    assert up.rejection is not None
    assert "pitched up" in up.rejection
    assert "REJECT" in up.summary()


def test_the_summary_says_which_of_the_two_failures_it_is():
    """They are different problems -- too much floor against no floor at all -- and a fleet
    report that prints one word for both tells the reader nothing about what to fix."""
    inside = Pose(-0.2, 95, 43.0, None, 36.0, 1080, 1920)
    below = Pose(-0.2, 30, -23.4, None, 1425.0, 1080, 1920)
    assert "grazing" in inside.summary()
    assert "pitched up" in below.summary()
    assert inside.rejection != below.rejection


def test_horizon_moves_out_of_frame_as_the_pitch_steepens():
    """The measured pair: 43 deg puts it inside a 1080-row frame, 50.2 deg does not."""
    focal = 766.0
    assert 0 <= horizon_row(43.0, 540.0, 1080) < 1080
    assert horizon_row(50.2, focal, 1080) < 0


def test_an_unscaled_pose_says_so_rather_than_guessing():
    p = Pose(-0.2, 70.0, 50.0, None, -300.0, 1080, 1920)
    assert "unscaled" in p.summary()


# --- scale -----------------------------------------------------------------


def test_a_known_length_fixes_the_height():
    h = height_from_known_length(
        pixel_length=200.0, metres=0.6, focal_px=766.0, pitch_deg=50.2, row=900.0, image_h=1080
    )
    assert 0.5 < h < 20.0  # a real number in metres, not a NaN or a negative


def test_a_row_above_the_horizon_is_not_on_the_floor():
    with pytest.raises(ValueError, match="horizon"):
        height_from_known_length(
            pixel_length=200.0,
            metres=0.6,
            focal_px=766.0,
            pitch_deg=5.0,
            row=10.0,
            image_h=1080,
        )


@pytest.mark.parametrize(("px", "m"), [(0.0, 0.6), (200.0, 0.0), (-5.0, 0.6)])
def test_an_impossible_known_length_is_refused(px, m):
    with pytest.raises(ValueError, match="positive"):
        height_from_known_length(px, m, 766.0, 50.2, 900.0, 1080)


# --- the mask, which is where a rectangular ROI went wrong -----------------


def test_edges_are_taken_only_inside_the_floor_mask():
    """A box around the lower frame catches counter edges and skirting, which are longer
    and straighter than a tile joint, and the Hough latches onto those instead."""
    grey = np.zeros((60, 80))
    grey[:, ::8] = 1.0
    grey[::10, :] = 1.0
    mask = np.zeros((60, 80), bool)
    mask[20:, :] = True
    pts = floor_edge_points(grey, mask)
    assert len(pts) > 0
    assert (pts[:, 1] >= 20).all()


def test_an_empty_floor_mask_returns_nothing_rather_than_everything():
    grey = np.random.default_rng(0).random((40, 40))
    assert len(floor_edge_points(grey, np.zeros((40, 40), bool))) == 0


def test_a_mask_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="differ"):
        floor_edge_points(np.zeros((40, 40)), np.zeros((20, 20), bool))


# --- oriented voting, which removes a starburst that looks like a result ---


def _oriented_grid():
    """The grid with each point's true gradient normal: vertical lines have normal 0,
    horizontal lines have normal pi/2."""
    vert = [
        (x0, y) for x0 in np.linspace(120, W - 120, 9) for y in np.linspace(80, H - 80, 200)
    ]
    horiz = [
        (x, y0) for y0 in np.linspace(120, H - 120, 6) for x in np.linspace(60, W - 60, 260)
    ]
    pts = np.array(vert + horiz, float)
    ang = np.concatenate([np.zeros(len(vert)), np.full(len(horiz), math.pi / 2)])
    return pts, ang


def test_a_dense_blob_of_noise_builds_a_ridge_without_orientations():
    """The failure `orientations` removes. Every point voting into all 180 thetas means a
    dense blob agrees about its own centroid whatever the lines are doing, and peaks
    extracted from that ridge render as thirty lines through one point -- a starburst a
    reader takes for a vanishing point."""
    rng = np.random.default_rng(0)
    blob = rng.normal([W / 2, H / 2], [40, 40], size=(3000, 2))
    unoriented = hough(blob, (H, W))
    # A structureless blob should not out-score a real grid. Without orientations it does.
    assert concentration(unoriented) > concentration(hough(_grid(), (H, W)))


def test_orientations_stop_the_blob_from_outscoring_a_real_grid():
    rng = np.random.default_rng(0)
    blob = rng.normal([W / 2, H / 2], [40, 40], size=(3000, 2))
    blob_ang = rng.uniform(0, math.pi, len(blob))  # structureless gradients
    pts, ang = _oriented_grid()
    assert concentration(hough(pts, (H, W), orientations=ang)) > concentration(
        hough(blob, (H, W), orientations=blob_ang)
    )


def test_a_point_does_not_vote_for_a_theta_its_gradient_disagrees_with():
    pts, ang = _oriented_grid()
    acc = hough(pts, (H, W), orientations=ang, spread_deg=12.0)
    n_theta = acc.shape[0]
    # 45 degrees is 90 bins of 180 across pi; no grid point's normal is within 12 of it.
    diagonal = acc[n_theta // 4]
    assert diagonal.sum() == 0
    assert acc[0].sum() > 0 and acc[n_theta // 2].sum() > 0


def test_omitting_orientations_is_the_old_behaviour_exactly():
    """Existing callers and every test above must be untouched by the new argument."""
    pts = _grid()
    assert np.array_equal(hough(pts, (H, W)), hough(pts, (H, W), orientations=None))


def test_mismatched_orientations_are_refused():
    with pytest.raises(ValueError, match="orientations has"):
        hough(_grid(), (H, W), orientations=np.zeros(3))


def test_edge_points_can_return_their_gradient_direction():
    grey = np.zeros((60, 80))
    grey[:, ::8] = 1.0  # vertical lines -> gradient points across x, normal near 0 or pi
    mask = np.ones((60, 80), bool)
    pts, ang = floor_edge_points(grey, mask, return_orientation=True)
    assert len(pts) == len(ang)
    assert ((ang < 0.3) | (ang > math.pi - 0.3)).mean() > 0.9
    assert np.array_equal(pts, floor_edge_points(grey, mask)), "default return unchanged"
