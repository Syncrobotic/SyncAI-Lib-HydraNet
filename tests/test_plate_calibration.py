"""The arithmetic every metre in this product rests on, which nothing was executing.

`pyproject.toml`'s coverage block records the measurement that prompted this file, taken
2026-08-28: of the two modules that hold the arithmetic every event rests on,
`plate_calibration.py` (pixels to metres) sits at **10%** and `floorplan.py` at **0%** --
the two least-covered modules in `syncai_bev3d`. Neither is dead code. They were outside
the instrument, because `[tool.coverage.run].source` named `syncai_hydranet` alone while
the wheel ships both packages.

Running `pytest --cov` against this module before this file existed reported *no data
collected*: not low coverage, **no import at all**. Every figure the README leads with, the
7.3 cm WILDTRACK reproduction, and all 71 service zones are downstream of code no test
loaded.

WHAT THESE TESTS ARE, AND WHAT THEY DELIBERATELY ARE NOT
--------------------------------------------------------
`run_depth` is not here: it downloads Depth-Anything V2 and wants a GPU, and a test that
skips on the box that ships is the pattern `test_figures_are_audited.py` opens by warning
about. Everything else in the pipeline is arithmetic on arrays, and arithmetic can be
checked against a case whose answer is known independently.

The strongest of these is `test_the_people_only_fit_recovers_the_pose_it_was_built_from`,
which is a *closed loop*: place people of exactly `ADULT_M` on a floor under a camera of
known height and pitch, project them to boxes through `GroundPlane.rotation`, and hand
those boxes back. The fit never sees the pose. It recovers it to **0.000 m and 0.0 deg** on
all three configurations here, so the chain from a person's box to a camera's height is
exact -- which is worth knowing precisely because §7.19 says the fleet's metres rest on it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from syncai_bev3d.plate_calibration import (
    ADULT_M,
    choose_floor,
    column_health,
    fit_pose_from_people,
    floor_candidates,
    floor_scale,
    load_person_boxes,
    person_checks,
    pick_daytime_slot,
    undistort_image,
)
from syncai_hydranet.geometry.ground import Camera, GroundPlane, undistort_points

H_PX, W_PX = 1080, 1920


def _synth_people(vfov: float, cam_h: float, pitch_deg: float, n: int = 24, seed: int = 0):
    """Person boxes for `n` adults of exactly `ADULT_M`, seen by a known camera.

    The forward map is `GroundPlane.rotation` applied by hand rather than
    `ground_to_pixel`, because that helper hard-codes the floor's `y` and a standing
    person needs a second height -- the head, `ADULT_M` above the same floor point.
    Boxes leaving the frame are dropped: a camera that cannot see a whole person cannot
    measure one, which is the same reason `box_extents` returns NaN for a clipped box.
    """
    cam = Camera.from_vfov(H_PX, W_PX, vfov)
    plane = GroundPlane(height=cam_h, pitch=math.radians(pitch_deg))

    def project(x, y_down, z):
        pts = np.stack([x, y_down, z], axis=-1) @ plane.rotation.T
        return (
            cam.fx * pts[..., 0] / pts[..., 2] + cam.cx,
            cam.fy * pts[..., 1] / pts[..., 2] + cam.cy,
        )

    rng = np.random.default_rng(seed)
    xs = rng.uniform(-3.0, 3.0, n)
    zs = rng.uniform(2.5, 9.0, n)
    u_feet, v_feet = project(xs, np.full_like(xs, cam_h), zs)
    _, v_head = project(xs, np.full_like(xs, cam_h - ADULT_M), zs)
    half = 0.25 * cam.fx / zs  # a ~0.5 m shoulder width at that range
    boxes = np.stack([u_feet - half, v_head, u_feet + half, v_feet], axis=-1)
    whole = (boxes[:, 0] > 0) & (boxes[:, 2] < W_PX) & (boxes[:, 1] > 0) & (boxes[:, 3] < H_PX)
    return boxes[whole]


# ---------------------------------------------------------------------------
# the people-only pose fit -- what §7.19 says the fleet's metres rest on


@pytest.mark.parametrize(
    ("vfov", "cam_h", "pitch_deg"),
    [(60.0, 2.8, 25.0), (70.4, 3.0, 20.0), (55.0, 2.5, 30.0)],
)
def test_the_people_only_fit_recovers_the_pose_it_was_built_from(vfov, cam_h, pitch_deg):
    """A closed loop: synthesise from a known pose, fit, and compare.

    Exact on all three, which is the useful form of this result. `fit_pose_from_people`
    is the only instrument behind every shipped `scale_source` -- every one of them reads
    `person_height_median_vs_1.7m_prior_nNN` -- so a systematic error here would be a
    systematic error in every metre the product reports, and would look like nothing.
    """
    boxes = _synth_people(vfov, cam_h, pitch_deg)
    assert len(boxes) >= 8, "the synthetic camera must see enough whole people to fit"
    spread, pitch, height, n = fit_pose_from_people(
        boxes, (H_PX, W_PX), vfov, heights=(1.0, 8.0), pitches=np.arange(5.0, 80.1, 0.5)
    )
    assert n == len(boxes)
    assert height == pytest.approx(cam_h, abs=0.01), (
        f"recovered {height} for a {cam_h} m camera"
    )
    assert pitch == pytest.approx(pitch_deg, abs=0.5), "pitch is searched on a 0.5 deg grid"
    assert spread < 1e-6, "an exact synthetic case should leave essentially no spread"


def test_a_wrong_vfov_is_a_confident_wrong_answer_and_the_residual_does_not_say_so():
    """The module's stated limitation, measured, because 22 of 23 cameras guess this input.

    The docstring says the vfov must be an input because a free one absorbs lens
    distortion. What it does not say is what a *wrong* input costs, and that matters here
    more than it would elsewhere: §7.19 records that vfov is a guess on all but one camera
    in the fleet.

    Measured on the 60 deg / 2.80 m / 25 deg case: feeding 70 deg -- a 10 deg error, which
    is an ordinary amount to be wrong by when guessing -- returns **2.639 m for a 2.80 m
    camera, a 5.8% error**, and every metre downstream carries it. The half of this worth
    a test is the second half: the fit's own residual moves from 0.0 to 9e-04, which is
    indistinguishable from a good fit. **Nothing in the output announces the error**, so a
    caller cannot gate on it, and a camera whose vfov was guessed wrong looks exactly like
    one whose vfov was measured.
    """
    truth_vfov, truth_h = 60.0, 2.8
    boxes = _synth_people(truth_vfov, truth_h, 25.0)
    kw = {"heights": (1.0, 8.0), "pitches": np.arange(5.0, 80.1, 0.5)}

    good_spread, _, good_h, _ = fit_pose_from_people(boxes, (H_PX, W_PX), truth_vfov, **kw)
    bad_spread, _, bad_h, _ = fit_pose_from_people(boxes, (H_PX, W_PX), 70.0, **kw)

    assert good_h == pytest.approx(truth_h, abs=0.01)
    assert good_spread < 1e-6, "the truthful vfov leaves essentially no spread"
    assert abs(bad_h - truth_h) / truth_h > 0.04, (
        "a 10 deg vfov error should move the height by several percent; if this fails the "
        "method has become robust to vfov and the fleet's guessed vfovs matter less"
    )
    assert bad_spread < 1e-3, (
        "the residual is expected NOT to expose the bad vfov -- that is the finding. If "
        "this fails, the residual has become a usable gate on vfov error and "
        "`person_checks` should start reporting it as one."
    )


def test_the_fit_refuses_rather_than_returning_a_pose_it_cannot_support():
    """Fewer than 8 usable heights at every pitch, so no pitch qualifies.

    `SystemExit` rather than `ValueError` is deliberate and load-bearing -- the module
    docstring records that `person_checks`'s `failed` branch catches exactly this -- so
    the type is asserted, not just the refusal.
    """
    boxes = _synth_people(60.0, 2.8, 25.0)[:5]
    with pytest.raises(SystemExit):
        fit_pose_from_people(
            boxes, (H_PX, W_PX), 60.0, heights=(1.0, 8.0), pitches=np.arange(5.0, 80.1, 0.5)
        )


def test_a_wrong_height_band_returns_a_different_pose_and_the_residual_does_say_so():
    """The band does not refuse -- it re-fits -- and this time the residual is a usable gate.

    Written expecting a refusal, which is wrong: constraining `heights` does not reject a
    camera outside the band, it selects a different pitch whose implied height lands
    inside it. Measured on the same 2.80 m / 25 deg camera:

        heights (1.0, 8.0) -> 2.800 m, pitch 25.0, spread 0.0
        heights (1.0, 2.0) -> 1.983 m, pitch 17.5, spread 8.4e-02
        heights (4.0, 8.0) -> 4.015 m, pitch 40.5, spread 9.9e-02

    The answer lands against whichever end of the band was wrong, which is what a caller
    passing a band from a rough site note would see.

    **Read this next to the vfov test above; the pair is the point.** Both are wrong
    inputs and they behave oppositely: a bad vfov moves the residual to 9e-04 and hides,
    a bad height band moves it to 8e-02 and shows. So `spread` is a usable gate on one
    class of input error and not on the other, and `person_checks` reports it without
    saying which -- a reader who learns to trust it from this case will trust it in the
    case where it does not work.
    """
    boxes = _synth_people(60.0, 2.8, 25.0)
    kw = {"pitches": np.arange(5.0, 80.1, 0.5)}
    good_spread, _, good_h, _ = fit_pose_from_people(
        boxes, (H_PX, W_PX), 60.0, heights=(1.0, 8.0), **kw
    )
    low_spread, _, low_h, _ = fit_pose_from_people(
        boxes, (H_PX, W_PX), 60.0, heights=(1.0, 2.0), **kw
    )
    high_spread, _, high_h, _ = fit_pose_from_people(
        boxes, (H_PX, W_PX), 60.0, heights=(4.0, 8.0), **kw
    )

    assert good_h == pytest.approx(2.8, abs=0.01) and good_spread < 1e-6
    assert low_h < 2.0 and high_h > 4.0, "each answer is pinned against the wrong bound"
    assert low_spread > 100 * good_spread + 1e-3
    assert high_spread > 100 * good_spread + 1e-3


# ---------------------------------------------------------------------------
# the lens


def test_undistorting_with_no_distortion_returns_the_image_itself():
    img = np.arange(12 * 16, dtype=np.uint8).reshape(12, 16)
    assert undistort_image(img, 0.0) is img


def test_undistort_image_preserves_shape_and_dtype_and_holds_the_centre():
    """The centre is at radius zero, so no division model can move it."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
    out = undistort_image(img, -0.12)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert np.array_equal(out[32, 48], img[32, 48])


def test_the_image_warp_is_the_inverse_of_the_point_map():
    """The one contract the two halves of the lens have with each other.

    `undistort_points` maps distorted -> undistorted; warping an image needs the other
    direction, and `undistort_image` inverts the model in closed form. If the two ever
    disagree -- a sign, a normalisation, a radius convention -- masks and metres drift
    apart silently, which is the failure `undistort_points`' own docstring warns about.
    So: take the radius `undistort_image` samples *from*, push it through
    `undistort_points`, and require the radius it lands on to be the one asked for.
    """
    h, w, k1 = 480, 640, -0.15
    radius = math.hypot(h, w) / 2.0
    cx, cy = w / 2.0, h / 2.0
    for r_u in (0.1, 0.3, 0.5, 0.7):
        # the closed-form inverse undistort_image applies, at this undistorted radius
        r_d = (1.0 - math.sqrt(1.0 - 4.0 * k1 * r_u**2)) / (2.0 * k1 * r_u)
        sampled = np.array([[cx + r_d * radius, cy]])
        back = undistort_points(sampled, k1, (cx, cy), radius)
        assert math.hypot(back[0, 0] - cx, back[0, 1] - cy) / radius == pytest.approx(
            r_u, rel=1e-9
        )


# ---------------------------------------------------------------------------
# floor selection: "lowest plausible", not "biggest"


def _cand(height, pitch_deg, count, roll_deg=0.0, lower=0.55, seed=0):
    return (
        GroundPlane(height, math.radians(pitch_deg), math.radians(roll_deg)),
        None,
        count,
        lower,
        seed,
    )


def test_the_lowest_plausible_plane_wins_even_with_fewer_inliers():
    """The docstring's claim, as a case that would pass under 'biggest plane' and must not.

    A display table is a large, flat, horizontal surface, and on a camera pointed at one
    it can carry more inlier pixels than the floor. Selecting on inlier count returns the
    table, every metre is then measured from the table top, and nothing downstream can
    tell -- the plane is perfectly good, it is just not the floor. The floor is by
    definition beneath every other horizontal surface, which is why height decides.
    """
    table = _cand(height=1.85, pitch_deg=30.0, count=9000)
    floor = _cand(height=2.80, pitch_deg=30.0, count=4000)
    plane, _residual, rows = choose_floor([table, floor])
    assert plane is not None
    assert plane.height == pytest.approx(2.80), (
        "the table won; selection is on count, not height"
    )
    assert all(r["plausible_floor"] for r in rows)


def test_a_starved_lower_plane_does_not_win():
    """Height decides only among candidates that are competitive: the 0.25 inlier floor.

    Without it, a handful of pixels on a low, badly-fitted plane outranks a well-supported
    floor purely by being lower, which is the failure mode the height rule would otherwise
    introduce on its own.
    """
    good = _cand(height=2.80, pitch_deg=30.0, count=9000)
    starved = _cand(height=3.60, pitch_deg=30.0, count=100)  # 1.1% of the best
    plane, _residual, _rows = choose_floor([good, starved])
    assert plane.height == pytest.approx(2.80)


@pytest.mark.parametrize(
    ("pitch_deg", "roll_deg", "why"),
    [
        (5.0, 0.0, "too shallow to be a floor this camera can see"),
        (89.0, 0.0, "past vertical"),
        (30.0, 35.0, "banked further than any mounted camera"),
    ],
)
def test_implausible_planes_are_refused_and_still_reported(pitch_deg, roll_deg, why):
    """No plane is returned, and every candidate still appears in `rows`.

    The refusal has to be legible: a caller that gets `None` needs to see what was
    rejected and why, or the only way to debug a camera that will not commission is to
    add prints to this function.
    """
    plane, residual, rows = choose_floor([_cand(2.8, pitch_deg, 9000, roll_deg=roll_deg)])
    assert plane is None and residual is None, why
    assert len(rows) == 1 and rows[0]["plausible_floor"] is False
    assert rows[0]["inliers_px"] == 9000


def test_no_candidates_at_all_is_a_refusal_not_a_crash():
    plane, residual, rows = choose_floor([])
    assert plane is None and residual is None and rows == []


# ---------------------------------------------------------------------------
# the derived numbers a caller gates on


def test_floor_scale_refuses_a_reference_pixel_that_is_not_on_the_floor():
    """A camera tilted far enough up that the reference row clears the horizon.

    `on_floor: False` rather than a very large number, for `pixel_to_ground`'s stated
    reason -- a number there would be read as a measurement.
    """
    cam = Camera.from_vfov(H_PX, W_PX, 60.0)
    out = floor_scale(cam, GroundPlane(2.8, math.radians(-35.0)), H_PX, W_PX)
    assert out["on_floor"] is False
    assert "m_per_px_u" not in out


def test_the_approach_to_the_horizon_is_finite_and_useless_before_it_is_a_refusal():
    """Where the refusal actually starts, and what the last few degrees before it report.

    The boundary is not where a reader would guess. Measured on a 2.8 m camera at 60 deg
    vfov, `on_floor` stays true down to about **-29 deg** of pitch, and on the way there
    the reported range runs away: -10 deg gives 13 m, **-20 deg gives 79.9 m**. Both are
    arithmetically right -- the ray does meet the plane -- and neither is a measurement of
    anything on a shop floor.

    So `on_floor` answers "does this ray meet the plane", not "is this number usable", and
    a caller gating only on the flag accepts an 80 m reading from a 2.8 m camera. Recorded
    as a property with its measured boundary rather than filed as a defect: the asymptote
    is what the projection is, and choosing a maximum credible range is a decision for the
    caller that has a store to compare it against.
    """
    cam = Camera.from_vfov(H_PX, W_PX, 60.0)
    near_horizon = floor_scale(cam, GroundPlane(2.8, math.radians(-20.0)), H_PX, W_PX)
    assert near_horizon["on_floor"] is True
    assert near_horizon["range_m"] > 50.0, "the finite-but-useless regime"

    ordinary = floor_scale(cam, GroundPlane(2.8, math.radians(30.0)), H_PX, W_PX)
    assert ordinary["range_m"] < 10.0
    assert near_horizon["m_per_px_u"] > 10 * ordinary["m_per_px_u"]


def test_a_pixel_is_worth_more_metres_the_further_away_the_floor_is():
    """Perspective, as a property rather than a fixed number.

    Both scales are metres per pixel at one reference row, so raising the camera pushes
    that row's floor point further away and each pixel spans more ground. Asserting the
    direction rather than the value keeps this a test of the projection and not of the
    0.85 reference constant.
    """
    cam = Camera.from_vfov(H_PX, W_PX, 60.0)
    low = floor_scale(cam, GroundPlane(2.0, math.radians(30.0)), H_PX, W_PX)
    high = floor_scale(cam, GroundPlane(4.0, math.radians(30.0)), H_PX, W_PX)
    assert low["on_floor"] and high["on_floor"]
    assert high["range_m"] > low["range_m"]
    assert high["m_per_px_u"] > low["m_per_px_u"]
    assert high["m_per_px_v"] > low["m_per_px_v"]


def test_column_health_counts_only_the_rows_that_reach_the_floor():
    cam = Camera.from_vfov(H_PX, W_PX, 60.0)
    out = column_health(cam, GroundPlane(2.8, math.radians(30.0)), H_PX, W_PX)
    assert 0 < out["floor_rows"] < H_PX, "a downward camera sees floor below the horizon only"
    assert out["range_near_m"] < out["range_far_m"]
    assert out["first_floor_row"] >= 0


def test_column_health_says_so_rather_than_inventing_a_range_when_nothing_is_floor():
    """A camera pointed at the ceiling: two rows are not enough to report a range."""
    cam = Camera.from_vfov(H_PX, W_PX, 60.0)
    out = column_health(cam, GroundPlane(2.8, math.radians(-45.0)), H_PX, W_PX)
    assert out["floor_rows"] < 2
    assert "range_near_m" not in out


# ---------------------------------------------------------------------------
# what reaches the fit: plate selection and the person-box gates


def test_the_brightest_daytime_plate_wins_and_the_slot_keys_are_utc(tmp_path):
    """Slot keys are UTC and the store is +8, so the window is not the one in the name.

    A plate written at `2326` UTC is 07:26 in the shop and is not daytime; one at `0530`
    is 13:30 and is. Getting this backwards picks a night plate for a camera that has a
    perfectly good day one, and every mask and metre downstream is fitted to a dark frame.
    """
    for slot, value in (
        ("20260816-233000", 250),
        ("20260816-033000", 100),
        ("20260816-053000", 200),
    ):
        Image.fromarray(np.full((20, 30, 3), value, np.uint8)).save(
            tmp_path / f"plate_{slot}.png"
        )
    assert pick_daytime_slot(tmp_path) == "20260816-053000", (
        "the 2330 plate is the brightest but is 07:30 store-local"
    )


def test_a_camera_with_no_daytime_plate_is_refused(tmp_path):
    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(
        tmp_path / "plate_20260816-233000.png"
    )
    with pytest.raises(SystemExit):
        pick_daytime_slot(tmp_path)


def _anns(tmp_path, annotations):
    path = tmp_path / "instances.json"
    path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bag"}],
                "images": [
                    {"id": 10, "file_name": "CamA__f0.jpg", "width": 960, "height": 540},
                    {"id": 11, "file_name": "CamB__f0.jpg", "width": 960, "height": 540},
                ],
                "annotations": annotations,
            }
        )
    )
    return path


def test_person_boxes_are_scaled_from_the_annotation_frame_to_the_plate(tmp_path):
    """`camera-json-is-calibrated-at-half-resolution` in one function.

    The annotations are 960x540 and the plate is 1920x1080, and the boxes must be scaled
    by two on the way in. Skipping it does not raise -- it returns metres, which is the
    failure `world_frame`'s `source_size_px` was added to stop elsewhere.
    """
    path = _anns(
        tmp_path,
        [{"image_id": 10, "category_id": 1, "bbox": [100, 100, 60, 180], "score": 0.9}],
    )
    boxes = load_person_boxes(path, "CamA", 1920, 1080, 0.0)
    assert boxes.tolist() == [[200.0, 200.0, 320.0, 560.0]]


@pytest.mark.parametrize(
    ("ann", "why"),
    [
        (
            {"image_id": 10, "category_id": 2, "bbox": [300, 100, 60, 180], "score": 0.9},
            "not a person",
        ),
        (
            {"image_id": 10, "category_id": 1, "bbox": [400, 100, 60, 180], "score": 0.2},
            "below the score gate",
        ),
        (
            {"image_id": 10, "category_id": 1, "bbox": [0, 100, 60, 180], "score": 0.9},
            "against the frame edge, so a crop",
        ),
        (
            {"image_id": 10, "category_id": 1, "bbox": [500, 100, 200, 180], "score": 0.9},
            "too wide to be standing",
        ),
        (
            {"image_id": 11, "category_id": 1, "bbox": [100, 100, 60, 180], "score": 0.9},
            "a different camera",
        ),
    ],
)
def test_every_gate_on_the_way_into_the_height_prior(tmp_path, ann, why):
    """Each of these boxes would move the median person height if it got through.

    The prior is a *median* over the survivors, so a gate that stops working does not
    fail, it shifts a camera's metres -- and `scale_source` would still read
    `person_height_median_vs_1.7m_prior_nNN` with a larger `NN`.
    """
    assert len(load_person_boxes(_anns(tmp_path, [ann]), "CamA", 1920, 1080, 0.0)) == 0, why


def test_person_checks_reports_the_prior_and_an_independent_bound():
    """The cross-check `person_checks` exists to be: the plane's answer beside a depth-free one.

    Handed a plane that is right, both agree and `scale_person` sits at 1.0 -- the ratio
    of the 1.70 m prior to what the boxes imply. The people-only fit is computed without
    the plane at all, which is what makes it a bound rather than a restatement.
    """
    cam_h, pitch_deg, vfov = 2.8, 25.0, 60.0
    boxes = _synth_people(vfov, cam_h, pitch_deg)
    cam = Camera.from_vfov(H_PX, W_PX, vfov)
    plane = GroundPlane(height=cam_h, pitch=math.radians(pitch_deg))
    out = person_checks(boxes, cam, plane, (H_PX, W_PX), vfov)
    assert out["boxes_used"] == len(boxes)
    assert out["implied_person_height_med_m"] == pytest.approx(ADULT_M, abs=0.01)
    assert out["scale_person"] == pytest.approx(1.0, abs=0.01)
    assert out["people_fit"]["height_m"] == pytest.approx(cam_h, abs=0.01)


def test_person_checks_records_the_failure_instead_of_dropping_the_key():
    """Too few boxes to fit: `people_fit` carries `failed`, and the caller can see why.

    A missing key and a recorded refusal read differently to anything downstream, and
    this is the branch the `SystemExit` type exists to reach.
    """
    boxes = _synth_people(60.0, 2.8, 25.0)[:3]
    cam = Camera.from_vfov(H_PX, W_PX, 60.0)
    plane = GroundPlane(height=2.8, pitch=math.radians(25.0))
    out = person_checks(boxes, cam, plane, (H_PX, W_PX), 60.0)
    assert "failed" in out["people_fit"]
    assert "implied_person_height_med_m" not in out, "fewer than five boxes measures nothing"


# ---------------------------------------------------------------------------
# floor selection through the real RANSAC, on depth this file synthesises


def _depth_of(cam: Camera, plane: GroundPlane, h: int, w: int, height_m=None, mask=None):
    """A depth frame for a plane `height_m` below the camera, in metres.

    The ray for each pixel is intersected with the plane analytically, so the frame is
    exactly what a perfect depth sensor would return -- which is the point: it lets
    `floor_candidates` be checked against an answer known independently of it. Beyond
    15 m the value is dropped rather than kept: a ray near the horizon meets the plane at
    hundreds of metres, arithmetically correct and not a reading any depth model produces.
    """
    v, u = np.mgrid[0:h, 0:w].astype(float)
    rays = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], axis=-1)
    level = rays @ plane.rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane.height if height_m is None else height_m) / level[..., 1]
    depth = np.where(np.isfinite(t) & (t > 0) & (t < 15.0), t, np.nan)
    return depth if mask is None else np.where(mask, depth, np.nan)


def test_the_floor_fit_recovers_the_plane_the_depth_was_built_from():
    """The other closed loop, and the one that runs the RANSAC rather than mocking it.

    `floor_candidates` is the step every commissioned camera's geometry comes out of, and
    nothing executed it. Given a depth frame computed from a known plane it returns that
    plane to **0.00 mm and 0.0000 degrees**, so the fit itself adds no error -- which is
    worth pinning separately from the depth model's, because in the field the two are only
    ever seen added together.
    """
    cam = Camera.from_vfov(240, 320, 60.0)
    truth = GroundPlane(height=2.8, pitch=math.radians(30.0))
    depth = _depth_of(cam, truth, 240, 320)

    plane, _residual, rows = choose_floor(floor_candidates(depth, cam, inlier_m=0.03))
    assert plane is not None and rows
    assert plane.height == pytest.approx(2.8, abs=0.005)
    assert math.degrees(plane.pitch) == pytest.approx(30.0, abs=0.05)
    assert math.degrees(plane.roll) == pytest.approx(0.0, abs=0.05)


def test_a_counter_across_the_seed_band_does_not_become_the_floor():
    """ "Lowest plausible", proven through the fit instead of on hand-built candidates.

    `fit_ground_plane` seeds from the lower part of the frame, so a counter in the near
    field is sampled *more* than the floor is -- the failure this rule exists for, and one
    a per-candidate check cannot see, because the counter's plane is a perfectly good
    plane. Here a surface 0.95 m above the floor fills the bottom 30% of the frame, and
    RANSAC genuinely proposes both: **h=1.85 with 23,040 inliers and h=2.80 with 39,360**.

    Selecting on inlier count would still pick the floor here, so the assertion is on the
    stronger property: both were offered, and the one further below the camera won.
    """
    cam = Camera.from_vfov(240, 320, 60.0)
    truth = GroundPlane(height=2.8, pitch=math.radians(30.0))
    counter = np.zeros((240, 320), dtype=bool)
    counter[int(240 * 0.70) :, :] = True

    floor = _depth_of(cam, truth, 240, 320)
    top = _depth_of(cam, truth, 240, 320, height_m=2.8 - 0.95, mask=counter)
    depth = np.where(np.isfinite(top), top, floor)

    plane, _residual, rows = choose_floor(floor_candidates(depth, cam, inlier_m=0.03))
    offered = {round(r["height_m"], 2) for r in rows}
    assert 1.85 in offered, "the counter must actually be proposed, or this proves nothing"
    assert 2.8 in offered
    assert plane.height == pytest.approx(2.8, abs=0.05), "the counter became the floor"


def test_a_depth_frame_with_nothing_in_it_yields_no_candidates():
    """`fit_ground_plane` needs 50 finite points in a band; below that it declines.

    An all-NaN frame is what a camera pointed at glass returns, and the refusal has to
    reach `choose_floor` as an empty list rather than as a plane fitted to noise.
    """
    cam = Camera.from_vfov(240, 320, 60.0)
    empty = np.full((240, 320), np.nan)
    assert floor_candidates(empty, cam, inlier_m=0.03) == []
    plane, residual, rows = choose_floor([])
    assert plane is None and residual is None and rows == []
