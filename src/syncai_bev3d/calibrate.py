"""Fixed-camera calibration from the floor tiles, in the order the parameters allow.

A shop's floor grout lines are straight by manufacture and its tiles are a known size.
That makes every frame its own calibration target, which matters because these are ceiling
cameras: nobody is walking a checkerboard under 48 of them, and a camera that never moves
only has to be solved once.

**The order is not a preference.** `scripts/fit_camera_from_people.py` recorded what
happens without it, measured on Taichung-cam01 on 2026-08-16: a pinhole fit to the tile
vanishing points gave 92.5 deg vfov and 65.4 deg pitch -- clean, prior-free, and it
measured 142 people at 1.0-1.2 m tall. **Every pinhole parameter was absorbing barrel
distortion, and the residual stayed healthy while it did**, so "lowest residual" selects
the most extreme lens rather than the real one. Undistorted first, the same tiles give
70.4 deg and 50.2 deg.

That script went in `500cdd2` and is at
`git show 500cdd2^:scripts/fit_camera_from_people.py`. Every number it is cited for in
this file is repeated above rather than left behind that pointer, which is the rule
`cli/scene.py` states: inline the measurement, point at the argument.

So:

1. **Distortion, from straightness alone.** No metric knowledge, no correspondences. Sweep
   the division model's `k1` and keep the value whose Hough accumulator is *sharpest*:
   straight grout makes a peak, curved grout smears across bins. On cam01 that found
   `k1 = -0.225` at a genuine **interior** maximum -- the first parameter in this project's
   camera work that did not run to its search boundary.
2. **Focal length and pitch, from vanishing points.** Still no metric knowledge: two
   orthogonal families of parallel floor lines fix the rotation and the focal length.
3. **Scale, from one known length.** A tile's pitch, measured once with a tape. On cam01
   that gave 2.38 m of camera height against 2.33 m from people under the corrected lens,
   where the uncorrected fit had said 1.96 m. A ceiling mount is 2.3-2.4 m.

**The acceptance test is binary and it is `horizon_row`.** At 43 deg the horizon sat at row
36 *inside* a 1080-row frame, so every row below it was a grazing ray manufacturing floor
that is not there -- the centre column spanned 0.07 to 3543 m, worst row 2275 m. At 50.2
deg it sits at row -379 and grazing rays are geometrically impossible: 0.19 to 8.9 m, worst
row 0.031 m. **If the horizon is inside the image, the calibration is wrong**, and a metric
range gate will hide that rather than fix it by putting the artefact's edge at a tidy number
the eye reads as the end of the room.

Pure numpy: this box has no cv2, no scipy and no skimage, and a deployment box need not
either. Everything here is a few hundred lines of transform that is easier to test than to
depend on -- `tests/test_calibrate.py` recovers a known `k1` from a synthetic grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from syncai_hydranet.geometry.ground import undistort_points


@dataclass(frozen=True)
class Pose:
    """What a calibration recovers, and what it still assumes."""

    k1: float
    vfov_deg: float
    pitch_deg: float
    height_m: float | None  # None until a known length fixes the scale
    horizon_row: float
    image_h: int
    image_w: int

    @property
    def horizon_is_inside(self) -> bool:
        """Literally: does the horizon fall within the image rows.

        Kept as the plain geometric fact. It is **not** the acceptance test, and reading it
        as one is the bug this class shipped with -- see ``rejection``.
        """
        return 0 <= self.horizon_row < self.image_h

    @property
    def rejection(self) -> str | None:
        """Why this pose is unusable for a ground projection, or None if it is usable.

        **The only acceptable horizon for a downward-looking camera is above the frame.**
        `horizon_is_inside` was the acceptance test and it admitted a pose whose horizon sat
        at row 1425 of 1080 -- outside the frame, and therefore "pass" -- when outside
        *below* means ``tan(pitch) < 0`` means the camera is pitched **up**. That was a
        ceiling camera pointed at a floor, and `[pass]` was what a fleet report would have
        printed for it in a table of 48.

        `GroundPlane` fixes the sign: pitch is positive when the camera looks down. So
        there are two distinct failures either side of the frame, and they are different
        enough to name separately:

        * horizon **inside** the image -- rows below it are grazing rays and manufacture
          floor that is not there. At 43 deg on cam01 the centre column spanned 0.07 to
          3543 metres.
        * horizon **below** the image -- the fit has the camera looking up, and there is no
          floor in view at all rather than too much of it.
        """
        if self.horizon_row >= self.image_h:
            return (
                f"horizon at row {self.horizon_row:.0f} is below the frame, so the fit has "
                f"the camera pitched up ({self.pitch_deg:+.1f}deg); a ceiling camera "
                f"looking at a floor cannot be"
            )
        if self.horizon_row >= 0:
            return (
                f"horizon at row {self.horizon_row:.0f} is inside the {self.image_h}-row "
                f"frame; every row below it is a grazing ray and manufactures floor"
            )
        return None

    def summary(self) -> str:
        why = self.rejection
        height = "unscaled" if self.height_m is None else f"{self.height_m:.2f} m"
        line = (
            f"k1={self.k1:+.3f}  vfov={self.vfov_deg:.1f}deg  pitch={self.pitch_deg:.1f}deg  "
            f"height={height}  horizon row {self.horizon_row:.0f}/{self.image_h}  "
            f"[{'pass' if why is None else 'REJECT'}]"
        )
        return line if why is None else f"{line}\n    {why}"


def floor_edge_points(
    grey: np.ndarray,
    floor_mask: np.ndarray,
    quantile: float = 0.92,
    return_orientation: bool = False,
):
    """Edge points on the floor, as (x, y), for the distortion fit to work on.

    With `return_orientation` it also returns each point's gradient direction on [0, pi),
    which `hough` needs to vote by orientation rather than into all 180 bins. **Quantile is
    a free parameter with no calibration behind it and the fitted k1 moves with it** --
    measured on Taichung-cam01, 0.92 gives -0.255, 0.96 gives -0.150, and 0.98 and above
    run to the search boundary where `is_interior_maximum` correctly refuses them. So a
    caller reporting one k1 without the band it is stable over is reporting a choice as a
    measurement.

    **Masked by the segmentation, not by a rectangular ROI**, and that detail cost a run.
    A box around the lower half of the frame catches counter edges, shelf bases and the
    skirting line, and the Hough then latches onto those instead of the grout -- they are
    longer, straighter and more contrasty than a tile joint. The floor mask is the model's
    own output and it is exactly the region where "the lines are straight by manufacture"
    is true.

    Sobel rather than a real edge detector: this needs the *locations* of high gradient,
    not thin one-pixel chains, because the Hough votes are what does the thinning.
    """
    grey = np.asarray(grey, float)
    if grey.shape != np.asarray(floor_mask).shape:
        raise ValueError(f"grey {grey.shape} and floor_mask {np.shape(floor_mask)} differ")
    gy, gx = np.gradient(grey)
    mag = np.hypot(gx, gy)
    inside = np.asarray(floor_mask, bool)
    if not inside.any():
        return np.zeros((0, 2))
    thresh = float(np.quantile(mag[inside], quantile))
    keep = inside & (mag > thresh)
    ys, xs = np.nonzero(keep)
    pts = np.stack([xs, ys], axis=1).astype(float)
    if not return_orientation:
        return pts
    # The Hough normal direction, which is the gradient direction, folded onto [0, pi)
    # because a line and its reverse are the same line.
    return pts, np.mod(np.arctan2(gy[keep], gx[keep]), math.pi)


def _renormalise(points: np.ndarray, centre, target_rms: float) -> np.ndarray:
    """Rescale a point set about the centre to a fixed RMS radius.

    **Without this the sweep measures contraction, not straightness.** A positive `k1`
    pulls every point toward the centre, which clusters the Hough votes and raises the
    concentration no matter how curved the lines still are -- so the objective ran to the
    positive end of the search range on synthetic grids with *zero* distortion. The
    division model is only defined up to scale for this purpose anyway, since a pinhole
    focal length absorbs it, so fixing the scale removes a degeneracy rather than
    discarding information.
    """
    centred = points - np.asarray(centre, float)
    rms = float(np.sqrt((centred**2).sum(axis=1).mean()))
    if rms < 1e-9:
        return points
    return centred * (target_rms / rms) + np.asarray(centre, float)


def hough(
    points: np.ndarray,
    shape,
    n_theta: int = 180,
    n_rho: int = 256,
    orientations: np.ndarray | None = None,
    spread_deg: float = 12.0,
) -> np.ndarray:
    """Accumulator over (theta, rho) for a set of edge points.

    Written out rather than imported: this box has no cv2 and a Jetson need not have one
    either, and the transform is fifteen lines whose behaviour the objective below depends
    on precisely.

    **`orientations` is not an optimisation, it is a correctness fix, and the failure it
    removes is worth stating because it looks like a result.** Without it every point votes
    into all 180 thetas, so a dense blob of edge pixels builds a ridge through its own
    centroid -- the votes agree about the *centroid* whatever the lines are doing. Peaks
    extracted from that ridge give thirty lines all passing near one point: a starburst,
    rendered over the frame, that a reader takes for a vanishing point.

    Given each point's gradient normal a point votes only near its own orientation, and the
    ridge cannot form because no point contributes to a theta its gradient disagrees with.
    Measured on Taichung-cam01: gradient orientations over 44,379 masked edge points are
    strongly bimodal, two modes about 90 degrees apart at 3221 and 5849 votes against a
    mid-range floor of 450-500. **The tile grid was in the data and the unoriented
    accumulator was not using it.**

    `spread_deg` is the half-width of that window: wide enough for gradient noise and the
    perspective fan within one family, narrow enough that two families 90 degrees apart do
    not bleed together.
    """
    h, w = shape
    if len(points) == 0:
        return np.zeros((n_theta, n_rho))
    theta = np.linspace(0.0, math.pi, n_theta, endpoint=False)
    rho_max = math.hypot(h, w)
    rho = (
        points[:, 0, None] * np.cos(theta)[None, :]
        + points[:, 1, None] * np.sin(theta)[None, :]
    )
    bins = np.clip(((rho + rho_max) / (2 * rho_max) * n_rho).astype(int), 0, n_rho - 1)

    weights = None
    if orientations is not None:
        if len(orientations) != len(points):
            raise ValueError(
                f"orientations has {len(orientations)} entries for {len(points)} points"
            )
        # Both angles live on a half circle, so the difference wraps at pi, not 2pi.
        delta = np.abs(np.asarray(orientations, float)[:, None] - theta[None, :])
        delta = np.minimum(delta, math.pi - delta)
        weights = (delta <= math.radians(spread_deg)).astype(float)

    acc = np.zeros((n_theta, n_rho))
    for t in range(n_theta):
        acc[t] = np.bincount(
            bins[:, t], weights=None if weights is None else weights[:, t], minlength=n_rho
        )
    return acc


def concentration(acc: np.ndarray, n_orientations: int = 2) -> float:
    """How sharply the accumulator peaks, measured **per orientation** and then averaged.

    Per orientation, not over the whole accumulator, and that is the whole reason this
    works. A floor grid has two line directions, so only 2 of 180 theta rows carry any
    straightness signal at all; the other 178 are smeared by construction whatever the
    lens is doing. Scoring the accumulator as a block let those 178 dominate, and the sweep
    then ran to whichever end of the range moved the most points -- which is the failure
    this project's earlier camera fits kept producing and calling a result.

    Within one theta row it is sum-of-squares over the square of the sum, so a row is
    normalised by its own vote count and cannot win by holding fewer votes.
    """
    if acc.size == 0:
        return 0.0
    totals = acc.sum(axis=1)
    good = totals > 0
    if not good.any():
        return 0.0
    per_theta = np.zeros(acc.shape[0])
    per_theta[good] = (acc[good] ** 2).sum(axis=1) / (totals[good] ** 2)
    k = max(1, min(n_orientations, acc.shape[0]))
    return float(np.sort(per_theta)[-k:].mean())


def fit_k1(
    points: np.ndarray, shape, k_range=(-0.6, 0.3), steps: int = 61
) -> tuple[float, np.ndarray]:
    """Sweep the division model and return the `k1` whose lines are straightest.

    Returns the argmax and the whole sweep, because **the shape of the curve is the result
    and the argmax alone is not.** A maximum at either end of the range means the objective
    is still improving where the search stopped, which is what every earlier camera fit in
    this project did, and it should be read as "no answer" rather than as the endpoint. The
    cam01 fit is trustworthy because `k1 = -0.225` sits in the interior.

    **The range may not reach `|k1| = 1`, and widening it to chase a boundary hit is the
    one thing never to do here.** The division model is `r_u = r_d / (1 + k1 r_d^2)` with
    `r` normalised so the image corner is `r_d = 1`, so its singularity `1/sqrt(-k1)` moves
    *inside the frame* once `|k1| >= 1`. Past that a few points map to enormous radii,
    `_renormalise` rescales the set to a fixed RMS radius to compensate, and everything else
    collapses into a dense blob at the centre -- which is precisely what `concentration`
    rewards. The objective stops scoring straightness and starts scoring collapse, and the
    false peak it grows sits at about **-1.14**.

    Measured 2026-08-31: on a `(-1.2, 0.3)` sweep a **full-frame, centred, exactly
    zero-distortion grid** returns `k1 = -1.140` and `is_interior_maximum` reports True,
    because that peak lands about four steps inside the bound rather than on it. So the
    refusal has to be here, on the range, rather than in the interior check.

    Inside `|k1| < 1` the interior check is enough: at the widest legal range a
    zero-distortion grid runs to the bound and is correctly refused. But that also means a
    genuine lens beyond about -0.6 cannot be measured by this objective at all -- the six
    `dingpu-1f` cameras all run to -0.6 on real floor points while their footprints return
    ~0 on the control, so those lenses are past what this can reach. **Before believing any
    sweep, run the same footprint with a synthetic straight grid and require ~0**; that flat
    control is what stops an artefact reading as a verdict.
    """
    if not all(abs(float(k)) < 1.0 for k in k_range):
        raise ValueError(
            f"k_range={k_range} reaches |k1| >= 1, where the division model's singularity "
            "is inside the frame and this objective rewards collapse rather than "
            "straightness (a zero-distortion grid fits -1.14 there). A boundary hit at a "
            "legal range means no answer, not a wider search."
        )
    h, w = shape
    centre = (w / 2.0, h / 2.0)
    radius = math.hypot(h, w) / 2.0
    ks = np.linspace(k_range[0], k_range[1], steps)
    target_rms = float(np.sqrt(((np.asarray(points, float) - centre) ** 2).sum(axis=1).mean()))
    scores = np.array(
        [
            concentration(
                hough(
                    _renormalise(
                        undistort_points(points, k, centre, radius), centre, target_rms
                    ),
                    shape,
                )
            )
            for k in ks
        ]
    )
    return float(ks[int(scores.argmax())]), np.stack([ks, scores])


def is_interior_maximum(sweep: np.ndarray, margin: int = 2) -> bool:
    """Whether the sweep's best value is a real peak or the edge of the search.

    The distinction that separates a measurement from a search-boundary artefact, and the
    one thing `fit_camera_from_people.py` could never satisfy: its residual fell
    monotonically with the assumed vfov all the way to 150 degrees.
    """
    i = int(sweep[1].argmax())
    return margin <= i < sweep.shape[1] - margin


class VanishingPoint(NamedTuple):
    """Where a family of lines meets, and how well it actually meets there.

    The residual is the reason this is not a bare array. Grouping image lines into two
    floor families is the step most likely to be subtly wrong, and a subtly wrong grouping
    produces a **plausible focal length** rather than an error: `pose_from_vanishing_points`
    only refuses the gross case, where the dot product comes out non-negative.

    Under perspective a family of parallel floor lines does not share a theta -- the lines
    fan toward their vanishing point -- so clustering by theta looks reasonable and is
    wrong. The residual is what distinguishes "these lines meet at a point" from "these
    lines were put in the same bucket".
    """

    point: np.ndarray
    residual_px: float

    def __array__(self, dtype=None, copy=None):
        """So a caller that only wants the point can keep treating it as one."""
        arr = np.asarray(self.point, dtype=dtype)
        return arr.copy() if copy else arr


def vanishing_point(lines: np.ndarray) -> VanishingPoint:
    """Least-squares intersection of a family of image lines given as (theta, rho).

    Parallel floor lines meet at a vanishing point under perspective; in homogeneous terms
    each line is `(cos t, sin t, -rho)` and the point is the null space of their stack.

    The residual is the RMS perpendicular distance from the recovered point to each line,
    in pixels, which is directly readable against the frame's own size -- a few pixels is a
    family, a few hundred is a bucket.
    """
    a = np.stack([np.cos(lines[:, 0]), np.sin(lines[:, 0]), -lines[:, 1]], axis=1)
    _, _, vt = np.linalg.svd(a)
    v = vt[-1]
    if abs(v[2]) < 1e-12:
        raise ValueError("lines are parallel in the image: no finite vanishing point")
    point = v[:2] / v[2]
    normals = np.stack([np.cos(lines[:, 0]), np.sin(lines[:, 0])], axis=1)
    residual = float(np.sqrt((((normals @ point) - lines[:, 1]) ** 2).mean()))
    return VanishingPoint(point, residual)


def pose_from_vanishing_points(v1, v2, shape) -> tuple[float, float]:
    """Focal length in pixels and down-pitch in degrees, from two orthogonal families.

    For orthogonal directions the vanishing points satisfy
    ``(v1 - c) . (v2 - c) + f^2 = 0`` about the principal point ``c``, which gives ``f``
    directly. The pitch then follows from where the horizon -- the line through both --
    sits relative to the principal point.
    """
    h, w = shape
    c = np.array([w / 2.0, h / 2.0])
    d1, d2 = np.asarray(v1, float) - c, np.asarray(v2, float) - c
    dot = float(d1 @ d2)
    if dot >= 0:
        raise ValueError(
            f"vanishing points are not consistent with orthogonal directions (dot={dot:.1f}); "
            "the two line families are probably not perpendicular on the floor"
        )
    focal = math.sqrt(-dot)
    # The horizon passes through both vanishing points; its perpendicular distance below
    # the principal point is what the pitch is measured from.
    direction = d2 - d1
    length = float(np.hypot(*direction))
    horizon_offset = float(np.cross(direction, -d1)) / max(length, 1e-9)
    return focal, math.degrees(math.atan2(horizon_offset, focal))


def horizon_row(pitch_deg: float, focal_px: float, image_h: int) -> float:
    """Image row of the horizon. Outside the frame is the only acceptable answer.

    Below it every ray is grazing, so a ground-plane projection manufactures floor that is
    not there: at 43 deg on a 1080-row frame the horizon sat at row 36 and the centre
    column spanned 0.07 to 3543 metres.
    """
    return image_h / 2.0 - focal_px * math.tan(math.radians(pitch_deg))


def height_from_known_length(pixel_length: float, metres: float, focal_px: float,
                             pitch_deg: float, row: float, image_h: int) -> float:  # fmt: skip
    """Camera height, from one measured length lying on the floor.

    The only metric input the whole procedure takes. Everything before this is scale-free,
    which is why it can be done from a recording with nobody on site.
    """
    if pixel_length <= 0 or metres <= 0:
        raise ValueError("pixel_length and metres must both be positive")
    dy = row - image_h / 2.0
    ground_angle = math.radians(pitch_deg) + math.atan2(dy, focal_px)
    if ground_angle <= 1e-6:
        raise ValueError("that row is at or above the horizon; it is not on the floor")
    # A length lying across the view at that row images at focal * L / range.
    range_m = metres * focal_px / pixel_length
    return range_m * math.sin(ground_angle)
