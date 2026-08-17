"""Fixed-camera calibration from the floor tiles, in the order the parameters allow.

A shop's floor grout lines are straight by manufacture and its tiles are a known size.
That makes every frame its own calibration target, which matters because these are ceiling
cameras: nobody is walking a checkerboard under 24 of them, and a camera that never moves
only has to be solved once.

**The order is not a preference.** `scripts/fit_camera_from_people.py` records what happens
without it, measured on Taichung-cam01 on 2026-08-16: a pinhole fit to the tile vanishing
points gave 92.5 deg vfov and 65.4 deg pitch -- clean, prior-free, and it measured 142
people at 1.0-1.2 m tall. **Every pinhole parameter was absorbing barrel distortion, and
the residual stayed healthy while it did**, so "lowest residual" selects the most extreme
lens rather than the real one. Undistorted first, the same tiles give 70.4 deg and 50.2 deg.

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

import numpy as np


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
        """The acceptance test. True means the fit manufactures floor that is not there."""
        return 0 <= self.horizon_row < self.image_h

    def summary(self) -> str:
        verdict = "REJECT" if self.horizon_is_inside else "pass"
        height = "unscaled" if self.height_m is None else f"{self.height_m:.2f} m"
        return (
            f"k1={self.k1:+.3f}  vfov={self.vfov_deg:.1f}deg  pitch={self.pitch_deg:.1f}deg  "
            f"height={height}  horizon row {self.horizon_row:.0f}/{self.image_h}  [{verdict}]"
        )


def floor_edge_points(grey: np.ndarray, floor_mask: np.ndarray, quantile: float = 0.92):
    """Edge points on the floor, as (x, y), for the distortion fit to work on.

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
    ys, xs = np.nonzero(inside & (mag > thresh))
    return np.stack([xs, ys], axis=1).astype(float)


def undistort_points(xy: np.ndarray, k1: float, centre, radius: float) -> np.ndarray:
    """Fitzgibbon's one-parameter division model, applied to points rather than an image.

    ``x_u = x_d / (1 + k1 * r^2)`` with ``r`` normalised by ``radius``, so ``k1`` is
    dimensionless and comparable between cameras of different resolutions. Points, not an
    image, because the objective below only ever looks at edge pixels and resampling the
    whole frame per candidate ``k1`` would cost a hundred times as much to answer the same
    question.
    """
    centred = np.asarray(xy, float) - np.asarray(centre, float)
    r2 = (centred**2).sum(axis=1) / (radius**2)
    return centred / (1.0 + k1 * r2)[:, None] + np.asarray(centre, float)


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


def hough(points: np.ndarray, shape, n_theta: int = 180, n_rho: int = 256) -> np.ndarray:
    """Accumulator over (theta, rho) for a set of edge points.

    Written out rather than imported: this box has no cv2 and a Jetson need not have one
    either, and the transform is fifteen lines whose behaviour the objective below depends
    on precisely.
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
    acc = np.zeros((n_theta, n_rho))
    for t in range(n_theta):
        acc[t] = np.bincount(bins[:, t], minlength=n_rho)
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
    """
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


def vanishing_point(lines: np.ndarray) -> np.ndarray:
    """Least-squares intersection of a family of image lines given as (theta, rho).

    Parallel floor lines meet at a vanishing point under perspective; in homogeneous terms
    each line is `(cos t, sin t, -rho)` and the point is the null space of their stack.
    """
    a = np.stack([np.cos(lines[:, 0]), np.sin(lines[:, 0]), -lines[:, 1]], axis=1)
    _, _, vt = np.linalg.svd(a)
    v = vt[-1]
    if abs(v[2]) < 1e-12:
        raise ValueError("lines are parallel in the image: no finite vanishing point")
    return v[:2] / v[2]


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
