"""The runtime contract between pixels and floor metres, applied every frame.

This module is the part of the geometry a *serving* process needs: the camera model, the
ground plane, the projections between them, and the lens undistortion those projections
assume. Everything that **produces** those parameters -- plate calibration, k1 fitting,
pose recovery, BEV rendering -- lives in `syncai_bev3d` and runs once per camera at
commissioning. The dependency rule follows: `syncai_bev3d` imports this module; nothing
on the serving path imports `syncai_bev3d` (`tests/test_package_boundaries.py` checks).

The docstring this one replaced explained per-frame plane fitting for a walking
quadruped. That line is history (`cc80fc3`); on a fixed CCTV camera the plane is a
constant, fitted once and cached in `camera.json`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics, in pixels, for the image the mask was computed on."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_vfov(cls, height_px: int, width_px: int, vfov_deg: float) -> Camera:
        """Assume square pixels and a vertical field of view. For clips with no calibration."""
        fy = (height_px / 2) / math.tan(math.radians(vfov_deg) / 2)
        return cls(fy, fy, width_px / 2, height_px / 2)

    @classmethod
    def from_calibration(cls, calib: dict, key: str = "color") -> Camera:
        """Read a recorder session's ``*_calibration.json``.

        The stored K already accounts for the recorder's subsample stride, so it belongs
        to the images on disk rather than to the topic they came from. Using the topic's
        K against subsampled frames is off by exactly that stride, silently.
        """
        k = calib[key]["K"] if key in calib else calib["K"]
        k = np.asarray(k, dtype=float).reshape(3, 3)
        return cls(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))

    def scaled_to(self, height_px: int, width_px: int, from_shape: tuple[int, int]) -> Camera:
        """Rescale intrinsics when the mask is not the size the calibration was taken at."""
        sy, sx = height_px / from_shape[0], width_px / from_shape[1]
        return Camera(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)


@dataclass(frozen=True)
class GroundPlane:
    """Where the floor is, relative to the camera.

    ``height`` is the camera's height above it in metres. ``pitch`` is positive when the
    camera looks down, ``roll`` positive when it banks right. Both in radians.
    """

    height: float
    pitch: float
    roll: float = 0.0

    @property
    def rotation(self) -> np.ndarray:
        """Level frame (x right, y down, z forward) -> camera frame."""
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        return r_roll @ r_pitch


def ground_to_pixel(x: np.ndarray, z: np.ndarray, cam: Camera, plane: GroundPlane):
    """Floor points (x lateral, z forward, metres) -> pixels, plus camera-frame depth."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    pts = np.stack([x, np.full_like(x, plane.height), z], axis=-1)
    cam_pts = pts @ plane.rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * (cam_pts[..., 0] / cam_pts[..., 2]) + cam.cx
        v = cam.fy * (cam_pts[..., 1] / cam_pts[..., 2]) + cam.cy
    return u, v, cam_pts[..., 2]


def pixel_to_ground(u: np.ndarray, v: np.ndarray, cam: Camera, plane: GroundPlane):
    """Pixels -> the floor point each ray lands on. NaN where the ray misses the floor.

    A ray at or above the horizon never meets the plane; those pixels come back NaN rather
    than as a very large distance, because a number there would be read as a measurement.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    rays = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], axis=-1)
    level = rays @ plane.rotation  # inverse of the rotation, which is orthonormal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = plane.height / level[..., 1]
    t = np.where(level[..., 1] > 1e-9, t, np.nan)
    return level[..., 0] * t, level[..., 2] * t


def unproject(depth_m: np.ndarray, cam: Camera) -> np.ndarray:
    """Metric depth image -> HxWx3 points in the camera frame. Zero depth becomes NaN."""
    h, w = depth_m.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = np.where(depth_m > 0, depth_m, np.nan)
    return np.stack([(uu - cam.cx) / cam.fx * z, (vv - cam.cy) / cam.fy * z, z], axis=-1)


def fit_ground_plane(
    depth_m: np.ndarray,
    cam: Camera,
    *,
    lower_fraction: float = 0.5,
    iterations: int = 120,
    inlier_m: float = 0.03,
    seed: int = 0,
) -> tuple[GroundPlane | None, np.ndarray]:
    """RANSAC the floor out of a depth frame.

    Returns the plane and a full-frame residual map in metres -- signed distance from the
    fitted plane, NaN where depth returned nothing. The residual is the second output, not
    a diagnostic: a polished floor scatters the projector's pattern away and comes back
    empty, and those pixels are the ones that fool the camera and the depth sensor at the
    same time. Reflections that do return sit *on* the plane; glass does not.

    Only the lower part of the frame seeds the fit, because that is where a forward-facing
    camera sees floor rather than wall.
    """
    pts = unproject(depth_m, cam)
    h = depth_m.shape[0]
    band = pts[int(h * (1 - lower_fraction)) :].reshape(-1, 3)
    band = band[np.isfinite(band).all(axis=1)]
    if len(band) < 50:
        return None, np.full(depth_m.shape, np.nan)

    rng = np.random.default_rng(seed)
    best_normal, best_d, best_count = None, 0.0, 0
    for _ in range(iterations):
        tri = band[rng.choice(len(band), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        if normal[1] > 0:  # up is -y in camera coords, so an up-pointing normal has y < 0
            normal = -normal
        d = -normal @ tri[0]
        count = int((np.abs(band @ normal + d) < inlier_m).sum())
        if count > best_count:
            best_normal, best_d, best_count = normal, d, count

    if best_normal is None or best_count < 50:
        return None, np.full(depth_m.shape, np.nan)

    # Refine on the inliers: three points set the hypothesis, every inlier sets the answer.
    inliers = band[np.abs(band @ best_normal + best_d) < inlier_m]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[1] > 0:
        normal = -normal
    d = float(-normal @ centroid)

    # The normal is the floor's up-vector seen by the camera, which for a level frame
    # rotated by (pitch, roll) is (cos p sin r, -cos p cos r, -sin p). Read the pose back
    # off those components rather than re-deriving it from the points.
    height = abs(d)
    pitch = math.asin(float(np.clip(-normal[2], -1.0, 1.0)))
    roll = math.atan2(float(normal[0]), float(-normal[1]))
    residual = np.einsum("hwc,c->hw", pts, normal) + d
    return GroundPlane(height=height, pitch=pitch, roll=roll), residual


def undistort_points(xy: np.ndarray, k1: float, centre, radius: float) -> np.ndarray:
    """Fitzgibbon's one-parameter division model, applied to points rather than an image.

    ``x_u = x_d / (1 + k1 * r^2)`` with ``r`` normalised by ``radius``, so ``k1`` is
    dimensionless and comparable between cameras of different resolutions.

    This lives here rather than with the k1 *fitting* in `syncai_bev3d.calibrate`
    because it is the application side of the lens contract: commissioning fits ``k1``
    once, and every runtime consumer of `camera.json` must undo the lens with the exact
    same model or the metres drift silently. One definition, both sides import it.
    """
    centred = np.asarray(xy, float) - np.asarray(centre, float)
    r2 = (centred**2).sum(axis=1) / (radius**2)
    return centred / (1.0 + k1 * r2)[:, None] + np.asarray(centre, float)


def distort_points(xy: np.ndarray, k1: float, centre, radius: float) -> np.ndarray:
    """The exact inverse of `undistort_points`: ideal pixels -> the raw frame's pixels.

    Needed by everything that draws a metre-space object back onto a real frame.
    `ground_to_pixel` returns *ideal* pixels -- the lens has been divided out of the
    camera model -- while a decoded frame, a commissioning plate and every mask beside it
    are raw. Overlaying one on the other without this puts a zone outline a few pixels off
    its own floor, which reads as "the calibration is a bit loose" and is nothing of the
    kind. It is the same silent drift `undistort_points` warns about, in the other
    direction, and the reason both halves live in one file.

    The division model ``x_u = x_d / (1 + k1 |x_d|^2 / R^2)`` inverts in closed form.
    Writing ``a = |x_u|`` and ``b = |x_d|`` gives ``(k1 a / R^2) b^2 - b + a = 0``; the
    root taken is the one that tends to ``a`` as ``k1`` tends to zero, which is the
    physical branch -- the other runs off to infinity and describes no lens.

    ``q < 0`` is beyond the model's turning point: no real pre-image exists, and inventing
    one would move a point to a place the lens cannot map from. Those come back NaN, which
    every consumer of this module already reads as "not measured".
    """
    xy = np.asarray(xy, float)
    if abs(k1) < 1e-12:
        return xy.copy()
    centre_a = np.asarray(centre, float)
    centred = xy - centre_a
    a = np.hypot(centred[:, 0], centred[:, 1])
    q = 1.0 - 4.0 * k1 * a**2 / (radius**2)
    safe_a = np.where(a > 1e-9, a, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (1.0 - np.sqrt(np.clip(q, 0.0, None))) * radius**2 / (2.0 * k1 * safe_a)
    scale = np.where(a > 1e-9, b / safe_a, 1.0)
    scale = np.where(q >= 0, scale, np.nan)
    return centred * scale[:, None] + centre_a


def height_above_floor_m(
    x_m: float, z_m: float, v_px: float, camera: Camera, plane: GroundPlane
) -> float:
    """How high above the floor a pixel row is, for a thing standing at ``(x_m, z_m)``.

    Given the floor position of an object and the image row of its topmost pixel, this
    returns the height that top must be at. Applied to a person box it is the top of the
    head: **a standing adult reads ~1.7 m, a person bent over a counter ~1.2 m, and a
    person on the floor under 0.5 m.**

    That last separation is the reason this is in the wheel rather than in the
    commissioning tool where it started. Two independent measurements say the posture
    angle cannot do it alone: NTU RGB+D's own ground-truth 3D puts `A43 falling down` at a
    74.5 deg median peak torso angle and `A06 pick up` at 76.3 deg, and on
    Kaohsiung-cam04 a shopper leaning over a counter produced a `fall` with a torso at
    69 deg and a box 21% shorter -- a bend passes every image-space test a fall passes.
    Height above the floor is what separates them, and it needs the camera pose that
    commissioning already measured.

    A head top at level-frame height ``plane.height - h`` projects linearly in ``h``, so
    this is one linear solve rather than a search. ``v_px`` must be in the pixel space
    `camera` was calibrated on -- several of this project's cameras are fitted at half
    the resolution their clips decode at.

    NaN when the ray is degenerate. The caller decides what an implausible answer means:
    for a person box it usually means the top was not a head top -- a merged box, or one
    truncated by the frame edge.
    """
    rot = plane.rotation
    a = rot @ np.array([x_m, plane.height, z_m])
    b = rot @ np.array([0.0, 1.0, 0.0])
    k = (v_px - camera.cy) / camera.fy
    den = b[1] - k * b[2]
    if abs(den) < 1e-9:
        return float("nan")
    return float((a[1] - k * a[2]) / den)


def pixel_row_at_height(
    x_m: float, z_m: float, h_m: float, camera: Camera, plane: GroundPlane
) -> float:
    """The image row a point ``h_m`` above the floor at ``(x_m, z_m)`` projects to.

    The inverse of `height_above_floor_m`, and it exists for a use the forward direction
    cannot serve: **cutting a mask at a class's plausible height.**

    A `display_table` component that reaches 2.44 m is not a 2.44 m table. It is a counter
    whose mask has been welded to the wall behind it -- PLAN §7.21 measured that welding
    directly, 263 of 503 merges on Taichung-cam01 being containment merges, because a
    counter standing in front of a wall is 100% inside the wall's mask from any single
    viewpoint. Dropping the component loses a real counter; keeping it draws a 2.4 m slab
    where the shop has a waist-high fixture. Cutting the mask at the row that 1.3 m
    projects to keeps the counter and discards the wall, and needs no depth at all, which
    matters because the depth model collapses on exactly the white surfaces this happens on.

    Same linear relation as the forward solve, rearranged. NaN on a degenerate ray.
    """
    rot = plane.rotation
    a = rot @ np.array([x_m, plane.height, z_m])
    b = rot @ np.array([0.0, 1.0, 0.0])
    den = a[2] - h_m * b[2]
    if abs(den) < 1e-9:
        return float("nan")
    return float(camera.cy + camera.fy * (a[1] - h_m * b[1]) / den)
