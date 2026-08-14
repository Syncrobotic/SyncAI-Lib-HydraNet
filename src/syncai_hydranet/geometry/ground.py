"""Camera geometry against the floor the robot is standing on.

The pose of the camera above the floor is fitted to the depth return every frame rather
than measured once. A wheeled robot's mount is a constant; a walking quadruped pitches
and rolls with every step, so a fixed homography smears, and it smears more the further
out you look -- which is exactly where a navigation stack is asking the question.

Fitting also beats reading the IMU, which gives angles but not height, and height changes
with gait too.
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
