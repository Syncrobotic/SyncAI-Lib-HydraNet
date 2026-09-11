"""Build scene geometry from one depth pass and the runtime camera contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points, unproject


def geometry_signature(cf: CameraFile) -> str:
    """Only fields that change geometry; adding a service zone does not invalidate depth."""
    payload = {
        key: asdict(cf)[key]
        for key in ("camera_id", "image_size_px", "camera", "plane", "lens")
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_geometry_cache(
    cf: CameraFile,
    depth: np.ndarray,
    *,
    depth_scale: float,
    frame_hw: tuple[int, int] = (1080, 1920),
) -> dict[str, np.ndarray]:
    """Undistorted optical-axis depth to raw-frame geometry, with calibration provenance.

    Depth must use the same uncropped field of view as camera.json. Scale is explicit:
    use the onboard calibration's scale for its own DA-V2 plate, never a hidden prior.
    """
    cf.validate()
    depth = np.asarray(depth, float)
    if depth.ndim != 2 or min(depth.shape) < 2:
        raise ValueError("depth must be an H x W image, at least 2 x 2")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    ph, pw = depth.shape
    h, w = frame_hw
    if h < 2 or w < 2:
        raise ValueError("frame_hw must be at least 2 x 2")
    cam = cf.camera.scaled_to(ph, pw, (cf.image_size_px[1], cf.image_size_px[0]))
    level = unproject(depth * depth_scale, cam) @ cf.plane.rotation
    height = cf.plane.height - level[..., 1]
    normal = np.cross(np.gradient(level, axis=1), np.gradient(level, axis=0))
    norm = np.linalg.norm(normal, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        horiz = np.abs(normal[..., 1]) / np.where(norm > 1e-12, norm, np.nan)
    vv, uu = np.mgrid[0:h, 0:w]
    px = np.column_stack(
        (uu.ravel() * cf.image_size_px[0] / w, vv.ravel() * cf.image_size_px[1] / h)
    )
    if cf.lens is not None:
        lens = cf.lens
        px = undistort_points(px, lens.k1, lens.centre_px, lens.radius_px)
    gx, gz = pixel_to_ground(px[:, 0], px[:, 1], cf.camera, cf.plane)
    su = px[:, 0].reshape(h, w) * pw / cf.image_size_px[0]
    sv = px[:, 1].reshape(h, w) * ph / cf.image_size_px[1]
    oob = (su < 0) | (su > pw - 1) | (sv < 0) | (sv > ph - 1)
    ui = np.clip(np.rint(su).astype(int), 0, pw - 1)
    vi = np.clip(np.rint(sv).astype(int), 0, ph - 1)
    return {
        "gx": gx.reshape(h, w).astype(np.float32),
        "gz": gz.reshape(h, w).astype(np.float32),
        "lx": level[..., 0][vi, ui].astype(np.float32),
        "lz": level[..., 2][vi, ui].astype(np.float32),
        "height": height[vi, ui].astype(np.float32),
        "horiz": horiz[vi, ui].astype(np.float32),
        "oob": oob,
        "geom_ok": ~oob & np.isfinite(height[vi, ui]),
        "geometry_signature": np.array(geometry_signature(cf)),
        "depth_scale": np.array(depth_scale),
    }


def save_geometry_cache(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write atomically; a render never sees a half-written npz."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, allow_pickle=False, **arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def floor_depth_scale(
    cf: CameraFile,
    depth: np.ndarray,
    floor_mask: np.ndarray,
) -> tuple[float, dict]:
    """Align depth to the current calibrated floor after a pose/focal correction.

    Alternating image blocks fit and check scale separately. This checks consistency
    with camera.json, not absolute accuracy: a wrong floor mask or camera can bias both.
    """
    mask = np.asarray(floor_mask, bool)
    if mask.ndim != 2:
        raise ValueError("floor_mask must be a raw-frame H x W mask")
    arrays = build_geometry_cache(cf, depth, depth_scale=1, frame_hw=mask.shape)
    down = cf.plane.height - arrays["height"]
    valid = mask & arrays["geom_ok"] & (down > 0)
    v, u = np.indices(mask.shape)
    held_out = (u // 64 + v // 64) % 2 == 1
    train, check = valid & ~held_out, valid & held_out
    if min(int(train.sum()), int(check.sum())) < 100:
        raise ValueError("floor scale needs 100 valid floor pixels in each spatial split")
    factor = float(np.median(cf.plane.height / down[train]))
    error = np.abs(cf.plane.height - factor * down[check])
    return factor, {
        "source": "depth aligned to calibrated floor mask; not independent metric truth",
        "fit_pixels": int(train.sum()),
        "check_pixels": int(check.sum()),
        "check_median_abs_height_m": float(np.median(error)),
        "check_p95_abs_height_m": float(np.percentile(error, 95)),
    }
