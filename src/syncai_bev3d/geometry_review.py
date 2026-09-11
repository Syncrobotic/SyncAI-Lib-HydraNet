"""Raw-frame metre grids and checks that cached geometry still matches its camera."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points

from .geometry_cache import geometry_signature
from .ground_control import floor_pixels


def cache_ground_error(cf: CameraFile, arrays) -> float:
    """Maximum sampled floor discrepancy in metres; infinity for incompatible caches.

    Legacy caches have no provenance. Their ground projection can still be compared to
    today's calibration. This detects geometry drift, not a changed depth teacher/plate.
    """
    gx, gz = arrays["gx"], arrays["gz"]
    if gx.ndim != 2 or gx.shape != gz.shape or min(gx.shape) < 2:
        return float("inf")
    h, w = gx.shape
    v, u = np.meshgrid(
        np.linspace(0, h - 1, 17).astype(int),
        np.linspace(0, w - 1, 17).astype(int),
        indexing="ij",
    )
    px = np.column_stack(
        (u.ravel() * cf.image_size_px[0] / w, v.ravel() * cf.image_size_px[1] / h)
    )
    if cf.lens is not None:
        lens = cf.lens
        px = undistort_points(px, lens.k1, lens.centre_px, lens.radius_px)
    x, z = pixel_to_ground(px[:, 0], px[:, 1], cf.camera, cf.plane)
    expected = np.column_stack((x, z))
    stored = np.column_stack((gx[v, u].ravel(), gz[v, u].ravel()))
    finite = np.isfinite(expected).all(axis=1)
    if not np.array_equal(finite, np.isfinite(stored).all(axis=1)) or not finite.any():
        return float("inf")
    return float(np.linalg.norm(expected[finite] - stored[finite], axis=1).max())


def load_geometry_cache(path: str | Path, cf: CameraFile) -> dict[str, np.ndarray]:
    """Refuse stale geometry before drawing metre-space fixtures from it."""
    with np.load(path, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    if "geometry_signature" in arrays and str(
        arrays["geometry_signature"]
    ) != geometry_signature(cf):
        raise ValueError(
            f"{cf.camera_id}: stale geometry cache signature; rebuild_geometry.py required"
        )
    error = cache_ground_error(cf, arrays)
    if error > 0.002:
        raise ValueError(
            f"{cf.camera_id}: stale geometry cache {path}; sampled floor drift {error:.4g} m. "
            "Rebuild with tools/commissioning/rebuild_geometry.py and the matching depth "
            "scale before rendering the scene."
        )
    return arrays


def render_metre_grid(
    plate: Image.Image,
    cf: CameraFile,
    *,
    spacing_m: float = 1.0,
    bounds_m: tuple[float, float, float, float] = (-10.0, 10.0, 0.0, 15.0),
) -> Image.Image:
    """Draw sampled metric lines through the lens onto the raw calibrated plate.

    Sampling matters: a straight line in the pinhole image is a curve in the raw image.
    The bounded drawing extent is displayed and is not a claim about the room boundary.
    """
    if plate.size != cf.image_size_px:
        raise ValueError("plate size must match camera.json; resize the raw plate explicitly")
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("spacing_m must be positive and finite")
    x0, x1, z0, z1 = bounds_m
    if not np.isfinite(bounds_m).all() or x1 <= x0 or z1 <= z0:
        raise ValueError("bounds_m must be finite, ordered x_min, x_max, z_min, z_max")
    if (x1 - x0 + z1 - z0) / spacing_m > 500:
        raise ValueError("grid would exceed 500 lines; increase spacing or reduce bounds")
    cf.validate()
    img = plate.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for axis, lo, hi, other_lo, other_hi in (
        (0, x0, x1, z0, z1),
        (1, z0, z1, x0, x1),
    ):
        for value in np.arange(np.ceil(lo / spacing_m), np.floor(hi / spacing_m) + 1):
            fixed = np.full(501, value * spacing_m)
            varying = np.linspace(other_lo, other_hi, len(fixed))
            points = np.column_stack((fixed, varying) if axis == 0 else (varying, fixed))
            px = floor_pixels(cf, points)
            visible = np.isfinite(px).all(axis=1)
            visible &= (px[:, 0] >= 0) & (px[:, 0] < width)
            visible &= (px[:, 1] >= 0) & (px[:, 1] < height)
            color = (255, 185, 55) if value == 0 else (40, 230, 180)
            for i in np.flatnonzero(visible[:-1] & visible[1:]):
                i = int(i)
                draw.line([tuple(px[i]), tuple(px[i + 1])], fill=color, width=1)
    draw.rectangle((0, 0, width, 34), fill=(20, 20, 20))
    draw.text(
        (6, 3), f"{cf.camera_id}: {spacing_m:g} m grid; raw lens projection", fill="white"
    )
    draw.text(
        (6, 18),
        f"Drawing extent {bounds_m} m; geometry review, not measured accuracy",
        fill="white",
    )
    return img
