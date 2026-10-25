"""Acceptance checks on the depth -> cloud stage, against an analytic scene.

`SyntheticDepth` blurs depth across the cube's outline the way a network does, so a
cloud built without edge filtering has flying points; these tests prove the filter
removes them and that walls come out flat. The model itself is not exercised here.

    uv run pytest test_pointcloud.py
"""

import math

import numpy as np
import pytest
import torch
from server import HEADER, MAGIC, CloudBuilder, CloudParams, SyntheticDepth, pack

BOX_X = 0.2
DEV = torch.device("cpu")


def surface_distance(p: np.ndarray) -> np.ndarray:
    """Distance from each viewer-frame point (Y up, -Z forward) to the nearest surface."""
    x, y, z = p[:, 0], -p[:, 1], -p[:, 2]  # back to the scene's OpenCV frame
    d = np.stack(
        [
            np.abs(x + 2.0),
            np.abs(x - 2.5),
            np.abs(y - 1.2),
            np.abs(y + 1.5),
            np.abs(z - 5.0),
        ]
    ).min(axis=0)
    on_face = (np.abs(x - BOX_X) < 0.4) & (np.abs(y - 0.8) < 0.4)
    return np.where(on_face, np.minimum(d, np.abs(z - 2.5)), d)


def build(params: CloudParams, frames: int = 10, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    src = SyntheticDepth(DEV, box_x=BOX_X)
    builder = CloudBuilder(params)
    rgb = np.zeros((291, 518, 3), np.uint8)
    pts = None
    for _ in range(frames):
        pts = builder(*src(rgb))
    assert pts is not None
    return pts.numpy()


def test_unfiltered_cloud_has_flying_points():
    """Sanity: without the edge filter the scene really does produce flying points,
    so the next test passing means something."""
    pts = build(CloudParams(edge_thresh=math.inf, edge_dilate=0))
    assert (surface_distance(pts) > 0.15).sum() > 20


def test_threshold_015_lets_smeared_edges_through():
    """Why the default is 0.05: at 0.15 relative change per pixel, a depth step smeared
    over 5 px (what the synthetic scene and a network both do) survives as flying points."""
    pts = build(CloudParams(edge_thresh=0.15))
    assert (surface_distance(pts) > 0.15).sum() > 20


def test_voxel_no_flying_points():
    pts = build(CloudParams())
    assert len(pts) > 2000
    assert (surface_distance(pts) > 0.15).sum() == 0


def back_wall(p: np.ndarray) -> np.ndarray:
    """Points of the back wall's interior, away from the corners and the cube's shadow."""
    x, y, z = p[:, 0], -p[:, 1], -p[:, 2]
    return p[(np.abs(z - 5.0) < 0.5) & (np.abs(x - 0.25) < 1.75) & (np.abs(y + 0.2) < 1.0)]


def test_voxel_wall_is_flat():
    pts = build(CloudParams())
    back = back_wall(pts)
    assert len(back) > 300
    # 1 % per-pixel noise at 5 m is 5 cm; EMA + voxel averaging must bring it well under.
    assert np.std(-back[:, 2] - 5.0) < 0.02


def test_ema_reduces_noise():
    one = build(CloudParams(), frames=1)
    many = build(CloudParams(), frames=15)

    def wall_std(p):
        return np.std(back_wall(p)[:, 2])

    assert wall_std(many) < 0.7 * wall_std(one)


def test_scanlines_are_rings_without_flying_points():
    p = CloudParams(scanlines=True)
    pts = build(p)
    assert len(pts) > 1000
    assert (surface_distance(pts) > 0.15).sum() == 0
    # Every point lies on one of the 32 elevation cones.
    elev = np.degrees(np.arctan2(pts[:, 1], np.hypot(pts[:, 0], pts[:, 2])))
    lines = np.linspace(p.elev_min, p.elev_max, p.n_lines)
    assert np.abs(elev[:, None] - lines[None]).min(axis=1).max() < 0.05
    # The camera sees only part of the -25..+15 fan, but more than a handful of lines.
    used = np.unique(np.abs(elev[:, None] - lines[None]).argmin(axis=1))
    assert len(used) >= 15
    # Azimuth sampling is the 0.25 deg grid.
    az = np.degrees(np.arctan2(pts[:, 0], -pts[:, 2]))
    frac = (az - 0.125) / 0.25
    assert np.abs(frac - np.round(frac)).max() < 0.01


def test_invalid_pixels_are_dropped():
    builder = CloudBuilder(CloudParams(edge_dilate=0))
    depth = torch.full((40, 60), 2.0)
    depth[:, :30] = math.inf  # MoGe marks sky / no-depth as inf
    pts = builder(depth, torch.tensor([1.0, 1.5, 0.5, 0.5])).numpy()
    assert len(pts) > 0
    assert np.isfinite(pts).all()
    assert (pts[:, 0] > -0.05).all()  # nothing from the invalid left half


@pytest.mark.parametrize("n", [0, 1, 1234])
def test_pack_roundtrip(n):
    pts = np.random.default_rng(0).uniform(-5, 5, (n, 3)).astype(np.float32)
    buf = pack(pts, seq=7, mode=1)
    magic, seq, count, mode, _ = HEADER.unpack_from(buf)
    assert (magic, seq, count, mode) == (MAGIC, 7, n, 1)
    planar = np.frombuffer(buf, np.float16, offset=HEADER.size).reshape(3, n)
    np.testing.assert_allclose(planar.T, pts, atol=5e-3)
