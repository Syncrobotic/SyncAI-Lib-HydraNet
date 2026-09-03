"""The heatmap grid maths, pinned where a wrong picture would just look plausible.

A heatmap has no error state: every bug renders as a slightly different pretty floor.
So the properties that make it a measurement are pinned here -- the two modes actually
answer their two different questions, the scale top is a percentile not a maximum, and
the walkable clip really drops cells.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from _cameras import FULL_RES_CAM, FULL_RES_PLANE, FULL_RES_SIZE
from syncai_bev3d.heatmap import LEVELS, accumulate, heat_colour, normalise, tile_specs
from syncai_hydranet.geometry.camera_json import CameraFile, Zone

FPS = 5.0


def pos(x, z, tid, frame=0):
    return {"x_m": x, "z_m": z, "track_id": tid, "frame": frame}


def test_dwell_counts_seconds_and_traffic_counts_tracks():
    """The whole reason two modes exist: five minutes of standing is one traversal."""
    lingerer = [pos(0.0, 0.0, 1, f) for f in range(50)]  # 10 s in one cell
    passers = [pos(2.0, 0.0, tid) for tid in range(2, 7)]  # 5 tracks, 1 frame each
    grid_d, x0, z0 = accumulate(lingerer + passers, FPS, "dwell")
    grid_t, _, _ = accumulate(lingerer + passers, FPS, "traffic")

    def cell(grid, x, z):
        return grid[int((x - x0) / 0.25), int((z - z0) / 0.25)]

    assert cell(grid_d, 0.0, 0.0) == pytest.approx(10.0)  # seconds
    assert cell(grid_d, 2.0, 0.0) == pytest.approx(5 / FPS)
    assert cell(grid_t, 0.0, 0.0) == 1.0  # one track, however long it stood
    assert cell(grid_t, 2.0, 0.0) == 5.0


def test_a_track_is_counted_once_per_cell_not_once_per_frame():
    revisits = [pos(0.0, 0.0, 1, f) for f in range(30)]
    grid_t, _, _ = accumulate(revisits, FPS, "traffic")
    assert grid_t.max() == 1.0


def test_the_scale_top_is_a_percentile_so_one_queue_cannot_eat_the_ramp():
    grid = np.zeros((10, 10))
    grid[::2, ::2] = 1.0  # a busy floor
    grid[0, 0] = 500.0  # one queue outlier
    norm, top = normalise(grid, sigma_cells=0.0, clip_pct=99.0)
    assert top < 500.0, "p99 must sit below the outlier"
    assert norm[0, 0] == 1.0  # the outlier clips to the top rather than stretching it


def test_tiles_are_clipped_to_the_walkable_polygon():
    cf = CameraFile(
        camera_id="Test-cam",
        image_size_px=FULL_RES_SIZE,
        camera=FULL_RES_CAM,
        plane=FULL_RES_PLANE,
        zones=(
            Zone("floor", "walkable", ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))),
        ),
    )
    norm = np.ones((2, 1))
    # one cell centre inside the walkable square, one far outside it
    tiles = tile_specs(norm, x0=-0.25, z0=-0.125, cell_m=0.25, cf=cf)
    inside = tile_specs(np.ones((1, 1)), x0=5.0, z0=5.0, cell_m=0.25, cf=cf)
    assert len(tiles) == 2 and len(inside) == 0


def test_an_unknown_mode_and_an_empty_window_are_refused():
    with pytest.raises(ValueError, match="mode"):
        accumulate([pos(0, 0, 1)], FPS, "vibes")
    with pytest.raises(ValueError, match="no positions"):
        accumulate([], FPS, "dwell")


def test_the_ramp_is_one_hue_light_to_bright():
    """Sequential colour discipline: R>G>B at every step (amber), monotone lightness."""
    steps = [heat_colour((k + 1) / LEVELS) for k in range(LEVELS)]
    assert all(r > g > b for r, g, b in steps)
    lum = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in steps]
    assert all(a < b for a, b in itertools.pairwise(lum)), (
        "intensity must brighten monotonically"
    )
