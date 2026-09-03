"""Floor heatmaps over tracked positions -- the grid math, without the CLI.

The construction is the retail-analytics standard: floor positions accumulate into a
metre grid, a Gaussian widens each sample to about the position noise, and the top of
the colour scale is a PERCENTILE rather than the maximum -- one queue otherwise eats
the whole ramp and every aisle reads empty. Two modes, because they answer different
questions and vendors that blur them mislead their users:

* ``dwell``   -- occupancy-seconds per cell: a queue and a served counter light up.
* ``traffic`` -- distinct tracks per cell: walkways light up, lingering does not add.

Colour is a SEQUENTIAL job (one magnitude), so the ramp is a single hue -- amber --
with lightness and alpha carrying intensity. Blue and green are spoken for by the
staff/customer figures and never appear here.

`tools/commissioning/heatmap3d.py` is the CLI; this module holds everything a test
can pin without rendering.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .figures import point_in_walkable

#: Cell edge. ~ the position noise of a smoothed foot point; finer cells sample noise,
#: coarser ones blur a till into its aisle.
DEFAULT_CELL_M = 0.25
#: Gaussian bandwidth in cells (0.375 m at the default cell) -- the industry-usual
#: 0.3-0.5 m that turns point samples into a field.
DEFAULT_SIGMA_CELLS = 1.5
#: The scale top. p99, not max: the busiest cell is an outlier by construction.
DEFAULT_CLIP_PCT = 99.0
#: Ramp resolution. Ten steps read as continuous at tile scale.
LEVELS = 10

MODES = ("dwell", "traffic")


def heat_colour(t: float) -> tuple[int, int, int]:
    """One amber, light to bright, for t in (0, 1]. Sequential = single hue."""
    return (int(120 + 135 * t), int(70 + 110 * t), int(20 + 30 * t))


def accumulate(
    positions, fps: float, mode: str = "dwell", cell_m: float = DEFAULT_CELL_M
) -> tuple[np.ndarray, float, float]:
    """Positions -> (grid, x0, z0). Grid units: seconds (dwell) or tracks (traffic).

    ``positions`` is `demo_tracks.json`'s shape: dicts with ``x_m``, ``z_m`` and
    ``track_id``. Traffic counts a track at most once per cell, which is the whole
    difference between the modes: five minutes of standing is one traversal.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not positions:
        raise ValueError(
            "no positions: a heatmap of nothing would render as an empty "
            "floor and read as a measurement"
        )
    xs = np.array([p["x_m"] for p in positions], float)
    zs = np.array([p["z_m"] for p in positions], float)
    x0, z0 = float(xs.min() - 0.5), float(zs.min() - 0.5)
    nx = int(np.ceil((xs.max() + 0.5 - x0) / cell_m))
    nz = int(np.ceil((zs.max() + 0.5 - z0) / cell_m))
    grid = np.zeros((nx, nz))
    ix = np.clip(((xs - x0) / cell_m).astype(int), 0, nx - 1)
    iz = np.clip(((zs - z0) / cell_m).astype(int), 0, nz - 1)
    if mode == "dwell":
        np.add.at(grid, (ix, iz), 1.0 / fps)
    else:
        tids = np.array([p["track_id"] for p in positions])
        seen = {(int(t), int(i), int(j)) for t, i, j in zip(tids, ix, iz, strict=True)}
        for _t, i, j in seen:
            grid[i, j] += 1.0
    return grid, x0, z0


def normalise(
    grid: np.ndarray,
    sigma_cells: float = DEFAULT_SIGMA_CELLS,
    clip_pct: float = DEFAULT_CLIP_PCT,
) -> tuple[np.ndarray, float]:
    """Smooth and scale to [0, 1]; returns (norm, top) with `top` in the grid's units.

    `top` is what the legend must print -- a heatmap without its scale is a picture
    of an opinion.
    """
    smoothed = gaussian_filter(grid, sigma=sigma_cells)
    occupied = smoothed[smoothed > 0]
    top = float(np.percentile(occupied, clip_pct)) if len(occupied) else 1.0
    return np.clip(smoothed / max(top, 1e-9), 0.0, 1.0), top


def tile_specs(
    norm: np.ndarray,
    x0: float,
    z0: float,
    cell_m: float,
    cf=None,
    min_t: float = 0.06,
) -> list[tuple[float, float, int, int]]:
    """(x_m, z_m, level, alpha) per visible cell, clipped to the walkable polygon.

    Clipping is what stops the field bleeding under fixtures: a foot point projected
    behind a counter lands on the counter's footprint, and painting heat there claims
    shoppers stand inside furniture. With no ``cf`` every cell passes (the caller has
    no polygon to clip against, and saying so beats silently not clipping).
    """
    out = []
    nx, nz = norm.shape
    for i in range(nx):
        for j in range(nz):
            t = float(norm[i, j])
            if t < min_t:
                continue
            x_m = x0 + (i + 0.5) * cell_m
            z_m = z0 + (j + 0.5) * cell_m
            if cf is not None and not point_in_walkable(cf, x_m, z_m):
                continue
            level = min(LEVELS - 1, int(t * LEVELS))
            out.append((x_m, z_m, level, int(90 + 140 * t)))
    return out
