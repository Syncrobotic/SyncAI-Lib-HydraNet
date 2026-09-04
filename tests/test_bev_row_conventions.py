"""Two grids over the same floor, with opposite row conventions, pinned.

`syncai_bev3d` holds two rasterisers of ground coordinates. `bev.BevGrid` puts row 0 at
the far edge, as a map is read; `floorplan.FloorRaster` puts it at the near edge, because
a contour tracer and `simplify_chain` want z ascending with the row index. Both are right
for their consumer, and a raster produced by one and indexed with the other's convention
is a vertical mirror that returns finite, plausible coordinates -- the failure shape this
project keeps naming.

They were both called `BevGrid`, in one package, until 2026-09-04. The names differ now;
these tests hold the conventions themselves, so a future edit to either has to be
deliberate rather than quiet.
"""

from __future__ import annotations

import numpy as np

from syncai_bev3d.bev import BevGrid
from syncai_bev3d.floorplan import FloorRaster


def test_bev_grid_puts_row_zero_at_the_far_edge():
    """`to_cell` documents this: row 0 is the largest z, as a map is read."""
    grid = BevGrid(x_min=-1.0, x_max=1.0, z_min=0.0, z_max=4.0, cell=1.0)
    rows, _ = grid.shape
    near = grid.to_cell(0.0, 0.5)  # closest to the camera
    far = grid.to_cell(0.0, 3.5)  # furthest away
    assert near is not None and far is not None
    assert far[0] < near[0], "far should sit at a LOWER row index than near"
    assert far[0] == 0
    assert near[0] == rows - 1


def test_floor_raster_puts_row_zero_at_the_near_edge():
    """The opposite, and deliberately: `raster` computes rows = (z - z0) / cell, so the
    row index rises with z, which is what the contour tracer downstream expects."""
    pts = np.array([[0.0, 0.5], [0.0, 3.5]])
    # pad > 0 and a cell that separates them: with pad=0 the x extent of two points
    # sharing an x is zero cells wide and the raster is empty.
    r = FloorRaster.over(pts, cell=0.5, pad=0.5)
    grid = r.raster(pts[:, 0], pts[:, 1])
    occupied = sorted(int(row) for row in np.nonzero(grid.any(axis=1))[0])
    assert len(occupied) == 2
    # the nearer point (smaller z) is the LOWER row index -- the mirror of BevGrid
    near_row, far_row = occupied[0], occupied[1]
    assert near_row < far_row


def test_the_two_conventions_are_mirrors_of_each_other():
    """Stated as one assertion so a future change to either side fails here rather than
    surfacing as a scene drawn upside down."""
    z_near, z_far = 0.5, 3.5
    grid = BevGrid(x_min=-1.0, x_max=1.0, z_min=0.0, z_max=4.0, cell=1.0)
    bev_near = grid.to_cell(0.0, z_near)[0]
    bev_far = grid.to_cell(0.0, z_far)[0]

    pts = np.array([[0.0, z_near], [0.0, z_far]])
    r = FloorRaster.over(pts, cell=0.5, pad=0.5)
    fr = r.raster(pts[:, 0], pts[:, 1])
    rows = sorted(int(x) for x in np.nonzero(fr.any(axis=1))[0])
    floor_near, floor_far = rows[0], rows[1]

    assert (bev_near > bev_far) and (floor_near < floor_far), (
        "the two grids no longer disagree about which end row 0 is; one of them changed"
    )
