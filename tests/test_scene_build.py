"""The commissioning scene build, on a store made of known parts.

`syncai_bev3d.scene_mesh` is what every render in this project draws -- the demo video's 3D
panel, the heads figure, `scene_overlay`'s wireframe, the social card -- and until it
moved out of `tools/` on 2026-08-29 **not one line of its scene build had ever been
executed by a test**. It was 13% covered, and the covered part was the caption.

The store here is synthetic and its parts are known, which is the only way to assert the
thing that actually matters. The user's standard for this scene is not accuracy, it is
**relative correctness** -- "I can accept imprecision, but the relative relationships must
be correct" -- and a relation needs two objects whose truth you hold. So the fixtures are
placed a known distance apart, the wall has floor on one side only, and the assertions are
about *pairs*: no two fixtures share floor, the wall is a wall and not a counter, and
every built box is inside its class's interval.

The geometry cache is a per-pixel `lx/lz/height/geom_ok` map, exactly as
`runs/site30k_qa/geometry_cache/<camera>.npz` is. It is written at 480x360 rather than
1920x1080 for speed, and that resolution is load-bearing in one direction: `cell_grids`
keeps a 6 cm floor cell only where **three** pixels land in it, so a plate too small for
the metric extent it claims produces an empty grid and a scene with no furniture in it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from syncai_bev3d import scene_mesh
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane

CAMERA = "Synthetic-cam01"
PLATE_W, PLATE_H = 480, 360
X_M = (-4.0, 4.0)  # metres across the plate, left to right
Z_M = (9.0, 1.0)  # metres away, top row to bottom row

# (name, row slice, column slice, height in metres). Chosen so that every box lands
# inside `PLAUSIBLE_M` and no two of them touch: the point of the fixture is that a
# failure is a real one and not the store's own fault.
TABLE = ("display_table", slice(180, 240), slice(120, 190), 0.78)
SHELF = ("display_shelf", slice(120, 165), slice(290, 420), 1.90)
WALL_ROWS, WALL_COLS = slice(20, 44), slice(40, 440)
FLOOR = (slice(80, 340), slice(40, 440))


def _png(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def a_store(tmp_path, fixtures=(TABLE, SHELF), wall=True):
    """A commissioned camera and its geometry cache, on disk, as `cell_grids` reads them."""
    root = tmp_path / "checkout"
    commission = root / "runs/commission01"
    commission.mkdir(parents=True)
    (root / "runs/site30k_qa/geometry_cache").mkdir(parents=True)

    lx = np.tile(np.linspace(*X_M, PLATE_W, dtype=np.float32), (PLATE_H, 1))
    lz = np.tile(np.linspace(*Z_M, PLATE_H, dtype=np.float32)[:, None], (1, PLATE_W))
    height = np.zeros((PLATE_H, PLATE_W), np.float32)

    mask_files = {}
    floor = np.zeros((PLATE_H, PLATE_W), bool)
    floor[FLOOR] = True
    _png(commission / "walkable.png", floor)
    mask_files["walkable"] = "walkable.png"

    for name, rows, cols, h in fixtures:
        m = np.zeros((PLATE_H, PLATE_W), bool)
        m[rows, cols] = True
        height[m] = h
        _png(commission / f"{name}.png", m)
        mask_files[name] = f"{name}.png"

    if wall:
        m = np.zeros((PLATE_H, PLATE_W), bool)
        m[WALL_ROWS, WALL_COLS] = True
        height[m] = 2.4
        _png(commission / "wall.png", m)
        mask_files["wall"] = "wall.png"

    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=lx,
        gz=lz,
        lx=lx,
        lz=lz,
        height=height,
        geom_ok=np.ones((PLATE_H, PLATE_W), bool),
    )
    CameraFile(
        camera_id=CAMERA,
        image_size_px=(PLATE_W, PLATE_H),
        camera=Camera(fx=380.0, fy=380.0, cx=PLATE_W / 2, cy=PLATE_H / 2),
        plane=GroundPlane(height=2.5, pitch=math.radians(50.0)),
        mask_files=mask_files,
    ).save(commission / f"{CAMERA}.camera.json")
    return root


def _built(root):
    """`(name, u0, u1, v0, v1)` per meshed fixture, keyed the way the assertions read."""
    _cf, items, _heights, shapes = scene_mesh.build_scene_regular(CAMERA, root)
    return items, shapes


# --------------------------------------------------------------------- the grid itself


def test_the_cell_grid_holds_the_classes_the_masks_named(tmp_path):
    root = a_store(tmp_path)
    _cf, grids, heights, _grid_h = scene_mesh.cell_grids(CAMERA, root)
    assert grids[1].any(), "no walkable floor: the plate is too small for its metric extent"
    for cid, name in scene_mesh.CLASS_NAMES.items():
        if name == "column":
            continue
        assert grids[cid].any(), f"{name} mask produced no cells"
    # The table's measured height comes back from the cache, not from the class default.
    assert heights[4] == pytest.approx(0.78, abs=0.05)


def test_a_class_with_no_mask_is_absent_rather_than_defaulted(tmp_path):
    """`column` has no mask in this store. The grid for it must be empty, not full."""
    root = a_store(tmp_path)
    _cf, grids, heights, _grid_h = scene_mesh.cell_grids(CAMERA, root)
    assert not grids[3].any()
    assert 3 not in heights


# ------------------------------------------------------------------- the relations


def test_no_two_fixtures_occupy_the_same_floor(tmp_path):
    """The regularise pass, stated as the relation it exists for.

    Until 2026-08-28 each component was meshed the moment it was fitted, so two boxes
    drawn through each other was a normal output and nothing in the scene had ever
    compared two fixtures with each other.
    """
    root = a_store(tmp_path)
    items, _shapes = _built(root)
    boxes = [
        scene_mesh.footprint_of(m)
        for m, name, _a, _s in items
        if name in ("display_table", "display_shelf")
    ]
    assert len(boxes) >= 2
    for i, (x0, x1, z0, z1, _t) in enumerate(boxes):
        for x2, x3, z2, z3, _t2 in boxes[i + 1 :]:
            overlap = max(0.0, min(x1, x3) - max(x0, x2)) * max(0.0, min(z1, z3) - max(z0, z2))
            assert overlap < 0.01, f"two fixtures share {overlap:.2f} m2 of floor"


def test_a_run_with_floor_on_one_side_is_kept_as_a_wall(tmp_path):
    root = a_store(tmp_path)
    _items, shapes = _built(root)
    assert any(name == "wall" for name, *_ in shapes), (
        "the far run has floor on one side only and is a room boundary; it was dropped"
    )


def test_a_run_with_shoppers_on_both_sides_is_not_a_wall(tmp_path):
    """`floor_both_sides`, which is what separates a wall from a counter on a white store.

    Half the runs fitted across the real fleet fail this, and they are the counters both
    teachers read as `wall`. The floor here is widened to reach behind the run; nothing
    about the run's own shape, size or height changes.
    """
    root = a_store(tmp_path)
    both = np.zeros((PLATE_H, PLATE_W), bool)
    both[10:340, 40:440] = True  # floor now reaches past the far run
    _png(root / "runs/commission01/walkable.png", both)
    _items, shapes = _built(root)
    assert not any(name == "wall" for name, *_ in shapes)


def test_every_built_fixture_is_inside_its_class_interval(tmp_path):
    root = a_store(tmp_path)
    _items, shapes = _built(root)
    assert scene_mesh.implausible(shapes) == []


def test_a_welded_table_is_reported_rather_than_drawn_silently(tmp_path):
    """The same store with the table stretched to 3.6 m -- a mask bridge, not furniture."""
    welded = ("display_table", slice(180, 240), slice(120, 400), 0.78)
    root = a_store(tmp_path, fixtures=(welded,))
    _items, shapes = _built(root)
    said = scene_mesh.implausible(shapes)
    assert any("display_table" in line and "span" in line for line in said), said
