"""A tall fixture stands where its mask meets the floor, not where its face was lowered to.

`cell_grids` builds a wall's, column's or cabinet's footprint from every mask pixel
lowered to the floor by its measured height. Where that height is wrong -- DA-V2 on a
white wall -- the lowered pixel lands metres into the aisle, and on 2026-09-10 four
commissioned cameras each showed a fixture standing on open floor that the plate did not
have. The gate keeps a lowered pixel only if it lands beside the class's own contact line.
"""

import math

import numpy as np
from PIL import Image

from syncai_bev3d import scene_mesh
from syncai_bev3d.scene_mesh import CELL, CONTACT_BAND_M, contact_cells
from syncai_hydranet.geometry.camera_json import Camera, CameraFile, GroundPlane

CAMERA = "cam-contact"
H, W = 240, 320
X_M, Z_M = (-3.0, 3.0), (1.0, 7.0)


def _lattice():
    gx = np.tile(np.linspace(*X_M, W, dtype=np.float32), (H, 1))
    gz = np.tile(np.linspace(*Z_M, H, dtype=np.float32)[:, None], (1, W))
    return gx, gz


# ------------------------------------------------------------------------ contact_cells


def test_contact_cells_takes_the_lowest_pixel_of_each_column():
    gx, gz = _lattice()
    mask = np.zeros((H, W), bool)
    mask[50:90, 100:140] = True
    pts = contact_cells(mask, gx, gz)
    assert len(pts) == 40  # one per column of the component
    assert np.allclose(pts[:, 1], gz[89, 0])  # the bottom row, not the top or the mean
    assert np.allclose(np.sort(pts[:, 0]), gx[0, 100:140])


def test_contact_cells_drops_speckle_but_not_a_second_fixture():
    gx, gz = _lattice()
    mask = np.zeros((H, W), bool)
    mask[50:90, 100:140] = True  # a fixture
    mask[150:190, 200:240] = True  # another, further down the frame
    mask[20:24, 10:14] = True  # 16 px of noise
    pts = contact_cells(mask, gx, gz, min_component_px=200)
    assert len(pts) == 80
    assert not np.isclose(pts[:, 1], gz[23, 0]).any()


def test_contact_cells_skips_a_pixel_the_geometry_could_not_place():
    gx, gz = _lattice()
    gx = gx.copy()
    gx[89, 120] = np.nan  # the contact pixel of one column has no ground hit
    mask = np.zeros((H, W), bool)
    mask[50:90, 100:140] = True
    pts = contact_cells(mask, gx, gz)
    assert len(pts) == 39 and np.isfinite(pts).all()


def test_contact_cells_of_nothing_is_nothing():
    gx, gz = _lattice()
    assert contact_cells(np.zeros((H, W), bool), gx, gz).shape == (0, 2)


# ------------------------------------------------------------------- the gate in place


def _store(tmp_path, monkeypatch, *, ghost_offset_m):
    """A wall whose upper half was lowered `ghost_offset_m` into the room.

    The lattice is orthographic: `gx/gz` are the pixel's own floor position. The wall's
    lower rows keep `lx/lz == gx/gz` (a correct lowering moves a contact pixel nowhere);
    its upper rows are lowered to a line `ghost_offset_m` further down the room, which is
    what a collapsed DA-V2 height does to a white wall.
    """
    root = tmp_path / "checkout"
    commission = root / "runs/commission01"
    commission.mkdir(parents=True)
    (root / "runs/site30k_qa/geometry_cache").mkdir(parents=True)
    gx, gz = _lattice()
    lx, lz = gx.copy(), gz.copy()
    wall = np.zeros((H, W), bool)
    wall[20:60, 40:280] = True
    lz[20:40, :] = gz[59, 0] + ghost_offset_m  # the upper half lands past the contact line
    floor = np.zeros((H, W), bool)
    floor[70:230, 40:280] = True
    for name, m in (("wall", wall), ("walkable", floor)):
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(commission / f"{name}.png")
    height = np.where(wall, 2.4, 0.0).astype(np.float32)
    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=gx, gz=gz, lx=lx, lz=lz, height=height, geom_ok=np.ones((H, W), bool),
    )  # fmt: skip
    CameraFile(
        camera_id=CAMERA,
        image_size_px=(W, H),
        camera=Camera(fx=380.0, fy=380.0, cx=W / 2, cy=H / 2),
        plane=GroundPlane(height=2.5, pitch=math.radians(50.0)),
        mask_files={"wall": "wall.png", "walkable": "walkable.png"},
    ).save(commission / f"{CAMERA}.camera.json")

    def load(path, _cf):
        with np.load(path) as cache:
            return {key: cache[key] for key in cache.files}

    # `raising=False`: at HEAD `cell_grids` reads the npz directly and the name does not
    # exist; on the branch that adds `load_geometry_cache` it validates the cache against
    # the camera, which a synthetic lattice cannot pass. Either way the file is the cache.
    monkeypatch.setattr(scene_mesh, "load_geometry_cache", load, raising=False)
    return root, gz[59, 0]


def _wall_rows_m(root):
    _cf, grids, _h, _gh = scene_mesh.cell_grids(CAMERA, root)
    rows = np.nonzero(grids[scene_mesh.WALL_CID].any(axis=1))[0]
    return rows * CELL


def test_a_face_lowered_into_the_aisle_leaves_no_footprint(tmp_path, monkeypatch):
    root, contact_z = _store(tmp_path, monkeypatch, ghost_offset_m=1.0)
    rows_m = _wall_rows_m(root)
    assert len(rows_m), "the wall's contact rows were gated away with the ghost"
    assert rows_m.max() < contact_z + CONTACT_BAND_M + CELL
    assert not np.any(np.abs(rows_m - (contact_z + 1.0)) < 2 * CELL), "the ghost was built"


def test_a_face_lowered_beside_its_contact_line_is_kept(tmp_path, monkeypatch):
    root, contact_z = _store(tmp_path, monkeypatch, ghost_offset_m=CONTACT_BAND_M / 2)
    rows_m = _wall_rows_m(root)
    assert np.any(np.abs(rows_m - (contact_z + CONTACT_BAND_M / 2)) < 2 * CELL)


def test_a_table_is_not_gated():
    """A table is mostly top, and its top lowered by its own height IS the footprint."""
    assert "display_table" not in scene_mesh.CONTACT_GATED
    assert {"wall", "column", "display_shelf"} <= scene_mesh.CONTACT_GATED


def test_a_pillar_whose_face_was_lowered_away_still_stands_at_its_foot(tmp_path, monkeypatch):
    """A column's lowered pixels all land far from it (a white pillar, DA-V2's height
    wrong on every pixel); its foot -- the mask's contact with the floor -- says where it
    is, and that is enough to build it."""
    root = tmp_path / "checkout"
    commission = root / "runs/commission01"
    commission.mkdir(parents=True)
    (root / "runs/site30k_qa/geometry_cache").mkdir(parents=True)
    gx, gz = _lattice()
    lx, lz = gx.copy(), gz.copy()
    column = np.zeros((H, W), bool)
    column[60:120, 150:182] = True  # a 32-px-wide (0.6 m) pillar, 60 px tall
    lz[column] = gz[119, 0] + 1.5  # every lowered pixel lands 1.5 m past the foot
    floor = np.zeros((H, W), bool)
    floor[125:230, 40:280] = True
    for name, m in (("column", column), ("walkable", floor)):
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(commission / f"{name}.png")
    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=gx, gz=gz, lx=lx, lz=lz,
        height=np.where(column, 2.4, 0.0).astype(np.float32),
        geom_ok=np.ones((H, W), bool),
    )  # fmt: skip
    CameraFile(
        camera_id=CAMERA,
        image_size_px=(W, H),
        camera=Camera(fx=380.0, fy=380.0, cx=W / 2, cy=H / 2),
        plane=GroundPlane(height=2.5, pitch=math.radians(50.0)),
        mask_files={"column": "column.png", "walkable": "walkable.png"},
    ).save(commission / f"{CAMERA}.camera.json")

    def load(path, _cf):
        with np.load(path) as cache:
            return {key: cache[key] for key in cache.files}

    monkeypatch.setattr(scene_mesh, "load_geometry_cache", load, raising=False)
    _cf, items, _h, shapes = scene_mesh.build_scene_regular(CAMERA, root)
    columns = [(m, k) for m, k, _a, _s in items if k == "column"]
    assert len(columns) == 1, shapes
    v = columns[0][0][0]
    foot_z = gz[119, 0]
    assert abs((v[:, 2].min() + v[:, 2].max()) / 2 - foot_z) < 0.5, (
        v[:, 2].min(),
        v[:, 2].max(),
        foot_z,
    )
