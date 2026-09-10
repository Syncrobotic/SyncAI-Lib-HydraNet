"""Two counters that touch are two fixtures when the object map says so.

`scene_mesh` re-labels the class map by cell connectivity, and every touching pair came
out as one welded box -- five of nine cameras on 2026-09-10. `masks_pass` now writes the
object id per pixel (`masks/objects.png`); a cell carrying an id belongs to that object,
and a cell without one keeps its connectivity label.
"""

import math

import numpy as np
from PIL import Image

from syncai_bev3d import scene_mesh
from syncai_hydranet.geometry.camera_json import Camera, CameraFile, GroundPlane

CAMERA = "cam-objects"
H, W = 240, 320
X_M, Z_M = (-3.0, 3.0), (1.0, 7.0)
LEFT = (slice(150, 200), slice(80, 150))
RIGHT = (slice(150, 200), slice(150, 220))  # shares its left edge with LEFT


def _store(tmp_path, monkeypatch, *, objects):
    root = tmp_path / "checkout"
    commission = root / "runs/commission01"
    commission.mkdir(parents=True)
    (root / "runs/site30k_qa/geometry_cache").mkdir(parents=True)
    gx = np.tile(np.linspace(*X_M, W, dtype=np.float32), (H, 1))
    gz = np.tile(np.linspace(*Z_M, H, dtype=np.float32)[:, None], (1, W))
    table = np.zeros((H, W), bool)
    table[LEFT] = True
    table[RIGHT] = True
    floor = np.zeros((H, W), bool)
    floor[20:230, 20:300] = True
    floor &= ~table
    mask_files = {"walkable": "walkable.png", "display_table": "display_table.png"}
    for name, m in (("walkable", floor), ("display_table", table)):
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(commission / f"{name}.png")
    if objects:
        ids = np.zeros((H, W), np.int32)
        ids[LEFT] = 1
        ids[RIGHT] = 2
        Image.fromarray(ids.astype(np.uint16)).save(commission / "objects.png")
        mask_files["objects"] = "objects.png"
    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=gx, gz=gz, lx=gx, lz=gz,
        height=np.where(table, 0.8, 0.0).astype(np.float32),
        geom_ok=np.ones((H, W), bool),
    )  # fmt: skip
    CameraFile(
        camera_id=CAMERA,
        image_size_px=(W, H),
        camera=Camera(fx=380.0, fy=380.0, cx=W / 2, cy=H / 2),
        plane=GroundPlane(height=2.5, pitch=math.radians(50.0)),
        mask_files=mask_files,
    ).save(commission / f"{CAMERA}.camera.json")

    def load(path, _cf):
        with np.load(path) as cache:
            return {key: cache[key] for key in cache.files}

    monkeypatch.setattr(scene_mesh, "load_geometry_cache", load, raising=False)
    return root


def _tables(root):
    _cf, _items, _h, shapes = scene_mesh.build_scene_regular(CAMERA, root)
    return sorted((max(w, d), min(w, d)) for n, w, d, h in shapes if n == "display_table")


def test_without_an_object_map_touching_counters_weld(tmp_path, monkeypatch):
    tables = _tables(_store(tmp_path, monkeypatch, objects=False))
    assert len(tables) == 1


def test_with_the_object_map_they_are_two(tmp_path, monkeypatch):
    tables = _tables(_store(tmp_path, monkeypatch, objects=True))
    assert len(tables) == 2
    # each about half the welded one along x (the lattice is 6 m over 320 px)
    for span, _short in tables:
        assert 0.9 < span < 1.7, tables


def test_the_object_grid_carries_the_ids(tmp_path, monkeypatch):
    root = _store(tmp_path, monkeypatch, objects=True)
    _cf, grids, _h, _gh, grid_obj = scene_mesh.cell_grids(CAMERA, root, with_objects=True)
    assert set(np.unique(grid_obj[4][grids[4]])) <= {0, 1, 2}
    assert (grid_obj[4] == 1).any() and (grid_obj[4] == 2).any()
    assert grid_obj[2] is None or not grid_obj[2].any()  # no wall in this store


def test_an_objects_stray_cells_do_not_widen_it(tmp_path, monkeypatch):
    """Cells wearing an object's id but lying apart from its body are the depth smear."""
    root = _store(tmp_path, monkeypatch, objects=True)
    commission = root / "runs/commission01"
    ids = np.asarray(Image.open(commission / "objects.png")).astype(np.int32)
    table = np.asarray(Image.open(commission / "display_table.png")) > 127
    # a stray patch of object 1, well away from it, in the table class
    stray = (slice(40, 60), slice(250, 290))
    ids[stray] = 1
    table[stray] = True
    Image.fromarray(ids.astype(np.uint16)).save(commission / "objects.png")
    Image.fromarray(np.where(table, 255, 0).astype(np.uint8)).save(
        commission / "display_table.png"
    )
    cache = root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz"
    with np.load(cache) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["height"] = np.where(table, 0.8, 0.0).astype(np.float32)
    np.savez(cache, **arrays)
    tables = _tables(root)
    assert len(tables) == 2, tables
    for span, short in tables:
        assert span < 1.7 and short < 1.3, (
            tables
        )  # the stray patch is 3 m away; it did not stretch anything
