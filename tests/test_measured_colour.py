"""A built item is painted the colour the plate shows under it, and the store axis is
fitted to the ungated evidence.

Both landed on 2026-09-10 after the user's verdict on the palette-coloured scene: the
model does not look like the real objects. The form is still convention; the colour is
now measured -- median RGB of the plate under the pixels each item was built from.
"""

import math

import numpy as np
from PIL import Image

from syncai_bev3d import scene_mesh
from syncai_bev3d.scene_mesh import PALETTE, Painted, colour_of
from syncai_hydranet.geometry.camera_json import Camera, CameraFile, GroundPlane

CAMERA = "cam-paint"
H, W = 240, 320
X_M, Z_M = (-3.0, 3.0), (1.0, 7.0)
TABLE_RGB, WALL_RGB, FLOOR_RGB = (200, 40, 40), (30, 190, 30), (60, 60, 200)
TABLE = (slice(180, 230), slice(120, 200))
TABLE2 = (slice(180, 230), slice(230, 270))  # a second table, in its own colour
TABLE2_RGB = (240, 220, 40)
WALL = (slice(20, 60), slice(40, 280))
FLOOR = (slice(70, 230), slice(40, 280))


def test_a_painted_key_is_still_the_class_name():
    k = Painted("wall", (1, 2, 3))
    assert k == "wall" and k in ("wall", "column") and isinstance(k, str)
    assert colour_of(k) == (1, 2, 3)
    assert colour_of("wall") == PALETTE["wall"]


def _store(tmp_path, monkeypatch, *, plate=True, ghost_offset_m=0.0):
    root = tmp_path / "checkout"
    commission = root / "runs/commission01"
    commission.mkdir(parents=True)
    (root / "runs/site30k_qa/geometry_cache").mkdir(parents=True)
    gx = np.tile(np.linspace(*X_M, W, dtype=np.float32), (H, 1))
    gz = np.tile(np.linspace(*Z_M, H, dtype=np.float32)[:, None], (1, W))
    lx, lz = gx.copy(), gz.copy()
    masks = {}
    for name, sl in (("wall", WALL), ("display_table", TABLE), ("walkable", FLOOR)):
        m = np.zeros((H, W), bool)
        m[sl] = True
        if name == "display_table":
            m[TABLE2] = True
        masks[name] = m
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(commission / f"{name}.png")
    if ghost_offset_m:
        lz[20:40, :] = gz[59, 0] + ghost_offset_m  # the wall's upper half lands in the room
    height = np.where(masks["wall"], 2.4, np.where(masks["display_table"], 0.8, 0.0))
    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=gx, gz=gz, lx=lx, lz=lz, height=height.astype(np.float32),
        geom_ok=np.ones((H, W), bool),
    )  # fmt: skip
    plate_file = None
    if plate:
        img = np.zeros((H, W, 3), np.uint8)
        img[FLOOR] = FLOOR_RGB
        img[WALL] = WALL_RGB
        img[TABLE] = TABLE_RGB
        img[TABLE2] = TABLE2_RGB
        Image.fromarray(img).save(root / "plate.png")
        plate_file = "plate.png"
    CameraFile(
        camera_id=CAMERA,
        image_size_px=(W, H),
        camera=Camera(fx=380.0, fy=380.0, cx=W / 2, cy=H / 2),
        plane=GroundPlane(height=2.5, pitch=math.radians(50.0)),
        mask_files={k: f"{k}.png" for k in masks},
        plate_file=plate_file,
    ).save(commission / f"{CAMERA}.camera.json")

    def load(path, _cf):
        with np.load(path) as cache:
            return {key: cache[key] for key in cache.files}

    monkeypatch.setattr(scene_mesh, "load_geometry_cache", load, raising=False)
    return root


def _keys(root):
    _cf, items, _h, _shapes = scene_mesh.build_scene_regular(CAMERA, root)
    return {str(k): k for _m, k, _a, _s in items}


def test_each_item_is_painted_the_plate_colour_under_it(tmp_path, monkeypatch):
    _cf, items, _h, _shapes = scene_mesh.build_scene_regular(
        CAMERA, _store(tmp_path, monkeypatch)
    )
    painted = {(str(k), colour_of(k)) for _m, k, _a, _s in items}
    # two tables of different colours, each its own -- not one class-wide median
    assert {("display_table", TABLE_RGB), ("display_table", TABLE2_RGB)} <= painted
    assert ("wall", WALL_RGB) in painted
    assert ("floor", FLOOR_RGB) in painted


def test_the_colour_is_the_lit_half_not_the_shadowed_mean(tmp_path, monkeypatch):
    """Half a counter's pixels are its shadowed front; the colour is the lit top's."""
    root = _store(tmp_path, monkeypatch)
    img = np.asarray(Image.open(root / "plate.png")).copy()
    rows, cols = TABLE
    img[rows.start : (rows.start + rows.stop) // 2, cols] = (20, 4, 4)  # the shadowed half
    Image.fromarray(img).save(root / "plate.png")
    _cf, items, _h, _shapes = scene_mesh.build_scene_regular(CAMERA, root)
    painted = {(str(k), colour_of(k)) for _m, k, _a, _s in items}
    assert ("display_table", TABLE_RGB) in painted


def test_without_a_plate_the_palette_stands(tmp_path, monkeypatch):
    keys = _keys(_store(tmp_path, monkeypatch, plate=False))
    assert not any(isinstance(k, Painted) for k in keys.values())
    assert colour_of(keys["display_table"]) == PALETTE["display_table"]


def test_too_few_pixels_do_not_paint(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_mesh, "COLOUR_MIN_PX", 10**9)
    keys = _keys(_store(tmp_path, monkeypatch))
    assert not isinstance(keys["display_table"], Painted)


def test_the_store_axis_is_fitted_to_the_ungated_evidence(tmp_path, monkeypatch):
    root = _store(tmp_path, monkeypatch, ghost_offset_m=1.0)
    gated = scene_mesh.cell_grids(CAMERA, root)[1][scene_mesh.WALL_CID].sum()
    ungated = scene_mesh.cell_grids(CAMERA, root, gated=False)[1][scene_mesh.WALL_CID].sum()
    assert ungated > gated, "the gate removed nothing; the test store has no ghost"
    seen = []
    real = scene_mesh.store_yaw

    def spy(grids):
        seen.append(int(grids[scene_mesh.WALL_CID].sum()))
        return real(grids)

    monkeypatch.setattr(scene_mesh, "store_yaw", spy)
    scene_mesh.build_scene_regular(CAMERA, root)
    assert seen == [int(ungated)]
