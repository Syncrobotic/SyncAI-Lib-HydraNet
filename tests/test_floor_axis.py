"""The store's axis is read off the floor's joints, and only when the floor has them.

On 2026-09-10 the fixture-blob axis was measured 36 deg off the tiles on Taichung-cam01;
the tiles were right. `floor_axis.floor_line_axis` reads the joints; `scene_mesh.store_axis`
uses it when the answer is sharp and falls back to the blobs when it is not.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d import floor_axis, scene_mesh
from syncai_hydranet.geometry.camera_json import Camera, CameraFile, GroundPlane
from syncai_hydranet.geometry.ground import ground_to_pixel

CAMERA = "cam-axis"
W, H = 640, 360
CAM = Camera(fx=420.0, fy=420.0, cx=W / 2, cy=H / 2)
PLANE = GroundPlane(height=2.6, pitch=math.radians(45.0))


def _tiled_plate(angle_deg, *, tile_m=0.6, noise=0.0, seed=0):
    """A plate whose floor is a tile grid laid at `angle_deg`, drawn through the camera."""
    img = Image.new("RGB", (W, H), (150, 150, 150))
    d = ImageDraw.Draw(img)
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    for k in np.arange(-12, 12.01, tile_m):
        for along in (np.linspace(-8, 8, 400),):
            # a line u = k (running along v) and a line v = k (running along u)
            for u, v in ((np.full_like(along, k), along), (along, np.full_like(along, k))):
                x, z = u * c - v * s, u * s + v * c
                u_px, v_px, depth = ground_to_pixel(x, z, CAM, PLANE)
                ok = np.isfinite(u_px) & np.isfinite(v_px) & (depth > 0) & (z > 0.5) & (z < 9)
                pts = [(float(a), float(b)) for a, b in zip(u_px[ok], v_px[ok], strict=True)]
                if len(pts) > 1:
                    d.line(pts, fill=(90, 90, 90), width=2)
    arr = np.asarray(img).astype(float)
    if noise:
        arr = arr + np.random.default_rng(seed).normal(0, noise, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _floor_geometry():
    """walk / gz / geom_ok for the whole lower half of the frame, and the projection."""
    ys, xs = np.mgrid[0:H, 0:W]
    from syncai_hydranet.geometry.ground import pixel_to_ground

    _gx, gz = pixel_to_ground(xs + 0.5, ys + 0.5, CAM, PLANE)
    geom_ok = np.isfinite(gz)
    walk = geom_ok & (gz > 0.5) & (gz < 9)

    def ground(px):
        x, z = pixel_to_ground(px[:, 0], px[:, 1], CAM, PLANE)
        return np.stack([x, z], axis=1)

    return walk, np.where(geom_ok, gz, 99.0), geom_ok, ground


def _axis(angle_deg, **kw):
    walk, gz, ok, ground = _floor_geometry()
    return floor_axis.floor_line_axis(_tiled_plate(angle_deg, **kw), walk, gz, ok, ground)


def test_a_tile_grid_is_read_at_its_own_angle():
    for laid in (0.0, 12.0, 37.0, 71.0):
        axis, second = _axis(laid)
        assert axis is not None, f"no axis at {laid} deg (second peak {second:.2f})"
        got = math.degrees(axis) % 90
        assert min(abs(got - laid), 90 - abs(got - laid)) < 2.0, (laid, got)
        assert second < floor_axis.SHARPNESS_MAX


def test_a_floor_without_lines_answers_none():
    walk, gz, ok, ground = _floor_geometry()
    plate = np.random.default_rng(1).integers(60, 200, (H, W, 3)).astype(np.uint8)
    axis, second = floor_axis.floor_line_axis(plate, walk, gz, ok, ground)
    assert axis is None and second > floor_axis.SHARPNESS_MAX


def test_too_little_floor_answers_none():
    walk, gz, ok, ground = _floor_geometry()
    walk = walk.copy()
    walk[:, :] = False
    walk[300:310, 300:310] = True
    assert floor_axis.floor_line_axis(_tiled_plate(20.0), walk, gz, ok, ground) == (None, 1.0)


# ------------------------------------------------------------- store_axis in the build


def _store(tmp_path, monkeypatch, *, plate):
    root = tmp_path / "checkout"
    commission = root / "runs/commission01"
    commission.mkdir(parents=True)
    (root / "runs/site30k_qa/geometry_cache").mkdir(parents=True)
    walk, _gz, ok, _ground = _floor_geometry()
    ys, xs = np.mgrid[0:H, 0:W]
    from syncai_hydranet.geometry.ground import pixel_to_ground

    gx, gz2 = pixel_to_ground(xs + 0.5, ys + 0.5, CAM, PLANE)
    gx, gz2 = np.nan_to_num(gx), np.nan_to_num(gz2, nan=99.0)
    # one long table blob, elongated along z (world), so the blob axis votes 90 deg
    table = np.zeros((H, W), bool)
    table[200:330, 300:330] = True
    for name, m in (("walkable", walk & ~table), ("display_table", table)):
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(commission / f"{name}.png")
    height = np.where(table, 0.8, 0.0).astype(np.float32)
    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=gx.astype(np.float32), gz=gz2.astype(np.float32),
        lx=gx.astype(np.float32), lz=gz2.astype(np.float32),
        height=height, geom_ok=ok,
    )  # fmt: skip
    plate_file = None
    if plate is not None:
        Image.fromarray(plate).save(root / "plate.png")
        plate_file = "plate.png"
    CameraFile(
        camera_id=CAMERA,
        image_size_px=(W, H),
        camera=CAM,
        plane=PLANE,
        mask_files={"walkable": "walkable.png", "display_table": "display_table.png"},
        plate_file=plate_file,
    ).save(commission / f"{CAMERA}.camera.json")

    def load(path, _cf):
        with np.load(path) as cache:
            return {key: cache[key] for key in cache.files}

    monkeypatch.setattr(scene_mesh, "load_geometry_cache", load, raising=False)
    return root


def test_the_build_takes_the_floor_axis_when_it_is_sharp(tmp_path, monkeypatch):
    root = _store(tmp_path, monkeypatch, plate=_tiled_plate(33.0))
    ev = scene_mesh.load_evidence(CAMERA, root)
    got = math.degrees(scene_mesh.store_axis(ev, CAMERA, root)) % 90
    assert abs(got - 33.0) < 2.0, got


def test_the_build_falls_back_to_the_blobs_without_lines(tmp_path, monkeypatch):
    root = _store(tmp_path, monkeypatch, plate=None)
    ev = scene_mesh.load_evidence(CAMERA, root)
    blob = scene_mesh.store_yaw(
        scene_mesh.cell_grids(CAMERA, root, gated=False, evidence=ev)[1]
    )
    assert scene_mesh.store_axis(ev, CAMERA, root) == blob


# ------------------------------------------------------------------ the joint period


def test_a_tile_grid_reports_its_own_pitch():
    walk, gz, ok, ground = _floor_geometry()
    for tile in (0.45, 0.60):
        plate = _tiled_plate(20.0, tile_m=tile)
        axis, _ = floor_axis.floor_line_axis(plate, walk, gz, ok, ground)
        period, strength = floor_axis.floor_period(plate, walk, gz, ok, ground, axis)
        assert period is not None and abs(period - tile) <= 0.02, (tile, period)
        assert strength > 0.2


def test_a_floor_without_joints_reports_no_period():
    walk, gz, ok, ground = _floor_geometry()
    plate = np.random.default_rng(2).integers(60, 200, (H, W, 3)).astype(np.uint8)
    period, strength = floor_axis.floor_period(plate, walk, gz, ok, ground, 0.0)
    assert period is None or strength < 0.2


# ------------------------------------------------------- two families, not one folded


def test_two_families_ninety_degrees_apart_fold_to_one_axis():
    rng = np.random.default_rng(3)
    ang = np.concatenate([rng.normal(7.0, 1.5, 4000), rng.normal(97.0, 1.5, 3000)]) % 180
    a1, a2, second = floor_axis._two_families(ang, np.ones(len(ang)))
    assert abs(a1 - 7.0) < 0.5 and abs(a2 - 97.0) < 0.5 and second < 0.1


def test_families_ten_degrees_off_square_still_give_the_stronger_one():
    """Tao-Hsin-cam15, 2026-09-10: planks at 7 and 107 deg in the assumed calibration.
    Folded mod 90 they were two peaks 10 deg apart and no axis; over 180 they are two
    families, the stronger is the axis, and the 10 deg is a reading on the calibration."""
    rng = np.random.default_rng(4)
    ang = np.concatenate([rng.normal(7.0, 1.5, 4000), rng.normal(107.0, 1.5, 3000)]) % 180
    a1, a2, second = floor_axis._two_families(ang, np.ones(len(ang)))
    assert abs(a1 - 7.0) < 0.5 and abs(a2 - 107.0) < 0.5 and second < 0.1


def test_a_lone_family_reports_no_partner():
    rng = np.random.default_rng(5)
    ang = rng.normal(30.0, 1.5, 4000) % 180
    a1, a2, _second = floor_axis._two_families(ang, np.ones(len(ang)))
    assert abs(a1 - 30.0) < 0.5 and a2 is None
