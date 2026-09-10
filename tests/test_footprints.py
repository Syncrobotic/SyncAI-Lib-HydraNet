"""A fixture's footprint comes from the image and one number (Gate D2).

The store here is drawn THROUGH a real camera: a box on the floor is projected into the
plate, its top face becomes the table mask with `horiz` = 1, its front face the shelf
mask; the cache's `gx/gz` are the camera's own floor hits. The per-pixel heights are
deliberately garbage on the faces (a white pillar's collapse), because the footprint must
not read them -- only the height scalar and the geometry.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d import footprints, scene_mesh
from syncai_hydranet.geometry.camera_json import Camera, CameraFile, GroundPlane
from syncai_hydranet.geometry.ground import pixel_to_ground

CAMERA = "cam-footprints"
W, H = 640, 360
CAM = Camera(fx=420.0, fy=420.0, cx=W / 2, cy=H / 2)
PLANE = GroundPlane(height=2.8, pitch=math.radians(42.0))


def _poly_px(points_m):
    """Scene points (x, y up, z) -> plate pixels through the camera."""
    out = []
    for x, y, z in points_m:
        pts = np.array([[x, PLANE.height - y, z]])
        cam = pts @ PLANE.rotation.T
        u = CAM.fx * cam[0, 0] / cam[0, 2] + CAM.cx
        v = CAM.fy * cam[0, 1] / cam[0, 2] + CAM.cy
        out.append((float(u), float(v)))
    return out


def _mask(polys):
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    for poly in polys:
        d.polygon(poly, fill=1)
    return np.asarray(img, bool)


def _store(tmp_path, monkeypatch, *, table, shelf=None, column=None):
    """`table` = (x0, x1, z0, z1, h); `shelf` = (x0, x1, z_front, h) along x; `column` = (x, z, side, h)."""
    root = tmp_path / "checkout"
    commission = root / "runs/commission01"
    commission.mkdir(parents=True)
    (root / "runs/site30k_qa/geometry_cache").mkdir(parents=True)
    ys, xs = np.mgrid[0:H, 0:W]
    gx, gz = pixel_to_ground(xs + 0.5, ys + 0.5, CAM, PLANE)
    ok = np.isfinite(gz)
    static = np.full((H, W), 255, np.uint8)
    objects = np.zeros((H, W), np.uint16)
    horiz = np.zeros((H, W), np.float32)
    height = np.zeros((H, W), np.float32)
    mask_files = {}
    x0, x1, z0, z1, th = table
    top = _mask([_poly_px([(x0, th, z0), (x1, th, z0), (x1, th, z1), (x0, th, z1)])])
    front = _mask([_poly_px([(x0, 0, z0), (x1, 0, z0), (x1, th, z0), (x0, th, z0)])])
    tab = top | front
    static[tab] = 4
    objects[tab] = 1
    horiz[top] = 1.0
    height[tab] = np.where(top[tab], th, th * 0.5)  # faces read wrong on purpose
    mask_files["display_table"] = "display_table.png"
    Image.fromarray(np.where(tab, 255, 0).astype(np.uint8)).save(
        commission / "display_table.png"
    )
    if shelf is not None:
        sx0, sx1, sz, sh = shelf
        face = _mask([_poly_px([(sx0, 0, sz), (sx1, 0, sz), (sx1, sh, sz), (sx0, sh, sz)])])
        static[face] = 5
        objects[face] = 2
        height[face] = 3.0  # nonsense per pixel, on purpose
        mask_files["display_shelf"] = "display_shelf.png"
        Image.fromarray(np.where(face, 255, 0).astype(np.uint8)).save(
            commission / "display_shelf.png"
        )
    if column is not None:
        cx, cz, side, ch = column
        face = _mask(
            [
                _poly_px(
                    [
                        (cx - side / 2, 0, cz),
                        (cx + side / 2, 0, cz),
                        (cx + side / 2, ch, cz),
                        (cx - side / 2, ch, cz),
                    ]
                )
            ]
        )
        static[face] = 3
        objects[face] = 3
        height[face] = 0.4  # nonsense per pixel, on purpose
        mask_files["column"] = "column.png"
        Image.fromarray(np.where(face, 255, 0).astype(np.uint8)).save(commission / "column.png")
    floor = ok & (gz > 0.5) & (gz < 12) & (static == 255)
    mask_files["walkable"] = "walkable.png"
    Image.fromarray(np.where(floor, 255, 0).astype(np.uint8)).save(commission / "walkable.png")
    Image.fromarray(objects).save(commission / "objects.png")
    mask_files["objects"] = "objects.png"
    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=np.nan_to_num(gx).astype(np.float32), gz=np.nan_to_num(gz, nan=99.0).astype(np.float32),
        lx=np.nan_to_num(gx).astype(np.float32), lz=np.nan_to_num(gz, nan=99.0).astype(np.float32),
        height=height, horiz=horiz, geom_ok=ok,
    )  # fmt: skip
    CameraFile(
        camera_id=CAMERA, image_size_px=(W, H), camera=CAM, plane=PLANE, mask_files=mask_files
    ).save(commission / f"{CAMERA}.camera.json")

    def load(path, _cf):
        with np.load(path) as cache:
            return {key: cache[key] for key in cache.files}

    monkeypatch.setattr(scene_mesh, "load_geometry_cache", load, raising=False)
    return root


def _by_name(fps):
    return {fp.name: fp for fp in fps}


def test_a_table_is_its_top_on_its_own_height_plane(tmp_path, monkeypatch):
    root = _store(tmp_path, monkeypatch, table=(-1.0, 1.2, 4.0, 5.0, 0.85))
    ev = scene_mesh.load_evidence(CAMERA, root)
    fp = _by_name(footprints.object_footprints(ev, 0.0))["display_table"]
    assert fp.source == "top"
    assert abs((fp.u1 - fp.u0) - 2.2) < 0.22 and abs((fp.v1 - fp.v0) - 1.0) < 0.15, fp
    assert abs((fp.u0 + fp.u1) / 2 - 0.1) < 0.1 and abs((fp.v0 + fp.v1) / 2 - 4.5) < 0.1, fp
    assert abs(fp.h - 0.85) < 0.02


def test_a_shelf_is_its_foot_with_the_class_depth_behind_it(tmp_path, monkeypatch):
    root = _store(
        tmp_path, monkeypatch, table=(-1.0, 1.2, 4.0, 5.0, 0.85), shelf=(-2.5, 0.5, 7.0, 1.9)
    )
    ev = scene_mesh.load_evidence(CAMERA, root)
    fp = _by_name(footprints.object_footprints(ev, 0.0))["display_shelf"]
    assert fp.source == "foot"
    assert abs(fp.u0 - -2.5) < 0.15 and abs(fp.u1 - 0.5) < 0.15, fp
    assert abs(fp.v0 - 7.0) < 0.1 and abs(fp.v1 - (7.0 + footprints.SHELF_DEPTH_M)) < 0.1, fp
    assert fp.h == footprints.HEIGHT_CLIP["display_shelf"][1] or fp.h >= 1.5  # clipped scalar


def test_a_column_is_square_on_its_foot(tmp_path, monkeypatch):
    root = _store(
        tmp_path, monkeypatch, table=(-1.0, 1.2, 4.0, 5.0, 0.85), column=(2.0, 6.0, 0.6, 2.4)
    )
    ev = scene_mesh.load_evidence(CAMERA, root)
    fp = _by_name(footprints.object_footprints(ev, 0.0))["column"]
    side = fp.u1 - fp.u0
    assert abs(side - 0.6) < 0.1 and abs((fp.v1 - fp.v0) - side) < 1e-6, fp
    assert abs((fp.u0 + fp.u1) / 2 - 2.0) < 0.1, fp


def test_a_box_on_its_own_object_scores_high_and_a_displaced_one_low(tmp_path, monkeypatch):
    root = _store(tmp_path, monkeypatch, table=(-1.0, 1.2, 4.0, 5.0, 0.85))
    ev = scene_mesh.load_evidence(CAMERA, root)
    fp = _by_name(footprints.object_footprints(ev, 0.0))["display_table"]
    good = footprints.reprojection_iou(fp, ev, 0.0)
    moved = footprints.Footprint(
        fp.name, fp.oid, fp.u0 + 2.0, fp.u1 + 2.0, fp.v0, fp.v1, fp.h, 0.0, fp.n_px, "top"
    )
    bad = footprints.reprojection_iou(moved, ev, 0.0)
    assert good > 0.6 and bad < 0.2, (good, bad)


def test_the_build_uses_the_object_footprints(tmp_path, monkeypatch):
    root = _store(tmp_path, monkeypatch, table=(-1.0, 1.2, 4.0, 5.0, 0.85))
    _cf, _items, _h, shapes = scene_mesh.build_scene_regular(CAMERA, root)
    tables = [(w, d, h) for n, w, d, h in shapes if n == "display_table"]
    assert len(tables) == 1
    w, d, h = tables[0]
    assert (
        abs(max(w, d) - 2.2) < 0.25 and abs(min(w, d) - 1.0) < 0.2 and abs(h - 0.85) < 0.02
    ), tables
