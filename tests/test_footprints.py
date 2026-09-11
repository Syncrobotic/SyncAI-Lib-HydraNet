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


# ------------------------------------------------------------ Gate D3: one surface, one box


def _store_two_tables(tmp_path, monkeypatch, *, gap_m, dh=0.0):
    """Two counters side by side along x, `gap_m` apart, the second `dh` higher."""
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
    tables = (
        (-1.6, -0.2 - gap_m / 2, 4.0, 5.0, 0.85),
        (-0.2 + gap_m / 2, 1.2, 4.0, 5.0, 0.85 + dh),
    )
    for i, (x0, x1, z0, z1, th) in enumerate(tables, start=1):
        top = _mask([_poly_px([(x0, th, z0), (x1, th, z0), (x1, th, z1), (x0, th, z1)])])
        front = _mask([_poly_px([(x0, 0, z0), (x1, 0, z0), (x1, th, z0), (x0, th, z0)])])
        tab = (top | front) & (objects == 0)
        static[tab] = 4
        objects[tab] = i
        horiz[top & tab] = 1.0
        height[tab] = th
    table = static == 4
    Image.fromarray(np.where(table, 255, 0).astype(np.uint8)).save(
        commission / "display_table.png"
    )
    floor = ok & (gz > 0.5) & (gz < 12) & ~table
    Image.fromarray(np.where(floor, 255, 0).astype(np.uint8)).save(commission / "walkable.png")
    Image.fromarray(objects).save(commission / "objects.png")
    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=np.nan_to_num(gx).astype(np.float32), gz=np.nan_to_num(gz, nan=99.0).astype(np.float32),
        lx=np.nan_to_num(gx).astype(np.float32), lz=np.nan_to_num(gz, nan=99.0).astype(np.float32),
        height=height, horiz=horiz, geom_ok=ok,
    )  # fmt: skip
    CameraFile(
        camera_id=CAMERA, image_size_px=(W, H), camera=CAM, plane=PLANE,
        mask_files={"display_table": "display_table.png", "walkable": "walkable.png", "objects": "objects.png"},
    ).save(commission / f"{CAMERA}.camera.json")  # fmt: skip

    def load(path, _cf):
        with np.load(path) as cache:
            return {key: cache[key] for key in cache.files}

    monkeypatch.setattr(scene_mesh, "load_geometry_cache", load, raising=False)
    return root


def _tables_after_d3(root):
    ev = scene_mesh.load_evidence(CAMERA, root)
    fps = footprints.regularise_footprints(footprints.object_footprints(ev, 0.0), ev, 0.0)
    return [fp for fp in fps if fp.name == "display_table"]


def test_two_counters_end_to_end_at_one_height_are_one_box(tmp_path, monkeypatch):
    tables = _tables_after_d3(_store_two_tables(tmp_path, monkeypatch, gap_m=0.05))
    assert len(tables) == 1, [(t.u0, t.u1, t.source) for t in tables]
    assert abs((tables[0].u1 - tables[0].u0) - 2.8) < 0.3 and tables[0].members == (1, 2)


def test_two_counters_with_an_aisle_between_stay_two(tmp_path, monkeypatch):
    tables = _tables_after_d3(_store_two_tables(tmp_path, monkeypatch, gap_m=0.8))
    assert len(tables) == 2


def test_two_counters_at_different_heights_stay_two(tmp_path, monkeypatch):
    tables = _tables_after_d3(_store_two_tables(tmp_path, monkeypatch, gap_m=0.05, dh=0.3))
    assert len(tables) == 2


def test_an_l_shaped_counter_row_is_two_boxes(tmp_path, monkeypatch):
    """One object whose top is an L: a box cannot hold it; two can."""
    root = tmp_path / "checkout"
    commission = root / "runs/commission01"
    commission.mkdir(parents=True)
    (root / "runs/site30k_qa/geometry_cache").mkdir(parents=True)
    ys, xs = np.mgrid[0:H, 0:W]
    gx, gz = pixel_to_ground(xs + 0.5, ys + 0.5, CAM, PLANE)
    ok = np.isfinite(gz)
    th = 0.85
    legs = ((-1.5, 1.5, 4.0, 4.8), (0.7, 1.5, 2.4, 4.0))  # along x, and a leg toward the camera
    top = np.zeros((H, W), bool)
    front = np.zeros((H, W), bool)
    for x0, x1, z0, z1 in legs:
        top |= _mask([_poly_px([(x0, th, z0), (x1, th, z0), (x1, th, z1), (x0, th, z1)])])
        front |= _mask([_poly_px([(x0, 0, z0), (x1, 0, z0), (x1, th, z0), (x0, th, z0)])])
    tab = top | front
    static = np.where(tab, 4, 255).astype(np.uint8)
    objects = np.where(tab, 1, 0).astype(np.uint16)
    Image.fromarray(np.where(tab, 255, 0).astype(np.uint8)).save(
        commission / "display_table.png"
    )
    floor = ok & (gz > 0.5) & (gz < 12) & ~tab
    Image.fromarray(np.where(floor, 255, 0).astype(np.uint8)).save(commission / "walkable.png")
    Image.fromarray(objects).save(commission / "objects.png")
    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=np.nan_to_num(gx).astype(np.float32), gz=np.nan_to_num(gz, nan=99.0).astype(np.float32),
        lx=np.nan_to_num(gx).astype(np.float32), lz=np.nan_to_num(gz, nan=99.0).astype(np.float32),
        height=np.where(tab, th, 0.0).astype(np.float32), horiz=top.astype(np.float32), geom_ok=ok,
    )  # fmt: skip
    CameraFile(
        camera_id=CAMERA, image_size_px=(W, H), camera=CAM, plane=PLANE,
        mask_files={"display_table": "display_table.png", "walkable": "walkable.png", "objects": "objects.png"},
    ).save(commission / f"{CAMERA}.camera.json")  # fmt: skip

    def load(path, _cf):
        with np.load(path) as cache:
            return {key: cache[key] for key in cache.files}

    monkeypatch.setattr(scene_mesh, "load_geometry_cache", load, raising=False)
    del static
    tables = _tables_after_d3(root)
    assert len(tables) == 2, [
        (round(t.u1 - t.u0, 2), round(t.v1 - t.v0, 2), t.source) for t in tables
    ]
    spans = sorted(
        (max(t.u1 - t.u0, t.v1 - t.v0), min(t.u1 - t.u0, t.v1 - t.v0)) for t in tables
    )
    assert all(short < 1.2 for _, short in spans), spans


# ------------------------------------------------------ Gate D4: walls from their feet


def _store_with_wall(tmp_path, monkeypatch, *, z_wall=8.0, to_horizon=False):
    """A wall across the room at `z_wall`, its face drawn through the camera; with
    `to_horizon` the mask runs on up past the horizon, as a real wall mask does."""
    root = tmp_path / "checkout"
    commission = root / "runs/commission01"
    commission.mkdir(parents=True)
    (root / "runs/site30k_qa/geometry_cache").mkdir(parents=True)
    ys, xs = np.mgrid[0:H, 0:W]
    gx, gz = pixel_to_ground(xs + 0.5, ys + 0.5, CAM, PLANE)
    ok = np.isfinite(gz)
    face = _mask(
        [
            _poly_px(
                [(-4.0, 0, z_wall), (4.0, 0, z_wall), (4.0, 2.4, z_wall), (-4.0, 2.4, z_wall)]
            )
        ]
    ).copy()
    if to_horizon:
        top = int(np.nonzero(face.any(axis=1))[0].min())
        face[: top + 1, :] = face[top, :]  # everything above the wall's top too
        face[:top, :] |= face[top, :]
    static = np.where(face, 2, 255).astype(np.uint8)
    objects = np.where(face, 1, 0).astype(np.uint16)
    Image.fromarray(np.where(face, 255, 0).astype(np.uint8)).save(commission / "wall.png")
    floor = ok & (gz > 0.5) & (gz < z_wall - 0.1) & ~face
    Image.fromarray(np.where(floor, 255, 0).astype(np.uint8)).save(commission / "walkable.png")
    Image.fromarray(objects).save(commission / "objects.png")
    np.savez(
        root / f"runs/site30k_qa/geometry_cache/{CAMERA}.npz",
        gx=np.nan_to_num(gx).astype(np.float32), gz=np.nan_to_num(gz, nan=99.0).astype(np.float32),
        lx=np.nan_to_num(gx).astype(np.float32), lz=np.nan_to_num(gz, nan=99.0).astype(np.float32),
        height=np.where(face, 1.2, 0.0).astype(np.float32), horiz=np.zeros((H, W), np.float32), geom_ok=ok,
    )  # fmt: skip
    CameraFile(
        camera_id=CAMERA, image_size_px=(W, H), camera=CAM, plane=PLANE,
        mask_files={"wall": "wall.png", "walkable": "walkable.png", "objects": "objects.png"},
    ).save(commission / f"{CAMERA}.camera.json")  # fmt: skip

    def load(path, _cf):
        with np.load(path) as cache:
            return {key: cache[key] for key in cache.files}

    monkeypatch.setattr(scene_mesh, "load_geometry_cache", load, raising=False)
    del static
    return root


def test_a_wall_stands_where_its_mask_meets_the_floor(tmp_path, monkeypatch):
    root = _store_with_wall(tmp_path, monkeypatch, z_wall=8.0)
    ev = scene_mesh.load_evidence(CAMERA, root)
    runs = footprints.wall_runs_from_feet(ev, 0.0)
    assert len(runs) == 1, runs
    assert runs[0].axis == "u" and abs(runs[0].perp - 8.0) < 0.2, runs[0]
    assert runs[0].hi - runs[0].lo > 5.0


def test_a_wall_mask_reaching_the_horizon_still_stands_at_its_foot(tmp_path, monkeypatch):
    root = _store_with_wall(tmp_path, monkeypatch, z_wall=8.0, to_horizon=True)
    ev = scene_mesh.load_evidence(CAMERA, root)
    runs = footprints.wall_runs_from_feet(ev, 0.0)
    assert len(runs) == 1 and abs(runs[0].perp - 8.0) < 0.2, runs


def test_the_build_draws_the_wall_from_its_feet(tmp_path, monkeypatch):
    root = _store_with_wall(tmp_path, monkeypatch, z_wall=8.0)
    _cf, items, _h, _shapes = scene_mesh.build_scene_regular(CAMERA, root)
    walls = [m for m, k, _a, _s in items if k == "wall"]
    assert len(walls) == 1
    v = walls[0][0]
    assert abs(v[:, 2].mean() - 8.0) < 0.25, v[:, 2].mean()


# --------------------------------------------------- the score is over what is seen


def test_a_box_behind_another_object_is_not_penalised_for_it(tmp_path, monkeypatch):
    """A wall behind a counter: the counter hides the wall's foot, and the box's pixels
    on the counter are neither hit nor miss."""
    root = _store_with_wall(tmp_path, monkeypatch, z_wall=8.0)
    ev = scene_mesh.load_evidence(CAMERA, root)
    run = footprints.wall_runs_from_feet(ev, 0.0)[0]
    clear = run.iou
    # a counter object in front of the wall, painted over the wall's lower pixels
    front = _mask(
        [_poly_px([(-2.0, 0, 6.0), (2.0, 0, 6.0), (2.0, 0.9, 6.0), (-2.0, 0.9, 6.0)])]
    )
    top = _mask(
        [_poly_px([(-2.0, 0.9, 6.0), (2.0, 0.9, 6.0), (2.0, 0.9, 7.0), (-2.0, 0.9, 7.0)])]
    )
    counter = front | top
    ev.objects[counter] = 2
    ev.static[counter] = 4
    occluded = footprints._wall_iou(run, ev, 0.0)
    assert occluded >= clear - 0.10, (clear, occluded)
    # and the plain IoU, which charges the box for the counter in front of it, would not
    sil = footprints._silhouette(
        footprints.Footprint(
            "wall", 1, run.lo, run.hi, run.perp - 0.075, run.perp + 0.075, 2.4, 0.0, 0, "foot"
        ),
        ev,
        0.0,
    )
    wall = ev.objects == 1
    naive = (sil & wall).sum() / (sil | wall).sum()
    assert naive < 0.4 < occluded, (naive, occluded)


def test_merchandise_on_a_counter_counts_as_the_counter(tmp_path, monkeypatch):
    root = _store(tmp_path, monkeypatch, table=(-1.0, 1.2, 4.0, 5.0, 0.85))
    ev = scene_mesh.load_evidence(CAMERA, root)
    fp = _by_name(footprints.object_footprints(ev, 0.0))["display_table"]
    bare = footprints.reprojection_iou(fp, ev, 0.0)
    # laptops on the top: pixels above the counter's top, inside its outline
    goods = _mask(
        [_poly_px([(-0.8, 0.85, 4.3), (1.0, 0.85, 4.3), (1.0, 1.15, 4.3), (-0.8, 1.15, 4.3)])]
    )
    ev.products = goods
    with_goods = footprints.reprojection_iou(fp, ev, 0.0)
    assert with_goods >= bare - 0.02, (bare, with_goods)


# ------------------------------------------ a welded object, cut by its instances


def test_two_counters_welded_into_one_object_split_by_their_instances(tmp_path, monkeypatch):
    """The object map says one thing; the SAM 3 instances behind it say two, side by
    side with an aisle between -- and the one box scores badly enough to ask them."""
    root = _store_two_tables(tmp_path, monkeypatch, gap_m=1.2)
    commission = root / "runs/commission01"
    objects = np.asarray(Image.open(commission / "objects.png")).astype(np.uint16)
    inst = [(objects == 1), (objects == 2)]
    objects[objects == 2] = 1  # welded: one object
    Image.fromarray(objects).save(commission / "objects.png")
    ih, iw = H // 2, W // 2  # instances live at masks_pass's half resolution
    packed = np.stack(
        [
            np.packbits(
                np.asarray(Image.fromarray(m).resize((iw, ih), Image.Resampling.NEAREST), bool),
                axis=-1,
            )
            for m in inst
        ]
    )
    np.savez_compressed(
        commission / "instances.npz",
        masks=packed, shape=np.array([ih, iw]), cluster=np.array([0, 0], np.int32),
        concept=np.array(["fixture", "fixture"]), prompt=np.array(["retail counter"] * 2),
        score=np.array([0.9, 0.9], np.float32),
    )  # fmt: skip
    cf = CameraFile.load(commission / f"{CAMERA}.camera.json")
    import dataclasses

    dataclasses.replace(cf, mask_files={**cf.mask_files, "instances": "instances.npz"}).save(
        commission / f"{CAMERA}.camera.json"
    )
    ev = scene_mesh.load_evidence(CAMERA, root)
    assert ev.instances is not None and len(ev.instance_masks(1)) == 2
    one = footprints.object_footprints(ev, 0.0)
    assert len(one) == 1 and one[0].iou < footprints.INSTANCE_SPLIT_MAX, one
    tables = footprints.regularise_footprints(one, ev, 0.0)
    assert len(tables) == 2, [(round(t.u0, 2), round(t.u1, 2), t.source) for t in tables]
    assert all("instance" in t.source for t in tables)
    assert max(t.u1 - t.u0 for t in tables) < 1.8


def test_a_wall_patch_ending_above_the_floor_cannot_locate_a_wall(tmp_path, monkeypatch):
    root = _store_with_wall(tmp_path, monkeypatch, z_wall=8.0)
    ev = scene_mesh.load_evidence(CAMERA, root)
    rows = np.flatnonzero((ev.objects == 1).any(axis=1))
    # The wall is visible only above an occluding fixture. Its visible lower edge
    # is now far from the observed floor, but still projects to finite floor metres.
    ev.objects[max(0, rows[-1] - 10) : rows[-1] + 1] = 0
    assert footprints.wall_runs_from_feet(ev, 0.0) == []


def test_a_rectangular_counter_does_not_become_a_round_table(tmp_path, monkeypatch):
    root = _store(tmp_path, monkeypatch, table=(-0.8, 0.8, 2.5, 3.5, 0.85))
    ev = scene_mesh.load_evidence(CAMERA, root)
    assert footprints._round_candidate(ev.objects == 1, ev, 0.0, 1) is None


def test_round_podium_fit_recovers_known_ground_position(tmp_path, monkeypatch):
    from scipy.spatial import ConvexHull

    root = _store(tmp_path, monkeypatch, table=(-0.8, 0.8, 2.5, 3.5, 0.85))
    ev = scene_mesh.load_evidence(CAMERA, root)
    # A cylinder with known dimensions, independently projected into the image.
    radius, z, height = 0.55, 3.0, 0.85
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    pixels = np.array(
        _poly_px(
            [
                (radius * np.cos(a), h, z + radius * np.sin(a))
                for h in (0, height)
                for a in angles
            ]
        )
    )
    mask = _mask([list(map(tuple, pixels[ConvexHull(pixels).vertices]))])
    fitted = footprints._round_candidate(mask, ev, 0.0, 1)
    assert fitted is not None and fitted.kind == "round"
    assert abs((fitted.u0 + fitted.u1) / 2) < 0.08
    assert abs((fitted.v0 + fitted.v1) / 2 - z) < 0.12
    assert abs((fitted.u1 - fitted.u0) / 2 - radius) < 0.08
    assert abs(fitted.h - height) < 0.10


def test_scene_build_does_not_draw_a_wall_across_a_glass_door(tmp_path, monkeypatch):
    root = _store_with_wall(tmp_path, monkeypatch, z_wall=5.0)
    # This test supplies observed floor right up to the threshold. The wall fixture
    # otherwise leaves a 10 cm unobserved strip, too wide at this distance to anchor it.
    r, c = np.mgrid[:H, :W]
    _gx, gz = pixel_to_ground(c + 0.5, r + 0.5, CAM, PLANE)
    floor = np.isfinite(gz) & (gz > 0.5) & (gz < 5.0)
    Image.fromarray(np.where(floor, 255, 0).astype(np.uint8)).save(
        root / "runs/commission01/walkable.png"
    )
    mask = _mask([_poly_px([(-0.7, 0, 5), (-0.7, 2.4, 5), (0.7, 2.4, 5), (0.7, 0, 5)])])
    directory = root / "runs/commission01" / CAMERA / "masks"
    directory.mkdir(parents=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(directory / "glass_door.png")
    _cf, items, _heights, _shapes = scene_mesh.build_scene_regular(CAMERA, root)
    assert any(key == "glass" for _mesh, key, _alpha, _shadow in items)
    walls = [mesh for mesh, key, _alpha, _shadow in items if key == "wall"]
    assert len(walls) == 2
    for vertices, _faces in walls:
        assert vertices[:, 0].max() < -0.5 or vertices[:, 0].min() > 0.5
