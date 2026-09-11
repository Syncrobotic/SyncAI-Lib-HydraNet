"""Transparent surfaces use a supported plane, not the depth behind the glass."""

import math
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d.surfaces import _contact_plane, fit_surface
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane, pixel_to_ground


def evidence(bottom=0.0, top=2.1):
    w, h = 640, 480
    cf = CameraFile(
        camera_id="surface",
        image_size_px=(w, h),
        camera=Camera(fx=420, fy=420, cx=w / 2, cy=h / 2),
        plane=GroundPlane(height=2.8, pitch=math.radians(25)),
    )
    world = np.array([[-1.5, bottom, 5], [-1.5, top, 5], [1.5, top, 5], [1.5, bottom, 5]])
    level = world.copy()
    level[:, 1] = cf.plane.height - level[:, 1]
    camera = level @ cf.plane.rotation.T
    px = camera[:, :2] / camera[:, 2, None] * 420 + [w / 2, h / 2]
    im = Image.new("1", (w, h))
    ImageDraw.Draw(im).polygon(list(map(tuple, px)), fill=1)
    mask = np.asarray(im, bool)
    r, c = np.mgrid[:h, :w]
    gx, gz = pixel_to_ground(c + 0.5, r + 0.5, cf.camera, cf.plane)
    walk = np.isfinite(gz) & (gz < 4.95) & (gz > 0) & ~mask
    ev = SimpleNamespace(cf=cf, z={"gx": gx, "gz": gz}, walk=walk)
    return mask, ev


def test_glass_ignores_depth_behind_it_and_recovers_its_contact_plane():
    mask, ev = evidence()
    ev.z["lx"] = np.full_like(ev.z["gx"], 99)
    ev.z["lz"] = np.full_like(ev.z["gx"], 99)
    result = fit_surface(mask, ev, kind="glass")
    assert result is not None
    assert np.allclose(result.points[:, 1], 5, atol=0.12)
    assert np.isclose(np.linalg.norm(np.diff(result.points, axis=0)), 3, atol=0.15)
    assert result.bottom == 0 and result.iou > 0.85


def test_floating_window_requires_a_known_wall_plane():
    mask, ev = evidence(bottom=0.8, top=2.0)
    assert fit_surface(mask, ev, kind="window") is None
    result = fit_surface(mask, ev, kind="window", walls=[("u", 5.0, -2.0, 2.0, 0.15)])
    assert result is not None and result.source == "wall plane"
    assert np.allclose(result.points[:, 1], 5, atol=0.01)
    assert abs(result.bottom - 0.8) < 0.08
    assert abs(result.height - 1.2) < 0.10


def test_window_cannot_attach_beyond_the_end_of_a_wall():
    mask, ev = evidence(bottom=0.8, top=2.0)
    assert fit_surface(mask, ev, kind="window", walls=[("u", 5.0, 4.0, 6.0, 0.15)]) is None


def test_contact_plane_rejects_occluding_edges():
    x = np.linspace(-2, 2, 200)
    sill = np.c_[x, 5 + 0.005 * np.sin(x)]
    occlusion = np.c_[np.linspace(-1, 1, 50), np.linspace(6, 9, 50)]
    along, normal, distance = _contact_plane(np.concatenate([sill, occlusion]))
    assert np.max(np.abs(sill @ normal - distance)) < 0.02
    assert abs(along[1]) < 0.02


def test_glass_door_carves_a_real_gap_in_the_wall():
    from syncai_bev3d.surfaces import Surface, wall_sections

    opening = Surface(np.array([[-0.5, 5.0], [0.5, 5.0]]), 0.0, 2.4, "glass_door", 0.9, "floor")
    sections = wall_sections([("u", 5.0, -2.0, 2.0, 0.15)], [opening], 0.0)
    assert sections == [("u", 5.0, -2.0, -0.5, 0.0, 2.4), ("u", 5.0, 0.5, 2.0, 0.0, 2.4)]


def test_window_keeps_the_wall_below_its_sill():
    from syncai_bev3d.surfaces import Surface, wall_sections

    opening = Surface(np.array([[-0.5, 5.0], [0.5, 5.0]]), 0.8, 1.2, "window", 0.9, "wall")
    sections = wall_sections([("u", 5.0, -2.0, 2.0, 0.15)], [opening], 0.0)
    assert ("u", 5.0, -0.5, 0.5, 0.0, 0.8) in sections
    assert ("u", 5.0, -0.5, 0.5, 2.0, 2.4) in sections


def test_a_glass_pane_in_front_of_a_wall_cannot_cut_it():
    from syncai_bev3d.surfaces import Surface, wall_sections

    opening = Surface(np.array([[-0.5, 3.0], [0.5, 3.0]]), 0.0, 2.4, "glass", 0.9, "floor")
    sections = wall_sections([("u", 5.0, -2.0, 2.0, 0.15)], [opening], 0.0)
    assert sections == [("u", 5.0, -2.0, 2.0, 0.0, 2.4)]


def test_adjacent_glass_does_not_double_the_reviewed_door_leaf():
    from syncai_bev3d.surfaces import Surface, separate_glazing

    pane = Surface(np.array([[-2.0, 5.0], [1.0, 5.0]]), 0.0, 2.4, "glass", 0.8, "shared")
    door = Surface(np.array([[-0.5, 5.0], [0.5, 5.0]]), 0.0, 2.4, "glass_door", 0.9, "shared")
    result = separate_glazing([pane, door])
    panes = [s for s in result if s.kind == "glass"]
    assert len(panes) == 2
    assert np.allclose(panes[0].points[:, 0], [-2.0, -0.5])
    assert np.allclose(panes[1].points[:, 0], [0.5, 1.0])
    assert any(s is door for s in result)
