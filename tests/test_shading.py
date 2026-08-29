"""The shared camera and light rig: the things that would make a picture *wrong*.

Shading tests that assert on colours are brittle and prove little, so these pin the
handedness, the invariants a refit must not break, and the one ordering rule that stops a
figure being painted over the fixture standing in front of it.

pytest tests/test_shading.py -v
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from syncai_bev3d.meshes import box, smooth_normals
from syncai_bev3d.shading import View, contact_shadows, draw_scene, shade

BG = (7, 9, 13)


def _view(focal=600.0, cx=0.0, cy=0.0):
    return View([0.0, 5.0, -3.0], [0.0, 0.0, 4.0], focal, cx, cy)


def test_world_x_is_screen_right():
    """The mirror this module was written to settle. `scripts/mesh_preview.py` (deleted in
    `500cdd2`) built its basis the other way round and put world +x on the left; nothing caught it because the
    scene it drew was hand-authored. A mirrored panel is a map that sends a robot the
    wrong way."""
    v = _view()
    u_left, _, _ = v.project(-1.0, 0.0, 4.0)
    u_right, _, _ = v.project(1.0, 0.0, 4.0)
    assert float(u_left) < float(u_right)


def test_a_straight_down_view_is_refused_rather_than_silently_degenerate():
    """It has no lateral axis, so the basis is not defined. Returning something anyway
    would produce a picture whose left-right meaning is whatever the rounding decided."""
    with pytest.raises(ValueError, match="lateral axis"):
        View([0.0, 5.0, 0.0], [0.0, 0.0, 0.0], 600.0)

    with pytest.raises(ValueError, match="no direction"):
        View([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 600.0)


def test_refitting_changes_the_lens_and_never_the_eye():
    """`_fit_view` sizes the scene to the panel. If a refit could move the camera it would
    be free to answer "does it fit" by choosing a different view, and the picture would
    stop being of the camera the caller asked for."""
    v = _view()
    w = v.with_intrinsics(1234.0, 55.0, -12.0)
    assert np.allclose(w.eye, v.eye)
    assert np.allclose(w.fwd, v.fwd)
    assert np.allclose(w.right, v.right)
    assert (w.focal_px, w.cx, w.cy) == (1234.0, 55.0, -12.0)


def test_the_basis_is_orthonormal():
    v = View([3.0, 4.0, -2.0], [-1.0, 0.5, 5.0], 700.0)
    for a in (v.right, v.up, v.fwd):
        assert float(np.linalg.norm(a)) == pytest.approx(1.0)
    assert float(v.right @ v.up) == pytest.approx(0.0, abs=1e-12)
    assert float(v.right @ v.fwd) == pytest.approx(0.0, abs=1e-12)
    assert float(v.up @ v.fwd) == pytest.approx(0.0, abs=1e-12)


def test_depth_is_returned_unclamped_so_behind_the_eye_is_distinguishable():
    """A caller culls on the sign. Clamping the depth to a small positive number instead
    turns "behind the camera" into "very close", which draws it enormous rather than not
    at all."""
    v = _view()
    _, _, behind = v.project(0.0, 5.0, -20.0)
    assert float(behind) < 0.0


def test_shading_stays_inside_the_byte_range_and_responds_to_the_key():
    verts, faces = box(1.0, 1.0, 1.0)
    n = smooth_normals(verts, faces)
    depth = np.full(len(faces), 4.0)
    lit = shade(n, (200, 200, 200), depth, _view(), bg=BG)
    assert lit.shape == (len(faces), 3)
    assert lit.min() >= 0 and lit.max() <= 255
    assert lit.max() > lit.min(), "a rig where every face gets the same value is not one"


def test_distance_fades_toward_the_background_and_only_when_asked():
    verts, faces = box(1.0, 1.0, 1.0)
    n = smooth_normals(verts, faces)
    v = _view()
    near = shade(n, (200, 200, 200), np.full(len(faces), 2.0), v, bg=BG)
    far = shade(n, (200, 200, 200), np.full(len(faces), 40.0), v, bg=BG)
    assert far.mean() < near.mean()
    unfogged = shade(n, (200, 200, 200), np.full(len(faces), 40.0), v, bg=BG, fog=False)
    assert unfogged.mean() > far.mean()


def test_one_sort_across_every_item_not_one_per_mesh():
    """The reason `draw_scene` exists. Sorting each mesh separately makes the order between
    two meshes an accident of the order the caller listed them in, so a far object listed
    last paints over a near one."""
    img = Image.new("RGB", (200, 200), BG)
    d = ImageDraw.Draw(img, "RGBA")
    near = (box(1.0, 1.0, 1.0)[0] + [0.0, 0.0, 3.0], box(1.0, 1.0, 1.0)[1])
    far = (box(3.0, 3.0, 1.0)[0] + [0.0, 0.0, 9.0], box(3.0, 3.0, 1.0)[1])
    v = View([0.0, 1.0, -2.0], [0.0, 0.8, 6.0], 220.0, 100.0, 100.0)

    # the far one listed *second*, which is the order that gets it wrong without a merge
    draw_scene(d, v, [(near, (255, 0, 0), 255), (far, (0, 0, 255), 255)], bg=BG)
    px = np.asarray(img).reshape(-1, 3)
    reds = int(((px[:, 0] > 100) & (px[:, 2] < 60)).sum())
    assert reds > 0, "the near object was painted over by the far one"


def test_contact_shadows_darken_the_floor_and_cost_nothing_when_there_is_nothing():
    v = _view(cx=110.0, cy=110.0)
    empty = contact_shadows((220, 220), v, [], blur_px=4)
    assert np.asarray(empty)[..., 3].max() == 0

    standing = (box(1.0, 1.8, 1.0)[0] + [0.0, 0.0, 4.0], box(1.0, 1.8, 1.0)[1])
    layer = np.asarray(contact_shadows((220, 220), v, [standing], blur_px=4))
    assert layer[..., 3].max() > 0, "nothing was drawn, so everything will hover"
    assert layer[..., 3].min() == 0, "a shadow covering the whole panel is not a shadow"
