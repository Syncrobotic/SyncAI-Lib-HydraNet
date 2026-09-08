"""The conversion between what an any-view model returns and what the bench can read.

`syncai_bev3d.geometry_teachers` runs Depth Anything 3 and VGGT, neither of which is a
dependency here and neither of which a CI runner will ever download. So the two model
calls do nothing but hand arrays to `reading_from_arrays`, and *that* is what these tests
hold: the vfov read off an intrinsics matrix, the aspect drift the loaders introduce,
the depth resize that must not turn holes into small numbers, and the refusal of a
revision that is not a commit id. Every one of those can be wrong quietly on a machine
that has the models, and none of them needs the models to be checked.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from syncai_bev3d import geometry_teachers as gt
from syncai_hydranet.geometry.ground import Camera

SHA = "0123456789abcdef0123456789abcdef01234567"


# ---------------------------------------------------------------------------
# the pin


@pytest.mark.parametrize("loose", ["main", "v1.0", SHA[:7], SHA.upper(), "", None])
def test_a_revision_that_is_not_a_commit_id_is_refused(loose):
    """`revision="main"` satisfies a keyword check and pins nothing; the same rule the
    pin test applies to constants, applied here to an argument."""
    with pytest.raises(ValueError, match="40-character commit id"):
        gt.pinned(loose)


def test_a_full_commit_id_passes_through_unchanged():
    assert gt.pinned(SHA) == SHA


# ---------------------------------------------------------------------------
# the lens


def test_vfov_from_intrinsics_inverts_camera_from_vfov():
    """`Camera.from_vfov` is how every metre downstream turns an angle into `fy`; the
    reading has to land on the same angle the fleet's 70.4 would produce."""
    for h, w, vfov in ((1080, 1920, 70.4), (294, 518, 70.4), (283, 504, 38.26)):
        cam = Camera.from_vfov(h, w, vfov)
        k = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
        assert gt.vfov_from_intrinsics(k, h) == pytest.approx(vfov, abs=1e-9)


def test_a_matrix_with_no_focal_has_no_vfov():
    with pytest.raises(ValueError, match="no positive focal"):
        gt.vfov_from_intrinsics(np.zeros((3, 3)), 1080)


def test_aspect_drift_is_zero_on_the_same_shape_and_reports_the_patch_rounding():
    """A 16:9 plate becomes 294x518 under VGGT's loader: 0.9% narrower, the height having
    been rounded up to the patch. That figure is what a reader compares a vfov
    disagreement against."""
    assert gt.aspect_drift((1080, 1920), (1080, 1920)) == 0.0
    assert gt.aspect_drift((1080, 1920), (294, 518)) == pytest.approx(-0.0089, abs=5e-4)
    # A square crop of a wide plate is not a rounding artefact and must not look like one.
    assert abs(gt.aspect_drift((1080, 1920), (518, 518))) > 0.4


def test_vggt_processed_hw_follows_the_loaders_crop_rule():
    """Width 518, height to a multiple of 14, cropped to 518 only when taller than that."""
    assert gt.vggt_processed_hw((1080, 1920)) == (294, 518)
    assert gt.vggt_processed_hw((540, 960)) == (294, 518)
    assert gt.vggt_processed_hw((1920, 1080)) == (518, 518)  # portrait: cropped
    h, w = gt.vggt_processed_hw((1080, 1920))
    assert h % gt.VGGT_PATCH == 0 and w % gt.VGGT_PATCH == 0


def test_vggt_preprocess_returns_chw_floats_in_unit_range_at_the_loader_shape():
    rgb = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    arr = gt.vggt_preprocess(rgb)
    assert arr.shape == (3, 294, 518)
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
    tall = gt.vggt_preprocess(np.zeros((1920, 1080, 3), np.uint8))
    assert tall.shape == (3, 518, 518), "a portrait plate is centre-cropped, as the loader does"


# ---------------------------------------------------------------------------
# the depth


def test_resize_depth_keeps_holes_as_nan_rather_than_as_small_depths():
    """A zero in a depth map is 'nothing here'. Bilinear interpolation of zeros next to
    3 m produces 1.5 m, which unprojects to a point in mid-air. The hole has to survive."""
    d = np.full((10, 20), 3.0, np.float32)
    d[:, 8:12] = 0.0
    out = gt.resize_depth(d, (20, 40))
    assert out.shape == (20, 40)
    assert np.isnan(out[:, 18:22]).all(), "the hole was filled"
    finite = out[np.isfinite(out)]
    assert finite.min() == pytest.approx(3.0, abs=1e-5), "hole values bled into the depth"


def test_resize_depth_on_the_same_lattice_only_masks():
    d = np.array([[1.0, 0.0], [np.inf, 2.0]], np.float32)
    out = gt.resize_depth(d, (2, 2))
    assert out[0, 0] == 1.0 and out[1, 1] == 2.0
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])


def test_resize_depth_refuses_anything_but_a_2d_map():
    with pytest.raises(ValueError, match="2-D"):
        gt.resize_depth(np.zeros((2, 3, 4, 5), np.float32), (3, 4))


# ---------------------------------------------------------------------------
# the reading


def _reading(scale="metric_claimed", processed=(283, 504)):
    ph, pw = processed
    cam = Camera.from_vfov(ph, pw, 61.0)
    k = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
    return gt.reading_from_arrays(
        source="da3",
        model=gt.DA3_MODEL,
        revision=SHA,
        depth=np.full((ph, pw), 2.5, np.float32),
        intrinsics=k,
        processed_hw=processed,
        input_hw=(1080, 1920),
        scale=scale,
    )


def test_a_reading_carries_the_vfov_at_the_processed_size_and_the_depth_at_the_plate():
    r = _reading()
    assert r.vfov_deg == pytest.approx(61.0, abs=1e-9)
    assert r.depth.shape == (1080, 1920)
    assert r.processed_hw == (283, 504)
    assert math.isfinite(r.aspect_drift) and abs(r.aspect_drift) < 0.01
    assert r.revision == SHA and r.model == gt.DA3_MODEL


def test_a_reading_refuses_an_unnamed_scale():
    """`metric_claimed` and `relative` are the two claims a caller can act on; a third
    string would let VGGT's depth reach the bench as metres by a typo."""
    with pytest.raises(ValueError, match="metric_claimed"):
        _reading(scale="metres")


def test_a_reading_refuses_intrinsics_that_are_not_3x3():
    with pytest.raises(ValueError, match="3x3"):
        gt.reading_from_arrays(
            source="vggt",
            model=gt.VGGT_MODEL,
            revision=SHA,
            depth=np.ones((4, 4), np.float32),
            intrinsics=np.ones((4,)),
            processed_hw=(4, 4),
            input_hw=(4, 4),
            scale="relative",
        )


def test_the_two_readers_are_the_two_models_named_at_the_top():
    assert set(gt.READERS) == {"da3", "vggt"}
