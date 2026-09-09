"""The letterbox geometry, in the one place it is now defined.

`preprocessing.letterbox_region` and `preprocessing.undo_letterbox` were written three
and two times respectively before this: `visualize.letterbox` computed the forward
geometry from PIL sizes, `serve_pilot.letterbox_filter` recomputed it as an ffmpeg filter
string, `clip_tracks.to_source_pixels` held the inverse, and `track_review` open-coded a
second one. Copies of a geometry are not checkable against each other -- a one-pixel
disagreement moves every box by one pixel and nothing raises -- so what these tests hold
is that there is one definition and that its inverse is really its inverse.

The two source shapes are the two the fleet actually has: 1920x1080, and the one camera
the corpus census found **mounted sideways** (docs/PLAN.md section 4.1). They are not
interchangeable here, and that is the point: at 16:9 the canvas pad is 5 px tall, and
rotated it is 380 px wide.

pytest tests/test_letterbox_geometry.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.analytics.clip_tracks import to_source_pixels
from syncai_hydranet.preprocessing import letterbox_region, undo_letterbox
from syncai_hydranet.utils.visualize import letterbox

# The shipped engine's canvas: `exports/pro6000_1120/res_640x1120`, (H, W).
CANVAS_HW = (640, 1120)
LANDSCAPE = (1920, 1080)
ROTATED = (1080, 1920)


def forward(points_px: np.ndarray, src_wh, canvas_hw=CANVAS_HW) -> np.ndarray:
    """Source pixels -> canvas pixels, the way a letterboxing preprocessor puts them."""
    x0, y0, cw, ch = letterbox_region(*src_wh, canvas_hw)
    return np.asarray(points_px, float) * (cw / src_wh[0], ch / src_wh[1]) + (x0, y0)


def test_the_two_source_shapes_pad_in_different_axes():
    """The measurement the rest of this file rests on: the pad is not a small constant.

    A rounding-sized pad on 16:9 is why the wrong call is nearly right there, and a 380 px
    pad on the sideways camera is why "nearly right" does not survive the fleet.
    """
    assert letterbox_region(*LANDSCAPE, CANVAS_HW) == (0, 5, 1120, 630)
    assert letterbox_region(*ROTATED, CANVAS_HW) == (380, 0, 360, 640)


def test_visualize_letterbox_returns_the_region_this_function_computes():
    """`letterbox` is the forward operation and this is its geometry; if they disagree,
    every consumer that inverts the region is inverting a different letterbox."""
    for src_w, src_h in (LANDSCAPE, ROTATED, (640, 480), (1280, 720)):
        img = Image.new("RGB", (src_w, src_h))
        _, region = letterbox(img, CANVAS_HW)
        assert region == letterbox_region(src_w, src_h, CANVAS_HW)


@pytest.mark.parametrize("src_wh", [LANDSCAPE, ROTATED])
def test_a_point_survives_the_round_trip(src_wh):
    src_w, src_h = src_wh
    pts = np.array(
        [[src_w * 0.5, src_h * 0.6], [src_w * 0.2, src_h * 0.75], [0.0, 0.0]],
    )
    region = letterbox_region(src_w, src_h, CANVAS_HW)
    back = undo_letterbox(forward(pts, src_wh), region, src_w, src_h)
    # Sub-pixel: the region's content size is rounded to whole pixels, so the scale it
    # inverts is the rounded one. A tolerance larger than that would hide a sign error.
    assert back == pytest.approx(pts, abs=0.5)


def test_boxes_and_points_take_the_same_path():
    """A box is two points, and `to_source_pixels` is this function under its box name."""
    region = letterbox_region(*LANDSCAPE, CANVAS_HW)
    box = np.array([[100.0, 60.0, 340.0, 500.0]])
    corners = box.reshape(-1, 2)

    as_box = undo_letterbox(box, region, *LANDSCAPE)
    as_points = undo_letterbox(corners, region, *LANDSCAPE)
    assert as_box.reshape(-1, 2) == pytest.approx(as_points)
    assert to_source_pixels(box, region, *LANDSCAPE) == pytest.approx(as_box)


def test_a_shape_that_is_neither_points_nor_boxes_is_refused():
    """(N,3) is the shape a keypoint array has, and it must not be silently read as a
    box with its confidence column treated as a coordinate."""
    with pytest.raises(ValueError, match=r"\(N,2\) points or \(N,4\)"):
        undo_letterbox(np.zeros((4, 3)), (0, 5, 1120, 630), *LANDSCAPE)


def test_the_canvas_is_not_describable_by_a_size():
    """Why `canvas_region` is a region: the sideways camera's pad is entirely lateral, so
    scaling canvas pixels by any single (w, h) ratio cannot recover the source -- and the
    error it leaves is **zero on the centre column**, which is where a spot check looks.
    """
    src_w, src_h = ROTATED
    region = letterbox_region(src_w, src_h, CANVAS_HW)
    centre = np.array([[src_w * 0.5, src_h * 0.6]])
    off_axis = np.array([[src_w * 0.2, src_h * 0.75]])

    for pts in (centre, off_axis):
        canvas = forward(pts, (src_w, src_h))
        # The plausible wrong call: treat the canvas as if it were the frame filmed.
        by_size = canvas * (src_w / CANVAS_HW[1], src_h / CANVAS_HW[0])
        correct = undo_letterbox(canvas, region, src_w, src_h)
        err = float(np.abs(by_size - correct)[0, 0])
        if pts is centre:
            assert err < 1.0, "the centre column is where the wrong call looks right"
        else:
            assert err > 100.0, "and off-axis is where it is not"
