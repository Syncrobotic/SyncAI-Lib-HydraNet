"""The counter-label pipeline's pure parts: the window, the merge, the mapping back.

pytest tests/test_counter_person_labels.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "counter_person_labels", ROOT / "tools/annotation/counter_person_labels.py"
)
assert _spec is not None and _spec.loader is not None
cpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpl)


def test_the_densest_window_lands_on_the_huddle_and_stays_inside_the_frame():
    rng = np.random.default_rng(0)
    huddle = rng.normal([1500, 300], [60, 40], size=(200, 2))
    scatter = rng.uniform([0, 0], [1920, 1080], size=(40, 2))
    x0, y0, x1, y1 = cpl.densest_window(np.vstack([huddle, scatter]), (1920, 1080), (960, 600))
    assert (x1 - x0, y1 - y0) == (960, 600)
    assert x0 >= 0 and x1 <= 1920 and y0 >= 0 and y1 <= 1080
    assert x0 <= 1500 <= x1 and y0 <= 300 <= y1
    assert cpl.densest_window(np.zeros((0, 2)), (1920, 1080), (960, 600)) == (
        480,
        240,
        1440,
        840,
    )


def test_inside_the_window_sam3_is_the_teacher_and_gdino_survives_only_where_unseen():
    gd = np.array([[100, 100, 200, 300, 0.6], [700, 100, 800, 300, 0.5]])
    sam = np.array(
        [[102, 98, 198, 302, 0.9], [400, 100, 500, 300, 0.8], [405, 104, 498, 296, 0.7]]
    )
    kept, survive = cpl.merge_window(gd, sam, merge_iou=0.3)
    assert len(kept) == 2 and sorted(kept[:, 4].tolist()) == [0.8, 0.9]  # NMS'd
    assert survive.tolist() == [False, True]  # under a SAM 3 box; unseen by SAM 3
    kept, survive = cpl.merge_window(gd, np.zeros((0, 5)), 0.3)
    assert len(kept) == 0 and survive.tolist() == [True, True]
    kept, survive = cpl.merge_window(np.zeros((0, 5)), sam, 0.3)
    assert len(kept) == 2 and len(survive) == 0


def test_the_crop_is_the_window_plus_a_margin_inside_the_frame():
    assert cpl.padded((460, 0, 1420, 600), (1920, 1080), 0.15) == (316, 0, 1564, 690)
    assert cpl.padded((0, 0, 960, 600), (1920, 1080), 0.15) == (0, 0, 1104, 690)


def test_boxes_come_back_from_the_upscaled_crop_to_frame_pixels():
    window = (460, 0, 1420, 600)
    crop_boxes = np.array([[100, 50, 300, 450, 0.9], [1900, 1190, 1920, 1200, 0.9]])
    out = cpl.to_frame(crop_boxes, window, upscale=2.0)
    assert len(out) == 1  # the 10x5 sliver at the crop's edge is dropped
    assert out[0, :4].tolist() == [510, 25, 610, 225]


def test_in_window_uses_the_box_centre():
    boxes = np.array([[0, 0, 100, 100], [900, 500, 1100, 700.0]])
    assert cpl.in_window(boxes, (50, 50, 1000, 600)).tolist() == [True, False]
