"""The dense head vouching for the box head, and the cases where it must not."""

import numpy as np
import pytest

from syncai_hydranet.serving.decode import (
    MIN_PERSON_FRACTION,
    confirm_with_dense,
    person_pixel_fraction,
)

PERSON_LABEL = 0
BAG_LABEL = 1
PERSON_ID = 5  # `retail_surfaces` puts person last


def class_map(h=100, w=100, person_box=None):
    m = np.ones((h, w), dtype=np.int64)  # 1 = floor
    if person_box is not None:
        x0, y0, x1, y1 = person_box
        m[y0:y1, x0:x1] = PERSON_ID
    return m


def result(boxes, labels):
    return {
        "boxes": np.asarray(boxes, dtype=float),
        "scores": np.full(len(boxes), 0.5),
        "labels": np.asarray(labels, dtype=np.int64),
    }


def test_a_box_over_person_pixels_is_kept():
    m = class_map(person_box=(10, 10, 30, 70))
    out = confirm_with_dense(
        result([[10, 10, 30, 70]], [PERSON_LABEL]), m, PERSON_LABEL, PERSON_ID
    )
    assert len(out["labels"]) == 1


def test_a_box_over_empty_floor_is_dropped():
    m = class_map()
    out = confirm_with_dense(
        result([[10, 10, 30, 70]], [PERSON_LABEL]), m, PERSON_LABEL, PERSON_ID
    )
    assert len(out["labels"]) == 0


def test_only_person_is_judged_because_only_person_has_a_dense_channel():
    """`bag`, `boxed_stock` and `device` have no class in `retail_surfaces`. Judging them
    against a map that cannot represent them would delete every one of them."""
    m = class_map()
    out = confirm_with_dense(
        result([[10, 10, 30, 70], [40, 40, 60, 60]], [PERSON_LABEL, BAG_LABEL]),
        m, PERSON_LABEL, PERSON_ID,
    )  # fmt: skip
    assert list(out["labels"]) == [BAG_LABEL]


def test_the_default_keeps_a_shopper_whose_legs_are_behind_a_counter():
    """The population this exists for: the dense head sees the visible upper body only,
    so the fraction is well under 1 and must still clear the bar."""
    m = class_map(person_box=(10, 10, 30, 40))  # upper half of a 10..70 box
    frac = person_pixel_fraction([10, 10, 30, 70], m, PERSON_ID)
    assert 0.45 < frac < 0.55
    assert frac >= MIN_PERSON_FRACTION


def test_every_field_is_filtered_together():
    m = class_map(person_box=(0, 0, 20, 20))
    out = confirm_with_dense(
        result([[0, 0, 20, 20], [50, 50, 70, 70]], [PERSON_LABEL, PERSON_LABEL]),
        m, PERSON_LABEL, PERSON_ID,
    )  # fmt: skip
    assert len(out["boxes"]) == len(out["scores"]) == len(out["labels"]) == 1


def test_an_empty_frame_survives_and_the_input_is_not_modified():
    m = class_map()
    r = result(np.zeros((0, 4)), np.zeros(0, dtype=np.int64))
    assert len(confirm_with_dense(r, m, PERSON_LABEL, PERSON_ID)["labels"]) == 0
    r2 = result([[10, 10, 30, 70]], [PERSON_LABEL])
    confirm_with_dense(r2, m, PERSON_LABEL, PERSON_ID)
    assert len(r2["labels"]) == 1


@pytest.mark.parametrize("box", ([50, 50, 50, 60], [-20, -20, -5, -5], [200, 200, 260, 260]))
def test_a_degenerate_or_offscreen_box_scores_zero_rather_than_raising(box):
    assert person_pixel_fraction(box, class_map(), PERSON_ID) == 0.0
