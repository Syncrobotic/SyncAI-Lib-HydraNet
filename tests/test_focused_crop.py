"""Focused detector views must preserve visible boxes and partial-supervision geometry."""

import copy
import random

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.data.transforms import (
    FocusedLetterboxScaleCrop,
    LetterboxScaleCrop,
    Sample,
    invert_geom,
)


def sample(anchor):
    boxes = np.array([anchor, [80, 40, 110, 65], [170, 75, 195, 95]], np.float32)
    labels = np.array([1, 9, 8])
    masks = {}
    rgb = np.zeros((100, 200, 3), np.uint8)
    for box, label in zip(boxes, labels, strict=True):
        x0, y0, x1, y1 = box.astype(int)
        mask = np.zeros((100, 200), np.uint8)
        mask[y0:y1, x0:x1] = 1
        masks[str(label)] = mask
        rgb[y0:y1, x0:x1, 0] = 255
    masks["negative_phone"] = np.zeros((100, 200), np.uint8)
    masks["negative_phone"][45:55, 120:140] = 1
    return Sample(image=Image.fromarray(rgb), boxes=boxes, labels=labels, masks=masks)


@pytest.mark.parametrize(
    "anchor",
    [[5, 5, 25, 25], [175, 5, 195, 25], [5, 75, 25, 95], [175, 75, 195, 95], [90, 40, 110, 60]],
)
def test_anchor_preserved_and_every_mask_tracks_clipped_boxes(monkeypatch, anchor):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    original = sample(anchor)
    for seed in range(8):
        random.seed(seed)
        result = FocusedLetterboxScaleCrop((100, 200), [1], (1, 1))(copy.deepcopy(original))
        assert result["geom"][:2] == (2, 2)
        box = result["boxes"][result["labels"] == 1][0]
        assert (box[:2] >= 2).all() and (box[2:] <= [198, 98]).all()
        np.testing.assert_allclose(invert_geom(box[None], result["geom"])[0], anchor)
        for box, label in zip(result["boxes"], result["labels"], strict=True):
            yy, xx = np.where(result["masks"][str(label)] == 1)
            np.testing.assert_allclose(box, [xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
        negative = result["masks"]["negative_phone"]
        assert set(np.unique(negative)) <= {0, 1, 255}
        yy, xx = np.where(negative == 1)
        if len(xx):
            mapped = invert_geom(
                np.array([[xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]]), result["geom"]
            )[0]
            assert (mapped[:2] >= [120, 45]).all()
            assert (mapped[2:] <= [140, 55]).all()


def test_no_eligible_device_preserves_original_rng_and_augmentation():
    original = sample([5, 5, 25, 25])
    random.seed(123)
    plain = LetterboxScaleCrop((100, 200), (0.9, 1.1))(copy.deepcopy(original))
    random.seed(123)
    focused = FocusedLetterboxScaleCrop((100, 200), [99], (0.9, 1.1))(original)
    np.testing.assert_array_equal(plain["image"], focused["image"])
    np.testing.assert_array_equal(plain["boxes"], focused["boxes"])
    for key in plain["masks"]:
        np.testing.assert_array_equal(plain["masks"][key], focused["masks"][key])


def test_oversized_anchor_falls_back_before_mutating(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    s = sample([0, 0, 200, 100])
    result = FocusedLetterboxScaleCrop((100, 200), [1], (1, 1))(s)
    np.testing.assert_array_equal(result["boxes"][0], [0, 0, 200, 100])
    assert result["geom"] == (1, 1, 0, 0)


def test_padding_stays_unknown_in_every_negative_channel(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    s = Sample(
        image=Image.new("RGB", (20, 200)),
        boxes=np.array([[5, 80, 15, 100]], np.float32),
        labels=np.array([1]),
        masks={k: np.ones((200, 20), np.uint8) for k in ["empty", "phone", "tablet"]},
    )
    result = FocusedLetterboxScaleCrop((100, 200), [1], (1, 1))(s)
    for mask in result["masks"].values():
        assert (mask == 255).any()
        assert (mask == 1).sum() == 2000
        assert set(np.unique(mask)) == {1, 255}
