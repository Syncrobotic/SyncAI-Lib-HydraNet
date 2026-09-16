"""Weak-class magnification keeps image/mask alignment and unknown pixels."""

import random

import numpy as np
from PIL import Image

from syncai_hydranet.data.transforms import Sample, SemanticFocusCrop


def test_focus_crop_retains_labelled_context_and_ignore_without_new_classes():
    mask = np.full((120, 180), 255, np.uint8)
    mask[20:100, 20:160] = 7
    mask[55:65, 85:95] = 10
    rgb = np.repeat(mask[..., None], 3, axis=2)
    random.seed(42)
    result = SemanticFocusCrop(probability=1, classes=(10,))(
        Sample(image=Image.fromarray(rgb), masks={"scene": mask})
    )
    cropped = result["masks"]["scene"]
    assert cropped.shape[0] < mask.shape[0] / 1.9
    assert np.array_equal(np.asarray(result["image"])[..., 0], cropped)
    assert (cropped == 10).sum() == 100
    assert set(np.unique(cropped)) <= {7, 10, 255}
    assert (cropped == 7).any()


def test_focus_crop_does_not_invent_targets_or_alter_full_frame_branch():
    mask = np.full((32, 48), 255, np.uint8)
    sample = Sample(image=Image.new("RGB", (48, 32)), masks={"scene": mask})
    assert SemanticFocusCrop(probability=1)(sample) is sample
    mask[10:12, 15:17] = 10
    assert SemanticFocusCrop(probability=0)(sample) is sample
