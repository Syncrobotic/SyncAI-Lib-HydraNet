"""Keypoints ride the same geometry as boxes -- checked, because nothing else would.

A keypoint that misses one transform site arrives at the loss in the wrong coordinate
frame and renders its Gaussian on the wrong cell; training still converges, to a model
that is confidently displaced. Each test pins one site.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from syncai_hydranet.data.transforms import (
    POSE_LR_SWAP,
    LetterboxResize,
    RandomHorizontalFlip,
    Sample,
    _paste,
    _resize,
)


def person(x, y, conf=0.9):
    kp = np.zeros((1, 17, 3), dtype=np.float32)
    kp[0, :, 0], kp[0, :, 1], kp[0, :, 2] = x, y, conf
    return kp


def test_resize_scales_keypoints_with_the_image():
    s = Sample(image=Image.new("RGB", (200, 100)), pose=person(100.0, 50.0))
    s = _resize(s, (50, 400))  # (H, W): x2 in x, x0.5 in y
    assert np.allclose(s["pose"][0, 0], [200.0, 25.0, 0.9])


def test_letterbox_puts_keypoints_where_the_pixels_went():
    s = Sample(image=Image.new("RGB", (1920, 1080)), pose=person(960.0, 540.0))
    s = LetterboxResize((640, 1120))(s)
    f = min(1120 / 1920, 640 / 1080)
    px, py = (1120 - round(1920 * f)) // 2, (640 - round(1080 * f)) // 2
    assert np.allclose(s["pose"][0, 0, :2], [960 * f + px, 540 * f + py], atol=1.0)


def test_flip_mirrors_x_and_swaps_left_right_identities():
    kp = np.zeros((1, 17, 3), dtype=np.float32)
    kp[0, 5] = [10.0, 20.0, 0.9]  # left_shoulder
    kp[0, 6] = [90.0, 20.0, 0.8]  # right_shoulder
    s = Sample(image=Image.new("RGB", (100, 50)), pose=kp)
    s = RandomHorizontalFlip(p=1.1)(s)
    assert np.allclose(s["pose"][0, 5], [100 - 90, 20.0, 0.8])  # left is the old right
    assert np.allclose(s["pose"][0, 6], [100 - 10, 20.0, 0.9])
    assert POSE_LR_SWAP[POSE_LR_SWAP[5]] == 5  # the swap is an involution


def test_paste_keeps_the_person_but_silences_offscreen_joints():
    kp = np.zeros((1, 17, 3), dtype=np.float32)
    kp[0, 0] = [10.0, 10.0, 0.9]
    kp[0, 1] = [90.0, 10.0, 0.9]
    s = Sample(image=Image.new("RGB", (100, 50)), pose=kp)
    s = _paste(s, (50, 100), -40, 0)  # shift left: joint 0 leaves the canvas
    assert s["pose"].shape[0] == 1
    assert s["pose"][0, 0, 2] == 0.0
    assert np.allclose(s["pose"][0, 1], [50.0, 10.0, 0.9])
