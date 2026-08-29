"""One implementation of inference preprocessing.

There were four: byte-identical in bev_video and in two ROS scripts of the quadruped line
(removed 2026-08-19),
once more in infer_image with a letterbox switch, and a fifth expanded inline in
infer_video. Four chances to feed the network something it was not trained on, and the
symptom would be a model that merely looks worse in one tool than in another.

pytest tests/test_preprocess.py -v
"""

import numpy as np
import torch
from PIL import Image

from syncai_hydranet.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from syncai_hydranet.utils.visualize import preprocess


def test_letterbox_path_matches_the_hand_written_copies():
    """Byte-for-byte what bev_video and the two removed ROS scripts each did."""
    from syncai_hydranet.utils.visualize import letterbox

    img = Image.new("RGB", (100, 50), (30, 60, 90))
    size = (64, 80)

    expected_img, expected_region = letterbox(img.convert("RGB"), size)
    arr = np.asarray(expected_img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    expected = torch.from_numpy(arr.transpose(2, 0, 1))[None]

    tensor, canvas, region = preprocess(img, size)
    assert torch.equal(tensor, expected)
    assert region == expected_region
    assert np.array_equal(np.asarray(canvas), np.asarray(expected_img))


def test_without_letterbox_it_stretches_and_the_region_is_the_whole_frame():
    """infer_image's variant, which follows data.letterbox in the config."""
    img = Image.new("RGB", (100, 50), (30, 60, 90))
    tensor, canvas, region = preprocess(img, (64, 80), use_letterbox=False)
    assert tensor.shape == (1, 3, 64, 80)
    assert canvas.size == (80, 64)
    assert region == (0, 0, 80, 64)


def test_the_tensor_is_normalised_not_merely_scaled():
    """A copy that dropped the mean/std step would still produce the right shape, run
    without error, and quietly halve the model's accuracy."""
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    tensor, _, _ = preprocess(img, (16, 16))
    expected = torch.tensor(-IMAGENET_MEAN / IMAGENET_STD, dtype=torch.float32)
    assert torch.allclose(tensor[0, :, 0, 0], expected, atol=1e-6)


def test_greyscale_input_is_accepted():
    """Frames arrive from ffmpeg, ROS and PIL; only some of them are already RGB."""
    tensor, canvas, _ = preprocess(Image.new("L", (32, 32), 128), (16, 16))
    assert tensor.shape == (1, 3, 16, 16)
    assert canvas.mode == "RGB"
