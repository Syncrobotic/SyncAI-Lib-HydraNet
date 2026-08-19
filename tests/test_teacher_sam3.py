"""The SAM 3 teacher's arithmetic, which no human pass will catch downstream.

Every number here ends up in a training set that is regenerated months apart, so a
silent change -- a mask composed under the wrong layer, a box that lands 3x off the
object it came from -- reaches the network before it reaches anyone's eyes.

`load_sam3` and the forward passes are not exercised: they need the `annotate` extra and
a 6.5 GB gated checkpoint. Everything between the model's output and the dataset is, with
`segment` and `vision_features` stubbed, which is where the arithmetic lives.

pytest tests/test_teacher_sam3.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.data.sam3_prompts import Concept
from syncai_hydranet.data.teachers import sam3
from syncai_hydranet.data.teachers.sam3 import (
    MAX_BOX_FRAC,
    _probe_image,
    compose,
    consensus,
    frame_boxes,
    frame_masks,
)
from syncai_hydranet.labels import IGNORE

TAXONOMY = {"void": 0, "floor": 1, "wall": 2, "person": 3}


def _concept(name: str, layer: int, prompts=("p",)) -> Concept:
    return Concept(name, list(prompts), layer, taxonomy=TAXONOMY)


def _claim(shape, box, score: float = 0.9) -> np.ndarray:
    y0, y1, x0, x1 = box
    out = np.zeros(shape, dtype=np.float32)
    out[y0:y1, x0:x1] = score
    return out


# --------------------------------------------------------------------------- compose


def test_a_higher_layer_wins_outright():
    """A person standing on a floor is a person; nobody would annotate it otherwise."""
    shape = (8, 8)
    floor, person = _concept("floor", 0), _concept("person", 2)
    out = compose(
        {"floor": _claim(shape, (0, 8, 0, 8)), "person": _claim(shape, (2, 6, 2, 6))},
        shape,
        [floor, person],
    )
    assert out[0, 0] == TAXONOMY["floor"]
    assert out[4, 4] == TAXONOMY["person"]


def test_the_same_layer_disagreeing_becomes_ignore_and_not_a_guess():
    """`table` vs `display table` is a judgement about the shop, not a race to write."""
    shape = (8, 8)
    a, b = _concept("floor", 1), _concept("wall", 1)
    out = compose(
        {"floor": _claim(shape, (0, 8, 0, 8)), "wall": _claim(shape, (0, 4, 0, 8))},
        shape,
        [a, b],
    )
    assert (out[0:4] == IGNORE).all()
    assert (out[4:8] == TAXONOMY["floor"]).all()


def test_two_prompts_of_one_class_are_a_union_not_a_conflict():
    """Eight `display_fixture` prompts each find a different fixture and all mean one id."""
    shape = (8, 8)
    c = _concept("floor", 1, prompts=("a", "b"))
    claims = {"floor": np.maximum(_claim(shape, (0, 4, 0, 8)), _claim(shape, (4, 8, 0, 8)))}
    out = compose(claims, shape, [c])
    assert (out == TAXONOMY["floor"]).all()


def test_a_pixel_no_prompt_claimed_is_ignore_and_not_void():
    """ "Nobody looked here" is not the class `void`; training them as one is the bug."""
    shape = (4, 4)
    out = compose({}, shape, [_concept("floor", 1)])
    assert (out == IGNORE).all()


# ------------------------------------------------------------------------- consensus


def test_consensus_keeps_only_what_the_frames_agree_on():
    """Disagreement across frames of a fixed camera is an error signal that costs nothing."""
    stable = np.full((4, 4), 1, dtype=np.uint8)
    frames = [stable.copy() for _ in range(10)]
    for i in range(4):  # one corner pixel that flips in 4 of 10 frames
        frames[i][0, 0] = 2
    modal, labelled = consensus(frames, 0.9)
    assert modal[0, 0] == IGNORE
    assert (modal[1:] == 1).all()
    assert labelled == pytest.approx(15 / 16)


def test_consensus_reports_the_share_it_kept():
    frames = [np.full((2, 2), 1, np.uint8), np.full((2, 2), 2, np.uint8)]
    modal, labelled = consensus(frames, 0.9)
    assert (modal == IGNORE).all()
    assert labelled == 0.0


# ----------------------------------------------------------------- the upscaled probe


def test_probe_image_is_the_identity_at_1x():
    img = Image.new("RGB", (32, 24))
    assert _probe_image(img, 1.0) is img


def test_probe_image_upscales_both_axes():
    got = _probe_image(Image.new("RGB", (32, 24)), 3.0)
    assert got.size == (96, 72)


# --------------------------------------------------- frame_masks / frame_boxes wiring


@pytest.fixture
def stub_model(monkeypatch):
    """Return one mask covering a named rectangle of the *probe* image, per prompt."""
    calls: list[dict] = []

    def fake_vision_features(_proc, _model, _image, _device):
        return "embeds"

    def fake_segment(_proc, _model, image, prompt, _min_score, device, _embeds=None):
        calls.append({"prompt": prompt, "size": image.size, "device": device})
        mask = np.zeros((image.height, image.width), dtype=bool)
        # 0.1 of each axis is 1% of the frame: over MIN_BOX_PIXELS, under the 2% cut.
        frac = {"tiny": 0.02, "big": 0.9}.get(prompt, 0.1)
        h = max(1, int(image.height * frac))
        w = max(1, int(image.width * frac))
        mask[:h, :w] = True
        return [(mask, 0.8)]

    monkeypatch.setattr(sam3, "vision_features", fake_vision_features)
    monkeypatch.setattr(sam3, "segment", fake_segment)
    return calls


def test_frame_masks_returns_the_source_image_not_the_probe(stub_model):
    """The mask is the probe's size; the image saved beside it must be the original."""
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    img, mask = frame_masks(
        None, None, frame, [_concept("floor", 1)], device="cpu", upscale=2.0
    )
    assert img.size == (60, 40)
    assert mask.shape == (80, 120)
    assert stub_model[0]["size"] == (120, 80)


@pytest.mark.usefixtures("stub_model")
def test_frame_boxes_scales_coordinates_back_to_the_saved_image():
    """`--upscale 3` must still produce boxes that land on the JPEG actually written."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    at_1x, _ = frame_boxes(None, None, frame, [("thing", ["p"])], device="cpu")
    at_3x, _ = frame_boxes(None, None, frame, [("thing", ["p"])], device="cpu", upscale=3.0)
    assert at_1x[0]["bbox"] == pytest.approx(at_3x[0]["bbox"], abs=1.0)
    assert at_3x[0]["bbox"][2] < 300 and at_3x[0]["bbox"][3] < 200


@pytest.mark.usefixtures("stub_model")
def test_frame_boxes_drops_a_speck():
    """A 3-pixel box trains a detector to fire on noise."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    boxes, dropped = frame_boxes(None, None, frame, [("thing", ["tiny"])], device="cpu")
    assert boxes == []
    assert not dropped  # too small is not the oversize counter


@pytest.mark.usefixtures("stub_model")
def test_frame_boxes_counts_what_the_oversize_cut_discarded():
    """`product box` fires on a whole display counter; the summary must be able to say so."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    boxes, dropped = frame_boxes(
        None, None, frame, [("thing", ["big"])], device="cpu", max_box_frac=MAX_BOX_FRAC
    )
    assert boxes == []
    assert dropped["thing"] == 1


@pytest.mark.usefixtures("stub_model")
def test_frame_boxes_dedupes_two_prompts_that_found_one_object():
    """One class, two prompts, one object -- one row, or the detector learns it is two."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    boxes, _ = frame_boxes(None, None, frame, [("thing", ["p", "q"])], device="cpu")
    assert len(boxes) == 1
