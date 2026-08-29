"""The Grounding DINO teacher, whose two jobs are a text prompt and a box array.

Thin on purpose -- the model does the work -- but both of the thin parts have bitten:
the phrase Grounding DINO is given is not the phrase a caller passes (it must be lower
case and end in a period, or the model matches nothing), and the post-processing call
changed signature between transformers releases, which is a run that dies after the
checkpoint is loaded rather than before.

pytest tests/test_teacher_gdino.py -v
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from syncai_bev3d.teachers.gdino import MODEL_ID, PERSON_THRESHOLD, detect


class _Inputs(dict):
    input_ids = "ids"

    def to(self, _device):
        return self


class _Proc:
    """Records the text it was handed, and answers with one box."""

    def __init__(self, *, legacy_signature: bool = False):
        self.legacy_signature = legacy_signature
        self.text: str | None = None
        self.kwargs: dict = {}

    def __call__(self, images, text, return_tensors):  # noqa: ARG002
        self.text = text
        return _Inputs()

    def post_process_grounded_object_detection(self, outputs, *args, **kwargs):  # noqa: ARG002
        if self.legacy_signature and args:
            raise TypeError("post_process_grounded_object_detection() takes 2 positional args")
        self.kwargs = kwargs
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                "scores": torch.tensor([0.7]),
            }
        ]


def _model(**_kwargs):
    return "outputs"


def _detect(proc, prompt="person", floor=0.3):
    return detect(proc, _model, Image.new("RGB", (8, 8)), prompt, floor, "cpu")


def test_the_prompt_is_normalised_to_what_the_model_expects():
    """Lower case and one trailing period. A caller's "Person" matches nothing without it."""
    proc = _Proc()
    _detect(proc, "  Person  ")
    assert proc.text == "person."


def test_a_prompt_that_already_ends_in_a_period_does_not_get_two():
    proc = _Proc()
    _detect(proc, "person.")
    assert proc.text == "person."


def test_boxes_come_back_as_xyxy_with_the_score_appended():
    got = _detect(_Proc())
    assert got.shape == (1, 5)
    assert got[0, :4].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert got[0, 4] == pytest.approx(0.7)  # float32 from the model, not a rounded copy


def test_the_floor_is_passed_as_both_thresholds():
    """A box threshold without a matching text threshold returns phrases nothing scored."""
    proc = _Proc()
    _detect(proc, floor=0.42)
    assert proc.kwargs["threshold"] == 0.42
    assert proc.kwargs["text_threshold"] == 0.42


def test_target_sizes_is_height_width_and_not_the_pil_order():
    """`Image.size` is (w, h) and the model wants (h, w); swapping them silently skews boxes."""
    proc = _Proc()
    detect(proc, _model, Image.new("RGB", (32, 16)), "person", 0.3, "cpu")
    assert proc.kwargs["target_sizes"] == [(16, 32)]


def test_an_older_transformers_signature_still_answers():
    """The fallback path exists because this raises after the checkpoint is already loaded."""
    proc = _Proc(legacy_signature=True)
    assert _detect(proc).shape == (1, 5)


def test_the_measured_person_threshold_is_pinned():
    """0.35 is the gap between day scores and a night clip's 0.326 ceiling, not a knob."""
    assert PERSON_THRESHOLD == 0.35
    assert MODEL_ID == "IDEA-Research/grounding-dino-base"
