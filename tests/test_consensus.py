"""Frame consensus: what it measures, and the three ways it would quietly mislead.

The measurement itself is a vote count, so the tests that matter are not about the
arithmetic. They are about the cases where a plausible number comes back and means
something other than what the reader will assume:

* a person walking through, which makes an aisle look inconsistent unless excluded
* a denominator that silently changed, which makes two cameras look comparable
* a model that is confidently wrong on every frame, which scores perfectly

The last one cannot be fixed and is not a bug. It is tested so the property is written
down somewhere executable rather than only in a docstring.

pytest tests/test_consensus.py -v
"""

import numpy as np
import pytest

from syncai_hydranet.engine.consensus import CAVEAT, FrameConsensus

CLASSES = ["void", "floor", "wall", "column", "fixture", "product", "person"]


def _const(value: int, shape=(4, 5)) -> np.ndarray:
    return np.full(shape, value, dtype=np.int64)


def _run(preds, exclude=("person",), thr=0.9):
    fc = FrameConsensus(CLASSES, exclude=exclude)
    for p in preds:
        fc.update(p)
    return fc.result(agree_threshold=thr)


# --- the property that is not a bug ----------------------------------------


def test_a_model_that_is_always_wrong_scores_perfectly():
    """The whole caveat, executable. Every frame says `wall` about a floor; consensus is
    1.0. This is why the result carries CAVEAT and why the tool refuses to be read as
    accuracy."""
    res = _run([_const(2) for _ in range(10)])
    assert res.stable_share == 1.0
    assert res.per_class[2].mean_agreement == 1.0
    assert "not a quality metric" in res.to_dict()["caveat"]
    assert res.to_dict()["caveat"] == CAVEAT


def test_a_pixel_that_flickers_evenly_is_unstable():
    preds = [_const(1) if i % 2 else _const(2) for i in range(10)]
    res = _run(preds)
    assert res.stable_share == 0.0
    assert res.unstable_share == 1.0


def test_agreement_is_the_modal_share_not_the_last_answer():
    preds = [_const(1)] * 9 + [_const(2)]
    res = _run(preds, thr=0.9)
    assert res.per_class[1].mean_agreement == pytest.approx(0.9)
    assert res.stable_share == 1.0


# --- moving people, which is the failure that makes it a people-counter ----


def test_a_person_walking_through_does_not_make_the_floor_look_unstable():
    """The regression `274a4a08` named: without exclusion one shopper drags the aisle
    they walked down into `unstable`, and the metric measures footfall."""
    preds = []
    for i in range(10):
        p = _const(1, (4, 20))  # floor everywhere
        p[:, i % 5] = 6  # a person, one column of pixels, walking the near aisle
        preds.append(p)

    without = _run(preds, exclude=())
    assert without.unstable_share > 0.2, "expected the walked-over floor to look unstable"

    with_exclusion = _run(preds)
    assert with_exclusion.stable_share == 1.0
    assert with_exclusion.excluded_pixels == 20  # every pixel the person ever touched
    assert with_exclusion.eligible_pixels == 60  # the rest of the shop, still settled


def test_a_camera_with_nothing_left_to_measure_returns_nan_not_zero():
    """Peak trading, people across the whole frame: every pixel is excluded and there is
    nothing to be consistent about. 0.0 there reads as "completely unstable", the opposite
    end of the scale from what happened -- and a fleet mean over six cameras would take
    the 0.0 and report a finding. NaN refuses to be averaged, which is the point."""
    preds = []
    for i in range(5):
        p = _const(1)
        p[:, i % 5] = 6
        preds.append(p)  # after 5 frames the person has touched every column

    res = _run(preds)
    assert res.eligible_pixels == 0
    assert not res.measurable
    assert np.isnan(res.stable_share)
    assert np.isnan(res.unstable_share)
    assert res.to_dict()["measurable"] is False


def test_exclusion_removes_every_pixel_the_class_ever_touched_not_just_where_it_is_now():
    preds = [_const(1) for _ in range(6)]
    preds[0][0, 0] = 6  # a person stood here once, six frames ago
    res = _run(preds)
    assert res.excluded_pixels == 1
    assert res.eligible_pixels == 19
    assert not res.eligible[0, 0]


def test_excluding_a_class_the_taxonomy_does_not_have_is_refused():
    """A silently-ignored exclusion leaves people in the vote, which is the one thing
    exclusion exists to prevent."""
    with pytest.raises(ValueError, match="shopper"):
        FrameConsensus(CLASSES, exclude=("shopper",))


# --- the denominator, which has to travel with the number ------------------


def test_the_denominator_is_eligible_pixels_and_it_is_reported():
    preds = [_const(1) for _ in range(5)]
    preds[0][0, :] = 6  # a whole row is people at some point
    res = _run(preds)
    d = res.to_dict()["pixels"]
    assert d["total"] == 20
    assert d["excluded"] == 5
    assert d["eligible"] == 15
    assert d["eligible_share"] == pytest.approx(0.75)
    # Shares are of the eligible set, not of the frame -- the distinction that makes two
    # cameras comparable or not.
    assert sum(c.share for c in res.per_class) == pytest.approx(1.0)


def test_frame_count_is_reported_because_a_threshold_alone_does_not_say_how_strong():
    """0.9 over 6 frames and 0.9 over 600 are different claims and the number cannot tell
    them apart."""
    assert _run([_const(1)] * 6).to_dict()["frames"] == 6
    assert _run([_const(1)] * 60).to_dict()["frames"] == 60


# --- the per-class breakdown, which is the finding -------------------------


def test_unstable_composition_says_where_the_instability_sits():
    """A single mean would hide it. `274a4a08`'s run put 71.9% of unstable pixels on
    `fixture`, and that was the result -- not their 31.1% mean."""
    preds = []
    for i in range(10):
        p = _const(2)  # wall, always settled
        p[0, :] = 4 if i % 2 else 5  # one row flickers fixture/product
        preds.append(p)
    res = _run(preds)

    assert res.unstable_share == pytest.approx(5 / 20)
    # Both flickering classes are named, and they account for all of the instability.
    assert set(res.unstable_composition) <= {"fixture", "product"}
    assert sum(res.unstable_composition.values()) == pytest.approx(1.0)
    assert res.per_class[2].stable_share == 1.0  # the wall is untouched by their argument


def test_a_class_that_never_appears_reports_zero_rather_than_being_absent():
    """Same reason `metrics.jsonl` emits every class: a missing key and a zero read very
    differently, and only one of them says the model was asked."""
    res = _run([_const(1)] * 4)
    names = [c.name for c in res.per_class]
    assert names == CLASSES
    assert res.per_class[3].pixels == 0
    assert res.per_class[3].mean_agreement == 0.0


# --- refusing the inputs that would produce a plausible wrong number -------


def test_a_second_camera_in_the_same_accumulator_is_refused():
    """Two scenes averaged describe neither."""
    fc = FrameConsensus(CLASSES)
    fc.update(_const(1, (4, 5)))
    with pytest.raises(ValueError, match="not of the same scene"):
        fc.update(_const(1, (8, 10)))


def test_class_ids_outside_the_taxonomy_are_refused():
    """Scoring a 13-class prediction against a 7-class scheme returns a number, not an
    error, unless something checks."""
    fc = FrameConsensus(CLASSES)
    with pytest.raises(ValueError, match="outside the 7 declared classes"):
        fc.update(_const(11))


def test_result_before_any_frame_is_refused():
    with pytest.raises(ValueError, match="no frames"):
        FrameConsensus(CLASSES).result()


@pytest.mark.parametrize("thr", [0.0, -0.1, 1.5])
def test_an_impossible_threshold_is_refused(thr):
    fc = FrameConsensus(CLASSES)
    fc.update(_const(1))
    with pytest.raises(ValueError, match="agree_threshold"):
        fc.result(agree_threshold=thr)


def test_a_non_2d_prediction_is_refused():
    fc = FrameConsensus(CLASSES)
    with pytest.raises(ValueError, match="HxW"):
        fc.update(np.zeros((1, 4, 5), dtype=np.int64))
