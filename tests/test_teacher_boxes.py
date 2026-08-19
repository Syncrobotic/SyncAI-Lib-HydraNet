"""The two greedy passes that decide how many objects a teacher says are there.

Each removes a duplicate the other cannot see -- `dedupe` across the prompts of one
frame, `drop_static` across the frames of one camera -- and both feed a detector
directly, so an off-by-one in either is a training set that teaches the wrong count.

The first test is the one that made this a module: a private scalar IoU sat in
`sam3_prelabel` next to `analytics.tracker.iou`, and the two were never compared.

pytest tests/test_teacher_boxes.py -v
"""

from __future__ import annotations

import numpy as np

from syncai_hydranet.analytics.tracker import iou
from syncai_hydranet.data.teachers.boxes import boxes_from_masks, dedupe, drop_static, nms


def _iou_xywh(a, b) -> float:
    """The scalar implementation `sam3_prelabel` carried, kept here as the oracle."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _box(cat: int, bbox, score: float) -> dict:
    return {"category_id": cat, "bbox": list(bbox), "score": score}


def test_the_shared_iou_answers_what_the_private_one_did():
    """One definition, and it has to agree with the definition it replaced.

    Random boxes rather than chosen ones: the cases that separate two IoU
    implementations are the degenerate ones -- zero area, no overlap, containment --
    and a hand-picked list is exactly where those go missing.
    """
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = rng.integers(0, 40, size=4).astype(float)
        b = rng.integers(0, 40, size=4).astype(float)
        got = iou(
            np.array([[a[0], a[1], a[0] + a[2], a[1] + a[3]]]),
            np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]]]),
        )[0, 0]
        assert got == np.float64(_iou_xywh(a, b)) or abs(got - _iou_xywh(a, b)) < 1e-12


def test_dedupe_collapses_two_prompts_that_found_one_object():
    """The 266-boxes-for-146-items failure, in miniature."""
    boxes = [_box(1, (10, 10, 20, 20), 0.9), _box(1, (11, 11, 20, 20), 0.7)]
    kept = dedupe(boxes)
    assert len(kept) == 1
    assert kept[0]["score"] == 0.9  # the confident one survives, not the first one


def test_dedupe_keeps_two_real_objects_of_the_same_class():
    boxes = [_box(1, (0, 0, 10, 10), 0.9), _box(1, (50, 50, 10, 10), 0.8)]
    assert len(dedupe(boxes)) == 2


def test_dedupe_never_compares_across_classes():
    """A person standing in front of a shelf is two objects in the same pixels."""
    boxes = [_box(1, (10, 10, 20, 20), 0.9), _box(2, (10, 10, 20, 20), 0.8)]
    assert len(dedupe(boxes)) == 2


def test_dedupe_output_does_not_depend_on_the_input_order_of_classes():
    a = [_box(3, (0, 0, 5, 5), 0.5), _box(1, (9, 9, 5, 5), 0.6)]
    assert [b["category_id"] for b in dedupe(a)] == [1, 3]
    assert [b["category_id"] for b in dedupe(a[::-1])] == [1, 3]


def test_nms_preserves_input_order_rather_than_score_order():
    """A file written next to its frame should be stable for a stable input."""
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0, 0.4],
            [50.0, 50.0, 60.0, 60.0, 0.9],
            [1.0, 1.0, 11.0, 11.0, 0.8],
        ]
    )
    kept = nms(boxes, 0.5)
    assert kept[:, 4].tolist() == [0.9, 0.8]  # index 1 then index 2, not sorted by score


def test_nms_is_a_no_op_below_two_boxes():
    one = np.array([[0.0, 0.0, 1.0, 1.0, 0.5]])
    assert nms(one, 0.5).shape == (1, 5)
    assert nms(np.zeros((0, 5)), 0.5).shape == (0, 5)


def test_boxes_from_masks_drops_an_empty_mask_rather_than_boxing_it():
    """A prompt that fired on nothing has no location; a zero-area target matches nothing."""
    full = np.zeros((10, 10), dtype=bool)
    full[2:5, 3:7] = True
    out = boxes_from_masks([(full, 0.9), (np.zeros((10, 10), dtype=bool), 0.8)])
    assert out.shape == (1, 5)
    assert out[0].tolist() == [3.0, 2.0, 7.0, 5.0, 0.9]  # xyxy, exclusive high edge


def test_drop_static_returns_both_sides():
    """It removed people the one time it was measured, so nothing may be deleted silently."""
    still = [0.0, 0.0, 10.0, 10.0, 0.9]
    per_frame = [
        np.array([still, [20.0, 20.0, 30.0, 30.0, 0.8]]),
        np.array([still, [40.0, 40.0, 50.0, 50.0, 0.8]]),
        np.array([still, [60.0, 60.0, 70.0, 70.0, 0.8]]),
    ]
    kept, dropped = drop_static(per_frame, iou_thr=0.7, share=0.5)
    assert [len(k) for k in kept] == [1, 1, 1]
    assert [len(d) for d in dropped] == [1, 1, 1]
    assert all(d[0][:4].tolist() == still[:4] for d in dropped)


def test_drop_static_at_share_zero_still_needs_two_frames_to_agree():
    """`need` floors at 2: one recurrence is a coincidence, and 0 would drop everything."""
    box = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])
    kept, dropped = drop_static([box, box.copy()], iou_thr=0.7, share=0.0)
    assert [len(k) for k in kept] == [1, 1]
    assert [len(d) for d in dropped] == [0, 0]
