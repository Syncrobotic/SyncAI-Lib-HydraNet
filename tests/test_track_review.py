"""The two pieces of `track_review` that can be wrong without anything erroring.

pytest tests/test_track_review.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from track_review import apply_merges, tracks_to_json

from syncai_hydranet.analytics.reid_metrics import idf1


class FakeTrack:
    def __init__(self, tid, frames, boxes):
        self.track_id, self.frames, self.boxes = tid, frames, boxes


def test_the_proposal_is_already_the_metric_s_input_format():
    """The reason the format is not invented here: a conversion step is a second place
    for a frame index to go wrong, and a ground-truth file with different frame indexing
    from the metric does not error -- it scores badly and reads as a bad tracker."""
    box = np.array([1.0, 2.0, 3.0, 4.0])
    out = tracks_to_json([FakeTrack(7, [0, 1], [box, box])])
    # round-trips through JSON and lands in `idf1` without a converter
    gt = {
        int(k): {int(f): np.array(b) for f, b in v.items()}
        for k, v in json.loads(json.dumps(out)).items()
    }
    assert idf1(gt, gt)["idf1"] == pytest.approx(1.0)


def test_merging_fragments_is_what_the_ground_truth_is_for():
    """Three fragments of one shopper -> one identity, and IDF1 against the unmerged
    proposal then reports the fragmentation instead of hiding it."""
    box = np.array([0.0, 0.0, 10.0, 20.0])
    proposal = tracks_to_json(
        [
            FakeTrack(1, [0, 1, 2, 3], [box] * 4),
            FakeTrack(2, [4, 5, 6, 7], [box] * 4),
            FakeTrack(3, [8, 9, 10, 11], [box] * 4),
        ]
    )
    merged = apply_merges(proposal, [[1, 2, 3]])["tracks"]
    assert len(merged) == 1 and len(merged["1"]) == 12

    gt = {int(k): {int(f): np.array(b) for f, b in v.items()} for k, v in merged.items()}
    pred = {int(k): {int(f): np.array(b) for f, b in v.items()} for k, v in proposal.items()}
    # 12 gt frames, 12 predicted, best one-to-one claims one 4-frame fragment:
    # idtp 4, idfn 8, idfp 8 -> 8 / 24. (I first wrote 8/16 here, forgetting idfp; the
    # code was right and the expected value was not, for the second time in this file's
    # short life. Worth the arithmetic being spelled out rather than asserted.)
    assert idf1(gt, pred)["idf1"] == pytest.approx(1 / 3, abs=1e-9)


def test_an_empty_merge_list_is_a_valid_answer():
    """ "Nothing needed merging" and "nobody looked" must not produce the same file."""
    p = tracks_to_json([FakeTrack(1, [0], [np.zeros(4)])])
    assert apply_merges(p, [])["tracks"] == p


def test_a_frame_claimed_twice_within_a_group_is_counted_not_dropped_silently():
    """Two fragments overlapping in time is a detector duplicate, not a judgement. The
    merge keeps one box and *says* how often it happened, because a ground-truth file
    that quietly lost frames would understate every recall computed against it."""
    box = np.array([0.0, 0.0, 10.0, 20.0])
    p = tracks_to_json([FakeTrack(1, [0, 1], [box] * 2), FakeTrack(2, [1, 2], [box] * 2)])
    r = apply_merges(p, [[1, 2]])
    assert r["merge_collisions"] == 1
    assert sorted(r["tracks"]["1"]) == ["0", "1", "2"]


def test_ids_outside_any_group_survive_untouched():
    box = np.array([0.0, 0.0, 10.0, 20.0])
    p = tracks_to_json([FakeTrack(i, [i], [box]) for i in (1, 2, 3)])
    r = apply_merges(p, [[1, 2]])["tracks"]
    assert sorted(r) == ["1", "3"], "3 was in no group and must remain its own identity"
