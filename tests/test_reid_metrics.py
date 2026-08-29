"""Identity metrics, against sequences whose right answer is countable by eye.

The point of hand-built fixtures here rather than a fixed regression number: every one of
these has an answer you can work out on paper, so a wrong implementation is visible rather
than merely different from last time.

pytest tests/test_reid_metrics.py -v
"""

import numpy as np
import pytest

from syncai_hydranet.analytics.reid_metrics import cmc_map, id_switches, idf1

BOX = np.array([10.0, 10.0, 30.0, 50.0])


def track(frames, box=BOX, dx=0.0):
    """One track: {frame: box}, optionally drifting so IoU stays high but boxes differ."""
    return {f: box + np.array([dx * i, 0, dx * i, 0]) for i, f in enumerate(frames)}


# ------------------------------------------------------------------------- IDF1


def test_a_perfect_tracker_scores_one():
    gt = {1: track(range(10))}
    assert idf1(gt, {7: track(range(10))})["idf1"] == pytest.approx(1.0)


def test_splitting_one_person_into_two_tracks_halves_idf1():
    """The measured failure, at its smallest. Ten frames of one person, cut in two.

    Detection is perfect and every box is right; only the identity is broken. MOTA would
    charge one switch out of ten frames and still read ~0.9, which is why IDF1 is the
    headline here: the best one-to-one mapping can claim only one of the two halves.
    """
    gt = {1: track(range(10))}
    pred = {1: track(range(5)), 2: track(range(5, 10))}
    r = idf1(gt, pred)
    assert r["idtp"] == 5 and r["idfn"] == 5 and r["idfp"] == 5
    assert r["idf1"] == pytest.approx(0.5)


def test_the_assignment_is_optimal_not_greedy():
    """IDF1 is defined as the *best* possible mapping, so a greedy pass understates it.

    A real trap rather than a plausible-looking one -- the first version of this test
    paired two tracks where greedy and optimal agree, so it proved nothing and failed on
    my own arithmetic instead. Here `A` overlaps both identities on 6 frames and `B`
    overlaps only identity 1, on 5:

        gt1 x A = 6    gt1 x B = 5
        gt2 x A = 6    gt2 x B = 0

    Greedy takes a 6 first and strands the other identity on 0, totalling 6. The optimal
    pairing is gt1->B and gt2->A, totalling 11.
    """
    far = BOX + np.array([200.0, 0, 200.0, 0])
    gt = {1: track(range(10)), 2: track(range(20, 30), box=far)}
    pred = {
        "A": {**track(range(6)), **track(range(20, 26), box=far)},
        "B": track(range(5, 10)),
    }
    assert idf1(gt, pred)["idtp"] == 11, "greedy would score 6"


def test_a_tracker_that_reports_nothing_scores_zero_not_nan():
    r = idf1({1: track(range(10))}, {})
    assert r["idf1"] == 0.0 and r["idfn"] == 10


def test_no_ground_truth_is_nan_rather_than_a_perfect_score():
    """An empty label set must not read as success -- that is how an unlabelled clip
    ends up reported as a working tracker."""
    assert np.isnan(idf1({}, {1: track(range(5))})["idf1"])


def test_boxes_that_miss_do_not_count_as_the_same_identity():
    gt = {1: track(range(10))}
    far = {1: track(range(10), box=np.array([500.0, 500.0, 520.0, 540.0]))}
    assert idf1(gt, far)["idtp"] == 0


# ----------------------------------------------------------- switches and fragments


def test_fragmentation_counts_tracks_per_person():
    """`tracks_per_identity` is the direct reading of this project's problem: one clip,
    1234 tracks, and the number of real shoppers far below that."""
    gt = {1: track(range(12))}
    pred = {1: track(range(4)), 2: track(range(4, 8)), 3: track(range(8, 12))}
    r = id_switches(gt, pred)
    assert r["tracks_per_identity"] == pytest.approx(3.0)
    assert r["switches"] == 2, "two changeovers across three fragments"


def test_a_clean_track_has_one_track_per_identity_and_no_switches():
    r = id_switches({1: track(range(10))}, {5: track(range(10))})
    assert r["tracks_per_identity"] == pytest.approx(1.0)
    assert r["switches"] == 0
    assert r["mostly_tracked"] == 1


def test_mostly_tracked_needs_eighty_percent_from_one_track():
    """An average hides whether fragmentation is everywhere or concentrated in a few."""
    gt = {1: track(range(10))}
    assert id_switches(gt, {1: track(range(9)), 2: track([9])})["mostly_tracked"] == 1
    assert id_switches(gt, {1: track(range(5)), 2: track(range(5, 10))})["mostly_tracked"] == 0


def test_an_identity_nobody_tracked_contributes_zero_not_one():
    r = id_switches({1: track(range(10))}, {})
    assert r["tracks_per_identity"] == pytest.approx(0.0)
    assert r["mostly_tracked"] == 0


# -------------------------------------------------------------------- retrieval


def _toy():
    """Three identities on two cameras, features that already separate them."""
    qf = np.array([[1.0, 0], [0, 1.0], [1.0, 1.0]])
    qp, qc = np.array([1, 2, 3]), np.array([1, 1, 1])
    gf = np.array([[1.0, 0], [0, 1.0], [1.0, 1.0], [0.9, 0.1]])
    gp, gc = np.array([1, 2, 3, 1]), np.array([2, 2, 2, 1])
    return qf, qp, qc, gf, gp, gc


def test_perfect_features_give_rank1_and_map_of_one():
    r = cmc_map(*_toy())
    assert r["rank1"] == pytest.approx(1.0)
    assert r["mAP"] == pytest.approx(1.0)
    assert r["queries_scored"] == 3


def test_same_camera_gallery_entries_are_excluded():
    """The protocol, and a large over-estimate if skipped rather than a small bias.

    Query 1 has a near-identical gallery entry on its *own* camera. Counting it would make
    the benchmark "can you match an image to itself", and the resulting number would not
    be comparable to any published Market-1501 result.
    """
    qf, qp, qc, gf, gp, gc = _toy()
    gf[3] = qf[0]  # an exact copy of query 1, same camera
    assert cmc_map(qf, qp, qc, gf, gp, gc)["rank1"] == pytest.approx(1.0)
    # and it is genuinely excluded: make the cross-camera match wrong and rank1 must fall
    gf[0] = np.array([-1.0, 0.0])
    assert cmc_map(qf, qp, qc, gf, gp, gc)["rank1"] < 1.0


def test_distractors_stay_in_the_gallery():
    """`pid == -1` is Market-1501's junk class. It is never a correct answer, but it must
    still occupy rank slots -- dropping it inflates every number reported."""
    qf, qp, qc, gf, gp, gc = _toy()
    gf = np.vstack([np.array([[1.0, 0.02]]), gf])  # ranks above the true match
    gp, gc = np.concatenate([[-1], gp]), np.concatenate([[2], gc])
    r = cmc_map(qf, qp, qc, gf, gp, gc)
    assert r["rank1"] == pytest.approx(1.0), "a distractor is skipped, not counted wrong"


def test_an_identity_on_only_one_camera_is_not_scored():
    """It cannot be retrieved cross-camera, so scoring it would count a guaranteed miss
    against the model. Market-1501's protocol drops it; `queries_scored` says how many
    survived so a caller can see the denominator rather than assume it."""
    qf, qp, qc, gf, gp, gc = _toy()
    gp = np.array([1, 2, 99, 1])  # identity 3 no longer appears in the gallery
    r = cmc_map(qf, qp, qc, gf, gp, gc)
    assert r["queries_scored"] == 2


def test_features_are_normalised_by_the_metric():
    """A caller passing raw backbone output must get cosine similarity, not magnitudes."""
    qf, qp, qc, gf, gp, gc = _toy()
    a = cmc_map(qf, qp, qc, gf, gp, gc)
    b = cmc_map(qf * 37.0, qp, qc, gf * 0.013, gp, gc)
    assert a["mAP"] == pytest.approx(b["mAP"])
