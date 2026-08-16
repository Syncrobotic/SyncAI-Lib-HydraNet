"""A tracker's mistakes are silent, and in analytics they are also cumulative.

Every failure this file pins produces a plausible number rather than an error: an id
that splits doubles footfall, an id that swaps corrupts a path, a track confirmed on one
frame turns noise into a shopper. None of it raises. Hence tests.

pytest tests/test_analytics.py -v
"""

import numpy as np
import pytest

from syncai_hydranet.analytics import Tracker, dwell_table, track_ground_path
from syncai_hydranet.analytics.dwell import ground_map
from syncai_hydranet.analytics.tracker import iou
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def walk(x0, y0, dx, dy, n, w=40, h=100):
    """A box walking in a straight line, one row per frame."""
    return [
        np.array([x0 + i * dx, y0 + i * dy, x0 + i * dx + w, y0 + i * dy + h]) for i in range(n)
    ]


# ------------------------------------------------------------------------ iou


def test_iou_of_a_box_with_itself_is_one():
    b = np.array([[0, 0, 10, 10]], dtype=float)
    assert iou(b, b)[0, 0] == pytest.approx(1.0)


def test_disjoint_boxes_score_zero():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[20, 20, 30, 30]], dtype=float)
    assert iou(a, b)[0, 0] == 0.0


def test_iou_handles_empty_input():
    assert iou(np.zeros((0, 4)), np.array([[0, 0, 1, 1]], dtype=float)).shape == (0, 1)


# -------------------------------------------------------------------- identity


def test_one_walker_keeps_one_id():
    """The failure this prevents inflates footfall in direct proportion to how often
    it happens, and nothing downstream can tell a split track from two shoppers."""
    tr = Tracker(min_hits=3)
    for i, b in enumerate(walk(0, 0, 5, 0, 20)):
        tr.update(np.array([b]), i)
    done = tr.finished()
    assert len(done) == 1, "one person walked across the frame and was counted twice"
    assert len(done[0].frames) == 20


def test_two_separated_walkers_keep_two_ids():
    tr = Tracker(min_hits=3)
    a, b = walk(0, 0, 5, 0, 20), walk(0, 400, 5, 0, 20)
    for i in range(20):
        tr.update(np.stack([a[i], b[i]]), i)
    assert len(tr.finished()) == 2


def test_a_gap_shorter_than_max_age_does_not_split_the_track():
    """Occlusion behind a fixture is the common case in a shop, not an edge case."""
    tr = Tracker(min_hits=3, max_age=5)
    boxes = walk(0, 0, 5, 0, 20)
    for i, b in enumerate(boxes):
        tr.update(np.zeros((0, 4)) if 8 <= i <= 10 else np.array([b]), i)
    assert len(tr.finished()) == 1


def test_a_gap_longer_than_max_age_does_split_it():
    """The other side of the same setting, so max_age cannot be raised without cost."""
    tr = Tracker(min_hits=3, max_age=2)
    boxes = walk(0, 0, 5, 0, 30)
    for i, b in enumerate(boxes):
        tr.update(np.zeros((0, 4)) if 10 <= i <= 20 else np.array([b]), i)
    assert len(tr.finished()) == 2


# ---------------------------------------------------------------- over-counting


def test_a_one_frame_blip_is_never_confirmed():
    """The single largest source of over-counting: a false positive that lasts one
    frame becomes a shopper if min_hits is 1."""
    tr = Tracker(min_hits=3)
    tr.update(np.array([[0, 0, 40, 100]], dtype=float), 0)
    for i in range(1, 10):
        tr.update(np.zeros((0, 4)), i)
    assert tr.finished() == []


def test_min_hits_one_would_have_counted_it():
    """States the cost of the default rather than assuming it: with min_hits=1 the same
    blip is a visit. Under-count over over-count is a choice, and this is it."""
    tr = Tracker(min_hits=1)
    tr.update(np.array([[0, 0, 40, 100]], dtype=float), 0)
    for i in range(1, 10):
        tr.update(np.zeros((0, 4)), i)
    assert len(tr.finished()) == 1


# --------------------------------------------------------------------- dwell


def test_dwell_is_measured_in_seconds_of_span_not_frames_observed():
    """A track that coasted through an occlusion was still present. Counting only the
    observed frames would shorten every dwell by however much the shop occludes."""
    tr = Tracker(min_hits=3, max_age=5)
    for i, b in enumerate(walk(0, 0, 5, 0, 20)):
        tr.update(np.zeros((0, 4)) if 8 <= i <= 10 else np.array([b]), i)
    row = dwell_table(tr.finished(), fps=10.0, last_frame=19)[0]
    assert row["dwell_s"] == pytest.approx(2.0)
    assert row["coasted"] == 3


def test_a_track_alive_at_the_last_frame_is_flagged_truncated():
    """Its dwell is a lower bound. Averaging it in silently makes clip length a
    parameter of the reported dwell."""
    tr = Tracker(min_hits=3)
    for i, b in enumerate(walk(0, 0, 5, 0, 20)):
        tr.update(np.array([b]), i)
    assert dwell_table(tr.finished(), fps=10.0, last_frame=19)[0]["truncated"] is True


# --------------------------------------------------------------- ground plane


def test_a_path_is_projected_to_metres_and_moves_away_as_the_box_rises():
    cam = Camera.from_vfov(1080, 1920, 55.0)
    plane = GroundPlane(height=2.4, pitch=0.7)
    tr = Tracker(min_hits=1)
    for i, b in enumerate(walk(900, 800, 0, -30, 5)):  # walking away: box rises
        tr.update(np.array([b]), i)
    path = track_ground_path(tr.finished()[0], cam, plane)
    assert path.shape == (5, 2)
    finite = path[np.isfinite(path).all(axis=1)]
    assert len(finite) >= 2
    assert np.all(np.diff(finite[:, 1]) > 0), "walking away must increase range"


def test_a_foot_point_above_the_horizon_is_nan_not_a_huge_distance():
    """`pixel_to_ground` refuses to turn a ray that misses the floor into a number,
    and this package must not undo that by dropping the row."""
    cam = Camera.from_vfov(1080, 1920, 55.0)
    plane = GroundPlane(height=2.4, pitch=0.05)  # nearly level: the top of frame misses
    tr = Tracker(min_hits=1)
    tr.update(np.array([[900, 0, 940, 10]], dtype=float), 0)
    assert not np.isfinite(track_ground_path(tr.finished()[0], cam, plane)).all()


# ------------------------------------------------------------------ ground map


def test_the_map_counts_track_frames_per_cell_and_reports_area():
    paths = [np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])]
    gm = ground_map(paths, cell_m=0.5)
    assert gm.cells.sum() == 3
    assert gm.visited_m2 == pytest.approx(2 * 0.25)
    assert gm.busiest(1)[0][2] == 2


def test_nan_positions_do_not_reach_the_map():
    paths = [np.array([[0.0, 0.0], [np.nan, np.nan]])]
    assert ground_map(paths, cell_m=0.5).cells.sum() == 1


def test_an_empty_input_is_an_empty_map_not_a_crash():
    assert ground_map([], cell_m=0.5).cells.sum() == 0
