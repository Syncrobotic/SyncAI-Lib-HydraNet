"""Tick semantics of the batching scheduler.

The contract under test is the serving decision recorded in the scheduler's
docstring: a tick never blocks, a late stream is skipped rather than waited for,
the freshest frame wins, and no frame is consumed twice.
"""

from __future__ import annotations

import pytest

from syncai_hydranet.serving.scheduler import BatchScheduler


def cams(n: int) -> list[str]:
    return [f"cam{i:02d}" for i in range(n)]


def test_tick_with_nothing_fresh_returns_empty_immediately():
    s = BatchScheduler(cams(4), batch=4)
    assert s.tick() == []


def test_late_stream_is_skipped_not_waited_for():
    s = BatchScheduler(cams(4), batch=4)
    s.offer("cam00", 1, "f0")
    s.offer("cam02", 1, "f2")
    items = s.tick()
    assert sorted(i.camera for i in items) == ["cam00", "cam02"]


def test_a_frame_is_consumed_exactly_once():
    s = BatchScheduler(cams(2), batch=2)
    s.offer("cam00", 1, "a")
    assert [i.camera for i in s.tick()] == ["cam00"]
    # No new frame arrived: the same frame must not be handed out again.
    assert s.tick() == []
    s.offer("cam00", 2, "b")
    assert [(i.camera, i.seq) for i in s.tick()] == [("cam00", 2)]


def test_freshest_frame_wins_and_the_replacement_counts_as_a_drop():
    s = BatchScheduler(cams(1), batch=1)
    s.offer("cam00", 1, "old")
    s.offer("cam00", 2, "new")
    items = s.tick()
    assert [(i.seq, i.frame) for i in items] == [(2, "new")]
    assert s.stats()["cam00"]["dropped"] == 1
    assert s.stats()["cam00"]["delivered"] == 1


def test_out_of_order_delivery_does_not_replace_a_newer_frame():
    s = BatchScheduler(cams(1), batch=1)
    s.offer("cam00", 5, "newer")
    s.offer("cam00", 3, "stale")
    assert [i.seq for i in s.tick()] == [5]
    # The stale frame was neither delivered nor counted as a drop of the newer one.
    assert s.stats()["cam00"]["dropped"] == 0


def test_slots_are_stable_per_camera_when_streams_fit_the_batch():
    s = BatchScheduler(cams(3), batch=16)
    for tick in range(3):
        for i in range(3):
            s.offer(f"cam{i:02d}", tick + 1, tick)
        slots = {i.camera: i.slot for i in s.tick()}
        assert slots == {"cam00": 0, "cam01": 1, "cam02": 2}


def test_fairness_when_more_streams_are_fresh_than_the_batch_has_slots():
    n, batch = 8, 4
    s = BatchScheduler(cams(n), batch=batch)
    for i in range(n):
        s.offer(f"cam{i:02d}", 1, i)
    first = {i.camera for i in s.tick()}
    assert len(first) == batch
    # Refresh everyone: the cameras skipped last tick must be served before the
    # cameras that already got a slot.
    for i in range(n):
        s.offer(f"cam{i:02d}", 2, i)
    second = {i.camera for i in s.tick()}
    assert second == set(cams(n)) - first


def test_unknown_camera_and_duplicate_names_fail_loudly():
    with pytest.raises(ValueError):
        BatchScheduler(["a", "a"], batch=2)
    s = BatchScheduler(["a"], batch=1)
    with pytest.raises(KeyError):
        s.offer("b", 1, None)
