"""The RTSP session policy, driven at the speed its failures actually need.

`CameraSession` takes every clock reading as an argument and holds no socket, so a week of
flapping runs here in microseconds and none of this needs GStreamer, a network or a camera.
That is the reason the policy was split out of the transport rather than a consequence of
it: the failures worth guarding against on this fleet are a camera that stalls with its
socket open and a camera that reconnects, and neither is reachable from a decoder test.

The headline is `test_a_reconnect_does_not_silence_the_camera_forever`. Everything else
here is a supporting property.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from syncai_hydranet.serving.ingest import (
    CameraSession,
    State,
    fleet_report,
)
from syncai_hydranet.serving.scheduler import BatchScheduler

MS = 1_000_000  # nanoseconds in a millisecond, since PTS is ns


def _frame() -> np.ndarray:
    """A frame for the scheduler. The session never sees one -- it returns a number."""
    return np.zeros((4, 4, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# seq: the trap a reconnect sets


def test_seq_tracks_pts_within_one_session():
    """Not a counter. Two cameras at 3 and 8 fps must not produce the same numbers."""
    s = CameraSession("cam01")
    s.begin_reconnect(now=0.0)
    first = s.on_frame(now=0.0, pts_ns=1000 * MS)
    second = s.on_frame(now=0.2, pts_ns=1200 * MS)
    assert second - first == 200 * MS, "the gap between frames is the camera's, not ours"


def test_a_reconnect_does_not_silence_the_camera_forever():
    """RTSP restarts its clock; `offer` drops anything not strictly newer. Both are right.

    Put together and unhandled they are a silent, permanent loss: every frame after the
    first reconnect is discarded by the scheduler, the session reports itself connected,
    and the camera contributes nothing for the life of the process -- indistinguishable
    from the six cameras that genuinely emit nothing.

    This drives a real `BatchScheduler` rather than asserting on the numbers, because the
    numbers are only interesting if the scheduler accepts them.
    """
    sched = BatchScheduler(["cam01"], batch=1)
    s = CameraSession("cam01")

    s.begin_reconnect(now=0.0)
    for i in range(5):
        pts = (10_000 + i * 200) * MS  # a session that has been up a while
        sched.offer("cam01", s.on_frame(now=i * 0.2, pts_ns=pts), _frame())
        sched.tick()

    s.on_error(now=1.0, message="teardown")
    s.begin_reconnect(now=2.0)

    delivered = 0
    for i in range(5):
        pts = i * 200 * MS  # the new session starts near zero, as RTSP does
        sched.offer("cam01", s.on_frame(now=2.0 + i * 0.2, pts_ns=pts), _frame())
        delivered += len(sched.tick())

    assert delivered == 5, (
        "the scheduler dropped frames from the reconnected session -- seq walked backwards"
    )


def test_a_repeated_timestamp_inside_one_session_still_moves_forward():
    """A camera that repeats a PTS would otherwise stop the stream at that frame."""
    s = CameraSession("cam01")
    s.begin_reconnect(now=0.0)
    a = s.on_frame(now=0.0, pts_ns=500 * MS)
    b = s.on_frame(now=0.1, pts_ns=500 * MS)
    c = s.on_frame(now=0.2, pts_ns=400 * MS)  # reordered, older
    assert a < b < c


# ---------------------------------------------------------------------------
# stall: the failure the socket cannot report


def test_a_live_session_that_goes_quiet_is_stalled_not_healthy():
    """One camera in this fleet does exactly this, with the socket still open."""
    s = CameraSession("cam01", stall_s=6.0)
    s.begin_reconnect(now=0.0)
    s.on_frame(now=0.0, pts_ns=0)

    assert s.check_stalled(now=5.0) is False
    assert s.health.state is State.LIVE
    assert s.check_stalled(now=7.0) is True
    assert s.health.state is State.STALLED


def test_a_session_that_never_connected_is_not_reported_as_stalled():
    """Stalling is a thing that happens to a stream that was working."""
    s = CameraSession("cam01", stall_s=6.0)
    assert s.check_stalled(now=1000.0) is False
    assert s.health.state is State.IDLE


# ---------------------------------------------------------------------------
# backoff


def test_the_backoff_doubles_and_then_stops_doubling():
    """Capped, so a camera that will never answer costs a request a minute, not a flood."""
    s = CameraSession("cam01", backoff_s=1.0, max_backoff_s=8.0)
    delays = []
    now = 0.0
    for _ in range(6):
        s.on_error(now=now, message="refused")
        delays.append(s.retry_delay(now))
        now += 100.0
    assert delays == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_a_delivered_frame_clears_the_backoff():
    """A stream that recovers must not carry its old penalty into the next blip."""
    s = CameraSession("cam01", backoff_s=1.0, max_backoff_s=64.0)
    for i in range(4):
        s.on_error(now=float(i), message="refused")
    s.begin_reconnect(now=10.0)
    s.on_frame(now=10.0, pts_ns=0)
    s.on_error(now=11.0, message="refused again")
    assert s.retry_delay(now=11.0) == 1.0


def test_reconnect_is_offered_only_once_the_backoff_has_elapsed():
    s = CameraSession("cam01", backoff_s=5.0)
    s.begin_reconnect(now=0.0)
    s.on_frame(now=0.0, pts_ns=0)
    s.on_error(now=1.0, message="dropped")
    assert s.should_reconnect(now=3.0) is False
    assert s.should_reconnect(now=6.0) is True


def test_an_untouched_camera_is_offered_a_first_connection():
    assert CameraSession("cam01").should_reconnect(now=0.0) is True


# ---------------------------------------------------------------------------
# the fleet view


def test_a_camera_that_never_delivered_is_counted_apart_from_one_that_broke():
    """Six of 48 emit nothing, and folding them into an error count is how a number stops
    being watched -- it can never reach zero, so nobody looks at it."""
    silent = CameraSession("cam-silent")
    silent.begin_reconnect(now=0.0)
    silent.on_error(now=1.0, message="no route")

    flapping = CameraSession("cam-flap")
    flapping.begin_reconnect(now=0.0)
    flapping.on_frame(now=0.0, pts_ns=0)
    flapping.on_error(now=1.0, message="dropped")

    healthy = CameraSession("cam-ok")
    healthy.begin_reconnect(now=0.0)
    healthy.on_frame(now=0.0, pts_ns=0)

    report = fleet_report({"cam-silent": silent, "cam-flap": flapping, "cam-ok": healthy})
    assert report["live"] == ["cam-ok"]
    assert report["never_delivered"] == ["cam-silent"]
    assert report["unhealthy"] == ["cam-flap"], "a camera that worked and broke is not silent"


def test_reconnects_are_counted_but_the_first_connection_is_not():
    """A count that starts at one for every healthy camera is a count nobody can read."""
    s = CameraSession("cam01")
    s.begin_reconnect(now=0.0)
    assert s.health.reconnects == 0
    s.on_frame(now=0.0, pts_ns=0)
    s.on_error(now=1.0, message="dropped")
    s.begin_reconnect(now=2.0)
    assert s.health.reconnects == 1


# ---------------------------------------------------------------------------
# the whole fleet's worth of flapping, at no wall-clock cost


@pytest.mark.parametrize("cycles", [50])
def test_a_thousand_reconnects_never_walk_seq_backwards(cycles):
    """The property that matters held over more churn than a day of operation.

    Each cycle restarts the camera's clock at zero, which is what an RTSP session does.
    If the offset were ever recomputed wrongly the scheduler would start dropping, and
    the assertion below is the same one the scheduler makes internally.
    """
    s = CameraSession("cam01")
    seqs = []
    now = 0.0
    for _ in range(cycles):
        s.begin_reconnect(now=now)
        for i in range(20):
            seqs.append(s.on_frame(now=now, pts_ns=i * 200 * MS))
            now += 0.2
        s.on_error(now=now, message="dropped")
        now += 1.0

    assert len(seqs) == cycles * 20
    assert all(b > a for a, b in pairwise(seqs)), "seq went backwards"


def test_a_connect_that_never_resolves_is_stalled_too():
    """The same failure one stage earlier, and the socket reports neither.

    A transport waiting on a camera that accepts the TCP connection and then never
    answers RTSP sits in CONNECTING forever: no frame arrives, so the live-stall check
    never applies, and no error arrives, so the backoff never starts. The stream is
    simply absent, which is how it would be mistaken for one of the six that emit
    nothing.
    """
    s = CameraSession("cam01", stall_s=6.0)
    s.begin_reconnect(now=0.0)
    assert s.health.state is State.CONNECTING
    assert s.check_stalled(now=5.0) is False
    assert s.check_stalled(now=7.0) is True
    assert s.health.state is State.STALLED
    assert s.should_reconnect(now=100.0) is True, "and it must be retried, not abandoned"
