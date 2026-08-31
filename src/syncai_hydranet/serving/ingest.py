"""Per-camera RTSP session policy: when a stream is dead, and what its frames are numbered.

The transport is not here. This is the half that decides things -- when a session that is
still connected has stopped being useful, how long to wait before trying again, and what
`seq` a frame carries into `BatchScheduler.offer`. A GStreamer pipeline, an ffmpeg pipe or
a test harness drives it; none of them is imported.

**The split is the point.** Decode throughput is already answered: PLAN 7 decision 8
measured NVDEC at ~7,800 f/s, about 520 streams, 5.4x the 96 this fleet needs. What is not
answered is session management, and the fleet's own census is why -- **six of 48 cameras
emit nothing at all, one stalls with the socket still open**, and the rest run 3-8 fps on
their own clocks. Those are the failures a decoder does not have an opinion about, and
they are all policy.

---------------------------------------------------------------------------
SEQ COMES FROM PTS, AND THE REASON RECONNECTS EXIST IS THE REASON THIS IS HARD

PLAN 2.3 states the rule: the time base is PTS, never a frame index, because every clip in
the corpus writes `30/1` into `r_frame_rate` regardless of its true variable rate. So a
counter cannot number these frames -- two cameras at 3 and 8 fps would produce the same
counter values for wildly different instants.

`BatchScheduler.offer` drops any frame whose `seq` is not greater than the one it holds.
That is correct and it is a trap: **RTSP timestamps restart near zero on a new session.**
A camera that reconnects would deliver frames numbered below the ones already in the
mailbox, and the scheduler would discard every one of them, for the life of the process,
without an error. The stream would be "connected" and contribute nothing -- which is
indistinguishable, from the outside, from the six cameras that genuinely emit nothing.

So a session's PTS is offset onto a monotonic timeline (:meth:`CameraSession.on_frame`),
and the offset is recomputed at each connect from the highest seq already issued. A
reconnect costs one frame period of headroom, never the stream.

---------------------------------------------------------------------------
"NEVER CONNECTED" IS NOT "DROPPED", AND THE FLEET MAKES THAT CONCRETE

Six cameras emit nothing. Treating them like a transient network blip means retrying every
few seconds forever and reporting them as unhealthy alongside a camera that is genuinely
flapping. They are a commissioning problem, not a runtime one, and the state machine says
so: :attr:`StreamHealth.ever_delivered` separates them, and the backoff for a stream that
has never produced a frame saturates at `max_backoff_s` rather than cycling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# One stalled RTSP session must not hold fifteen others hostage -- `scheduler.py`'s own
# tick semantics say so. This is the other half of that: a session whose socket is open
# and whose frames stopped is not detectable by the transport, only by a clock.
DEFAULT_STALL_S = 6.0
DEFAULT_BACKOFF_S = 1.0
DEFAULT_MAX_BACKOFF_S = 60.0


class State(Enum):
    """Where a camera is, as an operator would describe it."""

    IDLE = "idle"  # nothing attempted yet
    CONNECTING = "connecting"  # a session is being established
    LIVE = "live"  # frames are arriving
    STALLED = "stalled"  # connected, and no frame for longer than the stall window
    FAILED = "failed"  # the transport reported an error or EOS; waiting to retry


@dataclass
class StreamHealth:
    """What an operator, or the commissioning report, needs to tell these apart."""

    state: State = State.IDLE
    frames: int = 0
    reconnects: int = 0
    errors: int = 0
    last_frame_at: float | None = None
    last_error: str | None = None
    # A camera that has never produced a frame is a commissioning problem, not a network
    # one. Six of the fleet's 48 are in this state, and they should not be reported beside
    # a camera that is flapping.
    ever_delivered: bool = False


@dataclass
class CameraSession:
    """The policy for one camera. A transport calls in; this decides what happens.

    Holds no socket, no pipeline and no thread. Everything it knows about time arrives as
    an argument, so a test can run a week of flapping in microseconds and CI needs no
    GStreamer, no network and no camera.
    """

    camera: str
    stall_s: float = DEFAULT_STALL_S
    backoff_s: float = DEFAULT_BACKOFF_S
    max_backoff_s: float = DEFAULT_MAX_BACKOFF_S

    health: StreamHealth = field(default_factory=StreamHealth)
    # The monotonic timeline `offer` sees. Kept separate from the session's own PTS so a
    # reconnect cannot walk it backwards.
    _issued_seq: int = -1
    _pts_offset: int = 0
    _session_pts_seen: bool = False
    _retry_at: float | None = None
    _attempt: int = 0
    # When the current attempt started. A connect that never resolves is the same failure
    # as a live session going quiet, one stage earlier, and the socket cannot report it.
    _connecting_since: float | None = None

    # ---------------------------------------------------------------- transport events

    def on_connecting(self) -> None:
        """A new session is being established. Called before any frame from it."""
        self.health.state = State.CONNECTING
        # The next frame's PTS is from a fresh session and may be near zero; rebase it
        # above everything already issued. Deferred to the frame itself because the new
        # session's first PTS is not known yet.
        self._session_pts_seen = False

    def on_frame(self, now: float, pts_ns: int) -> int:
        """Record a frame's arrival and return the `seq` to hand `BatchScheduler.offer`.

        **The frame itself is not a parameter, and that is deliberate.** This decides a
        number; the caller passes the pixels to `offer`. A session that never touches a
        buffer cannot copy one, cannot hold one alive past its pool, and cannot become the
        place someone puts a colour conversion.

        The returned value is monotonic across reconnects. Within one session it is the
        camera's own PTS, so two frames an operator would call simultaneous carry the same
        distance apart whatever the wall clock did in between.
        """
        if not self._session_pts_seen:
            # +1 so the first frame of a new session strictly exceeds the last of the old
            # one even when a camera restarts its clock at exactly the same value.
            self._pts_offset = (self._issued_seq + 1) - pts_ns
            self._session_pts_seen = True
        seq = pts_ns + self._pts_offset
        # A camera that repeats or reorders inside one session still must not go
        # backwards; the scheduler would silently drop the rest of the stream.
        if seq <= self._issued_seq:
            seq = self._issued_seq + 1
            self._pts_offset = seq - pts_ns
        self._issued_seq = seq

        self.health.state = State.LIVE
        self.health.frames += 1
        self.health.last_frame_at = now
        self.health.ever_delivered = True
        self._attempt = 0
        self._retry_at = None
        return seq

    def on_error(self, now: float, message: str) -> None:
        """The transport reported an error or an unexpected EOS."""
        self.health.state = State.FAILED
        self.health.errors += 1
        self.health.last_error = message
        self._schedule_retry(now)

    # ---------------------------------------------------------------- policy questions

    def check_stalled(self, now: float) -> bool:
        """True if a session has gone quiet for longer than the stall window.

        Covers two stages, because they are the same failure: a LIVE session whose frames
        stopped, and a CONNECTING one that never resolved.

        The transport cannot answer this. An RTSP session whose camera has stopped sending
        is indistinguishable at the socket from one between frames, and the fleet has
        exactly one camera in that state. Only a clock separates them.
        """
        # A connect that never resolves is the same failure one stage earlier: the
        # transport is waiting on a socket that will not answer, and nothing times it out.
        # Found while removing an unused parameter, which is a poor way to find it.
        if self.health.state is State.CONNECTING:
            started = self._connecting_since
            if started is None or now - started <= self.stall_s:
                return False
            self.health.state = State.STALLED
            self._schedule_retry(now)
            return True
        if self.health.state is not State.LIVE:
            return False
        last = self.health.last_frame_at
        if last is None or now - last <= self.stall_s:
            return False
        self.health.state = State.STALLED
        self._schedule_retry(now)
        return True

    def should_reconnect(self, now: float) -> bool:
        """True when the backoff has elapsed and a new session should be started."""
        if self.health.state is State.IDLE:
            return True
        if self.health.state not in (State.FAILED, State.STALLED):
            return False
        return self._retry_at is not None and now >= self._retry_at

    def begin_reconnect(self, now: float) -> None:
        """Acknowledge that the caller is acting on :meth:`should_reconnect`."""
        if self.health.state is not State.IDLE:
            self.health.reconnects += 1
        self._connecting_since = now
        self.on_connecting()

    def _schedule_retry(self, now: float) -> None:
        # Doubling, capped. A stream that has never delivered saturates rather than
        # cycling: it is a commissioning problem and retrying it every second buys
        # nothing but log volume.
        delay = min(self.backoff_s * (2**self._attempt), self.max_backoff_s)
        self._attempt += 1
        self._retry_at = now + delay

    def retry_delay(self, now: float) -> float | None:
        """Seconds until the next attempt, or None when no retry is pending.

        Takes `now` rather than reading a clock, for the same reason nothing else here
        does: a session that cannot be driven from a test cannot be tested at the speed
        its failures actually need -- a week of flapping, in microseconds.
        """
        return None if self._retry_at is None else max(0.0, self._retry_at - now)


def fleet_report(sessions: dict[str, CameraSession]) -> dict[str, Any]:
    """One summary an operator can read, with the three failures kept apart.

    `never_delivered` is listed separately from `unhealthy` on purpose. A camera that has
    produced nothing since the process started is a commissioning fault -- the census found
    six of them -- and folding it into a live-error count means the number an operator
    watches never reaches zero and therefore stops being watched.
    """
    live = [c for c, s in sessions.items() if s.health.state is State.LIVE]
    never = [c for c, s in sessions.items() if not s.health.ever_delivered]
    unhealthy = [
        c
        for c, s in sessions.items()
        if s.health.ever_delivered and s.health.state is not State.LIVE
    ]
    return {
        "cameras": len(sessions),
        "live": sorted(live),
        "never_delivered": sorted(never),
        "unhealthy": sorted(unhealthy),
        "reconnects": sum(s.health.reconnects for s in sessions.values()),
        "frames": sum(s.health.frames for s in sessions.values()),
    }
