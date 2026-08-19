"""Fixed-batch tick scheduler over N camera streams.

The serving shape decided in the bench (runs/bench_pro6000): fixed-batch engines,
batch 16, because batch 1 computes 1,156 f/s against the 1,440 target and 32/64 add
nothing over 16. So frames are not inferred as they arrive -- they are collected
into batch-sized *ticks*.

Tick semantics, the part the tests pin down:

* **A tick never blocks.** The fleet's cameras emit 3-8 fps on their own clocks and
  six of 48 emit nothing at all (the corpus census); one stalled RTSP session must
  not hold 15 others hostage. ``tick()`` returns whatever is fresh right now.
* **Late stream = skip.** A camera with no new frame since its last consumed one is
  simply absent from the tick. Its slot in the engine batch still runs (fixed-batch
  engines compute all 16 either way) -- the caller just does no post-processing for
  stale slots.
* **Freshest frame wins.** If decode outruns consumption, the mailbox keeps only the
  newest frame and counts the replacement as a drop. Serving is live: an old frame's
  only value is being recent, and a backlog queue would add latency exactly when the
  system is already behind.
* **Fairness by longest-waiting.** When more streams are fresh than the batch has
  slots, the cameras served longest ago go first, so a fast camera cannot starve a
  slow one out of every tick.

Thread-safe: readers call ``offer()`` from their own threads, one consumer calls
``tick()``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TickItem:
    """One fresh frame assigned to one slot of the engine batch."""

    slot: int
    camera: str
    seq: int
    frame: Any  # HWC uint8 ndarray in practice; the scheduler never looks inside


@dataclass
class _Mailbox:
    frame: Any = None
    seq: int = -1  # seq of the frame currently held
    consumed_seq: int = -1  # last seq handed out through tick()
    dropped: int = 0  # frames replaced before anyone consumed them
    delivered: int = 0  # frames handed out through tick()
    last_served_tick: int = -1  # for longest-waiting-first fairness
    # Cameras keep their batch slot across ticks so per-slot device state (nothing
    # today, CUDA-graph capture later) stays camera-stable where possible.
    slot: int = field(default=-1)


class BatchScheduler:
    """Collect the freshest frame per stream into fixed-size ticks."""

    def __init__(self, cameras: list[str], batch: int):
        if not cameras:
            raise ValueError("scheduler needs at least one camera")
        if len(set(cameras)) != len(cameras):
            raise ValueError("duplicate camera names")
        self.batch = int(batch)
        self._lock = threading.Lock()
        self._boxes: dict[str, _Mailbox] = {c: _Mailbox() for c in cameras}
        # With <= batch cameras every camera owns a fixed slot. With more cameras
        # than slots (the 96-stream case runs 6 ticks per frame period) slots are
        # assigned per tick, in item order.
        self._fixed_slots = len(cameras) <= self.batch
        if self._fixed_slots:
            for i, c in enumerate(cameras):
                self._boxes[c].slot = i
        self.ticks = 0

    def offer(self, camera: str, seq: int, frame: Any) -> None:
        """A reader thread delivers a decoded frame. Newer replaces older."""
        box = self._boxes[camera]  # KeyError for an unknown camera is the right error
        with self._lock:
            if seq <= box.seq:
                return  # out-of-order or duplicate delivery; the newer frame stays
            if box.seq > box.consumed_seq:
                box.dropped += 1  # the held frame was never consumed
            box.frame = frame
            box.seq = seq

    def tick(self) -> list[TickItem]:
        """Assemble the next batch. Returns 0..batch items, immediately."""
        with self._lock:
            fresh = [
                (name, box) for name, box in self._boxes.items() if box.seq > box.consumed_seq
            ]
            # Longest-waiting first; insertion order breaks ties deterministically.
            fresh.sort(key=lambda nb: nb[1].last_served_tick)
            items: list[TickItem] = []
            free_slots = iter(range(self.batch))
            for name, box in fresh[: self.batch]:
                slot = box.slot if self._fixed_slots else next(free_slots)
                items.append(TickItem(slot=slot, camera=name, seq=box.seq, frame=box.frame))
                box.consumed_seq = box.seq
                box.delivered += 1
                box.last_served_tick = self.ticks
            self.ticks += 1
            return items

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                name: {"delivered": b.delivered, "dropped": b.dropped, "last_seq": b.seq}
                for name, b in self._boxes.items()
            }
