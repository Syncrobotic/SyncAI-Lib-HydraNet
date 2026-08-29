"""Multi-stream serving pipeline for the 96-stream target.

**96 streams at 5 fps = 480 frames/s** (PLAN 7.4, revised 2026-08-26). This file opened
with "96-stream x 15 fps" until 2026-08-27, which was the target when it was written and
the last place in the tree where the superseded 1,440 f/s still read as live. The stream
count is unchanged and is the part this package is built around; only the per-stream rate
moved, and it moved because every measurement this project makes of its own analytics
already runs at 5 fps and nothing downstream consumes 15.

Increment 1 of the plan recorded in
`git show b7457c2:docs/journal/2026-08-19-security-retail-teachers-and-methodology.md`:
the fp16 b16 engine computes 3,272 frames/s but synchronous fp32 H2D drags it to 1,553,
so the data path -- not the model -- is the frontier. That is why the pieces here are
shaped the way they are, and it is why the rate revision does not relax any of them: the
frontier was never the model's arithmetic.

* ``uint8_input``  -- graph surgery that moves cast+normalise into the graph so the
  host copies uint8 (4x less PCIe traffic).
* ``engine``       -- a TensorRT executor with pinned staging buffers and
  double-buffered, dual-stream H2D/compute overlap.
* ``scheduler``    -- fixed-batch ticks over N streams; a late stream is skipped,
  never waited for.
* ``camera``       -- per-camera state: label EMA, tracker, calibration handle,
  per-class working thresholds.
* ``decode``       -- host-side FCOS decode for the engine's raw det outputs.
"""

from .camera import DEFAULT_THRESHOLDS, CameraState, ClassThresholds
from .decode import FcosDecoder
from .scheduler import BatchScheduler, TickItem

__all__ = [
    "DEFAULT_THRESHOLDS",
    "BatchScheduler",
    "CameraState",
    "ClassThresholds",
    "FcosDecoder",
    "TickItem",
]
