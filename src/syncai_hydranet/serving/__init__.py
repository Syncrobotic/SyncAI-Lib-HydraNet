"""Multi-stream serving pipeline for the 96-stream x 15 fps target.

Increment 1 of the plan recorded in
docs/journal/2026-08-19-security-retail-teachers-and-methodology.md: the fp16 b16
engine computes 3,272 frames/s but synchronous fp32 H2D drags it to 1,553, so the
data path -- not the model -- is the frontier. The pieces here are shaped around
that measurement:

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
