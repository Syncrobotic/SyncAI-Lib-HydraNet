"""Retail people-flow analytics: tracking, dwell and ground-plane heatmaps.

**Post-processing, deliberately.** docs/ARCHITECTURE_REVIEW.md rules tracking out of the
network -- "Cross-frame post-processing. Putting it in the graph would break the
no-dynamic-control-flow property that makes TensorRT conversion work first time" -- and
this package is what that verdict implies rather than a departure from it. Nothing here
touches the exported graph. It reads the detection head's boxes frame by frame and adds
the one thing a per-frame model cannot have: identity over time.

Why analytics is a harder measurement than obstacle avoidance, which is the thing to
understand before reading any number this package produces:

    a robot asks   "is somebody there now"      -- instantaneous, self-correcting,
                                                   the next frame fixes a mistake
    analytics asks "how many distinct shoppers" -- an INTEGRAL, so errors accumulate
                                                   instead of averaging out

A tracker that splits one shopper into three tracks does not report a slightly noisy
footfall, it reports 300% of it. That asymmetry drives every default in `tracker.py`:
they are set to under-count rather than over-count, because an under-count is visible
against a manual audit and an over-count looks like a good day's trading.
"""

from .dwell import GroundMap, dwell_table, track_ground_path
from .tracker import Track, Tracker

__all__ = ["GroundMap", "Track", "Tracker", "dwell_table", "track_ground_path"]
