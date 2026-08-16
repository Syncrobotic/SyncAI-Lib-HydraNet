"""Where the network's "walkable" and the depth sensor's "I can see it" disagree.

The split is the whole point of the live view's right-hand panel. A traversability head
answers from appearance alone, so it will happily call a mirror, a glass storefront or a
polished floor open ground. The depth sensor answers from physics and returns nothing
from exactly those surfaces. Neither is wrong on its own; the interesting pixels are the
ones where they differ, and this is what names them.

That is also why the threshold lives here as an argument rather than inside the weights:
"further than I care about" is a property of the planner asking, not of the scene, so it
is a slider on a web page and not a training run.
"""

from __future__ import annotations

import numpy as np

# The traversability head's `go` class. The other two ids are blocked and caution.
GO = 2

# Reach ids, in the order REACH_COLORS paints them.
REACH_NONE = 0
REACH_WITHIN = 1
REACH_BEYOND = 2
REACH_NO_DEPTH = 3

# black / green / yellow / magenta.
#
# Magenta is the colour worth staring at: walkable by the model and returning no depth at
# all. Glass, mirrors and specular reflection off a polished floor all land there, and it
# is where this deployment's worst failure would appear first -- so it is given the one
# colour nothing else in the project uses rather than being blended into "unknown".
REACH_COLORS = np.array(
    [[0, 0, 0], [40, 220, 90], [250, 200, 40], [230, 60, 230]], dtype=np.uint8
)


def classify_reach(trav: np.ndarray, depth_m: np.ndarray, range_m: float) -> np.ndarray:
    """Split walkable pixels three ways against the depth frame.

    `trav` and `depth_m` must already be the same shape -- the caller has one at model
    resolution and one at camera resolution, and which of the two to resample is its
    decision, not this function's.

    Depth of 0 means "no return", not "at the camera". Treating it as a distance would
    put every glass panel in the nearest bucket, which is the reading this exists to
    prevent.
    """
    go = trav == GO
    valid = depth_m > 0
    reach = np.zeros_like(trav)
    reach[go & valid & (depth_m <= range_m)] = REACH_WITHIN
    reach[go & valid & (depth_m > range_m)] = REACH_BEYOND
    reach[go & ~valid] = REACH_NO_DEPTH
    return reach
