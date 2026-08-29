"""Security events: the second stage's *output* contract, and why none of it is a head.

`stage.py` names this stage and types what enters it (`StageFrame`). This types what
leaves it, for the security half of the deployment: a shopper entered a restricted zone,
too many people are at the counter, someone has stood in one place for four minutes, a
bag has been left, stock has gone off a shelf.

---------------------------------------------------------------------------
WHY THESE ARE NOT MODEL OUTPUTS, AND THE ARGUMENT IS ALREADY IN THIS REPOSITORY

`docs/PLAN.md` §5 refuses to put an answer in the weights that belongs in a config, and
refuses to let the rule layer reach into the model -- "four minutes is loitering" is an
argument. The retail doc that first made the case, against "walkable area within 5 m" as
a class, is at `git show b7457c2:docs/RETAIL.md`; its three reasons transfer to every
event here without modification:

1. **Annotators cannot label it consistently.** Where a restricted zone ends is a
   decision, not an appearance. Two annotators disagree by a metre and the model trains
   their average.
2. **It dies when the camera moves.** A zone drawn in pixels is wrong the moment the
   mount is knocked, and nothing in training reports it. A zone in metres on the floor
   survives, because the homography absorbs the change.
3. **It freezes a product parameter into the weights.** "Four minutes is loitering" and
   "six people is too many" are settings a store manager changes on a Tuesday. In a
   config they are a number; in a checkpoint they are a retraining run.

So every threshold below is an argument to a function, every zone is a polygon **in
metres on the ground plane**, and the model contributes exactly two things: boxes and a
terrain map.

---------------------------------------------------------------------------
WHAT AN EVENT CARRIES, AND WHAT IT DELIBERATELY DOES NOT

No score. These are threshold crossings on measured quantities, so an event carries the
`value` it measured and the `threshold` it crossed, and a reader can see how marginal it
was without a number that looks like a probability and is not one. A confidence here
would be invented, and this project has a standing rule against numbers whose production
nobody can name (`ARCHITECTURE.md` section 1).

`basis` names the instrument instead: which measurement produced this event. That is the
same rule applied to the event itself -- "before attributing a number to a mechanism,
check what produced the number".

---------------------------------------------------------------------------
WHAT IS NOT BUILT, AND IT IS NOT AN OVERSIGHT

`fall` and `fight` are in `EVENT_TYPES` because the schema is a contract with whatever
consumes these rows and adding a field later is more expensive than reserving one. They
have no producer here. Both need a temporal model over crops, and
`ARCHITECTURE.md`'s "what not to build" names the precondition nobody has met:
**confirm the behaviour occurs on these cameras before paying for the annotation.** MERL
Shopping is a grocery aisle of closed shelving; these are open display tables. A
plausible detector shipped unmeasured is the failure this project ranks worst.

`UNBUILT` below states the blocker per type, so a consumer that asks for one gets a
refusal naming what is missing rather than an empty list that reads like "no events".

---------------------------------------------------------------------------
THE PRECONDITION EVERY ROW HERE INHERITS

All of it stands on person tracks, and `RETAIL_DATA.md` measured what those are
worth today: a 4.6-minute clip fragments into **1234 tracks**. Occupancy counts tracks,
loitering measures how long one lasts, and an intrusion is one track crossing a line --
so fragmentation does not add noise to these numbers, it biases them, and in a known
direction. Nothing here fixes that; association does, and `reid_metrics.py` is where it
will be shown to have worked.

---------------------------------------------------------------------------
WHY THIS IS A PACKAGE, AND WHAT THAT DOES NOT CHANGE

One module of 1,425 lines, of which 788 were code. It was not incoherent -- its own
section rules split it into geometry helpers, the zone and line detectors, tier-1
behaviour and tier-2 pose -- and those rules are now the file boundaries, cut on the
lines they were already drawn on. Nothing moved between sections.

**The import surface is unchanged and is meant to stay that way.** Consumers reach this
as `from syncai_hydranet.analytics import events as ev` and then `ev.zone_events(...)`;
`scripts/mine_fall_candidates.py` and `tests/test_dispositions.py` import names off it
directly. Everything public is re-exported below, so both forms resolve exactly as they
did. `__all__` is the list of what a consumer may use -- the private helpers stay behind
their leading underscore in `_types` and `_geometry`, where the tier modules share them
instead of each keeping a copy.

One test had to change with it: `tests/test_stage_contract.py` reads the contract off the
source rather than restating it, and it found its two functions by parsing `ev.__file__`.
That is now a package `__init__`, so it parses the package directory. The test still reads
what the code does rather than what a second copy of the key list claims.
"""

# Private, and re-exported anyway: `scripts/pose_pilot.py` read `ev._torso` to plot the
# torso-angle distribution the fall detector would see. The script was measuring the
# instrument rather than calling it, which is a fair use of a private helper -- and the
# underscore still says it is not the contract, which is why it stays out of `__all__`.
# The pilot went in `500cdd2` once its verdict was recorded; `tools/pose/pose_overlay.py`
# is the live reader of `_torso`, so the export is still load-bearing rather than a
# leftover. `git show 500cdd2^:scripts/pose_pilot.py`.
from ._types import (
    CLIP_NAME,
    EVENT_TYPES,
    TIERS,
    UNBUILT,
    CountingLine,
    SecurityEvent,
    TrackSupport,
    Zone,
    clip_start_from_name,
    counting_lines,
    require_buildable,
    support_for,
    with_clip_start,
    zones_from_camera,
)
from .behaviour import crowd_events, fall_candidates, speed_events, tailgating_events
from .pose import (
    KEYPOINT_NAMES,
    KP,
    pose_posture_events,
    reach_to_shelf_events,
    require_keypoints,
)
from .pose import _torso as _torso
from .zones import (
    line_events,
    object_left_events,
    occupancy_events,
    stock_removed_events,
    zone_events,
    zone_stock_counts,
)

__all__ = [
    "CLIP_NAME",
    "EVENT_TYPES",
    "KEYPOINT_NAMES",
    "KP",
    "TIERS",
    "UNBUILT",
    "CountingLine",
    "SecurityEvent",
    "TrackSupport",
    "Zone",
    "clip_start_from_name",
    "counting_lines",
    "crowd_events",
    "fall_candidates",
    "line_events",
    "object_left_events",
    "occupancy_events",
    "pose_posture_events",
    "reach_to_shelf_events",
    "require_buildable",
    "require_keypoints",
    "speed_events",
    "stock_removed_events",
    "support_for",
    "tailgating_events",
    "with_clip_start",
    "zone_events",
    "zone_stock_counts",
    "zones_from_camera",
]
