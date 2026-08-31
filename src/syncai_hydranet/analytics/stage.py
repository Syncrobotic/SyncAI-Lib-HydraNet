"""The second stage: what it is, what enters it, and the contract its parts already share.

`HydraNet.forward` is a **single-frame pure convolution graph** -- its own docstring says
"this is exactly what gets exported to ONNX", and that property is why TensorRT conversion
works first time. Nothing that needs a second frame, or a region of interest cut from the
first, can live inside it.

Everything the retail analytics programme wants is therefore on this side of that line:
tracking, re-identification, dwell, temporal smoothing, and the crop encoder that will
carry attributes and action. **That is a system, and until this module it had no name.**
Four components of it, in three directories, sharing no stated interface:

    analytics/tracker.py       boxes           -> Track
    analytics/dwell.py         Track           -> dwell rows, ground occupancy
    analytics/reid_metrics.py  Track vs truth  -> IDF1, ID switches, CMC/mAP
    utils/temporal.py          frame + terrain -> smoothed terrain
    scripts/retail_flow.py     a clip          -> whatever it needed that day

The costs were paid before this file existed rather than predicted by it: `temporal.py`
reached 17% coverage owned by nobody and was deleted out of the tree by someone who
reasonably thought it was junk, and `retail_flow.py`'s output -- the only independent
answer to "was this pixel occupied at frame N" -- lived in a session scratchpad that
disappears with the session.

---------------------------------------------------------------------------
WHAT THIS MODULE DOES AND DOES NOT DO

It **names the boundary and types what crosses it**. It does not move any existing file.
A rename churns import paths for every consumer and settles nothing; what was missing was
never the directory, it was the statement of what the stage receives.

`Track` in `analytics/tracker.py` is **already** the contract on the output side, and it
already carries `frames` and `boxes`, which is why `reid_metrics.py` could consume it
without touching the tracker. That half needed documenting, not designing, and this module
does not restate it -- read `Track`.

What genuinely had no shape is the **input**: what one frame hands the stage. Every
consumer so far has reconstructed it from `HydraNet.predict`'s output dict by hand, which
is how `retail_flow.py` ended up with a private format. The types below are that shape,
written down once -- `StageFrame` for a producer that has everything, and the two narrower
ones for the halves a consumer actually reads. `StageFrame`'s docstring says why it is
three types rather than one, and the short version is that the one did not fit any of the
code it was written for.

---------------------------------------------------------------------------
WHY A TypedDict AND NOT A DATACLASS

The same reason `syncai_bev3d/scene_types.py` gives for the scene payloads: this describes a
dict that already flows between functions, and a dataclass would require every producer to
construct it. A `TypedDict` is a statement about the dict that exists, checkable by `ty`,
costing nothing at runtime -- so it can be adopted incrementally and by parts rather than
in one commit that touches every caller.
"""

# **No `from __future__ import annotations` in this module, and that is load-bearing.**
# It makes every annotation a string, and `TypedDict` does not resolve strings at class
# creation -- so `NotRequired[...]` becomes invisible and the key it marks comes out
# **required**. Measured here on 2026-08-31: with the future import, `StageFrame` reported
# seven required keys and none optional; without it, six and `terrain`. Nothing else in
# this file needs it -- `np.ndarray | None` is a runtime-valid expression on the 3.11
# floor -- and `test_stage_contract.py` holds the resulting key sets, which is the check
# that would have caught the mistake had it existed one revision earlier.
from typing import NotRequired, TypedDict

import numpy as np


class FrameRef(TypedDict):
    """The one key every second-stage payload carries, whatever else it holds.

    Position in the clip, not a timestamp. Tracking, dwell and every metric here count in
    frames and convert with `fps` at the edge, because a dropped frame changes the index
    and does not change the wall clock -- and it is the index that decides whether two
    boxes are adjacent observations or the same shopper seen twice.
    """

    frame_index: int


class BoxFrame(FrameRef):
    """What a consumer of *detections* needs. `events.zone_stock_counts` takes this.

    Boxes are in **image pixels, xyxy**, already mapped back through the letterbox. Named
    as the model's own space rather than the panel's: the letterbox mapping is the first
    stage's business and doing it twice is a class of bug this project has already paid
    for once.
    """

    boxes: np.ndarray  # [N, 4] float32
    scores: np.ndarray  # [N] float32
    labels: np.ndarray  # [N] int64, indices into `class_names`

    # The vocabulary those label indices mean. Carried per frame rather than assumed
    # global, because `--detection-classes` narrows an exported head and a narrowed engine
    # keeps only its binding names -- so a consumer that assumes COCO's 80 silently
    # renames every box against a `retail_analytics` build. Same failure
    # `coco_subsets.head_order` exists to prevent, one process boundary later.
    class_names: tuple[str, ...]


class TerrainFrame(FrameRef, total=False):
    """What a consumer of the *dense map* needs. `events.reach_to_shelf_events` takes this.

    Both keys are optional and the consumer says why for each: `terrain` because a
    detection-only config has no terrain head and the function raises a sentence about it
    rather than returning an empty list, and `image` because it is one of the two ways a
    payload can state which pixel space the map lives in (`image_size` is the other).
    """

    terrain: np.ndarray | None  # class ids, HxW
    image: np.ndarray  # HxWx3 uint8 RGB


class StageFrame(BoxFrame):
    """One frame's worth of first-stage output, as the second stage needs it.

    Deliberately **not** `HydraNet.predict`'s dict. That one is keyed by head name and
    shaped by what the network happens to have; this is keyed by what a consumer asks for,
    and the difference is the point -- a stage that reads the network's dict directly is
    coupled to which heads a config declared, and every retail config declares a different
    set.

    **Why this is three types and not one.** It was one, and the one did not fit. Measured
    across the module that declares it: `zone_stock_counts` reads five of its six required
    keys and never `image`; `reach_to_shelf_events` reads two, of which only
    `frame_index` is required here; and the only producer of a real payload
    (`scripts/pose_pilot.py`, since deleted in `500cdd2`) built `{frame_index, terrain}`,
    which satisfies none of it.
    A contract that no consumer needs in full and no producer can supply is not a
    contract -- it is a comment that type-checks, and annotating a signature with it would
    have made a working caller an error.

    So the split is by *what a consumer reads*, and this type is the full payload: a
    `StageFrame` is assignable wherever a `BoxFrame` or a `TerrainFrame` is asked for,
    because it has every key either requires. A producer that has everything states it
    with this; a producer that has a dense map and no boxes states `TerrainFrame` and is
    not lying about the rest.

    **This was two classes until 2026-08-31**, split only because `NotRequired` is 3.11+
    and this project's floor was 3.10. `requires-python` is `>=3.11` now, so the split has
    nothing left to do and the payload is one class again. The observation the split's
    docstring carried is kept, because the gap it names did not close with the floor:
    `NotRequired` on 3.10 fails at **import**, and `ty_ratchet.sh` does not catch that,
    because a type checker checks types rather than importing the module. What catches it
    is the test matrix's floor row -- which is why that row is `requires-python`'s floor
    and not whichever interpreter a contributor's venv resolved.
    """

    # Required here and optional in `TerrainFrame`, deliberately. A full first-stage
    # payload can always supply the image and the crop encoder is the whole reason this
    # stage exists, so a payload a crop cannot be cut from would have to be redefined the
    # moment it is built. `utils/temporal.py` needs it too, for its change-gate.
    image: np.ndarray

    terrain: NotRequired[np.ndarray | None]
