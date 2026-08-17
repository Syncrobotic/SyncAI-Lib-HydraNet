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
is how `retail_flow.py` ended up with a private format. `StageFrame` below is that shape,
written down once.

---------------------------------------------------------------------------
WHY A TypedDict AND NOT A DATACLASS

The same reason `geometry/scene_types.py` gives for the scene payloads: this describes a
dict that already flows between functions, and a dataclass would require every producer to
construct it. A `TypedDict` is a statement about the dict that exists, checkable by `ty`,
costing nothing at runtime -- so it can be adopted incrementally and by parts rather than
in one commit that touches every caller.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class _StageFrameRequired(TypedDict):
    """Split from `StageFrame` only to mark one key optional on Python 3.10.

    `NotRequired` is 3.11+, and this project's venv is 3.10 -- it was tried first and
    fails at *import*, which `ty_ratchet.sh` does not catch because a type checker checks
    types rather than importing the module. Worth the two extra lines rather than a
    `typing_extensions` import in `src/`, which would put a dependency question in the way
    of a two-key distinction.
    """

    # Position in the clip, not a timestamp. Tracking, dwell and every metric here count
    # in frames and convert with `fps` at the edge, because a dropped frame changes the
    # index and does not change the wall clock -- and it is the index that decides whether
    # two boxes are adjacent observations or the same shopper seen twice.
    frame_index: int

    # The image the predictions came from, HxWx3 uint8 RGB. Required rather than optional
    # even though tracking does not read it: the crop encoder is the whole reason this
    # stage exists, and a payload a crop cannot be cut from would have to be redefined the
    # moment it is built. `utils/temporal.py` needs it too, for its change-gate.
    image: np.ndarray

    # Detection boxes in **image pixels, xyxy**, already mapped back through the letterbox.
    # Named as the model's own space rather than the panel's: the letterbox mapping is the
    # first stage's business and doing it twice is a class of bug this project has already
    # paid for once.
    boxes: np.ndarray  # [N, 4] float32
    scores: np.ndarray  # [N] float32
    labels: np.ndarray  # [N] int64, indices into `class_names`

    # The vocabulary those label indices mean. Carried per frame rather than assumed
    # global, because `--detection-classes` narrows an exported head and a narrowed engine
    # keeps only its binding names -- so a consumer that assumes COCO's 80 silently
    # renames every box against a `retail_analytics` build. Same failure
    # `coco_subsets.head_order` exists to prevent, one process boundary later.
    class_names: tuple[str, ...]


class StageFrame(_StageFrameRequired, total=False):
    """One frame's worth of first-stage output, as the second stage needs it.

    Deliberately **not** `HydraNet.predict`'s dict. That one is keyed by head name and
    shaped by what the network happens to have; this is keyed by what a consumer asks for,
    and the difference is the point -- a stage that reads the network's dict directly is
    coupled to which heads a config declared, and every retail config declares a different
    set.

    Everything in `_StageFrameRequired` is mandatory; the keys below are not.
    """

    # Terrain class ids, HxW. Optional because a detection-only config has no terrain head
    # and tracking does not need one -- but `dwell.ground_map` and the free-space
    # derivation in `cli/scene.py` do, and a missing key is a better error than a plausible
    # empty mask.
    terrain: np.ndarray
