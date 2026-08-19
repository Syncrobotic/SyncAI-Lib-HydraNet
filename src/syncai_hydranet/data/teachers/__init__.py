"""Open-vocabulary teachers: the models that label site footage so nobody has to.

SAM 3 and Grounding DINO are not part of the network this project trains and ships. They
run once, offline, on a box with the `annotate` extra installed, and what they leave
behind is a dataset. Nothing under `serving/` or on the Orin imports any of this, and
neither does the wheel's install: every `transformers` import inside these modules is
inside the function that needs it, so a base install stays importable and an inference
box never carries a 6.5 GB checkpoint it will not run.

**Why they are in the package and not in `scripts/`, which is where they were written.**
A teacher's output is training data, so a change to one of these functions changes what
the network learns, and the pre-label runs are months apart -- there is no fast feedback
loop to catch a drift. `scripts/` sits outside the wheel, outside the type ratchet and
outside the coverage floor (`[tool.coverage.run] source` is `src/syncai_hydranet` and
nothing else), so the code with the longest feedback loop in the project was the code
with the least checking on it. It had already gone wrong twice by copying: four copies
of one tracking loop disagreed about lens correction (`analytics/clip_tracks.py`), and
one pixel test was written twice under two names (`photometry.py`).

`scripts/sam3_prelabel.py`, `scripts/gdino_person_boxes.py` and
`scripts/sam3_person_boxes.py` remain the command-line entry points. They keep their
argument parsing, their refusals and their run reports; what moved here is the part
another caller had a reason to import.
"""

from __future__ import annotations

from .boxes import boxes_from_masks, dedupe, drop_static, nms
from .photometry import MIN_CHROMA, MIN_LUMA, is_daylight, luma_chroma

__all__ = [
    "MIN_CHROMA",
    "MIN_LUMA",
    "boxes_from_masks",
    "dedupe",
    "drop_static",
    "is_daylight",
    "luma_chroma",
    "nms",
]
