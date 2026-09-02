"""Which trained run the tools ship, and which checkpoint inside it answers which question.

**Six places named the run and the best one was not among them.** Measured 2026-08-31:
`runs/hydranet_retail_security_b03_cw_xl-20260825-162131` beats the run every tool pointed
at on every *site* metric -- `terrain_mIoU/site_seg03` 0.6254 against 0.5652,
`detection_mAP50/site_boxes03` 0.3356 against 0.3020 -- while losing slightly on the two
web-dataset metrics. It had sat unpromoted since the day it finished. What that cost is
visible rather than abstract: on `dingpu-1f/test1` the old run's `best.pt` labels an entire
stone wall `floor` and this one does not, and the two floor masks agree to an IoU of 0.590.

Nothing here is a path a caller should retype. `stable_infer.py`, `onboard_camera.py`,
`demo_video.py`, `demo_gif.py`, `flicker_baseline.py` and `serve_pilot.py` each carried
their own copy of the string, so promoting a run meant finding all six and the next better
run would have drifted the same way.

**There is no single best checkpoint in a run, and this module refuses to pretend there
is.** `best.pt` is selected on one head's metric, so asking for "the" checkpoint is asking
a question with two answers. Measured on the shipped run, epoch 15 (`best.pt`) against
epoch 60 (`last.pt`):

===========================  ==========  ==========
metric                       best.pt     last.pt
===========================  ==========  ==========
terrain_mIoU/site_seg03      **0.6681**  0.6254
detection_mAP/coco_person    0.1447      **0.2022**
detection_mAP50/site_boxes03 0.3013      **0.3356**
===========================  ==========  ==========

`best.pt` is a **40% worse person detector**. A caller therefore names the head it is
judged on -- :func:`for_terrain` or :func:`for_detection` -- and gets the checkpoint that
run actually won on. `tools/commissioning/demo_video.py` had already worked this out for
itself and defaulted to `last.pt`; that reasoning is now in one place instead of one
comment.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The run every tool ships from. A timestamped directory rather than a stable name on
#: purpose: the name says when the weights were trained, and promoting a new run is an
#: edit here rather than a copy over the old one, so the previous run stays readable.
#:
#: **Promoted to person01 on 2026-09-02** (user's decision, PLAN 7c.20): it beats
#: `..._b03_cw_xl-20260825-162131` on every site metric and on the teacher-independent
#: `coco_person` (0.2157 against 0.2022), losing only ~0.003 mIoU on ADE20K.
SHIPPED_RUN = REPO / "runs/hydranet_retail_person01"

#: The config that trained it. Read from the run rather than from `configs/`, because a
#: config in `configs/` is what the *next* run will use and may already have moved.
SHIPPED_CONFIG = SHIPPED_RUN / "config.yaml"


def for_terrain() -> Path:
    """The checkpoint to use when the answer is judged on segmentation.

    For person01 the two saved checkpoints are the same model to three decimals --
    selection ran on `detection_mAP/site_person`, picked epoch 118, and the curve was
    flat: `last.pt` (epoch 120) reads `site_seg03` 0.6718 against `best.pt`'s 0.6716,
    and marginally >= on every other head in `selection.json`. So both questions get
    `last.pt`. The two-answers doctrine above still stands -- this run's answers happen
    to coincide, and a future run's may not; read its `selection.json` before assuming.
    """
    return SHIPPED_RUN / "last.pt"


def for_detection() -> Path:
    """The checkpoint to use when the answer is judged on finding people or objects.

    Figures, tracking, person counts, anything the detection head decides. `last.pt`,
    for the reason on :func:`for_terrain`: person01's saved checkpoints coincide, and
    `last.pt` is the marginally better of the two on every head.
    """
    return SHIPPED_RUN / "last.pt"
