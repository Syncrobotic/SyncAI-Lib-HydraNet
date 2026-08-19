"""Grounding DINO as a teacher: `person` boxes on a store frame.

The `person` teacher of record, and the reason is measured (2026-08-19, on the exact
frames SAM 3 had already labelled): day scores sit at or above 0.35 -- Taichung-cam01
median 0.61, Kaohsiung-cam04 94 of SAM 3's 103 -- while an empty night IR clip tops out
at **0.326**. Zero boxes at 0.35 against SAM 3's 229 at its own 0.50 default, a day:night
separation of 11-49x against SAM 3's 0.9x. The night population never reaches the
threshold, so unlike the SAM 3 path this one needs no daylight gate.

`PERSON_THRESHOLD` is that gap, not a tuned number, and it belongs to one clip's worth of
night evidence -- `docs/journal/2026-08-19-night-person-fleet-recheck.md` re-measured the
fleet and found 13 of 42 cameras putting a box over it on a shuttered store. What survives
that is the static veto in `data/night_person.py`, not a different threshold here.

`transformers` is imported inside `load_gdino` and nowhere else, so a base install stays
importable.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

MODEL_ID = "IDEA-Research/grounding-dino-base"

# The measured day/night gap; see the module docstring. A working threshold, not a floor:
# a run that wants both score populations visible passes something far lower and splits
# them afterwards.
PERSON_THRESHOLD = 0.35


def load_gdino(model_id: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.inference_mode()
def detect(proc, model, img: Image.Image, prompt: str, floor: float, device: str) -> np.ndarray:
    """Boxes as [x1, y1, x2, y2, score] above the floor, xyxy in source pixels."""
    text = prompt.strip().lower().rstrip(".") + "."
    inputs = proc(images=img, text=text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    kwargs = {"threshold": floor, "text_threshold": floor, "target_sizes": [img.size[::-1]]}
    try:
        (res,) = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, **kwargs
        )
    except TypeError:  # older signature without positional input_ids
        (res,) = proc.post_process_grounded_object_detection(outputs, **kwargs)
    boxes = res["boxes"].cpu().float().numpy().reshape(-1, 4)
    scores = res["scores"].cpu().float().numpy().reshape(-1, 1)
    return np.concatenate([boxes, scores], axis=1)
