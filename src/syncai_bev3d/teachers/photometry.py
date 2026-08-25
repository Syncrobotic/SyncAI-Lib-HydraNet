"""Is this frame colour daylight or monochrome IR, decided on the pixels.

The clock cannot answer this and the filename must not: a store's own lights, an
overcast afternoon and a camera that switches to IR early all move the boundary, and
`slot_of()` guessing from a timestamp is how a night frame reaches a daylight-only
teacher. Both numbers here are properties of the pixels.

**Why it is worth a module.** `person` is the class every security event is keyed on,
and SAM 3 returns 14 instances at score >= 0.5 on an empty IR frame of Taichung-cam09
at 23:59 -- every one an accessory packet hanging on the pegboard wall. Nine of 48
cameras return something for `person` at midnight. The daylight gate is what keeps
those frames out of the teacher, so the test that decides it belongs where the wheel,
the type ratchet and the coverage floor reach it.

**The thresholds are measured, not chosen.** The first `min_chroma` was 6.0, picked out
of the air, and it rejected Kaohsiung-cam08 outright -- a bright white-walled store at
luma 108 whose chroma is only 3.90 because the store itself is grey and white. Sampled
over all 48 cameras at two time slots: 21 of 96 frames sit at **exactly 0.00** (an IR
frame is monochrome by construction, so its per-pixel channel spread collapses) and
every colour frame is **>= 2.13**. The two populations do not overlap and nothing lies
between them, so the threshold belongs in that gap. `ARCHITECTURE.md` rule 2 -- a
threshold is relative to a measured baseline, never an absolute.

This replaces two copies of one formula. `gdino_person_boxes.luma_chroma` reported the
pair and `sam3_person_boxes.is_daylight` gated on it, and the first one's docstring said
"same pixel test as sam3_person_boxes.is_daylight" -- a comment doing the job an import
should. `analytics/clip_tracks.py` is what happens when that arrangement is left alone:
four copies of one loop disagreed about lens correction, and the disagreement changed
which observations the tracker linked.
"""

from __future__ import annotations

import numpy as np

# Defaults for the gate, from the distribution quoted above. Named here rather than
# repeated at each call site: three scripts passed `40.0, 1.0` as literals, one of them
# with a comment pointing at the script that owned the reasoning.
MIN_LUMA = 40.0
MIN_CHROMA = 1.0


def luma_chroma(frame: np.ndarray) -> tuple[float, float]:
    """Mean brightness, and mean distance of a channel from its pixel's own mean.

    Chroma is measured per pixel rather than over the frame: a frame that is red on one
    side and blue on the other has a neutral global mean and is plainly colour, and a
    global measure would call it monochrome.
    """
    luma = float(frame.mean())
    chroma = float(np.abs(frame.astype(np.int16) - frame.mean(axis=2, keepdims=True)).mean())
    return luma, chroma


def is_daylight(
    frame: np.ndarray,
    min_luma: float = MIN_LUMA,
    min_chroma: float = MIN_CHROMA,
) -> bool:
    """Bright enough and colour enough to be a daytime frame.

    Both conditions are required. Darkness alone would pass a colour frame of a dim
    stockroom to a night path it does not belong on, and colour alone would pass a
    floodlit IR frame -- which is the one that matters, because IR is where the hanging
    packets score as people.
    """
    luma, chroma = luma_chroma(frame)
    return luma >= min_luma and chroma >= min_chroma
