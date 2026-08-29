"""One answer per shopper, not one per frame.

Measured on Kaohsiung-cam04 with the crop encoder: **the same staff member is labelled
`F` and `M` in adjacent frames.** A per-frame attribute is an observation, and a shopper
is not a per-frame quantity -- so the frame-level output is the wrong unit to report and
always was. This is where it becomes a track-level one.

Three things this does that a majority vote does not, each because of something measured:

**1. It pools log-odds rather than counting decisions.** A vote throws away how sure each
frame was, and the frames disagree precisely where the crops are hard -- a shopper turning
away, half behind a counter. Summing logits weights each observation by its own
confidence, which is the whole content of the disagreement.

**2. It drops crops that are not of a whole person -- and this bought nothing measurable,
which is recorded rather than quietly dropped.** 23 of the 48 `fall_candidate` spans in
today's sweep touch the frame border, so truncation is common on these cameras, and a
crop cut by the frame edge is a torso whose lower-body attributes are a guess. Measured on
Kaohsiung-cam04 the gender decision flips on **16.2%** of consecutive frame pairs, and
excluding truncated and tiny crops moves that to **16.3%** -- nothing. So truncation is not
what makes gender unstable. The filter stays, because `Trousers` and `boots` cannot be read
off a torso whatever the flicker does, and because the reason for it was never the flicker;
what does not stay is the claim that it fixes this.

**3. It reports agreement beside every answer, and on this footage that is the whole
result.** Pooling removes the flicker *by construction* -- one track, one answer -- so the
absence of flicker afterwards is not evidence of anything. What can be measured without
site labels is how much the frames agreed, and they often do not: over 72 tracks on
Kaohsiung-cam04 the mean agreement on `Female` is **82.8%**, the minimum is **38.5%**, and
only **68%** of tracks clear 80%. For roughly a third of shoppers the gender answer is
close to a coin flip, and a pipeline that prints a letter without that number is claiming
something it did not measure.

Whether the pooled answer is *correct* is not measurable here at all: no site crop carries
an attribute label. This makes the output stable and states its own confidence; it does not
make it true.

---------------------------------------------------------------------------
AND ONE CORRECTNESS FIX THAT ONLY EXISTS AT THIS LEVEL

The three age flags are mutually exclusive by construction -- a person is under 18, or
18-60, or over 60 -- but the head emits three independent sigmoids, so a single frame can
answer yes to two of them or no to all three. Pooled over a track the bands can be
resolved as an argmax over the three, which is what they always were. `age_band` below is
that, and it is not available per frame because a single frame's three logits are exactly
the evidence that failed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.attributes import AGE_BANDS


@dataclass(frozen=True)
class Attribute:
    """One attribute for one track: the answer, how sure, and what it stands on."""

    name: str
    probability: float
    agreement: float  # share of used frames whose own decision matched the pooled one
    frames: int

    @property
    def value(self) -> bool:
        return self.probability >= 0.5

    def as_row(self) -> dict:
        return {
            "name": self.name,
            "value": bool(self.value),
            "probability": round(float(self.probability), 4),
            "agreement": round(float(self.agreement), 4),
            "frames": int(self.frames),
        }


def usable_crops(
    boxes: np.ndarray, width: int, height: int, pad: int = 3, min_height: int = 64
) -> np.ndarray:
    """Which of a track's boxes are whole people large enough to read.

    Two rejections and they are different in kind. A box touching the frame edge is
    *truncated* -- part of the person is outside the image and no amount of resolution
    recovers it. A short box is *small* -- the person is there and there are not enough
    pixels. Both are excluded, and 64 px is well under the 244-336 px median these
    cameras produce, so it removes the far field rather than the normal case.
    """
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(b):
        return np.zeros(0, dtype=bool)
    whole = (
        (b[:, 0] > pad) & (b[:, 1] > pad) & (b[:, 2] < width - pad) & (b[:, 3] < height - pad)
    )
    return whole & ((b[:, 3] - b[:, 1]) >= min_height)


def pool(logits: np.ndarray, use: np.ndarray | None = None) -> np.ndarray:
    """Mean log-odds over the used frames. Returns [A] pooled logits.

    The mean rather than the sum: a 200-frame track and a 5-frame one should produce
    comparable confidences, and a sum makes the long track certain of whatever it leaned
    towards. Length belongs in `frames`, where a reader can see it.
    """
    x = np.asarray(logits, dtype=float).reshape(len(logits), -1)
    if use is None:
        use = np.ones(len(x), dtype=bool)
    use = np.asarray(use, dtype=bool)
    if not use.any():
        return np.full(x.shape[1], np.nan)
    return x[use].mean(axis=0)


def track_attributes(
    logits: np.ndarray, names, use: np.ndarray | None = None, min_frames: int = 3
) -> dict[str, Attribute]:
    """Per-track attributes from per-frame logits. Empty when too few frames are usable.

    `min_frames` is a refusal rather than a smoothing constant. The measured median track
    is 26-34 frames, so a track with two usable crops is one where nearly every frame was
    truncated or tiny -- and an attribute computed from it would carry the same shape as
    one computed from 200 good crops. Returning nothing says so.
    """
    x = np.asarray(logits, dtype=float).reshape(len(logits), -1)
    if x.shape[1] != len(names):
        raise ValueError(f"{x.shape[1]} logits per frame against {len(names)} names")
    use = np.ones(len(x), dtype=bool) if use is None else np.asarray(use, dtype=bool)
    n = int(use.sum())
    if n < min_frames:
        return {}
    pooled = pool(x, use)
    per_frame = x[use] >= 0.0  # each frame's own decision, in logit space
    out = {}
    for i, name in enumerate(names):
        decision = pooled[i] >= 0.0
        agreement = float((per_frame[:, i] == decision).mean())
        out[name] = Attribute(
            name=name,
            probability=float(1.0 / (1.0 + np.exp(-pooled[i]))),
            agreement=agreement,
            frames=n,
        )
    return out


def age_band(attrs: dict[str, Attribute]) -> tuple[str, float]:
    """The one age band this track is in, as an argmax rather than three thresholds.

    Returns `(band, probability)`, or `("unknown", 0.0)` when the bands are absent.

    Per frame the head emits three independent sigmoids for three mutually exclusive
    states, so a frame can answer yes to two bands or no to all three. That is not a
    thing to fix per frame -- the three logits *are* the evidence -- but pooled over a
    track the exclusivity can be restored by taking the largest, which is what the label
    meant in the first place.

    Read the answer with `data/attributes.SUPPORT` in hand: `AgeOver60` was trained on
    1,127 crops and scored recall 0.119, so this function reporting `AgeOver60` is a
    statement about a channel that mostly says no.
    """
    present = [b for b in AGE_BANDS if b in attrs]
    if not present:
        return "unknown", 0.0
    best = max(present, key=lambda b: attrs[b].probability)
    return best, attrs[best].probability
