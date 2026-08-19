"""Night `person` boxes: what may be kept without a human looking at the frame.

The site `person` teacher is Grounding DINO at 0.35. That threshold was chosen from a
measured day/night gap and the night half of it **did not survive the fleet**: re-measured
on 2026-08-19 across all 42 live cameras at ~23:58 store-local, 13 of them put a box over
0.35 on an empty shuttered store, the worst at 0.594. See
`docs/journal/2026-08-19-night-person-fleet-recheck.md`.

This module is the veto that makes the rest usable. It does one thing:

    drop a box whose pixels never moved

against the camera's own midnight static plate, which `static_plates.py` already writes for
all 42 cameras from these very clips, so the reference illumination matches the frame. A
packet hanging on a pegboard scores near 1.0; a person standing still still breathes and
scores near 0.

WHAT IT IS AND IS NOT

**It is a veto.** It only ever removes. A low static share is *not* evidence that a box is a
person -- that inference is what produced two wrong verdicts in the first pass of this work,
and both were caught by opening the frames rather than by any number. Where the plate itself
is noisy, everything reads as moving:

  - `Kaohsiung-cam02` is an outdoor street bay. Two **parked scooters** score 0.00-0.09
    because headlights and IR gain change the whole scene between samples.
  - `Tao-Hsin-cam09` is a near-nadir product rack. A **hanging accessory** scores 0.23-0.29
    for the same reason, one level down.

Neither is removable at any threshold that keeps people, so both are excluded by name in
`UNPROTECTED` instead of being pretended about.

THE THRESHOLD, AND WHY IT IS NOT DELICATE

Swept against eye-verified ground truth -- 12 person boxes on Tao-Hsin-cam04, 72 false boxes
on 13 cameras, every one opened at native resolution:

    threshold      false removed    person lost
    0.30 .. 0.75      62 of 72           0
    0.80 .. 0.90      47 of 72           0
    0.95              41 of 72           0

`DROP_ABOVE` is the centre of a **0.45-wide plateau** on which the answer does not change at
all, which is a stronger form of ARCHITECTURE_DIRECTION rule 2 than a point in a gap. The 10
it never removes at any setting are the two cameras above.

RESIDUAL RISK, WHICH THE ANNOTATION LAYER CANNOT SOLVE

**A person who holds perfectly still for the whole clip is furniture to this veto.** The
static share measures pixels, not people, and nothing here would notice. That is a reason
for the alerting layer to put a human on night events; it is not something a labelling gate
can fix, and raising the threshold does not help -- a motionless person and a motionless
packet are the same measurement.

The narrower version of the same risk is that this only sees a box's own pixels: a person
standing in front of a static display inherits some of the display's stillness. The measured
margin absorbs it on this fleet (person max 0.282 against a threshold of 0.50), but that
margin is 12 boxes on one camera and should be re-measured when the campaign's night tranches
land.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Centre of the 0.30-0.75 plateau measured above. A box at or above this share is furniture.
DROP_ABOVE = 0.50

# The working threshold for the detector itself, unchanged; this module never applies it.
PERSON_SCORE = 0.35

# Below this the share is computed over too few plate pixels to mean anything -- the same
# concern `static_person_filter.py` raises: 1.00 over nine pixels is a box smaller than the
# mask's resolution, not a confident verdict. Such a box is kept and flagged, never dropped.
MIN_MASK_PIXELS = 64

# Cameras where the veto provably cannot separate the two populations, with the reason.
# Verified at native resolution on 2026-08-19; both are excluded from unreviewed night
# person labelling rather than being run through a gate that cannot see them.
UNPROTECTED = {
    "Kaohsiung-cam02": "outdoor street bay; two parked scooters score 0.00-0.09 because "
    "headlights and IR gain move the whole scene between plate samples",
    "Tao-Hsin-cam09": "near-nadir product rack; a hanging accessory scores 0.23-0.29 for "
    "the same reason -- the plate is noise-dominated at this exposure",
}

# Cameras that put a static box over 0.35 on an empty store. The veto is what makes them
# usable; listed so a caller can report coverage rather than discover it.
VETO_FIRES_ON = (
    "Kaohsiung-cam01",
    "Kaohsiung-cam07",
    "Kaohsiung-cam10",
    "Kaohsiung-cam12",
    "Kaohsiung-cam13",
    "Taichung-cam01",
    "Taichung-cam03",
    "Taichung-cam05",
    "Taichung-cam06",
    "Tao-Hsin-cam06",
    "Tao-Hsin-cam07",
)


def slot_of(file_name: str) -> str:
    """`Cam__archive_20260816-160053_20260816-160557__00125.jpg` -> `20260816-160053`.

    The slot key is **UTC**, as `static_plates.py` writes it, while the timestamp burned
    into the frame is store-local. Nothing here converts between them, deliberately: the
    pull that did convert silently is the one `pull_studioa.py` documents.
    """
    _, session, _ = file_name.rsplit(".", 1)[0].split("__")
    return session.split("_")[1]


def camera_of(file_name: str) -> str:
    return file_name.split("__")[0]


def static_share(mask: np.ndarray, bbox, img_w: int, img_h: int) -> tuple[float, int]:
    """Fraction of a box's area that never changed, and the plate pixels behind it.

    Identical to `scripts/static_person_filter.py`'s, which imports this rather than
    keeping a second copy: the two disagreeing about a box's static share would make the
    report that chose the threshold and the gate that applies it different instruments.
    """
    mh, mw = mask.shape
    x, y, w, h = bbox
    x0 = max(int(np.floor(x * mw / img_w)), 0)
    y0 = max(int(np.floor(y * mh / img_h)), 0)
    x1 = min(max(int(np.ceil((x + w) * mw / img_w)), x0 + 1), mw)
    y1 = min(max(int(np.ceil((y + h) * mh / img_h)), y0 + 1), mh)
    patch = mask[y0:y1, x0:x1]
    return (float(patch.mean()) if patch.size else 0.0), int(patch.size)


@dataclass(frozen=True)
class Decision:
    """One box's fate, with the number behind it rather than only the verdict."""

    bbox: tuple[float, float, float, float]
    score: float
    keep: bool
    reason: str
    share: float | None = None
    mask_pixels: int = 0


class NightPersonVeto:
    """Apply the static veto to one camera's night boxes.

        veto = NightPersonVeto(Path("datasets/studioa_static"))
        if veto.status(camera) == "excluded":
            ...                                  # do not label this camera unreviewed
        decisions = veto.apply(file_name, boxes, img_w, img_h)
        kept = [d for d in decisions if d.keep]

    `boxes` is any iterable of `(bbox_xywh, score)`. Boxes below `PERSON_SCORE` are dropped
    for being below the working threshold, and that is reported as its own reason so a
    caller can tell "the detector did not want it" from "the plate said furniture".
    """

    def __init__(
        self,
        static_root: Path,
        drop_above: float = DROP_ABOVE,
        score_thr: float = PERSON_SCORE,
    ) -> None:
        self.static_root = Path(static_root)
        self.drop_above = drop_above
        self.score_thr = score_thr
        self._masks: dict[tuple[str, str], np.ndarray | None] = {}

    def status(self, camera: str) -> str:
        """`excluded` | `veto_active` | `clean`, and only the first is a refusal."""
        if camera in UNPROTECTED:
            return "excluded"
        return "veto_active" if camera in VETO_FIRES_ON else "clean"

    def mask(self, camera: str, slot: str) -> np.ndarray | None:
        key = (camera, slot)
        if key not in self._masks:
            path = self.static_root / camera / f"static_{slot}.png"
            self._masks[key] = (
                np.asarray(Image.open(path).convert("L")) > 127 if path.exists() else None
            )
        return self._masks[key]

    def apply(self, file_name: str, boxes, img_w: int, img_h: int) -> list[Decision]:
        camera, slot = camera_of(file_name), slot_of(file_name)
        if camera in UNPROTECTED:
            return [
                Decision(tuple(b), float(s), False, f"camera excluded: {UNPROTECTED[camera]}")
                for b, s in boxes
            ]
        mask = self.mask(camera, slot)
        out: list[Decision] = []
        for bbox, score in boxes:
            if score < self.score_thr:
                out.append(Decision(tuple(bbox), float(score), False, "below score threshold"))
                continue
            if mask is None:
                # No plate is not a pass. A camera with no midnight plate has never been
                # checked, and the 13 that fail are not guessable from the frame.
                out.append(Decision(tuple(bbox), float(score), False, "no static plate"))
                continue
            share, npx = static_share(mask, bbox, img_w, img_h)
            if npx < MIN_MASK_PIXELS:
                out.append(
                    Decision(
                        tuple(bbox), float(score), True, "kept: plate too coarse", share, npx
                    )
                )
            elif share >= self.drop_above:
                out.append(
                    Decision(tuple(bbox), float(score), False, "static: furniture", share, npx)
                )
            else:
                out.append(Decision(tuple(bbox), float(score), True, "kept", share, npx))
        return out
