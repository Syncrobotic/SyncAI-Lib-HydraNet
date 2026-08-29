"""Choosing which frames of a clip are worth a human's time.

A fixed CCTV camera produces 900 near-identical frames in five minutes. Labelling 40 of
those at random buys perhaps three distinct scenes, and a train/val split over them
measures memorisation rather than generalisation. Picking frames that *disagree* with
each other is what makes an annotation hour worth what it costs.

This lives in the package rather than in ``scripts/`` because two pre-labellers need it
-- ``scripts/annotation_batch.py`` (model pre-labels) and ``scripts/sam3_prelabel.py``
(SAM 3 pre-labels) -- and the second used to get it by importing the first. A script
importing another script needs ``sys.path`` surgery to work at all, breaks the moment
either is invoked from a different directory, and is invisible to the test suite. The
functions were always library code; only their address was wrong.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DESCRIPTOR = (12, 16)  # a coarse grey thumbnail is enough to tell scenes apart


def describe(frame: np.ndarray) -> np.ndarray:
    """A tiny grey thumbnail, normalised. Cheap, and robust to compression noise."""
    img = (
        Image.fromarray(frame).convert("L").resize(DESCRIPTOR[::-1], Image.Resampling.BILINEAR)
    )
    v = np.asarray(img, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-6)


def farthest_first(descs: list[np.ndarray], k: int) -> list[int]:
    """Pick k frames that disagree with each other as much as possible.

    Starts from the frame furthest from the average -- the least typical one -- rather
    than from frame zero, so a clip that opens on an empty room does not anchor the whole
    selection on it.
    """
    if len(descs) <= k:
        return list(range(len(descs)))
    x = np.stack(descs)
    chosen = [int(np.argmax(np.linalg.norm(x - x.mean(0), axis=1)))]
    dist = np.linalg.norm(x - x[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(x - x[nxt], axis=1))
    return sorted(chosen)
