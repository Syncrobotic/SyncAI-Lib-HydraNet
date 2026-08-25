"""Person tracks for one clip: the loop four scripts were each keeping their own copy of.

`stage.py` names the second stage's boundary and types what crosses it. This is the first
producer on that boundary -- video in, `Track`s out -- and it exists because the same
twenty lines had been written four times and had stopped agreeing.

---------------------------------------------------------------------------
WHAT THE FOUR COPIES DISAGREED ABOUT, AND WHY IT MATTERS

    scripts/site_events.py          undistorted the boxes
    scripts/mine_fall_candidates.py did not
    scripts/retail_flow.py          did not
    scripts/track_review.py         did not

Same loop, same tracker settings, same detection head. One of them corrected the lens
before associating, three did not, and nothing in any of them said which choice it was
making. `scripts/ty_ratchet.sh` had already named this as "one of the only two genuine
code duplications in the repo, one of them with a behavioural divergence between the
copies" -- it was recorded and then left, which is how a known defect becomes an old one.

The cost is not that undistortion is large. At these radii it moves an edge by a pixel or
two. The cost is that **the tracker associates on IoU**, so a systematic shift applied to
some frames and not others changes which observations get linked -- and the fall-candidate
mining that concluded "none of the 48 spans is a posture" ran on the copy without it.

**So `k1` has no default here.** Every caller states its lens correction or states `None`,
in the call, where a reader sees it. A default would rebuild the thing this module was
written to remove: a choice being made silently and differently in four places.

---------------------------------------------------------------------------
WHY IN THE PACKAGE RATHER THAN A FOURTH SCRIPT

`retail_flow.py` had become the de-facto library for the other three -- they reach it with
a `sys.path` insert and `from retail_flow import PERSON, to_source_pixels`. A script that
other scripts import is a module in the wrong place: outside the wheel, outside the type
ratchet and outside the coverage floor, which is exactly the argument `live/__init__.py`
makes about the ROS view it absorbed.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Protocol

import numpy as np
import torch
from PIL import Image

from ..data.coco_subsets import COCO_NAMES
from ..data.transforms import invert_geom
from ..geometry.ground import undistort_points

# COCO's index for `person`, resolved from the name rather than written as an integer:
# the head is trained on COCO's 80 and the number is a property of that list, not of this
# project. A literal here goes wrong silently the first time a config narrows the head.
PERSON = COCO_NAMES.index("person")

__all__ = ["PERSON", "ClipTracks", "to_source_pixels", "track_clip", "undistort_boxes"]


class ClipTracks(NamedTuple):
    """Everything the four call sites between them needed out of one pass over a clip.

    One return type rather than three, because the copies differed here too: one returned
    `(tracks, frames, src_w, src_h)` and another `(tracks, frames, detections)`, so a
    reader moving between them had to re-check the tuple order every time.
    """

    tracks: list
    frames: int
    detections: int
    src_w: int
    src_h: int


class _Frames(Protocol):
    """What this module needs from a video reader, so the reader stays the caller's.

    Positional-only (`/`). A protocol that names its parameters also constrains the
    caller's spelling of them, and `data.video.frames` calls its first two `path` and
    `stride_fps`. Requiring a rename there to satisfy a structural type here would be the
    protocol dictating to the thing it describes.
    """

    def __call__(self, clip: str, w: int, h: int, fps: float, /): ...


def to_source_pixels(boxes: np.ndarray, region, src_w: int, src_h: int) -> np.ndarray:
    """Model-canvas boxes -> pixels of the frame that was actually filmed.

    `preprocess` letterboxes, so the canvas is content plus grey bars and a plain
    `src_w / model_w` scale is wrong by the padding. Getting it wrong does not look wrong:
    every box lands somewhere plausible, shifted by the bar width, and the ground
    projection then reports a floor position that is confidently a metre off.

    The arithmetic already exists as `invert_geom`, which the evaluator uses to put COCO
    boxes back on the original image; this only translates between the two ways the project
    describes the same letterbox. `region` is `(x0, y0, content_w, content_h)` from
    `preprocess`; `geom` is `(sx, sy, px, py)` with `out = (box - p) / s`. **A second copy
    of the arithmetic would be a second chance to get a sign wrong**, which is the whole
    argument this module is an instance of.
    """
    if not len(boxes):
        return np.zeros((0, 4))
    x0, y0, cw, ch = region
    return invert_geom(np.asarray(boxes, dtype=float), (cw / src_w, ch / src_h, x0, y0))


def undistort_boxes(boxes: np.ndarray, k1: float, w: int, h: int) -> np.ndarray:
    """Undo the lens on box corners, so the ground projection sees a pinhole camera.

    Corners rather than the foot point alone, and then the axis-aligned hull: the tracker
    associates on IoU and would otherwise compare undistorted foot points against distorted
    boxes. The hull is an approximation of a shape that is no longer a rectangle after a
    radial map; at these radii it moves an edge by a pixel or two, which is far below the
    error in the pose itself and does not pretend otherwise.
    """
    if not len(boxes):
        return boxes
    centre = (w / 2.0, h / 2.0)
    radius = math.hypot(h, w) / 2.0
    pts = boxes.reshape(-1, 2, 2).reshape(-1, 2)
    out = undistort_points(pts, k1, centre, radius).reshape(-1, 2, 2)
    return np.concatenate([out.min(axis=1), out.max(axis=1)], axis=1)


def track_clip(
    clip: str,
    model,
    size,
    device,
    tracker,
    *,
    frames: _Frames,
    preprocess,
    probe,
    fps: float,
    score_thr: float,
    k1: float | None,
    max_frames: int = 0,
    person_label: int = PERSON,
) -> ClipTracks:
    """Run the detector over a clip and associate its person boxes into tracks.

    `k1` is keyword-only and has **no default** on purpose -- see this module's header.
    Pass the camera's radial coefficient to correct the lens before associating, or `None`
    to state that this call deliberately does not. `0.0` is the identity of the division
    model and so behaves like `None`; both are accepted because a caller reproducing an
    older run wants to say "zero" rather than "no correction", and those are different
    sentences about the same arithmetic.

    `frames`, `preprocess` and `probe` are passed in rather than imported: the video reader
    and the letterbox live on the CLI side of the package, and importing them here would
    make `analytics` depend on `cli`, which is the one edge the layering forbids.
    """
    src_w, src_h, _ = probe(clip)
    n = detections = 0
    for frame in frames(clip, src_w, src_h, fps):
        x, _, region = preprocess(Image.fromarray(frame), size)
        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=score_thr)
        det = res["detection"][0]
        if len(det.get("labels", [])):
            lab = det["labels"].cpu().numpy()
            box = det["boxes"].cpu().numpy()[lab == person_label]
            box = to_source_pixels(box, region, src_w, src_h)
            if k1 is not None:
                box = undistort_boxes(box, k1, src_w, src_h)
        else:
            box = np.zeros((0, 4))
        detections += len(box)
        tracker.update(box, n)
        n += 1
        if max_frames and n >= max_frames:
            break
    return ClipTracks(tracker.finished(), n, detections, src_w, src_h)
