"""The numbers that turn a frame into a tensor, in one place.

Four constants, and what they have in common is that every one of them has to be
identical in three places at once: the augmentation pipeline that builds a training
batch, the inference path that builds a single frame, and the ONNX graph that carries
preprocessing inside itself for a board with no Python. A value that drifts between any
two of those does not raise. It feeds the network inputs it was never trained on, and the
only symptom is predictions that are slightly worse -- on padded frames only, which is
the subset nobody looks at.

They lived in `data/transforms.py`, which made `utils/visualize.py` import upward from
`utils` into `data` -- the one edge in the package that ran against the layering, and it
was already being dodged with function-local imports by someone who had noticed. Sitting
at the top level, this module depends on nothing and everything may depend on it.

`PAD_COLOR` in particular was not shared at all: `utils.visualize.letterbox` had the same
triple written out as a literal default with a comment pointing at the other copy. That is
the arrangement `tests/test_orin_standalone_copies.py` existed because of, one layer in and
with nothing watching it -- that test is gone with the Orin, and this module is why its
absence costs nothing: there is one definition now.

`test_export_preprocessing.py` keeps the wider contract honest: the exported graph
carries these exact values, so a host feeds raw 0-255 and the graph normalises. A second
test used to check that hand-copied Jetson copies of these constants still agreed; the
copies and the test went with the Orin on 2026-08-28, and folding the constants into the
graph is why losing that check costs nothing -- there is no second copy left to drift.
"""

from __future__ import annotations

import numpy as np

from .labels import IGNORE

# ImageNet statistics, in the 0-1 scale. The export path multiplies both by 255 because
# the graph is handed uint8 pixels; see cli/export_onnx.py.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Letterbox padding. Grey rather than black so that padding is not mistaken for a dark
# surface by a model that has learned floors are usually darker than walls.
PAD_COLOR = (114, 114, 114)

# Padded pixels are always ignore, never a class. They contribute no loss, which is the
# only honest thing to do with a region the camera never saw.
#
# Aliased rather than restated. `labels.IGNORE` is the mask-file contract -- the value an
# annotation PNG carries where nobody said, and the value the loss is told to skip -- and
# letterbox padding is a region nobody could have said anything about. Two names for one
# number is fine when one is defined as the other; two definitions of 255 is what
# `labels.py` was written to end, and this was the copy it did not reach.
PAD_LABEL = IGNORE


def letterbox_region(
    src_w: int, src_h: int, canvas_hw: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Where a source frame's real content sits inside a letterboxed canvas.

    Returns ``(x0, y0, content_w, content_h)`` -- the spelling `visualize.letterbox`
    already returns and `clip_tracks.to_source_pixels` already consumes, so this is not a
    new description of the letterbox, it is the one those two agreed on written down once.

    **It was written three times before this.** `visualize.letterbox` computed it from PIL
    sizes, `serve_pilot.letterbox_filter` recomputed it as an ffmpeg filter string, and
    `track_review` open-codes the inverse against a fourth spelling. A GPU letterbox in a
    streaming transport would have been the fifth. Copies of a geometry are not checkable
    against each other: a disagreement of one pixel moves every box by one pixel and
    nothing raises -- the argument this module's header already makes about `PAD_COLOR`,
    one function later.

    Same units as the callers it serves: `src_w`/`src_h` in the source frame's pixels,
    `canvas_hw` as ``(H, W)`` because that is how the project spells a network canvas
    (`CameraState.canvas_hw`, `preprocess`'s `size`).
    """
    s = min(canvas_hw[1] / src_w, canvas_hw[0] / src_h)
    nw, nh = max(round(src_w * s), 1), max(round(src_h * s), 1)
    return (canvas_hw[1] - nw) // 2, (canvas_hw[0] - nh) // 2, nw, nh


def undo_letterbox(
    coords: np.ndarray, region: tuple[int, int, int, int], src_w: int, src_h: int
) -> np.ndarray:
    """Canvas coordinates -> pixels of the frame that was actually filmed.

    Accepts ``(N, 2)`` points and ``(N, 4)`` xyxy boxes, because the two consumers need
    different ones: `clip_tracks.to_source_pixels` maps boxes, and the serving path's
    `WorldFrame` producer maps foot points.

    **Why this lives here and not next to `invert_geom`.** The arithmetic is the same --
    `to_source_pixels` delegates to this and says so -- but `data.transforms` imports
    torch and PIL, and `analytics.world` deliberately imports neither: it is produced
    every frame on the serving path, which is the reason its own module docstring gives
    for not living in `syncai_bev3d`. A pure-numpy geometry behind a torch import is a
    geometry the serving path cannot reuse, and "cannot reuse" is how the fourth copy
    gets written.

    **What getting this wrong looks like, measured 2026-09-09 on
    `Taichung-cam01.camera.json` (960x540 intrinsics, 1920x1080 stream, 640x1120
    canvas).** Handing canvas pixels to `world.world_frame` and naming the *stream* as
    their frame -- the honest-looking mistake, since the stream is what the camera sent --
    puts shoppers **2.4-2.7 m** from where they stand, with no NaN and no refusal, because
    the scaled points still land inside the calibrated canvas. Naming the *canvas* instead
    is off by 1-2 cm on a 16:9 source, which is inside this project's own floor spread, and
    by **1.37 m on the fleet's sideways-mounted camera** -- where the pad is 380 px wide
    instead of 5 px tall. That error is exactly zero on the frame's centre column and grows
    with lateral offset, so a spot check taken in the middle of the picture passes.
    A size cannot describe a letterboxed canvas; this region can.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 4):
        raise ValueError(
            f"undo_letterbox takes (N,2) points or (N,4) xyxy boxes, got {arr.shape}"
        )
    x0, y0, content_w, content_h = region
    out = arr.copy()
    # 0::2 selects x on both shapes (col 0, or cols 0 and 2); 1::2 selects y.
    out[:, 0::2] = (out[:, 0::2] - x0) / (content_w / src_w)
    out[:, 1::2] = (out[:, 1::2] - y0) / (content_h / src_h)
    return out


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PAD_COLOR",
    "PAD_LABEL",
    "letterbox_region",
    "undo_letterbox",
]
