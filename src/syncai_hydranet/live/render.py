"""One camera frame in, one live-view panel and its numbers out.

No ROS and no network: this takes two numpy arrays and a model, and returns the picture,
the statistics under it and the metric scene the 3D page reads. The subscriptions that
produce those arrays and the HTTP server that serves the result are both somebody else's
job, which is what makes this testable at all.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..data.coco_subsets import COCO_NAMES
from ..geometry.depth_scene import build_scene
from ..geometry.scene_types import DepthScene
from ..utils.visualize import TRAV_COLORS, crop_box, overlay, preprocess
from .reach import GO, REACH_BEYOND, REACH_COLORS, REACH_NO_DEPTH, REACH_WITHIN, classify_reach


@dataclass(frozen=True)
class LiveSettings:
    """What the viewer chose, as opposed to what the model believes.

    All three are set from the web page while the loop runs. They are a value rather than
    module state so that rendering a frame is a function of its arguments -- the loop can
    still keep a mutable copy and pass a snapshot of it.
    """

    range_m: float = 5.0
    score_thr: float = 0.30
    view: str = "both"


class LiveScene(DepthScene):
    """A `DepthScene` plus the range the viewer had selected when it was built.

    The 3D page reads it to draw the distance ring, so it is a field of the document that
    page is served -- but not one `geometry.depth_scene` produces, and saying so here is
    what stops the two drifting. Same split as `cli.scene.SceneReport`.
    """

    range_m: float


@dataclass
class LiveFrame:
    """The panel, the numbers under it, and the metric scene behind the 3D page."""

    panel: Image.Image
    jpeg: bytes
    stats: dict
    scene: LiveScene | None
    reach: np.ndarray


def _draw_detections(panel: Image.Image, det: dict, origin: tuple[int, int]) -> None:
    """Boxes on the left panel, in its own coordinates."""
    if not det or not len(det.get("boxes", [])):
        return
    x0, y0 = origin
    draw = ImageDraw.Draw(panel)
    for box, score, label in zip(
        det["boxes"].cpu().numpy(),
        det["scores"].cpu().numpy(),
        det["labels"].cpu().numpy(),
        strict=True,
    ):
        bx = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
        draw.rectangle(bx, outline=(255, 255, 0), width=2)
        draw.text((bx[0] + 2, bx[1] + 2), f"{_name_of(label)} {score:.2f}", fill=(255, 255, 0))


def _name_of(label) -> str:
    i = int(label)
    return COCO_NAMES[i] if i < len(COCO_NAMES) else str(i)


def _detections_in_frame(det: dict, origin, scale) -> list[dict]:
    """Boxes mapped from canvas coordinates into the colour frame's.

    Done before any depth is read through them: the canvas is letterboxed and decimated,
    so a box left in its coordinates would sample depth from beside the object.
    """
    if not det or not len(det.get("boxes", [])):
        return []
    x0, y0 = origin
    sx, sy = scale
    out = []
    for box, score, label in zip(
        det["boxes"].cpu().numpy(),
        det["scores"].cpu().numpy(),
        det["labels"].cpu().numpy(),
        strict=True,
    ):
        out.append(
            {
                "box": (
                    (box[0] - x0) * sx,
                    (box[1] - y0) * sy,
                    (box[2] - x0) * sx,
                    (box[3] - y0) * sy,
                ),
                "cls": _name_of(label),
                "score": float(score),
            }
        )
    return out


def render_frame(
    color: np.ndarray,
    depth_m: np.ndarray,
    model,
    device,
    *,
    size,
    terrain_colors,
    settings: LiveSettings,
    k: np.ndarray | None = None,
    extra_stats: dict | None = None,
) -> LiveFrame:
    """Run the model over one frame and build everything the viewer sees.

    `k` is the intrinsic matrix **of this frame**, already scaled for whatever stride the
    caller subsampled by. Without it the metric scene is skipped rather than guessed:
    a scene built from the wrong K is self-consistent and the wrong size, which is worse
    than not having one.
    """
    x, canvas, region = preprocess(Image.fromarray(color), size)
    with torch.no_grad():
        result = model.predict(x.to(device), score_thr=settings.score_thr)
    trav = crop_box(result["traversability"][0].cpu().numpy(), region)
    terr = crop_box(result["terrain"][0].cpu().numpy(), region)
    x0, y0, cw, ch = region
    vis = canvas.crop((x0, y0, x0 + cw, y0 + ch))

    # Walkable is the network's answer; range is the sensor's. They meet here, not inside
    # the weights -- so the threshold is a slider rather than a training run.
    trav_full = np.asarray(
        Image.fromarray(trav.astype(np.uint8)).resize(
            (color.shape[1], color.shape[0]), Image.Resampling.NEAREST
        )
    )
    reach = classify_reach(trav_full, depth_m, settings.range_m)

    show_terrain = settings.view == "terrain"
    left = overlay(
        vis,
        terr if show_terrain else trav,
        terrain_colors if show_terrain else TRAV_COLORS,
    )
    det = result.get("detection", [{}])[0]
    _draw_detections(left, det, (x0, y0))

    right_ids = np.asarray(
        Image.fromarray(reach.astype(np.uint8)).resize(vis.size, Image.Resampling.NEAREST)
    )
    right = overlay(vis, right_ids, REACH_COLORS, alpha=0.5)
    pair = Image.new("RGB", (left.width * 2 + 8, left.height), (16, 16, 16))
    pair.paste(left, (0, 0))
    pair.paste(right, (left.width + 8, 0))
    buf = io.BytesIO()
    pair.save(buf, format="JPEG", quality=80)

    go = trav_full == GO
    valid = depth_m > 0
    stats = {
        "range": f"{settings.range_m:.0f} m",
        "score": f"{settings.score_thr:.2f}",
        "go": f"{100 * go.mean():.1f}%",
        "go within range": f"{100 * (reach == REACH_WITHIN).mean():.1f}%",
        "go beyond range": f"{100 * (reach == REACH_BEYOND).mean():.1f}%",
        # Of `go`, not of the frame: the question is what share of what the model called
        # walkable the sensor could not see, and a share of the whole frame would shrink
        # as the floor does.
        "go with no depth": (
            f"{100 * (reach == REACH_NO_DEPTH).sum() / max(go.sum(), 1):.1f}% of go"
        ),
        "depth valid": f"{100 * valid.mean():.1f}%",
        "detections": int(len(det.get("boxes", [])) if det else 0),
        **(extra_stats or {}),
    }

    scene: LiveScene | None = None
    if k is not None:
        dets = _detections_in_frame(
            det, (x0, y0), (color.shape[1] / max(cw, 1), color.shape[0] / max(ch, 1))
        )
        scene = {**build_scene(trav_full, depth_m, k, dets), "range_m": settings.range_m}

    return LiveFrame(panel=pair, jpeg=buf.getvalue(), stats=stats, scene=scene, reach=reach)
