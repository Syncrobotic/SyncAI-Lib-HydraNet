"""Palettes, overlays, letterboxing and TensorBoard comparison grids.

Shared by the image and video inference CLIs and by the trainer's validation logging.

Why letterbox: ``data.transforms.Resize`` stretches straight to the target size. RUGD
is roughly 1.25:1 and the default 512x640 input is also 1.25:1, so training on it is
lossless. A phone clip is 16:9 (1.78:1) or portrait 9:16 (0.56:1); stretching that to
1.25:1 squeezes the frame horizontally by up to a factor of two. Inference always
letterboxes; training does so when ``data.letterbox`` is set.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Traversability: blocked / caution / go.
TRAV_COLORS = np.array([[220, 40, 40], [250, 200, 40], [40, 200, 80]], dtype=np.uint8)

# Terrain, 12 classes. Index order matches ``data.terrain_classes`` in the configs.
TERRAIN_COLORS = np.array(
    [
        [0, 0, 0],
        [139, 90, 43],
        [60, 180, 75],
        [0, 100, 0],
        [128, 128, 128],
        [210, 180, 140],
        [178, 34, 34],
        [135, 206, 235],
        [30, 100, 220],
        [255, 215, 0],
        [112, 128, 144],
        [255, 140, 0],
    ],
    dtype=np.uint8,
)


def overlay(
    base: Image.Image, mask: np.ndarray, palette: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blend a class-index map over an image using ``palette``."""
    color = palette[np.clip(mask, 0, len(palette) - 1)]
    out = (
        np.asarray(base, dtype=np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    ).astype(np.uint8)
    return Image.fromarray(out)


def letterbox(img: Image.Image, size, fill=(114, 114, 114)):
    """Scale to fit inside ``(H, W)`` preserving aspect ratio, then centre-pad.

    Returns ``(image, (x0, y0, w, h))``; the region locates the real content inside the
    canvas so predictions can be cropped back to the source aspect ratio.
    """
    h, w = size
    ow, oh = img.size
    s = min(w / ow, h / oh)
    nw, nh = max(round(ow * s), 1), max(round(oh * s), 1)
    canvas = Image.new("RGB", (w, h), fill)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (x0, y0))
    return canvas, (x0, y0, nw, nh)


def crop_box(arr: np.ndarray, region) -> np.ndarray:
    """Crop an ``HxW`` array produced on a letterboxed canvas back to its content."""
    x0, y0, w, h = region
    return arr[y0 : y0 + h, x0 : x0 + w]


# ---------------------------------------------------------------------------
# Training-time monitoring
# ---------------------------------------------------------------------------


def denormalize(t) -> np.ndarray:
    """Convert a normalized ``[3, H, W]`` tensor back to an ``[H, W, 3]`` uint8 array."""
    from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD

    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def prediction_grid(images, preds, gts, palette, max_n: int = 4, gap: int = 4) -> np.ndarray:
    """Build an "input | prediction | label" comparison grid, one row per sample.

    Returns ``[H, W, 3]`` uint8, ready for ``SummaryWriter.add_image(..., dataformats="HWC")``.
    Loss curves only tell you the loss is falling; this shows whether the model is
    calling an entire floor a wall, and how much of the frame is ignore padding.
    """
    rows = []
    n = min(max_n, len(images))
    for i in range(n):
        base = Image.fromarray(denormalize(images[i]))
        p = np.asarray(preds[i].detach().cpu().numpy(), dtype=np.int64)
        g = np.asarray(gts[i].detach().cpu().numpy(), dtype=np.int64)
        pred_img = np.asarray(overlay(base, p, palette))
        # Paint label ignore (255) black so unsupervised regions are obvious.
        gt_img = np.asarray(overlay(base, np.where(g == 255, 0, g), palette))
        gt_img = np.where((g == 255)[..., None], 0, gt_img).astype(np.uint8)
        sep = np.full((gt_img.shape[0], gap, 3), 255, np.uint8)
        rows.append(np.concatenate([np.asarray(base), sep, pred_img, sep, gt_img], axis=1))
    if not rows:
        return np.zeros((1, 1, 3), np.uint8)
    w = max(r.shape[1] for r in rows)
    hsep = np.full((gap, w, 3), 255, np.uint8)
    out = []
    for r in rows:
        if r.shape[1] < w:
            r = np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255)
        out += [r, hsep]
    return np.concatenate(out[:-1], axis=0)
