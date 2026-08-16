"""Joint augmentation of image, segmentation masks and detection boxes.

Implemented on numpy/PIL/torch rather than albumentations to keep the deployment
environment's dependency surface small.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

from ..preprocessing import IMAGENET_MEAN, IMAGENET_STD, PAD_COLOR, PAD_LABEL


class Sample(dict):
    """Keys: image (PIL), masks: dict[str, HxW array], boxes: [N,4], labels: [N],
    geom: the accumulated original-image -> network-input mapping (see below)."""


# geom = (sx, sy, px, py) with  x_net = x_orig * sx + px.  Every geometric step keeps it
# up to date, so evaluation can map predictions back to original image coordinates
# without re-deriving the transform from image sizes -- a derivation that is simply
# wrong once letterbox padding is involved.
GEOM_IDENTITY = (1.0, 1.0, 0.0, 0.0)


def _geom_scale(sample: Sample, fx: float, fy: float) -> None:
    g = sample.get("geom", GEOM_IDENTITY)
    if g is not None:
        sample["geom"] = (g[0] * fx, g[1] * fy, g[2] * fx, g[3] * fy)


def _geom_shift(sample: Sample, dx: float, dy: float) -> None:
    g = sample.get("geom", GEOM_IDENTITY)
    if g is not None:
        sample["geom"] = (g[0], g[1], g[2] + dx, g[3] + dy)


def invert_geom(boxes: np.ndarray, geom) -> np.ndarray:
    """Map ``[N,4]`` xyxy boxes from network coordinates back to the original image."""
    if geom is None:
        raise ValueError(
            "this sample's geometry is not invertible (a flip or other non-affine "
            "augmentation was applied); predictions cannot be mapped back to the "
            "original image. Evaluation must use the val transform pipeline."
        )
    sx, sy, px, py = geom
    out = np.asarray(boxes, dtype=np.float64).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - px) / sx
    out[:, [1, 3]] = (out[:, [1, 3]] - py) / sy
    return out


def _resize(sample: Sample, size: tuple[int, int]) -> Sample:
    h, w = size
    img = sample["image"]
    ow, oh = img.size
    _geom_scale(sample, w / ow, h / oh)
    sample["image"] = img.resize((w, h), Image.Resampling.BILINEAR)
    if "masks" in sample:
        sample["masks"] = {
            k: np.array(Image.fromarray(m).resize((w, h), Image.Resampling.NEAREST))
            for k, m in sample["masks"].items()
        }
    if "boxes" in sample and len(sample["boxes"]):
        sx, sy = w / ow, h / oh
        b = sample["boxes"].astype(np.float32)
        b[:, [0, 2]] *= sx
        b[:, [1, 3]] *= sy
        sample["boxes"] = b
    return sample


def _paste(sample: Sample, size: tuple[int, int], x0: int, y0: int) -> Sample:
    """Paste an already-scaled sample onto a ``size`` canvas at ``(x0, y0)``.

    Content outside the canvas is cropped; the remainder is filled with PAD_COLOR /
    PAD_LABEL. Negative offsets therefore express a random crop.
    """
    h, w = size
    img = sample["image"]
    _geom_shift(sample, x0, y0)
    canvas = Image.new("RGB", (w, h), PAD_COLOR)
    canvas.paste(img, (x0, y0))
    sample["image"] = canvas
    if "masks" in sample:
        out = {}
        for k, m in sample["masks"].items():
            c = np.full((h, w), PAD_LABEL, dtype=m.dtype)
            sx0, sy0 = max(-x0, 0), max(-y0, 0)
            dx0, dy0 = max(x0, 0), max(y0, 0)
            cw = min(m.shape[1] - sx0, w - dx0)
            ch = min(m.shape[0] - sy0, h - dy0)
            if cw > 0 and ch > 0:
                c[dy0 : dy0 + ch, dx0 : dx0 + cw] = m[sy0 : sy0 + ch, sx0 : sx0 + cw]
            out[k] = c
        sample["masks"] = out
    if "boxes" in sample and len(sample["boxes"]):
        b = sample["boxes"].copy()
        b[:, [0, 2]] += x0
        b[:, [1, 3]] += y0
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        keep = (b[:, 2] - b[:, 0] > 2) & (b[:, 3] - b[:, 1] > 2)
        sample["boxes"], sample["labels"] = b[keep], sample["labels"][keep]
    return sample


class Compose:
    def __init__(self, ts):
        self.ts = ts

    def __call__(self, s: Sample) -> Sample:
        for t in self.ts:
            s = t(s)
        return s


class Resize:
    """Stretch to ``(H, W)``, ignoring aspect ratio."""

    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, s):
        return _resize(s, self.size)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, s: Sample) -> Sample:
        if random.random() > self.p:
            return s
        w = s["image"].size[0]
        s["geom"] = None  # a mirror is not an (sx, sy, px, py) mapping; train-only anyway
        s["image"] = s["image"].transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if "masks" in s:
            s["masks"] = {k: np.ascontiguousarray(m[:, ::-1]) for k, m in s["masks"].items()}
        if "boxes" in s and len(s["boxes"]):
            b = s["boxes"].copy()
            b[:, [0, 2]] = w - b[:, [2, 0]]
            s["boxes"] = b
        return s


class RandomScaleCrop:
    """Random scale (0.75-1.5x) then crop back to the target size, stretching to fit.

    Scales below 1 shrink the frame and pad the remainder at a random position, so the
    zoom-out half of the range is a real augmentation rather than a round trip through a
    lower resolution.
    """

    def __init__(self, size, scale_range=(0.75, 1.5)):
        self.size = tuple(size)  # (H, W)
        self.scale_range = scale_range

    def __call__(self, s: Sample) -> Sample:
        h, w = self.size
        scale = random.uniform(*self.scale_range)
        sh, sw = max(int(h * scale), 1), max(int(w * scale), 1)
        s = _resize(s, (sh, sw))
        if sh <= h or sw <= w:
            x0 = random.randint(min(0, w - sw), max(0, w - sw))
            y0 = random.randint(min(0, h - sh), max(0, h - sh))
            return _paste(s, self.size, x0, y0)
        y0 = random.randint(0, sh - h)
        x0 = random.randint(0, sw - w)
        _geom_shift(s, -x0, -y0)
        s["image"] = s["image"].crop((x0, y0, x0 + w, y0 + h))
        if "masks" in s:
            s["masks"] = {k: m[y0 : y0 + h, x0 : x0 + w] for k, m in s["masks"].items()}
        if "boxes" in s and len(s["boxes"]):
            b = s["boxes"].copy()
            b[:, [0, 2]] -= x0
            b[:, [1, 3]] -= y0
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
            keep = (b[:, 2] - b[:, 0] > 2) & (b[:, 3] - b[:, 1] > 2)
            s["boxes"], s["labels"] = b[keep], s["labels"][keep]
        return s


class LetterboxResize:
    """Aspect-preserving scale then centre-pad to ``(H, W)``. Used for val and inference."""

    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, s: Sample) -> Sample:
        h, w = self.size
        ow, oh = s["image"].size
        f = min(w / ow, h / oh)
        nw, nh = max(round(ow * f), 1), max(round(oh * f), 1)
        s = _resize(s, (nh, nw))
        return _paste(s, self.size, (w - nw) // 2, (h - nh) // 2)


class LetterboxScaleCrop:
    """Aspect-preserving RandomScaleCrop: scale, then random crop or random-position pad.

    Scales above 1 mostly crop (zoom in), below 1 mostly pad (zoom out). Neither
    distorts object shape, which matters most for the detection head.
    """

    def __init__(self, size, scale_range=(0.75, 1.5)):
        self.size = tuple(size)
        self.scale_range = scale_range

    def __call__(self, s: Sample) -> Sample:
        h, w = self.size
        ow, oh = s["image"].size
        f = min(w / ow, h / oh) * random.uniform(*self.scale_range)
        nw, nh = max(round(ow * f), 1), max(round(oh * f), 1)
        s = _resize(s, (nh, nw))
        x0 = random.randint(min(0, w - nw), max(0, w - nw))
        y0 = random.randint(min(0, h - nh), max(0, h - nh))
        return _paste(s, self.size, x0, y0)


class ColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3):
        self.b, self.c, self.s = brightness, contrast, saturation

    def __call__(self, s: Sample) -> Sample:
        from PIL import ImageEnhance

        img = s["image"]
        for enh, mag in (
            (ImageEnhance.Brightness, self.b),
            (ImageEnhance.Contrast, self.c),
            (ImageEnhance.Color, self.s),
        ):
            if mag > 0:
                img = enh(img).enhance(1.0 + random.uniform(-mag, mag))
        s["image"] = img
        return s


class ToTensor:
    def __call__(self, s: Sample) -> Sample:
        img = np.asarray(s["image"], dtype=np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        s["image"] = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        if "masks" in s:
            s["masks"] = {
                k: torch.from_numpy(m.astype(np.int64)) for k, m in s["masks"].items()
            }
        if "boxes" in s:
            s["boxes"] = torch.from_numpy(np.asarray(s["boxes"], dtype=np.float32)).reshape(
                -1, 4
            )
            s["labels"] = torch.from_numpy(np.asarray(s["labels"], dtype=np.int64)).reshape(-1)
        return s


# Augmentation defaults. These were hard-coded until they became config: identical
# values, so runs before and after the change remain comparable.
AUGMENT_DEFAULTS = {
    "scale_range": (0.75, 1.5),
    "flip_p": 0.5,
    "brightness": 0.3,
    "contrast": 0.3,
    "saturation": 0.3,
}


def build_transforms(
    input_size, train: bool, letterbox: bool = False, augment: dict | None = None
) -> Compose:
    """``letterbox=False`` keeps the original stretch-to-fit behaviour.

    RUGD is about 1.25:1 and the default 512x640 input is 1.25:1 too, so the off-road
    configs lose nothing by stretching. Footage from your own camera rarely matches,
    so set ``data.letterbox: true`` there.

    ``augment`` overrides :data:`AUGMENT_DEFAULTS`. It comes from ``data.augment`` in the
    config, which means it is recorded in ``meta.json`` -- and augmentation strength
    belongs in a run's lineage as much as the learning rate does. It was hard-coded here
    until this project's own rule ("the most expensive mistake is a setting that never
    took effect") was noticed to apply to itself: a change of jitter strength between two
    runs was invisible, and the difference in their scores unexplainable.
    """
    a = {**AUGMENT_DEFAULTS, **(augment or {})}
    if train:
        geom_cls = LetterboxScaleCrop if letterbox else RandomScaleCrop
        geom = geom_cls(input_size, scale_range=tuple(a["scale_range"]))
        return Compose(
            [
                geom,
                RandomHorizontalFlip(p=float(a["flip_p"])),
                ColorJitter(
                    brightness=float(a["brightness"]),
                    contrast=float(a["contrast"]),
                    saturation=float(a["saturation"]),
                ),
                ToTensor(),
            ]
        )
    geom = LetterboxResize(input_size) if letterbox else Resize(input_size)
    return Compose([geom, ToTensor()])
