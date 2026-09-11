"""Decode Trans10K-v2 RGB annotations without confusing palette IDs with class IDs.

The cached rdyzakya/Trans10K-v2 mirror stores colour masks, including an opaque
alpha channel. Its numeric class ordering differs from other exports of Trans10K.
Unrecognised colours are ignored, never silently trained as background.
"""

from __future__ import annotations

import numpy as np

GLAZING_CLASSES = ("background", "glass_door", "glass", "window")
RGB_TO_GLAZING = {
    (0, 0, 0): 0,
    (4, 250, 7): 0,  # storage box
    (150, 5, 61): 0,  # bottle
    (204, 255, 4): 3,  # window
    (140, 140, 140): 0,  # eyeglass
    (6, 230, 230): 0,  # freezer
    (235, 255, 7): 0,  # jar / kettle
    (120, 120, 120): 1,  # glass door
    (255, 51, 7): 0,  # cup
    (255, 0, 0): 0,  # one cup has this alternate colour in the mirror
    (224, 5, 255): 2,  # glass wall
    (204, 5, 255): 0,  # bowl
    (120, 120, 70): 0,  # glass shelf, not an architectural opening
}


def glazing_labels(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError("Trans10K colour masks must be RGB or RGBA")
    values = rgb[..., :3].astype(np.uint32)
    packed = (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]
    labels = np.full(rgb.shape[:2], 255, np.uint8)
    for (red, green, blue), target in RGB_TO_GLAZING.items():
        labels[packed == ((red << 16) | (green << 8) | blue)] = target
    if rgb.shape[2] == 4:
        labels[rgb[..., 3] == 0] = 255
    return labels
