"""The tiny export-test model, in the one place its constraints can be stated.

Five test files carried near-verbatim copies of the same resnet18+FPN skeleton, and
the four-line comment justifying the input size lived in only one of them -- the other
four carried the constraint without the reason. The reason:

Small on purpose -- these tests exercise control flow and every one builds a model --
but not smaller than 128x160. At 64x80 the deepest FPN level is 1x1 and `F.group_norm`
in the FCOS tower refuses a 1-element map outright. That is a real constraint on any
export, not a test artefact: the smallest resolution this project ships is 384x512.

An underscore module by `_posture.py`'s argument: plain factories, not fixtures.
"""

from __future__ import annotations

INPUT_SIZE = [128, 160]


def tiny_trunk(out_channels: int = 32, num_repeats: int = 1, num_levels: int = 5) -> dict:
    """The backbone+neck stanza every export test shares."""
    return {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {
            "name": "fpn",
            "out_channels": out_channels,
            "num_repeats": num_repeats,
            "num_levels": num_levels,
        },
    }


def seg_head(num_classes: int = 3, channels: int = 32) -> dict:
    return {
        "type": "semantic_fpn",
        "num_classes": num_classes,
        "in_levels": [0, 1, 2],
        "channels": channels,
    }


def det_head(num_classes: int = 80, channels: int = 32, num_convs: int = 1) -> dict:
    return {
        "type": "fcos",
        "num_classes": num_classes,
        "channels": channels,
        "num_convs": num_convs,
    }
