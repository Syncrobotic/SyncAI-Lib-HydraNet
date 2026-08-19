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
the arrangement `tests/test_orin_standalone_copies.py` exists because of, one layer in and
with nothing watching it.

Two tests keep the wider contract honest: `test_export_preprocessing.py` checks the
exported graph carries these exact values, and `test_orin_standalone_copies.py` checks the
hand-copied Jetson scripts still agree with them.
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

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "PAD_COLOR", "PAD_LABEL"]
