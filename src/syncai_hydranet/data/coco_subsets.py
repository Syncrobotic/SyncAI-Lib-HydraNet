"""Named subsets of the COCO 80, so that a measurement made over one can be repeated.

A list of category names is not incidental to a detection number -- it *is* the
denominator. `runs/hydranet_joint_coco10/best.pt` scores mAP 0.3348 over all 80 and
0.3246 over the 25 below, and the only difference between those two figures is which
names were in the list. A baseline quoted without its subset is not a baseline.

The 0.3246 figure was recorded, argued from, and used to set the bar for a narrowed
detection head -- while the 25 names that defined it lived only in one shell command in
one session's history. This module is where they live now.
"""

from __future__ import annotations

# COCO's 80 categories in sorted-category-id order, which is the order CocoDetDataset
# assigns contiguous labels in: index i is class i out of the detection head.
#
# Spelled the way COCO spells them, spaces and all, because these names are matched
# against COCOeval's own categories -- `INDOOR_25` below is checked against this list for
# exactly that reason. `scripts/live_view_orin.py` keeps a hyphenated copy for drawing on
# frames; tests/test_orin_standalone_copies.py pins that the only difference is the
# hyphen, so the board cannot start naming a class something this list does not know.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]  # fmt: skip

# Reproduces mAP 0.3245647505782422 / mAP@50 0.49341168993493867 for
# runs/hydranet_joint_coco10/best.pt on val2017 with the COCO block at
# sample_ratio 0.1 -- see configs/eval_indoor25.yaml, which is the runnable form.
#
# Recovered verbatim from the session that produced that number; it was fixed before the
# evaluation ran, not fitted to it. No rationale for the individual entries was recorded
# and none is invented here -- what makes this list the right one is that it is the list
# the published figure was measured with. Anyone who wants a better-motivated subset
# should define a new one and measure it, rather than editing this and silently moving
# what 0.3246 refers to.
INDOOR_25 = [
    "person",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "bottle",
    "cup",
    "backpack",
    "handbag",
    "umbrella",
]
