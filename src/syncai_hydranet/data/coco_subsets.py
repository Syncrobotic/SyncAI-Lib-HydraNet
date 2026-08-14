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
