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

from collections.abc import Iterable, Sequence

# COCO's 80 categories in sorted-category-id order, which is the order CocoDetDataset
# assigns contiguous labels in: index i is class i out of the detection head.
#
# Spelled the way COCO spells them, spaces and all, because these names are matched
# against COCOeval's own categories -- `INDOOR_25` below is checked against this list for
# exactly that reason. `scripts/live_view_orin.py` kept a hyphenated copy for drawing on
# frames and a test pinned that the only difference was the hyphen; both went with the
# Orin on 2026-08-28 (`git show f64520c:scripts/live_view_orin.py`). This list is now the
# only spelling, which is the outcome that guard existed to protect.
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


# ---------------------------------------------------------------------------
# Reading COCO's answers as retail nouns.
#
# **This adds no knowledge and trains nothing.** It is a rename, and the distinction
# matters: everything below is the existing 80-way head's existing output, relabelled
# into the three groups configs/hydranet_retail_objects.yaml cares about. It cannot find
# a product the head did not already fire on, and it inherits every mistake the head
# makes. What it does is stop a human reading `73:0.42` off a frame and having to
# remember that 73 is `book` and that `book` in a phone shop is a boxed handset.
#
# The audit that produced it (runs/review_20260816/, 18 fixed cameras, 1,620 frames)
# swept the score threshold on two cameras, 40 frames each:
#
#     threshold   Taichung-cam01                        Kaohsiung-cam08
#       0.05      book 914, bottle 783, fridge 476      book 1683, fridge 384, oven 273
#       0.15      person 112, chair 108, fridge 77      book 311, toilet 39, mouse 19
#       0.25      person 72, fridge 40                  -- nothing
#       0.35      person 62                             -- nothing
#
# Kaohsiung-cam08 is an Apple store: a round podium of MacBooks, wall shelving of boxed
# accessories, two structural columns. There is no `laptop` in that column at any
# threshold and there are 1,683 `book`. The head is finding the merchandise and naming
# it by the nearest shape word it owns -- boxes as `book`, accessories as `bottle`,
# demo units and large fixtures as `refrigerator` or `tv`.
#
# So the grouping is by what the word means *in a shop*, not by what it means in COCO:
RETAIL_OBJECT_GROUP = {
    # Merchandise: small, on a fixture, and what a shopper picks up.
    "book": "product",  # boxed product -- the single strongest signal in the sweep
    "bottle": "product",
    "cup": "product",
    "laptop": "product",
    "cell phone": "product",
    "mouse": "product",
    "keyboard": "product",
    "remote": "product",
    "clock": "product",
    "vase": "product",
    "teddy bear": "product",
    "scissors": "product",
    "hair drier": "product",
    "toothbrush": "product",
    # Screens are the genuinely ambiguous group and they are called product here on
    # purpose: in the stores this project films, a screen on a table is a demo unit for
    # sale. In a supermarket the same box would be signage and this line would be wrong,
    # which is why it is one line to change rather than a training decision.
    "tv": "product",
    # Fixtures: the furniture merchandise sits on. `refrigerator`, `oven` and `toilet`
    # are here because that is what they fire on in this footage -- tall cabinets,
    # podiums and bins -- not because a shop contains white goods.
    "dining table": "fixture",
    "chair": "fixture",
    "couch": "fixture",
    "bench": "fixture",
    "bed": "fixture",
    "refrigerator": "fixture",
    "oven": "fixture",
    "microwave": "fixture",
    "sink": "fixture",
    "toilet": "fixture",
    "potted plant": "fixture",
    "person": "person",
}

# Belongings are deliberately absent: a backpack, handbag, suitcase or umbrella in a
# shop belongs to a customer, and calling it merchandise would put a shopper's bag in
# the stock count. They stay under their own COCO names.
RETAIL_BELONGINGS = ["backpack", "handbag", "suitcase", "umbrella", "tie"]


def retail_group(label_id: int) -> str | None:
    """The retail group for one detection-head label, or None if it has no reading.

    ``label_id`` is the head's contiguous index, i.e. an index into ``COCO_NAMES``.
    """
    if not 0 <= label_id < len(COCO_NAMES):
        return None
    return RETAIL_OBJECT_GROUP.get(COCO_NAMES[label_id])


def retail_box_label(label_id: int) -> str:
    """What to draw on a box: the retail group where there is one, the COCO name where
    there is not, and the bare index only if the label is off the end of the taxonomy.

    Both halves are shown for a grouped class -- ``product/book`` -- because the COCO
    word is the evidence for the group and hiding it would make a wrong grouping
    unfalsifiable from the rendered frame.
    """
    if not 0 <= label_id < len(COCO_NAMES):
        return str(label_id)
    name = COCO_NAMES[label_id]
    group = RETAIL_OBJECT_GROUP.get(name)
    return f"{group}/{name}" if group and group != name else name


# ---------------------------------------------------------------------------
# Export-time narrowing.
#
# RETAIL.md §4 draws the distinction these lists exist for: **train on 80, narrow
# at export.** `data.datasets[].classes` narrows at training time, which is the opposite
# trade -- it throws away COCO supervision the shared trunk gets for free, and buys the
# robot nothing, because a head trained on 80 and a head trained on 8 cost the same to
# run. What costs is decoding: on the AGX Orin, post-processing was 16.33 ms of a 37.8 ms
# frame, nearly all of it the sigmoid over 80 classes at 6,825 positions.
#
# So these are deployment decisions, applied to a checkpoint that already saw all 80.

# The eight classes RETAIL.md §4 names as changing a shop robot's behaviour. This
# is the *motion* list: what a planner has to react to.
ROBOT_8 = [
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "chair",
    "potted plant",
]

# The analytics list, derived rather than written out: everything RETAIL_OBJECT_GROUP
# gives a retail reading to, plus the belongings it deliberately leaves ungrouped.
#
# Derived on purpose. Written out, it would be a second copy of the grouping table that
# drifts the first time someone decides a `tv` in a phone shop is signage after all --
# and the failure would be silent, because an engine that never emits `tv` and a mapping
# that never reads it look identical from the frame.
#
# **Note it is not eight.** ROBOT_8 would delete `book`, which the cam08 sweep found
# 1,683 of and which is the single strongest merchandise signal the head produces. The
# two lists narrow for different deployments and neither is a default.
RETAIL_ANALYTICS = sorted(
    set(RETAIL_OBJECT_GROUP) | set(RETAIL_BELONGINGS), key=COCO_NAMES.index
)

EXPORT_SUBSETS = {
    "robot_8": ROBOT_8,
    "retail_analytics": RETAIL_ANALYTICS,
    "indoor_25": INDOOR_25,
}


def head_order(names: Iterable[str]) -> list[str]:
    """The order a detection head assigns these classes -- which is not the order they
    were written in.

    ``CocoDetDataset`` builds its label map from ``sorted(getCatIds(...))``, so the head's
    channel order is COCO category-id order regardless of how the config listed them.
    ``INDOOR_25`` below is written in neither order, and ``configs/eval_indoor25.yaml``
    has always relied on this sort happening inside the dataset.

    Getting it wrong is the quiet kind of wrong: every channel still decodes, every box
    still gets a name, and the names are someone else's.
    """
    unknown = sorted(n for n in names if n not in COCO_NAMES)
    if unknown:
        raise ValueError(f"not COCO category names: {', '.join(unknown)}")
    return sorted(set(names), key=COCO_NAMES.index)


def resolve_export_subset(spec: str) -> list[str]:
    """A named subset or a comma-separated list of COCO names, in head order."""
    if spec in EXPORT_SUBSETS:
        return head_order(EXPORT_SUBSETS[spec])
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise ValueError(
            f"empty class list. Give COCO names separated by commas, or one of: "
            f"{', '.join(sorted(EXPORT_SUBSETS))}"
        )
    return head_order(names)


def narrow_indices(keep: Iterable[str], trained: Sequence[str]) -> list[int]:
    """Channel indices of ``keep`` within a head trained on ``trained``.

    ``trained`` is the head's own class list -- ``COCO_NAMES`` for the usual 80-class
    head, or the dataset's ``classes`` in head order for a head that was already narrowed
    at training time. Narrowing an already-narrow head is legitimate; narrowing it to a
    class it never saw is not, and refusing here is the difference between an empty
    channel and one nobody notices is empty.
    """
    trained = list(trained)
    index = {name: i for i, name in enumerate(trained)}
    missing = [n for n in keep if n not in index]
    if missing:
        raise ValueError(
            f"class(es) {', '.join(sorted(missing))} are not in this head's {len(trained)} "
            f"trained classes, so no channel of it predicts them. Trained: "
            f"{', '.join(trained)}"
        )
    return [index[n] for n in keep]
