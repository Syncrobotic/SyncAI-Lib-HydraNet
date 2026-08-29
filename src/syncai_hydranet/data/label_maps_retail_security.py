"""One detection vocabulary for the retail *and* security questions on a store camera.

`configs/hydranet_retail_products.yaml` states the blocker this file removes, and states
it as a refusal to run rather than as a wish:

    `CocoDetDataset` remaps category ids to contiguous 0-based labels per dataset. Two
    detection datasets in one config therefore both produce a label `0` -- `person` in
    COCO's numbering and `boxed_stock` in this one -- and one head cannot hold both
    meanings.

That is exactly true of *per-dataset* numbering, and it is what keeps merchandise
detection in its own run today. A security deployment cannot accept that split: every
event the product asks for -- zone intrusion, occupancy, loitering, an object left
behind -- is keyed on `person`, and the same camera has to answer "what merchandise is
on this shelf" at the same time. Two networks on one camera is two trunks, two exports
and two latency budgets for one 6.7 ms frame.

So the numbering stops being per-dataset. A **vocabulary** names the classes and assigns
their ids once; every detection source maps its own category *names* into it. COCO's
`person` and the site's `boxed_stock` are then 0 and 2 in both files, because they were
0 and 2 before either file was opened.

---------------------------------------------------------------------------
WHAT THIS DOES NOT FIX, AND IT IS THE EXPENSIVE HALF

Sharing a head does not share supervision. A site frame full of shoppers carries **no
person boxes**, so every point on every shopper is a labelled negative for channel 0 --
which is not dilution but suppression, and this project has already measured what that
costs. `config_schema.minority_sourced_terrain_classes` documents the segmentation case:
ADE20K at 90.2% of steps containing zero `product` pixels held that class at IoU 0.000
for 22 consecutive epochs, on a run that looked entirely normal.

Two answers, and this file ships the cheap one:

1. **Mask the loss** -- a batch supervises only the channels its dataset can label. That
   is `DetVocab.class_mask`, consumed by `FCOSLoss`, and it turns a false negative into
   no gradient at all. It costs nothing and it is not a substitute for (2).
2. **Annotate person boxes on the site frames.** Masking means `person` is trained on
   COCO alone: web photography, eye level, uncompressed. The 24 store cameras are
   overhead, down-pitched and h.264, and `RETAIL.md` measured what that gap does to
   a class ADE20K *did* supply. Until site person boxes exist, person detection on site
   footage is a bootstrap and every number downstream of it -- tracks, dwell, occupancy,
   every event in `analytics/events.py` -- inherits that.

---------------------------------------------------------------------------
WHY FOUR CLASSES AND NOT SEVEN

`bag` is here because the security questions want it: an abandoned object in a shop is
overwhelmingly a bag, and COCO supplies `backpack`, `handbag` and `suitcase` in volume.
Three source categories, one channel, because nothing downstream asks which of the three
it was and three thin channels learn worse than one thick one.

What is deliberately **absent**: `trolley`, `basket`, `staff`. Every one of them is a
real ask and none of them has a source. COCO has no shopping trolley and no basket, and
`staff` is not a visual class at all -- it is an attribute of a person (uniform), which
belongs on the second stage's crop encoder rather than in a box vocabulary. Adding an
unsourced channel is not free and this project has the number: an empty channel is
"trained on nothing and reported as IoU 0.000" while the run looks normal. They enter the
vocabulary when annotation exists, and `UNSOURCED_CANDIDATES` keeps the list from being
forgotten rather than keeping it in the head.

Ids are frozen once site boxes are exported against them, exactly as
`label_maps_retail_objects.RETAIL_OBJECTS` is frozen. Appending is safe; renumbering
invalidates every box drawn and every checkpoint trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The vocabulary. Order defines the id, and the id is what a checkpoint holds.
RETAIL_SECURITY_DET = {
    "person": 0,  # every security event is keyed on this channel
    "bag": 1,  # backpack / handbag / suitcase -- the abandoned-object shape
    "boxed_stock": 2,  # site: merchandise in its box
    "device": 3,  # site: demo units, handsets, laptops
}

# Source category name -> vocabulary class. Names, not ids, because a category id is a
# property of one annotation file and a name is a property of the thing annotated. COCO's
# `person` is id 1; the site export could give it any id it liked and this table would
# not care.
#
# A source category absent from this table is dropped rather than mapped to a nearby
# class. `label_maps_retail_objects` argues the same point for `stairs`: guessing a class
# for something the taxonomy has no answer to trains a wrong label, and a wrong label is
# worse than a gap because a gap is visible.
DET_ALIASES = {
    "person": "person",
    "backpack": "bag",
    "handbag": "bag",
    "suitcase": "bag",
    "boxed_stock": "boxed_stock",
    "device": "device",
}

# Named so they are not silently dropped from the roadmap, and kept out of the head so
# they are not silently trained on nothing. Each entry is (class, what it would need).
UNSOURCED_CANDIDATES = (
    ("trolley", "no public dataset labels a shopping trolley; site annotation only"),
    ("basket", "same, and it is the more common of the two in these stores"),
    (
        "staff",
        "not a box class at all -- uniform is an attribute of a person, so it belongs on "
        "the second-stage crop encoder where PA-100K and RAP v2 already label clothing",
    ),
)


@dataclass(frozen=True)
class DetVocab:
    """A detection class list several datasets can share, and the maps that get them there.

    Frozen, and `aliases` is read but never written, so two datasets built from the same
    vocabulary cannot drift apart the way two per-dataset numberings do.
    """

    name: str
    classes: tuple[str, ...]
    aliases: dict[str, str]

    def label_of(self, category_name: str) -> int | None:
        """The vocabulary id for a source category, or None if this vocabulary drops it."""
        cls = self.aliases.get(category_name)
        return None if cls is None else self.classes.index(cls)

    def class_mask(self, labels) -> np.ndarray:
        """1.0 for each class the caller can label, 0.0 for the rest -- as a [C] float32.

        This is the array that decides whether an unlabelled shopper in a site frame is a
        negative for `person` or is nothing at all. It multiplies the classification loss
        channel-wise in `FCOSLoss`, so a zero here is not a small gradient, it is no
        gradient: the channel is neither rewarded nor punished on this dataset's images.

        Deliberately **not** applied to regression or centerness. Those are computed at
        positive points only, and a positive point exists only where this dataset drew a
        box, so they are already restricted to what it can label.
        """
        mask = np.zeros(len(self.classes), dtype=np.float32)
        for label in labels:
            mask[int(label)] = 1.0
        return mask


DET_VOCABS = {
    "retail_security": DetVocab(
        name="retail_security",
        classes=tuple(RETAIL_SECURITY_DET),
        aliases=dict(DET_ALIASES),
    ),
}


def get_det_vocab(name: str) -> DetVocab:
    if name not in DET_VOCABS:
        raise ValueError(
            f"unknown det_vocab {name!r}; known: {', '.join(sorted(DET_VOCABS))}. "
            "A vocabulary is added here rather than in a config, because its ids are "
            "frozen into every box exported against it."
        )
    return DET_VOCABS[name]
