"""The site30k campaign taxonomy, and the readings that let it be trained on.

`tools/site30k/recipe.py` writes eleven ids and 255 into `datasets/site30k_v1/masks`.
It is a finer taxonomy than anything already registered here, in two places:

  * `fixture` is SPLIT into `display_table` and `shelf`. The split is a real judgement
    made per camera-date by the campaign, and where the two prompt families both fire the
    pixel is IGNORE rather than guessed -- so the distinction in these masks is the one
    the teachers were confident about, not a coin toss.
  * merchandise is PAINTED, as four classes (`laptop`, `tablet`, `phone`, `boxed_stock`),
    where `RETAIL_SURFACES` deliberately has none.

Neither of those is free to train, so this module registers the readings and leaves the
choice to the config rather than to whoever writes the dataset path:

  `site30k_native`       eleven classes as written. Use it when the finer taxonomy IS the
                         experiment. Note what it commits to: a dense head at stride 8 has
                         2-3 feature cells across a phone, which is the measured reason
                         `RETAIL_SURFACES` dropped `product` in the first place (see
                         label_maps_retail_objects.py). Training the four product classes
                         densely is that argument's counter-experiment, not its refutation.
                         It also needs `ade20k_site30k` / `retail_objects_to_site30k`
                         below, or the sibling datasets stop supervising the terrain head.

  `site30k_to_surfaces`  the same masks read as `RETAIL_SURFACES`, so site30k_v1 can be
                         mixed with datasets/retail_objects_batch02 and batch03 under the
                         configs that already exist. `display_table` and `shelf` merge back
                         to `fixture`, and the four product classes follow
                         `RETAIL_OBJECTS_ID_TO_SURFACES`'s decision that merchandise falls
                         back to what it stands on.

There is no `glass` id here yet. The campaign labels the glazing -- and the street behind
it -- as `wall`, which is wrong and is recorded in
docs/journal/2026-08-20-site30k-campaign-plan.md; the fix stamps a new id, and the id is
added here in the same change that starts writing it, not before.
"""

from __future__ import annotations

from .label_maps_retail_objects import (
    ADE20K_ID_TO_RETAIL_SURFACES,
    RETAIL_OBJECTS_ID_TO_SURFACES,
    RETAIL_SURFACES,
)

# The ids `tools/site30k/recipe.py` writes. `void` is 0 by convention and is never
# written by the campaign (measured: 0.00% of a 244-mask sample); it is listed so that a
# mask arriving from an annotation tool, which exports unlabelled background as 0, is read
# as "nobody labelled this" rather than as a trainable class.
SITE30K = {
    "void": 0,
    "floor": 1,
    "wall": 2,
    "column": 3,
    "display_table": 4,
    "shelf": 5,
    "person": 6,
    "laptop": 7,
    "tablet": 8,
    "phone": 9,
    "boxed_stock": 10,
}

_PRODUCTS = ("laptop", "tablet", "phone", "boxed_stock")

SITE30K_NATIVE_ID = {0: 255, **{v: v for v in SITE30K.values() if v}}

# Only the floor is walkable; everything else in a shop is something to walk around.
# Stated rather than omitted because `SegFolderDataset` hands a falsy trav table to the
# **off-road** policy, where id 2 means `grass` -- an omission here would not disable the
# second head, it would supervise it with the wrong table.
SITE30K_TO_TRAV = {
    SITE30K["void"]: 255,
    SITE30K["floor"]: 2,
    **{SITE30K[n]: 0 for n in SITE30K if n not in ("void", "floor")},
}

# Reading the campaign masks as RETAIL_SURFACES. Two merges, both already argued for
# elsewhere and neither invented here:
#
#   display_table, shelf -> fixture   the split this campaign added, folded back
#   laptop/tablet/phone/boxed_stock -> fixture
#                                     merchandise stands ON a fixture, so its pixels fall
#                                     back to fixture rather than to open floor -- the
#                                     asymmetry `RETAIL_OBJECTS_ID_TO_SURFACES` documents,
#                                     which is why `person` is kept and `product` is not.
SITE30K_ID_TO_SURFACES = {
    0: 255,
    SITE30K["floor"]: RETAIL_SURFACES["floor"],
    SITE30K["wall"]: RETAIL_SURFACES["wall"],
    SITE30K["column"]: RETAIL_SURFACES["column"],
    SITE30K["display_table"]: RETAIL_SURFACES["fixture"],
    SITE30K["shelf"]: RETAIL_SURFACES["fixture"],
    SITE30K["person"]: RETAIL_SURFACES["person"],
    **{SITE30K[n]: RETAIL_SURFACES["fixture"] for n in _PRODUCTS},
}

# Reading the SIBLING datasets under the campaign taxonomy -- the other direction.
#
# `site30k_native` puts eleven classes on the terrain head, and nothing else on disk speaks
# them: datasets/ADE20K, retail_objects_batch02 and batch03 are annotated as
# RETAIL_SURFACES. Without a table here, choosing the native scheme silently costs the
# terrain head every one of those datasets, because a 6-class annotation cannot supervise
# an 11-class head.
#
# Four of the six classes carry over exactly. `fixture` does NOT: this taxonomy split it
# into `display_table` and `shelf`, and a mask that predates the split does not record
# which one it was. That is a judgement the annotation never made, so it becomes IGNORE
# rather than being guessed into the majority class -- the same rule the campaign applies
# where its own two prompt families both fire.
SURFACES_ID_TO_SITE30K = {
    0: 255,
    RETAIL_SURFACES["floor"]: SITE30K["floor"],
    RETAIL_SURFACES["wall"]: SITE30K["wall"],
    RETAIL_SURFACES["column"]: SITE30K["column"],
    RETAIL_SURFACES["person"]: SITE30K["person"],
    RETAIL_SURFACES["fixture"]: 255,  # display_table vs shelf was never decided there
}


# The two datasets that actually sit on disk reach the campaign taxonomy by composition,
# so there is one statement of the fixture rule above and not three copies of it.
#   ADE20K masks carry ADE ids; retail_objects_batch02/03 masks carry RETAIL_OBJECTS ids.
def _through(surf: int) -> int:
    """IGNORE stays IGNORE; anything else is read through the surfaces table."""
    return 255 if surf == 255 else SURFACES_ID_TO_SITE30K[surf]


ADE20K_ID_TO_SITE30K = {
    ade: _through(surf) for ade, surf in ADE20K_ID_TO_RETAIL_SURFACES.items()
}
RETAIL_OBJECTS_ID_TO_SITE30K = {
    obj: _through(surf) for obj, surf in RETAIL_OBJECTS_ID_TO_SURFACES.items()
}
# `product` in those datasets lands on `fixture` and therefore on IGNORE here, which is the
# right answer and not a lucky one: this taxonomy paints four product classes and a mask
# that only ever said "merchandise" cannot say which.
assert RETAIL_OBJECTS_ID_TO_SITE30K[5] == 255  # RETAIL_OBJECTS["product"]

# Every id the recipe can write must have a reading. A class added above and forgotten
# below would otherwise train as `void`, silently, on whichever share of pixels it holds.
assert set(SITE30K_ID_TO_SURFACES) == set(SITE30K.values()), (
    "SITE30K_ID_TO_SURFACES does not cover the taxonomy: "
    f"{set(SITE30K.values()) - set(SITE30K_ID_TO_SURFACES)}"
)
assert set(SITE30K_TO_TRAV) == set(SITE30K.values())
# `person` is 6 in the campaign and 5 in RETAIL_SURFACES. If that ever coincides, this
# table has become an identity map and the renumbering it exists for has been lost.
assert SITE30K["person"] != RETAIL_SURFACES["person"]
