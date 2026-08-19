"""Retail label maps for shops, supermarkets and showrooms.

A retail floor is an indoor floor with one thing on it that the indoor scheme has no
word for: the fixture that merchandise sits on. Everything else a store robot meets --
polished floor, glass storefront, escalator, wet floor after a mop, people -- is already
in ``label_maps_indoor.py``.

So this is not a new taxonomy. **Ids 0-11 are the indoor ones, unchanged**, and
``display_fixture`` is appended as 12. That buys two things:

* a retail run can warm-start from an indoor checkpoint by re-initialising one output
  channel rather than the whole terrain head;
* a mask annotated under ``indoor_native`` is readable under the retail scheme, so the
  site data already collected does not have to be re-exported.

``tests/test_retail_scheme.py`` pins that alignment, because it is exactly the property
a later tidy-up of the class list would break without any test failing.

Two things a store needs that are deliberately **not** classes here:

* **"walkable within 5 m."** Distance is geometry, not appearance. Baking it into labels
  asks an annotator to tell 4.8 m of floor from 5.2 m of the same floor, dies the moment
  the camera moves on its mount, and freezes a product parameter into the weights. A
  retail floor is flat and the extrinsics are fixed, so a ground-plane homography turns
  the segmentation into metric distance in post-processing -- or the LiDAR does, since
  the robot already carries one.
* **The merchandise *zone* on the floor.** It has no visual boundary; two annotators
  drawing "the area belonging to this display" will not agree, and the disagreement
  trains as noise. Derive it in bird's-eye view from the fixture footprint instead,
  where "within 1.5 m of a fixture" is a number rather than an opinion.

See [docs/RETAIL.md](../../../docs/RETAIL.md).
"""

from __future__ import annotations

from .label_maps_indoor import INDOOR_TERRAIN, INDOOR_TERRAIN_TO_TRAV

# The indoor 12, plus one. Do not renumber: 0-11 must stay identical to INDOOR_TERRAIN.
RETAIL_TERRAIN = {
    **INDOOR_TERRAIN,
    "display_fixture": 12,  # shelving / gondola / display table / counter / rack
}

RETAIL_TERRAIN_TO_TRAV = {
    **INDOOR_TERRAIN_TO_TRAV,
    12: 0,  # display_fixture -> blocked
}

# In-house retail annotations. 0 is ignore for the same reason as indoor: annotation
# tools export unlabelled background as 0, and the loss ignores 255 alone.
RETAIL_NATIVE_ID = {0: 255, **{v: v for v in RETAIL_TERRAIN.values() if v}}


# ---------------------------------------------------------------------------
# ADE20K -> retail 13.
#
# Identical to the indoor map except that the fixtures below are pulled out of
# obstacle_furniture into display_fixture. It is a bootstrap, not a substitute for store
# footage: ADE20K's indoor images are homes and restaurants, so what it can offer is the
# *shape* of a waist-height horizontal surface holding objects, not a shop.
#
# `table` (16) is deliberately NOT among them, though a display table is the exact thing
# Paul asked for. ADE20K's tables are domestic and dining, and mapping them here would
# teach the model that a dining table is a display fixture -- a public-data proxy that is
# nearly the right thing, which is the failure this project has already paid for once.
# Real display tables come from store annotation. Both classes are `blocked` either way,
# so the cost of leaving them out is semantic, not a safety regression.
# ---------------------------------------------------------------------------
_ADE_FIXTURES = {
    25: "shelf",
    46: "counter",
    56: "case",  # display case
    63: "bookcase",
    71: "countertop",
    78: "bar",
    89: "booth",
    100: "buffet",
}


def _build_ade20k_map() -> dict[int, int]:
    from .label_maps_indoor import ADE20K_ID_TO_INDOOR

    out = dict(ADE20K_ID_TO_INDOOR)
    for ade_id in _ADE_FIXTURES:
        out[ade_id] = RETAIL_TERRAIN["display_fixture"]
    return out


ADE20K_ID_TO_RETAIL = _build_ade20k_map()
