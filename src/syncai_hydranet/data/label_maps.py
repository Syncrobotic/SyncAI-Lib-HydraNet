"""The label-scheme registry: source annotations to unified terrain and traversability.

One pixel annotation drives two heads, which is the key to data efficiency here: the
terrain label is mapped to traversability through a policy table rather than being
labelled separately. Each scheme carries its own terrain vocabulary and its own policy
table; `SCHEMES` is the whole set, and `get_scheme` is how a config reaches one.

**The off-road schemes are gone, retired 2026-09-04 with the product line that wanted
them.** `rugd` and `rellis` mapped a 12-class outdoor vocabulary -- dirt, grass, rock,
stairs_log -- shared with nothing else here: the shipped terrain head emits `void, floor,
wall, column, fixture, person`, which overlaps it only at `void`. Their datasets have not
been on disk since the quadruped line was deleted (2026-08-19, `cc80fc3`), so nothing
could have trained on them either. Readable at
`git show 9215a67:src/syncai_hydranet/data/label_maps.py` if outdoor terrain comes back;
it will need a scheme that also carries floor, wall and person, so that will be a third
vocabulary rather than this one restored.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..labels import IGNORE
from . import label_maps_cocostuff as _cs
from . import label_maps_indoor as _ind
from . import label_maps_retail as _ret
from . import label_maps_retail_objects as _obj
from . import label_maps_site30k as _s30


def terrain_to_traversability(terrain_mask, trav_map):
    """Map an ``HxW`` terrain id array to traversability ids (0/1/2, 255 = ignore).

    **`trav_map` is required.** It defaulted to the off-road policy table until
    2026-09-04, so a caller that forgot it silently scored a retail mask against a
    quadruped's idea of what is crossable. Every real caller already passed one -- the
    default was only ever reachable by omission -- and the policy is a property of the
    scheme, which `get_scheme(...).trav` supplies.
    """
    import numpy as np

    out = np.full_like(terrain_mask, IGNORE)
    for t_id, trav in trav_map.items():
        out[terrain_mask == t_id] = trav
    return out


# ---------------------------------------------------------------------------
# Scheme registry
#
# A scheme is the whole path from a dataset's native annotation, through unified
# terrain, to traversability. Select one with ``label_map: <name>`` in a dataset's
# config block. Omitting it keeps the legacy auto-detection (RGB palette vs integer
# ids), so existing configs are unaffected.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelScheme:
    name: str
    fmt: str  # 'color' (RGB palette) or 'id' (single channel integers)
    mapping: dict  # (r,g,b) -> terrain, or source_id -> terrain
    trav: dict  # terrain -> traversability
    classes: tuple  # unified terrain names; index is the class id

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def _scheme(name, fmt, mapping, trav, terrain_dict):
    return LabelScheme(
        name, fmt, mapping, trav, tuple(sorted(terrain_dict, key=terrain_dict.get))
    )


SCHEMES: dict[str, LabelScheme] = {
    "ade20k_indoor": _scheme(
        "ade20k_indoor",
        "id",
        _ind.ADE20K_ID_TO_INDOOR,
        _ind.INDOOR_TERRAIN_TO_TRAV,
        _ind.INDOOR_TERRAIN,
    ),
    "indoor_native": _scheme(
        "indoor_native",
        "id",
        _ind.INDOOR_NATIVE_ID,
        _ind.INDOOR_TERRAIN_TO_TRAV,
        _ind.INDOOR_TERRAIN,
    ),
    # Retail is the indoor 12 with display_fixture appended as 12; ids 0-11 are
    # unchanged, so an indoor checkpoint warm-starts and indoor masks stay readable.
    "ade20k_retail": _scheme(
        "ade20k_retail",
        "id",
        _ret.ADE20K_ID_TO_RETAIL,
        _ret.RETAIL_TERRAIN_TO_TRAV,
        _ret.RETAIL_TERRAIN,
    ),
    "retail_native": _scheme(
        "retail_native",
        "id",
        _ret.RETAIL_NATIVE_ID,
        _ret.RETAIL_TERRAIN_TO_TRAV,
        _ret.RETAIL_TERRAIN,
    ),
    # ----------------------------------------------------------------- retail objects
    #
    # A second retail taxonomy answering "what object is this", where the one above
    # answers "can the robot step here". Separate on purpose: this one merges the two
    # fixture classes and splits `column` out of `wall`, and neither is expressible
    # under the ids-0-11-are-indoor invariant that tests/test_retail_scheme.py pins.
    # See label_maps_retail_objects.py for what the site audit found.
    "ade20k_retail_objects": _scheme(
        "ade20k_retail_objects",
        "id",
        _obj.ADE20K_ID_TO_RETAIL_OBJECTS,
        _obj.RETAIL_OBJECTS_TO_TRAV,
        _obj.RETAIL_OBJECTS,
    ),
    "retail_objects_native": _scheme(
        "retail_objects_native",
        "id",
        _obj.RETAIL_OBJECTS_NATIVE_ID,
        _obj.RETAIL_OBJECTS_TO_TRAV,
        _obj.RETAIL_OBJECTS,
    ),
    # The same taxonomy with `product` taken out of segmentation and left to the
    # detection head, which is the instrument that can actually resolve it: merchandise
    # is 2-3 feature cells across at stride 8, and a dense head has too few cells to put
    # an edge in. `person` stays, because unlike `product` its pixels would fall back to
    # `floor` and open walkable space where a shopper is standing. See
    # label_maps_retail_objects.py for the asymmetry, which is the whole argument.
    #
    # `retail_surfaces_from_objects` reads the site masks already on disk -- which carry
    # `RETAIL_OBJECTS` ids -- so dropping the class costs no re-annotation. It is a
    # renumbering as well as a merge: `person` is 6 there and 5 here.
    "ade20k_retail_surfaces": _scheme(
        "ade20k_retail_surfaces",
        "id",
        _obj.ADE20K_ID_TO_RETAIL_SURFACES,
        _obj.RETAIL_SURFACES_TO_TRAV,
        _obj.RETAIL_SURFACES,
    ),
    "retail_surfaces_native": _scheme(
        "retail_surfaces_native",
        "id",
        _obj.RETAIL_SURFACES_NATIVE_ID,
        _obj.RETAIL_SURFACES_TO_TRAV,
        _obj.RETAIL_SURFACES,
    ),
    "retail_surfaces_from_objects": _scheme(
        "retail_surfaces_from_objects",
        "id",
        _obj.RETAIL_OBJECTS_ID_TO_SURFACES,
        _obj.RETAIL_SURFACES_TO_TRAV,
        _obj.RETAIL_SURFACES,
    ),
    # The site30k campaign taxonomy: eleven ids as written, or the same masks folded
    # back onto RETAIL_SURFACES so they can be mixed with batch02/batch03 under the
    # configs that already exist. label_maps_site30k.py argues for both readings.
    "site30k_native": _scheme(
        "site30k_native",
        "id",
        _s30.SITE30K_NATIVE_ID,
        _s30.SITE30K_TO_TRAV,
        _s30.SITE30K,
    ),
    "site30k_to_surfaces": _scheme(
        "site30k_to_surfaces",
        "id",
        _s30.SITE30K_ID_TO_SURFACES,
        _obj.RETAIL_SURFACES_TO_TRAV,
        _obj.RETAIL_SURFACES,
    ),
    # The other direction: read a RETAIL_SURFACES dataset (ADE20K, batch02, batch03) so it
    # can supervise an 11-class site30k head. `fixture` becomes IGNORE, because the split
    # into display_table/shelf is a judgement those masks never made.
    "surfaces_to_site30k": _scheme(
        "surfaces_to_site30k",
        "id",
        _s30.SURFACES_ID_TO_SITE30K,
        _s30.SITE30K_TO_TRAV,
        _s30.SITE30K,
    ),
    "ade20k_site30k": _scheme(
        "ade20k_site30k",
        "id",
        _s30.ADE20K_ID_TO_SITE30K,
        _s30.SITE30K_TO_TRAV,
        _s30.SITE30K,
    ),
    "retail_objects_to_site30k": _scheme(
        "retail_objects_to_site30k",
        "id",
        _s30.RETAIL_OBJECTS_ID_TO_SITE30K,
        _s30.SITE30K_TO_TRAV,
        _s30.SITE30K,
    ),
    # Reads the retail-13 site masks already collected (SAM 3 consensus, pilot) under
    # the object taxonomy, so that work carries over. `column` does not survive the
    # trip -- get_scheme() says so out loud rather than leaving it to an IoU of 0.000.
    "retail_objects_migrated": _scheme(
        "retail_objects_migrated",
        "id",
        _obj.RETAIL_ID_TO_OBJECTS,
        _obj.RETAIL_OBJECTS_TO_TRAV,
        _obj.RETAIL_OBJECTS,
    ),
    # COCO-Stuff rides on the images already in datasets/coco. Its PNG values sit one
    # below the ids in the dataset's own labels.txt, and `person` is value 0 -- both are
    # handled inside the module and pinned by tests/test_cocostuff_scheme.py.
    "cocostuff_indoor": _scheme(
        "cocostuff_indoor",
        "id",
        _cs.COCOSTUFF_ID_TO_INDOOR,
        _ind.INDOOR_TERRAIN_TO_TRAV,
        _ind.INDOOR_TERRAIN,
    ),
    "cocostuff_retail": _scheme(
        "cocostuff_retail",
        "id",
        _cs.COCOSTUFF_ID_TO_RETAIL,
        _ret.RETAIL_TERRAIN_TO_TRAV,
        _ret.RETAIL_TERRAIN,
    ),
}


def get_scheme(name: str) -> LabelScheme:
    """Look up a scheme, warning about any mapping in it that was never verified.

    The warning fires where the scheme is used rather than where it is defined, because
    a comment in a table nobody opens does not reach the person about to train on it.
    """
    if name not in SCHEMES:
        raise ValueError(f"unknown label_map: {name}, available: {list(SCHEMES)}")
    if name == "retail_objects_migrated":
        import warnings

        # Same shape of warning as the rellis one this used to sit beside (retired
        # 2026-09-04 with the off-road taxonomy), and for the same reason: what is lost
        # here is lost silently. These masks were drawn under a taxonomy where a column
        # *is* wall, so migrating them cannot separate the two -- the run trains a
        # `column` channel on nothing and reports a plausible-looking 0.000 sixty epochs
        # later.
        warnings.warn(
            "label_map 'retail_objects_migrated' reads retail-13 masks under the object "
            "taxonomy. `column` (id 3) cannot survive the migration: those masks put "
            "columns inside `wall`, so this scheme supplies zero column pixels. Pair it "
            "with freshly drawn columns -- one polygon per fixed camera -- or the class "
            "is an empty channel.",
            stacklevel=2,
        )
    return SCHEMES[name]
