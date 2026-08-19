"""Off-road label maps: RUGD and RELLIS-3D to unified terrain and traversability.

One pixel annotation drives two heads, which is the key to data efficiency here: the
terrain label is mapped to traversability through a policy table rather than being
labelled separately.

That policy is robot-specific. A quadruped can cross a log that a wheeled base cannot,
so ``TERRAIN_TO_TRAV`` is the first thing to change when the platform changes, and it
requires no re-annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..labels import IGNORE
from . import label_maps_cocostuff as _cs
from . import label_maps_indoor as _ind
from . import label_maps_retail as _ret
from . import label_maps_retail_objects as _obj

# Unified 12-class terrain, aligned with ``data.terrain_classes`` in the configs.
TERRAIN = {
    "void": 0,
    "dirt": 1,
    "grass": 2,
    "tree_bush": 3,
    "pavement_concrete": 4,
    "gravel_mulch": 5,
    "building_wall": 6,
    "sky": 7,
    "water": 8,
    "vehicle_object": 9,
    "rock": 10,
    "stairs_log": 11,
}

# Terrain to traversability. Quadruped defaults: grass and gravel are passable but
# warrant caution, stairs and logs are crossable with a dedicated gait.
TERRAIN_TO_TRAV = {
    0: 255,  # void -> ignore
    1: 2,  # dirt -> go
    2: 1,  # grass -> caution (may hide holes or debris)
    3: 0,  # tree/bush -> blocked
    4: 2,  # pavement/concrete -> go
    5: 1,  # gravel/mulch -> caution
    6: 0,  # building/wall -> blocked
    7: 0,  # sky -> blocked
    8: 0,  # water -> blocked
    9: 0,  # vehicle/object -> blocked
    10: 1,  # rock -> caution
    11: 1,  # stairs/log -> caution
}

# RUGD ships 24 classes as an RGB palette. See http://rugd.vision/ for the colormap.
RUGD_COLOR_TO_TERRAIN = {
    (0, 0, 0): TERRAIN["void"],
    (108, 64, 20): TERRAIN["dirt"],
    (255, 229, 204): TERRAIN["gravel_mulch"],  # sand
    (0, 102, 0): TERRAIN["grass"],
    (0, 255, 0): TERRAIN["tree_bush"],  # tree
    (0, 153, 153): TERRAIN["building_wall"],  # pole -> obstacle
    (0, 128, 255): TERRAIN["water"],
    (0, 0, 255): TERRAIN["sky"],
    (255, 255, 0): TERRAIN["vehicle_object"],  # vehicle
    (255, 0, 127): TERRAIN["vehicle_object"],  # container / generic object
    (64, 64, 64): TERRAIN["pavement_concrete"],  # asphalt
    (255, 128, 0): TERRAIN["gravel_mulch"],  # gravel
    (255, 0, 0): TERRAIN["building_wall"],  # building
    (153, 76, 0): TERRAIN["gravel_mulch"],  # mulch
    (102, 102, 0): TERRAIN["rock"],  # rock-bed
    (102, 0, 0): TERRAIN["stairs_log"],  # log
    (0, 255, 128): TERRAIN["tree_bush"],  # bicycle (rare)
    (204, 153, 255): TERRAIN["vehicle_object"],  # person
    (102, 0, 204): TERRAIN["building_wall"],  # fence
    (255, 153, 204): TERRAIN["tree_bush"],  # bush
    (0, 102, 102): TERRAIN["vehicle_object"],  # sign
    (153, 204, 255): TERRAIN["rock"],
    (102, 255, 255): TERRAIN["building_wall"],  # bridge
    (101, 101, 11): TERRAIN["pavement_concrete"],  # concrete
}

# RELLIS-3D uses integer ids.
#
# Three of these were inferred rather than read off the official table, and they are
# listed in RELLIS_UNVERIFIED below. A wrong entry here is not a crash: it relabels a
# whole class silently, and the run looks normal. Rather than leave that as a comment
# nobody reads at the point of use, `get_scheme("rellis")` warns and names them.
RELLIS_ID_TO_TERRAIN = {
    0: TERRAIN["void"],
    1: TERRAIN["dirt"],
    3: TERRAIN["grass"],
    4: TERRAIN["tree_bush"],
    5: TERRAIN["building_wall"],  # pole
    6: TERRAIN["water"],
    7: TERRAIN["sky"],
    8: TERRAIN["vehicle_object"],
    9: TERRAIN["vehicle_object"],  # object
    10: TERRAIN["pavement_concrete"],  # asphalt
    12: TERRAIN["building_wall"],  # building
    15: TERRAIN["stairs_log"],  # log
    17: TERRAIN["vehicle_object"],  # person
    18: TERRAIN["building_wall"],  # fence
    19: TERRAIN["tree_bush"],  # bush
    23: TERRAIN["gravel_mulch"],  # unverified, may be concrete
    27: TERRAIN["building_wall"],  # barrier
    31: TERRAIN["gravel_mulch"],  # unverified: puddle -- water is arguably right
    33: TERRAIN["gravel_mulch"],  # unverified: mud
    34: TERRAIN["gravel_mulch"],  # rubble
}


# Source ids whose meaning was inferred from the class name rather than confirmed
# against RELLIS-3D's published table. 31 (puddle) is the one that matters: mapping it
# to gravel_mulch makes standing water `caution`, while `water` would make it `blocked`
# -- a safety-relevant difference on a real platform.
RELLIS_UNVERIFIED = {
    23: "may be concrete rather than gravel",
    31: "puddle -> gravel_mulch (caution); water (blocked) may be correct",
    33: "mud",
}


def terrain_to_traversability(terrain_mask, trav_map=None):
    """Map an ``HxW`` terrain id array to traversability ids (0/1/2, 255 = ignore)."""
    import numpy as np

    out = np.full_like(terrain_mask, IGNORE)
    for t_id, trav in (trav_map or TERRAIN_TO_TRAV).items():
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
    "rugd": _scheme("rugd", "color", RUGD_COLOR_TO_TERRAIN, TERRAIN_TO_TRAV, TERRAIN),
    "rellis": _scheme("rellis", "id", RELLIS_ID_TO_TERRAIN, TERRAIN_TO_TRAV, TERRAIN),
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
    if name == "rellis":
        import warnings

        detail = "; ".join(f"{k}: {v}" for k, v in sorted(RELLIS_UNVERIFIED.items()))
        warnings.warn(
            f"label_map 'rellis' contains {len(RELLIS_UNVERIFIED)} mappings inferred from "
            f"class names rather than confirmed against the official table ({detail}). "
            "A wrong entry relabels a whole class without any error. Confirm them before "
            "reporting numbers from RELLIS-3D.",
            stacklevel=2,
        )
    if name == "retail_objects_migrated":
        import warnings

        # Same shape of warning as rellis, and for the same reason: what is lost here is
        # lost silently. These masks were drawn under a taxonomy where a column *is*
        # wall, so migrating them cannot separate the two -- the run trains a `column`
        # channel on nothing and reports a plausible-looking 0.000 sixty epochs later.
        warnings.warn(
            "label_map 'retail_objects_migrated' reads retail-13 masks under the object "
            "taxonomy. `column` (id 3) cannot survive the migration: those masks put "
            "columns inside `wall`, so this scheme supplies zero column pixels. Pair it "
            "with freshly drawn columns -- one polygon per fixed camera -- or the class "
            "is an empty channel.",
            stacklevel=2,
        )
    return SCHEMES[name]
