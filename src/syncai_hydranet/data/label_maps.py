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

from . import label_maps_indoor as _ind

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
# TODO: entries marked below were inferred and should be checked against the official
# mapping before training on real RELLIS data.
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
    23: TERRAIN["gravel_mulch"],  # TODO: verify, may be concrete
    27: TERRAIN["building_wall"],  # barrier
    31: TERRAIN["gravel_mulch"],  # TODO: puddle, consider mapping to water
    33: TERRAIN["gravel_mulch"],  # mud
    34: TERRAIN["gravel_mulch"],  # rubble
}


def terrain_to_traversability(terrain_mask, trav_map=None):
    """Map an ``HxW`` terrain id array to traversability ids (0/1/2, 255 = ignore)."""
    import numpy as np

    out = np.full_like(terrain_mask, 255)
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
}


def get_scheme(name: str) -> LabelScheme:
    if name not in SCHEMES:
        raise ValueError(f"unknown label_map: {name}, available: {list(SCHEMES)}")
    return SCHEMES[name]
