"""Indoor label maps for lobbies, corridors and factory floors.

Lives alongside the off-road maps in ``label_maps.py``; a config picks one via
``label_map`` and the two never interfere.

Indoor traversability is closer to binary than off-road (floor vs not-floor), but the
risk concentrates in a few rare, dangerous classes: glass, descending stairs and wet
floor. Those may cover well under 1% of pixels, which is why the indoor config raises
``dice_weight``.
"""

from __future__ import annotations

# Unified 12-class indoor terrain, aligned with ``data.terrain_classes`` in
# configs/hydranet_indoor.yaml.
INDOOR_TERRAIN = {
    "void": 0,
    "floor_hard": 1,  # tile / stone / concrete / epoxy
    "floor_soft": 2,  # carpet / rubber mat
    "floor_metal": 3,  # grating / steel plate
    "wet_slippery": 4,  # standing water / freshly mopped
    "stairs": 5,  # stairs / escalator
    "threshold_ramp": 6,  # door sill / ramp / level change
    "wall": 7,  # wall / ceiling / column
    "glass": 8,  # glass door / curtain wall / mirror
    "door": 9,  # operable door
    "obstacle_furniture": 10,  # furniture / equipment / railing
    "person": 11,
}

# Indoor quadruped defaults. The three entries marked REVIEW depend on the platform.
INDOOR_TERRAIN_TO_TRAV = {
    0: 255,  # void -> ignore
    1: 2,  # floor_hard -> go
    2: 2,  # floor_soft -> go
    3: 1,  # floor_metal -> caution (REVIEW: foot size vs grating aperture)
    4: 1,  # wet_slippery -> caution (REVIEW: set 0 if the gait cannot handle slip)
    5: 1,  # stairs -> caution (REVIEW: set 0 if stair climbing is unsupported)
    6: 1,  # threshold_ramp -> caution
    7: 0,  # wall -> blocked
    8: 0,  # glass -> blocked (most dangerous: reads as an open path)
    9: 0,  # door -> blocked when closed; the opening becomes floor once ajar
    10: 0,  # obstacle -> blocked
    11: 0,  # person -> blocked
}

# In-house annotations: the labelling tool emits indoor class ids directly, so no
# translation is needed. Pin this table in the annotation spec so annotators and
# training cannot drift apart.
INDOOR_NATIVE_ID = {v: v for v in INDOOR_TERRAIN.values()}


# ---------------------------------------------------------------------------
# ADE20K SceneParsing (ADEChallengeData2016), 150 classes -> indoor 12.
#
# Annotations are single-channel PNGs: 0 is ignore, 1..150 are classes.
#
# TODO: indices below follow the standard objectInfo150.csv ordering but were written
# from memory; verify once against the dataset before the first real training run.
# Unlisted ids fall through to void -> ignore, so a mistake degrades coverage rather
# than corrupting labels.
#
# NOTE: ADE20K contains many purely outdoor images. Filter to indoor scenes using the
# official sceneCategories.txt first, otherwise sky and vegetation dominate the frame
# (they map to ignore, but still waste compute).
# ---------------------------------------------------------------------------
_T = INDOOR_TERRAIN

ADE20K_ID_TO_INDOOR = {
    # Walkable ground
    4: _T["floor_hard"],  # floor
    7: _T["floor_hard"],  # road
    12: _T["floor_hard"],  # sidewalk
    53: _T["floor_hard"],  # path
    29: _T["floor_soft"],  # rug
    # Level changes
    54: _T["stairs"],  # stairs
    60: _T["stairs"],  # stairway
    97: _T["stairs"],  # escalator
    122: _T["stairs"],  # step
    # Boundaries
    1: _T["wall"],  # wall
    6: _T["wall"],  # ceiling
    43: _T["wall"],  # column
    23: _T["wall"],  # painting
    44: _T["wall"],  # signboard
    101: _T["wall"],  # poster
    145: _T["wall"],  # bulletin board
    19: _T["wall"],  # curtain
    64: _T["wall"],  # blind
    # Glass and mirrors: the highest-consequence indoor failure mode
    9: _T["glass"],  # windowpane
    28: _T["glass"],  # mirror
    59: _T["glass"],  # screen door
    148: _T["glass"],  # glass
    15: _T["door"],  # door
    13: _T["person"],
    # Furniture, equipment, railings
    8: _T["obstacle_furniture"],  # bed
    11: _T["obstacle_furniture"],  # cabinet
    16: _T["obstacle_furniture"],  # table
    20: _T["obstacle_furniture"],  # chair
    24: _T["obstacle_furniture"],  # sofa
    25: _T["obstacle_furniture"],  # shelf
    31: _T["obstacle_furniture"],  # armchair
    32: _T["obstacle_furniture"],  # seat
    34: _T["obstacle_furniture"],  # desk
    36: _T["obstacle_furniture"],  # wardrobe
    37: _T["obstacle_furniture"],  # lamp
    39: _T["obstacle_furniture"],  # railing
    40: _T["obstacle_furniture"],  # cushion
    42: _T["obstacle_furniture"],  # box
    45: _T["obstacle_furniture"],  # chest of drawers
    46: _T["obstacle_furniture"],  # counter
    50: _T["obstacle_furniture"],  # fireplace
    51: _T["obstacle_furniture"],  # refrigerator
    56: _T["obstacle_furniture"],  # case
    63: _T["obstacle_furniture"],  # bookcase
    65: _T["obstacle_furniture"],  # coffee table
    70: _T["obstacle_furniture"],  # bench
    71: _T["obstacle_furniture"],  # countertop
    72: _T["obstacle_furniture"],  # stove
    74: _T["obstacle_furniture"],  # kitchen island
    75: _T["obstacle_furniture"],  # computer
    76: _T["obstacle_furniture"],  # swivel chair
    78: _T["obstacle_furniture"],  # bar
    83: _T["obstacle_furniture"],  # light
    86: _T["obstacle_furniture"],  # chandelier
    89: _T["obstacle_furniture"],  # booth
    90: _T["obstacle_furniture"],  # television
    94: _T["obstacle_furniture"],  # pole
    96: _T["obstacle_furniture"],  # bannister
    98: _T["obstacle_furniture"],  # ottoman
    100: _T["obstacle_furniture"],  # buffet
    106: _T["obstacle_furniture"],  # conveyer belt
    108: _T["obstacle_furniture"],  # washer
    111: _T["obstacle_furniture"],  # stool
    112: _T["obstacle_furniture"],  # barrel
    113: _T["obstacle_furniture"],  # basket
    119: _T["obstacle_furniture"],  # oven
    125: _T["obstacle_furniture"],  # microwave
    130: _T["obstacle_furniture"],  # dishwasher
    131: _T["obstacle_furniture"],  # screen
    133: _T["obstacle_furniture"],  # sculpture
    134: _T["obstacle_furniture"],  # hood
    135: _T["obstacle_furniture"],  # sconce
    139: _T["obstacle_furniture"],  # ashcan
    140: _T["obstacle_furniture"],  # fan
    144: _T["obstacle_furniture"],  # monitor
    147: _T["obstacle_furniture"],  # radiator
}

# Classes ADE20K does not cover; these only come from in-house annotation:
#   floor_metal     grating / steel plate (factory floors)
#   wet_slippery    standing water / freshly mopped floor
#   threshold_ramp  door sills and level changes
#
# No public dataset covers specular reflections on polished floors either. A reflected
# ceiling reads as an open corridor, which is the single most likely way an indoor
# quadruped misjudges traversability. That has to be taught from site data.
