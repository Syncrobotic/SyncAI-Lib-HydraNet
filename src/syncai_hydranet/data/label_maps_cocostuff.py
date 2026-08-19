"""COCO-Stuff (``stuffthingmaps_trainval2017``) -> the indoor 12 / retail 13.

Per-pixel labels for the images **already in** ``datasets/coco/`` -- the same 118,287
train frames the detection head reads, so this costs one 659 MB annotation download and
no new imagery. It exists to buy volume for the classes ADE20K starves:

    stairs   ADE20K supplies 314 training images; this adds ~2,700
    door     ~9,500      glass  ~17,500      floor_soft  ~8,400

It supplies **none** of ``floor_metal``, ``wet_slippery`` or ``threshold_ramp``, and no
free-standing display podiums. Those were tested against the full ADE20K release on
2026-08-13 and rejected; this download does not change the answer. See
docs/RETAIL.md -- site annotation is still the only source.

Three properties of the PNGs, all measured off the pixels rather than read off
``labels.txt``, because each one fails silently:

1. **The PNG value is one less than the ``labels.txt`` id.** The table below is written
   in ``labels.txt`` ids so it can be read against that file, and ``_shift`` applies the
   offset exactly once. Verified by class frequency against known COCO composition: PNG
   0 is in 54.4% of images and PNG 156 in 28.3%. Read unshifted those are ``unlabeled``
   and ``shelf``; shifted they are ``person`` and ``sky-other``, which is what COCO
   actually contains.
2. **``person`` is PNG value 0, and unlabelled is 255.** This is the exact inverse of the
   in-house convention in ``label_maps_indoor.INDOOR_NATIVE_ID``, where 0 means "nobody
   labelled this". Treating 0 as void here deletes every person in the dataset.
3. **Eleven ``labels.txt`` thing ids are dead** -- ``mirror`` (66), ``window`` (68),
   ``door`` (71) and others were removed from COCO before release and carry no pixels.
   The live entries are the stuff ones: ``mirror-stuff``, ``window-other``,
   ``door-stuff``. Mapping a dead id yields zero pixels, no error, and a class that
   quietly leaves the mIoU denominator.

Unlisted ids become 255 (ignore), never a trainable class, so this table is deliberately
conservative: COCO's outdoor majority -- sky, sea, grass, tree -- is left out rather than
forced into a class. The consequence is that most COCO images end up almost entirely
ignored, which is why a config should filter to images that actually carry these classes
instead of paying to load all 118,287.

``tests/test_cocostuff_scheme.py`` re-derives the offset from the shipped data, so the
one assumption that would corrupt every label cannot regress silently.
"""

from __future__ import annotations

from .label_maps_indoor import INDOOR_TERRAIN
from .label_maps_retail import RETAIL_TERRAIN

_T = INDOOR_TERRAIN

# Keyed by `labels.txt` id so the table can be diffed against that file by eye.
# `_shift` turns these into the PNG values the loader actually sees.
_LABELS_TXT_TO_INDOOR = {
    # -- walkable ground -------------------------------------------------------
    114: _T["floor_hard"],  # floor-marble
    115: _T["floor_hard"],  # floor-other
    116: _T["floor_hard"],  # floor-stone
    117: _T["floor_hard"],  # floor-tile
    118: _T["floor_hard"],  # floor-wood
    140: _T["floor_hard"],  # pavement   (indoor scheme maps ADE20K sidewalk the same way)
    149: _T["floor_hard"],  # road       (likewise ADE20K road)
    101: _T["floor_soft"],  # carpet
    152: _T["floor_soft"],  # rug
    131: _T["floor_soft"],  # mat
    # -- level changes ---------------------------------------------------------
    161: _T["stairs"],  # stairs        <- the reason this dataset is here
    # -- boundaries ------------------------------------------------------------
    171: _T["wall"],  # wall-brick
    172: _T["wall"],  # wall-concrete
    173: _T["wall"],  # wall-other
    174: _T["wall"],  # wall-panel
    175: _T["wall"],  # wall-stone
    176: _T["wall"],  # wall-tile
    177: _T["wall"],  # wall-wood
    102: _T["wall"],  # ceiling-other
    103: _T["wall"],  # ceiling-tile
    109: _T["wall"],  # curtain        (ADE20K curtain 19 -> wall)
    180: _T["wall"],  # window-blind   (ADE20K blind 64 -> wall)
    92: _T["wall"],  # banner
    # -- glass: architectural glazing only, never a drinking vessel -------------
    133: _T["glass"],  # mirror-stuff   (NOT 66 `mirror`, which is dead)
    181: _T["glass"],  # window-other   (NOT 68 `window`, which is dead)
    112: _T["door"],  # door-stuff     (NOT 71 `door`, which is dead)
    # -- people ----------------------------------------------------------------
    1: _T["person"],  # person -> PNG value 0. See note 2 in the module docstring.
    # -- furniture, equipment, railings ----------------------------------------
    15: _T["obstacle_furniture"],  # bench
    62: _T["obstacle_furniture"],  # chair
    63: _T["obstacle_furniture"],  # couch
    64: _T["obstacle_furniture"],  # potted plant
    65: _T["obstacle_furniture"],  # bed
    67: _T["obstacle_furniture"],  # dining table
    69: _T["obstacle_furniture"],  # desk
    70: _T["obstacle_furniture"],  # toilet
    72: _T["obstacle_furniture"],  # tv
    78: _T["obstacle_furniture"],  # microwave
    79: _T["obstacle_furniture"],  # oven
    81: _T["obstacle_furniture"],  # sink
    82: _T["obstacle_furniture"],  # refrigerator
    98: _T["obstacle_furniture"],  # cabinet
    99: _T["obstacle_furniture"],  # cage
    108: _T["obstacle_furniture"],  # cupboard
    110: _T["obstacle_furniture"],  # desk-stuff
    123: _T["obstacle_furniture"],  # furniture-other
    130: _T["obstacle_furniture"],  # light
    146: _T["obstacle_furniture"],  # railing
    165: _T["obstacle_furniture"],  # table
}

# Fixtures, for the retail scheme only. `shelf` and `counter` match what
# `ade20k_retail` already does with ADE20K's shelf (25) and counter (46), so this is
# consistency rather than a new liberty.
#
# `table` (165) is deliberately absent, exactly as ADE20K's `table` (16) is: a domestic
# dining table is a proxy that is nearly a display table, and teaching the model that a
# near-miss is the real thing is the mistake this project has already paid for once.
_LABELS_TXT_FIXTURES = {
    156: "shelf",
    107: "counter",
}


def _shift(mapping: dict[int, int]) -> dict[int, int]:
    """`labels.txt` id -> the PNG value the annotation actually stores.

    One subtraction, in one place. Everything above is written in `labels.txt` ids so it
    stays readable against the file that ships with the dataset.
    """
    return {label_id - 1: terrain for label_id, terrain in mapping.items()}


COCOSTUFF_ID_TO_INDOOR = _shift(_LABELS_TXT_TO_INDOOR)

COCOSTUFF_ID_TO_RETAIL = _shift(
    {
        **_LABELS_TXT_TO_INDOOR,
        **dict.fromkeys(_LABELS_TXT_FIXTURES, RETAIL_TERRAIN["display_fixture"]),
    }
)
