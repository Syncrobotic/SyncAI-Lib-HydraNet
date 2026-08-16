"""Retail *object* labels: what is in the shop, not whether a robot can walk on it.

This is a second taxonomy for retail, not a replacement for ``label_maps_retail.py``.
The two answer different questions and the difference is not cosmetic:

``label_maps_retail``   "can the robot step here" -- everything above the floor
                        collapses toward ``blocked``, so a column and the wall behind
                        it are one class *on purpose*, and a display table and a dining
                        table are kept apart *on purpose*.
``label_maps_retail_objects`` (this file)
                        "what object is this pixel" -- a column is a column, and every
                        surface merchandise sits on is one thing.

Why a new scheme rather than two more ids on the retail 13
----------------------------------------------------------
``tests/test_retail_scheme.py`` pins ids 0-11 to ``INDOOR_TERRAIN`` so that a retail run
warm-starts from an indoor checkpoint and indoor site masks stay readable. The object
taxonomy has to *merge* ids 10 and 12 and *split* id 7, and neither is expressible under
that invariant. Extending the retail scheme would mean deleting the tests that exist
specifically to stop the extension, which is the wrong way round. So the robot line keeps
its taxonomy untouched and this one is free to be shaped by the question it answers.

What the audit found, which is what this taxonomy is built from
--------------------------------------------------------------
18 fixed-camera site clips, 1,620 frames, scored with the 60-epoch checkpoint
(``runs/review_20260816/class_audit.json``):

* ``wall`` is **39.8%** of every frame at 93.7% temporal agreement -- solid, and the
  columns are inside it, unrecoverable. ADE20K's ``column`` (43) maps to ``wall`` in
  ``label_maps_indoor.py`` and always has.
* Shop fixtures arrive split across **two** classes: ``obstacle_furniture`` 17.6% and
  ``display_fixture`` 10.7%. In one Apple-store frame the round podium is
  ``obstacle_furniture`` while the wall shelving three metres away is ``display_fixture``.
  Same object, two labels, because ADE20K's ``table`` was deliberately withheld from
  fixtures -- correct under traversability, where both are ``blocked`` and the cost is
  only semantic; not correct here, where the semantics *are* the output.
* ``display_fixture`` test IoU **0.336**, the lowest of any class with real data.
* Merchandise has no class at all, in either taxonomy.

Ids
---
Grouped by what a pixel is, coarsest first. Nothing here is aligned to another scheme,
so the numbering is free -- but once site masks are exported against it, it is not.
"""

from __future__ import annotations

from .label_maps_indoor import ADE20K_ID_TO_INDOOR, INDOOR_TERRAIN
from .label_maps_retail import RETAIL_TERRAIN

RETAIL_OBJECTS = {
    "void": 0,
    "floor": 1,  # every walking surface, hard or soft
    "wall": 2,  # wall / ceiling / glazing / door -- the room's shell
    "column": 3,  # free-standing structural column or pillar
    "fixture": 4,  # display table / cabinet / shelving / counter / gondola
    "product": 5,  # merchandise, boxed or on display
    "person": 6,
}

# The loss ignores 255 alone, and annotation tools export unlabelled background as 0.
# An identity table would make "nobody labelled this" a trainable class -- the same trap
# INDOOR_NATIVE_ID documents, and it costs nothing to avoid twice.
RETAIL_OBJECTS_NATIVE_ID = {0: 255, **{v: v for v in RETAIL_OBJECTS.values() if v}}

# A traversability policy for a taxonomy whose configs do not train a traversability
# head. It is here anyway, and it is not dead code: `SegFolderDataset` passes
# `scheme.trav` straight to `terrain_to_traversability`, which falls back to the
# **off-road** policy when handed a falsy map. A scheme shipping no table would not
# disable the second head, it would silently supervise it with the RUGD/RELLIS policy
# where id 2 means `grass`. Stating it costs seven lines and removes that failure.
#
# It is also the honest reading of these classes: only the floor is walkable, and
# everything else in a shop is something to walk around.
RETAIL_OBJECTS_TO_TRAV = {
    0: 255,  # void -> ignore
    1: 2,  # floor -> go
    2: 0,  # wall -> blocked
    3: 0,  # column -> blocked
    4: 0,  # fixture -> blocked
    5: 0,  # product -> blocked
    6: 0,  # person -> blocked
}


# ---------------------------------------------------------------------------
# Migrating the site masks already collected.
#
# Everything under datasets/retail_sam3_consensus* and datasets/retail_cctv_pilot* is
# annotated against the retail 13. This table reads those masks under the object
# taxonomy, so the SAM 3 consensus work is not thrown away.
#
# ONE CLASS CANNOT SURVIVE THE TRIP, and it is the one this taxonomy was extended for:
# every column pixel in those masks is already inside `wall` (id 7) and there is no
# signal left to separate them. Migrated data therefore contains **zero** `column`
# pixels. `column` has to be drawn fresh -- which is cheap, because the cameras are
# fixed and a column is one polygon per camera that is correct for every frame that
# camera will ever produce. See `columns_from_migration_only` below, which refuses to
# let that fact stay quiet.
#
# `stairs` maps to ignore rather than to floor. An escalator is not a floor and it is
# not a fixture; guessing either would train a wrong label over the 0.27% of pixels
# where the site footage actually has one.
# ---------------------------------------------------------------------------
_R = RETAIL_TERRAIN
_O = RETAIL_OBJECTS

RETAIL_ID_TO_OBJECTS = {
    0: 255,  # void -> ignore
    _R["floor_hard"]: _O["floor"],
    _R["floor_soft"]: _O["floor"],
    _R["floor_metal"]: _O["floor"],
    _R["wet_slippery"]: _O["floor"],
    _R["threshold_ramp"]: _O["floor"],
    _R["stairs"]: 255,  # neither floor nor fixture; do not guess
    _R["wall"]: _O["wall"],
    _R["glass"]: _O["wall"],  # a glass wall is a wall once nobody has to walk into it
    _R["door"]: _O["wall"],
    _R["obstacle_furniture"]: _O["fixture"],  # <- the merge, half one
    _R["display_fixture"]: _O["fixture"],  # <- the merge, half two
    _R["person"]: _O["person"],
}


def columns_from_migration_only(*scheme_names: str) -> str | None:
    """Warn when ``column`` can only come from migrated masks, where it does not exist.

    Returns the warning text, or None when some scheme in the run can actually supply
    the class. Called by the config validator so the message arrives before the GPU
    hours, not after a run reports IoU 0.000 on a channel nobody could have filled.

    This is the ``wet_slippery`` failure, generalised: that class has been in the
    taxonomy since the first commit and is still 0.000 today, because no dataset in any
    config ever contained one. A class nobody draws is a permanently empty channel, and
    nothing in a training run says so.
    """
    supplying = {"ade20k_retail_objects", "retail_objects_native"}
    if supplying & set(scheme_names):
        return None
    if "retail_objects_migrated" not in scheme_names:
        return None
    return (
        "`column` (id 3) is trained on nothing: the only scheme in this run is "
        "'retail_objects_migrated', and every column pixel in those masks is already "
        "inside `wall` (retail id 7) with no signal left to separate them. The head "
        "will report IoU 0.000 for it after a full run. Draw columns fresh -- the "
        "cameras are fixed, so it is one polygon per camera, not per frame."
    )


# ---------------------------------------------------------------------------
# ADE20K -> retail objects. A bootstrap for the trunk, and explicitly not more.
#
# The rule that shaped this table: **the concern that kept `table` out of
# display_fixture inverts here.** label_maps_retail.py withholds ADE20K's `table` (16)
# because mapping a dining table to `display_fixture` would teach the model that a
# domestic proxy is the retail thing, while a competing `obstacle_furniture` class stood
# ready to take it. In this taxonomy there is no competing class -- `fixture` means "a
# waist-height surface or storage volume that merchandise sits on or in", and a dining
# table is a near-perfect *shape* prior for exactly that. What was a contamination there
# is the intended signal here. The audit is what makes the difference concrete: the
# split cost `display_fixture` its 0.336 IoU, and the podium/shelving disagreement in a
# single frame is that number made visible.
#
# Seating is NOT a fixture. Chairs, sofas, stools, benches and ottomans fall through to
# ignore rather than being forced into a class -- a shop has all of them and none is
# what the question is about. Ignore costs coverage; a wrong label costs the class.
#
# `product` has NO ADE20K source and none is possible: no public segmentation dataset
# labels merchandise. It is filled from site annotation and SAM 3 pre-labels
# (data/sam3_prompts_objects.py), and until it is, it is an empty channel. The config
# validator says so rather than letting a run discover it at epoch 60.
# ---------------------------------------------------------------------------
_ADE_FLOOR = (4, 7, 12, 53, 29)  # floor, road, sidewalk, path, rug
_ADE_WALL = (
    1,  # wall
    6,  # ceiling -- a ceiling camera sees plenty of it, and it is shell, not object
    23,  # painting
    44,  # signboard
    101,  # poster
    145,  # bulletin board
    19,  # curtain
    64,  # blind
    9,  # windowpane -- glazing is shell here; the robot scheme keeps `glass` apart
    28,  # mirror
    15,  # door
    59,  # screen door
)
# 43 only. ADE20K's `pole` (94) was considered and left out: it covers lamp posts and
# sign posts, and this class means a structural column standing in a room.
_ADE_COLUMN = (43,)
_ADE_FIXTURE = (
    25,  # shelf
    46,  # counter
    56,  # case -- display case
    63,  # bookcase
    71,  # countertop
    78,  # bar
    89,  # booth
    100,  # buffet
    16,  # table        <- withheld by the robot scheme, wanted here
    11,  # cabinet      <-
    34,  # desk         <-
    45,  # chest of drawers
    36,  # wardrobe
    74,  # kitchen island
    65,  # coffee table
)
_ADE_PERSON = (13,)


def _build_ade20k_map() -> dict[int, int]:
    out: dict[int, int] = {}
    for ids, name in (
        (_ADE_FLOOR, "floor"),
        (_ADE_WALL, "wall"),
        (_ADE_COLUMN, "column"),
        (_ADE_FIXTURE, "fixture"),
        (_ADE_PERSON, "person"),
    ):
        for ade_id in ids:
            out[ade_id] = RETAIL_OBJECTS[name]
    return out


ADE20K_ID_TO_RETAIL_OBJECTS = _build_ade20k_map()


# Which ADE20K ids the robot scheme uses and this one drops, and why. Kept as data
# rather than prose because tests/test_retail_objects_scheme.py asserts against it: a
# silent addition here is a class boundary moving without anyone deciding to move it.
DROPPED_FROM_ADE20K = {
    "seating": (20, 24, 31, 32, 76, 111, 70, 98, 8),  # chair..bed
    "appliances_and_lamps": (37, 51, 72, 83, 86, 108, 119, 125, 130, 134, 135, 140, 147),
    "small_objects": (40, 42, 112, 113, 139, 148),  # cushion, box, barrel, basket, ...
    "screens": (75, 90, 131, 144),  # computer, television, screen, monitor
    "other_structure": (39, 94, 96, 106, 133, 50),  # railing, pole, bannister, ...
    "level_change": (54, 60, 97, 122),  # stairs, stairway, escalator, step
}

# `screens` deserves a note, because it is the one group a reader will query. A wall of
# demo iPads is merchandise and an in-store signage panel is not, and ADE20K cannot tell
# them apart -- both are id 131. The detection head already separates them better than a
# segmentation label could: the audit found it firing `tv` and `refrigerator` on exactly
# these objects at 0.05-0.15. So they go to ignore here and are answered there.


def unsourced_classes(*mappings: dict[int, int]) -> tuple[str, ...]:
    """Taxonomy classes no supplied mapping can ever produce.

    The check the project keeps re-learning: `wet_slippery`, `floor_metal` and
    `threshold_ramp` have been in the retail taxonomy since the beginning, are all still
    IoU 0.000, and every run that trained them looked completely normal. An empty output
    channel is invisible at training time. This makes it a config-time answer.
    """
    produced = {v for m in mappings for v in m.values() if v != 255}
    return tuple(name for name, tid in RETAIL_OBJECTS.items() if tid and tid not in produced)


# Sanity: the object taxonomy must not accidentally re-derive the indoor one. If a
# future edit makes these equal, the migration table above is silently an identity map.
assert set(RETAIL_OBJECTS) != set(INDOOR_TERRAIN)
assert set(ADE20K_ID_TO_RETAIL_OBJECTS) <= set(ADE20K_ID_TO_INDOOR), (
    "every ADE20K id used here must exist in the verified indoor table; a new id has "
    "not been checked against objectInfo150.txt"
)
