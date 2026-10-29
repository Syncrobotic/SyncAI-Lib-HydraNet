"""The retail_v3 taxonomy: the eleven ids the CVAT packages and `studioa-segment` use.

This is the scheme on every mask under `datasets/retail_v3*`, and the one
`SyncAI-Data-Core/datasets/studioa-segment/dataset.yaml` calls the single source of
truth. The palette was checked against a mask on disk, colour by colour, before this
file was written -- all eleven match.

**It is not `SITE30K`, although both have eleven ids.** The two taxonomies collide
numerically and disagree on what the numbers mean:

    id   retail_v3          SITE30K
     2   stairs             wall
     3   structure          column
     4   glass              display_table
     5   door               shelf
     7   column             laptop

Training retail_v3 masks under `site30k_native` would therefore not fail -- it would
quietly supervise `wall` with staircases and `laptop` with pillars. A separate scheme is
the only safe way to read these masks, which is why one exists rather than a reuse.

`structure` is this taxonomy's merge of wall and ceiling; `column` is split out of it
because a pillar is the thing a shopper walks around and a wall is the thing they walk
along. `display_fixture` is the furniture body, while `goods_table` and `goods_wall` are
the merchandise on it, split by the surface it rests on -- a distinction the masks make
geometrically (DA3 plane normals), not by prompt.
"""

from __future__ import annotations

RETAIL_V3 = {
    "void": 0,
    "floor": 1,
    "stairs": 2,
    "structure": 3,
    "glass": 4,
    "door": 5,
    "person": 6,
    "column": 7,
    "display_fixture": 8,
    "goods_table": 9,
    "goods_wall": 10,
}

# id -> id, with 0 lifted to the ignore index. `void` is "nobody labelled this", not a
# twelfth class: an annotation tool exports unlabelled background as 0, and training it
# as a class teaches the head to predict "unlabelled" over whatever was really there.
RETAIL_V3_NATIVE_ID = {0: 255, **{v: v for v in RETAIL_V3.values() if v}}

# Walkability for the optional second head. Floor is walkable; stairs are traversable
# but not flat, which is the distinction the off-road table already encodes (2 = clear,
# 1 = passable with care, 0 = obstacle). Stated in full rather than defaulted:
# `SegFolderDataset` hands a falsy table to the off-road policy, where id 2 means
# `grass` -- omitting this would not switch the head off, it would supervise it from the
# wrong taxonomy.
RETAIL_V3_TO_TRAV = {
    RETAIL_V3["void"]: 255,
    RETAIL_V3["floor"]: 2,
    RETAIL_V3["stairs"]: 1,
    **{
        RETAIL_V3[n]: 0
        for n in RETAIL_V3
        if n not in ("void", "floor", "stairs")
    },
}
