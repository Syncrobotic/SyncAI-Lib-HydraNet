"""SAM 3 prompts for the object taxonomy: floor, wall, column, fixture, product, person.

The retail table in [sam3_prompts.py](sam3_prompts.py) targets the robot's 13 classes and
every prompt finding in it still holds -- the footage did not change, only what the
labels are for. This file reuses those findings and differs in three places, each of
which follows from the taxonomy rather than from a new measurement.

**The `table` / `display table` conflict disappears.** That is the single biggest change
and it is free. sam3_prompts.py sends both prompts to LAYER_THING so their overlap
becomes 255, because "is this shop table a display or a desk" is a judgement an
annotator has to make -- ``display_fixture`` and ``obstacle_furniture`` are different
answers there. Here they are the *same class*, so the overlap is not a conflict at all;
it is two prompts agreeing, which ``compose`` already treats as a union. The ~37% ignore
that sweep recorded was substantially this collision, and merging the classes reclaims
it without touching a threshold.

**`column` gets its own prompts and sits above `wall`.** sam3_prompts.py already carries
``column`` and ``pillar``, buried inside the ``wall`` concept's prompt list and switched
off by default. They were there to fill structural holes next to the floor, and the
comment notes RETAIL_SCOPE.md s5 found a *column* was exactly what the baseline painted
``caution`` on in 97.4% of frames. Same prompts, promoted to a class.

The layer matters: ``column`` is on the thing layer and ``wall`` stays on the ground
layer, so a column paints over the wall behind it. That is the ordinary case -- a column
stands in front of a wall, and "column" is the more specific claim of the two. Leaving
both on the ground layer would send every column to 255, which is the opposite of the
point.

**`product` is new and has no prior measurement at all.** Nothing in either sweep asked
SAM 3 for merchandise, so every number below is absent rather than zero, and the first
run of this class is a measurement rather than a production pass. What is known comes
from the detection head instead: the audit in runs/review_20260816/ found the COCO head
firing on boxed stock as ``book`` 1,683 times over 40 frames of Kaohsiung-cam08 at score
0.05. That says merchandise is findable in this footage; it does not say SAM 3 finds it.
Check the first batch before trusting the second.
"""

from __future__ import annotations

from .label_maps_retail_objects import RETAIL_OBJECTS
from .sam3_prompts import DEFAULT_MIN_SCORE, Concept

# Four layers rather than three, because merchandise sits *on* the thing that holds it
# and that is not a judgement call -- it is the whole spatial relationship a shop is
# arranged around. Products therefore paint over fixtures outright.
#
# `compose` only ever compares these as integers, so the values are a sort order and
# nothing more.
LAYER_GROUND = 0  # what the room is made of: floor, wall
LAYER_THING = 1  # what stands in it: fixtures, columns
LAYER_PRODUCT = 2  # what sits on the things
LAYER_PERSON = 3  # always foreground


def _c(name, prompts, layer, **kw) -> Concept:
    return Concept(name, prompts, layer, taxonomy=RETAIL_OBJECTS, **kw)


CONCEPTS = (
    # --- the merge, and the reason this taxonomy exists ---------------------------
    #
    # Every prompt from `display_fixture` plus every prompt from `obstacle_furniture`,
    # in one class. In sam3_prompts.py these are two concepts on the same layer whose
    # overlap is discarded as contested; here the union is the answer.
    #
    # `table` is the case worth naming. It fires on 6/8 frames over largely the same
    # pixels as `display table`, and under the robot taxonomy that ambiguity is real and
    # unresolvable from a ceiling camera. Under this one there is nothing to resolve:
    # both are `fixture`, whatever the shop uses the table for.
    #
    # `trolley` and `trash can` come along from the obstacle list. A trolley is not a
    # display fixture in any careful sense, and it is kept because the alternative is
    # worse -- it is the most common dynamic obstacle in a shop, COCO has no class for
    # it, and leaving it unclaimed puts a hole in the middle of the aisle. Flagged here
    # so that a later split has a place to start rather than a surprise.
    _c(
        "fixture",
        (
            # from display_fixture -- coverage figures are sweep B's, on the retail class
            "display table",  # 6/8, 0.787 -- the island podiums
            "product display stand",  # 5/8, 0.639
            "shelving unit",  # 5/8, 0.651 -- the wall runs
            "showcase cabinet",  # 5/8, 0.598
            "display podium",  # 4/8, 0.727
            "retail counter",  # 4/8, 0.620
            "merchandise rack",  # 4/8, 0.629
            "display case",  # 3/8, 0.561
            "shelf",
            "display counter",
            "checkout counter",
            "product display rack",
            # from obstacle_furniture -- no longer a competing class
            "table",  # 6/8; the contested one, now uncontested
            "office furniture",
            "trolley",
            "trash can",
        ),
        LAYER_THING,
        note="display_fixture + obstacle_furniture merged; the `table` conflict dissolves",
    ),
    # --- split out of `wall` ------------------------------------------------------
    #
    # Above `wall` on purpose: a column stands in front of the wall behind it, and
    # `column` is the more specific of the two claims. On the ground layer with `wall`
    # every column would come back 255.
    #
    # No coverage figure: these prompts have only ever run as two entries in the `wall`
    # concept's list, where whatever they found was recorded as wall. What is known is
    # that they fire -- RETAIL_SCOPE.md s5 tracked a column through 97.4% of frames of
    # one camera, as a false `caution`.
    _c(
        "column",
        ("column", "pillar", "structural column", "square pillar", "round column"),
        LAYER_THING,
        note="promoted out of wall's prompt list; sits above wall so it is not erased",
    ),
    # --- measured 2026-08-17, after the first batch ---------------------------------
    #
    # min_score is 0.25 here rather than the 0.40 default, and this is now the sweep the
    # note below asked for. Union coverage of `product` against the threshold, on three
    # frames chosen for three kinds of merchandise:
    #
    #                                  0.20    0.25    0.30    0.35    0.40
    #   Taichung-cam10 laptop counter 22.29%  21.77%  19.46%  15.01%  12.92%
    #   Kaohsiung-cam07 packet wall   38.69%  38.65%  38.61%  33.69%  33.67%
    #   Taichung-cam01 mixed floor    19.94%  18.18%  16.86%  15.05%  14.94%
    #
    # 0.20 buys almost nothing over 0.25 and 0.30 starts costing; the knee is the same
    # on all three. What 0.40 was discarding is visible rather than statistical: on the
    # laptop counter `laptop on a table` returns 14 instances at 0.20 and **4** at 0.40,
    # against seven or eight laptops actually on the table, so most of them came back
    # unlabelled and a human reviewing the batch spotted it by eye.
    #
    # Why this class needs its own number, which is the part worth keeping: SAM 3 scores
    # furniture far higher than goods. On the same frame `display table` peaks at 0.926
    # and `retail counter` at 0.922, while `merchandise on a shelf` peaks at 0.543 and
    # `boxed product` at 0.500. A fixture is one large object with a clean silhouette; a
    # product is small, dense and half-occluded by its neighbours. One threshold across
    # both is right for the first and too strict for the second.
    #
    # Two prompts return nothing at any score on this footage -- `tablet on display` and
    # `packaged goods` -- and are kept rather than deleted: one frame set is not enough
    # to retire a vocabulary entry, and the cost is a forward pass, not a wrong label.
    #
    # The original note, which asked for exactly this and was right to refuse to guess:
    # the 0.40 in sam3_prompts.py came from a seven-threshold sweep on classes that had
    # coverage figures, and inventing a number for a class nobody had run would have been
    # the kind of unsourced constant this project keeps deleting.
    #
    # The prompt list deliberately mixes three registers, because the sweep's lesson was
    # that SAM 3's vocabulary is web English rather than trade jargon (`gondola shelf`
    # scored 0/8 despite being the correct term): a generic noun, the packaging, and the
    # specific goods these stores sell.
    _c(
        "product",
        (
            "product box on a shelf",
            "boxed product",
            "packaged goods",
            "merchandise on a shelf",
            "smartphone on display",
            "laptop on a table",
            "tablet on display",
            "headphones box",
        ),
        LAYER_PRODUCT,
        min_score=0.25,
        note="min_score 0.25, swept on 3 cameras; 0.40 kept 4 of ~8 laptops on a counter",
    ),
    # --- carried over unchanged ---------------------------------------------------
    #
    # Same caveat as the retail table and it is worth repeating where it will be read:
    # on LAYER_PERSON this overwrites everything, and SAM 3 segments people *reflected*
    # in mirrors and seen *through* storefront glass as people in the room. A false
    # person is the one false positive nothing downstream can outvote.
    _c(
        "person",
        ("person",),
        LAYER_PERSON,
        note="8 instances, mean 0.805; segments reflections and people seen through glass",
    ),
    # --- off by default, for the reasons the retail table already established -------
    #
    # The floor is one static polygon per fixed camera, which beats a pre-label per
    # frame; RETAIL_SCOPE.md s5 measured that. `wall` is adequately covered already.
    # Both are on the ground layer so that fixtures, columns, products and people all
    # paint over them.
    #
    # Note `wall` here absorbs what the robot taxonomy splits into wall, glass and door
    # -- so the glass failure documented at length in sam3_prompts.py stops being a
    # class-level hazard and becomes an ordinary mislabel inside one class. That is a
    # genuine simplification and not a fix: a pane still reads as the pavement behind
    # it, and those pixels are still wrong, they are just wrong within `wall`.
    _c(
        "floor",
        ("tiled floor", "hard floor", "floor", "tile floor", "carpet", "floor mat"),
        LAYER_GROUND,
        default_on=False,
        note="fixed camera: one polygon per camera beats a pre-label per frame",
    ),
    _c(
        "wall",
        ("wall", "ceiling", "glass panel", "glass door", "door", "doorway"),
        LAYER_GROUND,
        default_on=False,
        note="absorbs glass and door; the glass failure is now contained, not solved",
    ),
)

BY_NAME = {c.name: c for c in CONCEPTS}
DEFAULT_ON = tuple(c.name for c in CONCEPTS if c.default_on)

# See sam3_prompts.MOVING. `product` is deliberately NOT here: merchandise is restocked
# and picked up, so some of it does move between frames -- but on the timescale of one
# clip a shelf of boxes is static, and excluding the class from the vote would forfeit
# the only error signal available for the one class with no prior measurement.
MOVING = ("person",)

assert set(BY_NAME) <= set(RETAIL_OBJECTS), "a concept names a class the taxonomy lacks"
assert DEFAULT_MIN_SCORE == 0.40, "the min_score note above quotes this value"


def resolve(include: list[str] | None, exclude: list[str] | None) -> list[Concept]:
    """The concepts to run, starting from the default set. Mirrors sam3_prompts.resolve."""
    chosen = dict.fromkeys(DEFAULT_ON)
    for name in include or ():
        if name not in BY_NAME:
            raise SystemExit(f"unknown class {name!r}; known: {', '.join(sorted(BY_NAME))}")
        chosen[name] = None
    for name in exclude or ():
        if name not in BY_NAME:
            raise SystemExit(f"unknown class {name!r}; known: {', '.join(sorted(BY_NAME))}")
        chosen.pop(name, None)
    return [BY_NAME[n] for n in chosen]


# ---------------------------------------------------------------------------
# Instance detection: merchandise, as boxes rather than pixels.
#
# A semantic mask cannot count. Thirty identical boxes on a shelf are one connected
# region to a per-pixel classifier, which is why the site run reads `product` as 93.1%
# `fixture`: merchandise physically sits on the thing that holds it, and adjacency plus
# shared texture is all a segmentation head has to go on. Counting stock, spotting a gap
# on a shelf, and "how many phones are out on this table" are all instance questions.
#
# SAM 3 already returns instances -- `segment()` hands back one mask per detection and
# `compose()` then throws that identity away by painting into a single-channel id map.
# These classes are the same forward pass read the other way, so the boxes cost no extra
# GPU time when `--boxes` is on.
#
# MEASURED, 22 prompts x 4 site cameras at threshold 0.40 (2026-08-16). The finding that
# shaped the vocabulary, and it is the same one `gondola shelf` produced in
# sam3_prompts.py: **brand names do not work, category nouns do.** SAM 3's vocabulary is
# web English, and the web's `iphone` is a product shot, not thirty pixels of handset on
# a ceiling camera.
#
#     prompt              instances   mean score
#     product box              116        0.604
#     boxed product            106        0.609
#     mobile phone              17        0.601
#     smartphone                12        0.665
#     phone on display          10        0.713
#     laptop                     8        0.800   <- highest score of any prompt
#     laptop computer            7        0.814
#     imac                       7        0.715
#     ipad                       5        0.651
#     tablet                     5        0.609
#     macbook                    4        0.478   <- against `laptop` 0.800
#     iphone                    <2          --    <- essentially silent
#     apple watch                0          --    <- silent
#
# Two classes, not six, and deliberately coarse:
#
#   * `headphones` (54) and `phone case` (23) both fired on the *same* wall of mixed
#     hanging accessories on Tao-Hsin-cam01. Each is right about half of it and neither
#     can be told which half. A finer taxonomy buys contested pixels, not resolution.
#   * `product box` scored 73 instances on Taichung-cam01 and **0** on Tao-Hsin-cam01,
#     where the stock hangs in packets rather than sitting in boxes. One prompt, one
#     retail brand, 73 against 0 -- so no vocabulary fixed from one store is safe.
#
# Splitting `device` into phone / tablet / laptop is the natural next step and is better
# done as a second-stage classifier on the crop, where the resolution problem that sinks
# the brand names is much smaller.
DETECTION_CLASSES = (
    ("boxed_stock", ("product box", "boxed product")),
    (
        "device",
        (
            "smartphone",
            "mobile phone",
            "phone on display",
            "tablet",
            "ipad",
            "laptop",
            "laptop computer",
            "imac",
        ),
    ),
)

DETECTION_BY_NAME = dict(DETECTION_CLASSES)
