"""What to ask SAM 3 for, and what its answer is allowed to become.

Self-training cannot invent a class. ``configs/hydranet_retail_cctv.yaml`` says so in
prose and [docs/RETAIL.md](../../../docs/RETAIL.md) measures it: the site
pseudo-labels moved ``display_fixture`` by **-0.0096**, because the free-standing podiums
came back unlabelled and the model was then taught that fixture-shaped things in a shop
are not fixtures. The label source was the model itself, so the loop could only harden.

SAM 3 breaks that loop for one reason, and it is worth being precise about it: **its
knowledge is external**. It has never seen this model's mistakes, so a podium it segments
is new information rather than an echo. That is the whole argument for this module, and
it is also its limit -- SAM 3 is a different model with different errors, not an oracle.
Everything here is a *pre-label*, and the contract in
[docs/METHODOLOGY.md](../../../docs/METHODOLOGY.md) still gates it.

Ids are imported, never typed. ``RETAIL_TERRAIN`` is the one place the numbering lives,
and a table of hand-written ints here would be a second one -- the exact drift the
annotation CLI exists to prevent.

Two independent runs feed this table, and they disagree usefully. Both are recorded,
because where they agree the finding is much stronger than either alone, and where they
disagree the reason is always the *footage*, never the model.

**Sweep A** (2026-08-15): eight frames of `hls-record`, 352x240 and 704x480, one frame
per camera, upscaled 3x, threshold 0.5. ``N/8`` notes below come from here. One store,
one flat floor, four dead cameras -- so a zero here means very little on its own, which
is a lesson this file paid for twice (see ``stairs``).

**Sweep B** (2026-08-15, parallel session): 30 frames, 1920x1080, three stores, 29
prompts x 7 thresholds. Percentages below come from here. Its own caveats, stated by the
run that produced it: every frame is local 10:00-22:00 colour, so **nothing here says
anything about the IR monochrome these cameras switch to overnight**, and it is three
stores at seed 0.

Neither is precision. Both say a prompt *fires*, not that what it fired on was right.
Precision needs hand-labelled frames, and at time of writing this domain has **none** --
every mask in ``datasets/retail_cctv_pilot*`` is SAM 3 output with no human pass. That
makes it usable as a pre-label and unusable as a test split: scoring a model against it
measures agreement with SAM 3. See RETAIL.md s6 for why the test split has to be
built by hand and built first.
"""

from __future__ import annotations

from .label_maps_retail import RETAIL_TERRAIN

# --------------------------------------------------------------------------- layers
#
# SAM 3 returns overlapping instances; a single-channel id mask has room for one answer
# per pixel. Paint order resolves the overlaps that have an obvious winner -- a person
# standing on a floor is a person, and no annotator would disagree.
#
# What paint order must NOT be used for is a disagreement that is genuinely a judgement
# call. Two classes on the same layer claiming one pixel is not resolved here; it becomes
# 255 and a human decides. See ``compose`` in scripts/sam3_prelabel.py.
#
# This is a deliberate divergence from sweep B's generator, which ordered all 29 prompts
# in one list and let the later one win every overlap outright. That yields less ignore
# (20.4% against this file's ~37% on the same frames) and it is the better choice if the
# output is being read as a map. It is the worse choice here, because the output is being
# read by an annotator: a pixel where `table` and `display table` both fired is a real
# question about what the shop uses that table for, and paint order answers it by
# whichever prompt happened to be typed last. Three layers resolve the overlaps that have
# an obvious answer and refuse the ones that do not. The cost is a hole; holes get drawn.
LAYER_GROUND = 0  # what the room is made of
LAYER_THING = 1  # objects standing in it -- the contested layer
LAYER_PERSON = 2  # always foreground, never ambiguous


# The default score below which an instance is not written at all.
#
# 0.40 is measured, not chosen. Sweep B ran 29 prompts over 6 frames at seven
# thresholds, reusing one forward pass per prompt so the only variable is the cut:
#
#     threshold   ignore   floor_hard   wall   display_fixture
#       0.30      15.3%      23.5%     29.6%       25.6%
#       0.35      16.9%      30.5%     22.6%       24.2%
#       0.40      19.9%      30.4%     21.3%       22.7%
#       0.50      24.4%      30.4%     19.9%       19.7%
#       0.60      25.4%      30.3%     19.3%       19.5%
#
# Read the two ends. **0.30 is a trap**: floor collapses to 23.5% while wall climbs to
# 29.6%, which is low-confidence wall and ceiling masks eating the floor beneath them --
# more coverage, and the extra coverage is wrong. **0.50 quietly loses fixtures**: an
# entire wall of shelving in Taichung-cam01 scores in the 0.40-0.50 band, so the cut
# this file originally shipped (0.5) wrote it as ignore.
#
# 0.40 over 0.35 is the weaker call: at 0.35 the hedge band below the cut is only 0.05
# wide, narrow enough to be decorative, and the band is the entire point.
DEFAULT_MIN_SCORE = 0.40


class Concept:
    """One terrain class, the prompts that reach it, and how far to trust them.

    ``default_on`` is False for the classes where a pre-label is not the cheap way to
    get the answer, rather than for the ones SAM 3 is bad at. Two different reasons to
    leave a class alone, and conflating them would hide both.
    """

    def __init__(
        self,
        name,
        prompts,
        layer,
        min_score=DEFAULT_MIN_SCORE,
        default_on=True,
        note="",
        taxonomy=None,
    ):
        self.name = name
        self.prompts = prompts
        self.layer = layer
        self.min_score = min_score
        self.default_on = default_on
        self.note = note
        # Which class list the name resolves against. Defaults to the retail 13 -- the
        # only taxonomy that existed when this file was written -- so every concept
        # below is unchanged. `sam3_prompts_objects.py` passes the object taxonomy
        # instead. The id is still never typed: it is looked up, in one place, from
        # whichever taxonomy the concept belongs to.
        self.taxonomy = taxonomy if taxonomy is not None else RETAIL_TERRAIN

    @property
    def terrain_id(self) -> int:
        return self.taxonomy[self.name]


CONCEPTS = (
    # --- the class this whole exercise exists for -------------------------------
    #
    # No single prompt covers it, and the sweep is unambiguous about why: "display
    # table" finds the island podiums, "shelving unit" finds the wall runs, "retail
    # counter" finds the service desk, and each misses the other two. They are one
    # class to this taxonomy and three different objects to SAM 3, so the prompts are
    # a union and the masks are merged before anything else looks at them.
    #
    # "gondola shelf" is dropped: 0/8 despite being the trade term for exactly the
    # fixture in these frames. Worth knowing -- SAM 3's vocabulary is web English, not
    # retail jargon, so prompts should read like a caption rather than a spec sheet.
    #
    # Sweep B reaches 24.31% of labelled pixels with the union below, against 13.98%
    # before `shelf` / `display counter` / `checkout counter` / `product display rack`
    # were added and the cut dropped to 0.40. That is the single largest movement in
    # this table, on the single class the project cannot buy anywhere else.
    Concept(
        "display_fixture",
        (
            "display table",  # 6/8, 0.787 -- the island podiums, the priority case
            "product display stand",  # 5/8, 0.639
            "shelving unit",  # 5/8, 0.651 -- the wall runs
            "showcase cabinet",  # 5/8, 0.598
            "display podium",  # 4/8, 0.727
            "retail counter",  # 4/8, 0.620 -- the service desk cam9 missed entirely
            "merchandise rack",  # 4/8, 0.629
            "display case",  # 3/8, 0.561
            "shelf",  # sweep B -- bare noun; `shelving unit` alone misses runs
            "display counter",
            "checkout counter",
            "product display rack",
        ),
        LAYER_THING,
        note="the only class no public dataset supplies; RETAIL.md s5",
    ),
    # --- collides with the above, deliberately and unresolvably ------------------
    #
    # "table" fires on 6/8 frames over largely the same pixels as "display table".
    # ADE20K's `table` is deliberately NOT mapped to display_fixture in
    # label_maps_retail.py, precisely so a dining table does not teach the model it is
    # shop furniture -- and SAM 3 reproduces that ambiguity rather than resolving it.
    #
    # So both stay on LAYER_THING and the overlap becomes 255. A shop table that is a
    # display and a shop table that is a desk look the same from the ceiling; this is a
    # question for the annotator, and guessing it in either direction is how the
    # -0.0096 happened the first time.
    # `trolley` and `trash can` come from sweep B: a shopping trolley is the most common
    # dynamic obstacle in a shop and COCO has no class for it (RETAIL.md s8 raises
    # this as an open question). Here it is at least segmented as an obstacle.
    Concept(
        "obstacle_furniture",
        ("chair", "stool", "table", "office furniture", "trolley", "trash can"),
        LAYER_THING,
        note="`table` overlaps display_fixture on purpose -- contested pixels go to 255",
    ),
    # --- reliable, and cheap to correct -----------------------------------------
    # The one caveat, and it is not about accuracy: on Tao-Hsin-cam14 SAM 3 segmented a
    # convex security mirror as `glass` (correct -- the taxonomy puts mirrors there) and
    # then segmented **the person reflected in it** as `person`. A reflection is not a
    # person in the scene. Same failure through a storefront: pedestrians on the
    # pavement, seen through the pane, come back as `person` inside the shop.
    #
    # It is on LAYER_PERSON, so it overwrites everything -- a false person is the one
    # false positive nothing downstream can outvote. Check mirrors and windows.
    Concept(
        "person",
        ("person",),
        LAYER_PERSON,
        note="8 instances, mean 0.805; segments reflections and people seen through glass",
    ),
    Concept("door", ("door", "doorway"), LAYER_THING, note="6/8, 0.671"),
    # --- SAM 3 cannot do this one, and it is the one that matters most -----------
    #
    # Off by default. Across two sessions and 112 frames glass has now failed in **four
    # distinct ways**, which is what makes it class-level rather than a cut to tune:
    #
    #   1. finds nothing            sweep B, 30 pilot frames: 12 phrasings (glass /
    #                               window / storefront / mirror / reflective glass ...)
    #                               x 7 thresholds, all <=0.12% coverage.
    #   2. finds what is behind it  sweep A, hls-record cam1: 34% of labelled pixels at
    #                               0.65, the mask enclosing pavement, a parked truck
    #                               and two pedestrians seen *through* the pane.
    #   3. finds glossy surfaces    Tao-Hsin-cam06: an entire teal workbench top. No
    #                               transparency involved at all -- colour and specular
    #                               highlight were enough.
    #   4. finds people in it       Tao-Hsin-cam14: a convex security mirror correctly
    #                               marked glass, and then the person *reflected in it*
    #                               marked `person`. See the person concept above.
    #
    # Note the aggregate moves the wrong way: glass reads 3.3% over the 82-frame camera
    # sweep against 0.12% on the pilot, and that rise is modes 2-4, not recovery.
    #
    # None of these is correctable at the threshold, and glass is the class whose
    # failure mode is a robot walking through a pane it read as an open aisle. A
    # confident wrong pre-label here is worse than a blank, because a blank gets drawn.
    #
    # METHODOLOGY.md already ranks glass the highest-priority annotation target.
    # It stays a human class. --include glass only to reproduce the finding.
    Concept(
        "glass",
        ("glass panel", "glass door", "glass partition", "mirror"),
        LAYER_THING,
        min_score=0.65,
        default_on=False,
        note="class-level failure, both directions -- human annotation only",
    ),
    # --- present in the footage, but not worth a model run ----------------------
    #
    # Off by default, and not because SAM 3 fails on them -- "tiled floor" hits 4/8 at
    # 0.706 over 28% of the frame, which is fine. It is that RETAIL.md s5 already
    # measured the cheaper answer: on a camera that never moves the floor is a single
    # static region, so one polygon drawn once is correct for all 1,830 frames that
    # camera will ever produce. Per camera, not per frame.
    #
    # Enable with --include floor_hard if a camera is not in fact fixed.
    Concept(
        "floor_hard",
        ("tiled floor", "hard floor", "floor", "tile floor"),
        LAYER_GROUND,
        default_on=False,
        note="fixed camera: one polygon per camera beats a pre-label per frame",
    ),
    Concept("floor_soft", ("carpet", "mat", "floor mat"), LAYER_GROUND, default_on=False),
    # `ceiling`, `column` and `pillar` are sweep B's, and they earn their place for a
    # reason that is not their own accuracy: they fill the structural holes that would
    # otherwise sit next to the floor as ignore, and RETAIL.md s5 documents a
    # *column* as the exact thing the baseline painted `caution` on in 97.4% of frames.
    # All four are `blocked`, so mislabelling one as another costs nothing to the robot.
    Concept(
        "wall",
        ("wall", "ceiling", "column", "pillar"),
        LAYER_GROUND,
        default_on=False,
        note="6/8; adequately covered already, but see the column false-positive in s5",
    ),
    # --- the only two classes that reach `caution` at all ------------------------
    #
    # Both returned 0/8 on this sweep, and reading that as "SAM 3 cannot find stairs"
    # was wrong. The sweep ran on one store -- a single flat floor with no stairs, no
    # ramp and no sill -- so the 0/8 measured the *building*, not the model. A parallel
    # run over the three-store pilot found `stairs` at **5.2% of labelled pixels** once
    # `escalator` was added to the prompts, which is the term this footage wanted and
    # which is missing from the list this sweep tried.
    #
    # That correction is worth more than its size, because of what it unblocks. Nothing
    # else in the taxonomy maps to `caution`: RETAIL_TERRAIN_TO_TRAV sends only
    # floor_metal, wet_slippery, stairs and threshold_ramp there, and the other two are
    # materials SAM 3 cannot see (below). So `stairs` is the *only* class a pre-label
    # can supply that gives the traversability head any `caution` signal whatsoever. A
    # batch without it trains a three-class head on two classes.
    #
    # The general lesson, since it will recur: a 0/8 on a rare class says nothing until
    # you know the footage contains the thing. Check the site before blaming the prompt.
    Concept(
        "stairs",
        ("stairs", "staircase", "escalator", "step"),
        LAYER_THING,
        note="the only pre-labellable route to `caution` -- 5.2% on the pilot",
    ),
    # --- nothing to pre-label, for two different reasons ------------------------
    #
    # threshold_ramp is the same shape of question as stairs and stays off only because
    # no run has yet found one: it is a solid geometric concept and worth re-testing on
    # footage known to contain a sill or ramp.
    #
    # wet_slippery and floor_metal are a different problem entirely. They are
    # *materials* -- a wet floor is the same object as a dry floor -- and a promptable
    # concept model has no handle on them. METHODOLOGY.md ranks wet_slippery second
    # by priority and calls all three annotate-only; nothing here changes that. Draw
    # them by hand, and film a mopped floor first, because zero frames contain one.
    Concept(
        "threshold_ramp",
        ("ramp", "door sill", "threshold"),
        LAYER_THING,
        default_on=False,
        note="0/8, but so was stairs before `escalator` -- re-test on a site with a sill",
    ),
    Concept(
        "wet_slippery",
        ("wet floor", "puddle", "spilled liquid on floor"),
        LAYER_GROUND,
        default_on=False,
        note="0/8 -- a material, not an object. SAM 3 has no handle on it; hand-draw.",
    ),
    Concept(
        "floor_metal",
        ("metal grating", "steel floor plate", "drain cover"),
        LAYER_GROUND,
        default_on=False,
        note="0/8 -- a material, not an object. SAM 3 has no handle on it; hand-draw.",
    ),
)

BY_NAME = {c.name: c for c in CONCEPTS}
DEFAULT_ON = tuple(c.name for c in CONCEPTS if c.default_on)

# Classes whose pixels change between frames of a fixed camera, and which the consensus
# vote in scripts/sam3_prelabel.py must therefore skip. A person is the whole list:
# people move, so their pixels agree with nothing across a clip and voting on them
# deletes them. Everything else in a shop stays where it was put.
#
# Stated here rather than derived from `layer`, which currently gives the same answer.
# "always paints over everything" and "moves between frames" are different claims that
# happen to coincide for `person`, and a class that acquired the first would otherwise
# be dropped from the vote without anyone deciding that.
MOVING = ("person",)


def resolve(include: list[str] | None, exclude: list[str] | None) -> list[Concept]:
    """The concepts to run, starting from the default set.

    ``--include`` names a class that is off by default; ``--exclude`` drops one that is
    on. Both take class names rather than prompts, so the caller cannot half-enable a
    class by naming three of its eight prompts and then read the coverage table as if
    the class had been searched for properly.
    """
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
