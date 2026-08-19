# 2026-08-19 — the immovable classes, decided per object

The floor recipe (v4.2) was with the user for approval when they asked for the immovable
classes next: wall, column, display_table, shelf. This entry records what forced the
design and what it costs, because two of the measurements invalidate a rule the campaign
had already adopted.

## The camera cannot see high, and that kills the v2 height bands

`runs/onboard01/Kaohsiung-cam04.calib.json`: height 2.17 m, pitch 52.33°, vfov 70.4°.
The top edge of the frame therefore sits at 52.33 − 35.2 = **17.1° below horizontal** —
every ray in the image points downward. At 3.5 m (the accessory wall) the top of the
image reaches 1.09 m above the floor; the measured ceiling of the entire height map is
**1.75 m**, and only 184k of 2.07M px exceed 1.5 m, all in one region of the left wall.

The v2 fixture split classified pixels by height band with `shelf ≥ 1.40 m OR (vertical
AND h ≥ 1.00)`. On this camera that upper band is essentially unreachable, so the rule
falls back to painting *parts* of objects — which is also how a table's side panel came
to be called `shelf` in the v1 pilot. The bands were swept honestly (
`runs/site30k_qa/fixture_height_sweep.json`); what was not checked was whether the view
contains the range the band names.

**Ruling: the class is decided once per OBJECT, not per pixel.** A per-pixel veto is
also precisely what produced the speckle and the stair-stepped edges the user rejected
in v1–v4.1, so this serves both problems at once.

## 21.6% of the frame has no measured geometry

`undistort_image` resamples to the undistorted frame **at the same canvas size**. With
k1 = −0.225 (barrel) the undistorted radius is larger than the distorted one, so the
outer ring of the frame maps outside the canvas: for 448,569 px (21.6%) the sample
coordinate is clamped to the plate border and the height/horiz values there are the
border's, not the pixel's. The constant-height blobs at the frame corners (h = 1.72 m
everywhere) are this artifact.

It reaches the reviewed floor too: **22.0% of the v4.2 floor sits in that ring**, though
76.6% of those pixels are carried by the b03 floor channel, which needs no geometry.
The remainder rests on a clamped on-plane test. Not corrected here — recorded, and the
v5.0 rules abstain wherever a decision would depend on clamped geometry.

## The recipe

SAM 3 static concepts run on the **plate** (person-free by construction, so a shopper
cannot take a cabinet's label with her) → 89 instances → 18 clusters by mask IoU. Each
cluster is named by three teachers, and each teacher is there for a failure the others
cannot catch:

| teacher | what it decides | the failure it catches |
| --- | --- | --- |
| prompt family | the object's name | — |
| b03 terrain channel | independent confirmation | `display table` 0.78 standing on 94% floor pixels (rejected) |
| metric geometry | table vs shelf, and column sanity | a `column` claim whose surface is 69% horizontal (rejected) |

`column` is exempt from the b03 gate, and this frame shows why in the strongest possible
terms: **b03 emits zero column pixels in the entire frame and reads the real structural
pillar as `wall` at 0.98.** It is the documented site failure; it cannot be its examiner.

Gates, each read off this frame's own spread:

- b03 wall share ≥ 0.60 (accepted walls measure 0.70–1.00)
- b03 fixture share ≥ 0.50 (accepted 0.76–1.00; the rejects are 0.00)
- column: ≥ 0.50 measurable geometry, top-surface share ≤ 0.15 (real pillar 0.04)
- fixture score ≥ 0.65 — the midpoint of the widest gap in the frame's accepted-score
  distribution (0.95/0.91/0.84/0.80 against 0.50/0.43). The two below the gap are the
  counter's dark service recess traced as `shelving unit`, confirmed by eye at 4x. **A
  one-frame reading; re-measure it on the 10-frame batch.**
- a claim with ≥ 80% of its floor footprint inside a stronger claim's footprint (± 0.15 m,
  the calib's own stated scale band at that range) and a *different* class is a part of
  that object, not an object. Same class → no action; it is one object seen twice.

Painting: whole objects, strongest score first, so a weaker claim never overwrites a
stronger one and no veto can punch a hole. Enclosed holes are filled for every class —
a hole surrounded by wall is signage, a hole surrounded by shelf is one packet among the
hundreds the same mask already covers, and products are carried by the **detection** head
as boxes, so filling the surface class loses nothing. Clutter that is *open* to the
outside (the signage heaped on the counter's right end) is not enclosed, stays
unlabelled, and is named in the accounting. Withdrawal for people and changed pixels is
at superpixel granularity, so a hole has the shape of the occluder and not of the noise
threshold.

## What it cost the floor, and why that is the point

The floor/object conflict is 7,053 px (0.34% of frame) and now resolves to the **object**.
That reads like overruling a reviewed decision and is the opposite: v4.2 was delivered
with exactly one named residual — the thin collar at the base of a pillar or fixture,
inside tol(r) of the ground plane, which the floor takes because |h| cannot separate a
skirting board from the tile it stands on. An object mask is the missing evidence for
those pixels. Resolving the contest per superpixel (the first attempt) handed the collar
back to the floor anyway, because a SLIC cell at the pillar base holds more floor than
skirt. **The pillar skirt is now `column`, verified by eye at 2x.**

## Result

Labelled 86.0% of the frame: floor 15.3, wall 27.6, column 3.0, display_table 14.2,
shelf 8.0, person 17.9. The unlabelled 14.0% is named: changed_vs_plate 2.13,
refused_claim 0.63, unclaimed_background 11.19 — and 60% of that last bucket is b03
`fixture`, i.e. goods and equipment on the counter, which this taxonomy carries as
detection boxes and has no pixel class for.

Verified personally at 2x on five crops (counter edge, pillar base, shelf wall, podium,
door/wall). Object boundaries follow real image edges. **The staircase that remains at
the wall/floor junction and around the cabinet base is the v4.2 floor's own SLIC snap
imprinting its 60 px cell where the image has no edge to snap to** — worth revisiting if
the user still sees it.

One call is flagged for the user rather than decided: the white cylinder top-right scores
`display table` 0.95 against `round column` 0.83 (margin 0.101) and its geometry abstains
(23% measurable — it sits at the frame's top edge). By eye it is a display podium with
phones on it and a base ellipse on the floor, so it is kept as `display_table` and
flagged.

## Where the code is — a durability problem

The v5.0 driver is `single_frame_v50.py` in the planner session's scratchpad and is
**not in the repository**: `scripts/` carries an upward-only ratchet on
script-imports-script (`tests/test_scripts_are_not_libraries.py`) and the composition
logic belongs in the package anyway. It lands after the user's verdict, as a package
module with the acceptance criteria as tests.

More urgently: **the reviewed v4.2 floor recipe exists only as `single_frame_v42.py` in
the campaign agent's scratchpad** (`.../2673d35b-.../scratchpad`). It is reproduced
verbatim inside the v5.0 cache builder, but if that session's scratchpad is cleared
before either is productised, the recipe the user approved is gone.
