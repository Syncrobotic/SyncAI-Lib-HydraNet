# 2026-08-19 — the 30-frame batch breaks three of the gates that one frame set

The user approved the v5.2 recipe on Kaohsiung-cam04 frame 0002 and asked for a small
batch at the same standard: 30 frames, 3 pilot cameras x 3 daytime slots (10:58 / 14:30 /
19:27 local), night clips excluded. It ran to completion (`runs/site30k_qa/batch30`,
systemd unit `hydra-batch30.service`). Labelled coverage is 79-88% per clip.

**The batch did its job: three gates that separated cleanly on one frame do not
transfer.** None of this is visible in the coverage numbers, which look healthy.

## 1. The floor needed changing before the batch could run at all

The approved recipe intersected the floor with "unchanged against the plate" on every
frame. On frame 0002 that cost 3.4%; on frame **0000 of the same clip** it cost 2.5
points of floor and brought back the fragmentation the user had already rejected — that
frame simply has more people, so more shadow and reflection move. A shadow on the floor
is floor, and the plate is already the person-free median of the clip.

So the floor is now decided **once per clip, on the plate**, and each frame subtracts only
what stands on it. Measured on K04 063027: 13.3% -> 16.9 / 18.3 / 16.6% across three
frames, speckle gone. This is a change from what the user signed off and is recorded here
for that reason.

## 2. A static object changes class between time slots

`column` is accepted in exactly one slot per camera and never on Taichung-cam04:

| camera | 10:58 | 14:30 | 19:27 |
| --- | --- | --- | --- |
| Kaohsiung-cam04 | wall x8, table x1, shelf x2 | wall x8, table x2, shelf x2, **column x1** | wall x9, table x1, shelf x1 |
| Taichung-cam01 | wall x16, table x7, shelf x2 | wall x13, table x6, shelf x4, **column x1** | wall x13, table x6, shelf x3 |

The pillar does not move between 10:58 and 19:27. In the slots where no `column` survives,
the wall family takes it — b03 already reads that pillar as `wall` at 0.98, and without a
winning `column` prompt nothing contests it. The same happens to Kaohsiung-cam04's white
podium: accepted as `display_table` (0.96) at 14:30, rejected at 10:58 and 19:27.

A pillar that is `column` in one tranche and `wall` in the next teaches the model that the
two classes are interchangeable. This is the failure mode the repo already has a memory
for — a sourced class suppressed by the dominant one.

**Root cause, and the fix I'd propose:** the class of a static object is decided per clip,
from that clip's plate. It should be decided **per camera**, pooling the camera's plates
across slots and voting — the same reasoning that just moved the floor from per-frame to
per-clip. Static things should be decided on the most reliable evidence available and then
applied, not re-derived from whichever lighting the tranche happened to have.

## 3. The column flat-share gate has no separating power off K04

`COLUMN_FLAT_MAX = 0.15` came from K04 frame 0002: the real pillar measured 0.04 and the
far-field junk 0.69. Across the batch:

    real pillars (>30k px), rejected:   0.18, 0.18   (Taichung-cam01, two slots)
    junk claims (<12k px), rejected:    0.20, 0.21, 0.31, 0.34, 0.43, 0.45, 0.48

There is no valley. 0.18 against 0.20 is not a boundary, and the K04 gap was one camera's
accident. On this evidence the flat-share test should be **withdrawn**, not retuned; what
does separate here is size (57k against <= 11k), which is camera- and range-dependent and
needs its own measurement before it can be a rule.

## 4. The b03 fixture gate costs a major fixture on every Taichung-cam04 frame

`B03_FIXTURE_MIN = 0.50` was read off K04, where accepted fixtures sat at 0.76-1.00 and
the rejects at 0.00. Elsewhere it rejects large, high-confidence furniture:

    Taichung-cam04  display_table  52k px  score 0.88  b03 fixture 0.04-0.14  (all 3 slots)
    Taichung-cam01  display_table  36k px  score 0.76  b03 fixture 0.20
    Kaohsiung-cam04 display_table  47k px  score 0.96  b03 fixture 0.21-0.36  (2 of 3 slots)

b03's fixture channel does not cover white and reflective furniture, which is most of what
an Apple reseller has. Unlike the column gate this is a consistent miss rather than a
flicker — Taichung-cam04 loses the same counter in every frame of the batch.

## Status

The batch stands as delivered, at the approved standard plus the floor change in section 1;
nothing was retuned to make these findings go away. **I have not verified the 30 frames by
eye**: the IDE host went unreachable when the user closed their session, so image reads
fail. Everything above is from the per-frame decision log (`report.json`), which is why it
is about gates and counts rather than about edges. A contact sheet is written for the user
at `runs/site30k_qa/batch30/CONTACT_SHEET.jpg`; the eye pass is the first task of the next
session.

Proposed next step, for the user's decision: decide static objects per camera rather than
per clip (section 2), withdraw the flat-share gate (section 3), and re-measure the b03
fixture concordance across cameras before it gates anything (section 4) — then re-run the
same 30 frames and compare, rather than adding more cameras on top of gates that are known
not to transfer.

## 5. The eye pass (user, same day) — three defects, two mechanisms

The user reviewed all 30 frames and reported five items; the rest passed. Each is
reproduced below with the measurement that explains it. Boxes are in 1920x1080 frame
coordinates, `dev` is the per-pixel mean |frame - plate| the recipe thresholds on.

**(a) Kaohsiung-cam04 06:30 — a person-free strip of floor on the left is unlabelled.**
The clip's own plate is dirty exactly there: `plate_20260816-063027.png` carries a
smeared standing person across the left floor (visible by eye, blue-green ghost at
x 300-760, y 560-1000). Two independent rules then remove that ground twice over —
`dirty = dilate(b03 == person)` deletes it from `plate_floor`, and the frame disagrees
with the smear anyway: in that box `dev` median is 33.7 for the unlabelled pixels against
9.0 for the accepted floor. Because the floor is decided ONCE PER CLIP (section 1), a
frame in which nobody stands there cannot recover it — no frame is ever re-examined.
This is the per-clip decision meeting a dirty plate, not a teacher failure.

**(b,c) Taichung-cam01 03:00 and 11:30 — the aisle floor is full of holes.**
Opposite mechanism, and the plate is innocent: `plate_20260816-030041.png` is a clean,
empty, person-free aisle. In the aisle box (300,500)-(900,1080) the unlabelled pixels are
statistically indistinguishable from the pixels that WERE labelled floor — dev median 1.7
vs 1.7, 76% of them under dev 3. Nothing changed; they were never claimed. The holes are
baked into `plate_floor` itself, and the floor-diag numbers say why: on this camera b03's
floor channel reaches 15.63% of the frame and the metric on-plane test 14.01%, union
~21.1% — against Taichung-cam04's 37.16 / 36.68. The on-plane share is also saturated
(15.23% at tol 0.10 -> 15.37% at 0.30), so widening the tolerance cannot help: the depth
does not put the pale glossy tile on the plane at any tolerance, and b03 does not call it
floor. Same white-and-reflective blind spot as section 4, now costing floor rather than
fixtures.

**(d,e) Taichung-cam04 06:31 and 11:30 — the computer on the right counter is missed.**
It is a desktop monitor with a keyboard in front of it, and the batch driver's product
families have no concept for one: `laptop` is `("laptop", "open laptop",
"notebook computer")` at min_score 0.45, while `keyboard` sits in the EXCLUSION list.
So the same object fails two different ways in two slots — at 06:31 no claim clears 0.45
and it is not even refused (no `laptop` line in that clip's `products_refused`), and at
11:30 weak claims 0.46/0.48/0.50 are refused as `contested with keyboard` 0.35-0.46. The
repo's own prompt library already carries what is missing: `computer monitor` (27/48
cameras, peak 0.930) and `imac` in `sam3_prompts_objects`. Two fixes, both narrow: give
the computer family those prompts, and stop letting an exclusion that lies INSIDE the
claim veto it — a keyboard in front of a monitor is part of the workstation, not evidence
against it.

Together (a) and (b,c) say the floor needs the same move the static objects need: decide
it per CAMERA over pooled plates, so a slot whose plate is dirty in one place borrows a
slot where that place is clean, and so a camera whose teachers are weak gets every plate's
evidence rather than one clip's.

## 6. v5.3 rerun — the same 30 frames, the three fixes, measured

`hydra-batch30-v53.service` re-labelled the identical 30 frames with the driver at
`scratchpad/batch30_v53.py` (v5.2's driver untouched, so the two runs are comparable).
Output `runs/site30k_qa/batch30_v53/`, log `runs/site30k_qa/batch30_v53.log`, per-frame
deltas `runs/site30k_qa/v52_vs_v53/compare.json`. Exit 0, 30/30 frames.

What changed, and nothing else was retuned:

1. The floor is decided once per CAMERA over all three of its day plates — dirty only
   where every plate is dirty, evidence wherever any plate saw floor.
2. Candidacy is the union of the two floor teachers: `(on-plane & b03 context) |
   (b03 floor)`. v5.2 required the geometry, so b03's own floor was evidence that could
   never be admitted.
3. `computer monitor` and `imac` join the laptop family; `keyboard` and `computer mouse`
   stop counting as exclusions for that family only.

Result over the batch: mean labelled 84.5% -> 86.9%, mean floor share 22.80% -> 25.50%,
floor grew on 30 of 30 frames, laptop share grew on 22 of 30. The four frames that gate
the pooled floor show it clearest — Kaohsiung-cam04 06:30 frame 0000 goes 81.0 -> 86.3%
labelled with floor 16.88 -> 22.15%, and Taichung-cam01 03:00 frame 0000 goes
77.9 -> 83.6% with floor 11.99 -> 20.40%.

Verified by eye, at 2x, on the five frames of the eye pass:

* K04 06:30 left strip — the dark floor band in front of the counter is floor now; in the
  box (300,560)-(760,1000) unlabelled falls 38.3% -> 17.2% and floor rises 31.2 -> 52.3%.
* T01 03:00 and 11:30 aisle — the aisle is floor across its width. Box floor 43.1 ->
  52.6% at 11:30.
* T04 06:31 — the service-counter monitor is painted whole, and its `laptop` refusals are
  gone from the decision log (0 laptop refusals across all seven T04 frames, against 6 at
  11:30 in v5.2).
* T04 11:30 — the same machine is painted on frames 0001/0002. On frame 0000 it is
  occluded by a person standing in front of it, which is the correct outcome, not a miss.

**Known residual, measured, not fixed:** a specular patch in Taichung-cam01's aisle stays
unlabelled in ALL THREE slots — 42,810 px, 12.3% of the aisle box, unlabelled in every
slot's frame 0000. Pooling cannot help what every plate refuses: it is a bright reflection
that b03 does not read as floor and whose depth is not on the plane. It is the same white
and reflective blind spot as sections 4 and 5, and the next thing to attack if T01's floor
needs to be complete.

## 7. Second and third eye passes, and the decisions they forced

The user reviewed the v5.3 rerun and reported two more defects; a third came out of my own
check of the fix. Each was measured before it was changed, and each fix is bounded by that
measurement. Runs: `batch30_v54` (third floor source), `batch30_v56` (chroma-gated
withdrawal, six parallel workers), `batch30_v57` (object-hole veto widened).

**Taichung-cam01's aisle hole was not a geometry failure — it was b03.** The hole
(component #4, 36,042 px, refused in all three slots) measures |height| p10/p50/p90 =
0.002 / 0.007 / 0.015 m with 94.9% of its pixels horizontal: it IS the ground. b03 reads
it as `fixture` in every plate (100% / 96.8% / 99.4%) because the polished tile mirrors
the counter. Fix: a THIRD floor source — on-plane at |height| <= min(tol, 0.05 m) and
horizontal, whatever b03 says. Cost measured before running: it adds 1.90% of frame on
T01, 0.65% on K04, 0.21% on T04, of which 96.6-100% is currently unlabelled, and it claims
no pixel of any accepted wall, table or shelf.

**The middle cabinet on Taichung-cam04 lost SLIC-shaped blocks to the change-withdrawal
rule.** The blocks are a person's shadow and reflection on a glossy cabinet that has not
moved. In YCbCr the separation is clean: the lost blocks move dY 15 / dC 2 at the median
(94% under dC 4), the person's own pixels dY 53 / dC 7 (39% under dC 4). Gating the change
test on dC > 4 keeps 100% of the cabinet's blocks, drops withdrawn area from 7.45% to
0.52% of frame, and still withdraws the person's cells. Same principle the floor already
uses: a shadow on the floor is floor.

**The third source claimed a tray of accessories on T01's counter** (3,112 px the depth
puts at ground level). Vetoed by the object layer: floor inside a hole fully enclosed by
one accepted object is part of that object. Measured over all 30 frames, that rule touches
9,336 px total and every one of them is this tray. A connectivity variant was tried first
and measured not to work (the tray shares a component with the real floor); it was removed
rather than kept as decoration.

Result over the batch: mean labelled 84.5% (v5.2) -> 86.9% (v5.3) -> 89.3% (v5.7); the
T01 aisle box goes 45.8% unlabelled -> 12.0%; the T04 cabinet box 23.6% -> 2.6%.

## 8. Throughput: the pipeline is not GPU-bound

Measured while a batch ran: GPU utilisation 0-19% with 11 GB of 97 GB used, and ONE of 24
CPU cores busy. Six workers together still leave the card at 0-24%. The per-frame cost is
34 SAM 3 prompt calls (1 moving + 16 product + 17 exclusion), each a separate
tokenise/forward/post-process round trip, plus numpy stages that never touch the GPU
(SLIC 5.3 s and the guided filter 1.3 s per clip, measured).

So the batch now runs as six processes over one output directory. The floor is pooled per
camera, so a worker is given its camera's THREE clips and renders only the ones in
`--only`: the recipe is bit-for-bit what one process produces. Wall clock for the same 30
frames: 14.6 min -> **4.9 min**. Filling the card needs prompt batching inside
`sam3_prelabel.segment`, which is the next throughput item -- at 35 s/frame, 30k frames is
292 GPU-hours.

## 9. Decisions taken by the user on this batch

* `pet` is REMOVED from the campaign scope. The census (2477 boxes over 336 frames) tops
  out at 0.5427 with p99 = 0.2018 and no threshold was ever defensible, so the class would
  ship with zero positives and never fire. It leaves the taxonomy rather than sitting in
  it unlabelled.
* Approved and queued, in order: (1) a human-judged baseline over these 30 frames, which
  is the only route from teacher-agreement to accuracy; (2) static objects decided PER
  CAMERA over pooled plates, the same move the floor made -- `column` is currently
  accepted in one slot per camera and never on Taichung-cam04, and the student cannot
  learn a class whose label flickers between tranches (site IoU 0.178 today);
  (4) products decided once on the plate and only subtracted per frame, so an occluded
  product stops splitting into two or three boxes; (5) prompt batching.
