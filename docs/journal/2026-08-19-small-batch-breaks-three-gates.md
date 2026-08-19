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
