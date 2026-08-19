# 2026-08-19 — the `column` supplement was leaking, and re-sweeping for more cameras found none

Worker-session record, opened from a question that had a wrong premise: *"can't SAM 3
segment walls and columns?"* It can, and it is the designated teacher for both
([METHODOLOGY.md](../METHODOLOGY.md) §2). The claim that had been mixed up with it is that
**ADE20K's** `column` does not transfer — a fact about the data source, not about SAM 3.
Chasing the real question (why `column`, `pillar` and the display tables train badly)
turned up a leak, a retraction, and two negative results worth not repeating.

Related same-day work: the fleet census (`runs/camera_census01/REPORT.md`) and the site30k
campaign, both in the planner session.

---

## 1. RETRACTED: the `column` 0.454 → 0.505 from the columns supplement

`runs/hydranet_retail_surfaces_columns{,_seed7,_seed13}` measured `column`/site_seg at
0.449 / 0.502 / 0.563 against the base run's 0.451 / 0.426 / 0.486. **That comparison is
withdrawn.** Not because the runs were wrong — they were honest when they ran — but
because the val they were scored on no longer exists. `split.json` was rewritten at 19:47
the same evening, three hours after the last seed finished, and
[RETAIL_DATA.md](../RETAIL_DATA.md) §4 already says of that resplit: *none of them is
comparable to anything measured after this change.* That sentence covers these runs too.
The seed ranges also overlapped, so the delta was not separable from seed noise even on
its own terms.

## 2. The leak, and why nothing caught it

```
14:51  datasets/retail_objects_columns_clean built — 9 cameras "in neither val nor test"
15:10 ┐
15:29 ├ the three hydranet_retail_surfaces_columns seeds finish
15:41 ┘
19:47  split.json rewritten: the third `moves` entry promotes Taichung-cam10 to val
```

`Taichung-cam10` is 12 of the supplement's 108 training frames and is now **val** in both
batches; `Kaohsiung-cam02` is 12 more and is now **batch03 test**.

`resplit_selling_floor.py` could not have caught it. Its `BATCHES` names batch02 and
batch03, and the supplement carried **no `split.json` at all** — train-only by config, so
there was no assignment to move or refuse. The config's own comment had predicted the
shape and got the direction backwards: it warned that a batch *selected on a property*
selects against the split it belongs to. This is the reverse — **a static supplement
invalidated by the split moving underneath it**, with no record of which split it was ever
clean against.

**Fixed structurally, not by vigilance.** A dataset may now carry a `clean_against` block
naming the batches it was built against and a hash of their assignment at the time.
`resplit_selling_floor.py` verifies every such block before a move and refuses by name:

```
REFUSED: this move invalidates a train-only supplement:
  datasets/retail_objects_columns_v2 trains on Taichung-cam11, which this move sends to val
```

A supplement with no block is reported as *unprotected* rather than assumed safe — "nothing
to check" and "checked and fine" are the two states this failure confused.

`datasets/retail_objects_columns_v2` is the clean rebuild: the same masks hardlinked, minus
the two cameras, 7 cameras and 84 frames. `columns_clean` is **kept and marked superseded**,
with a digest deliberately set to a value that can never match — it is what those three
seeds actually trained on, and deleting it would make them unreproducible.

## 3. Two negative results, so nobody pays for them twice

**The rotation arm is zero.** The census found five sideways ~90° mounts (K-05, K-09, K-10,
T-02, TH-05), not the one the roll census had counted, and sweep C had asked all five for a
"column" without rotating them — where a column is a horizontal bar. Re-asked upright,
**not one of the five gains a column at 0.50**. Reasonable hypothesis, wrong. Do not retry.

**The 0.25–0.50 band yields no trainable camera.** `scripts/column_camera_sweep.py`
reproduces sweep C and, unlike it, writes the identities down; `runs/column_sweep03/REPORT.md`
is the output. Of everything it surfaces, exactly two cameras are both new and trainable, and
**both were opened at native resolution and rejected**:

| camera | score | what the mask is actually on |
|---|---|---|
| Tao-Hsin-cam11 | 0.594, evening only | street bollards seen **through a glass door** — glass failure mode 2, "finds what is behind it" |
| Taichung-cam09 | 0.359, never clears 0.50 | a narrow vertical strip beside a display podium — the documented drift onto "narrow vertical thing" |

The trainable population for `column` therefore stays at **6 cameras**, and the priority-4
line "more cameras, not more frames" is exhausted inside the current pull.

## 4. Tranche consistency, which is the one reusable thing here

Sweep C had two frames per camera. This sweep has four store-local tranches, and the count
of tranches in which a camera clears 0.50 separates a column from a lighting artefact
nearly by itself:

| tranches @0.50 | cameras | in the shipped 14 |
|---|---|---|
| 4 of 4 | 11 | 11 |
| 3 of 4 | 2 | 2 |
| 2 of 4 | 1 | 0 |
| **1 of 4** | **3** | **1** |

**Require 3 of 4 before a camera supplies a training pixel.** The rule also flags a sitting
member: `Taichung-cam04` is in the supplement and clears 0.50 at midnight only, on a fragment
behind the back counter that never reaches the floor. Flagged in `columns_v2`'s `split.json`
rather than removed — that is a judgement about the class, not about this leak.

## 5. `datasets/studioa_clips/_survey` was a trap; renamed

Every plate in it came from `manifest_2026-08-16_1130-1600UTC-mislabelled.json` — the pull
whose UTC-vs-local bug `pull_studioa.py` documents — so all 48 are ~19:30 store-local. The
first run of the sweep used them as its "daylight" arm and was thrown away.

Renamed to `_survey_evening_1930local`, with a README, because the name was the only thing
that could have prevented it. **The fleet census is unaffected** — checked rather than
assumed: its plates for K01 / K08 / T09 sit at mean |diff| 3.7 / 4.0 / 1.8 against the midday
frames versus 25.6 / 4.1 / 5.3 against the evening ones, so it used the correct pull.

## 6. What this does not settle

Every number above is still an **agreement with SAM 3**, not an accuracy. R3 is unsatisfied,
the test masks are the teacher's own output, and a `column` mask this sweep calls stable
across four tranches is stable — not right. The priority-1 line has not moved: until the
72 test frames are human-corrected, this class has no site number at all.
