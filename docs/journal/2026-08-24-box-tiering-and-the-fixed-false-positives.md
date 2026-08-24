# Confidence tiering over the existing boxes, and 43% of the hallucinations have a postcode

[PLAN.md](../PLAN.md) §7 step 4's zero-GPU item, run 2026-08-24 over
`datasets/site30k_v1/annotations/instances_all_*.json` — the raw Grounding DINO population
`box_pass.py` deliberately kept down to score 0.10 with no NMS, "so the score population
stays visible and a future threshold question can be answered without GPU time". This is
that question, and no GPU was used.

## 1. What the file actually contains

`instances_all_*.json` holds **only `person`**. The `device` and `boxed_stock` boxes exist
only in `instances_*.json` and carry **no score**, because they are connected components of
the segmentation mask's own product ids (7–10) rather than a detector's opinion. So
confidence tiering is a `person` question; the product boxes need a different check (§4).

| split | images | raw person boxes | median score |
|---|---:|---:|---:|
| train | 16,146 | 198,234 | 0.15 |
| val | 9,828 | 101,572 | 0.15 |
| test | 3,237 | 75,725 | 0.14 |

## 2. NMS, and the tiers

The population has no NMS. At IoU 0.5 on the test split, **17.5% of boxes are duplicates**
(75,725 → 62,468), leaving **19.30 person boxes per image** on a shop-floor frame that
typically holds one to four people.

| tier | rule | share after NMS | per image |
|---|---|---:|---:|
| **Gold** | score ≥ 0.50 | 10.2% | 1.96 |
| **Silver** | 0.35 ≤ score < 0.50 | 2.1% | 0.40 |
| **Gray** | score < 0.35 | **87.7%** | **16.93** |

## 3. The accept/reject pass — 60 crops per tier, judged by eye

Sheets rendered as crop grids and inspected (`scratchpad/tiers/tier_{gold,silver,gray}.png`):

* **Gold: 60 of 60 are real people.** Well clear of PLAN §7's ≥95% gate. Partial bodies —
  a head over a counter, a torso behind a display — are common but they are all people.
* **Gray: roughly 3–4 of 60.** Precision on the order of 5–7%.

**This settles the training rule with a number.** Seventeen Gray boxes per frame cannot be
negatives — a model taught that would learn to suppress `person` across the whole frame —
and they cannot be positives either. `Gray → ignore region` was the right call in the plan
and is now measured rather than assumed.

Incidental but useful: the staff uniform is a **blue STUDIO A polo**, unmistakable in nearly
every Gold crop. `staff / customer` looks very learnable from a crop, which supports it being
the attribute worth building first.

## 4. The finding: 43.4% of Gray boxes come from a handful of fixed positions

Binning Gray box centres into 64 px cells per camera, and keeping only cells that fire on
**more than half of that camera's frames** — a threshold no moving person can meet:

| camera | fixed hotspots | share of that camera's Gray boxes |
|---|---:|---:|
| Tao-Hsin-cam03 (**the test camera**) | 6 | **65.1%** |
| Taichung-cam07 | 6 | 57.7% |
| Taichung-cam11 | 11 | 57.7% |
| Taichung-cam04 | 3 | 49.4% |
| Taichung-cam05 | 2 | 30.9% |
| Taichung-cam10 | 4 | 24.1% |
| Tao-Hsin-cam04 | 3 | 22.5% |
| Taichung-cam01 | 2 | 21.2% |
| Kaohsiung-cam04 | 0 | 0.0% |
| **all nine** | **37** | **43.4%** (127,460 of 293,381) |

Rendered over a real frame per camera, the hotspots are identifiable by eye and fall into
exactly two families:

* **hanging packaged accessories on wall displays** — Taichung-cam11's accessory wall,
  cam07's back-left merchandise wall, cam10's right-hand wall. A grid of person-shaped
  blister packs.
* **printed people** — Tao-Hsin-cam03's "The Power of Sound" standee by the glass doors,
  which is the single most repeated crop in the whole Gray sample; posters on cam04 and
  cam05.

This is the daytime, quantified version of a failure already recorded at night
(`person` firing on hanging packets on an empty IR frame, 14 false people, consensus voting
unable to remove them). It is not a night problem. It is a **fixed-object** problem, and it
accounts for nearly half the hallucination population.

Kaohsiung-cam04 is the instructive exception: zero fixed hotspots, and its sample frame shows
four real people crowded at a counter. Its Gray boxes are near-misses on real people, not
furniture — a different failure needing a different fix.

## 5. What this earns: a rung-1 fix, derived not drawn

Per [PLAN.md](../PLAN.md) §1, an answer that is constant on a fixed camera belongs in a
config. These hotspots are constants, and — importantly for §5's no-manual-annotation policy
— they are **measured, not hand-drawn**: the cells come from the data, so commissioning can
*propose* the polygon list and a human only accepts or rejects it.

Add to §4 commissioning: a per-camera **known-false-positive polygon list**. Cost at runtime:
nothing. Effect: removes ~43% of the Gray population before it ever reaches the ignore-region
logic, and removes the same boxes from inference, where today they would become 17 phantom
tracks per frame feeding occupancy, dwell and crowd events.

## 6. Product boxes survive the mask defect

The `device` / `boxed_stock` boxes are derived from the segmentation mask, and that mask was
measured on 2026-08-20 to collapse on 10 of Tao-Hsin-cam03's 29 dates (the counter bank
flipping to `wall`). Checking whether the product boxes flip with it, on the test camera:

| mask state | frames | device/frame | boxed_stock/frame |
|---|---:|---:|---:|
| ok (19 dates) | 2,067 | 13.6 | 1.4 |
| collapsed (10 dates) | 1,170 | 11.5 | 1.2 |

A 15% dip, not a collapse — because the product ids are painted by SAM 3's own product
prompts and are independent of the structure decision that moved counters between
`display_table` and `wall`. **The detection output of the 2026-08-20 campaign is materially
more usable than its segmentation output.**

## 7. Where this leaves step 4

* Gold passes its gate by eye at 60/60; a larger sample would tighten it, but not change the
  verdict.
* Silver's 0.40 boxes per image is a thin tier. Worth asking whether 0.35–0.50 earns a
  separate weight at all, or whether the useful split is simply Gold versus ignore.
* CoTracker3 temporal filtering was to be decided by this measurement. **It is still worth
  running, but for a smaller job than planned:** the fixed-hotspot polygons remove 43% of the
  Gray population for free, and Kaohsiung-cam04 shows the remainder is a different problem —
  near-misses on real people — which is exactly what temporal consistency is good at.
