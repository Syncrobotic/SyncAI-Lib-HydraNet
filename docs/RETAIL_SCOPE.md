# Retail scope: what to build, and what not to build

The ask: a network for retail stores that finds **walkable area within 5 m of the camera**,
**objects** (COCO classes), and **display tables / merchandise areas**.

Two of those three are perception. One of them is geometry wearing a perception costume,
and this document is mostly about telling them apart — because the difference decides
whether the annotation effort is reusable or thrown away when the camera moves 10 cm on
its mount.

Related: [METHODOLOGY.md](METHODOLOGY.md) for the process, [ANNOTATION_SETUP.md](ANNOTATION_SETUP.md)
for the labelling contract, [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) for what the
existing heads are measured to be worth.

---

## 1. Not a new network — a new config

`configs/hydranet_retail.yaml` and `data/label_maps_retail.py` are the whole change. The
shared-trunk premise is measured: 84.4% of parameters sit in the backbone and neck, and a
fourth head would cost 3–9%. Retail and indoor overlap almost entirely — polished floor,
glass storefront, escalator, wet floor after a mop, people, doors — so a second network
would mean training the same trunk twice and maintaining two of everything.

The taxonomy is the indoor 12 with **one class appended**:

| id | class | traversability |
|---|---|---|
| 0–11 | unchanged from `label_maps_indoor.py` | unchanged |
| **12** | **`display_fixture`** — shelving, gondola, display table, counter, rack | blocked |

Ids 0–11 are byte-identical on purpose. A retail run can warm-start from an indoor
checkpoint by re-initialising a single output channel instead of the whole terrain head,
and any masks already annotated under `indoor_native` stay readable. `tests/test_retail_scheme.py`
pins that alignment, because renumbering the list is exactly the kind of tidy-up that breaks
nothing visibly and invalidates every mask and checkpoint at once.

Whether retail and indoor share a *checkpoint* is a separate question from sharing the code.
With the data volumes this project has today, train them separately.

---

## 2. "Within 5 m" is not a class

This is the important one.

Distance is geometry. It is not a property of how a patch of floor looks, and a network
that predicts it from appearance is doing depth estimation badly, in disguise. Three
consequences, in increasing order of expense:

1. **Annotators cannot label it consistently.** 4.8 m of floor and 5.2 m of the same floor
   are the same pixels. Whatever line gets drawn is one annotator's guess, and the
   disagreement between annotators trains as noise — in the class that the robot's motion
   depends on.
2. **It dies when the camera moves.** Mount height, pitch, lens, a knock in a loading bay:
   any of these changes which pixels are within 5 m, and every label ever drawn becomes
   wrong at once. Nothing in training will report this.
3. **It freezes a product parameter into the weights.** 5 m is a planning horizon, and
   planning horizons change — per store, per robot speed, per aisle width. In
   post-processing it is a config value; in the weights it is a retraining run.

[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) already reached the same conclusion from
the other direction, for a depth head: *do not build it, LiDAR measures it better than a
monocular head ever will.*

There is also a price tag on the alternative. This model scores `stairs` at 0.32 — a large,
high-contrast, geometrically distinctive class with **314 training images** behind it.
Anything subtle that has to be *learned* costs far more than it looks like it will.
Distance, computed geometrically, is a few lines of post-processing with an exact answer;
distance, learned, is several hundred annotations and a result nobody can bound.

### What to do instead

The network outputs walkable area as semantics. The 5 m test happens after it, on the
ground plane, in one of two ways:

**A. Ground-plane homography — recommended for retail.** A shop floor is flat and the
extrinsics are fixed, which are exactly the conditions under which a homography is exact
rather than approximate. Calibrate once per camera mount:

1. Put four markers on the floor at known positions (a 2 m × 2 m square works; measure it,
   do not assume it).
2. Click the four image points, solve for `H` with `cv2.findHomography`.
3. Precompute a **per-pixel ground-distance lookup table** once at startup. Runtime cost of
   the 5 m test then drops to a threshold compare against a cached array — on the order of
   0.1 ms, against the 37.8 ms frame we measured on the AGX Orin.
4. Verify with a tape measure at 2, 4 and 6 m before trusting it.

Failure modes to know: it is only valid on the ground plane (a shelf face at 3 m is not at
its footprint's distance), it degrades on ramps and thresholds, and re-calibration is
required whenever the mount changes — which is a five-minute job rather than a relabelling
project.

**B. LiDAR mask.** The robot already carries LiDAR, it needs no flat-floor assumption, and
it is the more accurate of the two. It costs an extrinsic calibration between LiDAR and
camera, which is more work than A and worth it if floors are not reliably flat.

Either way the model stays honest about what it can see, and the horizon stays a number in
a config file.

---

## 3. The merchandise *zone* is not a class either

"Display table" and "the merchandise area around it" are two different things, and only one
of them is annotatable.

- **The fixture** is a physical object with visible edges. Two annotators will draw nearly
  the same mask. That is `display_fixture`, id 12.
- **The zone** — the floor that belongs to a display, where customers stop and browse — has
  no visual boundary at all. Ask two annotators to draw it and they will disagree by a
  metre, consistently, and the model will learn the average of their disagreement.

Derive the zone in bird's-eye view instead: project the fixture mask through the same
homography, dilate its footprint by whatever the product decides (1.5 m, say), and the
answer is a number rather than an opinion — and adjustable per store without touching the
model. This is the same argument as the 5 m rule, and it has the same fix, which is a sign
the fix is the right shape.

If instance identity is later needed — "how many displays, which one is this" — that is
instance segmentation and a real scope increase. Do not build it now; the semantic mask
plus connected components in BEV answers most of the question for free.

---

## 4. Objects: train on 80, export about 8

Keep `detection.num_classes: 80` in training. COCO supervision is free signal for the
shared trunk, and dropping classes at training time does not make the robot faster.

Narrow at **export**. On the AGX Orin the measured breakdown was: GPU inference 5.12 ms
(14% of the frame) and post-processing 16.33 ms (43%), nearly all of it the sigmoid over 80
classes at 6,825 positions — 546,000 values per frame. About eight classes change a shop
robot's behaviour (`person`, `chair`, `backpack`, `handbag`, `suitcase`, `bottle`, `cup`,
`potted plant`); the rest are traffic. Narrowing is a ~10× reduction in post-processing
work, and it is the single largest latency win available.

What COCO does **not** contain: shopping trolleys, baskets, price signage, or any product
SKU. If the product needs those, they are annotation, and they are detection-format
annotation — a different labelling job from the segmentation one, with a different tool
configuration. Decide whether it is in scope before promising it.

---

## 5. The data, which is the actual project

Same conclusion as indoor, and it is worth restating because it is the one that gets
staffed wrongly: **the model is not the constraint.**

| output | can public data bootstrap it? |
|---|---|
| objects | Yes — COCO, already integrated, nothing to do |
| walkable area | Partly — ADE20K floors work, but it is human-height web photography. The indoor runs measured a ceiling coming back **25% "go"** from the robot's viewpoint. |
| display fixtures | Barely — ADE20K's shelves, counters and cases give the shape of a waist-height surface holding objects, in homes and restaurants. Not a shop. |

Note what `ade20k_retail` deliberately does *not* map: ADE20K's `table` (id 16) stays
`obstacle_furniture`, even though a display table is precisely what was asked for. Its
tables are domestic and dining, and mapping them would teach the model that a proxy which
is nearly the right thing is the right thing. This project has already paid for that
mistake once, with COCO diluting the segmentation heads. Both classes are `blocked`
regardless, so the cost of the omission is semantic, not a safety regression.

### Capture rules, in priority order

1. **Robot camera height and pitch.** Not eye level. This is the single biggest gap between
   public data and field performance.
2. **Aisles, not just open floor.** Gondola runs, end caps, narrow aisles, and the reflective
   floor under strip lighting.
3. **Storefront glass and mirrors**, backlit entrances, glare on polished floor. `glass` is
   the highest-consequence class indoors and a shop is made of it.
4. **After the floor is mopped.** `wet_slippery` has zero examples today and no public
   dataset supplies it.
5. One directory per capture session, logged.

### Splitting: by store, not by session

Stricter than the indoor rule. A single shop's aisles repeat, its lighting is uniform and
its fixtures are identical by design — so a val split taken inside the same store measures
memorisation of that store, and will look excellent. **Hold out entire stores.** Three
stores is the practical minimum for a number that means anything.

---

## 6. Two traps, both already measured on the indoor model

### Multi-task dilution: keep COCO at `sample_ratio: 0.1`

Swept at 60 epochs and scored on the held-out test split:

| | segmentation only | **COCO 0.1** | COCO 1.0 |
|---|---|---|---|
| `glass` IoU | 0.5021 | **0.5462** | 0.0692 |
| `caution` IoU | 0.3252 | **0.3346** | 0.1064 |
| `stairs` IoU | 0.3249 | 0.3174 | 0.0725 |
| terrain mIoU | 0.5718 | **0.5968** | 0.3591 |
| detection mAP | — | 0.1985 | **0.3348** |

At 0.1, detection is free: every segmentation number matches or beats the
segmentation-only baseline and mAP 0.20 arrives anyway. At 1.0, the model goes very nearly
blind to glass — 0.502 to 0.069 — in exchange for mAP 0.199 → 0.335.

For a shop that trade is unacceptable in a way it might not be elsewhere. A retail floor is
*made* of glass: storefronts, display cases, mirrored columns. Glass is also the first class
to collapse under dilution, and the one whose failure mode is a robot walking into a pane it
read as an open aisle. If detection needs to be better than mAP 0.2, buy it with detection
data or a wider detection head — not by starving the segmentation heads.

### Report rare classes on the test split only

The val split selects `best.pt`, and that selection is not free: each candidate gets cut at
a high point of its own val noise, so val comparisons of a rare class measure the selection
as much as the model. Same three indoor checkpoints, same classes:

```
              val (285 images)              test (329 images)
caution       0.1241 / 0.0844 / 0.0057      0.3252 / 0.3281 / 0.3346
              spread 0.118                  spread 0.009
```

On val those three models look wildly different. On test they are equivalent, which is the
truth. **`display_fixture` will be a low-frequency class for the whole first phase of
annotation, so its numbers belong on `test` and nowhere else.** Use val to select, test to
report, and never quote a rare-class val IoU in a decision — the indoor work reversed its
conclusions three times before this was understood.

A corollary worth stating: build the retail `test` split *before* the first training run,
not after the numbers look interesting.

---

## 7. What to do first

1. Train the ADE20K + COCO baseline from `hydranet_retail.yaml`. It is a day, it is not the
   product, and its only job is to give the annotation effort something to be measured
   against.
2. **Start annotating store footage in parallel.** It is the critical path.
   `hydranet-annotation labels` emits the CVAT schema; `hydranet-annotation check` gates the
   result before any of it reaches training.
3. Calibrate the homography and build the distance LUT. This is independent of the model
   and can be finished before the first store mask is drawn.
4. Narrow the detection export to the eight classes that matter, and re-measure on the
   Orin. Expect most of the 16.33 ms post-processing to go away.

## 8. Open questions for whoever owns the product

- Is the merchandise zone a *reporting* output (analytics: dwell time, coverage) or a
  *motion* output (the robot must not enter it)? The two want different accuracy and
  different failure behaviour, and the answer changes whether §3's BEV derivation is
  sufficient.
- Are shopping trolleys and baskets in scope? They are the most common dynamic obstacle in
  a shop and COCO has neither.
- Does the robot operate during trading hours? If so, `person` density is an order of
  magnitude above anything in the indoor footage, and occlusion — not classification —
  becomes the hard problem.
