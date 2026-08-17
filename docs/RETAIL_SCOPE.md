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

**C. The depth camera, if the platform has one.** `10.8.140.130` carries an Intel RealSense
**D435I** publishing metric depth already registered to the colour frame, so the 5 m test
looks like `depth < 5.0` — no calibration, no extrinsics, no flat-floor assumption.

**It is not sufficient on its own, and the reason is exactly the floor retail has.** Run
live on that robot in a lobby with a polished stone floor:

> The figure that stood here was deleted in `f9d4fcf`. It is in history at `4eb8b56` as
> `assets/lobby_polished_floor_depth_holes.jpg`, and **nothing replaces it**: reproducing
> it needs that robot, that lobby and that RealSense, and no site camera in
> `datasets/studioa_clips` carries depth at all. The measurement below is the finding; the
> picture was its illustration.

It showed traversability beside the same frame split by depth: green within range, yellow
beyond it, **magenta walkable with no depth return at all**. The magenta is scattered right
across the near floor — 10.6% of all walkable pixels in that frame — because a polished
floor is specular and reflects the projector's IR pattern away from the sensor instead of
back to it. Depth-only range gating therefore punches holes in the walkable mask precisely
where the robot is about to drive.

Two consequences worth carrying into the design:

1. **Prefer the homography for the range gate and use depth to confirm obstacles.** A
   homography does not care whether a surface returns IR; it maps floor pixels to floor
   distances geometrically. On specular retail floors that makes A more robust than C, which
   inverts the usual ordering.
2. **"No depth return" is ambiguous, and geometry disambiguates it.** Glass at eye level
   returns nothing and is lethal; polished floor below the horizon returns nothing and is
   perfectly walkable. The same pixel value means opposite things depending on where it sits
   relative to the ground plane — which is one more thing the homography answers and a
   depth threshold alone cannot.

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
| display fixtures | **Partly, and the split is sharp — see below.** |

### What the bootstrap transfers, and what it cannot

The retail baseline was run and its pre-labels put over real store CCTV:

![the bootstrap splits one kind of fixture across two classes](../assets/retail_bootstrap_fixture_split.jpg)

`runs/hydranet_retail_base` — ADE20K and COCO only, no site data — on a held-out shop
camera, `Taichung-cam06`. Left is free space with detection boxes, right is the 13-class
terrain map. **Wall-mounted shelving comes back purple as `display_fixture`; the
free-standing podiums in the middle of the same floor come back red as
`obstacle_furniture`.** Measured on this frame: `wall` 30.7%, `obstacle_furniture` 29.2%,
`floor_hard` 23.9%, `display_fixture` 15.0%.

That is the split, and it is worth stating precisely because the older version of this
section got it wrong in the model's favour: it said the podiums came back *unlabelled*,
"the model saying it does not know rather than guessing". It does not say that. It answers
confidently, with a different class for the same kind of object, which is the worse
failure of the two — an unlabelled region is visible as a gap, and a wrong label is not.
It is also the exact frame `label_maps_retail_objects.py` was written from: "the round
podium is `obstacle_furniture` while the wall shelving three metres away is
`display_fixture`", and `display_fixture` carries the lowest test IoU of any class with
real data at 0.336.

Reproduce with:

    hydranet-infer-image --config runs/hydranet_retail_base/config.yaml \
      --checkpoint runs/hydranet_retail_base/best.pt --input <frame.jpg> \
      --output out/ --vocab retail

> The figure that stood here was deleted in `f9d4fcf` (history: `55ec787`,
> `assets/retail_prelabel_gap.jpg`). What replaces it is stronger than a picture, because
> the gap it showed has since been **counted** rather than pointed at. Per-class support on
> the ADE20K splits, mapped through `ade20k_retail_objects`: `product` appears in
> **0 of 285 val images and 0.00% of labelled pixels**, against `wall` at 61.86% and
> `floor` at 24.60%. A figure showing one unlabelled podium is an example; a class that is
> absent from every image of the source dataset is the whole claim.
>
> What was done about it is also now on disk rather than proposed:
> `datasets/retail_objects_batch02` — 288 frames over 24 shop-floor cameras, split by
> camera, 10,517 merchandise boxes — and `configs/hydranet_retail_products.yaml`, which
> stops asking a dense head what fraction of a shelf is product and gives the boxes to the
> instrument that can resolve them. Read `split.json`'s `test_provenance` before quoting a
> number off it: the human pass was never completed, so those numbers are agreement with
> SAM 3, not accuracy.

That split is not arbitrary. A wall of shelves looks like ADE20K's `bookcase` and `shelf`,
so the bootstrap reaches it. A waist-height island podium with three phones on it has no
analogue in any public dataset — it is a retail fixture and nothing else is shaped like
one. **No amount of public data will supply it**, which is what makes it the first thing
worth annotating rather than a nice-to-have.

It also sets the annotation priority within the class: the wall runs are already close
enough for a correction pass, the islands have to be drawn from scratch.

Note what `ade20k_retail` deliberately does *not* map: ADE20K's `table` (id 16) stays
`obstacle_furniture`, even though a display table is precisely what was asked for. Its
tables are domestic and dining, and mapping them would teach the model that a proxy which
is nearly the right thing is the right thing. This project has already paid for that
mistake once, with COCO diluting the segmentation heads. Both classes are `blocked`
regardless, so the cost of the omission is semantic, not a safety regression.

### Self-training on the site's own frames: what it bought, and what it did not

Measured 2026-08-14, `runs/hydranet_retail_cctv`. Twelve epochs from the retail baseline's
E22 checkpoint, adding 400 frames from two site clips labelled by that same baseline at
`sample_ratio: 3.0`, with the input raised to 512x896 for the 16:9 source.

**The fine-tune changes three things at once**, so a run with `site_cctv_pseudo` removed and
everything else identical was done alongside it (`runs/hydranet_retail_cctv_noself`). Without
that control none of what follows can be attributed, because a fresh low-LR cosine restart
buys gains on its own. All three checkpoints on the 329 held-out ADE20K test images under
`configs/hydranet_retail.yaml`, nothing differing but the weights. The two runs were trained
several commits apart, so all three were re-scored in one pass under one code state before
the differences below were believed — they reproduced to the last digit.

| | baseline | restart only | restart + site | the restart | the site data |
|---|---|---|---|---|---|
| traversability mIoU | 0.6685 | 0.6785 | 0.6766 | **+0.0100** | −0.0019 |
| terrain mIoU | 0.5410 | 0.5460 | 0.5453 | **+0.0050** | −0.0007 |
| glass IoU | 0.5461 | 0.5574 | 0.5587 | **+0.0114** | +0.0013 |
| caution IoU | 0.2217 | 0.2464 | 0.2393 | **+0.0248** | −0.0072 |
| stairs IoU | 0.2080 | 0.2605 | 0.2446 | **+0.0524** | −0.0159 |
| person IoU | 0.6503 | 0.6777 | 0.6804 | **+0.0274** | +0.0027 |
| door IoU | 0.2985 | 0.2760 | 0.2745 | −0.0226 | −0.0015 |
| **display_fixture IoU** | **0.3668** | **0.3475** | **0.3379** | −0.0193 | **−0.0096** |

**On the source domain the pseudo-labels do nothing.** Every gain in the middle column is
the low-LR restart; the site data's own contribution is a wash and slightly negative on most
classes. There is no source-domain cost either — the expected trade never appeared — but do
not read the rise as evidence that self-training works. Without the control this run would
have been written up as "adaptation improved ADE20K too", which is false.

Two other explanations were checked and eliminated: it is **not the evaluation resolution**
(the fine-tune scores 0.6725 / 0.5459 / 0.5585 at its native 512x896, the same picture), and
it is **not simply more epochs** (the baseline peaked at E22 and then ran 38 further epochs
on the same ADE20K without beating it, ending at val trav 0.6255).

**`display_fixture` is where the site data does its own damage.** It is the only class the
pseudo-labels move meaningfully on their own (−0.0096, on top of the restart's −0.0193), and
the mechanism is the one this file predicted: the podiums come back unlabelled, so the
pseudo-labels teach "fixture-shaped things in a shop are not `display_fixture`". Note where
that lands — on ADE20K's *wall shelving*, the half of the class the bootstrap could already
reach. Self-training did not merely fail to add the podiums; it eroded the part that worked.

**Benefit side, which lives entirely on the target domain.** The third site clip
(`archive_20260802-125220`) is not in the training data. It is not perfectly clean either:
eight of its frames are the run's val split, so it did influence which epoch became
`best.pt` — a weak channel against ADE20K's val set, but not zero.

The baseline paints `caution` on a **fixed structural column**. On a camera that
never moves, a static false hazard is in every frame the site will ever produce — it does not
average out, and it is the failure that matters most here. A column is never `caution`, so
that region's answer is known in advance and the false positives can simply be
counted over all 1830 frames rather than eyeballed (`scripts/count_false_caution.py`):

| | frames with the column marked `caution` | mean area of the box |
|---|---|---|
| baseline | 1782 / 1830 (97.4%) | 21.3% |
| restart only | 909 / 1830 (49.7%) | 9.8% |
| restart + site | **117 / 1830 (6.4%)** | **0.6%** |

> A side-by-side figure and a close-up of the column stood above this table until
> 2026-08-17, when they were deleted along with the three `assets/cmp_clip3_*.mp4` renders
> they were cut from. **The table is the record, and always was** — the figure was four
> frames chosen by eye, and the paragraph below is about how badly that sampling misled.
> `scripts/count_false_caution.py` still names the three renders it reads; they were
> gitignored from the start, so it has never been runnable by anyone but their author, and
> regenerating them from a checkpoint is what re-running it requires.

**This is what the site data bought, and only the target domain can see it.** The restart
halves the false positive; the pseudo-labels remove most of what is left. Four frames
sampled by eye had suggested the opposite — that the restart did the work — because the four
happened to be frames where the control was clean. The full count is the whole result.

Also observed, by eye and not counted: recovered floor is more contiguous under the
fine-tune, and **the free-standing podiums are still unmarked by all three models** —
unchanged, as predicted.

The honest summary: **judge domain adaptation on the target domain or not at all.** The
source-domain split, which is the only place with labels, scores the pseudo-labels at
approximately zero and would have got the decision wrong. What adaptation cannot do is
create the missing class — it made that one slightly worse. Annotation is still the only
thing that supplies the podiums.

### A fixed camera makes the floor one polygon, not a labelling campaign

Measured on `archive_20260802-125220`, 610 frames through
`runs/hydranet_retail_cctv/best.pt`. Because the camera never moves, every pixel can be
asked the same question 610 times, and the answers split three ways:

| how often the pixel came back `go` | share of frame |
|---|---|
| ≥ 98% — settled floor | **2.2%** |
| 80–98% | 17.2% |
| **20–80% — the model cannot decide** | **16.7%** |
| ≤ 2% — settled not-floor | 59.6% |

The unstable 16.7% is not scattered. It sits in three places: the **brighter, more
specular near-field floor** — the same tiles as the stable region, differing only in how
much light they throw back; the **bases of the podium and the shelf runs**; and the paths
people walk, which is the only honest part of it. The first is the failure this file
already predicts for polished retail floors, arriving without any depth sensor involved.

**None of the training options touch it.** Another self-training round draws its labels
from the model that is confused there, so it hardens the confusion — `hydranet_retail_cctv.yaml`
says as much. COCO-Stuff is web photography of homes and public interiors and knows
nothing about this shop's lighting. More epochs only make a contradictory signal more
confident.

What does touch it is one annotator and one image. **On a fixed camera the floor is a
single static region**, so the polygon drawn once is correct for every frame that camera
will ever produce — 1,830 in this clip alone. Thresholding the frame-consensus at 90/5%
leaves 12.7% of the frame confidently floor, 61.2% confidently not, and **26.1% needing a
human**, which is a ten-minute correction rather than a labelling campaign. Per camera,
not per frame: N cameras cost N polygons.

That is also the right order of work. Real labels for the floor first, then a fine-tune on
them; a fine-tune before them is training on the model's own uncertainty.

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
4. ~~Narrow the detection export to the eight classes that matter, and re-measure on the
   Orin. Expect most of the 16.33 ms post-processing to go away.~~ **Done 2026-08-17, and
   two of the three predictions in that sentence were wrong.** See below.

### What the export narrowing actually bought

`hydranet-export-onnx --detection-classes` landed and was measured end to end on a GB10
(TRT 10.16), not the AGX Orin — so the numbers below are a different board and are not
comparable to the 37.8 ms frame quoted above. Treat the ratios as the result.

**The prediction that held.** Narrowing collapses the host sigmoid exactly as argued:
545,600 → 218,240 values a frame at `retail_analytics`, and the detect stage went 1.73 →
0.64 ms, a measured 2.7× against a predicted 2.5×.

**The first thing that was wrong: eight is the wrong list for this deployment.** §4 names
eight classes that change a shop *robot's* behaviour. For retail *analytics* that list
deletes `book`, which is the single strongest merchandise signal the head produces — 1,683
instances on Kaohsiung-cam08, and the whole subject of
[RETAIL_OBJECTS.md](RETAIL_OBJECTS.md)'s audit. So there are two subsets, `robot_8` and
`retail_analytics` (32 classes), and neither is a default. The eight-class figure is real
and it is for the robot.

**The second thing that was wrong: the post-processing was not mostly the sigmoid.** On a
6.69 ms frame the detect stage was 1.70 ms and the **host argmax over the segmentation
logits was 2.53 ms** — larger than the engine's 2.09 ms, and on nobody's list here or in
DEPLOY_JETSON.md §4, which offers four ways to shrink the engine. `--argmax-seg` folds it
into the graph: 6.69 → 4.00 ms, and 3.29 ms with both flags.

The reusable part is not the millisecond count. **This document reasoned carefully about
the largest item in a frame it had measured, and named the second largest** — because the
16.33 ms figure was a single bucket labelled "post-processing" and nobody had split it. A
number that is 43% of the frame and undivided is not a finding, it is a place to look.

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
