# Retail and security on one store camera

**This is the line the project leads with**: fixed store CCTV, no LiDAR, one model
answering two questions. Retail analytics asks *what merchandise, where do people go, how
long do they stay*; security asks *who entered where, how many, how long, what did they
do*. They are not two products — they are two readings of one camera, one export and one
latency budget.

One file, merged 2026-08-19 from `RETAIL_SCOPE.md`, `RETAIL_OBJECTS.md` and
`RETAIL.md`, because the three had grown a shared boundary that every reader had
to reconstruct. Its companion is [RETAIL_DATA.md](RETAIL_DATA.md) — the splits, the
teachers, the datasets and what may be believed off them. The architecture argument behind
both is [ARCHITECTURE.md](ARCHITECTURE.md) Part II.

**The most expensive mistake available here is putting an answer in the weights that
belongs in a config**, and most of what follows is a way of telling those apart.

---

## 1. The output contract: three layers, and the line that matters is the first

| layer | what it is | what it produces |
|---|---|---|
| **L0** | `HydraNet.forward` — a single-frame pure convolution graph, exported to ONNX | `terrain` logits `[B,6,H,W]`; `det_cls` / `det_reg` / `det_ctr` over 4 box classes |
| **L1** | the second stage, host-side ([`analytics/stage.py`](../src/syncai_hydranet/analytics/stage.py)) | `Track`s, and — when the pose model exists — 17 keypoints per person per frame |
| **L2** | events ([`analytics/events.py`](../src/syncai_hydranet/analytics/events.py)) | one row per event, with the value it measured and the threshold it crossed |

**Nothing crosses from L2 into L0.** `RETAIL.md` §2 refuses to make "within 5 m" a
class and gives three reasons; all three transfer to every security event without
modification. Annotators cannot draw a zone boundary consistently, because it has no
appearance. A zone in pixels dies the first time the mount is knocked, and nothing in
training reports it. And "four minutes is loitering", "six people is too many" are
settings a store manager changes on a Tuesday — a config value, or a retraining run.

So every zone in L2 is a polygon **in metres on the floor**, every threshold is a function
argument, and the model contributes boxes and a terrain map.

### L0's outputs, exactly

```
terrain     [B, 6, H, W]   void, floor, wall, column, fixture, person
detection   FCOS triple    person, bag, boxed_stock, device
```

`person` is in **both**, and that is not redundancy to optimise away.
`ARCHITECTURE.md` §3 has the measurement: these configs drop the traversability
head and *derive* free space from terrain, so removing `person` from segmentation leaves a
person-shaped hole of walkable floor — drawn in metres, with no error anywhere. The
detection channel feeds the second stage; the segmentation channel feeds the geometry.
Neither is derivable from the other.

---

## 2. One detection head for both questions

`hydranet_retail_products.yaml` refused to train COCO alongside the site boxes, and the
reason was exact: `CocoDetDataset` numbers each annotation file's categories from zero, so
COCO's `person` and the site's `boxed_stock` are both label 0 and one head cannot hold
both meanings. Every security event is keyed on `person`, so that refusal is the blocker.

[`data/label_maps_retail_security.py`](../src/syncai_hydranet/data/label_maps_retail_security.py)
lifts it. A **vocabulary** assigns the ids once and every source maps its category *names*
into it:

| id | class | sourced by |
|---|---|---|
| 0 | `person` | COCO — 64,115 images, 257,253 boxes |
| 1 | `bag` | COCO `backpack` + `handbag` + `suitcase` — 11,989 images, 27,168 boxes |
| 2 | `boxed_stock` | site `retail_objects_batch02` |
| 3 | `device` | site `retail_objects_batch02` |

Three COCO categories behind one `bag` channel because nothing downstream asks which of
the three it was, and three thin channels learn worse than one thick one. The ground truth
is merged in memory the same way the labels are, so mAP is scored over the merged class
rather than counting every `handbag` as an unmatchable category.

**`trolley`, `basket` and `staff` are deliberately absent.** All three were asked for and
none has a source: COCO has neither trolley nor basket, and `staff` is not a box class at
all — a uniform is an attribute of a person, so it belongs on the second stage's crop
encoder. An unsourced channel is trained on nothing and reports 0.000 while the run looks
entirely normal, and this project has paid for that four times over.

### The half a shared vocabulary does not fix

A site frame is full of shoppers and carries **no person boxes**. Without help, every
point on every shopper is a labelled *negative* for channel 0 — which is not dilution but
suppression, and it is the mechanism that held `product` at IoU 0.000 for 22 consecutive
epochs (`config_schema.minority_sourced_terrain_classes`).

So `FCOSLoss` takes a **class mask** and each dataset supplies its own: a COCO batch trains
channels 0–1 and leaves 2–3 untouched, a site batch does the reverse. Untouched means *no
gradient*, not a small one. `MultiTaskLoader.detection_class_steps()` prints the realised
share at startup, counted from the schedule rather than estimated from `sample_ratio`.

What that buys: a minority source is now **slow rather than suppressed**, so the COCO ratio
can be set on throughput grounds instead of defensively.

What it does not buy, and this is the sentence to carry out of this section:

> **`person` is trained on COCO alone** — web photography, eye level, uncompressed —
> against 23 overhead, down-pitched, h.264 selling-floor cameras. Its mAP is measured on
> COCO val and *nowhere else*, because the site annotation file has no person category to
> score against. A good number there is not evidence about a store.

**The teacher for this class was swapped on 2026-08-19** — Grounding DINO at a
measured threshold gap, replacing SAM 3, which turned an unusable night into a usable one.
The measurement, the fleet batch and the caveats are in
[RETAIL_DATA.md](RETAIL_DATA.md) §6.

`column` is the precedent and it is not a mild one: sourced from ADE20K, passing every
config gate, scoring 0.40–0.51 on val, and predicting **0.00% of pixels** across four
daytime shop-floor cameras. Site person boxes are the critical path, not a nice-to-have.

---

## 3. Behaviour, split by the instrument that can resolve it


Behaviour is the half of "security" that usually arrives attached to an annotation budget.
A good share of it needs no annotation at all, and the part that does is blocked on
something other than data. The split is by instrument, which is the same judgement that
moved `product` out of the dense head — ask what can answer the question before asking what
to call it. `events.TIERS` states it per type.

### Tier 1 — track geometry. Nothing new required.

`running` (m/s over a window), `tailgating`, `crowd_forming`, `loitering`,
`zone_intrusion`, `line_cross`, `occupancy_exceeded`, `object_left`, `stock_removed`.

All computed from floor positions in **metres per second**, not pixels per frame, which is
why a threshold set on one camera transfers to the next. A pixel speed tuned on
Kaohsiung-cam08 is a statement about that lens and that mount.

### Tier 2 — pose. One second-stage model; the data is already on disk.

Three measurements decide that this is the right instrument, and all three are already in
this repository ([RETAIL_DATA.md](RETAIL_DATA.md)):

| measured | value | consequence |
|---|---|---|
| person box height | 244–336 px median | a crop encoder is comfortable |
| head region | 31–42 px | anything routed through a face reports on noise |
| **track length** | **median 9–16 frames at 5 fps** | **a clip-level action model never receives a whole person** |

The third one decides it. A clip-level action classifier wants 16–64 consecutive frames of
one identity; the median track does not contain one. **Pose is per-frame, so fragmentation
shortens it instead of invalidating it** — nine frames of a track is nine poses, and a fall
is a change of torso angle over one or two seconds.

It is also the only behaviour direction that needs no dataset purchase to start:
`datasets/coco/annotations/person_keypoints_train2017.json` is on disk.

From keypoints: `fall`, `crouch`, and `reach_to_shelf`. That last one is worth naming
separately — **it is the one output where the retail model and the security model are the
same model.** It needs the segmentation head's `fixture` region *and* the second stage's
wrist keypoint, and neither instrument can produce it alone. "Shared trunk" is usually a
claim about parameters; this is a claim about an answer.

Two design details that were forced by measurement rather than chosen:

* Both quantities are computed **inside one frame**. `crouch` compares hip-to-ankle extent
  against the torso length of the same person in the same frame, not against that track's
  own history. A self-baseline over a 9–16 frame fragment compares a crouch against a
  crouch and reports nothing — the case that matters, not an edge case. (The unit test
  found this; the first version used a per-track median and detected neither posture.)
* `fall` uses shoulder-to-hip only. Ankles are the first thing a display table hides, so
  the safety-relevant type is the one that survives occlusion.

### Tier 3 — a temporal model over a whole track. `fight`, and it is not blocked on data.

RWF-2000 and UCF-Crime are buyable. The blocker is **association**: at 9–16 frames the
pipeline cannot deliver the input such a model needs, so buying the data first trains a
model on an input that does not exist here. `ARCHITECTURE.md` §5's ordering —
metric, then association, then attributes, then action — is hard at this step.
`events.UNBUILT` refuses `fight` with that blocker named, rather than returning an empty
list that reads as "nothing happened".

### The fall proxy is a miner, not an alarm

Tier 1 carries `fall_candidate`: a box that went wide and short and stayed that way. As an
alarm it is close to useless in retail, and the false positives are not exotic — **bending
to a low shelf, crouching at bottom stock, and legs occluded by a fixture** are the three
most common things a shopper does, and `dwell.py` already records that fixtures are exactly
where shoppers stand.

Its job is the other one. Run it over the site clips already on disk, look at what it
returns, and that answers the question ARCHITECTURE.md insists is answered
*before* any behaviour annotation is paid for: **do these events occur on these cameras at
all.** The type name carries the distinction, because a type name survives being copied into
a dashboard and a docstring does not.

**Answered 2026-08-18, and the answer is no.** `runs/fall_mining01`, over 192 clips / 48
cameras / 188,523 person boxes, returned **48 `fall_candidate` spans**. 23 of them touch the
frame edge and the rest sit on near-nadir or rotated cameras. **None is a posture.** On this
corpus the proxy measures camera mounting, so the behaviour-annotation spend this miner was
built to justify is not justified — which is exactly what it was built to be able to say.

### The frame rate, which was assumed and is now counted

An earlier version of this section said the clips are **1920×1080 at 30 fps** and proposed
re-running tracking at 5 / 10 / 15 / 30 fps to buy back temporal resolution. The 30 came
from `cli/infer_video.probe`, which reads `r_frame_rate` — and `r_frame_rate` is the
container's nominal rate.

Counted with `ffprobe -count_frames`: Kaohsiung-cam04 holds **2,400 frames over 300.1 s,
8.0 fps**, and Taichung-cam01 **2,130 over 304.3 s, 7.0 fps**. `avg_frame_rate` matches the
counted value to three decimals on both and costs no decode, which is what
`scripts/mine_fall_candidates.source_fps` now reads.

**The proposal it invalidates is its own.** There is no 30 fps of content to recover, so
the sweep tops out at 7–8 and the headroom over the current 5 is about 1.6×, not 6×. That
changes a conclusion rather than a detail: **fragmentation cannot be fixed by sampling
faster**, which leaves appearance-based association as the only instrument for it instead
of one of two. It also lowers the ceiling on tier 3 — a temporal action model over these
clips gets 7–8 samples a second no matter what is spent on it.

This is the sixth instance of the pattern §1 of ARCHITECTURE.md names: a number
was attributed to a mechanism without checking what produced it, and the thing that
produced this one was a metadata field that is allowed to be nominal. `probe` reports 30
for every clip in `datasets/studioa_clips`, so anything else in this repository that reads
its fps on this footage inherits the same error.

---

## 4. How to read anything this produces


* **A detection comparison runs three seeds or it reports nothing.** Measured spread on
  this family with nothing changed but the seed: `detection_mAP` 0.0655 / 0.0821 / 0.0648,
  range 0.0173. A single-run comparison of two `sample_ratio` values decides nothing.
* **Val selects, test reports.** `best.pt` is chosen on `detection_mAP/site_boxes` over 36
  images; rare-class numbers go on test, and `split.json`'s `test_provenance` says the
  human pass was never completed — those are agreement with SAM 3, not accuracy.
* **Every event row carries a `basis`** naming the measurement that produced it, and no
  score. These are threshold crossings, so the row carries `value` and `threshold`; a
  confidence would be invented, and this repository has a standing rule against numbers
  whose production nobody can name.
* **Every event inherits the fragmentation.** A 4.6-minute clip becomes 1234 tracks.
  Occupancy counts tracks, loitering needs one to survive, tailgating needs two to be
  distinct. Read an occupancy alarm as an upper bound until association is fixed and
  measured.
* **The `settings` block names what decided the numbers, and not where anything lives.**
  Every row is a threshold crossing, so the argument list is provenance and belongs in the
  report — but it used to be `vars(args)` verbatim, which meant each report also carried
  `/home/paul/SyncAI-Lib-HydraNet/datasets/studioa_clips/…`: an operator, a home directory,
  a repository checkout and a dataset root, in a file that looks like output rather than
  like a document. `analytics/delivery.report_settings` keeps the last two path components
  — `hydranet_retail_security_b03/best.pt`, `Taichung-cam01/archive_…mp4` — which is enough
  to identify and not enough to locate. A basename alone was tried first and is worse than
  the leak: `best.pt` names nothing, there are forty of them under `runs/`, and a report
  that cannot say which weights produced it is not auditable.
* **Every event row carries a wall clock, and it carries its offset.** `started_at` and
  `ended_at` are ISO-8601 with a UTC offset, derived from the clip's `archive_<UTC>_` name
  and the store's zone. Frames remain the unit of record for every computation — a dropped
  frame changes the index and not the clock, which is why `stage.StageFrame` insists on the
  index — but the conversion happens at the edge, because `frame_start: 5312` does not
  locate anything in a DVR. This was missing until 2026-08-19: the rows were correct and
  undeliverable.

  **The clip filenames are UTC and the stores are UTC+8.** `scripts/site_events.py` takes
  `--utc-offset` (default 8) and writes every time in the store's zone; the offset is in
  the string so nobody has to know which convention was used. `scripts/pull_studioa.py`
  learnt this the expensive way, having asked for "16:00, the busy hour" and received
  greyscale IR of a shop with the shutters down. A clip whose name carries no time gets
  `null` rather than an invented start — the events are still correct in frames, and a
  guessed timestamp would look exactly as authoritative as a real one.

---

## 5. Geometry is not a class, and neither is a zone

Two of the three things originally asked for are geometry wearing a perception costume,
and telling them apart decides whether the annotation effort is reusable or thrown away
when the camera moves 10 cm on its mount.


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

[ARCHITECTURE.md](ARCHITECTURE.md) already reached the same conclusion from
the other direction, for a depth head: *do not build it, LiDAR measures it better than a
monocular head ever will.* **That verdict has since been overturned** — the robot has no
LiDAR, `models/heads/depth.py` was built, and a depth head is head ⑤ of the occupancy
direction ([RESEARCH_OCCUPANCY.md](RESEARCH_OCCUPANCY.md)). The conclusion *this* section
reaches does not depend on it: "within 5 m" stays out of the weights because it is a
product parameter and an annotator cannot draw it, which is true whatever measures the
distance.

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

**B. LiDAR mask.** ~~The robot already carries LiDAR~~ — **it does not** (corrected
2026-08-19; the Lite3 is monocular plus two ultrasound returns, and a LiDAR variant is one
of the sensors E-prep would justify *buying*). Kept as an option because the reasoning holds
the day one is fitted: it needs no flat-floor assumption and is the more accurate of the
two, at the cost of a LiDAR↔camera extrinsic. **Today it is not available**, which makes A
the only calibrated route on the robot and leaves the retail line — a fixed camera on a flat
shop floor — squarely in A's best case anyway.

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


### The merchandise *zone* is not a class either

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

## 6. The taxonomies, and why there are two of them


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

### The audit that produced the second taxonomy

18 fixed-camera site clips across three stores, 1,620 frames sampled at 3 fps, scored
with the 60-epoch checkpoint `runs/hydranet_retail/best.pt`. Raw output was in
`runs/review_20260816/`, **which is no longer on disk** (checked 2026-08-19) — the tables
below are the surviving record, and re-deriving them means re-running the sweep.

The camera does not move, so a pixel that changes class between frames is one the model
was unsure about — the same error signal the SAM 3 consensus pass uses, turned on the
model itself. `agree` below is the share of frames on which a pixel keeps its modal
class.

| class | share of frame | agree | test IoU |
|---|---|---|---|
| `wall` | 39.8% | 93.7% | 0.835 |
| `floor_hard` | 26.2% | 93.5% | 0.826 |
| `obstacle_furniture` | 17.6% | 86.8% | 0.707 |
| `display_fixture` | 10.7% | 79.3% | **0.336** |
| `door` | 3.5% | 71.0% | 0.380 |
| `person` | 1.4% | 76.3% | 0.760 |
| `glass` | 0.6% | 62.8% | 0.493 |
| `stairs` | 0.3% | 55.6% | 0.108 |
| `floor_metal`, `wet_slippery`, `threshold_ramp` | 0.00% | — | 0.000 |

Per-camera terrain stability runs from 92.9% (Kaohsiung-cam12) to 61.6%
(Tao-Hsin-cam05); the four worst are the cameras with polished near-field tiles or a
large glazed frontage.

#### Three findings

**1. Fixtures are split across two classes.** In a single Kaohsiung-cam08 frame the
round MacBook podium is `obstacle_furniture` and the wall shelving three metres away is
`display_fixture`. Both are shop fixtures. The cause is in
[`label_maps_retail.py`](../src/syncai_hydranet/data/label_maps_retail.py): ADE20K's
`table` (16) is deliberately withheld from `display_fixture` so that a dining table does
not teach the model it is shop furniture. Under traversability that is correct and the
cost is only semantic — both classes are `blocked`. Under object segmentation the
semantics *are* the output, and the cost is `display_fixture`'s 0.336.

**2. Columns are inside `wall`, by construction.** `ADE20K_ID_TO_INDOOR` maps `column`
(43) to `wall`, and the class comment has read "wall / ceiling / column" since the first
commit. `wall` is 39.8% of every frame, so the largest and most stable class in the model
is also the one hiding a structure the store cares about. RETAIL.md §5 already
tracked a column through 97.4% of one camera's frames — as a false `caution`.

**3. Merchandise has no class, and the detection head is already finding it.** Sweeping
the score threshold over 40 frames each:

| threshold | Taichung-cam01 | Kaohsiung-cam08 |
|---|---|---|
| 0.05 | 100.0 boxes/frame — `book` 914, `bottle` 783, `refrigerator` 476 | 84.7 — `book` 1683, `refrigerator` 384 |
| 0.15 | 11.8 — `person` 112, `chair` 108 | 10.0 — `book` 311, `mouse` 19 |
| 0.25 | 3.4 — `person` 72 | **0.0** |
| 0.35 | 1.6 — `person` 62 | **0.0** |

Kaohsiung-cam08 is an Apple store. There is no `laptop` in that column at any threshold
and there are 1,683 `book`: the head is finding the merchandise and naming it with the
nearest shape word COCO owns. **The localisation already works; the vocabulary and the
calibration do not.** At the shipped viewing threshold that camera returns no detections
at all.

---

### The object taxonomy

[`label_maps_retail_objects.py`](../src/syncai_hydranet/data/label_maps_retail_objects.py)

| id | class | notes |
|---|---|---|
| 0 | `void` | ignore; never annotated |
| 1 | `floor` | hard and soft merged |
| 2 | `wall` | absorbs ceiling, glazing and doors |
| 3 | `column` | **split out of wall** |
| 4 | `fixture` | **`obstacle_furniture` + `display_fixture` merged** |
| 5 | `product` | **new; no public source** |
| 6 | `person` | |

#### Why a second scheme rather than two more ids

`tests/test_retail_scheme.py` pins retail ids 0–11 to `INDOOR_TERRAIN` so a retail run
warm-starts from an indoor checkpoint and indoor masks stay readable. This taxonomy has
to merge two of those ids and split a third, neither of which is expressible under that
invariant — extending would mean deleting the tests that exist to stop the extension.
The robot's taxonomy is untouched.

The reversal on ADE20K's `table` is the clearest illustration:
`tests/test_retail_scheme.py` asserts it is **not** a display fixture and
`tests/test_retail_objects_scheme.py` asserts it **is** a fixture. Both are right, which
is the argument for two taxonomies.

#### The traversability head is gone

On segmentation data the traversability target is `terrain` run through a lookup table —
no new information. The 60-epoch run measured the consequence: `head_disagreement`
0.0091, the two heads agreeing on 99.1% of pixels because one is a deterministic function
of the other. Under the traversability goal that redundancy still bought a head that was
exactly the deployment question. Here it buys nothing.

`model.heads.traversability: null` in a config removes an inherited head. The deletion
happens once, before validation, so the schema, `unsupervised_heads`, the exporter and
`meta.json` all agree it does not exist.

---

### Migrating the masks already drawn

Everything under `datasets/retail_sam3_consensus*` is annotated as the retail 13. The
`retail_objects_migrated` label map reads it under the new taxonomy with no re-export.
Verified on all 216 consensus frames — **those directories, and `datasets/retail_cctv_pilot*`
cited further down, are no longer on disk** (checked 2026-08-19), so the table below is the
record and the check is not re-runnable as written. The label map it verifies is, and
`tests/test_retail_objects_scheme.py` still pins it:

```
retail_native              retail_objects_migrated
  floor_hard    42.28%       floor      42.71%
  wall          26.43%       wall       31.34%   = wall + glass + door, pixel-exact
  door           4.60%
  obstacle_f.    1.14%       fixture    20.89%   = obstacle_furniture + display_fixture,
  display_fix.  19.55%                            pixel-exact (5,558,232 px)
  person         5.01%       person      5.06%
  stairs         1.00%       (-> ignore)
  ignore        62.04%       ignore     62.41%
                             column      0 px
                             product     0 px
```

**`column` does not survive the migration and cannot.** Those masks put columns inside
`wall`; there is no signal left to separate them. `get_scheme("retail_objects_migrated")`
warns, `columns_from_migration_only()` reports it to the config validator, and
`hydranet-annotation check` shows it as 0.00% flagged `<- priority`. Three places,
because the failure it prevents — a permanently empty output channel reporting IoU 0.000
after sixty epochs with no error anywhere — is the one this project has already shipped
three times (`floor_metal`, `wet_slippery`, `threshold_ramp`).

> **Correction, 2026-08-17: the predicted failure mode is the milder member of its family.**
> The 60-epoch run happened, and `product` did not report IoU 0.000. It reported **nothing**
> — no `IoU/terrain/05_product` key at any epoch, and `terrain_mIoU_classes` 5.0 for all 60
> rows, because a class absent from the ground truth never enters the confusion matrix. A
> 0.000 in the table is visible and looks wrong; an absent row looks like a taxonomy with
> five classes and reads as normal. The three warnings above were right to exist and none of
> them fired at the moment that mattered, because none of them ran: `unsourced_classes()`
> had no caller outside its own tests until `config_schema.py` gained one the same day.
> Per-class support now travels with per-class IoU in `metrics.jsonl` for the same reason.

---


### What is still missing, in order

**1. Columns — sourced 2026-08-17, still unlearned.** ADE20K supplies id 43, which is atrium
and lobby columns rather than a shop's, and the 60-epoch run showed exactly what that buys:
`column` scores val IoU 0.40–0.51 and predicts **0.00% on every daytime site clip measured**.
Taichung-cam11 has a clad pillar with STUDIO A on it, centre of frame, and the model calls it
`wall`. The gap was never hidden — `configs/hydranet_retail_objects.yaml:84-86` says it in a
comment written before the run — and nothing executed that comment.

Two things the val number could not show, both from the site pre-labels. SAM 3 finds those
columns at 7.14% of labelled pixels across 24 shop-floor cameras, so the footage carries the
signal and the model does not. And `column`'s entire val IoU stands on **22 images and 0.66%
of labelled pixels**, printed in the same format as `wall`'s 283 images and 61.86%.

**2. Products — measured 2026-08-17, and the prompts work.** No public segmentation dataset
labels merchandise; site pre-labels are the only source, and their coverage figure was the
experiment. Result, from `datasets/retail_objects_batch01`: `product` is **19.28%** of
labelled pixels in train and **17.23%** in test, present in every frame, plus 10,524
merchandise instance boxes.

That result overturns a figure worth naming, because it would otherwise have talked someone
out of the material that works. The earlier `product box` measurement of **0 instances** on
accessory walls was a property of 352×240 footage and one prompt out of eight, not of the
footage: at 1920×1080 with `--upscale 1.0`, `packaged goods` and `merchandise on a shelf`
segment each hanging packet individually, 33% of labelled pixels on Kaohsiung-cam07.

```bash
hydranet-annotation labels --scheme retail_objects --out cvat_objects.json
python3 scripts/sam3_prelabel.py --scheme retail_objects --boxes \
    --out datasets/retail_objects_batch01 --frames 6 --upscale 1.0 /path/to/cam*/clip.mp4
hydranet-annotation check datasets/retail_objects_batch01 --scheme retail_objects
```

`--consensus 0.9` is deliberately absent from that command. It buys precision by writing 255
wherever the clip's frames disagree, which is the right trade when nobody will correct the
masks — but the pixels it discards are the class boundaries, which is what an IoU compares.
Measured on the six site test cameras, only **31.1%** of static pixels are stable at 0.9,
and of the unstable remainder `fixture` is 71.9%, `product` 17.4%, `column` 10.6%. A
consensus pass here would have produced a high-precision dataset that had thrown away the
argument.

**Which cameras, and which are held back for measurement, is
[RETAIL_DATA.md](RETAIL_DATA.md).** Split by camera and never by frame,
six cameras reserved, and a rule that a per-class number standing on fewer than two test
cameras is not reported at all.

The consensus pass removes only *random* error — a fixture SAM 3 consistently misses
survives untouched — and it cannot produce a test split, because scoring a model against
SAM 3's masks measures agreement with SAM 3.

**3. A hand-drawn test split. Still the critical path, and still absent — now designed and
reserved, which is not the same thing.** Six cameras are held out under
[RETAIL_DATA.md](RETAIL_DATA.md) and their 72 frames sit in the `test`
split, but **every mask in that split is still SAM 3 output with no human pass**, exactly as
in `datasets/retail_cctv_pilot*` and `datasets/retail_sam3_consensus*`. Scoring against them
would measure agreement with SAM 3, and since the training masks come from SAM 3 too, the
systematic errors are shared and cancel. Until someone corrects those 72 frames, this
taxonomy has **no site number at all** — not a low one.

What the reservation does buy, before anyone draws anything: the cameras are decided in
advance, from measured class share rather than from a contact sheet, so the split cannot be
chosen after seeing which cameras a model happens to do well on. RETAIL.md §6 argues
why this has to come first; nothing here changes that.

**4. Detection calibration.** `SCORE_THR_RETAIL = 0.20` restores boxes on the cameras
that return none at 0.30, and it is a stopgap, not a fix: scores are not comparable
between these cameras, so one global cut cannot be right for both. Per-class thresholds
fitted to hand-labelled site boxes are the fix, and they need item 3 first.

`--vocab retail` reads the existing head's existing output as shop nouns
(`product/book`, `fixture/refrigerator`). It **adds no knowledge and trains nothing** —
it is a rename, and it inherits every mistake the head makes.

---

### Running the object taxonomy

```bash
# train. Still the ADE20K bootstrap alone: datasets/retail_objects_batch01 exists but is
# not wired into this config, because its masks have had no human pass. Adding it means a
# second dataset entry with label_map: retail_objects_native -- at which point the
# unsourced-class warning for `product` should stop firing, and if it does not, the
# pre-labels are not reaching id 5 and that is worth knowing before sixty epochs.
hydranet-train --config configs/hydranet_retail_objects.yaml

# inference; there is no traversability panel, so the terrain overlay carries the boxes
hydranet-infer-video --config configs/hydranet_retail_objects.yaml \
    --checkpoint runs/hydranet_retail_objects/best.pt \
    --input clip.mp4 --output pred.mp4 --fps 6 \
    --vocab retail --score-thr 0.20
```

`hydranet-scene` refuses this config by design: every panel it draws starts from free
space, which is the head that was removed.

---

## 7. Objects: train on 80, narrow at export


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
[RETAIL.md](RETAIL.md)'s audit. So there are two subsets, `robot_8` and
`retail_analytics` (32 classes), and neither is a default. The eight-class figure is real
and it is for the robot.

**The second thing that was wrong: the post-processing was not mostly the sigmoid.** On a
6.69 ms frame the detect stage was 1.70 ms and the **host argmax over the segmentation
logits was 2.53 ms** — larger than the engine's 2.09 ms, and on nobody's list here or in
DEPLOY.md §4, which offers four ways to shrink the engine. `--argmax-seg` folds it
into the graph: 6.69 → 4.00 ms, and 3.29 ms with both flags.

The reusable part is not the millisecond count. **This document reasoned carefully about
the largest item in a frame it had measured, and named the second largest** — because the
16.33 ms figure was a single bucket labelled "post-processing" and nobody had split it. A
number that is 43% of the frame and undivided is not a finding, it is a place to look.

---

## 8. Traps, both already paid for on the indoor model


### Multi-task dilution: keep COCO at `sample_ratio: 0.1`

Swept at 60 epochs and scored on the held-out test split:

| | segmentation only | **COCO 0.1** | COCO 1.0 |
|---|---|---|---|
| `glass` IoU | 0.5021 | **0.5462** | 0.0692 |
| `caution` IoU | 0.3252 | **0.3346** | 0.1064 |
| `stairs` IoU | 0.3249 | 0.3174 | 0.0725 |
| terrain mIoU | 0.5718 | **0.5968** | 0.3591 |
| detection mAP | — | 0.1985 | **0.3348** |

**Which run is which column:** `hydranet_indoor` (no coco block), `hydranet_indoor_det60`
(0.1), `hydranet_joint_coco03` (0.3) and `hydranet_joint_coco10` (**1.0**). `coco10` means
share **1.0**, not 0.10, and `coco03` means 0.3 — the two names are not on the same scale,
and reading them as decimals swaps the strongest segmenter for the weakest. Take the ratio
from the run's `config.yaml`; [ARCHITECTURE.md](ARCHITECTURE.md) carries the
full table and the cross-checks.

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
## 9. What not to build

* **No ROI, temporal, keypoint or attribute head in HydraNet.** A keypoint head needs the
  detection head's boxes and the `Head` protocol's `forward_into(out, feats, size)` has no
  way to express that; time is not in `forward(images)` at all. The second stage gets its
  own small ONNX export.
* **No fine-grained age.** 31–42 px of head. Coarse bands or nothing.
* **No behaviour annotation before `fall_candidate` has been run over the existing clips.**
* **No `fight` model before the association metric exists.**


---

## 10. What to do first

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

## 11. Open questions for whoever owns the product

**Two of the three were answered on 2026-08-18, and the premise of the third moved.** This
document was written for a shop *robot*; the deployment that exists is fixed ceiling CCTV
([RETAIL.md](RETAIL.md)), so read "the robot" below as "the camera's
consumer" where it still makes sense.

- ~~Is the merchandise zone a *reporting* output or a *motion* output?~~ **Answered:
  reporting, as coverage rather than count**, derived by SAM 3 with no human labour
  (`runs/coverage01`: Tao-Hsin-cam02 35.0%, Taichung-cam04 30.9%, Kaohsiung-cam08 25.6%).
  §3's BEV derivation is therefore sufficient.
- ~~Are shopping trolleys and baskets in scope?~~ **Answered: not in the detection
  vocabulary**, and the reason is sourcing rather than scope — COCO has neither, so a
  channel for them would be trained on nothing and report 0.000 while the run looked
  normal. RETAIL.md §2 records the same verdict for `staff`.
- Does the robot operate during trading hours? **Still open, and now mostly a question about
  a different product.** The CCTV line runs during trading hours by construction, and the
  consequence this bullet predicted is measured: a 4.6-minute clip fragments into 1234
  tracks, so occlusion rather than classification is the hard problem
  ([RETAIL_DATA.md](RETAIL_DATA.md)).
