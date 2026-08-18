# Retail + security on one store camera: the output contract, and the data behind each row

**Scope.** Fixed store CCTV, no LiDAR, the retail analytics programme of
[ARCHITECTURE_DIRECTION.md](ARCHITECTURE_DIRECTION.md). Not the robot platform;
[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md)'s verdicts stand and nothing here
overturns them.

The ask is two products on one camera: retail analytics (what merchandise, where do
people go, how long do they stay) and security (who entered where, how many, how long,
what did they do). This document says what the model outputs, what the *pipeline* outputs,
and which of the two each answer belongs to — because the most expensive mistake available
here is putting an answer in the weights that belongs in a config.

---

## 1. Three layers, and the line that matters is the first one

| layer | what it is | what it produces |
|---|---|---|
| **L0** | `HydraNet.forward` — a single-frame pure convolution graph, exported to ONNX | `terrain` logits `[B,6,H,W]`; `det_cls` / `det_reg` / `det_ctr` over 4 box classes |
| **L1** | the second stage, host-side ([`analytics/stage.py`](../src/syncai_hydranet/analytics/stage.py)) | `Track`s, and — when the pose model exists — 17 keypoints per person per frame |
| **L2** | events ([`analytics/events.py`](../src/syncai_hydranet/analytics/events.py)) | one row per event, with the value it measured and the threshold it crossed |

**Nothing crosses from L2 into L0.** `RETAIL_SCOPE.md` §2 refuses to make "within 5 m" a
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
`ARCHITECTURE_DIRECTION.md` §3 has the measurement: these configs drop the traversability
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
> against 24 overhead, down-pitched, h.264 store cameras. Its mAP is measured on COCO val
> and *nowhere else*, because the site annotation file has no person category to score
> against. A good number there is not evidence about a store.

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
this repository ([PERSON_ATTRIBUTES.md](PERSON_ATTRIBUTES.md)):

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
model on an input that does not exist here. `ARCHITECTURE_DIRECTION.md` §5's ordering —
metric, then association, then attributes, then action — is hard at this step.
`events.UNBUILT` refuses `fight` with that blocker named, rather than returning an empty
list that reads as "nothing happened".

### The fall proxy is a miner, not an alarm

Tier 1 carries `fall_candidate`: a box that went wide and short and stayed that way. As an
alarm it is close to useless in retail, and the false positives are not exotic — **bending
to a low shelf, crouching at bottom stock, and legs occluded by a fixture** are the three
most common things a shopper does, and `dwell.py` already records that fixtures are exactly
where shoppers stand.

Its job is the other one. Run it over the 3.2 GB of site clips already on disk, look at
what it returns, and that answers the question ARCHITECTURE_DIRECTION.md insists is
answered *before* any behaviour annotation is paid for: **do these events occur on these
cameras at all.** The type name carries the distinction, because a type name survives being
copied into a dashboard and a docstring does not.

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

This is the sixth instance of the pattern §1 of ARCHITECTURE_DIRECTION.md names: a number
was attributed to a mechanism without checking what produced it, and the thing that
produced this one was a metadata field that is allowed to be nominal. `probe` reports 30
for every clip in `datasets/studioa_clips`, so anything else in this repository that reads
its fps on this footage inherits the same error.

---

## 4. The data, in the order it unblocks things

| # | data | for | state |
|---|---|---|---|
| 1 | **site person boxes**, same 288 frames, same 24 cameras | `person` in domain | **not started — critical path** |
| 2 | **one labelled site clip** (person tracks) | `idf1` / `id_switches`, which do not exist today | **not started — blocks all of tier 3** |
| 3 | COCO `person`+`bag` | bootstrap for channels 0–1 | on disk, wired in `hydranet_retail_security.yaml` |
| 4 | COCO keypoints | tier-2 pose bootstrap | **on disk, nothing reads it yet** |
| 5 | CrowdHuman | occlusion and density, which is what fragments tracks | not downloaded; research licence |
| 6 | RAP v2 | attributes, indoor mall CCTV distribution | licence unconfirmed — one question, changes the plan more than anything else here |
| 7 | PA-100K / PETA / Market-1501 | attribute mass, benchmark, re-ID floor | on disk; `reid_metrics.py` already scores an untrained resnet18 at mAP 0.0318 |
| 8 | RWF-2000 / UCF-Crime | `fight` | **do not buy yet** — see tier 3 |
| 9 | URFD / Le2i fall sets | `fall` | **rejected as a training source**: staged, domestic, eye-level. The `column` failure in advance |

Three rules that apply to every row of that table and have each already been paid for once:

* **Hold out whole stores, not sessions.** A shop's aisles repeat and its fixtures are
  identical by design, so a val split inside one store measures memorisation of that store
  and looks excellent.
* **Build the test split before the first run**, not after the numbers look interesting.
* **Keep an out-of-domain set a minority by *step share*, which is not file count.** And
  read `MultiTaskLoader.detection_class_steps()` rather than inferring it from
  `sample_ratio` — the ratio is not the share, and the config that measured well was
  already two-thirds COCO by steps while its comment said 0.1.

---

## 5. How to read anything this produces

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

## 6. What not to build

* **No ROI, temporal, keypoint or attribute head in HydraNet.** A keypoint head needs the
  detection head's boxes and the `Head` protocol's `forward_into(out, feats, size)` has no
  way to express that; time is not in `forward(images)` at all. The second stage gets its
  own small ONNX export.
* **No fine-grained age.** 31–42 px of head. Coarse bands or nothing.
* **No behaviour annotation before `fall_candidate` has been run over the existing clips.**
* **No `fight` model before the association metric exists.**
