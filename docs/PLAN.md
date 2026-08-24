# Plan: the model, the data, and the order to build it

Supersedes `hydranet-decisions.md` and `DESIGN.md`. Written 2026-08-22 after the drift audit
in [journal/2026-08-22-drift-audit-and-course-correction.md](journal/2026-08-22-drift-audit-and-course-correction.md).
Read [VISION.md](VISION.md) first — it says what success is; this file says how.

Every number below is either **measured** (and says where) or marked **unmeasured**. Nothing
here is an estimate presented as a fact.

---

## 1. The placement rule

For any new capability, walk from the top and stop at the first rung that can answer it.
**Never start at the bottom.** This rule exists because every mistake of 2026-08-19/20 was
a jump to rung 6 or 7 for something rung 1 or 3 could answer.

| # | instrument | cost | needs |
|---|---|---|---|
| 1 | a **config value** — polygon, threshold, schedule | nothing | written once per camera |
| 2 | a **geometric derivation** in metres | nothing | the ground plane (§4) |
| 3 | a **rule over track time series**, CPU | nothing | tracks in metres |
| 4 | a **branch on the person-crop encoder**, ~1/45 of frame rate | small model, crop labels | existing detections |
| 5 | a **track-level temporal model**, CPU, <100K params | clip labels, 2–3 s each | tracks |
| 6 | a **new head on the shared trunk** | **+0.27 M params, +3% throughput** (measured) | dense or box labels |
| 7 | a **new class in an existing head** | invalidates every checkpoint's comparability | labels **and** a re-baseline |

**The test that catches a misplacement:** if the answer is a constant on a fixed camera, it
is rung 1. Floor, walls, shelf positions, glass, zone boundaries, opening hours, the till's
location — constants, every one. None of them is a class.

---

## 2. The five layers

```
L0  pixels  -> boxes + keypoints        shared trunk, every frame, GPU
L1  boxes   -> tracks in METRES         tracker + homography, every frame, CPU
L2  tracks  -> per-person facts         crop heads, ~every 3 s per track, GPU
L3  facts   -> events                  rules in metres and seconds, CPU
L4  events  -> judgement                VLM on trigger, GPU queue
```

Two invariants, both already enforced in the tree:

* **Nothing crosses from L3/L4 down into L0.** `analytics/events/zones.py` is built this way
  — every event there reads only floor polygons in metres and track positions, and touches
  no model output.
* **Heads read only the neck, never each other** (`models/hydranet.py` design rule 2), and
  `forward` is pure convolution so the ONNX/TensorRT export stays clean. This is what makes
  rung 6 cost 3% instead of a rewrite.

---

## 3. L0 — the per-frame network

### 3.1 Trunk, frozen

| part | value | why not something else |
|---|---|---|
| backbone | RegNetX-800MF | `DEPLOY.md` swept it: `regnet_x_400mf` inside noise |
| neck | BiFPN ×2, 96 ch, P3–P7 | 64 ch measured **slower** (2.45 vs 2.31 ms); `num_repeats: 1` inside noise |
| input | **640 × 1120**, FP16 | §3.3. Do not reduce |
| precision | FP16 | `--best` measured slower (2.48 vs 2.31 ms) |

### 3.2 Heads

| head | output | status |
|---|---|---|
| **detection** (FCOS, P3–P7) | `person`, `bag`, `device`, `boxed_stock`, `stack` | 4 classes trained today; `stack` is new and `device` carries a collected-but-merged sub-label `iphone \| ipad \| macbook` |
| **pose** | 17 keypoints per person | **the second head to land.** Today ViTPose offline; pilot passed 2026-08-19 |
| ~~terrain~~ | — | **not trained, not exported.** Static structure is a commissioning artefact — §4 |

**Why pose and not segmentation as the second head.** Walking the placement rule over every
output the product sells, nothing requires per-frame dense segmentation:

* occupancy, dwell, path, heatmap, line-cross, intrusion, loiter, queue, tailgating, crowd →
  **one floor point per person**. A bbox foot point through the homography gives it.
* "is the shelf occluded, can I trust the count" → a person **bbox** overlapping the shelf
  ROI is sufficient, and being conservative there is correct behaviour anyway.
* "something was left on the floor" → background subtraction against the static plate, on
  the CPU, gated on "persistent and no track explains it". `events/zones.py` already has
  `object_left_events`.
* "is the device in a hand or on the shelf behind them" → a mask cannot resolve depth
  ordering in a projective view either. What resolves it is the item's support point through
  the homography against the known shelf position. **Geometry, not segmentation.**

Pose, by contrast, **cannot be cached** — it differs every frame — and three events on the
list need it (`reach_to_shelf`, `crouch`, `fall`). That is what makes it the head worth
having.

**Measured, RTX PRO 6000 idle, FP16 eager, 640×1120, batch 16, random weights:**

| configuration | params | trunk share | frames/s |
|---|---:|---:|---:|
| shared trunk, two heads | 7.97 M | 88.2% | **941** |
| detection head only | 7.70 M | 91.3% | 970 |
| second head only | 7.30 M | 96.2% | 1221 |
| **two separate networks** | — | — | **541** |

So: trunk 7.03 M, second head 0.27 M, detection head 0.67 M. **A shared trunk buys the
second task for 3%; two networks cost 74%.** At the 1,440 frames/s target, 941 is in range
with TensorRT and 541 is hopeless. That is HydraNet's whole reason to exist, and it is
contingent on genuinely needing a second per-frame task — which pose is and segmentation is
not.

Batch saturates at 8–16 (batch 32 is *slower* than 16), so `exports/pro6000/xl_b64.onnx` is
pointless and the serving tick should fill 8–16. `channels_last` is slower than contiguous
(865 vs 938).

> **Unmeasured:** the pose head's actual cost. The 0.27 M / 3% figures were measured with
> the segmentation head in that slot. Pose has more output channels and may cost more.
> This is gate 2's job.

### 3.3 Why the input stays at 640 × 1120

FCOS's finest level is stride 8, so a box whose short side is under 8 px does not span one
feature cell — not hard to learn, **unlearnable**.

| input | canvas content | median object short side | boxes under 8 px |
|---|---|---|---|
| 512 × 640 | 70.3% | 17.3 px | 8.0–12.5% |
| 512 × 896 | 98.4% | 24.3 px | 1.6–2.0% |
| **640 × 1120** | 98.4% | **~30 px** | — |

And the jump was measured on a pair of configs differing in nothing else:
`detection_mAP/site_boxes` **0.1210 → 0.1469, +21% relative** — larger than batch03's extra
annotation (+0.013) and class weights (+0.002) combined.

### 3.4 Two-scale inference for small objects

Whole-frame upscaling is the expensive way to find small objects. People are large in a
ceiling view; devices are not. So:

* **full frame, downscaled** → `person`, `bag`, `stack`
* **shelf ROI crops at native 1080p** → `device`, `boxed_stock`

The shelf ROIs come free from commissioning (§4) — the cached fixture mask *is* the ROI
list. Higher accuracy on the small classes at lower cost than raising whole-frame
resolution.

> **Unmeasured.** Worth a measurement before committing: mAP on `device` at native-ROI
> versus whole-frame 640×1120.

---

## 4. Commissioning — per camera, once, offline

This is a product surface, not a setup chore: onboarding cost is what decides whether the
system scales past one chain. Target **under 20 minutes per camera**, of which ~4 are human.

| step | what | human? | status |
|---|---|---|---|
| 4a | temporal-median **static plate** from a daytime clip; people and moving stock disappear | no | ✅ `scripts/static_plates.py` |
| 4b | **4 ground points** on the undistorted plate (a floor tile of known pitch, or a measured 1 m square) → homography, pixels ↔ floor metres | **yes, ~4 clicks** | ❌ tool to build |
| 4c | **SAM 3 + Grounding DINO once** on the plate → `floor` / `wall` / `display_table` / `display_shelf` masks, glass polygon, shelf ROI list | no | ✅ `tools/site30k/recipe.py` — this is its real home |
| 4d | **zone polygons** in metres: entrance line, till area, premium shelf, stockroom door | **yes, drawn once** | ❌ tool to build |
| 4e | store the plate for **tamper and fixture-change detection** | no | ❌ |
| 4f | **known-false-positive polygons** — hanging accessory walls and printed people, *derived* from the box population, not drawn | accept/reject the proposed list | ❌ tool to build; measurement done |

Output: one `camera.json` per camera — homography, zone polygons in metres, cached masks,
shelf ROIs, plate reference. Valid until the camera is moved.

**Why a teacher run and not a trained head.** A per-camera constant is best represented per
camera: the cached mask never has to generalise, because it is fitted to that camera by
construction. A learned dense head must generalise across cameras to be useful, and this
project has measured that it does not (`column`: 0.86–0.88 trained, 0.00–0.51 unseen). At 48
cameras and ~40 s of teacher time each, the cache wins outright.

**This is scale-dependent and the threshold is computable.** At tens of thousands of
self-installed cameras, per-camera teacher runs stop being affordable and a generalising
segmentation head becomes the right answer. Revisit when the deployment target passes ~1,000
cameras.

**Tamper detection is not optional.** The entire metric layer rests on a fixed homography. A
camera knocked once silently invalidates every metre in the system without raising an error.
Monitor current-frame similarity against the stored plate; flag the camera for
re-commissioning on divergence.

---

## 5. Data strategy

### 5.1 The frame that decides everything

Public datasets and auto-annotation solve **different** problems, and conflating them is
what burned 10.4 GPU-hours on 2026-08-20:

* **Public data buys class knowledge** — what a laptop, a bag, a box looks like. Cheap, huge,
  **wrong viewpoint**.
* **Auto-annotation buys domain adaptation** — what those look like from *this* ceiling
  camera at *this* resolution under *this* compression. Costly per camera, **right
  viewpoint**.
* **Neither buys a measurement.** §6.

The measured bottleneck in this project is **viewpoint, not class count**: `person` is
trained on COCO's eye-level uncompressed photography against ceiling-mounted h.264.

> **Therefore: do not auto-annotate public data — it already has boxes. Auto-annotate only
> your own footage, because in-domain labels are the ones you cannot buy.**

### 5.2 Already on disk and unused — spend this first, it is free

| dataset | size | why it matters |
|---|---|---|
| **PoseLift** | 68 M | WACV 2025. A **real retail store, 6 indoor cameras, pose sequences + person IDs + frame-level shoplifting labels.** The only real, labelled, in-domain data for the behaviour model (rung 5) — and it is exactly the concealment problem. No config references it |
| **hm3d_cctv** | 501 M | Synthetic renders at **the measured store camera pose** (height 2.38 m, pitch 50.2°, vfov 70.4°). Perfect boxes and keypoints, right viewpoint, unlimited quantity, zero annotation cost |
| **RAP-v2** | 5.4 G | Surveillance-domain attributes — closer to this problem than PA-100K, whose `AgeOver60` recall is 0.119 and degrading |
| **`_incoming/attr_bundle`** | 11 G | Never moved into `datasets/`. Identify before downloading anything else |

`PoseLift` ships poses, not pixels — it is training and evaluation data for the **behaviour
model and shoplifting anomaly detection**, not for the pose head.

### 5.3 Public datasets to acquire — ordered by whether they fix viewpoint

**Fix viewpoint (priority — this is the measured bottleneck):**

| dataset | content | why | caveat |
|---|---|---|---|
| **CEPDOF** (9 GB), **HABBOF**, **WEPDTOF** — BU COSSY | overhead person detection, human-annotated, spatio-temporal IDs, low-light included | the only human-labelled top-down person data. COCO cannot supply foreshortened, radially-oriented bodies seen from above | **fisheye projection**, not rectilinear. Mitigation: rectify each fisheye into 4 virtual perspective views — the repo already has a k1 division-model undistort |
| **WILDTRACK**, **MultiviewX** | calibrated multi-view pedestrians with **ground-truth ground-plane positions** | **lets you measure L1 without labelling anything** — real truth for the homography → floor-position pipeline | outdoor/synthetic, but what is being validated is geometry, not appearance |
| **MOT17 / MOT20** | dense pedestrian tracking | crowding and compression artefacts | oblique, not top-down |

**Add class knowledge (cheaper, secondary):**

| dataset | content | why |
|---|---|---|
| **LVIS** | fine-grained annotations over **the COCO 2017 images already on disk** — `cellular_telephone`, `laptop_computer`, `tablet_computer` | the cheapest acquisition in this table: annotations only, ~1 GB, zero new pixels |
| **SKU-110K** | 11,743 shelf images, **1.7 M boxes, ~147 per image, single class** | the right source for `stack` and dense `boxed_stock`. Class-agnostic, which is exactly what is wanted — detect box instances, do not classify SKUs |
| **RPC** | 30,000 **top-down** checkout-tray scenes, ~12 products, 200 SKUs | correct viewpoint for product appearance from above |
| **Open Images V7** (subset) | `Mobile phone`, `Tablet computer`, `Laptop`, `Handbag`, `Backpack`, `Suitcase` | highest volume for `device` and `bag`. Take only the needed classes |

**Do not acquire:** DukeMTMC (withdrawn on privacy grounds); Kinetics / UCF101 / HMDB
(eye-level web video, negligible transfer to ceiling surveillance).

**No public source exists** for `staff / customer`, or for overhead `fall` / `crouch`. Those
are necessarily in-domain: 3 uniform reference photos plus per-track VLM voting for staff,
and 50–100 staged clips for fall. Staged clips are **data generation, not annotation**.

### 5.4 Auto-annotation — what the large models are for now

With no segmentation head to train, the teachers' role narrows, which is a simplification:

1. **Commissioning masks** (§4c) — per camera, once. Not training data; a cached artefact.
2. **Box tightening.** Grounding DINO's boxes are loose; SAM 2/3 mask → tight bbox. Zero
   human input, measurable quality gain.
3. **Temporal consistency filtering — the highest-value step, and it was skipped.** A
   detection a point tracker follows coherently across frames is real; one whose tracked
   points scatter is a hallucination. This is what turns a score-0.14 box into either a
   usable label or a discard. **The raw material is already on disk:**
   `datasets/site30k_v1/annotations/instances_all_*.json` keeps every box down to score 0.10
   with no NMS, precisely so this question could be answered later without GPU time.
4. **Cross-model agreement as the confidence signal.** Grounding DINO + SAM 3 + the current
   student: 2-of-3 agreement → **Gold**; disagreement → **ignore region, never a negative**.
   Gold trains at full weight, Silver at 0.5.
5. **What they cannot do:** produce a measurement. A student trained on SAM 3 and scored
   against SAM 3 shares its errors — recorded in `METHODOLOGY.md` §0 and in each
   `split.json`'s `test_provenance`.

### 5.5 How to mix public, synthetic and in-domain — the exchange rate is measured

`METHODOLOGY.md` §0: the COCO share is a **monotonic trade, not a collapse**. Swept at four
ratios and scored on *test*: at `sample_ratio: 0.1` every segmentation metric matches or
beats the segmentation-only baseline and detection arrives free; above it, segmentation falls
monotonically. There is no sweet spot to find — only an exchange rate that worsens as you
climb.

So: **public data at low `sample_ratio` as a trunk prior; in-domain at high ratio as the
body.** And synthetic and pseudo-labels are complementary, not alternatives:

* **synthetic (`hm3d_cctv`) supplies geometry and viewpoint** — exact labels at the real
  camera pose, unlimited
* **in-domain pseudo-labels supply appearance** — real lighting, real h.264, real fixtures

### 5.6 Split discipline — unchanged, and it binds

From `RETAIL_DATA.md`: split by camera never by frame (R1); a test camera never supplies a
training frame (R2); **every rare class on at least two test cameras (R4)**; a per-class
number over fewer than two test cameras is **not reported** — write "not measured" and the
camera count (R7).

`site30k_v1`'s test split is one camera and therefore reports nothing. Fixing it needs more
cameras, which is §7 step 3, not more frames.

---

## 6. Evaluation — how the "no manual annotation" and "need a human-graded test set" contradiction resolves

The plan forbids manual annotation. `METHODOLOGY.md` §0 says the binding constraint is the
absence of a human-corrected test split. Both cannot hold — and the resolution is standard
industry practice:

> **Shadow mode.** The system runs live but raises no alerts to the store; every alert it
> *would* have raised is logged. An operator grades them accept / reject.

That grading is the human-corrected test set. It is free (the operator was going to read
alerts anyway), it is perfectly in-distribution, it accumulates forever, and it labels
exactly the cases the system gets wrong. It also produces the one number
[VISION.md](VISION.md) §3 defines as success: actionable alerts per camera per day.

Model-level metrics stay as debugging instruments, reported with their camera counts per R7.

---

## 7. Build order — one artefact and one gate per step

The last cycle failed as a long chain with no gates. Each step here produces one thing that
can be looked at, and states what makes it fail.

| # | step | artefact | gate |
|---|---|---|---|
| 1 | freeze [VISION.md](VISION.md) and §1–§4 here | these two documents | you agree with the placement rule, the L0 table and the commissioning flow |
| 2 | **throughput ceiling**, with the pose head in place | one number: frames/s under TensorRT + CUDA graphs, against 1,440 | ≥1,440 → resolution and two heads stand. Below → we know exactly what to trade, measured. `trtexec` is absent from this machine; either install the TensorRT CLI or drive the Python API |
| 3 | **commissioning, all 48 cameras** | per-camera `camera.json` + **a 1 m floor grid rendered on a real frame, one image per camera** | the grid looks right to your eye on every camera |
| 4 | **detection first** (`RETAIL_DATA` order, and the plan's own §5 #7) | `site_boxes` mAP; Gold/Silver/Gray tiers built from the existing `instances_all_*.json` at **zero GPU cost** | Gold precision ≥95%, Silver ≥85% on a 300-frame sample. **Tiering pass done 2026-08-24 — Gold 60/60 by eye, Gray 3–4/60, and 43.4% of Gray traced to 37 fixed hotspots across nine cameras. See [journal/2026-08-24](journal/2026-08-24-box-tiering-and-the-fixed-false-positives.md)** |
| 5 | **pose head resident** | keypoint error against the ViTPose teacher, plus the throughput number from step 2 re-taken | pose good enough for `reach_to_shelf` and `crouch` to fire correctly on a clip you watch |
| 6 | **L3 end to end, one camera** | an event log readable against the video it came from | the events match what you see when you watch the clip |
| 7 | **shadow mode, one store** | alerts/camera/day, and the operator's accept/reject log | the rate is in single digits and the rejects have a pattern we can act on |

Steps 2 and 3 are independent of each other and of everything below. Nothing after 6 starts
until 6 passes: the crop heads, the behaviour model and the VLM all consume L3's output, so a
wrong L3 makes all three unmeasurable.

**Done 2026-08-24 — step 4's tiering pass over `instances_all_*.json`, zero GPU.** What the
10.4 GPU-hours of 2026-08-20 left behind: a `person` population that is 87.7% hallucination
after NMS, of which **43.4% comes from 37 fixed positions** — hanging accessory walls and
printed people — and product boxes that survive the segmentation defect with a 15% dip rather
than a collapse. The next zero-GPU item is 4f: derive the known-false-positive polygons from
those hotspots and put them through an accept/reject pass.

---

## 8. Open — each blocks a specific step

1. **The 1,440 frames/s target has never been measured** (step 2). Eager reaches 941 with
   two heads. `scripts/bench_pro6000.sh` cannot run: `trtexec` is not installed, only
   TensorRT's Python bindings.
2. **`stack` as a new detection class** takes the head from 4 to 5 classes — rung 7, so it
   invalidates every checkpoint's comparability. And a vocabulary change silently empties
   `analytics/events/zones.py:346`, whose default class list is `("boxed_stock", "device")`:
   `wanted` becomes empty, `zone_stock_counts` returns 0 for every frame, and the
   stock-removal alarm stops firing **without raising an exception**.
3. **`bag` was dropped in the earlier plan.** It is free from COCO (11,989 images, 27,168
   boxes) and concealment into a bag is the core theft action. Recommend keeping it.
4. **Night is unscoped.** `person` is measured firing on hanging packets on an empty IR
   frame — 14 false people, and consensus voting cannot remove them. Yet `after_hours_person`
   is the first entry on the VLM trigger whitelist. Either night enters with a measurement,
   or the trigger comes off the list.
5. **Delivery target undefined** — who receives v1, when, and whether 96 streams is a
   customer requirement or design headroom. This changes step 2's verdict and the
   commissioning-versus-trained-head threshold in §4.
6. **Retail dashboards are unscoped.** Not a sequencing question — security and retail are
   one system reading one camera ([VISION.md](VISION.md) §1), and the analytics outputs fall
   out of L1 for free once the ground plane exists. What is genuinely missing is the
   presentation layer: which numbers a store manager opens, at what cadence, in what form.
   That is a product-surface gap, not a modelling one, and it blocks nothing before step 6.
