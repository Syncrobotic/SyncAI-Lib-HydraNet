# SyncAI CCTV analytics — the plan

Written 2026-08-25, replacing the entire previous documentation set in one file.
Everything it replaces is readable at `git show b7457c2:docs/<file>` (VISION.md, PLAN.md,
METHODOLOGY.md, RETAIL_DATA.md, the journal, and the rest). Where this file cites a
measurement, the provenance is either a config/dataset in this tree or a journal entry at
that commit.

Every number below is **measured** (and says where) or marked **unmeasured**. Nothing here
is an estimate presented as a fact.

Amended the same day after an adversarial self-review; choices made with the user in that
review are marked **decided 2026-08-25** in place.

---

## 1. What this is

A vision system for the **security and retail CCTV analytics domain** — nothing else. It
reads fixed, already-installed store cameras (no LiDAR, no new hardware) and answers two
families of question from the same frames:

* **Security / loss prevention** — who entered where, how many, how long, what did they do,
  did stock leave unpaid.
* **Retail analytics** — footfall, dwell, paths, queues, which display draws attention.

One camera, one model, one export, two readings. The first vertical is an Apple-reseller
chain: white fixtures, glass shopfronts, small high-value devices on open tables — the
environment where generic web-trained models fail hardest.

**Success is one number:** actionable alerts per camera per day, and the incidents missed.
Model metrics (mAP, IoU) are debugging instruments, never the reported result. And no site
figure is an *accuracy* until a human has graded it — until then it is an *agreement* with
the teacher models (§5.5 resolves this via shadow mode).

## 2. The two packages

```
src/syncai_bev3d      runs ONCE per camera, offline    →  camera.json
src/syncai_hydranet   runs EVERY frame, continuous     →  boxes + keypoints → tracks
                                                          in metres → events → alerts
```

The boundary is the load-bearing design decision: **anything constant on a fixed camera is
computed once by `syncai_bev3d` and cached; only what changes frame-to-frame is computed by
`syncai_hydranet`.** The contract between them is one `camera.json` per camera.

**Dependency rule — as built (2026-08-25).** `syncai_bev3d` imports `syncai_hydranet`'s
core along four named edges: the labels contract (`IGNORE`), the runtime geometry it
writes parameters for, the single `iou` in `analytics.tracker`, and the prompt tables in
`data.sam3_prompts` (which stay in hydranet because the label maps read them too).
Hydranet's offline CLI may import bev3d (`cli/scene.py` does, for the BEV renderer). The
**serving path — `serving`, `analytics`, `models`, `engine`, `geometry` — never imports
bev3d**: at runtime `camera.json` is the only crossing, and its loader lives in
`syncai_hydranet.geometry.camera_json` because a reader barred from the producer's
package still has to read the file. `tests/test_package_boundaries.py` enforces both
directions, so the rule is a failing test rather than a memory.

Why the boundary sits here — three measurements:

* A trained dense head does not generalise across these cameras: `column` scored 0.86–0.88
  on cameras it trained on and **0.00–0.51 on cameras it never saw**. A commissioning cache
  never has to generalise — it is fitted to its camera by construction.
* The throughput margin is thin: **1,552 fps under TensorRT against the 1,440 target
  (1.08×, measured 2026-08-24)** — engine-only, and with the cheaper terrain head in the
  second slot. Every per-frame head spends this margin; a head whose answer never changes
  spends it on nothing.
* A shared trunk buys a second task for **+3% cost; two separate networks cost +74%**
  (941 vs 541 fps eager, batch 16, 640×1120 FP16). The trunk is worth sharing only between
  tasks that genuinely need to run every frame.

Scale caveat: at ~1,000+ self-installed cameras, per-camera teacher runs stop being
affordable and a generalising segmentation head becomes the right answer. Revisit then.

### 2.1 `syncai_bev3d` — first-pass analysis, calibration, 3D scene

Per camera, once, offline. Target under 20 minutes, of which ~4 are human.

| step | what | human? | status |
|---|---|---|---|
| a | temporal-median **static plate** — people and moving stock disappear | no | ✅ `scripts/static_plates.py` |
| b | **4+ ground points** on the undistorted plate → homography by least squares. Four is the minimum, not the target: a 4-point fit has zero redundancy and its error grows unquantified toward the frame edges, exactly where speed rules read it | **~4–8 clicks** | ❌ tool to build |
| c | **SAM 3 + Grounding DINO, one pass on the plate** → structure masks: `floor`, `wall`, `column`, `door`, `display_table`, `display_shelf`, plus `product` subclassed `iphone/ipad/macbook/boxed_stock` (measured usable at 1080p — the "0 instances" verdict was 352×240), completed by depth and floor-boundary geometry where the teachers go blind. `glass` stays human-drawn: 112 frames, four failure modes | no | ✅ run for all 8 commissioned cameras (`tools/commissioning/`) |
| d | **walkable / non-walkable** = floor mask − fixture masks − forbidden zones; indoor/outdoor split is a polygon where a camera sees through the shopfront | no | derived from c |
| e | **zone polygons in metres** — entrance line, till, premium shelf, stockroom door | drawn once | ❌ tool to build |
| f | **shelf ROI list** (from the fixture masks) — drives two-scale product inference (§2.2) | no | derived from c |
| g | **known-false-positive polygons** — derived from the box population, not drawn (hanging accessory walls, printed people) | accept / reject | ✅ derived, reviewed and written for all 8 — 20 polygons covering 7–78% of each camera's hallucination population (`fp_polygons.py`; verdict 2026-08-25: every candidate was a poster or merchandise) |
| h | store the plate for **tamper / fixture-change detection** — a knocked camera silently invalidates every metre downstream | no | ❌ |
| i | **3D scene / BEV render** for verification: a 1 m floor grid drawn on a real frame | checked by eye | partial — `geometry/bev3d.py`, `geometry/depth_scene.py` |

Output: `camera.json` — homography, structure masks, walkable polygon, zones in metres,
shelf ROIs, FP polygons, plate reference. Valid until the camera moves **or the store
does**: seasonal display resets invalidate the fixture masks, shelf ROIs and human-drawn
zones while leaving the homography intact.

**Decided 2026-08-25 — the refresh is a nightly job, not a person.** In closed-store
hours the pipeline captures a clip (an empty store barely needs the temporal median),
compares it against the stored plate, and on divergence re-runs the teachers and
re-derives the masks, ROIs and walkable polygon. Zones follow automatically where they
can: floor-anchored zones (entrance line, till) do not move with displays, and
fixture-anchored zones (premium shelf) re-attach to the matched fixture in the new mask.
A human sees only a **morning accept/reject when the diff exceeds a threshold** — fully
silent zone drift is how a knocked zone raises wrong alerts for a month. A homography
change (the camera itself moved) always escalates to re-commissioning, never auto-heals.

**Depth lives here, not in a per-frame head.** Every spatial output the product sells
(dwell, paths, heatmaps, queues, zone occupancy) needs *the person's floor position in
metres*, and the homography gives that exactly, in metric units, from 4 clicks. A monocular
depth head gives relative depth at per-frame GPU cost, needs a scale calibration anyway,
and has no right-viewpoint training data.

The package exists (2026-08-25): `syncai_bev3d` holds bev, bev3d, calibrate,
plate_calibration, depth_scene, meshes, shading, scene_types and `teachers/` (sam3,
gdino, boxes, photometry). The runtime side — `Camera`, `GroundPlane`, the projections,
`undistort_points` and the `camera.json` loader — stayed in `syncai_hydranet.geometry`,
because commissioning *fits* the parameters and serving *applies* them, and both sides
sharing one definition is what keeps the metres honest. `scripts/static_plates.py` and
the `tools/site30k` recipe remain thin front ends over the package.

### 2.2 `syncai_hydranet` — the per-frame network

Trunk, frozen by measurement: **RegNetX-800MF + BiFPN ×2, 96 ch, P3–P7, input 640 × 1120,
FP16.** (64 ch measured slower; input below 640×1120 measured −21% relative site mAP —
boxes under 8 px do not span a stride-8 cell and are unlearnable, not merely hard.)

**Two heads:**

| head | output | status |
|---|---|---|
| **detection** (FCOS) | `person`, `bag`, `device`, `boxed_stock` (+ `stack`, open — §7) | trained today. `device` carries collected-but-merged sub-labels `iphone \| ipad \| macbook` — they stay merged until each sub-class exists on ≥2 test cameras (§4.4), then splitting is a deliberate re-baseline |
| **pose** | 17 keypoints / person | next to land. ViTPose is the offline teacher; measured over 66,599 verified person boxes, **99.9% of people clear the 32 px bottom-up floor** at network scale (median height 178 px) |

**Two-scale inference for small objects:** full frame at 640×1120 for `person`/`bag`/
`stack`; shelf-ROI crops at native 1080p for `device`/`boxed_stock`. The ROIs come free
from commissioning (§2.1f). **Decided 2026-08-25: the ROI path polls at 0.2–1 fps per
camera, not frame rate** — shelf stock is a slow variable and the stock-removal alarm
tolerates tens of seconds of latency. This matters for the budget: the 1.08× throughput
margin was measured **whole-frame only** — the ROI passes were never in it, and at frame
rate they would multiply the per-camera load past the margin. **Unmeasured:** `device` mAP
at native-ROI vs whole-frame, and the ROI path's cost at the chosen cadence — both taken
in gate 3's re-measure.

**No dense segmentation head, no depth head, no behaviour head.** The static structure
those would predict is a per-camera constant (§2.1); single-frame behaviour classification
is ill-posed (walking vs standing is motion, invisible in one frame — and no teacher can
label it from one frame either). Behaviour lands in §2.3.

**The risk the architecture rests on:** distilling top-down ViTPose (per-crop) into a
bottom-up whole-frame head is standard but not free. If it fails, pose becomes a crop-stage
model, L0 has one head, and the honest move is an off-the-shelf detector. Gate 3 in §6
exists to answer this early.

### 2.3 The layers above — where behaviour actually lives

```
L0  pixels  →  boxes + keypoints      hydranet, every frame, GPU
L1  boxes   →  tracks in METRES       tracker + camera.json homography, CPU
L2  tracks  →  per-person facts       crop heads, ~every 3 s per track, GPU
L3  facts   →  events                 rules in metres and seconds, CPU
L4  events  →  judgement              VLM on trigger, GPU queue
```

Two L1 rules that are cheap to write and expensive to discover missing:

* **Floor position is the foot point through the homography — except under fixture
  occlusion.** A person behind a display table has their bbox cut at the table edge; the
  naive foot point lands metres off, and the bias concentrates exactly at the
  highest-value positions (table-edge dwell, shelf reach). When ankle keypoints are
  missing or low-confidence, L1 falls back to head/shoulder keypoints plus a height prior
  — the pose head pays for itself here a second time.
* **The time base is PTS, never frame index.** Measured: every clip in
  `gs://studioa-recording` writes `30/1` into `r_frame_rate` regardless of the true,
  variable rate. Speeds in m/s over nominal fps silently misclassify walk as run on a
  slow stream, and the failure surfaces only as unexplainable run alerts.

* **walk / stand / run** — speed thresholds over L1 tracks. CPU, free, and the thresholds
  are config a manager can change, not classes.
* **sit / crouch / fall** — a <100K-parameter temporal model over pose sequences, CPU.
  Training data is already on disk: **PoseLift** (real retail store, 6 indoor cameras,
  pose sequences + person IDs + frame-level shoplifting labels). For the actions PoseLift
  lacks — fall, crouch — in-store staging is ruled out entirely (decided 2026-08-25,
  off-hours included), so the source is **3D action / mocap data projected to the
  measured store camera pose**: NTU RGB+D's falling and squatting classes, CMU MoCap
  skeletons, projected through the same camera parameters `hm3d_cctv` renders at (height
  2.38 m, pitch 50.2°, vfov 70.4°). The model consumes keypoint sequences, not pixels, so
  the projection *is* the domain adaptation; sim-to-real transfer is **unmeasured** and
  is checked at step 6 on watched clips. **Caveat, unmeasured:**
  PoseLift's sequences came from a different pose estimator whose noise (jitter spectrum,
  occlusion failure modes) differs from our distilled head's; train with noise
  augmentation matched to our head's error profile and measure the transfer after gate 3.
* **loiter, intrusion, line-cross, queue, dwell, tailgating** — rules over tracks in
  metres. `analytics/events/zones.py` is already built this way.
* **intent / concealment** — VLM on trigger only.

Invariants already enforced in the tree: nothing crosses from L3/L4 down into L0, and heads
read only the neck, never each other (`models/hydranet.py`), so `forward` stays pure
convolution and the ONNX/TensorRT export stays clean.

#### 2.3.1 What L1 emits — the vector space, as a contract

**Built 2026-08-25: `analytics/world.py` (`WorldFrame`, `WorldObject`, `world_frame`).**
`analytics/stage.py` typed what enters the second stage and typed it in **pixels**; the
metre side had no type, so `dwell.track_ground_path`, `events/zones.py` and `cli/scene.py`
each called `pixel_to_ground` and kept the answer in a private shape — the same failure
`stage.py` records as its own reason for existing, one coordinate system later.

`WorldFrame` is one camera's floor at one instant: `frame_index`, PTS `time_s`, `space`,
and a list of `WorldObject{track_id, name, x_m, z_m, vx_ms, vz_ms, yaw_rad, height_m,
observed, basis}`. Four decisions carry it:

* **It lives in `syncai_hydranet.analytics`, not `syncai_bev3d.scene_types`**, even though
  `PlaneObject`/`DepthObject` already describe almost this shape and `DepthObject` already
  carries `yaw_rad`. The serving path may not import bev3d (§2), and this is produced every
  frame on the serving path. Same precedent as `geometry/camera_json.py`.
* **`space` names which metric frame the coordinates live in**, carried per frame rather
  than assumed — the rule `BoxFrame.class_names` exists for. Today the only value is
  `camera_floor(<camera_id>)`, because **there is no store frame**: nothing in `CameraFile`
  maps a camera's metres onto a store plan, so two cameras' `x_m` are two different x. That
  is open question 5's missing piece and it is a commissioning artefact (a 2D similarity
  transform per camera, 2–3 correspondences against a store plan), not a model change. It
  needs a `SCHEMA_VERSION` bump, so the 8 shipped `camera.json` get regenerated.
* **Every key is required and unfillable values are `None`**, so "not supplied" and "not
  measured" stay different claims. `yaw_rad` is `None` until the pose head lands (shoulder
  line → body yaw, which is what §1's "which display draws attention" actually needs);
  `height_m` is `None` until a serving-side producer exists.
* **`basis` names the instrument**, as `SecurityEvent.basis` does: `foot_point` today,
  `keypoint_ankle` / `keypoint_prior` reserved for the occlusion fallback above,
  `above_horizon` for the refusal `pixel_to_ground` already makes.

**A correction it carries and `dwell` does not.** `camera_json.py` states that the lens
applies to points on their way to the floor, and `undistort_points` states that a runtime
consumer that skips it makes the metres drift silently. Nothing on the serving path did:
the only callers are two commissioning modules, and `track_ground_path` projects the raw
foot point. `world_frame` undistorts. `track_ground_path` is deliberately left alone —
fixing it moves every dwell, path and heatmap number already reported, which is a
re-baseline and belongs in its own commit next to a measurement of what moved.

**State: one producer, no consumer.** `dwell` and `events/zones.py` still take `Track` and
keep taking it until moving them is measured to change no number. Adopting it is step 5's
work, not this commit's.

## 3. The requirements → where each one lands

The four capability families this plan must deliver, and their placement:

| requirement | instrument | cost |
|---|---|---|
| walkable / non-walkable area (indoor, outdoor) | `syncai_bev3d` §2.1d, cached mask | once per camera |
| person boxes | detection head | exists |
| floor / column / wall / door / glass / display table / display shelf | `syncai_bev3d` §2.1c, cached masks | once per camera |
| products — iphone / ipad / macbook / boxed-stock | detection head `device` + `boxed_stock`, native-res shelf-ROI crops; sub-labels collected, split when measurable | exists + ROI path unmeasured |
| depth / 3D space | homography + BEV scene (§2.1b, i) — metric, not relative | 4 clicks per camera |
| behaviour: walk / stand / run | L1 speed rules | free, CPU |
| behaviour: sit (+ crouch, fall) | pose head → temporal model on PoseLift | pose is the next head |

The general placement rule, kept from the previous plan because every 2026-08-19/20 mistake
was a violation of it: **for any new capability, walk down from "a config value" and stop
at the first rung that answers it — never start at "a new head".**

## 4. Data — minimum human labelling, teachers do the rest

### 4.1 The onsite corpus

`gs://studioa-recording` — the store's own CCTV. Fleet reality, measured: **48 cameras, 6
emit nothing, 1 mounted sideways, 24 annotated so far.** Two rules with measurements behind
them:

* **More cameras buy generalisation; more frames from the same cameras do not.** The
  2026-08-20 counter-example: 29,211 new frames, 10.4 GPU-hours, zero new cameras, no gain.
* **Viewpoint, not class count, is the bottleneck.** Auto-annotate only our own footage —
  public data already has boxes; in-domain labels are the ones money cannot buy.

### 4.2 The teachers — SAM 3 + Grounding DINO

Both already live in the wheel (`data/teachers/`). Their jobs, in value order:

1. **Commissioning masks** (§2.1c) — per camera, once. A cached artefact, not training data.
2. **Detection pseudo-labels on site footage.** Grounding DINO for `person` (measured
   day:night separation 11–49× against SAM 3's 0.9×); SAM 3 mask → tight box refinement.
   SAM 3 product prompts need adequate resolution — the "0 instances" result was 352×240.
3. **Temporal-consistency filtering** — a box a point tracker follows coherently is real; a
   scattering one is a hallucination. Raw material already on disk:
   `datasets/site30k_v1/annotations/instances_all_*.json` keeps every box to score 0.10,
   no NMS. Zero GPU cost.
4. **Cross-model agreement as confidence:** GDINO + SAM 3 + current student, 2-of-3 →
   **Gold** (full weight); disagreement → **ignore region, never a negative**. The tiering
   pass ran 2026-08-24: Gold verified **60/60 by eye**, and 43.4% of the Gray tier traced
   to the 37 fixed hotspots that become §2.1g's FP polygons.

### 4.3 Supporting datasets

* **On disk, unspent:** `PoseLift` (behaviour model, §2.3); `hm3d_cctv` (synthetic renders
  at the measured store camera pose — geometry and viewpoint, unlimited, exact labels);
  `RAP-v2` (surveillance-domain attributes, for the L2 crop heads later).
* **Worth acquiring, in order:** CEPDOF/HABBOF/WEPDTOF (the only human-labelled overhead
  person data; fisheye — rectify into virtual perspective views), WILDTRACK/MultiviewX
  (ground-truth floor positions — validates L1 geometry without labelling anything), LVIS
  (device-class annotations over COCO images already on disk), SKU-110K (dense box stacks),
  RPC (top-down product appearance).
* **Mixing rule, measured:** the COCO share is a monotonic trade — at `sample_ratio: 0.1`
  detection arrives free; above it, in-domain performance falls monotonically. Public data
  is a low-ratio trunk prior; in-domain is the body.
* **Necessarily in-domain:** `staff/customer` (3 uniform reference photos + per-track VLM
  voting). `fall`/`crouch` lost its planned staged-clip source — **all in-store staging is
  ruled out (decided 2026-08-25, off-hours included)** — and moves to pose space: the
  temporal model reads keypoints, not pixels, so public 3D action data projected to the
  measured camera pose replaces staged clips entirely (§2.3).

### 4.4 Split discipline — binds every reported number

Split by **camera**, never by frame. A test camera never supplies a training frame. Every
rare class on ≥2 test cameras, or the number is written "not measured" with the camera
count. One fixed camera is one scene measured N times, not N samples.

### 4.5 Where human labour goes — all of it

4 calibration clicks + zone drawing per camera (§2.1), accept/reject passes on teacher
proposals, and **shadow-mode grading**: the system runs live raising no alerts, an operator
grades what it would have raised. That grading is the human test set — free,
in-distribution, accumulating, and it produces the success number itself.

**Shadow grading measures precision only.** A missed theft never becomes an alert to
grade, so recall has no instrument in that loop. Decided 2026-08-25 (staged in-store
theft tests are not an option): recall is measured by **reconciliation against the
store's shrinkage counts and incident reports**, weekly. The signal is weak and delayed
by weeks — it is also the only one available, so it is reported with that caveat rather
than not at all.

## 5. What it must never do

1. Never put an answer in the weights that belongs in a config.
2. Never let the rule layer reach into the model — "4 minutes is loitering" is an argument.
3. Never report a per-class number resting on one camera.
4. Never buy coverage with more frames from already-annotated cameras.
5. Never infer age/gender on customers; no face recognition; no cross-store tracking.
   `staff/customer` is a uniform, not an identity, and stays in.
6. No SKU-level checkout; no new camera hardware; no per-frame dense scene understanding.

## 6. Build order — one artefact, one gate per step

| # | step | artefact | gate |
|---|---|---|---|
| 1 | **package split** — create `src/syncai_bev3d`, move the §2.1 code, define the `camera.json` schema; archive non-current configs | two importable packages, green tests | **DONE 2026-08-25** (`64eafd8`, `6e0eb36`, `1395e2c`) — 1,824 tests green, boundary enforced by `test_package_boundaries.py`, GPU idle and no training unit running when the tree moved. Config archiving deferred: which of the 30 yamls are current needs a decision, not a guess |
| 2 | **commission all 48 cameras** — build the 4-click tool and zone tool, run the pipeline | per-camera `camera.json` + a 1 m grid rendered on one real frame per camera | the grid looks right **to my own eyes on every camera**. **In progress 2026-08-25**: geometry pass done for the 9 scale-measured cameras of the 2026-08-19 onboard scan — 8 `camera.json` shipped (grids eyeballed per camera, `runs/commission01/REVIEW.md`), Taichung-cam05 withdrawn (two furniture checks agree its cells are over-scaled). **Masks + 3D scenes done 2026-08-25 for all 8** (`tools/commissioning/masks_pass.py`, `scene3d.py`): structure masks, walkable, shelf ROIs written into each camera.json; commissioning-only 3D scenes with depth-measured fixture heights (tables 0.78–0.97 m fleet-wide). 5 mask sets clean, Tao-Hsin pair partial (white-fixture teacher blindness — iteration item), Kaohsiung-cam04 misses its pillar. Blocked on the physical world for the rest: 14 cameras need a per-store visual reference, 4 need mount-type triage, 7 Kaohsiung cameras need the person-score investigation; Taichung-cam05 needs its NVR stream setting. Remaining tools: 4-point ground-calibration click tool, zone drawing, FP polygons (4f), tamper reference (4h) |
| 3 | **pose head resident** — **IN PROGRESS 2026-08-25**: the box-conditioned P3 head is built and tested (`ef9b53c`), ViTPose labelled the Gold boxes (train 13,930 / val 8,479 / test 2,862 images), and `hydranet_retail_pose01` is training under systemd (60 epochs, pose loss 0.47 → 0.10 by epoch 2). **Its run directory is `runs/hydranet_retail_security_b03_cw_xl-20260825-162131`** — the pose config inherited `output_dir` from the security run and the trainer appended a timestamp rather than overwrite it, so the directory is named for a different model and only the `config.yaml` inside it is authoritative. `tools/pose/eval_student.py` had never been executed (it died on `torch.from_numpy` over an already-tensor) — **fixed and run 2026-08-25** (`2fb6f0c`), and `--render` now writes per-person crops with both skeletons, because a gate decided by eye has to be lookable-at. **Throughput: `scripts/bench_e2e.py` (`65ce3b5`) is the end-to-end instrument** and it refuses a busy GPU. First legs, measured while training shared the box so they are lower bounds: host NMS 450 fps/thread (3.2 of 24 cores at target — not a risk), tracker 7,694 fps/thread (0.2 cores), **CPU decode saturates at ~950–1,030 fps = 63–69 streams against the 96 required**. **Decode resolved 2026-08-25**: the box had no NVDEC path at all; PyNvVideoCodec 2.2.1 installed (user's call, §7.8) and wired into the harness with `usedevicememory=True`, so frames never cross PCIe. Measured with training still on the card, i.e. lower bounds: **NVDEC saturates ~7,800 fps ≈ 520 streams at 15 fps, 5.4× the target**, against the CPU pipe's 63–69 streams. Decode is no longer the binding leg. **Still open: the exporter drops the pose head** — `cli/export_onnx.py`'s `ExportWrapper` emits segmentation and detection outputs only, so an engine built today measures a model without pose and reports it as the model with pose. Must be fixed before the engine leg is measured; blocked until the training unit stops (never edit `src/` under a running job). **Training finished 2026-08-25 20:01**, 60/60 epochs, final val `coco_person` mAP 0.2022 / mAP@50 0.3891. The checkpoint trap fired exactly as forecast: `primary_metric` is `terrain_mIoU/site_seg03`, so `best.pt` is **epoch 15**. Both candidates run on the full test split (2,862 images / 6,360 persons / 93,967 judged joints):

| checkpoint | PCK@0.2h | L2 p50 | mean | p90 |
|---|---|---|---|---|
| `best.pt` (E15, terrain-selected) | 0.840 | 13.8 px | 35.4 | 96.3 |
| **`last.pt` (E60)** | **0.898** | **9.3 px** | **25.1** | **67.7** |

**Use `last.pt`.** Taking the handoff's `best.pt` would have gated on a figure 5.8 points lower with 48% more error. Trajectory E15 → E29 → E60 is 0.840 → 0.882 → 0.898, still rising at the end. The run logs **no pose validation metric at all**, so there is no per-epoch curve and these two checkpoints are the only candidates that survive a run — fix the config's `primary_metric` before the next one (§7.9).

**Throughput, end to end, 2026-08-25 — the gate MISSES.** `cli/export_onnx.py` was
dropping the pose head entirely (`a0c51de`), so every engine number before this described
the model with pose replaced. With `pose_heatmap_p3` in the graph, E60 weights, batch 16,
fp16, NVDEC decode into device memory, one small RL neighbour on the card:

| leg | measured | against 1,440 f/s |
|---|---|---|
| decode (NVDEC, 16 streams) | 8,647 f/s | **6.01×** |
| **engine (pose resident)** | **1,324 f/s** | **0.92× — binding** |
| post (host FCOS decode + NMS) | 501 f/s/thread | 2.9 of 24 cores |
| track | 7,969 f/s/thread | 0.2 cores |

One card therefore holds **88 streams at 15 fps, not 96**. Levers measured, not guessed:
batch 32 gives 1,300 f/s (*worse* — the card is saturated, so batch size is not a lever);
pruning the terrain head from the graph gives 1,386 f/s (+4.7%, still 3.7% short). The
1,552 f/s of 2026-08-24 that made the target look comfortable was engine-only *and*
without pose. Remaining levers untested: int8/fp8 (`--best`), a narrower backbone, or
re-opening §7.4's 96 | keypoint error vs ViTPose; throughput re-measured **end to end — NVDEC decode → engine → host NMS → tracker** — with pose in the slot and the ROI path at its 0.2–1 fps cadence | `reach_to_shelf` and `crouch` fire correctly on a watched clip; fps ≥ 1,440 stands end to end, not engine-only — 96 streams is a requirement now (§7.4), and decode/NMS/PCIe were named the real risk when the target was set |
| 4 | **detection uplift at zero GPU** — ~~temporal-consistency tiers~~ **REFUTED for `person`, 2026-08-26, measured**: `instances_all_train.json` is not a pool of unused good labels, it is the same Grounding DINO teacher dumped at 0.10 instead of 0.35 — median score 0.149, and `instances_train` is simply that pool thresholded (48,005 vs 47,465 at 0.35). Worse, persistence is the wrong signal here and it is wrong in the dangerous direction: score vs run-length correlation is **+0.006**, the 0.10–0.20 band moves a median **0.3 px per frame**, and **41,160 sub-threshold boxes persist ≥8 frames** — posters, mannequins and hanging packets are the most persistent things in a store. A consistency tier would promote exactly them. The measured detection problem is **recall, not label count**: on one clip the trunk's own dense head marks **1.30× as many people as the box head returns** (146 blobs vs 112 boxes over 60 frames) while scoring `person` IoU 0.885 against detection mAP@50 0.302 — the features find people the box head cannot emit. **Re-scoped and then measured on the fleet, 2026-08-26**: `serving.decode.confirm_with_dense` admits low-scoring boxes where the dense head puts person pixels under them. Three arms over the same eight clips — A shipped 0.35, B threshold-only 0.15, C 0.15 + confirmation:

| arm | person dets | tracks | boxes dropped | fall | crouch |
|---|---|---|---|---|---|
| A 0.35 | 11,363 | 202 | 0 | 1 | 1 |
| B 0.15 | 27,715 | 569 | 0 | 4 | 5 |
| C 0.15 + confirm | 25,713 | 520 | 2,002 | 4 | 5 |

**The recall gain is real and the confirmation does not protect the event layer.** C drops 12–14% of boxes and produces *exactly* B's events, on the whole fleet and again with Kaohsiung-cam04 excluded (2 fall / 3 crouch both arms). The boxes it removes were never the ones firing events: the extra events come from real people detected at low confidence, whose keypoints are noisier, making more posture runs. So the next mechanism is not another box filter — **the event layer has to see the detection confidence a track was built from**; a track of 0.15 boxes cannot be trusted for posture the way a 0.6 track is. Kaohsiung-cam04 also explodes 3,353 → 13,595 detections at 0.15, which is its open person-score investigation surfacing, so the threshold is a **per-camera** decision. Re-scope before spending a retrain. Original scope kept below: temporal-consistency tiers from `instances_all_*.json`, FP polygons from the 37 hotspots; **plus the night pass**: measure whether FP polygons + temporal consistency remove the IR ghost persons | new Gold/Silver training set; night `person` precision figure | Gold precision ≥95%, Silver ≥85% on a 300-frame sample; `after_hours_person` stays on the VLM trigger list **only if** the night figure passes |
| 5 | **L1 validation** — homography accuracy against WILDTRACK/MultiviewX ground-truth floor positions; tracking quality (ID switches) on in-domain clips watched end to end | position error in metres; ID-switch count per watched 10-minute clip | position error small enough that zone events land in the right zone; dwell/loiter durations survive — an ID switch mid-loiter resets the clock, so switches on the watched clips must be rare enough not to |
| 6 | **L3 end to end, one camera** | an event log readable against its video | events match what the clip shows |
| 7 | **shadow mode, one store** | alerts/camera/day + operator accept/reject log + **the first shrinkage reconciliation** | single-digit rate; rejects show an actionable pattern |

Throughput ceiling: already passed 2026-08-24 (1,552 fps TRT, batch 16) — but engine-only
and with terrain in slot 2, which is why step 3 re-takes it. Steps 1 and 2 are independent;
nothing after 6 starts until 6 passes, and 6 is not attempted before 5 — an L3 event log
over unvalidated tracks cannot be attributed when it is wrong.

## 7. Open questions — each blocks a specific step

1. **`stack` as a 5th detection class** invalidates checkpoint comparability, and a
   vocabulary change silently empties `analytics/events/zones.py:346`'s default class list
   — the stock-removal alarm stops firing without an exception. Decide at step 4.
2. ~~Night is unscoped~~ — **decided 2026-08-25: night is in v1, gated on a measurement.**
   Step 4 carries the night pass (14 IR ghost persons on one empty frame, measured);
   `after_hours_person` triggers only if the night precision figure passes.
3. **Pose distillation risk** (§2.2) — the claim the two-head architecture rests on.
   Answered by step 3's gate. The scope question above it is settled: **decided
   2026-08-25, fall/second-level behaviour is in v1**, so pose stays a per-frame L0 head.
4. ~~Delivery target undefined~~ — **decided 2026-08-25: v1 is 96 concurrent streams
   analysed on one RTX PRO 6000.** 96 × 15 fps = 1,440 frames/s is a **binding
   requirement, not headroom** — so gate 3's re-measure is end-to-end (decode, NMS,
   tracking, PCIe), and NVDEC capacity for 96 × h.264 joins the measurement list. At 96
   cameras the commissioning cache remains the right answer; §2's ~1,000-camera
   threshold is far away.
8. ~~No NVDEC backend exists on this box~~ — **decided 2026-08-25: PyNvVideoCodec.**
   Raised because gate 3 names NVDEC and the box had nothing to run it on (no
   PyNvVideoCodec, no DALI, no PyAV, and the only ffmpeg on PATH offering `vdpau`
   alone). Installed 2.2.1 — one wheel, no dependency of the running training job
   touched. Decode went from 63–69 streams to ~520 and stopped being the binding leg.
   `data/video.py` is still the CPU pipe: migrating the *serving* path to NVDEC is the
   work this decision authorises, and is not done.

9. **The pose run has no pose validation metric** — raised 2026-08-25. `metrics.jsonl`
   for `hydranet_retail_pose01` carries terrain IoU and detection mAP and nothing about
   pose, while `primary_metric` is `terrain_mIoU/site_seg03`. So a 60-epoch pose run
   selects `best.pt` on segmentation and leaves no per-epoch pose curve behind; the only
   checkpoints that survive are `best.pt` and `last.pt`. Gate 3's number has to be
   produced post hoc by `eval_student.py` over those two. The config change — a pose
   metric logged per epoch and named as the primary — cannot be made while the unit
   runs, and applies to the *next* run either way.

5. **Retail dashboard surface unscoped** — the numbers fall out of L1 free; what a store
   manager opens, at what cadence, is a product question. Blocks nothing before step 6.
   One modelling gap hides inside it: **store-level footfall needs cross-camera dedup**
   (overlapping views double-count a person). Single-store re-linking is in scope,
   cross-store is banned; the mechanism is unscoped.
6. ~~Fall/crouch training source~~ — **decided 2026-08-25: no in-store staging, ever.**
   Resolved in pose space: the temporal model reads keypoint sequences, so public 3D
   action data (NTU RGB+D fall/squat, CMU MoCap) projected to the measured camera pose
   replaces staged clips (§2.3). What stays open is only the sim-to-real transfer
   measurement, taken at step 6.
7. ~~Recommissioning ownership~~ — **decided 2026-08-25: a nightly closed-hours job**
   re-derives plate, masks, ROIs and re-anchors zones automatically; a human gets a
   morning accept/reject only when the diff exceeds a threshold, and a moved camera
   always escalates instead of auto-healing (§2.1). To build alongside step 2's tooling.
