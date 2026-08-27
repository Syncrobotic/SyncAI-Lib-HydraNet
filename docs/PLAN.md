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
| e | **zone polygons in metres** — the walkable outline, plus one **service zone** per fixture: the floor *beside* it, which is the part a shopper can stand on | no | ✅ `tools/commissioning/service_zones.py` — 71 zones across the 8 commissioned cameras, 2026-08-26. Fully automatic; the kind ships as `display` because which fixture is the till is a store fact, not a teacher's. See step 2 |
| f | **shelf ROI list** (from the fixture masks) — drives two-scale product inference (§2.2) | no | derived from c |
| g | **known-false-positive polygons** — derived from the box population, not drawn (hanging accessory walls, printed people) | accept / reject | ✅ derived, reviewed and written for all 8 — 20 polygons covering 7–78% of each camera's hallucination population (`fp_polygons.py`; verdict 2026-08-25: every candidate was a poster or merchandise) |
| h | store the plate for **tamper / fixture-change detection** — a knocked camera silently invalidates every metre downstream | no | ❌ |
| i | **3D scene / BEV render** for verification: a 1 m floor grid drawn on a real frame | checked by eye | partial — `syncai_bev3d/bev3d.py`, `syncai_bev3d/depth_scene.py` |

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

**State 2026-08-26: adopted.** `analytics/journey.py` is the first consumer, and `dwell`
and `events/zones.py` still take `Track` — moving *them* stays measured-first, but the
payload is no longer a contract nobody reads.

**`journeys()` — what L1 actually answers.** One `Journey` per track: an ordered list of
`Visit`s with durations, the `transitions` between them, the floor distance walked, and
the detector confidence the positions were built from. That is the "walked from A to B and
stood at C for how long" question, in metres, and it is the shape the retail dashboard
(open question 5) and step 6's event log both read. Four refusals are built in, each one a
measurement this project already paid for: two cameras' floors are not one route (`space`
must match, since every origin is under its own camera); a position `pixel_to_ground`
declined to measure is not a place and adds no distance; a *missed observation* does not
end a visit but a sustained observation outside does; and no clock means `seconds` is
`None`, never a nominal 30/1. It is a **track's** journey, not a customer's — the events
package measures 1,234 tracks in a 4.6-minute clip — so `track_id` is the field name and
nothing merges two journeys. Association is step 5.

**A defect the first real run found, and it produced numbers rather than errors.**
Taichung-cam01's intrinsics are fitted on 960×540 and its clips decode at 1920×1080, while
`clip_tracks.track_clip` returns boxes in the decoded stream's pixels. Feeding those
straight in put three shoppers at x 0.4–8.8 m — metres outside the commissioned walkable
polygon — with a 38 m walk in 60 s, and **nothing was NaN**. `camera_json.py`'s header had
always stated the contract ("pixels on the raw stream frame at `image_size_px`") and
nothing enforced it. `world_frame`/`world_frames` now take `source_size_px`: stating it
scales the points, omitting it is checked, and a point more than 1.5× outside the
calibrated canvas is refused with the mismatch named. Corrected, the same clip reads
x −1.9…1.0 m, z 1.6…5.1 m — inside the commissioned floor — and one shopper stands 20.2 s
in one 1.5 m cell and 23.4 s in the next (`assets/journey_cam01_frame150.png`).

**Confidence travels with the position.** `WorldObject` carries `score`, and
`Journey.score_p50` reports it, for the reason §4/step 4's sweep established: a track
built from 0.15 boxes is not the claim a 0.6 track is.

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
* **NTU RGB+D 60 is on disk and surveyed, 2026-08-26.** `datasets/_incoming/ntu60/NTU60_CS.npz`,
  the Kaggle mirror (`jarex616/ntu-rgb-d-60-skeleton-data-npz`) — a third-party
  redistribution, taken at the user's instruction while the ROSE Lab application is still
  queued; the licence *purpose* is satisfied (research, non-commercial) and the per-researcher
  agreement is not. 40,091 train / 16,487 test sequences, `(N, 300, 150)` float32 = 300
  frames × 2 bodies × 25 Kinect joints × 3D, one-hot over 60 balanced classes (271–276
  each in test). `tools/temporal/ntu_survey.py` is the instrument; `runs/ntu_survey01/`
  the numbers. **The projection route survives the file**: coordinates are *centred* —
  frame-0 spine base sits at (0.00, −0.295, 0.065) m in every sample, std ~0.03, so the
  Kinect's absolute distance is gone — but they are **not** canonically rotated (the
  facing direction still varies sample to sample) and **not** rescaled (torso length
  median **0.482 m**, std 0.026; shoulder width **0.336 m**, std 0.035 — between-subject
  anatomy, not a normalised constant). The floor is recoverable from the feet: a standing
  clip's foot height has std **4.4 mm**. So we supply the floor position and the camera,
  the file supplies the body, which is exactly what §2.3 asks for.
* **The licensed copy landed 2026-08-26 and it carries what the mirror had removed.**
  The ROSE Lab application came through, so `datasets/_incoming/ntu_rose/` now holds
  `nturgbd_skeletons_s001_to_s017.zip` (6.18 GB, **56,880** `.skeleton` files, NTU-60) and
  `nturgbd_skeletons_s018_to_s032.zip` (4.78 GB, 57,600 files, the NTU-120 extension) —
  skeletons only; the RGB is ~1.3 TB and this project reads keypoints, not pixels. The
  raw files are in **absolute Kinect camera coordinates**: spine base measures z = 2.84 to
  3.78 m across samples, where the Kaggle mirror had centred it to 0.065. Three things
  follow that the mirror could not give:
  - the person's real distance and pose *relative to a camera*, which is the quantity
    §2.3's projection route re-imposes and the one a view-normalised tensor has already
    thrown away;
  - **the same action from three cameras at once** — `S001C001P001R001A043`,
    `...C002...`, `...C003...` are one fall recorded simultaneously at z 3.78 / 3.51 /
    3.32 m — so a projection can be *validated* rather than assumed: project one view's
    skeleton through another view's parameters and check it reproduces that recording;
  - per joint, the Kinect's own `depthX/depthY` and `colorX/colorY`, which is a free
    ground truth for any projection code written here.
  25 joints per frame, as the survey found. `tools/temporal/ntu_survey.py` measured the
  mirror; the same instrument should be re-run against these before either is trained on.
* **And the survey refuted a per-frame feature before it was trained on.** In NTU's own
  ground-truth 3D, peak shoulder-to-hip angle from vertical: `A43 falling down` **74.5°**
  median with **38/40** clips over `events/pose.py`'s shipped 55° threshold — and
  `A06 pick up` **76.3°** with **35/40** over it. `A08 sit down` is 67.3°/30. The controls
  behave: `A42 staggering` 28.2°/3, `A27 jump up` 27.3°/1, `A01 drink water` 12.8°/1.
  **The torso angle does not separate a fall from a shopper reaching a bottom shelf** —
  it separates "bent" from "upright", which is a different question. What must carry the
  distinction is what happens *after* the peak: a fall stays down. `pose_posture_events`
  already encodes that as `sustained_seconds` plus the box-height cross-check, and the
  temporal model must be trained on the sequence for the same reason rather than on a
  per-frame posture score. **NTU has no crouch/squat class**; the nearest retail posture
  is `A06 pick up`, and it is the class a fall detector must not fire on.

### 4.4 Split discipline — binds every reported number

Split by **camera**, never by frame. A test camera never supplies a training frame. Every
rare class on ≥2 test cameras, or the number is written "not measured" with the camera
count. One fixed camera is one scene measured N times, not N samples.

### 4.5 Where human labour goes — all of it

4 calibration clicks per camera (§2.1b), accept/reject passes on teacher
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
| 2 | **commission all 48 cameras** — build the 4-click tool and zone tool, run the pipeline | per-camera `camera.json` + a 1 m grid rendered on one real frame per camera | the grid looks right **to my own eyes on every camera**. **In progress 2026-08-25**: geometry pass done for the 9 scale-measured cameras of the 2026-08-19 onboard scan — 8 `camera.json` shipped (grids eyeballed per camera, `runs/commission01/REVIEW.md`), Taichung-cam05 withdrawn (two furniture checks agree its cells are over-scaled). **Masks + 3D scenes done 2026-08-25 for all 8** (`tools/commissioning/masks_pass.py`, `scene3d.py`): structure masks, walkable, shelf ROIs written into each camera.json; commissioning-only 3D scenes with depth-measured fixture heights (tables 0.78–0.97 m fleet-wide). 5 mask sets clean, Tao-Hsin pair partial (white-fixture teacher blindness — iteration item), Kaohsiung-cam04 misses its pillar. Blocked on the physical world for the rest: 14 cameras need a per-store visual reference, 4 need mount-type triage, 7 Kaohsiung cameras need the person-score investigation (**answered per camera 2026-08-27, §7.12 — they do not behave as a group and one of them is not a people camera at all**); Taichung-cam05 needs its NVR stream setting. **Zones answered 2026-08-26, and not by drawing them.** `tools/commissioning/service_zones.py`: SAM 3 instances on the plate (the prompts `data/sam3_prompts.py` already measured), the floor pixels touching each instance, projected to metres, then every floor cell within 0.9 m of a fixture given to its nearest one. Fully automatic, and it corrects a mistake worth recording -- two earlier attempts chased fixture *footprints*, hit the fact that one camera cannot see behind a counter, and concluded a human must draw the far edge. **A footprint is a region no shopper can ever occupy**, so as a zone it never fires; what a zone is for is the floor *beside* the fixture, which is entirely in view. **Applied to all 8 commissioned cameras 2026-08-26: 71 service zones**, 6-13 per camera, written into `camera.json` beside the walkable outline (`runs/service_zones01/`, per-camera renders eyeballed). On a real 60 s clip of Taichung-cam01 the three tracks read `fixture_02, 60.0 s, 298/300` (the shopper at the left counter), `fixture_06, 60.0 s, 300/300` (the one at the right-hand shelving) and a route for the member of staff who walks. Two defects were found and fixed by that measurement, not by inspection: an instance mask can swallow the floor in front of it (`merchandise rack` at 0.90 returned a fifth of the frame), so an instance is cut against the floor mask before its contact ring is taken; and **a counter has two sides** -- keeping only the largest connected piece of a fixture's service floor discarded the staff aisle every run, which is exactly where the track that fell outside every zone was standing. Caveat inherited: Taichung-cam10's metres are 1.21x too large (§7.10), so its 48.4 m2 of zone area is inflated by the same factor. Two gaps stated: the right-hand accessory wall got no zone (and a track stands there), and prompt-to-kind attribution is weak -- which fixture is the till is a store fact, not a 0.745 from a shape-shaped prompt. Remaining tools: the 4-point ground-calibration click tool (§2.1b) and the tamper reference (§2.1h) — the last two ❌ in that table |
| 3 | **pose head resident** — **IN PROGRESS 2026-08-25**: the box-conditioned P3 head is built and tested (`ef9b53c`), ViTPose labelled the Gold boxes (train 13,930 / val 8,479 / test 2,862 images), and `hydranet_retail_pose01` is training under systemd (60 epochs, pose loss 0.47 → 0.10 by epoch 2). **Its run directory is `runs/hydranet_retail_security_b03_cw_xl-20260825-162131`** — the pose config inherited `output_dir` from the security run and the trainer appended a timestamp rather than overwrite it, so the directory is named for a different model and only the `config.yaml` inside it is authoritative. `tools/pose/eval_student.py` had never been executed (it died on `torch.from_numpy` over an already-tensor) — **fixed and run 2026-08-25** (`2fb6f0c`), and `--render` now writes per-person crops with both skeletons, because a gate decided by eye has to be lookable-at. **Throughput: `scripts/bench_e2e.py` (`65ce3b5`) is the end-to-end instrument** and it refuses a busy GPU. First legs, measured while training shared the box so they are lower bounds: host NMS 450 fps/thread (3.2 of 24 cores at target — not a risk), tracker 7,694 fps/thread (0.2 cores), **CPU decode saturates at ~950–1,030 fps = 63–69 streams against the 96 required**. **Decode resolved 2026-08-25**: the box had no NVDEC path at all; PyNvVideoCodec 2.2.1 installed (user's call, §7.8) and wired into the harness with `usedevicememory=True`, so frames never cross PCIe. Measured with training still on the card, i.e. lower bounds: **NVDEC saturates ~7,800 fps ≈ 520 streams at 15 fps, 5.4× the target**, against the CPU pipe's 63–69 streams. Decode is no longer the binding leg. **Still open: the exporter drops the pose head** — `cli/export_onnx.py`'s `ExportWrapper` emits segmentation and detection outputs only, so an engine built today measures a model without pose and reports it as the model with pose. Must be fixed before the engine leg is measured; blocked until the training unit stops (never edit `src/` under a running job). **Training finished 2026-08-25 20:01**, 60/60 epochs, final val `coco_person` mAP 0.2022 / mAP@50 0.3891. The checkpoint trap fired exactly as forecast: `primary_metric` is `terrain_mIoU/site_seg03`, so `best.pt` is **epoch 15**. Both candidates run on the full test split (2,862 images / 6,360 persons / 93,967 judged joints):

| checkpoint | PCK@0.2h | L2 p50 | mean | p90 |
|---|---|---|---|---|
| `best.pt` (E15, terrain-selected) | 0.840 | 13.8 px | 35.4 | 96.3 |
| **`last.pt` (E60)** | **0.898** | **9.3 px** | **25.1** | **67.7** |

**Use `last.pt`.** Taking the handoff's `best.pt` would have gated on a figure 5.8 points lower with 48% more error. Trajectory E15 → E29 → E60 is 0.840 → 0.882 → 0.898, still rising at the end. The run logs **no pose validation metric at all**, so there is no per-epoch curve and these two checkpoints are the only candidates that survive a run — fix the config's `primary_metric` before the next one (§7.9).

**pose02 FINISHED 2026-08-26 19:19, and the accuracy leg of this gate is met.** 120/120 epochs. On the full test split (2,862 images / 6,360 persons / 93,967 judged joints) `eval_student.py` reads **PCK@0.2h 0.915, L2 p50 7.7 px** (mean 21.5, p90 54.7), against pose01's `last.pt` at 0.898 / 9.3 px. **`best.pt` and `last.pt` return identical figures** this time -- the run selected on `pose_PCK@0.2h`, which picked epoch 119 -- so pose01's trap, where a terrain-selected checkpoint was the run's *worst* pose model, is structurally gone rather than avoided by hand. **The curve is flat at the end**: 0.9339, 0.9344, 0.9345, 0.9345, 0.9345 over the last five epochs on the val prefix, so 120 epochs was enough, which is precisely the question pose01 could not answer about itself. The cost of selecting on pose is on the record in `selection.json`: terrain mIoU was 0.698 at epoch 17 and is 0.637 at the selected epoch. Superseded note: **pose02 was running as of 2026-08-26 14:08** — `hydranet-pose02.service`, `runs/hydranet_retail_pose02`, 120 epochs, ~7 min/epoch. The 2026-08-25 overnight attempt **never started**: the trainer refused a dirty tree over another session's uncommitted `analytics/world.py`, the two eval steps behind it then failed on a checkpoint that was never written, and the card sat idle for 17 hours. It selects on `pose_PCK@0.2h` (§7.9), so this run leaves a per-epoch pose curve and a `best.pt` chosen by the head it exists to train.

**Throughput re-measured 2026-08-26 on an idle card — the gate PASSES at the shipped
canvas, and the margin is 3.7%.** The 1,324 f/s below was pose01's weights *with an RL
neighbour on the card*; with pose02's `last.pt`, batch 16, fp16 and nothing else on the
GPU the engine measures **1,494 f/s = 100 streams**, against 1,440 required. So the leg
that was 8% short was 8% short of a *shared* card. Read the two together rather than
either alone: a co-tenant costs 11%, and 3.7% of headroom does not survive one.

Found on the way, and it invalidated the 2026-08-25 resolution sweep: `bench_trt.py`
read the batch from the **filename** (`_b(\d+)`, else 1). That sweep's files were named
`res_640x1120.onnx` and exported at batch 16, so every row divided the true throughput by
sixteen and `meets_target_compute` compared the sixteenth against 1,440 — recording a
**False** for a 576x1008 engine that clears the target. It now reads the batch from the
graph and refuses a dynamic-batch one rather than guessing.

**The resolution trade is measured now, both halves, on the same weights** (`runs/res_trade01/`,
`eval_student.py --input-size`). It has been carried as "an unmeasured accuracy cost" since
the target was set:

| input | engine f/s | streams | PCK@0.2h | vs shipped | L2 p50 |
|---|---|---|---|---|---|
| **640x1120 (shipped)** | **1,494** | **100** | **0.915** | — | 7.7 px |
| 576x1008 | 1,806 | 120 | 0.911 | −0.004 | 8.3 px |
| 512x896 | 2,314 | 154 | 0.908 | −0.007 | 9.2 px |
| 448x784 | 3,010 | 201 | 0.897 | −0.018 | 10.7 px |

Every canvas clears the target, so none of them is needed to pass — they are headroom, and
now priced. The bottom row is the one to keep in mind: **448x784 doubles the throughput
for 0.018 PCK, and 0.897 is exactly the accuracy pose01's `last.pt` was about to ship
yesterday.** Superseded: **Throughput, end to end, 2026-08-25 — the gate MISSES.** `cli/export_onnx.py` was
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
re-opening §7.4's 96 **Gate 3's third condition met 2026-08-26, and one thing it does not cover is worse than
the things it does** (`runs/pose_events03/`, pose02 `last.pt`, the eight sweep clips, 24
minutes). `reach_to_shelf_events` had never run outside a test; wired into
`pose_overlay.py` it produces **280 reach, 4 crouch, 3 fall**. Verified the way this gate
says — by looking:

* **`reach_to_shelf` fires correctly.** Two firing frames read by eye, both right, wrist
  genuinely over the counter at fixture fractions 0.66 and 1.00 against 0.35.
* **`crouch` fires correctly.** Taichung-cam01 frame 810, ratio 0.30 against a 0.60
  threshold: a member of staff folded down at a low cabinet, exactly what the type means.
* **So the gate passes as written** — and the alert is not usable, for a reason the clip
  makes plain. 280 reaches is 11.7 a minute, 36.7 on Kaohsiung-cam04, and every person in
  the Taichung-cam01 clip wears the shop's blue polo: there is no customer in it. Staff
  working a till reach over a fixture every few seconds. `staff/customer` is already
  listed in §4.3 as necessarily in-domain, and this is the first measurement of what a
  specific output costs without it.
* **`fall` is not in this gate's wording and it is the one that would wake somebody.**
  Three in 24 minutes of a phone shop. Both Kaohsiung-cam04 cases were read by eye and
  **both are false**, with the cause visible in the frame and predicted in
  `models/heads/pose.py` before any of it ran: "two people whose boxes overlap can steal
  each other's peaks inside the intersection". At a crowded counter the overlapping boxes
  tangle their skeletons into a horizontal torso. The same frames show the detector
  finding **3 people in a frame holding about ten** — that camera's open person-score
  investigation, in a second symptom. A safety alert at one false alarm per eight minutes
  does not ship, and the fix is not a threshold: it is the crop-stage fallback
  `models/heads/pose.py` reserved for exactly this case.

| keypoint error vs ViTPose; throughput re-measured **end to end — NVDEC decode → engine → host NMS → tracker** — with pose in the slot and the ROI path at its 0.2–1 fps cadence | `reach_to_shelf` and `crouch` fire correctly on a watched clip; fps ≥ 1,440 stands end to end, not engine-only — 96 streams is a requirement now (§7.4), and decode/NMS/PCIe were named the real risk when the target was set |
| 4 | **detection uplift at zero GPU** — ~~temporal-consistency tiers~~ **REFUTED for `person`, 2026-08-26, measured**: `instances_all_train.json` is not a pool of unused good labels, it is the same Grounding DINO teacher dumped at 0.10 instead of 0.35 — median score 0.149, and `instances_train` is simply that pool thresholded (48,005 vs 47,465 at 0.35). Worse, persistence is the wrong signal here and it is wrong in the dangerous direction: score vs run-length correlation is **+0.006**, the 0.10–0.20 band moves a median **0.3 px per frame**, and **41,160 sub-threshold boxes persist ≥8 frames** — posters, mannequins and hanging packets are the most persistent things in a store. A consistency tier would promote exactly them. The measured detection problem is **recall, not label count**: on one clip the trunk's own dense head marks **1.30× as many people as the box head returns** (146 blobs vs 112 boxes over 60 frames) while scoring `person` IoU 0.885 against detection mAP@50 0.302 — the features find people the box head cannot emit. **Re-scoped and then measured on the fleet, 2026-08-26**: `serving.decode.confirm_with_dense` admits low-scoring boxes where the dense head puts person pixels under them. Three arms over the same eight clips — A shipped 0.35, B threshold-only 0.15, C 0.15 + confirmation:

| arm | person dets | tracks | boxes dropped | fall | crouch |
|---|---|---|---|---|---|
| A 0.35 | 11,363 | 202 | 0 | 1 | 1 |
| B 0.15 | 27,715 | 569 | 0 | 4 | 5 |
| C 0.15 + confirm | 25,713 | 520 | 2,002 | 4 | 5 |

**The recall gain is real and the confirmation does not protect the event layer.** C drops 12–14% of boxes and produces *exactly* B's events, on the whole fleet and again with Kaohsiung-cam04 excluded (2 fall / 3 crouch both arms). The boxes it removes were never the ones firing events: the extra events come from real people detected at low confidence, whose keypoints are noisier, making more posture runs. So the next mechanism is not another box filter — **the event layer has to see the detection confidence a track was built from**; a track of 0.15 boxes cannot be trusted for posture the way a 0.6 track is. Kaohsiung-cam04 also explodes 3,353 → 13,595 detections at 0.15, which is its open person-score investigation surfacing, so the threshold is a **per-camera** decision. Re-scope before spending a retrain. Original scope kept below: temporal-consistency tiers from `instances_all_*.json`, FP polygons from the 37 hotspots; **the night pass, done 2026-08-26** (`scripts/night_ghosts.py`, `runs/night_ghosts01/`): the shipped detector produces **zero `person` boxes in 2,250 frames of an empty shop at 23:58 store-local, across 15 cameras**, at the shipped 0.35. The ghosts were always the *teachers*' -- SAM 3's `person` prompt returned 14 hanging accessory packets as people on Taichung-cam09, and Grounding DINO at 0.35 fired on an empty store on 13 of 42 cameras -- and the student did not inherit either. **Taichung-cam09 itself returns zero.** So the false-positive rate that `after_hours_person` had to be quoted against, and which did not exist, is now measured and is 0. Two caveats, because the sample is what it is: only 15 of 48 cameras have a 23:58 clip on this box, and only 2 of the 11 cameras where Grounding DINO fired at night are among them (both zero). `data/night_person.py`'s static-plate veto stays where it is -- it protects the *teacher's* output, and the serving path turns out not to need it. Superseded: measure whether FP polygons + temporal consistency remove the IR ghost persons | new Gold/Silver training set; night `person` precision figure | Gold precision ≥95%, Silver ≥85% on a 300-frame sample; `after_hours_person` stays on the VLM trigger list **only if** the night figure passes |
| 5 | **L1 validation** — homography accuracy against WILDTRACK/MultiviewX ground-truth floor positions; tracking quality (ID switches) on in-domain clips watched end to end. **First fleet measurement of what the layer produces, 2026-08-26** (`scripts/site_journeys.py`, `runs/journeys01/`): 8 cameras, 24 minutes, the same clips as the threshold sweep — **11,363 person detections and 202 confirmed tracks, reproducing the sweep's arm A exactly**, so the chain is consistent with what was already measured. Of those 202 tracks **86 (43%) ever enter a zone**, and the split is a length split: a track that reaches a zone has a median 26 observations and 5.5 s of span, one that never does has 9.5 and 2.4 s. Across 197 visits the **median is 3.2 s, p90 13.8 s, and only 8 visits in the whole 24 minutes last 30 s or more** (longest 105.8 s, Taichung-cam01). Visits are well observed once they exist — seen fraction p50 1.00, p10 0.79 — so this is not a detection gap: **the tracks are too short to be visits**, which is the 1,234-tracks-per-clip fragmentation stated as a retail number for the first time. That is what this step exists to fix, and it now has a baseline to move. **And the cause is not fleet-wide** (`scripts/track_endings.py`, `runs/endings03/`, same clips, no labels needed). What holds: an ending is an **exit** when the last box sits against a frame edge, and otherwise the track died mid-view -- a shopper does not vanish from a shop floor. That split is rule-independent and did not move across three runs: **80 of 191 endings are exits (42%), so 58% die mid-view.** Per camera it is not uniform, and that is the finding -- **Tao-Hsin-cam03 83% and Kaohsiung-cam04 78%, against Taichung-cam04 13%, Taichung-cam10 14% and Taichung-cam11 17%**, with the same tracker on all of them. Two cameras carry this, and one is the camera whose person-score investigation has been open since step 2. **Answered 2026-08-27, §7.11: 86% of these deaths have a box still on the person at a median 0.338 against the 0.35 threshold, so fragmentation is a threshold problem before it is a tracker problem.** **What does NOT hold is the split of those mid-view deaths into tracker-lost and detector-gone**, and an earlier revision of this plan asserted it: three matching rules gave 72/39, 72/39 and 101/10 on identical endings while `exit` stayed at 80 every time. A 1 m radius matched the *neighbouring* shopper -- its reappearances were a median 0.71 m away one frame later, 3.5 m/s -- and a 0.35 m per frame speed limit fixed that end and opened the other, reaching 3.5 m at gap 10. Distinguishing "the same person, a metre on" from "a different person, a metre away" is what an appearance model is for; `reid_metrics.cmc_map` is its metric, and nothing should be concluded from that split until something can tell two shoppers apart. `lost_detail` keeps every gap, IoU and floor distance so a later rule can be tried against the same endings. `idf1`/`id_switches` are *not* the way in here -- both need ground-truth tracks and `reid_metrics.py` records that no labelled site clip exists, which is the human cost SS4 refuses by default | position error in metres; ID-switch count per watched 10-minute clip | position error small enough that zone events land in the right zone; dwell/loiter durations survive — an ID switch mid-loiter resets the clock, so switches on the watched clips must be rare enough not to |
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
3. ~~Pose distillation risk~~ (§2.2) — **closed 2026-08-26 by step 3's gate.** The
   student agrees with ViTPose at **PCK@0.2h 0.915 / L2 p50 7.7 px** on the full test
   split, and both consumers were verified firing correctly on real footage by eye:
   `crouch` on a member of staff folded at a low cabinet, `reach_to_shelf` on a wrist over
   a counter. The two-head architecture's claim holds. What did *not* hold is `fall`,
   and that is a grouping failure in crowds rather than a distillation one -- see gate 3.
   Original: the claim the two-head architecture rests on. Answered by step 3's gate. The scope question above it is settled: **decided
   2026-08-25, fall/second-level behaviour is in v1**, so pose stays a per-frame L0 head.
4. ~~Delivery target undefined~~ — **decided 2026-08-25, and the frame rate revised
   2026-08-26: v1 is 96 concurrent streams on one RTX PRO 6000, analysed at 5 fps.**
   **96 × 5 = 480 frames/s**, not the 1,440 first written down, and the revision is not a
   relaxation of ambition — it is the target catching up with the system. Every
   measurement this project has made of its own analytics runs at 5 fps: the journey run,
   the ending analysis, the pose events, `retail_flow`, `site_events`. Nothing downstream
   consumes 15.

   **The card is exclusive and the headroom is spoken for.** The engine measures 1,494
   f/s at the shipped canvas, so analytics needs **32% of it** and the rest is budgeted
   for a **VLM on the same card** (§4.3's `staff/customer` voting, the `after_hours_person`
   trigger list, and whatever adjudicates a zone's kind). That is why no resolution
   reduction is taken: 448x784 would double analytics throughput nobody needs and cost
   0.018 PCK. Superseded: 96 × 15 fps = 1,440 frames/s was a **binding
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

10. ~~Taichung-cam10's metres are ~1.21x too large~~ — **decided 2026-08-26: do not
   re-pin.** Re-pinning would invalidate every metre already reported for that camera; the
   cost of not re-pinning is that **cam10 can never appear in a table beside another
   camera** — its areas, path lengths and speeds are inflated by 1.21x and its durations
   are not. Any fleet aggregate must exclude it or scale it explicitly. Original:
   **Taichung-cam10's metres are ~1.21x too large.** Recovered stature median 1.96 m with
   nobody under 1.72; `--metre-scale 0.8824` renders true metres. `camera.json` is
   **untouched** because re-pinning changes every metre already reported for that camera --
   its 13 service zones, their 48.4 m2 of area, every path length and every speed in
   `runs/journeys01`. Durations are unaffected. Raised 2026-08-26 and still undecided; it
   blocks nothing today and it invalidates a comparison the moment two cameras' metres are
   put in one table.

9. ~~The pose run has no pose validation metric~~ — **closed 2026-08-26** (`cf1ddfb`).
   Raised because `hydranet_retail_pose01`'s `metrics.jsonl` carried terrain IoU and
   detection mAP and nothing about pose, while `primary_metric` was
   `terrain_mIoU/site_seg03` — so a 60-epoch pose run selected `best.pt` on segmentation
   and left no per-epoch pose curve, and the selected checkpoint turned out to be the
   run's *worst* pose model (PCK 0.840 against `last.pt`'s 0.898). Validation now emits
   `pose_PCK@0.2h`, `pose_L2_p50` and `pose_L2_p90` per pose head, computed the way gate
   3 computes them and decoded from the **teacher's own boxes** — pose is a
   box-conditioned head, so scoring it against predicted boxes would move the pose curve
   whenever detection moved. That required `PoseKeypointsDataset` to emit
   `targets["boxes"]`, which made keypoints and boxes parallel arrays through the
   transforms and surfaced a real bug: `_paste` dropped a cropped-away person's box and
   kept their skeleton, after which person i's heatmap window was read from person i+1's
   box. `train.pose_val_max_persons` (default 4,000 of the val split's 22,241) caps the
   per-epoch cost at ~7 s. Sanity check against pose01's E60 weights: PCK 0.925 / p50
   8.8 px on a 1,501-person val prefix, next to `eval_student.py`'s 0.898 / 9.3 px on
   test. **The number that gates is still `eval_student.py` over the whole test split**;
   this is the curve that says when to stop.

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

11. **The Kaohsiung person-score investigation is two investigations, and the fix is not
   on the inference side.** Opened 2026-08-27 from step 2's blocker list and step 5's
   "two cameras carry the fragmentation". The name was wrong on both counts: it is not
   one fault, and on the camera it is named after the scores are fine.

   **The split.** Measured on *solo frames* — frames where the dense head, the head not
   under test, finds exactly one person-sized blob — over all four clips of each camera
   (`scripts/person_score_probe.py solo`, 4 clips x 300 frames at 1 fps):

   | camera | solo best-box score p50 | solo ≥0.35 | mid-view track death (§6 step 5) |
   |---|---|---|---|
   | Taichung-cam04 | 0.55 | 96% | 13% |
   | Taichung-cam01 | 0.54 | 95% | — |
   | **Kaohsiung-cam04** | **0.50** | **99%** | **78%** |
   | Taichung-cam07 | 0.43 | 80% | — |
   | Tao-Hsin-cam04 | 0.42 | 80% | — |
   | Taichung-cam11 | 0.39 | 63% | 17% |
   | Taichung-cam10 | 0.27 | 43% | 14% |
   | **Tao-Hsin-cam03** | **0.26** | **32%** | **83%** |

   **Kaohsiung-cam04 is one of the fleet's best cameras with one person in it** and
   collapses to 0.19 / 20% in its evening counter crowd. **Tao-Hsin-cam03 fails with one
   person in an empty shop**, and it is the camera whose commissioning masks already
   record the white-fixture / plank-floor teacher blindness. One fix applied to both
   would miss cam03's fault entirely.

   **Crowding is the cause on Kaohsiung-cam04, measured with the gradient it needs.**
   Pooling that camera's quiet, mid and busy clips and binning by dense person pixels
   (`scripts/person_score_probe.py density`, 3 clips x 400 frames at 2 fps): recall at the shipped
   0.35 goes **1 detection at ~1 person → 2 at ~7 → 4 at ~13**, while at 0.15 it goes
   **1 → 5 → 15**. The boxes are present the whole way; they sit between 0.15 and 0.35.
   The **top score never moves** (0.55 / 0.44 / 0.50 across the bands), so this is not a
   frame-wide depression — one or two people per cluster keep their score and the rest
   are demoted. This is also *why* §6 step 4's arm B recovered people and made the event
   layer worse: at 0.15 the demoted people and the duplicate fragments arrive together.

   **The demotion is in the classifier, not in centerness, and not in NMS**
   (`scripts/person_score_probe.py factors`, `score = sigmoid(cls) * sigmoid(centerness)` split at
   every location the dense head calls person, 100 frames each):

   | | cls p90 | centerness p50/p90 | ≥0.35 at NMS 0.6 / 0.75 / 0.9 |
   |---|---|---|---|
   | Kaohsiung-cam04 quiet | 0.61 | 0.38 / 0.62 | 1 / 1 / 1 |
   | Kaohsiung-cam04 crowd | **0.44** | 0.39 / 0.59 | 4 / 4 / 5 |
   | Tao-Hsin-cam03 solo | **0.45** | 0.38 / 0.58 | 2 / 2 / 3 |
   | Taichung-cam01 | 0.67 | 0.38 / 0.58 | 3 / 3 / 3 |

   Centerness is **identical in all four** — quiet, crowded, good camera, bad camera —
   so the "a crowd steals a shopper's central locations and centerness punishes what is
   left" hypothesis is refuted. Opening the NMS IoU from 0.6 to 0.9 moves the crowded
   count from 4 to 5, so NMS is not suppressing the crowd either. **Both inference-side
   fixes are closed.** What moves is `cls`: the network is less sure those pixels are a
   person when there are many people.

   **The candidate cause, and it is a training-time one.** FCOS assigns a location to the
   smallest box containing it, so a shopper standing in front takes the locations of the
   shopper behind. Measured on the teacher labels, Kaohsiung-cam04 loses **a median 10.1%
   of a person's area to smaller boxes (every other camera: 0.0%) and 22.9% of its people
   lose more than 25%** (others 6.5–12.5%); person-on-person max IoU p90 is 0.302 against
   0.192–0.257. The same appearance is therefore labelled positive in one frame and
   negative in the next, and an unsure classifier is what that trains. Centerness is
   untouched because it is regressed only on the positives that survived — which is
   exactly the asymmetry the table above shows. **Untested.** The fix it implies is a
   crowd-aware assigner (ATSS/OTA) and that is a retrain; **no run has been started.**

   **Refuted on the way, each single-variable, so none of these is worth re-opening:**
   *stream spec* — h264 / 1920x1080 / 30 fps on all 8, cam04 at 583 kb/s against a fleet
   range of 453–600; *roll* — de-rotating the frame by the camera's own +12.9° gives
   ≥0.35 counts of 3→3, 3→2, 3→3; *teacher quality* — cam04 has the fleet's **largest**
   teacher boxes (p50 425 px against 168–306) and its second-highest teacher score
   (0.616), and the teacher's score *rises* with person size fleet-wide (0.42 at <60 px
   to 0.65 at 460–660); *training representation* — cam04 is in the training set with
   10,744 person boxes over 3,198 images, second most in the fleet; *time-of-day
   coverage* — 1,092 of those are 19h store-local and their labels hold p90 9 and max 14
   people, so crowded frames were labelled and trained on.

   **And one measurement that answered nothing, recorded because it looks like it did.**
   Binning the *evening clip's own* frames by crowd size is null: that clip never holds
   fewer than about twelve people, its person pixels move only 120k→213k, and top score
   is flat at 0.46–0.51 across every band. It cannot be read as "crowding does not
   matter" — the independent variable did not move. The pooled three-clip run above is
   the one that has a gradient.

   **Tao-Hsin-cam03, and a fleet-wide effect found while chasing it.** Its cls p90 of
   0.45 with one person in an empty shop is the same factor failing for a different
   reason. Looking at one frame gave the reason: it holds a **0.60** shopper in the open
   aisle and a **0.06 / 0.07 / 0.10** pair at the right-hand counter, all four segmented
   cleanly by the dense head. So the fault is positional, and the position is a counter.

   Split without any human labels, using the terrain head's own `floor` / `fixture` /
   `person` classes to say what the strip under each person blob stands on — feet on the
   floor, cut off by a fixture, or standing behind someone (`scripts/person_score_probe.py standing`,
   6 cameras x 4 clips x 200 frames at 1 fps). The student's terrain was rendered and
   eyeballed on the deciding frame first: it does call those white counters `fixture`,
   so the bins are real on the camera that matters:

   | camera | grounded ≥0.35 | cut by a fixture ≥0.35 | drop |
   |---|---|---|---|
   | Taichung-cam11 | 0.98 (n 372) | **0.14** (n 222) | −0.84 |
   | **Tao-Hsin-cam03** | 0.64 (n 544) | **0.16** (n 196) | −0.48 |
   | Taichung-cam10 | 0.81 (n 1043) | 0.59 (n 694) | −0.22 |
   | Taichung-cam01 | 0.95 (n 839) | 0.75 (n 445) | −0.20 |
   | Kaohsiung-cam04 | 0.92 (n 766) | 0.75 (n 173) | −0.17 |
   | Taichung-cam04 | 0.62 (n 867) | 0.50 (n 911) | −0.12 |

   **A person a fixture cuts off scores worse on every camera in the fleet, six for six**,
   and that is new. **Tao-Hsin-cam03's failure lives in that bin**: 0.64 grounded against
   0.16 cut, which is what drags its solo median to 0.26.

   **It does not unify the two cameras, and two things refuse it separately.**
   Taichung-cam11 has the *steepest* cut penalty in the fleet and only **17%** mid-view
   track death, so the penalty does not predict fragmentation. And Tao-Hsin-cam03 is only
   0.64 even when grounded, against 0.95 and 0.98 on cam01 and cam11 — a second, smaller
   deficit that has nothing to do with counters and is **not explained**.

   **What this measurement cannot see, stated because its table looks complete.** It cuts
   people apart with connected components, so a crowd merges into one blob and counts
   once: Kaohsiung-cam04 contributes 992 people over 800 frames, about 1.2 per frame,
   while its evening clip holds twelve. **It says nothing about that camera's crowd**, and
   its comfortable 0.92 / 0.75 there is a statement about its quiet frames only.

   Tao-Hsin-cam03, not Kaohsiung-cam04, is the fleet's worst mid-view track death at 83%.

   **THE PAYOFF: 86% of tracks that die mid-view die with a box still on the person, a
   median 0.338 against a 0.35 threshold.** Everything above measures *scores*; step 5
   cares about *fragmentation*, and the two were not connected -- Taichung-cam11 has the
   steepest cut penalty in the fleet and the second-lowest mid-view death, so the score
   findings did not predict it. `scripts/track_endings.py` now answers the link with a
   **witness pass** (`runs/endings04/`, same eight clips, same weights, same settings):
   pass one is untouched and reproduces **exit 80 / lost 101 / gone 10 exactly**, and a
   second pass re-runs the model at 0.03 over the three frames after each mid-view death.

   This is deliberately **not** a fourth matching rule. `lost`/`gone` asks *who*
   reappeared and that is what moved 72/39 to 101/10; this asks only whether **anybody**
   was still standing where the track died, which needs no identity:

   | verdict | n | share | what it means |
   |---|---|---|---|
   | **demoted** | **95** | **86%** | a box is there, below the shipped threshold |
   | available | 15 | 14% | a box at ≥0.35 overlapped by the **tracker's own** IoU rule and was not associated — a tracker failure |
   | vacated | 1 | 1% | the dense head sees nobody either |
   | boxless | **0** | 0% | the box head is never wholly blind |

   **110 of 111 mid-view deaths had a person box sitting on the spot.** The premise this
   whole measurement rests on — a shopper does not vanish from a shop floor — is now
   measured rather than assumed.

   **The number that matters is how far below the threshold those boxes are, not that
   they are below it** (that part is definitional): **p10 0.304, p50 0.338, p90 0.367**,
   and 95% of them are at or above 0.25. A detection failure would put this distribution
   at 0.05. It sits in a band 0.05 wide, immediately under the threshold. And the score
   quoted is the **maximum over three frames**, so these people held ~0.34 for at least
   three frames — a stable just-below state, not a one-frame dip. The boxes are the
   person's own, not a neighbour's: **best IoU p50 0.90, 85% at ≥0.7**.

   **Per camera the two faults separate again.** Kaohsiung-cam04: **63 of its 66 mid-view
   deaths are demoted**, 3 available — almost purely a score problem, exactly as its crowd
   measurement said. Tao-Hsin-cam03: 17 demoted **and 12 available**, which is 12 of the
   fleet's 15 tracker failures on one camera. Its second deficit, unexplained above, has a
   shape now: it is losing tracks the tracker could have kept.

   **What the witness pass cannot show.** Its dense-head check confirms a person in only
   28% of demoted cases, and that is the connected-component limit again rather than an
   absence: in a crowd the components merge and the merged bounding box cannot reach IoU
   0.2 against one shopper. It does not weaken the verdict — `demoted` is decided on the
   box, and only the single `vacated` case turned on the dense signal at all.

   **AND THE FIX MAY ALREADY BE IN THE TREE, UNUSED.** If tracks die at 0.338 against a
   0.35 threshold, what is needed is not a lower birth threshold — that arm was measured
   on 2026-08-26 and made the event layer worse — but a **survival** band under an
   unchanged birth threshold. `analytics/bytetrack.py` is exactly that (births at the high
   band, survives down to the low one) and its own header says the comparison against the
   shipped `analytics/tracker.py` **has never been run on this footage**. It has now
   (`runs/endings05/`, `scripts/track_endings.py --two-stage`, same eight clips, same
   weights, births still at 0.35, survival to 0.20):

   | | shipped `tracker.py` | `bytetrack` two-stage |
   |---|---|---|
   | tracks | 202 | **100** |
   | endings judged | 191 | 83 |
   | **mid-view deaths** | **111 (58%)** | **38 (46%)** |
   | Kaohsiung-cam04 tracks | 87 | 29 |
   | median exit-track length, Kaohsiung-cam04 | 15 frames | **61** |
   | median exit-track length, Taichung-cam04 | 11 | **40** |
   | median exit-track length, Tao-Hsin-cam03 | 10 | **20** |
   | median exit-track length, Taichung-cam01 | 38 | **492** |

   **Half the tracks and each 2–4× longer is the signature of fragments being joined**;
   people being dropped would halve the tracks and leave the lengths where they were.

   **It moves two things at once and the comparison must not be quoted as one.**
   `bytetrack` brings a Kalman filter as well as the band, and `tracker.py` refuses that
   filter on stated grounds — no measured noise model exists for this footage. So this is
   the second tracker, not the first with hysteresis added.

   **And no metric here can see an ID switch**, which produces the same shape: fewer,
   longer tracks. `reid_metrics.py` records that no labelled site clip exists, so it was
   settled the way everything else today was — by looking. The three longest
   Taichung-cam01 tracks, sampled twelve times each across their lives:

   * **711 observations (142 s): one person in all twelve.** Same blue polo, grey
     wide-leg trousers, lanyard, black waist bag. The identity genuinely held.
   * 551 observations: the same man in eleven of twelve, and **the twelfth is a different
     person** — somebody crossed in front and the box went with them.
   * 433 observations: the same person in eleven of twelve, and **the twelfth is an empty
     counter** — the box coasted off after they left.

   **So the joining is real and the tails are not.** The Kalman coasts past a departure,
   onto a neighbour or onto a fixture. That is not free: **a track that coasts forty
   frames onto a counter inflates dwell**, and dwell is a number the retail half sells.
   Adopting two-stage tracking therefore trades fragmentation against dwell accuracy and
   is a decision, not a patch. **Not adopted; nothing in the serving path changed.**

12. **The seven blocked Kaohsiung cameras split, and the block was never a group
   property.** §7.11 answered what the person-score investigation was; this answers what
   it *blocks*, which is `runs/commission01/REVIEW.md`'s "the Kaohsiung person-score
   investigation before its 7 cameras can use the person-height path". Measured
   2026-08-27 with `scripts/person_score_probe.py solo` over all four clips of each,
   against Kaohsiung-cam04's 0.50 / 99%:

   | camera | solo frames | score p50 | ≥0.35 | stream |
   |---|---|---|---|---|
   | Kaohsiung-cam03 | 245 | 0.41 | **81%** | **704x480** |
   | Kaohsiung-cam08 | 48 | 0.54 | **79%** | 1080p |
   | Kaohsiung-cam12 | 58 | 0.38 | 59% | 1080p |
   | Kaohsiung-cam05 | 276 | 0.31 | 43% | 1080p |
   | Kaohsiung-cam06 | 408 | 0.27 | **29%** | 1080p |
   | Kaohsiung-cam11 | 139 | 0.32 | **22%** | 1080p |
   | Kaohsiung-cam07 | **0** | — | — | 1080p |

   **None of the seven behaves like cam04**, so a single verdict for the group would have
   been wrong whichever way it went. cam03 and cam08 clear the threshold on four fifths of
   their solo shoppers and can carry the person-height path; cam05, cam06 and cam11 cannot
   be leaned on at 43 / 29 / 22%.

   **Kaohsiung-cam07's zero is correct and is a scope fact, not a detector fault.** Its
   frame was read: it is a **merchandise camera** — an accessory wall of hanging cases and
   screen protectors filling the view, with about a metre and a half of floor at the
   bottom. Nobody stands in it. It cannot use the person-height path at any threshold and
   should be out of every footfall and dwell figure; what it is for is stock (§7.1). It is
   also the Kaohsiung twin of the scene behind
   [[person-fires-on-hanging-packets-at-night]] — a wall of hanging packets is what SAM 3
   returned fourteen people from on Taichung-cam09.

   **Kaohsiung-cam06's 29% is a real deficit.** Its frame was read too and it is a proper
   selling floor: stairwell, demo table, laptop bench, open floor, 408 solo frames. So it
   fails the way Tao-Hsin-cam03 fails, and for a reason still unexplained.

   **Resolution is not the driver, and it was worth checking because it looked like it
   would be.** The best of the seven is the one 704x480 camera. A census of the corpus
   found **7 of 48 cameras deliver a 704x480 sub-stream** — Kaohsiung-cam03, -cam15,
   -cam16 and Tao-Hsin-cam08, -cam10, -cam13, -cam16. That is a **physical-world action
   item of the same kind as Taichung-cam05's**: an NVR channel setting, not a modelling
   problem. It matters beyond `person`: [[sam3-product-prompts-need-resolution]] measured
   0 product instances at 352x240 and usable subclasses at 1080p, and **704x480 sits
   between those two and has never been measured.**

   Found by `data.video.DecodeError` firing rather than by a survey: pointing the probe at
   cam03 raised it after 49 frames, because the probe hardcoded 1920x1080 the way several
   scripts do and `frames()` does not scale. The eight cameras of §7.11 are all 1080p
   (verified), so nothing there is affected; the probe now asks `probe()` instead.
