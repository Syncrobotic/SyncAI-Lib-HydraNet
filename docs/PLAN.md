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
the teacher models (§4.5 resolves this via shadow mode).

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
* ~~The throughput margin is thin~~ — **this leg of the argument no longer carries
  weight, and saying so is cheaper than letting it be quoted.** It read: 1,552 fps under
  TensorRT against the 1,440 target (1.08×, measured 2026-08-24), engine-only and with
  the cheaper terrain head in the second slot; every per-frame head spends that margin
  and a head whose answer never changes spends it on nothing. **§7.4 revised the target
  to 96 × 5 fps = 480 f/s on 2026-08-26**, and at 480 the shipped engine's 1,494 f/s is
  **3.1× the requirement, not 1.08×**. The margin is not thin. The boundary still stands
  on the other two reasons — the `column` generalisation failure and the +3% vs +74%
  trunk-sharing measurement — so it is now a two-legged argument and should be argued as
  one.
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
in one 1.5 m cell and 23.4 s in the next (rendered by `scripts/site_journeys.py`; the
frame itself is not in `assets/`, whose allowlist keeps customer shop floors out of the
history unless a figure earns a line in `.gitignore`).

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

### 4.6 Retention — what is kept, for how long, and why the tiers differ

Decided 2026-08-30. Until then this document said nothing about retention and neither did
anything else in the tree: `grep -rniE "retention|PDPA|GDPR|consent"` over `src`, `tools`,
`scripts`, `docs` and CONTRIBUTING returned one line, and it was about git history being
uneditable. A product that records customers in a shop in Taiwan, where the 個人資料保護法
applies, had no stated answer to "how long do you keep this".

**The tiers are short on imagery and long on numbers**, because those carry different risk
and different value. A frame of a shop floor is the thing a person could be recognised in
and is the thing a model no longer needs once it has been trained on. A row saying
`reach_to_shelf` fired at `basis=wrist_over_fixture value=0.66 threshold=0.35` identifies
nobody and is the only record of why an alert was raised months later.

| what | where | kept | why that number |
|---|---|---|---|
| raw store clips | `datasets/studioa_clips/` | **30 days** | long enough for one incident investigation and one re-cut of a figure; past that a clip is a liability holding no answer the measurements have not already extracted |
| imagery derived from them | `runs/**/*.{jpg,png,gif,mp4}` | **90 days** | one model iteration. A crop sheet is evidence for a verdict, and a verdict is re-read when the next training set is assembled |
| measurements | `runs/**/*.{json,jsonl,npz,log,yaml}` | **kept** | numbers, not pictures. Deleting these deletes the apparatus behind every claim in this document |
| static plates | `datasets/studioa_static/` | **kept** | the temporal median removes every moving person by construction; a plate is the empty shop |
| disposition log | the JSONL store | **kept** | `frame_ref` is a *pointer* — clip path plus frame offsets — not an image. When the clip expires the pointer stops resolving, which is the deletion, and the operator's verdict survives as a number |
| published figures | `assets/` | **permanent, and unerasable** | they are in git history. CONTRIBUTING already states this; §4.6 restates it as the reason the audit gate in front of `assets/` is the strictest one here |

**Two properties of the design carry most of the weight, and neither was added for this.**
The disposition log stores a pointer rather than a frame, so expiring a clip severs the
link without touching the log. And everything except `assets/` is gitignored, so deletion
is deletion — the only holding that cannot be revoked is the five tracked figures, which
is why the thing guarding that directory is an allowlist plus a per-figure audit verdict
rather than a convention.

**What this does not decide, and it is not a code question.** Whether the deployment
partner or this project is the 蒐集者 under the PDPA changes who owes notice to whom, and
that changes the numbers above rather than the mechanism. Stated here so the omission is
visible: the tiers are engineering defaults chosen to be defensible, not a legal position.

`scripts/retention_sweep.py` enforces the table and `tests/test_retention_policy.py` holds
the two to each other, because a policy nothing executes is the failure mode this
repository has the most notes about.


## 5. What it must never do

1. Never put an answer in the weights that belongs in a config.
2. Never let the rule layer reach into the model — "4 minutes is loitering" is an argument.
3. Never report a per-class number resting on one camera.
4. Never buy coverage with more frames from already-annotated cameras.
5. Never infer age/gender on customers; no face recognition; no cross-store tracking.
   `staff/customer` is a uniform, not an identity, and stays in.
6. No SKU-level checkout; no new camera hardware; no per-frame dense scene understanding.

## 6. Build order — one artefact, one gate per step

**When §7 answers a step, edit the step's row in the same commit.** §7 is where the work
lands and this table is what a reader consults, and nothing holds the two together. Twice
on 2026-08-29 a row still stated as current something §7c had already superseded — step 8
said the NTU projection was "not yet written" and `pose_sequence.py` had "zero consumers"
while `tools/temporal/` held both and §7c.17 recorded the gate's numeric half passing;
step 5 said no labelled site clip exists while `runs/gt_cam01/idf1.json` held the IDF1.
A reader of this table got a picture the same document contradicts further down, and no
test can catch that. The convention is the only mechanism.

| # | step | artefact | gate |
|---|---|---|---|
| 1 | **package split** — create `src/syncai_bev3d`, move the §2.1 code, define the `camera.json` schema; archive non-current configs | two importable packages, green tests | **DONE 2026-08-25** (`64eafd8`, `6e0eb36`, `1395e2c`) — 1,824 tests green, boundary enforced by `test_package_boundaries.py`, GPU idle and no training unit running when the tree moved. Config archiving deferred: which of the 30 yamls are current needs a decision, not a guess |
| 2 | **commission all 48 cameras** — build the 4-click tool and zone tool, run the pipeline | per-camera `camera.json` + a 1 m grid rendered on one real frame per camera | the grid looks right **to my own eyes on every camera**. **8 of 48 shipped** (`runs/commission01/REVIEW.md`), Taichung-cam05 withdrawn — two furniture checks agree its cells are over-scaled. **"Scale-measured" means the 1.70 m person-height prior, not a tape measure** (§7.19): every shipped `scale_source` reads `person_height_median_vs_1.7m_prior_nNN`, on 15–37 boxes. Masks, walkable outline, shelf ROIs and a commissioning 3D scene are written for all 8 (`masks_pass.py`, `scene3d.py`; tables 0.78–0.97 m fleet-wide); 5 mask sets clean, the Tao-Hsin pair partial and Kaohsiung-cam04 missing its pillar. **The Tao-Hsin caveat is measured (§7.21) and is the whole of what limits the 3D render's coverage** — the fixtures are proposed and then classified `wall`, and neither clustering nor paint order moves that camera's map by a pixel. **71 service zones across the 8, fully automatic** (§7.19): a zone is the floor *beside* a fixture, not its footprint, because a footprint is a region no shopper can occupy. Blocked on the physical world for the rest: 14 cameras need a per-store visual reference, 4 need mount-type triage, Taichung-cam05 needs its NVR stream setting; the 7 Kaohsiung person-score cameras are answered per camera in §7.12. Remaining tools: the 4-point ground-calibration click tool (§2.1b) and the tamper reference (§2.1h) |
| 3 | **pose head resident** — trained, 60/60 epochs, final val `coco_person` mAP 0.2022. **Two traps this step paid for, both live.** Its run directory is `runs/hydranet_retail_security_b03_cw_xl-20260825-162131`: the pose config inherited `output_dir` from the security run, so the directory is named for a different model and only the `config.yaml` inside it is authoritative. And `primary_metric` is `terrain_mIoU/site_seg03`, so **`best.pt` is epoch 15 and is not the pose checkpoint** — read `selection.json` before using either. **The exporter's shape is the third**: `cli/export_onnx.py`'s `ExportWrapper` lists the ModuleDicts it knows about, so every new head is silently absent from every engine until someone adds a line, and the only thing that catches it is a config whose *only* head is the missing one — which is how a graph with **zero outputs** passed `onnx.checker` and onnxsim and left CI's export-parity job red for nine days. **Throughput** (§7.8): NVDEC saturates at ~520 streams against the CPU pipe's 63–69, so decode is no longer the binding leg; host NMS and the tracker cost 3.4 of 24 cores at target. Both checkpoints on the full test split (2,862 images / 6,360 persons / 93,967 judged joints):

| checkpoint | PCK@0.2h | L2 p50 | mean | p90 |
|---|---|---|---|---|
| `best.pt` (E15, terrain-selected) | 0.840 | 13.8 px | 35.4 | 96.3 |
| **`last.pt` (E60)** | **0.898** | **9.3 px** | **25.1** | **67.7** |

**Use `last.pt`.** Taking the handoff's `best.pt` would have gated on a figure 5.8 points lower with 48% more error. Trajectory E15 → E29 → E60 is 0.840 → 0.882 → 0.898, still rising at the end. The run logs **no pose validation metric at all**, so there is no per-epoch curve and these two checkpoints are the only candidates that survive a run — fix the config's `primary_metric` before the next one (§7.9).

**pose02 FINISHED 2026-08-26 19:19, and the accuracy leg of this gate is met.** 120/120 epochs. On the full test split (2,862 images / 6,360 persons / 93,967 judged joints) `eval_student.py` reads **PCK@0.2h 0.915, L2 p50 7.7 px** (mean 21.5, p90 54.7), against pose01's `last.pt` at 0.898 / 9.3 px. **`best.pt` and `last.pt` return identical figures** this time -- the run selected on `pose_PCK@0.2h`, which picked epoch 119 -- so pose01's trap, where a terrain-selected checkpoint was the run's *worst* pose model, is structurally gone rather than avoided by hand. **The curve is flat at the end**: 0.9339, 0.9344, 0.9345, 0.9345, 0.9345 over the last five epochs on the val prefix, so 120 epochs was enough, which is precisely the question pose01 could not answer about itself. The cost of selecting on pose is on the record in `selection.json`: terrain mIoU was 0.698 at epoch 17 and is 0.637 at the selected epoch. Superseded note: **pose02 was running as of 2026-08-26 14:08** — `hydranet-pose02.service`, `runs/hydranet_retail_pose02`, 120 epochs, ~7 min/epoch. The 2026-08-25 overnight attempt **never started**: the trainer refused a dirty tree over another session's uncommitted `analytics/world.py`, the two eval steps behind it then failed on a checkpoint that was never written, and the card sat idle for 17 hours. It selects on `pose_PCK@0.2h` (§7.9), so this run leaves a per-epoch pose curve and a `best.pt` chosen by the head it exists to train.

**Throughput re-measured 2026-08-26 on an idle card — the gate PASSES at the shipped
canvas, and the margin is 3.7%.** The 1,324 f/s below was pose01's weights *with an RL
neighbour on the card*; with pose02's `last.pt`, batch 16, fp16 and nothing else on the
GPU the engine measures **1,494 f/s = 100 streams**, against the 1,440 then required (**§7.4 revised the requirement to 480 f/s the same day, so read this as 3.1× rather than 3.7% of margin**). So the leg
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

| leg | measured | against 1,440 f/s (the target of the day; §7.4 revised it to 480) |
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

| keypoint error vs ViTPose; throughput re-measured **end to end — NVDEC decode → engine → host NMS → tracker** — with pose in the slot and the ROI path at its 0.2–1 fps cadence | `reach_to_shelf` and `crouch` fire correctly on a watched clip; fps ≥ **480** stands end to end, not engine-only (the gate was written at 1,440 and §7.4 revised it to 96 × 5 fps on 2026-08-26; the leg passed either way, at 1,494 f/s) — 96 streams is a requirement now (§7.4), and decode/NMS/PCIe were named the real risk when the target was set |
| 4 | **detection uplift at zero GPU** — ~~temporal-consistency tiers~~ **REFUTED for `person`, 2026-08-26, measured**: `instances_all_train.json` is not a pool of unused good labels, it is the same Grounding DINO teacher dumped at 0.10 instead of 0.35 — median score 0.149, and `instances_train` is simply that pool thresholded (48,005 vs 47,465 at 0.35). Worse, persistence is the wrong signal here and it is wrong in the dangerous direction: score vs run-length correlation is **+0.006**, the 0.10–0.20 band moves a median **0.3 px per frame**, and **41,160 sub-threshold boxes persist ≥8 frames** — posters, mannequins and hanging packets are the most persistent things in a store. A consistency tier would promote exactly them. The measured detection problem is **recall, not label count**: on one clip the trunk's own dense head marks **1.30× as many people as the box head returns** (146 blobs vs 112 boxes over 60 frames) while scoring `person` IoU 0.885 against detection mAP@50 0.302 — the features find people the box head cannot emit. **Re-scoped and then measured on the fleet, 2026-08-26**: `serving.decode.confirm_with_dense` admits low-scoring boxes where the dense head puts person pixels under them. Three arms over the same eight clips — A shipped 0.35, B threshold-only 0.15, C 0.15 + confirmation:

| arm | person dets | tracks | boxes dropped | fall | crouch |
|---|---|---|---|---|---|
| A 0.35 | 11,363 | 202 | 0 | 1 | 1 |
| B 0.15 | 27,715 | 569 | 0 | 4 | 5 |
| C 0.15 + confirm | 25,713 | 520 | 2,002 | 4 | 5 |

**The recall gain is real and the confirmation does not protect the event layer.** C drops 12–14% of boxes and produces *exactly* B's events, on the whole fleet and again with Kaohsiung-cam04 excluded (2 fall / 3 crouch both arms). The boxes it removes were never the ones firing events: the extra events come from real people detected at low confidence, whose keypoints are noisier, making more posture runs. So the next mechanism is not another box filter — **the event layer has to see the detection confidence a track was built from**; a track of 0.15 boxes cannot be trusted for posture the way a 0.6 track is. Kaohsiung-cam04 also explodes 3,353 → 13,595 detections at 0.15, which is its open person-score investigation surfacing, so the threshold is a **per-camera** decision. Re-scope before spending a retrain. Original scope kept below: temporal-consistency tiers from `instances_all_*.json`, FP polygons from the 37 hotspots; **the night pass, done 2026-08-26** (`scripts/night_ghosts.py`, `runs/night_ghosts01/`): the shipped detector produces **zero `person` boxes in 2,250 frames of an empty shop at 23:58 store-local, across 15 cameras**, at the shipped 0.35. The ghosts were always the *teachers*' -- SAM 3's `person` prompt returned 14 hanging accessory packets as people on Taichung-cam09, and Grounding DINO at 0.35 fired on an empty store on 13 of 42 cameras -- and the student did not inherit either. **Taichung-cam09 itself returns zero.** So the false-positive rate that `after_hours_person` had to be quoted against, and which did not exist, is now measured and is 0. Two caveats, because the sample is what it is: only 15 of 48 cameras have a 23:58 clip on this box, and only 2 of the 11 cameras where Grounding DINO fired at night are among them (both zero). `data/night_person.py`'s static-plate veto stays where it is -- it protects the *teacher's* output, and the serving path turns out not to need it. Superseded: measure whether FP polygons + temporal consistency remove the IR ghost persons | new Gold/Silver training set; night `person` precision figure | Gold precision ≥95%, Silver ≥85% on a 300-frame sample; `after_hours_person` stays on the VLM trigger list **only if** the night figure passes |
| 5 | **L1 validation** — tracking quality on in-domain clips, and homography accuracy against ground truth. **Geometry: done** (§7.13) — 7.3 cm median floor error over 19,824 WILDTRACK observations, yaw to <0.2°, and it did not need the images. **Tracking: has a number and a caveat** (§7c) — `runs/gt_cam01/idf1.json` on 900 frames of Taichung-cam01, single-stage IDF1 0.739 / 6 switches, two-stage 0.742 / 3; read `provenance.json` before quoting either, because a model labelled those identities by eye. **What the step exists to fix is fragmentation**: 202 tracks over 24 minutes, 43% ever enter a zone, median visit 3.2 s, and 58% of endings are mid-view deaths concentrated on two cameras (§7.11 — 86% still have a box on the person at a median 0.338 against a 0.35 threshold, so it is a threshold problem before it is a tracker one). The tracker-lost/detector-gone split is **not** established and nothing should be concluded from it until an appearance model can tell two shoppers apart (§7.11) | position error in metres; ID-switch count per watched 10-minute clip | position error small enough that zone events land in the right zone; dwell/loiter durations survive — an ID switch mid-loiter resets the clock, so switches on the watched clips must be rare enough not to |
| 6 | **L3 end to end, one camera** | an event log readable against its video | events match what the clip shows |
| 8 | **behaviour head** — the one component of the head set that has never been trained. `analytics/pose_sequence.py` holds its view-invariant features (limb angles as cos/sin, limb lengths over torso). **The numeric half of this gate PASSES (§7c.17)**: `tools/temporal/train_posture.py`, two 1-D convolutions, **32,933 parameters** against the 100K budget, held out by performer, clearing the §7.16 linear floor on all five pairs (fall vs pick_up 0.963 against a 0.944 floor). Training data is **NTU RGB+D's official skeletons** in real camera-frame metres, projected to our measured camera pose by `tools/temporal/ntu_project.py`; **PoseLift** supplies the shopping half, and its cameras look near-straight-down (height/torso 0.9 against our 2.6), which is why the features are angles and ratios rather than coordinates. **What is NOT done**: no consumer on the serving path, and `fall`/`crouch`/`sit` not yet watched firing on site footage | a temporal classifier under 100K parameters, CPU, plus the NTU→camera-pose projection tool | **it must beat the geometry it replaces, which now has a number**: §7.14's tuned rule reads 96% fall recall at a 29% `A06 pick up` false rate on NTU. The model is held out **by subject**, not by clip — NTU repeats each action per performer, and a random split puts the same body on both sides. And the standing rule applies unchanged: `fall`, `crouch` and `sit` verified firing correctly on watched site footage, by eye |
| 9 | **`staff/customer`** — listed in §4.3 as necessarily in-domain since the plan was written, blocking two steps, and until 2026-08-27 not a step of its own. First measurement in §7.15: **0.893 balanced accuracy** held out by camera, from nine torso-colour statistics that beat every embedding in the tree. What remains is not a better model — the two failures were read off the frames: a camera whose staff wear jackets over the polo, and a near-top-down camera whose crops are hair and shoulders. So: the **three uniform reference photos per store** already on the waiting list, the **1,181 crops in `pool/`** already extracted, and top-down views in the labelled set | a per-store uniform reference fitted from the photos, and `staff` on `Track` beside `TrackSupport` | **stated by its consumer rather than by the classifier.** `reach_to_shelf` measures 11.7 alerts a minute on a clip where every person is staff (§6 step 3); the gate is that the same clip's alert rate falls to something a person would read. The classifier's own figure is reported held out **by camera**, never pooled — 16 of 32 labelled cameras carry a single class, where "which camera" is "which class". **Half done (§7.23)**: `analytics/staff.py` persists a fitted model with the camera its accuracy was held out on, `Track.staff_scores` is the wire, and `demo_video --staff-colours` is the first consumer. Torso colour alone, because that arm is the one with per-camera numbers. Licensed on Kaohsiung-cam04, Taichung-cam01 and Taichung-cam11, **refused on Tao-Hsin-cam04 at 0.417** by a floor of 0.90. Still missing: the **per-store uniform reference fitted from three photographs**, which is the thing that would move Tao-Hsin |
| 7 | **shadow mode, one store** | alerts/camera/day + operator accept/reject log + **the first shrinkage reconciliation** | single-digit rate; rejects show an actionable pattern |

Throughput ceiling: already passed 2026-08-24 (1,552 fps TRT, batch 16) — but engine-only
and with terrain in slot 2, which is why step 3 re-takes it. Steps 1 and 2 are independent;
nothing after 6 starts until 6 passes, and 6 is not attempted before 5 — an L3 event log
over unvalidated tracks cannot be attributed when it is wrong.

**Steps 8 and 9 are numbered last and do not run last.** Both are prerequisites for 6 and
7: an L3 event log without `staff/customer` is 11.7 alerts a minute of staff working, and
without a behaviour head `sit`, `crouch` and `fall` remain geometric rules whose
thresholds §7.14 measured against ground truth and found marginal. They carry 8 and 9
because **1–7 are addresses**: step 2, 3 and 5 are cited by number in twelve files outside
this document — `commissioning.py`, `journey.py`, `track_endings.py`, `site_journeys.py`,
`zone_draw.py`, two test modules, README and three configs — and renumbering to put them
in dependency order would break every one of them to make a table read better. The
position is stated here instead, which is what a number cannot carry and prose can.

Added 2026-08-27, and the reason they were missing is worth keeping: both are components
§2.3 and §4.3 describe in full, and neither had an artefact or a gate anywhere in this
order. A component with no step is not scheduled, it is assumed.

## 7. Open questions — each blocks a specific step

### 7a. Still open

1. **`stack` as a 5th detection class** invalidates checkpoint comparability, and a
   vocabulary change silently empties `analytics/events/zones.py:346`'s default class list
   — the stock-removal alarm stops firing without an exception. Decide at step 4.

29. ~~The serving target is bound by PCIe~~ — **cleared 2026-09-01, and the entry that
   said otherwise was scored against a target this document had already replaced.**
   `bench_trt.py` still held `TARGET_FPS = 1440` — 96 streams at 15 fps — six days after
   §7.4 revised delivery to **96 x 5 fps = 480 f/s**, so the first version of this entry
   read a 3.5x shortfall where the real figure was 0.77x.

   Measured on the RTX PRO 6000 at 640x1120, fp16, batch 16, end to end with both copies:

   | export | compute | + copies | vs 480 |
   |---|---|---|---|
   | plain | 916.2 | 368.8 | 0.77x |
   | `--argmax-seg` | 848.0 | 433.3 | 0.90x |
   | `--argmax-seg --uint8-input` | 848.9 | **633.3** | **1.32x** |

   **`--uint8-input` is what clears it** (`exports/pro6000_xl20260825/xl_last_u8in_b*`).
   The image is the larger half of the round trip once the outputs are class ids rather
   than logits — 137 MB of fp32 against 32 MB of uint8 output at batch 16 — and a byte
   holds 0-255 exactly, so the pixels are identical and `--check-parity` holds it to that
   (worst 1.52e-05 against a 1e-4 tolerance). The binding is renamed `image_rgb_255_u8`
   so a float host fails to find it rather than misreading it.

   Two things this cost on the way, both now fixed: `bench_trt.py`'s end-to-end figure
   never copied the outputs back (`69ede66`), and its fp16 conversion left an input-side
   `Cast` saying FLOAT while the constants beside it became FLOAT16, which TensorRT
   refuses — the shape any non-float input contract produces.

   ~~**Still not measured on an idle card.**~~ **Re-measured 2026-09-02 on an idle card,
   three runs, and the contamination was worth 56% of the end-to-end figure.** The only
   neighbour was `tools/lite3_web/server.py` holding 790 MB at 0% GPU utilisation, against
   the three ~800 MB processes of the first attempt. `runs/bench_idle_20260902{,_r2,_r3}`:

   | export | compute | + copies | spread | vs 480 | vs 960 | old + copies |
   |---|---|---|---|---|---|---|
   | plain | 1471.7 | 476.5 | 8.6 | 0.99x | 0.50x | 368.8 |
   | `--argmax-seg` | 1369.5 | 592.6 | 7.8 | 1.23x | 0.62x | 433.3 |
   | `--argmax-seg --uint8-input` | 1364.4 | **990.5** | 16.1 | **2.06x** | **1.03x** | 633.3 |

   Run-to-run spread is **1.6%** on the figure that matters, not the factor of 1.6 the old
   note warned about -- that spread was the neighbours, not the card. The ordering is
   unchanged and `--uint8-input` is still what clears the target.

   **So 96 x 10 fps = 960 f/s passes the engine and the PCIe path, at 1.03x** (985.0,
   985.5 and 1001.1 over the three runs; every run clears it, by 2.6-4.3%). Two things
   that follow, and the second is the one that decides:

   * **3% is not a shipping margin.** Compute drifts down within a session (plain reads
     1506.7, 1479.1, 1429.3 across the three runs, -5%), so a figure this close to the
     line is a thermal question as much as an architectural one.
   * **The serving path cannot feed 96 streams at any frame rate.** `data/video.py` is
     still the CPU decode pipe at 63-69 streams; NVDEC reaches ~520 and is what decision 8
     chose, and migrating serving onto it "is not done" by that decision's own words. The
     engine is no longer the binding leg for 96 x 10 -- decode is, and it binds below 96
     streams before frame rate enters the question.

28. **The best terrain checkpoint in the tree is not the one anything uses.** Measured
   2026-08-31, final epoch of each run (which is what `last.pt` is), the same recipe:

   | metric | `runs/…b03_cw_xl` (the default) | `runs/…b03_cw_xl-20260825-162131` |
   |---|---|---|
   | `terrain_mIoU/site_seg03` | 0.5652 | **0.6254** |
   | `terrain_mIoU/site_seg` | 0.5658 | **0.6178** |
   | `detection_mAP50/site_boxes03` | 0.3020 | **0.3356** |
   | `terrain_mIoU/ade20k` | **0.7255** | 0.7043 |
   | `detection_mAP/coco_person` | **0.2103** | 0.2022 |

   Better on every **site** metric, slightly worse on the two web-dataset ones. Nothing
   points at it: `stable_infer.py`, `onboard_camera.py`, `demo_video.py`, `demo_gif.py`,
   `flicker_baseline.py` and `serve_pilot.py` all name the default run, and **both README
   figures were drawn with it**. What the gap costs is visible rather than abstract — on
   `dingpu-1f/test1` the default's `best.pt` labels a whole stone wall `floor` and the
   dated run's `last.pt` does not (`assets/dev/dingpu/test1_floor_mask_two_checkpoints.png`,
   floor share 0.561 vs 0.337, IoU 0.590).

   Two things to settle before promoting, because promotion re-cuts every published
   figure and re-runs the fleet: whether the dated run was deliberately not promoted, and
   whether the two web-metric regressions matter for anything shipped. **Deferred by the
   user on 2026-08-31 until there is more information.**

   The `selection.json` discrepancy noted here on 2026-08-31 — the file names
   `terrain_mIoU/site_seg03` as its primary metric but records 0.7030, which is that run's
   best plain `terrain_mIoU`, while its `site_seg03` best is 0.6681 — **was a reporting
   bug, fixed 2026-09-01.** `runmeta.HEAD_METRICS` matched metric names exactly, and
   `evaluator._det_metrics` writes an unqualified key only for a run with a single
   detection val set. Every retail run has three or four, so the report skipped the
   detection head entirely and showed the bare `terrain_mIoU` in place of the qualified
   primary. Replayed over the same `metrics.jsonl`, it now adds a third figure to the
   promotion decision: **the dated run's `best.pt` (epoch 15) gives up 0.0576 on
   `detection_mAP/coco_person`** against the 0.2022 it reaches at epoch 60, which is the
   `last.pt` number tabled above. The table compares `last.pt` to `last.pt` and is
   unaffected; what changes is that promoting the dated run and then reading its `best.pt`
   for anything person-shaped costs more than the table shows.

5. **Retail dashboard surface unscoped** — the numbers fall out of L1 free; what a store
   manager opens, at what cadence, is a product question. Blocks nothing before step 6.
   One modelling gap hides inside it: **store-level footfall needs cross-camera dedup**
   (overlapping views double-count a person). Single-store re-linking is in scope,
   cross-store is banned; the mechanism is unscoped.

### 7b. Decided — the answer, and what it cost

2. ~~Night is unscoped~~ — **decided 2026-08-25: night is in v1, gated on a measurement.**
   Step 4 carries the night pass (14 IR ghost persons on one empty frame, measured);
   `after_hours_person` triggers only if the night precision figure passes.

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

6. ~~Fall/crouch training source~~ — **decided 2026-08-25: no in-store staging, ever.**
   Resolved in pose space: the temporal model reads keypoint sequences, so public 3D
   action data (NTU RGB+D fall/squat, CMU MoCap) projected to the measured camera pose
   replaces staged clips (§2.3). What stays open is only the sim-to-real transfer
   measurement, taken at step 6.

7. ~~Recommissioning ownership~~ — **decided 2026-08-25: a nightly closed-hours job**
   re-derives plate, masks, ROIs and re-anchors zones automatically; a human gets a
   morning accept/reject only when the diff exceeds a threshold, and a moved camera
   always escalates instead of auto-healing (§2.1). To build alongside step 2's tooling.

8. ~~No NVDEC backend exists on this box~~ — **decided 2026-08-25: PyNvVideoCodec.**
   Raised because gate 3 names NVDEC and the box had nothing to run it on (no
   PyNvVideoCodec, no DALI, no PyAV, and the only ffmpeg on PATH offering `vdpau`
   alone). Installed 2.2.1 — one wheel, no dependency of the running training job
   touched. Decode went from 63–69 streams to ~520 and stopped being the binding leg.
   `data/video.py` is still the CPU pipe: migrating the *serving* path to NVDEC is the
   work this decision authorises, and is not done.

3. ~~Pose distillation risk~~ (§2.2) — **closed 2026-08-26 by step 3's gate.** The
   student agrees with ViTPose at **PCK@0.2h 0.915 / L2 p50 7.7 px** on the full test
   split, and both consumers were verified firing correctly on real footage by eye:
   `crouch` on a member of staff folded at a low cabinet, `reach_to_shelf` on a wrist over
   a counter. The two-head architecture's claim holds. What did *not* hold is `fall`,
   and that is a grouping failure in crowds rather than a distillation one -- see gate 3.
   Original: the claim the two-head architecture rests on. Answered by step 3's gate. The scope question above it is settled: **decided
   2026-08-25, fall/second-level behaviour is in v1**, so pose stays a per-frame L0 head.

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

### 7c. Investigations — finished, and kept for their chains

Answered work, not open questions. Each entry is kept whole because its number is a
measurement and its prose is what that measurement cannot show, and the pair is what a
decision rests on -- reading either half alone is how a figure gets quoted for
something it does not support.

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

13. **Step 5's geometry gate: 7.3 cm, and what it is agreement with.** Run 2026-08-27 with
   `scripts/wildtrack_ground_eval.py`, and run **before the archive finished downloading**:
   a zip stores every file behind its own local header, so all 7 calibration XMLs and all
   400 annotation JSONs were already in the bytes on disk and could be inflated out of the
   partial file. The images are not needed — the boxes are in the JSON.

   | view | n | rigid p50 | p90 | free-scale p50 | recovered scale |
   |---|---|---|---|---|---|
   | CVLab1 | 4,343 | 0.064 m | 0.134 | 0.035 | 0.991 |
   | CVLab2 | 3,960 | 0.072 | 0.153 | 0.038 | 0.989 |
   | CVLab3 | 2,971 | 0.076 | 0.128 | 0.048 | 0.991 |
   | CVLab4 | 952 | 0.059 | 0.147 | 0.051 | 0.989 |
   | IDIAP1 | 1,450 | 0.065 | 0.094 | 0.021 | 0.986 |
   | IDIAP2 | 4,728 | 0.086 | 0.158 | 0.038 | 0.990 |
   | IDIAP3 | 1,420 | 0.104 | 0.200 | 0.063 | 0.982 |

   Pooled: **p50 7.3 cm, p90 15.3 cm over 19,824 observations.** The recovered scale is
   0.982–0.991, so there is no scale fault of the Taichung-cam10 kind here (§7.10).

   **The independent check that makes the number believable.** The rigid fit has three
   free parameters and never sees the extrinsics; it is fitted only to our projected floor
   points against the ground truth. It lands on **each camera's actual pose**: yaw within
   **0.02°–0.18°** and centre within **2.7–13.1 cm** of the values in `extr_*.xml`. A
   broken chain cannot do that with three parameters.

   **What this is NOT, and the evidence is in the shape of the error.** The error is
   **flat with range** — 10.3 cm at 0–5 m, 6.9 at 10–15, 6.1 at 20–25, 8.2 beyond 25 —
   and a foot-row pixel error costs metres that grow like range squared (measured on
   these very points: **6.2 cm per pixel** at the median 16.3 m). Flat therefore means the
   residual is not dominated by anything in image space, which fits how WILDTRACK was
   annotated: its ground-truth positions and its per-view boxes come from **the same
   annotation act**. So this measures **agreement with a professional multi-view
   calibration and its annotation model, to 7 cm — not a measurement against a tape
   measure.** Step 5's gate should be read that way, and the plan's wording ("ground-truth
   floor positions") invites the stronger reading it does not support.

   **And the regime is not ours.** WILDTRACK's cameras sit at **8.7–20.1° of pitch**
   (recovered by `to_our_model`, heights 1.68–3.40 m); the shipped fleet sits at
   **38.8–52.3°**. A shallow ray grazes the floor, so this is the harder end and the gate
   is conservative — but "our cameras are 7 cm out" is not a sentence this run supports.
   Taichung-cam10 remains 1.21x wrong (§7.10) and nothing here contradicts that: a
   per-camera scale error is exactly what a fleet-independent dataset cannot see.

   **One rule that transfers immediately.** Boxes touching a frame edge are 9% of the
   population and carry **double the error** — 13.4 cm against 6.9 cm. `track_endings.py`
   already treats an edge box as an `exit`; this says an edge box's *metres* should be
   distrusted too, which nothing in the serving path does yet.

   **Two bugs this file caught before they became numbers**, both of which would have
   produced a plausible wrong answer rather than a crash: the world frame is
   **centimetres** (a camera 987 m up if read as metres), and the reprojection needs
   WILDTRACK's **distortion** because the boxes are drawn on original frames. And the grid
   convention was decided by measurement rather than memory — `check` reprojects each
   truth position and reads off the distance to the annotated box foot: **24.1 px** for the
   row-major reading against **1,399.9** and **1,507.4** for the alternatives. The script's
   own default had been one of the losers until that run.

14. **The `fall` height threshold is measured now, and on its own it does not work.**
   `analytics/events/pose.py` says of its own defaults that "none is measured", and
   `fall_head_height_m = 0.80` was added 2026-08-26 on the evidence of one 24-minute clip
   where it took the fleet from 3 false falls to 0. Measured against NTU RGB+D's official
   3D skeletons — real camera-frame metres, read by `data/ntu_skeletons.py`, 120 clips per
   class, `tools/temporal/ntu_fall_discriminator.py`, `runs/ntu_fall01/`:

   | class | n | min head above floor, p50 | **below 0.80 m** | peak torso p50 | over 55° |
   |---|---|---|---|---|---|
   | **A43 falling down** | 120 | 0.47 m | **0.99** | 60.9° | 0.60 |
   | **A06 pick up** | 119 | 0.64 m | **0.92** | 58.8° | 0.62 |
   | A08 sit down | 119 | 0.86 m | 0.34 | 47.3° | 0.30 |
   | A42 staggering | 120 | 1.25 m | 0.03 | 20.3° | 0.06 |
   | A27 jump up | 120 | 1.18 m | 0.03 | 16.5° | 0.01 |
   | A01 drink water | 120 | 1.34 m | 0.03 | 8.2° | 0.01 |

   **99% of falls put the head below 0.80 m and so do 92% of pick-ups.** The height test
   separates a fall from staggering, jumping and standing still — all at 3% — and does
   **not** separate it from the one posture that matters, a shopper reaching a bottom
   shelf. §7.11's torso-angle result was the same shape (0.60 against 0.62), so *both*
   published conditions of the shipped rule fail the control on their own.

   **What separates them is how long the head stays low, and the shipped rule does not
   test that.** `sustained_seconds = 1.0` is applied to the *angle*; the height is
   instantaneous. The longest run below 0.80 m, per class:

   | sustain required | fall recall | pick-up false rate | sit | others |
   |---|---|---|---|---|
   | instantaneous (shipped) | 0.99 | **0.92** | 0.34 | 0.03 |
   | 0.5 s | 0.98 | 0.61 | 0.11 | 0.02 |
   | **0.7 s** | **0.96** | **0.29** | 0.04 | 0.02 |
   | 0.8 s | 0.85 | 0.15 | 0.03 | 0.02 |
   | 1.0 s | 0.42 | 0.03 | 0.01 | 0.02 |

   The knee is at **0.7 s: 96% of falls kept, 71% of the pick-up false rate removed.**
   For a safety alarm recall is the side to protect — a missed fall costs more than an
   extra alert — so 0.7 s is the defensible point and 1.0 s is not, because at 1.0 s the
   rule misses more falls than it catches.

   **This does not make the alarm shippable and the honest reading is that it moves it
   from unusable to marginal.** 29% of bottom-shelf reaches still alarming is too many for
   a room where that is the commonest posture there is. What it does is replace two
   unmeasured defaults with one measured one and say where the remaining gap is.

   **Two things this cannot show.** NTU's `A06 pick up` lifts from the **floor**; a shop's
   lowest merchandise sits at roughly 0.4 m, so a real bottom-shelf reach spends less time
   under 0.80 m and the real false rate is probably lower than 29% — **unmeasured**. And
   the vertical is estimated per clip from the subject's own standing pose, because a
   Kinect's +y is only gravity if that Kinect was level; it sits a median **13.8°** off
   the camera axis, so the estimate is doing work rather than reproducing one.

   **A hypothesis raised and refuted rather than left standing**: the 42% recall at 1.0 s
   looked like it might be NTU clips ending shortly after the fall. It is not. A43 clips
   run a median 2.42 s and keep recording a median **1.50 s past the end of the low
   spell**, with **0%** stopping within 0.3 s of it. The 42% is a property of falls in
   this data, not of the recording.

15. **`staff/customer` has its first measurement, and nine colour numbers beat every
   embedding in the tree.** 421 crops from 142 people were extracted by
   `scripts/staff_crops.py` and sorted by hand into 154 staff / 223 customer / 44
   unclear — 54, 75 and 16 people. Probed by `scripts/staff_probe.py`,
   `runs/staff_probe01/`, **leave one camera out** over the 16 cameras carrying both
   classes, 230 held-out crops:

   | frozen feature | balanced accuracy | staff precision | recall |
   |---|---|---|---|
   | colour + ImageNet backbone | **0.893** | — | — |
   | **torso colour, 9 numbers** | **0.880** | 0.867 | 0.848 |
   | ImageNet backbone, 512-d, no person training | 0.850 | 0.824 | 0.815 |
   | `rapv2_crop01` (RAP v2, 256-d) | 0.777 | 0.703 | 0.772 |
   | `crop_encoder01` (PA-100K, 256-d) | 0.723 | 0.612 | 0.772 |

   **Both attribute-trained encoders are worse than untrained ImageNet features**, which
   is the Market-1501 result of `runs/rapv2_eval01` arriving on a second task: that
   fine-tune took association from the ImageNet floor's mAP 0.0318 to 0.0113, and it
   costs here too. **The production signal is a colour, not an embedding** — nine
   statistics over the chest band match a 512-d backbone to within 3 points and beat both
   fine-tunes by 10 and 16. A per-store colour reference is also cheaper, interpretable,
   and re-fittable from three photographs when the shop changes its shirts, which an
   embedding is not.

   **0.893 does not gate anything yet.** `reach_to_shelf` at 11.7 alerts a minute needs
   staff removed reliably, and one in nine wrong is not that. What it establishes is the
   shape of the answer and the two things standing in the way, both visible by eye on the
   worst cameras:

   * **Tao-Hsin-cam01 (0.40–0.60): the uniform is not in the crop.** It is an entrance
     camera in blown-out daylight and its staff wear jackets over the polo. The labeller
     could tell from context; the crop cannot, and the model only ever sees the crop.
   * **Taichung-cam08 (0.27 on the RAP features): viewpoint.** Near top-down over a
     counter, so a crop is mostly hair and shoulders while the 15 training cameras see
     people from the side.

   Neither is fixed by a better model. The first wants the **three uniform reference
   photos per store** already on the waiting list; the second wants top-down cameras in
   the training set, and `pool/` holds **1,181 further crops** already extracted.

   **Two defects found by running it rather than by reading it.** One person in the batch
   is a tracker identity switch — Kaohsiung-cam04 t0002 is a customer at 14:30:27 and a
   member of staff at 14:32:25 under one id — which the labeller correctly sorted into two
   folders and which is an extraction fault. The single-stage tracker was chosen precisely
   to avoid merges, so the **measured merge rate is at least 1 in 142**, and only
   cross-class merges are detectable at all. And the first version of the probe reported
   an "ImageNet floor" that was a **random projection**: `CropEncoder.embed` is an
   untrained `nn.Linear` on a checkpoint-free encoder, so two runs of identical code on
   identical data gave 0.757 and 0.839. It reads the pooled backbone now and reproduces
   exactly.

   **What the split cannot measure.** 16 of the 32 labelled cameras carry exactly one
   class — three all-staff, six all-customer among them — so on those "which camera" *is*
   "which class". They are excluded from the fold rather than averaged in, which means
   this number describes the 16 mixed cameras and says nothing about the other half.

   **The attribute line stays, decided 2026-08-28.** `train_attributes.py`,
   `eval_attributes.py` and `prep_rapv2.py` have no caller and were up for deletion on
   that basis. They have a consumer, it is just not an import: `staff_probe.py` loads
   `runs/crop_encoder01/last.pt` (PA-100K) and `runs/rapv2_crop01/last.pt` (RAP v2) on
   every run, 45 MB each and both present, and skips the row rather than failing when one
   is missing. `runs/` is gitignored, so those two checkpoints exist nowhere the repo can
   rebuild them from. Delete the three scripts and the finding above — that both
   fine-tunes score *below* the untrained backbone — stops being a live comparison and
   becomes an assertion nobody can re-run. That is the question
   [[unused-symbols-are-outdated-or-early]] says to ask of an uncalled symbol: who was it
   for. Here it is for a measurement that is still standing.

16. **Step 8's precondition holds: the features separate the classes, and they beat the
   geometric rule they would replace.** Asked before a trainer was written, because
   `analytics/pose_sequence.py` has been in the tree tested and with **zero consumers**,
   so nothing had ever measured what its features can tell apart, and building a pipeline
   first and discovering the features were the problem is the expensive order.

   `tools/temporal/ntu_project.py` (`runs/ntu_project01/`) takes NTU's official skeletons
   in real metres, gravity-aligns each clip by the subject's own standing pose, spins it
   to a sampled yaw, stands it at a sampled floor position inside our camera's view
   (height 2.38 m, pitch 50.2°, vfov 70.4° — the parameters `hm3d_cctv` renders at), maps
   Kinect's 25 joints to COCO's 17 and projects. **The projection is the domain
   adaptation**: the model consumes keypoints, so once the viewpoint is re-imposed there
   is no appearance gap left to cross. Linear model, features pooled over time by mean and
   max, **held out by performer** — NTU repeats each action per subject, and a clip-level
   split puts the same body on both sides:

   | pair | balanced accuracy | n | performers |
   |---|---|---|---|
   | pick_up vs sit_down | 0.924 | 137 | 11 |
   | **fall vs pick_up** | **0.896** | 137 | 11 |
   | fall vs stand_still | 0.892 | 148 | 11 |
   | fall vs stagger | 0.871 | 139 | 10 |
   | fall vs sit_down | 0.838 | 148 | 11 |

   **`fall` against `pick_up` is the pair both shipped geometric conditions fail** — 0.99
   against 0.92 on height (§7.14), 0.60 against 0.62 on angle (§7.11) — and the features
   read **0.896** on it. §7.14's tuned rule, at 96% recall and a 29% false rate, implies a
   balanced accuracy near 0.835, so a **linear** model on **time-pooled** features already
   sits above the rule before any temporal model exists. That is the floor step 8 has to
   beat, not the target.

   **What this is not.** It is not a trained behaviour head and not a transfer measurement:
   every number here is NTU projected into our geometry, and no site footage is in it. The
   sim-to-real question PLAN §2.3 leaves open — whether a model trained this way fires
   correctly on a real shopper — is still open and is step 6's to answer.

   **Three things the projection has to get right, each checked rather than assumed.**
   Gravity comes from the subject, not the sensor, because a Kinect's +y sits a median
   13.8° off vertical (§7.14). Yaw and floor position are sampled, because a model trained
   at one heading learns that heading and foreshortening is not uniform across a frame.
   And a placement that puts any joint outside the frame is **dropped rather than
   clipped** — a clipped skeleton is a different posture, not the same one at the edge.
   The pinhole is the shipped one: `_project` generalises `ground_to_pixel` by a single
   term and the tool asserts on every run that the two agree exactly at zero height.

17. **Step 8 has a model, and it passes — after failing first, which is the useful half.**
   `tools/temporal/train_posture.py`, two 1-D convolutions over time with masked mean and
   max pooling, **32,933 parameters** against §2.3's 100K budget, trained on NTU projected
   through our camera pose, **held out by performer**, no early stopping on the held-out
   fold.

   **First run, 505 sequences: it failed.** 0.931 on `fall` against `pick_up` where a
   linear model on time-pooled features read 0.944, and below that floor on three of five
   pairs. Reported as a failure rather than smoothed, because step 8's gate was written as
   a number and not as "it fires correctly on a watched clip" — under the usual wording
   0.931 would have been a success.

   **Second run, 2,925 sequences from 24 performers: it passes all five**, and against the
   like-for-like linear baseline on the *same* data rather than the smaller one:

   | pair | temporal model | linear, same data | linear floor §7.16 |
   |---|---|---|---|
   | **fall vs pick_up** | **0.963** | 0.926 | 0.944 |
   | fall vs stand_still | 0.966 | 0.965 | 0.932 |
   | pick_up vs sit_down | 0.952 | 0.940 | 0.932 |
   | fall vs stagger | 0.929 | 0.897 | 0.923 |
   | fall vs sit_down | 0.902 | 0.858 | 0.835 |

   Multi-class recall held out by performer: fall 0.853, pick_up 0.904, sit_down 0.845,
   stagger 0.841, stand_still 0.948. **So the first failure was data volume and not
   architecture, and the bar is what told those apart** — a demonstration gate would have
   accepted the first model and never asked.

   **A frame-rate correction sits underneath all of it.** NTU records at 30 fps, this
   fleet analyses at 5 (§7.4), and `sequence_features` takes a **per-frame** difference for
   its velocity term — so the rate is inside the features and a 2.4 s fall is 72 frames at
   one and 12 at the other. Resampling at the source moved the linear floor from 0.896 to
   0.944, which is the direction that makes sense: a 33 ms difference is mostly noise and a
   200 ms one is the movement.

   **What is still not true.** No site footage is in any of this. The model has never seen
   a real shopper, PLAN §2.3's sim-to-real transfer is untouched, and it is **not wired
   into the serving path** — `events/pose.py` still decides `fall` by geometry, now with
   §7.14's measured 0.7 s sustain. Step 6 is where a projected-data model meets a real one,
   and until then this is a number about NTU.

18. **Two-stage tracking's identity was measured, the instrument was wrong first, and
   fixing it is a component step 9 also needs.** Opened 2026-08-27 against decision 1:
   `runs/endings05/` cuts mid-view track deaths 111 → 38 and doubles track length, and the
   objection to adopting it was that the Kalman coasts onto neighbours and inflates dwell.
   **Length cannot tell those apart — recovering a chopped-up shopper and merging two
   shoppers produce the same number.** Only identity can, which is what §6 step 5 already
   says: "distinguishing 'the same person, a metre on' from 'a different person, a metre
   away' is what an appearance model is for."

   **The design, and it needs no chosen threshold.** `scripts/track_identity.py` runs both
   trackers over the **same detections** in one inference pass. A **join** is where a
   two-stage track changes single-stage identity — it stitched fragment A to B — and the
   distance between their mean torso colours is read against two references built from the
   same statistic on the same clip: `within`, one single-stage track's first half against
   its second (**same person by construction**: the shipped tracker never associates across
   the low band, so it cannot make this mistake), and `between`, two single-stage tracks
   **co-present in a frame** (**different people by construction**). The overlap of the two
   is itself part of the answer.

   **First run (`runs/identity01/`, 8 cameras, 7,200 frames): 127 joins, 124 with a usable
   reference, `pct_between` median 0.03, nine past the between-person median.** The
   references separated best exactly where they had to — Kaohsiung-cam04, the crowded
   camera with 78% mid-view deaths, went 87 → 29 tracks over **69 joins with none flagged**
   at `within` p50 0.413 against `between` p50 4.401 over 312 pairs; Tao-Hsin-cam03 (83%)
   ran 18 joins with one flagged. Three cameras have no usable `between` population at all
   (n = 0, 1, 1) and were not read; all three are near-empty shops, so **the instrument
   fails where it does not matter.**

   **Then the nine were looked at, and all nine are one person each** (`runs/identity02/`,
   crop strips either side of each join, read by eye). The cause is the same in all nine
   and it is in the instrument: **`TORSO_BAND` is a fraction of the detected box**, and a
   box reframes along a track — full body at the back of a shop, tight upper body walking
   under the camera, clipped at an edge. When it reframes, 18–55% of the box moves off the
   shirt onto hair or trousers and the nine statistics change completely for somebody who
   did nothing. The chain that looked most like the feared failure — Taichung-cam01's
   `3→6→7→9`, 565 observations absorbing three more tracks — **is one member of staff in a
   blue polo walking toward the camera and away again.**

   `tests/test_appearance.py` had already written down the risk in the abstract ("where the
   window sits is not something a reader can check, and it silently stops meaning anything
   if a crop convention changes"). What it did not anticipate is that the convention
   changes **inside one clip**, which is the case that costs something.

   **The fix is `appearance.torso_region`**: shoulders and hips from the resident pose head
   (PCK@0.2h 0.915) bound an actual torso, which the box cannot move. It refuses — returns
   `None` — on anything less than **one shoulder and one hip**, because two hips fix the
   wrong end of a torso and a region built from them is a fraction of something again
   wearing a keypoint's name; a caller that falls back to the band **records that it did**,
   since the fallback is a different measurement rather than a degraded one. Fleet fallback
   rate: **20,217 of 21,473 detections anchored, 94.2%**, lowest on Taichung-cam11 (88.1%)
   and Kaohsiung-cam04 (91.1%) — the near-empty camera and the crowded one.

   **The re-run is a controlled comparison and it was predicted wrongly.**
   `runs/identity03/` (and `runs/identity04/`, which adds the split below to the artefact
   and reproduces all 127 distances exactly) produces **the same 127 joins** as
   `runs/identity01/`, because the
   tracker did not change and only the statistic did. The prediction written down before it
   ran was that Taichung-cam01's chain would stop being flagged; **two of its four flags
   survived, and Kaohsiung-cam04 and Taichung-cam04 each grew a new one**. Flags went 9 → 8,
   which read at first as the fix having achieved nothing.

   **It had, and the count was the wrong statistic.** Read as distances rather than
   percentiles, the fix separates the references it should: `within` p50 / `between` p50
   improves on five of the seven cameras that have a reference, and **Tao-Hsin-cam04, where
   the instrument had been inverted — one person reading *further apart* than two people,
   1.996 — comes back to 0.317**. Then the survivors have one thing in common:

   | | joins | flagged, box band | flagged, keypoints |
   |---|---|---|---|
   | **both sides ≥ 6 observations** | **67** | **3** | **0** |
   | one side a 2–5 observation stub | 57 | 6 | 8 |

   Zero on **every camera separately**, not only in the total, and the largest share of it
   is Kaohsiung-cam04's — **35 well-observed joins on the crowded camera, none flagged**.

   **Every surviving flag is a stub join**, and the floor that exposes it was already in the
   code: `within` is built from halves of tracks with at least `MIN_OBS_FOR_HALVES = 6`
   observations, and joins were being read against it with no floor at all. A 2-observation
   mean colour is one moment's lighting. That inconsistency was mine, built in when the
   reference was written, and the script now reports the two populations as a split rather
   than filtering the short ones away — **"46% of joins cannot be judged this way" is a
   result, and a silently shorter list is not.**

   So the measurement, stated at the end of both fixes: **of the 67 joins where two-stage
   association stitched two well-observed fragments, none joins two people the appearance
   can tell apart.** The three that the box band flagged among those 67 are exactly the
   three read by eye in `runs/identity02/` and found to be one person each, so the two
   corrections agree.

   **What this still does not decide.** A colour proves a join wrong and cannot certify one
   right: two shoppers in the same colour are the same nine numbers, and `between` is what
   prices how often that happens in these shops. The industry answer to decision 1 is
   IDF1/HOTA against hand-labelled track IDs, `analytics/reid_metrics.idf1` has been armed
   since it was written and has never run on a site clip, and `scripts/track_review.py`
   exists to make that labelling minutes of judgement rather than hours of drawing. **That
   is the measurement decision 1 should be settled on**; this one narrows what has to be
   labelled and has already removed the reason to think two-stage is unsafe.

   **One premise above is false, found 2026-08-28 while labelling that clip, and it is the
   one the `within` reference is built on.** This section twice says the single-stage
   tracker "never associates across the low band, so it has no way to make this mistake" —
   `scripts/track_identity.py`'s docstring says it a third time — and uses it to call one
   track's first half against its second half *same person by construction*. On
   Taichung-cam01's 900-frame clip, **proposed track 8 is two people**: frames 711–820 are
   the long-haired member of staff crouching at the left counter, and from **f821** the box
   is on the one in the blue hooded jacket standing beside her, who is separately boxed as
   track 5 five frames earlier. The box height goes **211, 227, 242, 424** across f818–821,
   and the crop read by eye at f821 has the first person still crouching inside it after
   the jump — a local figure from `scripts/track_review.py`, never published, because
   `assets/` requires a passing face audit and this is an unaudited shop-floor frame. Two people came together, and the high band alone was enough.

   What that costs: `within` on this camera pools at least one pair that is two people, so
   the same-person reference is **wider than it should be**, which biases the instrument
   toward calling a two-stage join ordinary. The direction is the safe one for the
   conclusion above — it makes flags *less* likely, and the conclusion was that none of the
   67 well-observed joins flagged — but "by construction" is now "on this evidence", and a
   camera where the shipped tracker merges more would loosen the reference further. The
   fix is not a threshold: it is that `within` should be built from labelled identities
   where they exist, which for the first time they do.

   **Decision 1 is answered, 2026-08-28, on two labelled clips — and the answer took three
   goes, two of which were the instrument rather than the trackers.** The false starts are
   kept because the last one is the reusable lesson: *first* it was called settled on
   Taichung-cam01 alone; *then* Tao-Hsin-cam03 appeared to reverse it, two-stage identity
   precision collapsing 0.508 to 0.235; *then* the reversal turned out to be
   `track_idf1.py`'s own doing. The ground truth is labelled from a proposal made at
   `score_thr 0.25`, the two-stage tracker's entire mechanism is the band underneath it,
   and **196 of its 443 observations on that clip (44%) sat below 0.25 against 0 of the
   single-stage arm's 238.** A box the labels cannot contain is an identity false positive
   *by construction*, so two-stage was losing twenty IDF1 points for doing the thing it
   exists to do. `high_band_only()` now drops every observation below the labelled band
   from both arms before scoring — tracking still uses the low band, which is its job, and
   scoring uses the band the labels cover, which is also the band a serving path emits.

   With that fixed, both clips agree and the self-check passes on both (the single-stage
   arm reproduces `tracks.json` exactly, which is what licenses reading the other):

   | | tracks | IDF1 | IDP | IDR | switches | tracks/identity | low band dropped |
   |---|---|---|---|---|---|---|---|
   | **Taichung-cam01** (5 identities, 42% mid-view deaths) | | | | | | | |
   | single-stage (shipped) | 10 | 0.7388 | 0.7388 | 0.7388 | 6 | 2.20 | 0 / 2,343 |
   | **two-stage** | 8 | **0.7418** | 0.7418 | 0.7418 | **3** | **1.60** | 6 / 2,349 (0.3%) |
   | **Tao-Hsin-cam03** (4 identities, 83%) | | | | | | | |
   | single-stage (shipped) | 18 | 0.5095 | 0.5084 | 0.5105 | 14 | 4.50 | 0 / 238 |
   | **two-stage** | 15 | **0.5455** | **0.5344** | **0.5570** | **7** | **2.50** | 196 / 443 (44%) |

   **Two-stage wins on both, and wins by more where fragmentation is worse** — +0.003 IDF1
   on the mild camera, **+0.036** on the severe one — with identity precision up rather
   than down on both. The objection this section was written to test is not observed. The
   dropped-band column is the mechanism in one number: on Tao-Hsin-cam03 the low band
   carries 44% of the two-stage arm's observations and on Taichung-cam01 it carries 0.3%,
   which is why the gain follows fragmentation.

   **And what neither tracker touches is still the larger figure.** `idfn` is 612 → 605 on
   cam01 and 116 → 105 on cam03: **26% and 49% of the labelled boxes lost to fragmentation
   under a one-to-one identity mapping**, of which association tuning recovers 7 frames and
   11. The loss decomposes exactly on cam01 — identity 1 is 873 frames split across `#1`
   (433), `#5` (363), `#8b` (56) and `#13` (21), so a one-to-one mapping keeps 433 and
   charges 440; identity 3 charges 172 the same way; 440 + 172 = 612 — and the gaps
   responsible are **24 frames** (`#1`→`#5`) and **61** (`#5`→`#13`) against `max_age = 5`.
   No association over a five-frame coasting window bridges a twelve-second absence.
   Tao-Hsin-cam03 makes the point without arithmetic: its dominant identity is a member of
   staff leaning on a counter for 58 s **in thirteen fragments**, all at the same image
   position, with **no box at all** in the gaps while the blue sleeve is plainly on the
   counter, read by eye off the same `track_review.py` crops.

   **Step 5's other sentence is measured now, 2026-08-28, and on one of the two cameras it
   fails** (`scripts/zone_dwell.py`, `runs/gt_*/zone_dwell.json`). The gate reads "dwell/
   loiter durations survive — an ID switch mid-loiter resets the clock, so switches on the
   watched clips must be rare enough not to", and IDF1 does not answer it: what the layer
   above reports is a duration in a zone, so the product's own chain — `world_frames` →
   `journeys` → visits in metres through the camera's commissioned geometry — is run three
   times over the same clip, over the labelled identities and over each tracker's output.

   **Taichung-cam01 passes.** 35 true visits totalling 335.2 s; both arms report 35 visits
   totalling 326.6 s and 328.2 s, survival median 1.00, and **5 of the 6 loiters ≥ 10 s
   are kept whole**. One single-stage visit spans two identities and it is exactly track 8,
   the two-person track above; two-stage has none.

   **Tao-Hsin-cam03 fails, and not narrowly.** The truth is **4 visits totalling 68.6 s**.
   The shipped tracker reports **17 visits totalling 42.2 s** — four times the count and
   38% of the time gone. Its one loiter is the member of staff at the counter, `fixture_05`,
   **57.6 s**, and it arrives at the event layer like this:

   | | reported visits covering it | longest single | as a fraction |
   |---|---|---|---|
   | single-stage (shipped) | **13** | 4.8 s | **8%** |
   | two-stage | 5 | 24.4 s | 42% |

   Single-stage splits it into 4.8, 4.0, 3.6, 3.0, 3.0, 2.4, 2.0, 2.0, 1.6, 1.6, 1.6, 1.6
   and 1.2 seconds. **A person who stood there for a minute is a 4.8-second dwell**, so any
   loiter rule with a threshold above five seconds cannot fire on this camera at all —
   which is the gate's own sentence about the clock, with a number on it for the first
   time. Two-stage takes 8% to 42%: a real improvement, and still not "survive".

   **So step 5 does not pass on the fleet, and the failing half is not the geometry.**
   §7.13's 7.3 cm stands; what breaks is that the identity a duration is attached to does
   not last as long as the duration. Step 6 is gated on step 5 and should stay gated:
   an L3 event log built on this camera would report a shopper's minute as thirteen
   visits, and every one of them would be true.

   **So: adopt two-stage, and do not expect it to fix fragmentation.** Bridging 5–12 second
   absences is what §6 step 5 already names an appearance model for, and this is the first
   measurement that prices it — 26 to 49 IDF1 points of headroom, against the 0.3 to 3.6
   that association tuning returned.

   **Two more clips were proposed for this and neither could be labelled, which is itself
   a measurement about the fix above.** `runs/gt_cam04_v3` (Kaohsiung-cam04, 300 frames,
   **45 tracks**) and `runs/gt_tc04` (Taichung-cam04, 900 frames, **26 tracks**) are both
   built with the shipped detector and both instrumented, and both were abandoned at the
   reading stage: on the first, four tracks wear the same maroon top, two the same black
   vest and ponytail, two the same cream ruffles, under a near-top-down view; on the
   second, **nearly every track is a blue polo torso at the same scale** — `19, 21, 22, 23,
   25, 27, 28` are not separable by eye at all. A ground truth that over-fragments
   penalises the tracker that correctly merges, which is the direction that would flatter
   the conclusion above, so guessing was refused rather than hedged.

   **That failure is evidence about the appearance model, not only about the labeller.**
   The 26-49 points priced above are priced for something that has to tell these people
   apart, and in these stores the staff wear one shirt — the same fact §7.15 measures as
   `staff/customer` being carried by nine torso-colour statistics that fail on the camera
   whose staff wear jackets, and the same fact §7.18's `between` reference exists to price.
   **Whatever bridges a twelve-second absence here cannot be torso colour**, and the two
   clips that could be labelled are the two where people happened to look different.

   Caveats, all of them: **two clips, two cameras, nine identities**, and the labels were
   read by a model rather than by a person (both `provenance.json` files say so in their
   own words). The labels cover the detector's boxes only, so this measures association and
   not detection — which bites hardest exactly where the fragmentation is, since a shopper
   the detector never saw is absent from the truth as well as from the prediction. A third
   clip, `runs/gt_cam11`, is labelled against `configs/hydranet_indoor.yaml` — a different
   detector — so `track_idf1.py` refuses it and it is not quoted here.

19. **"Scale-measured" meant the person-height prior all along, and a second fleet arrived
   that has more of it than the first.** Opened 2026-08-27 when a new corpus —
   `gs://syncai-rtsp-recordings`, ten RTSP channels of a second store, wood floor, not
   STUDIO A — was asked for 3D scenes and the first answer given was "these need a
   physical reference first". **That answer was wrong, and checking it corrected this
   plan rather than the new cameras.**

   **Every shipped camera's scale comes from the 1.70 m adult prior.** Read straight out
   of `runs/onboard01/*.calib.json`:

   | camera | height | `scale_source` |
   |---|---|---|
   | Taichung-cam01 | 2.49 m | `person_height_median_vs_1.7m_prior_n37` |
   | Taichung-cam10 | 2.87 m | `..._n31` |
   | Tao-Hsin-cam04 | 2.91 m | `..._n22` |
   | Tao-Hsin-cam03 | 2.61 m | `..._n21` |
   | Taichung-cam04 | 2.30 m | `..._n19` |
   | Kaohsiung-cam04 | 2.17 m | `..._n15` |
   | Taichung-cam11 | 2.29 m | `..._n15` |

   And **vfov is a fleet assumption on 22 of 23**: `vfov_source: fleet_hardware_assumed`
   at 70.4°, pinned by tile grid on Taichung-cam01 alone. The 14 cameras "blocked on a
   visual reference" are `scale_source: unmeasured`, which `onboard_camera.py` emits when
   **fewer than 10 person boxes** pass its gates — they are short of people, not short of
   a tape measure. `visual_reference.status` is `needs_visual_reference` on cam01 too,
   the camera whose numbers everything else is compared against.

   **So the new fleet is not held to a bar the old one clears.** People-only pose fitting
   over `plate_calibration.fit_pose_from_people`, with the shop-ceiling prior 2.2–3.6 m
   and vfov 70.4:

   | ch | person boxes | fit | note |
   |---|---|---|---|
   | ch2 | **1,425** | 2.47 m / 35.5°, spread 0.043 | interior solution |
   | ch3 | **468** | 2.58 m / 36.5°, spread 0.081 | interior solution |
   | ch8 | 838 | 2.20 m / 32.0° | **on the 2.2 m bound** |
   | ch5 | 37 | 2.22 m / 38.0° | on the bound, 37 boxes |
   | ch6 | 27 | 2.27 m / 50.5° | 27 boxes |
   | ch1 | 33 | no fit | floor only in one corner |
   | ch4 | 405 | no fit | people stand behind a counter, so box bottoms are not feet |
   | ch7 | 1,251 | no fit | same, and 1,251 boxes do not fix it |
   | ch9, ch10 | 22, 45 | — | near top-down; **0 in a first 300-frame sample, which was the
     sample and not the camera** — over 6 clips x 720 frames they yield 22 and 45, against
     ch2's 1,394. Whether that clears `onboard_camera.py`'s ten-usable-heights floor is the
     tool's answer, not a judgement made here |

   ch2 and ch3 rest on more evidence than **any** shipped camera. Three cautions kept
   because they are what the table cannot show: **a fit pinned to the bound of a prior is
   the prior, not a measurement** (ch5, ch6, ch8); ch2 is the doorway camera, so its plane
   is the pavement rather than the selling floor; and pitch is coupled to vfov — 65° vs
   70.4° moves ch3 to 34.0°/2.66 m, about **2% of scale**, which is fine for a layout and
   not fine for a claim at §7.13's 7.3 cm.

   **The prior did all the work and the first run had it wrong.** The first sweep passed
   `heights=(1.0, 8.0)` and returned 1.0–1.2 m at 5–9.5° for every vfov — a camera sitting
   on a table — with pitch pinned at the search's own lower bound and the objective flat
   across vfov 50–80. That was read as "these cameras cannot be calibrated from video".
   It was the range being wrong: with 2.2–3.6 m, **all 468 boxes enter the fit** instead
   of 318–348, and height comes out stable across every vfov. A boundary solution is not
   a failed camera, it is a question asked with the wrong prior.

   **One check attempted and void, recorded so it is not retried.** The vertical vanishing
   point of standing people would confirm pitch independently — but a detector box is
   *axis-aligned*, so its side edges are exactly vertical in the image by construction,
   every such line is parallel, and the least-squares intersection is degenerate. The
   working version of that check needs **keypoints** (mid-ankle to mid-shoulder leans; a
   box does not), and the pose head is resident.

   Corpus facts found on the way: hevc, 1920x1080, **6 fps**, 10-minute files. **Some are
   never finalised** — `moov atom not found`, identical after a clean re-download, so the
   file in the bucket is broken rather than the transfer. ch7's 10:00, 12:00 and 14:00 are
   three in a row; ch5's 14:00, ch3's and ch9's 16:00 are singles. The rate over a whole
   day is not measured and matters for whether this corpus can be trained on.

20. **The `person` detector has never seen a labelled shopper from these stores, and half
   the site training images teach it that shoppers are background.** Opened 2026-08-28 to
   confirm §7.11's premises before spending its retrain, and both of them moved.

   **No config trains detection on site person labels.** Read across every yaml:
   `pose01`, `pose02`, `security_b03` and `security_b03_cw_xl` all supervise detection
   from `retail_objects_batch02` + `batch03` + `coco_person`. Those two site datasets
   carry **`boxed_stock` and `device` only** — 4,758 and 5,839 annotations, zero people.
   So the shipped retail `person` channel is trained from **COCO at `sample_ratio 0.05`**
   and nothing else. `datasets/site30k_v1` holds **47,465 site person boxes** and is
   wired into the same configs for **pose alone**.

   ~~**And the site images are not neutral about it.**~~ **Withdrawn 2026-08-28, and the
   mechanism it says is missing is the one already shipped.** The claim was that
   `detection.py` has no ignore mechanism, so each of the **69 of 120 batch02 training
   images that contain a person (58%), and 63 of 120 in batch03 (52%)**, teaches the
   `person` channel that a shopper is background. Those two counts stand; the consequence
   does not. `DetVocab.class_mask` is exactly that mechanism at channel granularity, it is
   consumed by `FCOSLoss`, and `label_maps_retail_security.py` was written for this case
   and says so in its own header. Built from `hydranet_retail_pose02.yaml` and read off
   the objects: **`site_boxes` and `site_boxes03` both carry `class_mask [0, 0, 1, 1]`**
   — `person` is neither rewarded nor punished on any batch02/03 frame — against
   `coco_person`'s `[1, 1, 0, 0]`. So the site frames are neutral about `person` by
   construction, which is the same fact as the paragraph above stated from the other side:
   masking is why `person` is trained on COCO **alone**.

   **This explains §7.11 better than the assigner does, and it explains the parts the
   assigner could not.** Centerness is identical across quiet, crowded, good and bad
   cameras (§7.11) because centerness is regressed on positives and **there are no site
   person positives at all**. `cls` falls in crowds because a counter crowd at 3-5 m under
   a 50-degree ceiling mount is as far from COCO as this fleet gets. Score rising with
   person size, and the two worst cameras being the two most unlike COCO, follow the same
   way. §7.11's own mechanism — FCOS giving a location to the smallest box containing it
   — additionally required checking: when the stealing box is another *person* the class
   target is still `person`, so what it corrupts is the regression, not `cls`. A class
   flip needs a smaller **non-person** box, and that case cannot arise in training here,
   because no site frame carries a person box for a bag to steal from.

   **So decision 2 changes shape.** The intervention is not a crowd-aware assigner
   (ATSS/OTA) and not new labelling: it is **putting `site30k_v1`'s person boxes into
   detection supervision**, where they already sit on disk inside the same configs. It
   fixes the missing-label pressure in the same move, because those frames then carry the
   labels they were missing. The cost to state: they are Grounding DINO teacher output
   thresholded at 0.35, score p10/p50/p90 **0.43 / 0.60 / 0.70** — teacher error, not
   human ground truth, and `stack`-class comparability (§7a.1) still applies to any
   vocabulary change.

   **The first gate is answered 2026-08-28, and it needed one fact nobody had read off the
   file**: `site30k_v1/annotations/instances_train.json` is **not person-only**. It carries
   **person 47,465, boxed_stock 168,970 and device 117,893 — 334,328 boxes over 16,146
   images** — against batch02+batch03's **10,597** product boxes. Added whole it would not
   disturb the product classes, it would replace them, at better than 30:1, with teacher
   output on the same footage.

   It does not have to be added whole. `classes:` on a `coco` dataset already does this and
   `coco_person` already uses it; the restriction is not a filter bolted on top but the
   thing that drives `class_mask`. Built and read off the object: **`classes: [person]`
   over `datasets/site30k_v1` gives 14,654 training images and `class_mask [1, 0, 0, 0]`**
   — every product channel gets no gradient on those frames, and the 286,863 product boxes
   are left out rather than mapped somewhere. So the intervention is one dataset block, and
   the gate it was blocked on cannot fire.

   Sizing, for the same reason: `coco_person` is 65,744 images at `sample_ratio 0.05` =
   **~3,287 images an epoch**, against `site_boxes` 120 x 3.0 = 360 and `site_boxes03`
   144 x 1.0 = 144. A `site_person` block at **0.2** draws ~2,931 an epoch, which makes
   site footage half of the `person` supervision rather than all of it — the first run
   should be able to attribute a change to the new labels, and a ratio that drowns COCO
   cannot.

   ~~**Still open, and it still gates the run**: what `primary_metric` the run selects
   on.~~ **Decided `detection_mAP/site_person`** — taken in
   `configs/hydranet_retail_person01.yaml` on 2026-08-28 and confirmed 2026-09-01 before
   spending the card. It is the head the run exists to move, on the footage it exists to
   move it on, which is what §7.9's trap asks for; `detection_val_interval` is therefore 1,
   which `config_schema._check_detection_val_interval` requires.

   **The caveat in the question above is not answered by the choice, it is answered by
   reading three numbers.** Those val person boxes are Grounding DINO at 0.35, the same
   teacher as the training labels, so this metric rises with agreement and cannot
   distinguish a better detector from a better imitation. It selects `best.pt`; it is not
   the run's verdict. The verdict is:

   | number | what it answers | prior |
   |---|---|---|
   | `detection_mAP/site_person` | did the new labels move the head at all | none — first run |
   | `detection_mAP/coco_person` | is real person detection still there | 0.2106 (pose02 `last.pt`) |
   | `detection_mAP/site_boxes03` | did the product classes pay for it | 0.1496 (pose02, epoch 65) |

   A `site_person` gain bought with a `coco_person` collapse is the failure mode, not the
   result. Against pose02 the comparison is **`last.pt` to `last.pt`**, both at epoch 120
   of the same recipe, for the reason 7a.28's table is built that way: `best.pt` is
   selected by one head and is a different question.

   **What made this safe to leave as-is rather than re-pointed at `coco_person`**: until
   `2387ef6` the selection report could not see any detection metric at all, so a
   checkpoint that traded real person detection for teacher agreement would have shipped
   with `selection.json` naming `terrain_mIoU` and nothing else. It now prints the trade.

   **ANSWERED 2026-09-02: the site person boxes help every head and cost nothing
   measurable.** The run completed 07:33, all 120 epochs, `runs/hydranet_retail_person01`.
   `last.pt` to `last.pt` against pose02, same recipe, one dataset block apart:

   | metric | pose02 | person01 | delta |
   |---|---|---|---|
   | `detection_mAP/coco_person` | 0.2106 | **0.2157** | +0.0050 |
   | `detection_mAP/site_boxes` | 0.1246 | **0.1444** | +0.0198 |
   | `detection_mAP/site_boxes03` | 0.1446 | **0.1518** | +0.0071 |
   | `detection_mAP/site_person` | n/a | 0.7386 | new |
   | `pose_PCK@0.2h` | 0.9345 | 0.9327 | -0.0019 |
   | `terrain_mIoU/site_seg03` | 0.5843 | **0.6718** | +0.0874 |
   | `terrain_mIoU/ade20k` | 0.6798 | 0.6770 | -0.0028 |

   **The failure mode did not occur.** `coco_person` is the teacher-independent number --
   COCO person boxes are human-labelled -- and it rose. The run did not buy teacher
   agreement with real person detection. Read the shape rather than the endpoint, though:
   person01 was far ahead early (0.1890 at epoch 15 against pose02's 0.1342) and the lead
   closed to +0.0050 by epoch 120. What the site labels bought on COCO is **speed**; the
   endpoint difference is within what one seed can say.

   **The largest gain is not in detection and not in the `person` class.** Terrain rose
   +0.0874 on `site_seg03` and -0.0028 on `ade20k`, and the per-class split is floor
   +0.0739, wall +0.0620, fixture +0.0560, `05_person` **-0.0047**. So this is not the
   `person` channel helping itself: the block's 14,654 site images supervise detection
   only and carry `class_mask [1, 0, 0, 0]`, but they still pass through the shared
   backbone and neck, and they are the same domain as `site_seg`/`site_seg03`. The site
   person block acted as site-domain representation data. `03_column` reads +0.1158 and
   should not be quoted: it is the class that scores on val and predicts nothing in the
   store, so a val IoU on it is not a measurement of the class.

   **`detection_mAP/site_person` 0.7386 is agreement with Grounding DINO at 0.35, its own
   training teacher, and has no prior.** It is not 74% accuracy at finding people and must
   not be quoted as one.

   **Costs, all visible for the first time because of `2387ef6`.** `best.pt` is epoch 118
   and gives up 0.0341-0.0415 of terrain mIoU, which peaked at epochs 18-22; every
   detection head and the pose head sit within 0.001 of their own best at that checkpoint.
   Before that commit `selection.json` would have carried one line and none of this.

   **Provenance.** systemd-oomd SIGKILLed the unit five times (00:19, 00:55, 01:12, 01:37,
   02:12) on PSI memory pressure, not real OOM -- a second training run,
   `quadhydra-train`, started at 00:51 and competed for RAM and the card. Each restart
   resumed from `last.pt`; all 120 epochs are present, with epochs 34 and 40 logged twice
   where a resume re-ran them. Wall clock 8 h 33 min. **No throughput or epoch-time figure
   from this run is usable** -- it shared the GPU for most of its length.

   **Single seed.** These deltas have no seed-variance context; `hydranet_retail_security_seeds`
   is the shape that would give them one.

   **PROMOTED 2026-09-02** (the user's call): `shipped.SHIPPED_RUN` now names this run.
   Both `for_terrain` and `for_detection` return `last.pt` -- the saved checkpoints
   coincide to three decimals because selection picked epoch 118 off a flat tail; the
   per-head doctrine stands and the reasoning is on the functions. Every tool renders
   with person01 from here on, so its numbers may now be quoted beside the figures.

   **Pre-flight, 2026-09-01, built on CPU from the config rather than read off it**:
   `split_leaks` clean, `check_config` silent, `site_person` 14,654 train images at 0.2 →
   91 detection steps carrying `class_mask [1, 0, 0, 0]`. `detection_class_steps` is
   person 193/206 (94%), bag 102/206 (50%), `boxed_stock` and `device` 13/206 (6%) — the
   product classes fall as a share and not as an amount, 13 steps before and 13 after.
   The epoch grows 287 → 378 steps (1.32x; the *detection* steps grow 115 → 206, 1.79x)
   and detection validation grows to 11,609 images every epoch. Estimated ~9 h from
   pose02's measured 139 s non-detection epoch plus 26 s per 2,961 detection val images,
   so a 23:00 start finishes around 08:00 — consistent with the timer unit's own estimate.

21. **`masks_pass` is not the bev-3d bottleneck, and three explanations for the missing
   furniture are ruled out.** Opened 2026-08-28 on the handoff's statement that the render
   covers about a third of the store because "half the furniture is never found". The
   apparatus is `tools/commissioning/masks_diagnose.py` (the recipe's front half, with the
   per-cluster verdicts kept instead of printed as a total, plus the SAM 3 proposals cached
   bit-packed so a later rule costs no GPU) and `tools/commissioning/cluster_rules.py`
   (replays those proposals through an alternative merge rule and paint order into the
   recipe's own `decide_structure`). Records in `runs/masks_diag01/<camera>/`; the offline
   replay reproduces the online run exactly on all three cameras checked first, which is
   what licenses the comparison.

   **The furniture is proposed.** Tao-Hsin-cam03's largest object is 663 kpx -- 32% of the
   frame, one blob spanning the left counter run, the right counter run and the wall behind
   both -- and it ranks `wall 0.969 / table 0.945`. The counters are in the vote record.
   Fleet-wide the first pass accepts **120 walls against 41 fixtures** over eight phone
   shops (wall 20/18/14/28/14/9/8/9, table 7/8/2/3/4/1/2/2, shelf 3/1/0/3/2/0/1/2).

   **`cluster` treats image containment as identity, and that is a real defect.** A
   proposal joins an existing object at IoU >= 0.6 **or** when 85% of it lies inside that
   object, and the object's mask is frozen to the largest proposal that arrived first. A
   counter standing in front of a wall is 100% inside the wall's mask from any single
   viewpoint, so it can never become its own object: **263 of 503 merges on Taichung-cam01
   are containment merges, 266 of 325 on Tao-Hsin-cam03**. Guarding it -- containment
   counts only between masks of comparable size (`sized`, ratio 0.5) -- splits them back
   out: objects 11 -> 46 on Tao-Hsin-cam03, accepted `shelf` 1 -> 13 on Taichung-cam04,
   2 -> 9 on Kaohsiung-cam04.

   **And it changes nothing downstream, which is the finding.** `scene_mesh` opens the
   class mask PNGs and takes connected components (`syncai_bev3d/scene_mesh.py`, in
   `cell_grids`); the object identity clustering produces is discarded before the scene
   is built. Measured on the
   painted map, all four arms -- current/sized x paint-by-score/paint-by-area -- give
   Tao-Hsin-cam03 **72.6, 72.8, 72.6, 72.8 % wall with `table` at 7.3% and 4 fixture
   components in every one**. Taichung-cam04 and -cam11 do not move either. Only
   Taichung-cam10 (wall 14.4 -> 7.5, shelf 26.5 -> 31.2) and Tao-Hsin-cam04 (wall
   64.5 -> 57.0, table 9.7 -> 19.6) gain, and Kaohsiung-cam04 loses shelf (8.1 -> 5.6).

   **What is left is the one already named in step 2: on a white-fixture store both
   teachers call a counter a wall.** After splitting, Tao-Hsin-cam03's counters are still
   classified `wall` -- SAM 3's `wall` prompt reads 0.94-0.97 on them and b03's wall share
   is 1.00, so `decide_structure`'s wall gate passes on the first test it applies.
   **`flat` cannot arbitrate it**, and that was checked before writing any code:
   Taichung-cam01's k14 is a wall at flat 0.92 and Taichung-cam07's k13 a wall at 0.62,
   while four of Tao-Hsin-cam04's accepted tables read 0.17-0.26. `flat` answers "how much
   of what I can see of this is horizontal", which is a fact about the viewing angle.

   **Three plausible fixes are therefore ruled out by measurement rather than by argument**:
   more or better prompts (the fixtures are already proposed), a merge-rule fix (the map
   does not move), and a paint-order fix (nor does that). The clustering guard is still
   worth landing for the object record -- `shelf_rois_px` and any future per-object
   consumer read it -- but it is not a coverage fix and must not be quoted as one. Note
   the blast radius before landing it: `cluster` and `decide_structure` are the same
   functions `recipe.py:794` uses to generate the **site30k segmentation labels**, so a
   change there changes the next dataset and breaks comparability with `site30k_v1`.

   One thing found on the way and not chased: on Taichung-cam11 three of the clusters
   rejected as `no column claim, class inherited` are **chairs**. No prompt in the
   `fixture` family names a chair or a stool, so a chair can only be picked up by the
   `column` prompts and is then rejected by the column gate. Fleet-wide that reason
   accounts for 14 of the 52 rejections.

22. **A merchandise wall was being drawn as a small cabinet on every camera turned the
   other way, and the tripwire could not see it.** Found 2026-08-28 by a reviewer asking
   why the README figure's fixtures were placed as they were — the render, not a number,
   which is the third time today that a defect surfaced only when somebody opened the
   picture (§7.21, and the p85 wall heights before it).

   `scene_mesh.build_scene_regular` fits each component's extents in the store frame and
   then caps the shelving depth: `min(d, SHELF_MAX_DEPTH_M)`, where `d` is the extent
   along **v**. That is the depth only if the run lies along **u**. Taichung-cam10's three
   shelf components measure **u1.33 x v6.29, u0.97 x v2.70 and u0.55 x v1.08** — so the cap
   took a **6.29 m accessory wall down to 0.45 m** and left its 1.33 m depth untouched. The
   store's whole merchandise run arrived in the scene as a **1.35 x 0.45 m cabinet standing
   in the aisle**, and the same happened on Tao-Hsin-cam04 (2.74 m -> 1.38).

   **`PLAUSIBLE_M` is why it survived a round that added a tripwire.** `display_shelf`
   allows span (0.4, 9.0) and short (0.2, 1.4); 1.35 x 0.45 sits inside it exactly as
   6.30 x 0.45 does. A truncated run is a plausible small unit, so the check that exists
   to say "that is not furniture" had nothing to say. **A plausibility interval catches a
   shape that is impossible and never one that is merely wrong.**

   **And the camera it was measured on could not show it.** The 0.45 m cap was read off
   cam11's plates in the same round that introduced it, and cam11's runs are
   u4.02 x v0.40 and u2.36 x v0.42 — long side on `u`, where the cap is a no-op. One
   camera's orientation was enough to make the parameter look right.

   Fixed by capping whichever side is shorter, which the `wall` branch fifteen lines above
   has always done, plus a quarter turn of the placement when the run lies along v.
   Fitted shelving before -> after: **cam10 1.35/0.95/0.55 -> 6.30/2.70/1.10**,
   Tao-Hsin-cam04 1.38 -> 2.75, **cam11 4.02/2.36 unchanged** — which is the regression
   check that matters. The back panel now also faces away from the room's centre; nothing
   had asked where `shelving`'s back pointed, and on cam10 it pointed into the aisle.

   **Found underneath it: the eye is a fixed (+x, -z) diagonal and nothing asks whether
   something tall lies along it.** With the 6.3 m wall restored, the commissioning still
   for cam10 was taken from behind it. Scoring the four diagonals by occlusion was tried
   and reverted the same hour — it fixes cam10 and makes cam11 worse, so which corner to
   stand in is a composition decision rather than a scoring one. The demo video's panel
   was never affected, because `demo_video.content_crop` reframes on the content after
   rendering.

   **Answered 2026-08-29 (`ad78d08`), by fading the occluder rather than moving the eye.**
   The composition is untouched, so no earlier framing judgement is invalidated; the
   offending object stays visibly present and the room behind it becomes readable, which is
   what an architectural viewer does with a near wall. The measure is deliberately not "is
   this object close" — a stool by the lens hides nothing — but the share of the projected
   area of everything *wholly behind* an object that the object covers. The eye is still
   fixed; that was the mechanism, and it was the still being unreadable that was the
   complaint.

23. **The staff/customer classifier became something that can be applied, and the number
   that licensed it is per camera rather than the headline.** Done 2026-08-28, for the
   demo colouring the user asked for: staff blue, customers green.

   §7.15 measured 0.893 and stopped. `scripts/staff_probe.py` fit a probe, printed a
   figure and wrote `probe.json`; **nothing was persisted**, so step 9's deliverable —
   `staff` on `Track` — had nothing to load. `analytics/staff.py` is the missing half:
   the same arithmetic, plus the standardisation it was fitted with and a record of which
   camera its accuracy was held out on.

   **The headline is not the adoptable model, and `probe.json` says so rather than a
   preference.** The 0.893 arm is colour + the ImageNet embedding, and the probe's second
   loop keeps only a pooled `balanced_accuracy` — **no `per_camera` block at all**. So for
   any camera we might deploy on, that model's held-out accuracy cannot be looked up.
   Colour alone scores 0.880 and has all sixteen, and they are what the decision is made
   from:

   | | | | |
   |---|---|---|---|
   | Kaohsiung-cam04 **1.00** (n=15) | Taichung-cam01 **1.00** (n=18) | Taichung-cam11 **1.00** (n=15) | Kaohsiung-cam05 0.91 (n=11) |
   | Taichung-cam03 0.73 (n=15) | **Tao-Hsin-cam04 0.42** (n=12) | **Tao-Hsin-cam01 0.40** (n=15) | |

   **0.893 is an average over sixteen cameras and two of them are below a coin toss.**
   That is the same shape as every other entry in this section: the statistic was true and
   answered a different question than the one the deployment decision asks.

   **The gate has two conditions, because one would not have caught it.** A model must
   name the camera **and** clear a 0.90 floor on it. `model_Tao-Hsin-cam04.json` exists,
   names its own camera and scores 0.417; a gate that only matched the camera would have
   waved it through and coloured half that store's staff as shoppers, confidently, on
   every frame. The floor is derived, not felt: a three-minute clip carries 4–10 distinct
   people, so 0.90 keeps the expected miscoloured count below one.

   **What the render had to get right and would not have announced.** Features are taken
   from the source frame **before** either blur instrument: `_blur_region` covers the top
   45% of a box plus padding and the torso band is 0.18–0.55 of it, so the face blur lands
   exactly on the pixels the classifier reads. The 3D panel's palette key follows the
   verdict rather than `track_id % 8`, or two tracks with different verdicts share a key
   and the last write repaints one of them. A track under `MIN_OBSERVATIONS` is drawn grey,
   because otherwise every shopper arrives as a confident green for their first second.

   **Verification, not assertion.** The whole probe re-run after the refactor reproduces
   every §7.15 headline to three decimals (0.850 / 0.723 / 0.777 / 0.880 / 0.893) and all
   sixteen per-camera accuracies to four. Fitting on the training half only — the probe
   standardises transductively — was measured rather than argued, and changes none of the
   sixteen. The four refusals in `tests/test_staff_model.py` and the two tracker guards
   were each verified by deleting the defence and watching the test go red.

   **Which cameras this licenses.** Kaohsiung-cam04 and the two Taichung cameras.
   **Tao-Hsin gets a figure without staff colours**, and the reason is stated rather than
   averaged over: 0.42 on the only Tao-Hsin camera that reconstructs shelving at all.
   That store's uniform is the §7.15 failure mode already read off the frames, and what it
   needs is the three physical reference photographs on the waiting list, not a better fit.

   **Found while doing it, and fixed separately: `--no-blur` was writing its unblurred
   render into `assets/demo_<camera>.mp4`** — the one filename every command, the README
   and any person treats as this camera's render, from a flag whose own help says "never
   for anything shared". `tests/test_assets_allowlist.py` cannot reach it: that test
   governs what may be committed, and this is a file that gets copied by hand.

   **Was open here: `tools/commissioning/heads_video.py` did not blur at all**, and wrote
   `assets/heads_<camera>.mp4` under the same shared-name convention. **Closed 2026-08-29
   (`65a6b78`).** Both instruments now run **before any panel is drawn**, because three of
   its four panels are the source frame with something painted on them and blurring after
   would leave two of them showing what the third had covered. It takes a second forward
   pass rather than reusing one at the lower threshold: the pose rows are index-aligned
   with the detections, so widening the display set to reach the blur set would put
   skeletons on 0.07 boxes and change what the figure claims. `--no-blur` no longer claims
   the shared filename, and renders now go to `assets/dev/`, which is ignored wholesale.

24. **The face blur was missing people, and the instrument that found it had to be built
   wrong twice first.** Found 2026-08-28 while cutting the two store figures the user
   asked for. This is the privacy path, so it is written out in full.

   **The defect.** `demo_video` blurs at `BLUR_THR`, deliberately below the shipped
   detection threshold, on the argument that a detector missing a shopper costs a track
   while a blur missing one publishes a face. The value was 0.10, chosen as "below 0.35"
   and never measured against whether it was low enough. It was not. Over the busiest 120
   frames of Tao-Hsin-cam04, the detector re-run at 0.03 finds **954 person boxes, 132 of
   which have a head outside every blurred rectangle**, scoring around 0.08. One of them,
   cropped from the source frame and read by eye, is **two shoppers at a shelf with the
   man's profile plainly recognisable**. The static-plate instrument — the second one,
   which exists precisely because it cannot fail the way a detector fails — missed the
   same people.

   **The fix is nearly free, and that is measured rather than assumed.** Over the same
   windows: 0.10 → **0.07** takes the readable count to **0** on Tao-Hsin-cam04 while the
   blurred fraction of the frame moves 7.0% → 7.2%, and on Kaohsiung-cam04 it changes
   nothing at all (0 readable either way, 32.4% both). 0.03 costs no more again and is
   **deliberately not taken**: the audit runs at 0.03, and a render that blurs at exactly
   the threshold its auditor inspects at is checking itself with its own answer.

   **The instrument, and the two versions of it that were wrong.** The first ran the
   detector on the *rendered panel* and refused any head that still had fine texture. It
   is worth recording because it looked entirely reasonable: over 120 frames it returned
   921 boxes with head-gradient ratios p50 0.99 and p90 3.17, no separation in any score
   band, and **zero boxes above 0.35 on a window where the render itself had people in
   every frame**. The rendered panel is half resolution, carries drawn boxes and grey
   slabs and has been through h264, so the detector is not reading an image it was trained
   on — and gradient energy cannot tell a face from merchandise in any case. Both halves
   answered a different question. The second version ran on the source frames but
   reconstructed the blur set from the *current* `BLUR_THR` rather than the one the render
   used, which would have reported every old figure cleaner than it was; `demo_tracks.json`
   now records `blur_score_thr` and `blur_faces`, and the audit refuses a render that
   predates the field or was made with `--no-blur`.

   **What this says about the two-instrument design.** Both instruments missed the same
   people on the same frames. Contact sheets and a person's eyes missed them too — the
   sheets were rendered and read, and a face 47 px wide in a 480 px thumbnail does not
   announce itself. Only the audit found it. So the honest statement is not "two
   instruments make this safe" but "every mechanism here has now failed at least once, and
   the only one that has not is the one that checks the others".

   **Consequences.** `tests/test_figures_are_audited.py` requires every tracked
   `assets/demo_*.gif` to carry a tracked `.audit.json` written whether the audit passed or
   failed, with zero failing boxes, a non-zero checked count, and a blur threshold at least
   as strict as the current constant. That last clause fired immediately on
   **`assets/demo_Taichung-cam10.gif`, the README's front-page figure, which was cut at
   0.10** — its own audit found zero at the time and there is no reason to doubt that
   camera, but "zero at a threshold since found insufficient on another camera of the same
   fleet" is inherited rather than measured, so it was re-rendered.

   **Also fixed here, and it is the same shape one layer out**: `--no-blur` was copying its
   unblurred render into `assets/demo_<camera>.mp4`, the filename every command and person
   treats as this camera's render, from a flag whose own help says "never for anything
   shared". `tests/test_assets_allowlist.py` cannot reach that — it governs what may be
   committed, and this is a file that gets copied by hand.

25. **"Relative relationships must be correct" turned out to be a different requirement
   from "the numbers must be right", and it is the one the scene was failing.** Stated by
   the user on 2026-08-28 after reading the renders: *precision is negotiable; a cabinet is
   not at 45 degrees and a wall is not several disconnected panes.* Every defect below is
   invisible to a per-object check, which is why `PLAUSIBLE_M` passed all of them — a 7.9 m
   wall 15 cm thick is plausible in every dimension it has.

   **The 45 degrees was `main()`, not the geometry.** Two scene builders exist.
   `build_scene_regular` fits each fixture as a box in the store's own frame, so everything
   is parallel or perpendicular to everything else *by construction*; the older
   `build_scene` tiles `rect_decompose` rectangles aligned to the **world** axes, and a shop
   30 degrees off those comes out as staircases of small blocks. Every real consumer —
   `demo_video`, `heads_video`, `scene_overlay` — had moved to the regular path already.
   `main()` had not, and `main()` is what writes `assets/commission_mesh_*.png` and what the
   social preview card was cut from, so **the two most widely seen images in the project
   were the only ones still built the old way**. `--regular` is now `--ragged`.

   **The panes were not walls.** In the store frame, Taichung-cam11's five `wall`
   components measure 1.12x1.07, 1.44x0.85, 1.22x0.27, 1.23x0.86 and 1.11x0.63 m, and
   Taichung-cam04's eight are 0.6–1.5 m long. A shop wall is four to eight metres. **What
   the mask holds is not walls but the patches of white surface still visible between the
   fixtures standing in front of them**, and drawing each patch as its own 2.4 m slab is
   the row of floating panes. Merging the boxes afterwards was the obvious fix, was tried
   first, and barely moved anything: 5 → 5, 8 → 7, 6 → 6. **Boxes fitted to fragments are
   not collinear enough to merge**, and no amount of care in the merging reaches that.

   `wall_runs` does the step the scan-to-BIM sequence actually specifies, in its order: fit
   the wall **axes** to the whole point set, split each into runs at gaps wider than a door
   leaf, then intersect perpendicular runs at corners. Runs go to 2–3x their previous
   length. **Non-maximum suppression across the run is not optional** — one wall votes in
   several adjacent bins, and without it the fleet went *up* to 26 and 28 runs per camera,
   longer than before and more numerous, which is worse. Closing the runs into a room
   polygon — the last step of that sequence — is deliberately **not** done: a fixed camera
   sees part of one store, and closure would draw walls along the edge of the field of view.

   **And roughly half of what it fitted was still not a wall.** `floor_both_sides` is the
   relation, and no shape or size can substitute for it: a room boundary has floor on one
   side and the outside world on the other; a shopper can stand on both sides of a counter.
   Measured across four cameras, about half the runs fail it — **Tao-Hsin-cam04's two
   longest at 7.9 m (724 floor cells one side, 550 the other) and 7.2 m (626 / 430)**,
   Taichung-cam01's 4.7 m, Kaohsiung-cam04's 3.0 m. Those are the counter runs §7.21
   measured as classified `wall` by both teachers on a white-fixture store — and extracting
   them as long continuous runs had made them **more** convincing, not less. They are
   **dropped**, and the attempt to do better than that is worth recording because it broke
   the requirement this whole entry is about. Re-classifying them as merchandise -- the
   mask holds a real object, only its class is wrong -- put the fixture in the wrong
   **place**: the line is fitted to the `wall` point set, which on a white-fixture store is
   the counter *and the wall behind it*, so it sits at the wall, a metre behind the counter
   that failed the relation. Projected back through the camera, a box on that line hovers
   above the counter standing in front of it. Measured by rendering `scene_overlay` at
   `619975b` and at the change: **Tao-Hsin-cam03 went from four well-placed meshes to
   six**, the two extra ones floating over the counter runs, and Kaohsiung-cam04 from three
   to five with a sliver standing on open floor. Reverted in `f63c0d2`. **A fixture the
   mask cannot place is better absent than present and wrong** -- and an earlier revision
   of this entry claimed those five boxes as a gain.

   **Fit, regularise, mesh — in that order, which this file did not have.** Nothing ever
   compared one fitted component with another, so two boxes drawn through each other was a
   normal output; a reviewer calls that a jumble and no per-object check can see it.
   `resolve_overlaps` absorbs a box more than 60% inside another (the class mask and a
   re-classified run can both cover one counter — Tao-Hsin-cam03 goes 5 shelves → 2 on that
   alone) and otherwise shrinks the smaller along the axis it overlaps least, which is the
   direction its extent was least certain in. `snap_to_walls` moves a fixture within 20 cm
   of a wall flush to it, near face only, so it keeps its measured depth.

   **What an independent reader saw, and what it recommended.** A second model was given
   the four plate-versus-reconstruction pairs cold. Its verdict: nothing is skewed — the
   single-global-yaw design genuinely delivers that half — but **zero of four cameras have
   a room**, two of four have a fixture layout a customer would recognise, and the column is
   missing on every camera that has one. Its ranked recommendations, kept here because the
   reasoning transfers even where the answer does not: (1) commission the wall line by hand,
   one click-path per camera, since §7.21 proved by measurement that no automatic route
   supplies it — **declined by the user, who requires this stay automatic**; (2) the BEV
   plausibility pass, done here; (3) a vanishing-point room frame as a cross-check on
   `store_yaw`, which today votes with the least reliable objects in the scene; (4) an
   image-space sibling of `floor_both_sides` — a true wall has wall pixels *above* its top
   edge, a counter does not — which needs no depth and so is immune to the white-surface
   collapse; (5) extend the fixture catalogue (chair, round podium, door constrained to lie
   on a wall line). It named as **not worth doing**: floorplan vectorisation networks
   (Floor-SP / HEAT / RoomFormer — wrong input distribution for a single partial monocular
   view), CAD retrieval (Scan2CAD / ROCA — retail gondolas are not in their vocabularies),
   better monocular depth for wall heights, and replacing the 1.70 m person prior (a uniform
   scale error cannot break a relative relationship).

   **Open after this, in the order they are worth doing**: the room boundary with no human
   click — the honest automatic route is the occupancy-grid distinction between an
   *obstacle* boundary and a *frontier*, so that the field-of-view edge stays open rather
   than becoming a wall; the vertical wall test above; columns, which are lost on every
   camera; and the door, which is currently drawn standing in the middle of the floor.

26. **The checkout counter is not a trained class, and it is being classified as
   merchandise shelving.** Reported by the user 2026-08-29 from the renders, and it is a
   labelling gap rather than a geometry one, so none of §7.25's work touches it.

   The fixture vocabulary is `wall / column / display_table / display_shelf`. A till point
   is neither a table nor a shelf: it is a counter with a person permanently behind it,
   and every retail output that matters treats it differently from a merchandise run — a
   queue forms at it, dwell there is a transaction rather than interest, and
   `reach_to_shelf` firing at a till is an operator picking up a scanner, not a shopper
   handling stock. Classified as `display_shelf` it inherits the merchandise
   interpretation of all three.

   **Why this is likely the same root as §7.21 rather than a separate defect.** Both
   teachers read a white counter as a wall on a white-fixture store, which §7.21 measured
   and proved unfixable by clustering, prompt or paint order. A till point in these shops
   is a white counter, so the same surface that defeats the `wall`/`table` distinction
   defeats the `table`/`till` one. What separates a till from a merchandise counter is not
   its surface: it is **what is on it and who stands behind it** — a monitor, a card
   terminal, a printer, and a person on the staff side for most of the trading day.

   That last part is the useful observation, because this project already measures it.
   `staff/customer` is fitted per camera (§7.23) and `service_zones` already gives every
   fixture the floor beside it (§7.19's 71 zones). **A fixture whose adjacent floor is
   occupied by `staff` for most of the clip is a till; one whose adjacent floor is
   occupied by customers is a merchandise run.** That is a relation between two things the
   tree already computes, not a new model — the same shape as `floor_both_sides` in §7.25,
   and it needs no new labels at all.

   **Measured 2026-08-29, and the answer is better than the one proposed above.** Two
   findings, one negative and one decisive.

   *The prompt cannot do it, and now there is a number.* `prompt_group == "till"` covers
   **48 of the 71 service zones (68%)** across the eight cameras, because `retail counter`
   sits in that group and it matches every merchandise counter in these shops --
   9 of 12 zones on Taichung-cam04, 10 of 13 on Taichung-cam10. Only four cameras carry an
   instance whose prompt is literally `checkout counter`, at scores from 0.484 to 0.915.
   §7.19's caveat that "which fixture is the till is a store fact, not a 0.745 from a
   shape-shaped prompt" is confirmed rather than merely suspected.

   *The two sides of one instance are the discriminator.* `service_zones` already splits
   an instance into a zone per side -- same `instance_px`, `side` 0/1/2 -- and that is what
   makes the relation checkable without a new model. On Kaohsiung-cam04, over 3,304 track
   placements from a 900-frame clip, the counter SAM 3 calls `checkout counter` at 0.915
   splits perfectly:

   | zone | side | staff | customer |
   |---|---|---|---|
   | `fixture_02` | 0 | 0 | 724 |
   | `fixture_04` | 1 | 1,137 | 0 |

   One counter, one side entirely customers and the other entirely staff. So the rule is
   not "staff for most of the clip" as written above -- it is **the contrast between an
   instance's own sides**, which is a relation between two things and needs no threshold
   anybody has to defend.

   **And the absolute form would have misfired, which is why the contrast matters.** On
   Taichung-cam01 the staff share is 0.98-1.00 on *every* fixture zone in the store,
   because that clip's people are all staff -- the same clip §6 step 3 uses. An absolute
   threshold would have called all seven zones tills. That is `0298d7a` in a third place:
   **a population that is all one class licenses nothing**, and the till rule needs the
   same refusal `analytics.staff.require_camera` already carries for the classifier.

   Still to do: two cameras (Taichung-cam10, Tao-Hsin-cam04) have no staff verdicts in
   their track logs yet, so the fleet evidence is one decisive camera and one that proves
   the refusal is needed. Then the `till` kind has to reach `camera.json` -- today
   `prompt_group` is written to `runs/service_zones01/<camera>.service_zones.json` and
   **flattened to `kind: "display"`** on the way in, so even the weak signal is discarded
   at the boundary, which is the same shape as §7.27.

27. **`implausible()` names seven fixtures across the fleet and nothing reads it.**
   Measured 2026-08-29 over all eight commissioned cameras, on the build the renders
   actually use (`build_scene_regular`), because the previous entries in this section
   name the function repeatedly and none of them says how much it is finding.

   **85 fixtures built, 7 outside their class interval, on 4 of the 8 cameras**, and the
   identities matter more than the total because they are three different defects:

   | camera | fixture | what it is |
   |---|---|---|
   | Taichung-cam01 | `display_table` 2.60x2.45 m | the welding §7.21 measured -- two counters bridged |
   | Taichung-cam01 | `display_table` 2.05x2.00 m | the same |
   | Taichung-cam01 | `display_shelf` 0.55x0.20 m | a sliver: what a neighbour's `resolve_overlaps` left of it |
   | Taichung-cam04 | `display_table` 3.10x1.90 m | welding |
   | Tao-Hsin-cam03 | `display_table` 1.85x1.80 m | welding |
   | Tao-Hsin-cam03 | `door` 2.60 m span | shopfront glazing, not a door |
   | Tao-Hsin-cam04 | `door` 3.70 m span | the same, and named in §7.25 already |

   **Every one of them is drawn, and the sentence naming it goes to a terminal nobody is
   reading.** `implausible` is called in `tools/commissioning/scene_mesh.py`'s `main()`
   and nowhere else -- not by `demo_video`, `heads_video`, `scene_overlay` or
   `social_card`, which are what produce every published figure. So the four cameras
   above ship pictures with a fixture the code has already decided is not furniture, and
   the picture says nothing. That is the shape this repository keeps writing down: a
   check that talks and is not heard, the sibling of the green light wired to nothing.

   **What it must not become.** Deleting the offender is the obvious consumer and it is
   the one §7.25 measured and reverted: cutting a welded mask at the class ceiling took
   display tables 5 -> 1 on Taichung-cam01 and to zero on two cameras, because a welded
   component is a real fixture *plus* its neighbour, not a phantom. And "absent beats
   present-and-wrong" was written about a fixture the mask **cannot place**; a welded
   table is in the right place and the wrong size, which is a different trade.

   Open. The candidate consumers, in the order they are worth trying: the render says it
   in the picture rather than on stdout, so a reader of the figure sees what the code
   already knows; and the commissioning record carries it per camera, so the count is a
   number that can be watched rather than a line in a scrollback.

30. **The mesh figure can be driven by the pose head, and the thing that stops it is
   constraint design rather than data.** Opened 2026-09-02 on the question "can the 3D
   figure replicate what the person is doing". Four solvers, one subject: the single
   confident person on `Kaohsiung-cam04` at 10:58 local, standing at the counter typing,
   keypoint confidence min 0.69 / median 0.90.

   **The mesh side needs nothing built.** `meshes.human()` is not a rigged model, it is one
   `_tube(start, end, r0, r1)` per limb, and `_P` names exactly the joints COCO gives —
   shoulder, elbow, wrist, hip, knee, ankle. Twelve of the seventeen keypoints land on a
   tube endpoint. Driving it is replacing the standing constants with measured joints; no
   skinning, no SMPL, no new training.

   **What each solver fixed and what it broke**, on that one frame, with the absolute
   heights that decide it:

   | | bone ratio | shoulder→wrist | hip sep | head height | torso lean |
   |---|---|---|---|---|---|
   | target | 1.00 | 0.50–0.65 m | 0.204 m | ~1.8 m | ~15° |
   | A, fronto-parallel | 0.67–1.15 | **0.50 / 0.49** | 0.144 | **1.62** | **16°** |
   | B, per-bone depth solve | **1.00** | 0.23 / 0.19 | 0.100 | 1.73 | 23° |
   | A-seeded + constraints | 0.84–1.04 | 0.52 / 0.58 | 0.136 | **0.73** | **65°** |
   | same, 60-frame track | 0.51–1.13 | 0.33–0.64 | **0.182** | — | 53° |

   **A is the only one that puts the person in the right place.** It forces every joint
   onto one vertical plane at the feet's range, which is wrong about depth by construction
   — and that same construction is what preserves the vertical structure. Its costs are
   the bone lengths and a foot 0.26 m in the air, which is a forward foot with its depth
   turned into height.

   **B's failure is worth keeping.** Each bone was individually correct — ratio 1.00 across
   the board — and the arm was impossible: 0.19 m from shoulder to wrist on a person whose
   upper arm alone is 0.371 m, because a per-bone sign chosen with nothing holding the limb
   together folds the elbow back through the shoulder.

   **The reported win was an artefact of the metrics.** The A-seeded solve was written up
   as the best of the three on bone ratio, arm span and hip separation. All three are
   *relative*: a skeleton that sinks to the floor and folds at the hips satisfies every one
   of them. It put a 1.95 m person's head at 0.73 m. The check that catches it — absolute
   joint height against stature — was printed by the first version of route A and dropped
   from the comparison table when the solver changed.

   **The scale hypothesis is half right and is worth recording as a measurement.** Every
   solver wanted a bigger person than the geometry allowed, pinning against whatever clamp
   it was given. Sweeping the stature with everything else fixed puts the residual minimum
   at **1.70 m** — this fleet's own calibration prior (§7c.19), so `stature_m`'s 1.95 m
   reading for this subject is not independent evidence about him. At 1.70 m: residual
   297 → 137 mm single-frame and 371 → 233 mm over the track, hip separation 0.136 → 0.171
   against a 0.177 target, its across-frame spread 57 → 17 mm, and the fitted scale stops
   sitting on the clamp. **The picture does not improve**: the head goes 0.73 → 1.02 m
   against ~1.6. So 1.70 m fixes the skeleton's argument with itself and not the pose.

   **Where this leaves it.** Bone lengths do not encode "the torso is upright" or "the head
   is above the hips", so a constraint set made only of them can fold. A got the heights
   right and the depths wrong; the solve that followed freed both and lost the half that
   was already correct. The next version holds A's per-joint height and solves only depth
   against the bone lengths. That is a different constraint structure, not a tuning pass,
   and nothing here needs a model or a label that does not exist.

   Prototypes are scratch, not in the tree. The subject frame, the four joint sets and the
   renders are reproducible from `runs/hydranet_retail_person01/last.pt` plus
   `runs/commission01/Kaohsiung-cam04.camera.json`.

   **Addendum 2026-09-02: route A shipped behind a bone gate, and confidence cannot be
   the gate.** Route A drove the demo figures for one afternoon (the user reverted the
   look the same day; the capability stays behind `demo_video --posed-figures` and is
   heads_video's default). Shipping it surfaced the occlusion failure the single-subject
   comparison could not: a person whose lower body a counter hides lifts to metre-long
   tubes, roughly one placement in ten on Kaohsiung-cam04's gif window (20 of 321 past
   3 m, p99 90 m, max 7.6 km). Two measurements worth keeping:

   * **Keypoint confidence does not separate the exploded lifts — it inverts.** Their
     minimum limb confidence (p50 0.649) is HIGHER than the clean lifts' (0.481): the
     pose head is confidently wrong about joints it cannot see, so no confidence
     threshold exists that keeps the good and drops the bad.
   * **The bones separate perfectly.** Coherent lifts top out at a 1.46 m longest edge;
     exploded ones start past 3 m; the band between is empty on both README cameras.
     `lift_fronto_parallel` therefore refuses any skeleton edge over `MAX_BONE_M = 1.5`
     (`syncai_bev3d/figures.py`, numbers inline) and the figure stands instead. Same
     family as §7.19's lesson: the check that works is external to the model being
     checked.
