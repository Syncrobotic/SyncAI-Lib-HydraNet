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
| g | **known-false-positive polygons** — derived from the box population, not drawn (hanging accessory walls, printed people) | accept / reject | ✅ derived, reviewed and written for all 8 — 20 polygons covering 7–78% of each camera's hallucination population (`fp_polygons.py`; verdict 2026-08-25: every candidate was a poster or merchandise). **Nothing on the analytics or serving path reads them** (verified 2026-09-09): the only consumers of `CameraFile.false_positive_polygons_px` are `syncai_bev3d/figures.py`, which draws them, and the tool that writes them. A guard that half-covers is worse than an absent one because its presence is read as coverage — §8's structural finding, one artefact later. They would not have caught step 4's Tao-Hsin-cam04 night detection either: that camera's single polygon sits at x 80–144, y 432–496 and the box is elsewhere, because the polygons were derived from the **daytime** box population |
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

**Section 9 reads this table as a whole** — how far the statuses below, taken
together, leave the project from section 1's definition of success. It cites rather
than restates, so this table stays the thing a reader consults for status.

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
| 2 | **commission all 48 cameras** — build the 4-click tool and zone tool, run the pipeline | per-camera `camera.json` + a 1 m grid rendered on one real frame per camera | the grid looks right **to my own eyes on every camera**. **8 of 48 shipped** (`runs/commission01/REVIEW.md`), Taichung-cam05 withdrawn — two furniture checks agree its cells are over-scaled. **"Scale-measured" means the 1.70 m person-height prior, not a tape measure** (§7.19): every shipped `scale_source` reads `person_height_median_vs_1.7m_prior_nNN`, on 15–37 boxes. Masks, walkable outline, shelf ROIs and a commissioning 3D scene are written for all 8 (`masks_pass.py`, `scene3d.py`; tables 0.78–0.97 m fleet-wide); 5 mask sets clean, the Tao-Hsin pair partial and Kaohsiung-cam04 missing its pillar. **The Tao-Hsin caveat is measured (§7.21) and is the whole of what limits the 3D render's coverage** — the fixtures are proposed and then classified `wall`, and neither clustering nor paint order moves that camera's map by a pixel. **71 service zones across the 8, fully automatic** (§7.19): a zone is the floor *beside* a fixture, not its footprint, because a footprint is a region no shopper can occupy. Blocked on the physical world for the rest: 14 cameras need a per-store visual reference, 4 need mount-type triage, Taichung-cam05 needs its NVR stream setting; the 7 Kaohsiung person-score cameras are answered per camera in §7.12. Remaining tools: the 4-point ground-calibration click tool (§2.1b) and the tamper reference (§2.1h). **Inventoried 2026-09-03**: of 48 cameras, 23 are `selling_floor` and ALL 23 are onboarded (calib.json exists, `fleet_hardware_assumed` vfov on 22, tile-grid on Taichung-cam01); 8 of the 23 are commissioned, so the real stage-0 backlog is **15 selling-floor cameras**, each needing only the teacher passes plus the zones/FP human confirmation — no new calibration work. The other 25 are 17 back_of_house (out of product scope), 6 dead (luma 1.1), and 2 already counted. |
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

**The recall gain is real and the confirmation does not protect the event layer.** C drops 12–14% of boxes and produces *exactly* B's events, on the whole fleet and again with Kaohsiung-cam04 excluded (2 fall / 3 crouch both arms). The boxes it removes were never the ones firing events: the extra events come from real people detected at low confidence, whose keypoints are noisier, making more posture runs. So the next mechanism is not another box filter — **the event layer has to see the detection confidence a track was built from**; a track of 0.15 boxes cannot be trusted for posture the way a 0.6 track is. Kaohsiung-cam04 also explodes 3,353 → 13,595 detections at 0.15, which is its open person-score investigation surfacing, so the threshold is a **per-camera** decision. Re-scope before spending a retrain. Original scope kept below: temporal-consistency tiers from `instances_all_*.json`, FP polygons from the 37 hotspots; **the night pass, done 2026-08-26** (`scripts/night_ghosts.py`, `runs/night_ghosts01/`): the shipped detector produces **zero `person` boxes in 2,250 frames of an empty shop at 23:58 store-local, across 15 cameras**, at the shipped 0.35. The ghosts were always the *teachers*' -- SAM 3's `person` prompt returned 14 hanging accessory packets as people on Taichung-cam09, and Grounding DINO at 0.35 fired on an empty store on 13 of 42 cameras -- and the student did not inherit either. **Taichung-cam09 itself returns zero.** So the false-positive rate that `after_hours_person` had to be quoted against, and which did not exist, is now measured and is 0. Two caveats, because the sample is what it is: only 15 of 48 cameras have a 23:58 clip on this box, and only 2 of the 11 cameras where Grounding DINO fired at night are among them (both zero). **The first caveat came true on 2026-09-09** (step 6's fleet run, with the shipped person01 weights rather than the checkpoint measured above): **Tao-Hsin-cam04 returns a `person` box in 1,518 of 1,518 frames** of its 00:02 local clip, at score 0.47–0.57, and it raised the only `loitering` alert that survives a 240 s threshold. It is a static object — the box moves **0.14 px in x and 0.28 px in y over five minutes**, sub-pixel, which no person is, and the scene is otherwise motionless. That camera is not among the 15. So the night figure is 0 *on the cameras it was taken on*, and this is the population it did not see; `data/night_person.py`'s static-plate veto, recorded above as something "the serving path turns out not to need", is what would have caught it. `data/night_person.py`'s static-plate veto stays where it is -- it protects the *teacher's* output, and the serving path turns out not to need it. Superseded: measure whether FP polygons + temporal consistency remove the IR ghost persons | new Gold/Silver training set; night `person` precision figure | Gold precision ≥95%, Silver ≥85% on a 300-frame sample; `after_hours_person` stays on the VLM trigger list **only if** the night figure passes |
| 5 | **L1 validation** — tracking quality on in-domain clips, and homography accuracy against ground truth. **Geometry: done** (§7.13) — 7.3 cm median floor error over 19,824 WILDTRACK observations, yaw to <0.2°, and it did not need the images. **Tracking: has a number and a caveat** (§7c) — `runs/gt_cam01/idf1.json` on 900 frames of Taichung-cam01, single-stage IDF1 0.739 / 6 switches, two-stage 0.742 / 3; read `provenance.json` before quoting either, because a model labelled those identities by eye. **What the step exists to fix is fragmentation**: 202 tracks over 24 minutes, 43% ever enter a zone, median visit 3.2 s, and 58% of endings are mid-view deaths concentrated on two cameras (§7.11 — 86% still have a box on the person at a median 0.338 against a 0.35 threshold, so it is a threshold problem before it is a tracker one). The tracker-lost/detector-gone split is **not** established and nothing should be concluded from it until an appearance model can tell two shoppers apart (§7.11) | position error in metres; ID-switch count per watched 10-minute clip | position error small enough that zone events land in the right zone; dwell/loiter durations survive — an ID switch mid-loiter resets the clock, so switches on the watched clips must be rare enough not to |
| 6 | **L3 end to end** — **RUN 2026-09-09, and it produced the number section 1 defines success as.** `scripts/step6_events.py`: a commissioned `camera.json` (not a hand-fitted pose), its service zones, the event layer, and every event filed through `dispositions.record_alert` — the store section 9.2 found empty now holds **263 rows**. 8 cameras × 4 clips × 5 min = **162 minutes**, 47 min GPU, one code version (the `src/` content hash is identical before and after, once a file another session added mid-run is excluded). **Open hours: 129.6 alerts per camera-hour** = 1,478 per camera-day over 12 trading hours, at this run's demonstration `loiter_seconds=8`. **The threshold is the lever, not the geometry**: the same footage re-read at 30 s gives 427 per camera-day, at 120 s gives 30, and **at the 240 s section 5 itself uses as the archetype it gives 6 — inside step 7's single-digit gate**. Closed hours: 1 event in 40.5 camera-minutes. **Two alerts read against their video by eye** on the first single-clip run: one correct-but-useless (a member of staff at their own workstation, which is step 9's measured problem) and one whose foot point the frame decided rather than the floor (now refused by `geometry.FrameBounds`). **What is NOT done, and it is the gate**: none of the 263 has been graded. 15 survive the 240 s threshold and 14 of those are `occupancy_exceeded`, which its own docstring says counts *tracks* and so over-counts by exactly the fragmentation rate | an event log readable against its video | events match what the clip shows |
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

29. ~~The serving target is bound by PCIe~~ — **cleared 2026-09-01.** The delivery
   target is §7.4's **96 x 5 fps = 480 f/s**; `bench_trt.py` had held `TARGET_FPS = 1440`
   for six days after that revision, which is what a stale constant costs a conclusion.

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

   **Measured on an idle card, 2026-09-02, three runs: contamination was worth 56% of
   the end-to-end figure.** The only
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

32. **Two any-view geometry models are wired up as commissioning witnesses, and neither
   has run.** Opened 2026-09-06. Depth Anything 3 (`DA3NESTED-GIANT-LARGE`, 1.15B, metres
   as it claims them) and VGGT-1B (relative depth, intrinsics) are asked the two
   questions the fleet cannot answer from inside: **what is the lens** and **how tall is
   the furniture**. `syncai_bev3d/geometry_teachers.py` runs either on one undistorted
   plate; `tools/commissioning/map_anything_eval.py intrinsics --backend da3|vggt` puts
   the vfov beside the anchor the way MapAnything's was, and
   `tools/commissioning/geometry_bench.py --source da3|vggt-floorfit` scores the depth
   against the floor with `flat(ctl)` beside it. **The commit ids are arguments, not
   constants**: the machine that built the instruments could not reach the Hub, a pin
   copied from memory pins nothing, and `geometry_teachers.pinned` refuses anything but a
   full 40-character id -- the first run records the one it used, and promotion to a
   constant takes it from that run.

   **The predictions, written before the runs so the runs can be wrong about them:**

   * *vfov on Taichung-cam01 (tile grid 70.4°).* Both disagree the way MapAnything did
     (38.26°, a 2.03x focal ratio). Single-image focal estimation resolves the focal/scale
     ambiguity from learned object-size priors, and a phone shop seen from a ceiling
     corner is not what those were fitted on. A backend inside 5° of 70.4 would be the
     first independent confirmation of the fleet's assumed vfov on 21 cameras; anything
     else is a third estimate beside two others and changes nothing.
   * *Depth, eight commissioned cameras.* DA3 holds the floor within DA-V2's 0.06 m and
     improves the relief half on the two white-fixture cameras (Tao-Hsin-cam03/-cam04),
     because the collapse on textureless surfaces is a prior-strength failure and the
     model is three times the size. VGGT's relief is not better than DA-V2's: single view
     is the case its README says it was never trained for, and its floor columns are the
     fit rather than a measurement, which the `floorfit` in its label says.
   * *What neither moves.* The far edge of a fixture the camera cannot see, and nothing
     that gates stage 2 -- foot-point occlusion at the counter and the detector's recall
     are §6 step 4's, and a depth teacher does not touch them.

   **What each outcome costs.** DA3 winning both halves means replacing the depth teacher
   in `plate_calibration.run_depth` and re-cutting every `camera.json`, mask completion
   and published figure; the 0.847 NYU scale constant in `data/nyu_depth.py` goes with V2,
   since DA3's metres are tied to the vfov explicitly rather than to NYU's cameras. VGGT
   winning the lens question means running it on the other 21 selling-floor cameras and
   retiring `fleet_hardware_assumed`. Neither winning means closing this item with the
   numbers and leaving the 4-point ground-calibration tool (§2.1b) as the only route to a
   measured vfov -- which is where the risk sits regardless of how this comes out.

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

### 7c. Investigations — finished: the claim each one settled

**Collapsed 2026-09-10, 2011 lines to 171.** Each entry keeps its number and its opening
claim; the evidence, tables and reasoning that produced it are in git, not here:

    git show 87d5cdc:docs/PLAN.md

The numbers are addresses — `§7.15` and `PLAN 7c.30` are dialled from nine files outside
`docs/`, and `tests/test_plan_citations_resolve.py` holds them — so no entry is renumbered
or removed, only shortened. What was lost is the argument; what is kept is what it
concluded, which is what a plan needs from a finished investigation.

**Why it was cut.** This section had grown to 61% of the document, and a plan that is
mostly a lab notebook stops being read as a plan. §1 states the goal — one camera, one
model, two readings, security and retail — and it was being reached through two thousand
lines of settled arguments.


Answered work, not open questions. Each entry is kept whole because its number is a
measurement and its prose is what that measurement cannot show, and the pair is what a
decision rests on -- reading either half alone is how a figure gets quoted for
something it does not support.

11. **The Kaohsiung person-score investigation is two investigations, and the fix is not
   on the inference side.** Opened 2026-08-27 from step 2's blocker list and step 5's
   "two cameras carry the fragmentation". The name was wrong on both counts: it is not
   one fault, and on the camera it is named after the scores are fine.

12. **The seven blocked Kaohsiung cameras split, and the block was never a group
   property.** §7.11 answered what the person-score investigation was; this answers what
   it *blocks*, which is `runs/commission01/REVIEW.md`'s "the Kaohsiung person-score
   investigation before its 7 cameras can use the person-height path". Measured
   2026-08-27 with `scripts/person_score_probe.py solo` over all four clips of each,
   against Kaohsiung-cam04's 0.50 / 99%:

13. **Step 5's geometry gate: 7.3 cm, and what it is agreement with.** Run 2026-08-27 with
   `scripts/wildtrack_ground_eval.py`, and run **before the archive finished downloading**:
   a zip stores every file behind its own local header, so all 7 calibration XMLs and all
   400 annotation JSONs were already in the bytes on disk and could be inflated out of the
   partial file. The images are not needed — the boxes are in the JSON.

14. **The `fall` height threshold is measured now, and on its own it does not work.**
   `analytics/events/pose.py` says of its own defaults that "none is measured", and
   `fall_head_height_m = 0.80` was added 2026-08-26 on the evidence of one 24-minute clip
   where it took the fleet from 3 false falls to 0. Measured against NTU RGB+D's official
   3D skeletons — real camera-frame metres, read by `data/ntu_skeletons.py`, 120 clips per
   class, `tools/temporal/ntu_fall_discriminator.py`, `runs/ntu_fall01/`:

15. **`staff/customer` has its first measurement, and nine colour numbers beat every
   embedding in the tree.** 421 crops from 142 people were extracted by
   `scripts/staff_crops.py` and sorted by hand into 154 staff / 223 customer / 44
   unclear — 54, 75 and 16 people. Probed by `scripts/staff_probe.py`,
   `runs/staff_probe01/`, **leave one camera out** over the 16 cameras carrying both
   classes, 230 held-out crops:

16. **Step 8's precondition holds: the features separate the classes, and they beat the
   geometric rule they would replace.** Asked before a trainer was written, because
   `analytics/pose_sequence.py` has been in the tree tested and with **zero consumers**,
   so nothing had ever measured what its features can tell apart, and building a pipeline
   first and discovering the features were the problem is the expensive order.

17. **Step 8 has a model, and it passes — after failing first, which is the useful half.**
   `tools/temporal/train_posture.py`, two 1-D convolutions over time with masked mean and
   max pooling, **32,933 parameters** against §2.3's 100K budget, trained on NTU projected
   through our camera pose, **held out by performer**, no early stopping on the held-out
   fold.

18. **Two-stage tracking's identity was measured, the instrument was wrong first, and
   fixing it is a component step 9 also needs.** Opened 2026-08-27 against decision 1:
   `runs/endings05/` cuts mid-view track deaths 111 → 38 and doubles track length, and the
   objection to adopting it was that the Kalman coasts onto neighbours and inflates dwell.
   **Length cannot tell those apart — recovering a chopped-up shopper and merging two
   shoppers produce the same number.** Only identity can, which is what §6 step 5 already
   says: "distinguishing 'the same person, a metre on' from 'a different person, a metre
   away' is what an appearance model is for."

19. **"Scale-measured" meant the person-height prior all along, and a second fleet arrived
   that has more of it than the first.** Opened 2026-08-27 when a new corpus —
   `gs://syncai-rtsp-recordings`, ten RTSP channels of a second store, wood floor, not
   STUDIO A — was asked for 3D scenes and the first answer given was "these need a
   physical reference first". **That answer was wrong, and checking it corrected this
   plan rather than the new cameras.**

20. **The `person` detector has never seen a labelled shopper from these stores, and half
   the site training images teach it that shoppers are background.** Opened 2026-08-28 to
   confirm §7.11's premises before spending its retrain, and both of them moved.

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

22. **A merchandise wall was being drawn as a small cabinet on every camera turned the
   other way, and the tripwire could not see it.** Found 2026-08-28 by a reviewer asking
   why the README figure's fixtures were placed as they were — the render, not a number,
   which is the third time today that a defect surfaced only when somebody opened the
   picture (§7.21, and the p85 wall heights before it).

23. **The staff/customer classifier became something that can be applied, and the number
   that licensed it is per camera rather than the headline.** Done 2026-08-28, for the
   demo colouring the user asked for: staff blue, customers green.

24. **The face blur was missing people, and the instrument that found it had to be built
   wrong twice first.** Found 2026-08-28 while cutting the two store figures the user
   asked for. This is the privacy path, so it is written out in full.

25. **"Relative relationships must be correct" turned out to be a different requirement
   from "the numbers must be right", and it is the one the scene was failing.** Stated by
   the user on 2026-08-28 after reading the renders: *precision is negotiable; a cabinet is
   not at 45 degrees and a wall is not several disconnected panes.* Every defect below is
   invisible to a per-object check, which is why `PLAUSIBLE_M` passed all of them — a 7.9 m
   wall 15 cm thick is plausible in every dimension it has.

26. **The checkout counter is not a trained class, and it is being classified as
   merchandise shelving.** Reported by the user 2026-08-29 from the renders, and it is a
   labelling gap rather than a geometry one, so none of §7.25's work touches it.

27. **`implausible()` names seven fixtures across the fleet and nothing reads it.**
   Measured 2026-08-29 over all eight commissioned cameras, on the build the renders
   actually use (`build_scene_regular`), because the previous entries in this section
   name the function repeatedly and none of them says how much it is finding.

30. **The mesh figure can be driven by the pose head, and the thing that stops it is
   constraint design rather than data.** Opened 2026-09-02 on the question "can the 3D
   figure replicate what the person is doing". Four solvers, one subject: the single
   confident person on `Kaohsiung-cam04` at 10:58 local, standing at the counter typing,
   keypoint confidence min 0.69 / median 0.90.

31. **The shipped model was walked into three foreign venues — metro, mall, airport —
   and the failure is a confidence slide, not a collapse.** 2026-09-03, seven public
   fixed-camera clips (Taipei MRT platform day x2 + elevated night, UK shopping-centre
   entrance + retail store, Hanoi terminal gate hall timelapse, LaGuardia apron) beside
   a Taichung-cam01 in-domain control; person01 `last.pt` EMA at 1 fps sampling.
   Apparatus and per-frame JSONs: `runs/domain_probe_20260903/` (probe + stats scripts
   copied in), sources and caveats in `datasets/domain_probe_20260903/SOURCES.md`.

33. **The text-embedding head, tested with a vocabulary it was built for, trains to
   parity and does not win. 2026-09-08.** `heads/text_classifier.py` was built 2026-08-17
   against a measured failure -- the detection head over Kaohsiung-cam08 returns 1,683
   `book` at score 0.05 and no `laptop` at any threshold -- and then never revisited, with
   no decision on record. The reason it stalled is now measured: **the only matrix ever
   built for it embedded the two internal alarm names** in generic templates ("a photo of
   boxed stock"). Those are not English phrases, and CLIP reads `device` loosely enough to
   collide with `person` at excess **0.79**, above `make_text_embeddings.EXCESS_SIMILARITY`
   -- which is why `runs/hydranet_retail_openvocab` could only carry two classes.

34. **person01's throughput on the pro6000, measured on an idle card 2026-09-08**
   (`runs/bench_person01/`, `runs/bench_person01_argmax/`; `scripts/bench_trt.py`, 15 s per
   engine, H2D **and** D2H in the end-to-end figure). The target is 96 streams x 5 fps =
   480 f/s.

35. **70.4 is not the wrong number, and this is the first independent check of it.
   2026-09-08.** Three any-view models now answer the lens with one voice -- MapAnything
   38.27, DA3 Metric-Large 39.8-42.0, DA3 nested Giant-Large 37.7-40.6 -- against a tile
   grid measured on Taichung-cam01 at 70.4. Two witnesses agreeing is 7.19's own warning
   (GeoCalib and HumanFoV matched to 0.16 deg and were both wrong), and 7c.32 showed DA3's
   number does not move for a zero baseline, so it is a resolution prior rather than a
   reading. But "the models are not measuring" does not establish that the tile grid is.

36. **The counting-line escape from fragmentation was tried on Kaohsiung-cam04 and the
   camera cannot supply it, 2026-09-07.** `line_events` needs identity across ONE frame
   step -- 0.2 s at 5 fps -- not across a visit, so a crossing count should survive
   fragmentation that halves the track count. That reasoning is sound and this camera is
   the wrong subject for it: re-derived from 900 frames (87 tracks, against the
   `runs/zones01` proposal's 5 events from 300 frames, which says of itself "weak
   evidence, confirm against the plate"), **all 87 births and deaths cluster at the right
   frame edge and at the counter's near end. There is no door in this view.** A line at
   the frame edge counts "entered the field of view", and a track born inside it never
   crosses it. The birth/death map is a cheap per-camera test of "does this camera see an
   entrance", and it is worth running before any footfall line is drawn.

37. **The backbone carries no novelty signal, and the control is what says so.
   2026-09-08.** The tree has no anomaly capability -- "anomaly" appears nowhere in
   `src/` -- so every intrusion alarm raised on the UCF-Crime probe was a rule written
   outside the model: person, after hours, survived `night_person`'s static veto. The
   model detected the burglar because a burglar is a person; it did not detect the
   burglary, and would have raised the same alarm for a cleaner. Option A of the
   redesign was the cheap one: expose the neck feature, fit a per-camera "normal" at
   commissioning, score the distance -- Avigilon UMD's shape, which learns a scene for
   two weeks and flags what departs from it.

## 8. What the health audit changed, and what it taught

A best-practice audit ran on 2026-09-04 over the whole tree (8 sweeps: packaging,
config, duplication, comment truth, tests, ML engineering, repo hygiene, privacy),
raising 34 items. All 34 are closed. The working list they were tracked on is gone,
as it said it would be: the fixes are in the code, the arguments are in the commits
that made them, and what generalises is here.

**The structural finding, which is why the P0 list looked the way it did: this
project's guards were trusted more widely than they reached.** `split_leaks` covered
one dataset type of five; `deterministic` warned instead of enforcing; `check_parity`
gated the ONNX and not the fp16 engine that ships; the face-blur tests could not see a
blur -- their fixture was `np.zeros`, and a Gaussian blur of a uniform image is the
identity, so all five passed against a no-op `blur_region`. A guard that half-covers is
worse than an absent one, because its presence is read as coverage. Every fix in this
pass was verified by reverting it and watching the test go red.

**Four of the 34 findings were wrong, and the pattern in how is the useful part.** Each
was wrong because it was read off a name or a grep rather than measured:

* *"`place_boxes` and `track_ground_path` have drifted apart."* They had not.
  `clip_tracks.tracks_for_clip` undistorts upstream and passes `k1` as a keyword-only
  argument with no default, so the second call site cannot silently skip it.
* *"`select_weights`'s 'every caller goes through this' is bypassed by three scripts."*
  All three go through it. But it pointed at a real defect one level down: the
  *fallback* was silent, and `cli/export_onnx.py` had already worked that out and
  recovered the answer with an identity test on the returned dict.
* *"Configs for the deleted quadruped line survive as fixtures."* The dangling dataset
  path was inside a comment, and the config that looked orphaned was the only test of a
  taxonomy `src/` still shipped. Retiring the taxonomy was the real question, and it was
  taken deliberately rather than as cleanup.
* *"The 26 absolute `ROOT` constants encode a real assumption -- leave them."* 26 of the
  27 named the repo root itself, not `runs/` or `datasets/`, and each sat exactly two
  levels below it, so `parents[2]` was the same directory computed. The failure they
  caused was also worse than the one they were weighed against: with two checkouts on
  one box, a tool run from the second reads the first's `runs/` and answers about the
  wrong tree, with no error at all.

**Three decisions, so they are not rediscovered as findings:**

* **No LFS.** The 113 MB was 88 MB of loose objects, not history; one `git gc` took
  `.git` to 78 MB. LFS would have rewritten history on a repo three branches and several
  sessions share, to reclaim space that was never in it. Re-run `git gc` if it grows.
* **`dev` is meant to lag `main`.** `dev` sits at `0.1.0` with no CHANGELOG while `v0.4.0`
  is released, because release-please writes to `main` and nothing flows back. That is the
  design. A version bump is metadata *about a release* and putting it on the unreleased
  branch would make `dev` claim something untrue. Do not back-merge to make them agree.
* **Privacy is a publication-time control here, not a runtime one.** `face_blur` is used
  by the figure tools and one CLI, and by nothing on the serving or analytics path;
  retention is `scripts/retention_sweep.py` rather than policy in code. Coherent as
  designed -- the analytics path keeps positions and dwell times, not faces, and the faces
  that leave this repository leave through a figure. Do not read the blur as a runtime
  guarantee it does not make.

**Two failures worth not repeating.** A shell glob expanded `[[...]]` while editing this
document and replaced per character; the same class of accident had earlier eaten three
backtick references out of a docstring in `tools/commissioning/service_zones.py`, where it
sat unnoticed through two commits. Edit prose through a file-based script, not a shell
one-liner. And a figure's audit must be re-cut *after* the code commit lands, not before,
or it records a version the tree has already left -- which put a red `dev` on the board
once in this pass.


## 9. The distance to the product, read across the steps

Written 2026-09-09. **This section measures nothing new.** Every figure in it is cited
from the section that took it, and where a status here and a row in section 6 disagree,
section 6 is right and this is stale -- it is a reading of the table, not a second copy of
it. What is mine and not the evidence's is the **ranking**; the ordering below is a
planner's judgement about which gap makes the others unreadable, and it is arguable.

It exists because sections 6 and 7 answer "what is the state of each piece" and nothing
answered "how far is this from the thing section 1 says success is" -- which is
*actionable alerts per camera per day, and the incidents missed*.

### 9.1 What is solid

Named first because the list below is long and it would otherwise read as a verdict on
the whole system, which it is not.

* **The geometry is validated independently of this fleet.** 7.3 cm median floor error
  over 19,824 WILDTRACK observations, yaw under 0.2 deg, and it did not need the images
  (section 7.13).
* **The two-head architecture's bet paid.** Distilled pose reads PCK@0.2h 0.915 /
  L2 p50 7.7 px with a flat end-of-run curve (step 3).
* **Throughput is not a constraint.** 1,494 f/s against the 480 f/s requirement, 3.1x
  (step 3, section 7.4).
* **The night false-positive rate is measured and is zero.** 2,250 frames of an empty shop
  at 23:58 across 15 cameras, zero `person` boxes at the shipped 0.35 (step 4). The ghosts
  were the teachers' and the student did not inherit them.

### 9.2 Nothing has been graded, so no number here is an accuracy

Section 1 states the rule: *no site figure is an accuracy until a human has graded it --
until then it is an agreement with the teacher models.* Section 4.5 names the instrument
that would fix it (shadow grading) and calls it the human test set, free and accumulating.

**Verified 2026-09-09: that instrument had never been switched on.**
`serving/dispositions.py` is a complete schema -- append-only JSONL by UTC day, alert row
plus operator verdict joined by `alert_id`, calibration hash, checkpoint and commit -- and
there was **no store on disk**. **Amended the same day**: step 6's first fleet run wrote
**263 alert rows**. What that changes is that the instrument now has input; it does not
change this entry, because **not one of the 263 has been graded by a person**, and an
ungraded alert row is not a label. And the one ground-truth artefact in the tree,
`runs/gt_cam01/provenance.json`, declares in its own text that it was
`"labelled_by": "Claude (Opus 5) ... by eye from the clip"` and is
`"not_a_human_label_set"`; the tracking IDF1 0.739 rests on it (step 5).

So `detection_mAP/site_person` 0.7387 means *agrees closely with Grounding DINO*. The
same teacher is measured returning people on an empty store on 13 of 42 cameras, and
SAM 3's `person` prompt returned 14 hanging accessory packets as people (step 4).

**The cost is not that the figures are uncertain. It is that improvement has no
verifiable direction.** Raising site mAP and raising agreement-with-the-teacher are the
same movement under every instrument this project currently owns, and they are not the
same thing. Every gain booked since the teachers became the label source carries that
ambiguity, and no amount of further training resolves it.

### 9.3 The known recall failure sits exactly where the product sells

Measured 2026-08-26 over four commissioned cameras, 40 frames each, and recorded in
`serving/decode.py`: at the shipped 0.35 threshold the dense head marks **73 person
regions with no box at all -- 20% more people than the box head returned** -- and every
one, cropped and looked at, is a shopper whose lower body is behind a counter or a display
table. Dense `person` IoU is 0.885 against detection mAP@50 0.302.

Table-edge dwell, shelf reach and queue position are the readings section 1 sells, and
they are the same geometry. `confirm_with_dense` recovers part of it, and the same sweep
found the boxes it admits **were never the ones firing events** (step 4) -- so the
recovery is real for positions and has not been shown to reach the event layer.

### 9.4 Tracks do not survive long enough to measure a duration

Section 7.11: 202 tracks over 24 minutes, 43% ever enter a zone, **median visit 3.2 s**,
and 58% of endings are mid-view deaths -- of which **86% still have a box on the person**,
at a median score 0.338 against a 0.35 threshold.

Dwell, loiter, queue and path are durations. An instrument whose median observation is
3.2 s cannot measure a 4-minute loiter, and section 7.11 also records that the
tracker-lost / detector-gone split is not established and will not be until an appearance
model can tell two shoppers apart. This is a threshold problem before it is a tracker one,
which is the cheap half and is not done.

### 9.5 The metres are a population prior, fleet-wide

Every commissioned camera's `scale_source` reads
`person_height_median_vs_1.7m_prior_nNN`, **on 15-37 boxes** (step 2, section 7.19), and
vfov is `fleet_hardware_assumed` on **22 of the 23** onboarded cameras -- Taichung-cam01's
tile-grid pin is the exception, and §7c.31 refused the same pin at two other venues
on a flat k1 sweep. Taichung-cam05 was withdrawn when two furniture checks disagreed.

Section 9.1's 7.3 cm validates the **arithmetic**. It says nothing about this fleet's
**parameters**, and every speed threshold, zone verdict and dwell figure is denominated in
them. Coverage compounds it: 8 of 48 cameras commissioned, 15 selling-floor cameras still
in the stage-0 backlog.

### 9.6 The chain has never run end to end

Verified 2026-09-09, before the commits below: `world_frame`, `pixel_to_ground`,
`SecurityEvent` and `dispositions` appeared **zero times** in `serving/` and in
`scripts/serve_pilot.py`. The serving path stopped at L0 plus a tracker, with boxes still
on the letterboxed network canvas -- the pixel frame `analytics/world.py` gained
`canvas_region` for the same day, and whose two wrong readings cost 2.4-3.4 m in metres
that carry no NaN and raise nothing.

**Amended the same day, and the amendment is the smaller half.** `CameraState` now holds
the commissioned `camera.json` and the canvas region and produces a `WorldFrame`
(`serving/camera.py`), and the tracker it runs records the score each track was built
from and converts to the producer's type (`bytetrack.Fragment.scores`, `as_track`) --
that second gap is the one docs/PLAN.md step 4 named as its next mechanism and had no
owner. `tests/test_serving_world.py` runs the chain in metres.

**Amended again the same day: step 6 has now run** (`scripts/step6_events.py`, section 6),
over 8 commissioned cameras and 162 minutes, filing 263 alert rows. Two things that
follow, and the second is the one that matters:

* It runs the **offline** chain -- `clip_tracks.track_clip` over stored clips -- not the
  serving path. `scripts/serve_pilot.py` still emits no positions, no events and no alert
  rows, so the live half of this entry stands unchanged.
* **Step 7 (shadow mode) is what remains, and nothing about it is a code problem.** The
  chain produces gradeable rows; what it has never had is a person grading them.

### 9.7 Two components have no consumer on the serving path

* **`staff/customer`** (step 9) is half done: 0.893 balanced accuracy held out by camera,
  and **refused on Tao-Hsin-cam04 at 0.417** against a 0.90 floor. Its own gate is stated
  by its consumer, and the consumer's number is that `reach_to_shelf` fires **11.7 alerts
  a minute** on a clip where every person is staff.
* **The behaviour head** (step 8) has passed its numeric half -- 32,933 parameters,
  fall vs pick_up 0.963 against a 0.944 linear floor -- and has **no consumer on the
  serving path**, so `fall`, `crouch` and `sit` are still the geometric rules section 7.14
  measured as marginal. The crowd failure is grouping, not classification: three people
  found in a frame holding about ten, and **one false alarm per eight minutes on a safety
  alert does not ship** (step 3).

Both are prerequisites for steps 6 and 7, as section 6 already states.

### 9.8 Out of domain it is measured to fail, and that is a scope statement

§7c.31: out of domain the person-score distribution slides into the threshold band
(a 0.15 to 0.30 cut costs 36-67% against 11% at home), ad posters detect as people every
frame, night OSD text mints phantom devices, and `fixture` is approximately zero outside
retail so zones cannot be drawn at all.

This is consistent with section 1 -- the first vertical is an Apple-reseller chain -- and
is recorded here as the boundary rather than as a defect. What follows from it is that
**self-calibration is the product** (ruled 2026-09-03), and that ruling is not yet built.

### 9.9 The read

**The gap is instruments, not model capability.** The model may already be good enough for
step 6; nothing in this tree can currently answer that, because every site number is
agreement with a teacher or with a model-labelled set, and the one instrument that would
break the circle is a schema with no rows in it.

The counterweight, stated because it is unusual and is this repository's strongest asset:
**every failure above is one this project recorded about itself, with a number.**
`provenance.json` volunteers that its labeller was a model; `dispositions.py` volunteers
that it cannot measure recall; section 2 volunteers that one leg of its own boundary
argument no longer carries weight. The problem is not blindness. It is that steps 1-5
build instruments and steps 6-7 use them, and 6 has not started.

### 9.10 What this ordering implies for the next steps

Not a build order -- section 6 is the build order -- but the sequence this reading argues
for inside it:

1. ~~Step 6~~ **done 2026-09-09** (section 6): 263 alert rows over 8 cameras, and the
   measured finding is that **the loiter threshold, not the geometry, is what stands
   between 1,478 alerts per camera-day and 6**. What follows is not another gate:
   it is a person grading a day of them, which is step 7.
2. **Then shadow mode, even at one camera for one week.** The first operator verdicts are
   worth more than any retrain, because they are the first signal that is not the
   teachers' opinion.
3. **Fragmentation (the 0.338-against-0.35 half) and the per-store uniform reference**
   are what make 1 and 2 readable rather than noisy.
