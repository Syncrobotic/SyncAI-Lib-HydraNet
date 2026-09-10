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


### 1.1 One model, two readings — what they share, where they part

The reason this is one system and not two products, stated as a design decision rather
than left implicit in the section that follows.

**Shared, all of it, down to L3.** One network on one frame (§2.2), one `camera.json` per
camera (§2.1), one set of tracks in metres (§2.3 L1), one event layer in metres and
seconds (§2.3 L3). Security does not get a detector and retail another. A shopper standing
at a fixture and an intruder standing in a shut shop are **the same measurement** — a
person, a floor position, a duration — and nothing above L3 is asked to tell them apart.

**They part at the threshold, and that is measured, not asserted.** The same 162 minutes of
fleet footage, re-read at four loiter thresholds, gives **1,478 / 427 / 30 / 6 alerts per
camera-day** at 8 / 30 / 120 / 240 seconds (§6 step 6). Security wants the 6 and will pay
recall for it, because every alert costs a person's attention. Retail wants the 1,478,
because it aggregates and a missed dwell is a rounding error. **One model, one export, two
thresholds** — the divergence is a config value, which is §3's placement rule holding.

**What each is blocked on, and neither is a model problem.**

* **Security** — step 7: a person grading a day of alerts. The chain produces gradeable
  rows (§9.6); none has been graded, so no number here is an accuracy (§9.2).
* **Both** — `staff/customer` (§6 step 9, blocking two steps). Without it an L3 log is
  **11.7 alerts a minute of staff working**, and retail counts a member of staff at their
  own workstation as customer dwell. It is a uniform, not an identity, which is what §5 rule 5
  permits.

**The two additions under discussion, and where each would land.**

* **A VLM / visual judgement head.** It already has a place: **L4, on trigger** (§2.3),
  and the card is already budgeted for it — analytics needs 32% of the engine and the
  remainder is held for exactly this (§7.4). It is an escalation layer, not a fifth head:
  it reads the frames an event already selected. That keeps §3's rule — walk down from a
  config value and stop at the first rung that answers the question.
* **Memory.** Absent, and the only one of the two that changes the shape of the system.
  What it must NOT be is re-identification across visits or stores — §5 rule 5 forbids it. What
  it can be is a **per-camera, per-time-of-day normal**: what this floor usually looks like
  at this hour, so an event is scored against its own camera's history rather than a fleet
  constant. §7.37 refuted one route to it (a distance in the backbone's feature space,
  killed by its own control) and left the other untouched: a model of *normal* learned from
  ordinary footage, of which this fleet has 48 cameras' worth and almost no anomalies.
  Before it is built it needs the thing step 7 produces — graded alerts — or it will be
  learning the teachers' opinion again.

  **What the market's "self-learning" is, read from the patents rather than the
  datasheets (surveyed 2026-09-10).** Avigilon's Unusual Motion Detection
  (US10878227B2) keeps a histogram per macroblock — direction in 30° bins, speed, and a
  no-motion bin — over hourly intervals clustered into at most four day/week patterns,
  with an exponential average whose window is about eight hours of frames; Unusual
  Activity Detection (US11302117B2) is the same framework over detected-object counts on
  a spatial grid. Two weeks and one week of learning respectively, events reported while
  learning, no operator feedback into the model, and a rarity slider as the only control.
  iCetana (server GPU, 400 cameras a server) begins alerting at 24 h and calls a week a
  baseline. Lumana keeps per-camera speed / dwell / time-of-day statistics and puts them
  in search, not alerts. Every camera-native vendor (Axis, Hanwha, Hikvision, Dahua,
  Bosch, Verkada) ships pre-trained detectors plus hand-drawn rules; no edge SoC runs a
  behaviour baseline. Operator feedback reaches a model in three places — the UAD patent
  allows a false-alarm mark to alter the statistics, Hikvision's Learn-by-Example, and
  Irisity — and none publishes what it changes. Retail loss prevention is a different
  mechanism again: Everseen and StopLift score the gap between video object events and
  the POS scan list; Veesion is supervised gesture classification with staff
  accept/reject as the training feed, and the one independent figure is a store
  manager's "three-quarters false". **No vendor publishes alerts per camera per day**;
  the only published rates are IPVM's for perimeter analytics — under one false alert
  per camera per month at best, one per camera per night in rain and glare. And the
  academic route is measured not to travel: normality models score 0.70 AUC on the
  camera they were fitted on and 0.50 on any other, which at 0.9 recall is about 26,000
  false alarms an hour (Rashidi 2026, arXiv 2606.29506); Sultani 2018's reconstruction
  baseline fires on 27% of normal video. That is *why* the market's normal is per
  camera, per cell, per hour, and statistical. So the memory this plan should build
  first is what the market's learned-normal actually is — per-camera, per-hour
  statistics over L1 output — and the part the market does not have is the graded
  disposition store, which is the only route to the number no vendor publishes. §9.10
  orders the work.


## 2. The two packages

```
src/syncai_bev3d      runs ONCE per camera, offline    →  camera.json
src/syncai_hydranet   runs EVERY frame, continuous     →  boxes + keypoints → tracks
                                                          in metres → events → alerts
```

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


### 2.2 `syncai_hydranet` — the per-frame network

Trunk, frozen by measurement: **RegNetX-800MF + BiFPN ×2, 96 ch, P3–P7, input 640 × 1120,
FP16.** (64 ch measured slower; input below 640×1120 measured −21% relative site mAP —
boxes under 8 px do not span a stride-8 cell and are unlearnable, not merely hard.)
| head | output | status |
|---|---|---|
| **detection** (FCOS) | `person`, `bag`, `device`, `boxed_stock` (+ `stack`, open — §7) | trained today. `device` carries collected-but-merged sub-labels `iphone \| ipad \| macbook` — they stay merged until each sub-class exists on ≥2 test cameras (§4.4), then splitting is a deliberate re-baseline |
| **pose** | 17 keypoints / person | next to land. ViTPose is the offline teacher; measured over 66,599 verified person boxes, **99.9% of people clear the 32 px bottom-up floor** at network scale (median height 178 px) |


### 2.3 The layers above — where behaviour actually lives

```
L0  pixels  →  boxes + keypoints      hydranet, every frame, GPU
L1  boxes   →  tracks in METRES       tracker + camera.json homography, CPU
L2  tracks  →  per-person facts       crop heads, ~every 3 s per track, GPU
L3  facts   →  events                 rules in metres and seconds, CPU
L4  events  →  judgement              VLM on trigger, GPU queue
```

#### 2.3.1 What L1 emits — the vector space, as a contract

**Built 2026-08-25: `analytics/world.py` (`WorldFrame`, `WorldObject`, `world_frame`).**
`analytics/stage.py` typed what enters the second stage and typed it in **pixels**; the
metre side had no type, so `dwell.track_ground_path`, `events/zones.py` and `cli/scene.py`
each called `pixel_to_ground` and kept the answer in a private shape — the same failure
`stage.py` records as its own reason for existing, one coordinate system later.
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

### 4.2 The teachers — SAM 3 + Grounding DINO

Both already live in the wheel (`data/teachers/`). Their jobs, in value order:

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

### 4.6 Retention — what is kept, for how long, and why the tiers differ

Decided 2026-08-30. Until then this document said nothing about retention and neither did
anything else in the tree: `grep -rniE "retention|PDPA|GDPR|consent"` over `src`, `tools`,
`scripts`, `docs` and CONTRIBUTING returned one line, and it was about git history being
uneditable. A product that records customers in a shop in Taiwan, where the 個人資料保護法
applies, had no stated answer to "how long do you keep this".
| what | where | kept | why that number |
|---|---|---|---|
| raw store clips | `datasets/studioa_clips/` | **30 days** | long enough for one incident investigation and one re-cut of a figure; past that a clip is a liability holding no answer the measurements have not already extracted |
| imagery derived from them | `runs/**/*.{jpg,png,gif,mp4}` | **90 days** | one model iteration. A crop sheet is evidence for a verdict, and a verdict is re-read when the next training set is assembled |
| measurements | `runs/**/*.{json,jsonl,npz,log,yaml}` | **kept** | numbers, not pictures. Deleting these deletes the apparatus behind every claim in this document |
| static plates | `datasets/studioa_static/` | **kept** | the temporal median removes every moving person by construction; a plate is the empty shop |
| disposition log | the JSONL store | **kept** | `frame_ref` is a *pointer* — clip path plus frame offsets — not an image. When the clip expires the pointer stops resolving, which is the deletion, and the operator's verdict survives as a number |
| published figures | `assets/` | **permanent, and unerasable** | they are in git history. CONTRIBUTING already states this; §4.6 restates it as the reason the audit gate in front of `assets/` is the strictest one here |

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
| 2 | **commission all 48 cameras** — build the 4-click tool and zone tool, run the pipeline | per-camera `camera.json` + a 1 m grid rendered on one real frame per camera | the grid looks right **to my own eyes on every camera**. **8 of 48 shipped** (`runs/commission01/REVIEW.md`), Taichung-cam05 withdrawn — two furniture checks agree its cells are over-scaled. **"Scale-measured" means the 1.70 m person-height prior, not a tape measure** (§7.19): every shipped `scale_source` reads `person_height_median_vs_1.7m_prior_nNN`, on 15–37 boxes. Masks, walkable outline, shelf ROIs and a commissioning 3D scene are written for all 8 (`masks_pass.py`, `scene3d.py`; tables 0.78–0.97 m fleet-wide); 5 mask sets clean, the Tao-Hsin pair partial and Kaohsiung-cam04 missing its pillar. **The Tao-Hsin caveat is measured (§7.21) and is the whole of what limits the 3D render's coverage** — the fixtures are proposed and then classified `wall`, and neither clustering nor paint order moves that camera's map by a pixel. **71 service zones across the 8, fully automatic** (§7.19): a zone is the floor *beside* a fixture, not its footprint, because a footprint is a region no shopper can occupy. Blocked on the physical world for the rest: 14 cameras need a per-store visual reference, 4 need mount-type triage, Taichung-cam05 needs its NVR stream setting; the 7 Kaohsiung person-score cameras are answered per camera in §7.12. Remaining tools: the 4-point ground-calibration click tool (§2.1b) and the tamper reference (§2.1h). **Inventoried 2026-09-03**: of 48 cameras, 23 are `selling_floor` and ALL 23 are onboarded (calib.json exists, `fleet_hardware_assumed` vfov on 22, tile-grid on Taichung-cam01); 8 of the 23 are commissioned, so the real stage-0 backlog is **15 selling-floor cameras**, each needing only the teacher passes plus the zones/FP human confirmation — no new calibration work. The other 25 are 17 back_of_house (out of product scope), 6 dead (luma 1.1), and 2 already counted. **This gate has no instrument in the tree, found 2026-09-10**: `render_metre_grid` drew the picture the eight cameras were judged on and was deleted in `5c209c7` as uncalled — correctly, since the scripts that drove it had already left — so the 15-camera backlog cannot be judged until it is rebuilt. The converter feeding it is in the same state: `commissioning.from_onboard_calib` turns a calib scan into `camera.json` and its only caller anywhere is a test, so step 0-4 of the runbook in README has no command. |
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
| 6 | **L3 end to end** — **RUN 2026-09-09, and it produced the number section 1 defines success as.** `scripts/step6_events.py`: a commissioned `camera.json` (not a hand-fitted pose), its service zones, the event layer, and every event filed through `dispositions.record_alert` — the store section 9.2 found empty now holds **263 rows**. 8 cameras × 4 clips × 5 min = **162 minutes**, 47 min GPU, one code version (the `src/` content hash is identical before and after, once a file another session added mid-run is excluded). **Open hours: 129.6 alerts per camera-hour** = 1,478 per camera-day over 12 trading hours, at this run's demonstration `loiter_seconds=8`. **The threshold is the lever, not the geometry**: the same footage re-read at 30 s gives 427 per camera-day, at 120 s gives 30, and **at the 240 s section 5 itself uses as the archetype it gives 6 — inside step 7's single-digit gate**. Closed hours: 1 event in 40.5 camera-minutes. **Two alerts read against their video by eye** on the first single-clip run: one correct-but-useless (a member of staff at their own workstation, which is step 9's measured problem) and one whose foot point the frame decided rather than the floor (now refused by `geometry.FrameBounds`). **What is NOT done, and it is the gate**: none of the 263 has been graded. 15 survive the 240 s threshold and 14 of those are `occupancy_exceeded`, which its own docstring says counts *tracks* and so over-counts by exactly the fragmentation rate. **Re-read 2026-09-10 by `scripts/step6_reread.py`, which is the arithmetic behind those four numbers made an instrument, and it corrects one of them**: the 6 at 240 s was a single event, and it is Tao-Hsin-cam04's 00:02 local clip — a closed shop, one track for all 303 s, the camera whose night detection step 4 already names — counted into the open-hours rate. Open hours alone read **1,472 / 421 / 24 / 0** loitering per camera-day at 8 / 30 / 120 / 240 s, with the 14 `occupancy_exceeded` at every threshold; past 120 s the log *is* occupancy, and occupancy counts tracks. **The tracker moved the same day**: the first run tracked at a single 0.35 threshold while serving already ran the birth 0.35 / keep 0.20 band that §7.11 measured at 91% of two-stage's gain; `step6_events.py` now runs the band and `--single-threshold` reproduces the first log. **The band re-run landed the same evening** (`runs/step6_fleet02`, 36 clips, 64 min GPU with the CPU shared; read with `--common` on the 8 cameras the two runs share): tracks **1,279 → 1,185**, loitering **248 → 297**, `occupancy_exceeded` **14 → 39**; per camera-day, open hours, loitering **1,763 / 475 / 18 / 0** at 8 / 30 / 120 / 240 s. Fewer tracks and more events is the band doing what §7.11 measured — low boxes continue a track instead of ending it, and a continued track clears a threshold a fragment could not — and the tripled occupancy is the same thing seen from the other side: more people held at once, on cameras where §7.11 counts eight to twelve real people at a counter with four at 0.35. Whether those are shoppers or fragments is what step 7 grades. **Three of the longest new loiters looked at by eye**: two are a member of staff at their own counter for 108 and 115 s (Kaohsiung-cam04, step 9's problem, continuous now where the first run had no event); the third named a zone its floor points were never in — and that was not the event layer but **the fleet's `camera.json` files rewritten by another session at 16:48–16:52 while the run was reading them**, fixtures renumbered, caught only because every alert row carries the calibration hash (`sha256:09cb…` on the rows, `11353b…` on disk, and the first run's rows carry a third). The review surface now refuses to draw a row against a file it was not filed under. A rewritten calibration is the same failure as a knocked camera, one artefact earlier, which is §10.2 A3's case | an event log readable against its video | events match what the clip shows |
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

11. **The Kaohsiung person-score investigation is two investigations, and the fix is not on the
   inference side.**

12. **The seven blocked Kaohsiung cameras split, and the block was never a group property.**

13. **Step 5's geometry gate: 7.3 cm, and what it is agreement with.**

14. **The `fall` height threshold is measured now, and on its own it does not work.**

15. **`staff/customer` has its first measurement, and nine colour numbers beat every embedding
   in the tree.**

16. **Step 8's precondition holds: the features separate the classes, and they beat the
   geometric rule they would replace.**

17. **Step 8 has a model, and it passes — after failing first, which is the useful half.**

18. **Two-stage tracking's identity was measured, the instrument was wrong first, and fixing it
   is a component step 9 also needs.**

19. **"Scale-measured" meant the person-height prior all along, and a second fleet arrived that
   has more of it than the first.**

20. **The `person` detector has never seen a labelled shopper from these stores, and half the
   site training images teach it that shoppers are background.**

21. **`masks_pass` is not the bev-3d bottleneck, and three explanations for the missing
   furniture are ruled out.**

22. **A merchandise wall was being drawn as a small cabinet on every camera turned the other
   way, and the tripwire could not see it.**

23. **The staff/customer classifier became something that can be applied, and the number that
   licensed it is per camera rather than the headline.**

24. **The face blur was missing people, and the instrument that found it had to be built wrong
   twice first.**

25. **"Relative relationships must be correct" turned out to be a different requirement from
   "the numbers must be right", and it is the one the scene was failing.**

26. **The checkout counter is not a trained class, and it is being classified as merchandise
   shelving.**

27. **`implausible()` names seven fixtures across the fleet and nothing reads it.**

30. **The mesh figure can be driven by the pose head, and the thing that stops it is constraint
   design rather than data.**

31. **The shipped model was walked into three foreign venues — metro, mall, airport — and the
   failure is a confidence slide, not a collapse.**

33. **The text-embedding head, tested with a vocabulary it was built for, trains to parity and
   does not win. 2026-09-08.**

34. **person01's throughput on the pro6000, measured on an idle card 2026-09-08**

35. **70.4 is not the wrong number, and this is the first independent check of it. 2026-09-08.**

36. **The counting-line escape from fragmentation was tried on Kaohsiung-cam04 and the camera
   cannot supply it, 2026-09-07.**

37. **The backbone carries no novelty signal, and the control is what says so. 2026-09-08.**

38. **The store's axis is read off the floor's joints, and the fixture-blob vote was 36 deg
   off them on Taichung-cam01. 2026-09-10.** `floor_axis.floor_line_axis`, sharpness-gated,
   blobs as fallback; five of nine cameras take the floor.

39. **Cameras over one floor disagreed by 28% under the person prior, and the project now
   picks its own rulers: the catalogue tile by store consensus, the store's counter height,
   the prior -- two agreeing to move. 2026-09-10.** `rulers`, `scale_rulers.py`; first run
   moved Taichung-cam01 x1.277 and nothing else.

40. **Five of nine cameras' welded tables were `masks_pass` objects thrown away at the class
   PNG. 2026-09-10.** `masks/objects.png`; one box per object.


## 8. What the health audit changed, and what it taught

A best-practice audit ran on 2026-09-04 over the whole tree (8 sweeps: packaging,
config, duplication, comment truth, tests, ML engineering, repo hygiene, privacy),
raising 34 items. All 34 are closed. The working list they were tracked on is gone,
as it said it would be: the fixes are in the code, the arguments are in the commits
that made them, and what generalises is here.

## 9. The distance to the product, read across the steps

Written 2026-09-09. **This section measures nothing new.** Every figure in it is cited
from the section that took it, and where a status here and a row in section 6 disagree,
section 6 is right and this is stale -- it is a reading of the table, not a second copy of
it. What is mine and not the evidence's is the **ranking**; the ordering below is a
planner's judgement about which gap makes the others unreadable, and it is arguable.

### 9.1 What is solid

Named first because the list below is long and it would otherwise read as a verdict on
the whole system, which it is not.

### 9.2 Nothing has been graded, so no number here is an accuracy

Section 1 states the rule: *no site figure is an accuracy until a human has graded it --
until then it is an agreement with the teacher models.* Section 4.5 names the instrument
that would fix it (shadow grading) and calls it the human test set, free and accumulating.

### 9.3 The known recall failure sits exactly where the product sells

Measured 2026-08-26 over four commissioned cameras, 40 frames each, and recorded in
`serving/decode.py`: at the shipped 0.35 threshold the dense head marks **73 person
regions with no box at all -- 20% more people than the box head returned** -- and every
one, cropped and looked at, is a shopper whose lower body is behind a counter or a display
table. Dense `person` IoU is 0.885 against detection mAP@50 0.302.

### 9.4 Tracks do not survive long enough to measure a duration

Section 7.11: 202 tracks over 24 minutes, 43% ever enter a zone, **median visit 3.2 s**,
and 58% of endings are mid-view deaths -- of which **86% still have a box on the person**,
at a median score 0.338 against a 0.35 threshold.

The band is in the step 6 tracker since 2026-09-10 (1,279 → 1,185 tracks over the same
32 clips, §6 step 6), the appearance gate reaches the clip loop (`describe` on
`track_clip`), and the three-arm endings measurement -- single, band, band + gate -- is
`runs/endings04`, `06` and `07`; §10.2 A2 reads it. **By the night of 2026-09-10 the
tracker is two-stage on both paths and its share of what remains is measured at zero**
(`runs/endings09`: 40 witnessed mid-view deaths, 24 demoted, 13 taken by a neighbour's
track, 3 vacated, 0 available). The 3.2 s median visit is now the detector's number to
move, at the crowded counter, which is §9.3 -- the same failure seen from L1.

### 9.5 The metres are a population prior, fleet-wide

Every commissioned camera's `scale_source` reads
`person_height_median_vs_1.7m_prior_nNN`, **on 15-37 boxes** (step 2, section 7.19), and
vfov is `fleet_hardware_assumed` on **22 of the 23** onboarded cameras -- Taichung-cam01's
tile-grid pin is the exception, and §7c.31 refused the same pin at two other venues
on a flat k1 sweep. Taichung-cam05 was withdrawn when two furniture checks disagreed.
Since 2026-09-10 the prior is checked, not trusted: §7c.39's rulers read the floor's
joints and the store's counters, and Taichung-cam01 is the first camera whose metres
come from the tile (x1.277 over its prior; the counter ruler says x1.15 -- the truth is
between, and one witness does not move it again).

### 9.6 The chain has never run end to end

Verified 2026-09-09, before the commits below: `world_frame`, `pixel_to_ground`,
`SecurityEvent` and `dispositions` appeared **zero times** in `serving/` and in
`scripts/serve_pilot.py`. The serving path stopped at L0 plus a tracker, with boxes still
on the letterboxed network canvas -- the pixel frame `analytics/world.py` gained
`canvas_region` for the same day, and whose two wrong readings cost 2.4-3.4 m in metres
that carry no NaN and raise nothing.

**Closed on the pilot, 2026-09-10** (§10.2 A1, A4, A5): `serving/alerts.py` calls
`CameraState.world_frame` → `events/live.py`'s `ZoneMonitor` → `record_alert` after every
`CameraState.update` on a commissioned camera, `scripts/serve_pilot.py` builds one per
commissioned stream, and the first row the serving path ever filed was a loitering alert
on Kaohsiung-cam04's 00:00 clip — an empty shop, a phantom person on an IR frame, and the
first alert graded through `deploy/retail-security/review_server.py`: rejected. The live
rules are held to the offline ones by `tests/test_live_events.py` (same events; `value`
and `frame_end` at the crossing). **The two trackers became one the same evening**:
step 6 defaults to the shipped two-stage tracker built by `bytetrack.shipped_forward`
exactly as the pilot builds it (the four-arm measurement in §10.2 A2 put two-stage at
38 mid-view deaths against the band's 48, and `band_probe01` had shown its dwell is not
inflated by coasting), and the appearance gate runs on it in both places. What still
keeps a row from being byte-comparable across the two paths is the **engine**: the pilot
runs the uint8 TensorRT plan on a 512×640 canvas for throughput, the offline runner the
checkpoint at 640×1120. Same tracker, different detector, until the pilot runs the
1120 plan.

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
* **The two trackers are one since 2026-09-10 evening.** §7.11 recorded two-stage as
  measured and not adopted; it is adopted now, on the four-arm measurement (§10.2 A2),
  and `scripts/step6_events.py --tracker band|single` is how the older logs are re-read.

### 9.8 Out of domain it is measured to fail, and that is a scope statement

§7c.31: out of domain the person-score distribution slides into the threshold band
(a 0.15 to 0.30 cut costs 36-67% against 11% at home), ad posters detect as people every
frame, night OSD text mints phantom devices, and `fixture` is approximately zero outside
retail so zones cannot be drawn at all.

### 9.9 The read

**The gap is instruments, not model capability.** The model may already be good enough for
step 6; nothing in this tree can currently answer that, because every site number is
agreement with a teacher or with a model-labelled set, and the one instrument that would
break the circle is a schema with no rows in it.

### 9.10 What this ordering implies for the next steps

Not a build order -- section 6 is the build order -- but the sequence this reading argues
for inside it. Completed 2026-09-10, when the user asked where the model goes next --
memory in the network, or a VLM distilled into it -- and the answer is neither, yet:

1. **Step 7, shadow mode**, because it produces the human test set that every item
   below needs. Until a person has graded alerts, memory and distillation alike learn
   the teachers' opinion a second time (§1.1).
2. **Track persistence at L1** -- the measured 3.2 s (§9.4). The survival band is the
   91% arm (§7.11) and on 2026-09-10 it became the tracker step 6 runs; the appearance
   gate (`appearance_thr`, licensed on five cameras) is the next lever, and neither is a
   model change.
3. **The VLM at L4, on trigger**, with its verdict logged beside the operator's in the
   disposition store. That is distillation data that costs nothing and is the only kind
   here that is not a teacher's opinion.
4. **A per-camera, per-hour baseline computed from L1 output** -- counts, dwell and
   speed on the world frame, no network. It is what the market's "self-learning" is
   when the patents are read (§1.1), and it can be audited by hand.
5. **Only then**: VLM verdicts distilled into L2 crop heads (`staff/customer`, the
   printed-person false positives, a zone's kind), and a learned model of normal.

What it is not: memory inside the per-frame network, or a VLM inside it. §5 rule 6 and
§7.4's budget both refuse that, and RegNetX-800MF would carry a VLM's semantics no
better than it carries its weight.


## 10. The next plan — three gates over section 6's steps

Written 2026-09-10 at the user's request for a CTO / architect's plan. It adds no steps:
section 6 is the build order and section 9 the ranking of gaps. What this adds is
**three gates in calendar order, each defined by the one artefact that makes the next
gate readable**, the architecture decisions the gates rest on, what is deliberately not
done, and the decisions only the user can take. Dates are targets from 2026-09-10 with
one person plus AI sessions and one shared GPU; every "done" below still means section
6's gate, not this section's summary of it.

### 10.1 The principle every gate follows

**A number nobody has graded is not a result, and the model is not what is blocking the
grading.** Section 9.9's read stands: the gap is instruments. So the order is (1) make
the chain run live and put its alerts in front of a person, (2) fix the three measured
failures against that person's verdicts, (3) scale from one store to the fleet. Capability
work that does not have a graded consumer waits — that is what section 1.1 says about
memory and the VLM, and it is the rule here, not an exception for those two.

### 10.2 Gate A — one store, live, graded. Target 2026-10-03.

The artefact: **two weeks of operator dispositions from one store**, filed by the
serving path rather than by an offline script, and the first precision number this
project has ever had.

| # | work | closes | done when |
|---|---|---|---|
| A1 | the serving path carries L1 → L3 → dispositions: `world_frame`, `zone_events`, `record_alert` called from `serving/`, not only from `scripts/step6_events.py` | §9.6 | `serve_pilot.py` on one live stream files the same rows step 6 files offline, byte-comparable on a recorded clip. **Wired 2026-09-10** (`serving/alerts.py`, `events/live.py`); the two paths run one tracker since the same evening, and byte-comparability is now blocked only on the pilot's 512×640 uint8 engine against the offline 640×1120 checkpoint, see §9.6 |
| A2 | tracker in serving = the measured band **plus the appearance gate** (`appearance_thr` per camera; a descriptor hook in `clip_tracks.track_clip`) | §9.4 | median track life on the eight sweep clips, single-threshold vs band vs band+gate, by `scripts/track_endings.py`; identity checked by eye on the ten longest. **Measured 2026-09-10** (`runs/endings04/05/06/07`, same eight 900-frame clips, same weights): tracks **202 / 100 / 109 / 115** and mid-view deaths **111 / 38 / 48 / 53** for single / two-stage / band / band+gate. The band alone reproduces `band_probe01`'s 109 exactly and takes **86% of two-stage's mid-view-death reduction** with no Kalman; what the filter still buys is ten deaths, eight of them on Kaohsiung-cam04. The gate moves only that camera: nine re-associations refused after a gap (the dead track's ending then reads as `lost` at gap 1, IoU 0.55–0.65 — the detection it refused started a new track), four fewer of the band's own, net +5 tracks, at the camera's calibrated 0.39. **Whether those nine are two shoppers or one is unverified**: the endings instrument records no frame for a refusal, so the eye check this row asks for needs `Tracker` to record the refused pair with its frame first. **Decided the same evening: two-stage is the tracker, on both paths**, with the gate on it (`bytetrack.OfflineForward(appearance_thr=...)`, every refusal recorded as a `Refusal` with its frame — the instrument the eye check needed); step 6 defaults to it and the pilot cuts descriptors from the canvas frame for the five licensed cameras. The band and single arms stay as flags for reading old logs. Smoke on Taichung-cam01's first clip with the gate at 0.335: 29 tracks, 3 events, 3 refusals; the two staff loiters (48.6 s, 49.4 s) unchanged from the band arm. **And the tracker's own share of the remaining deaths is zero** (`runs/endings09`, two-stage, 2026-09-10 night): the endings instrument now records where each mid-view death happened and whether the box "available" next to it was already observed by another live track; of 40 witnessed deaths, **24 demoted, 13 taken, 3 vacated, 0 available**. Every one of the 13 that read as a tracker failure the same afternoon was, on its strip, a huddle at a counter with the dead track's last box at 0.20–0.30 and the high-band box beside it on the neighbour — §7.11's crowd demotion wearing the tracker's label. What is left for the tracker to fix is nothing; what is left is B1 |
| A3 | **plate drift / tamper check** (§2.1h): nightly static plate diffed against the commissioned one; a moved camera stops filing metres and says so | §9.5's silent failure | a deliberately nudged camera is refused within one night. **The same failure arrived from the other side on 2026-09-10**: the calibration files were rewritten under a running log and the rows' calibration hash was what caught it (§6 step 6); the serving path should refuse to file against a `camera.json` whose hash changed since it started, which is this row's check with the plate as the second witness |
| A4 | **the grading surface**: a daily review sheet generated from the disposition store — frame crop (blurred), the event row, accept / reject / "staff" — writing disposition rows back. HTML from a script, no server | §4.5 | an operator grades a day in under fifteen minutes. **Built 2026-09-10** as `deploy/retail-security/review_server.py` (a localhost stdlib server rather than a static sheet, so a verdict is one click); the frame is drawn from the calibration alone, and the fifteen-minute gate is unmeasured until a store grades a day |
| A5 | **per-store policy as a file**: loiter seconds, occupancy, open hours, after-hours rule; the demonstration values in step 6 leave the code | §3's placement rule | `step6_events.py` and serving read the same file. **Done 2026-09-10**: `configs/policy/demo.yaml` via `analytics/policy.py`, read by step 6, the re-read and the pilot; the after-hours rule is deliberately not a field yet because no producer reads it |
| A6 | commission **every camera of the pilot store**, not eight of the fleet; the floor bench scores each; k1 / vfov attributed per camera as now | step 2 | one real frame per camera with the 1 m grid, seen by eye |

Exit: alerts per camera per day at the store's thresholds is single-digit **after**
`staff` rows are excluded, and the reject log is readable as a pattern (which camera,
which zone, which kind). If the rate is not single-digit the thresholds move, not the
model — that is what section 1.1 measured.

### 10.3 Gate B — the three measured failures, scored on graded rows. Target 2026-11-07.

The artefact: **precision on the graded set moves, and the operator's list of missed
incidents shrinks**. Nothing in this gate ships without a before/after on Gate A's rows.

| # | work | evidence it rests on | done when |
|---|---|---|---|
| B1 | **occluded-person recall**: the dense head's 20% of shoppers behind counters become boxes — the crop-stage fallback `models/heads/pose.py` reserved, or a dense-to-box proposal at the counter zones only | §9.3 | detection mAP unchanged elsewhere, counter-zone recall up on the graded misses |
| B2 | **`staff/customer` licensed on every pilot camera**: the three uniform photos per store, and a VLM at L2 as the teacher for the cameras the colour statistics refuse | step 9, §7.15 | balanced accuracy ≥ 0.90 held out by camera on all pilot cameras; `reach_to_shelf` no longer fires on staff at their workstation |
| B3 | **the VLM at L4, on trigger**, on the card's reserved budget: reads the frames of an alert already filed and writes its verdict as a disposition row beside the operator's | §1.1, §7.4 | agreement between VLM and operator measured per event kind; the disagreements are the next training set |
| B4 | **per-camera, per-hour baseline** from L1 output: counts, dwell, speed on the world frame, kept beside `camera.json`, no network; each event gains a `rarity` field against its own camera's history | §1.1's survey; §7.37 | rarity separates accepted from rejected rows better than the fleet constant does — or it is dropped |
| B5 | the behaviour head gets a serving consumer, or is shelved: `fall` / `crouch` from the trained sequence model instead of the geometric rules | step 8, §9.7 | one false alarm per eight minutes becomes one per shift on the crowded camera, else it stays off |
| B6 | **crowd-aware assigner retrain** (ATSS / OTA) — only if Gate A's misses concentrate in crowds | §7.11 | a re-baseline, run as one systemd unit, compared on the graded set |

Exit: precision on the graded rows reported per camera and per event kind; the missed
list carries a cause per item (recall, tracking, staff, policy).

### 10.4 Gate C — from one store to a chain. Target 2026-12-19.

The artefact: **the fleet under the 20-minute commissioning budget, serving at the
measured 96 streams, and the data engine closing**.

| # | work | done when |
|---|---|---|
| C1 | all 48 cameras commissioned; the 4-click tool exists (step 2b) and a human's part is under five minutes a camera | `runs/commission01` holds 48 files, each with a grid seen by eye, each with a floor-bench score |
| C2 | serving end to end at 96 × 5 fps: NVDEC decode, engine, NMS, tracker, events, dispositions, on one card, measured not engine-only | the §7.4 number re-taken with L1–L3 in the loop |
| C3 | the retail reading scoped (§7a.5): footfall, dwell, paths, queue length from the same L1 tracks, as a daily table per store; no new model | a store manager reads it beside their till data |
| C4 | **distillation, now with a test set**: VLM and operator verdicts from B3 into L2 crop heads (`staff`, printed-person false positives, zone kind) | the small head matches the VLM's agreement with the operator on held-out cameras |
| C5 | the data engine closes: dispositions → training set with provenance (checkpoint, commit, calibration hash already on every row) → the next model, scored on rows it never saw | one retrain whose test set is graded rows, and whose gain is stated in alerts, not mAP |

### 10.5 Architecture decisions this plan fixes

* **The world frame is the contract** (§2.3.1). Every layer above L1 reads metres and
  seconds from `WorldFrame`; nothing above L1 sees pixels. Pixel-frame bugs have cost
  2.4–3.4 m silently twice (§9.6); this is the boundary that stops the third.
* **The disposition store is the spine of the data engine.** Every training set, every
  precision figure and every distillation cites rows in it; a model that cannot say which
  graded rows it was scored on is not compared.
* **The VLM is a teacher and an adjudicator, never a student's target.** It labels at
  commissioning, judges at L2 every few seconds per track, and adjudicates at L4 on
  trigger, on the reserved 68% of the card. Nothing distils *into* the per-frame network
  from it; what distils is verdicts into crop heads (C4).
* **Memory is statistics on L1 output, per camera, per hour, with no learned weights**
  (B4). It is the mechanism the market ships under that name (§1.1), it is auditable by
  hand, and it produces a `rarity` field, not a new alert type. A learned model of normal
  waits for enough graded anomalies to score it.
* **One network, frozen by measurement.** RegNetX-800MF + BiFPN at 640 × 1120 stays until
  a graded miss names a failure the heads cannot fix. No fifth class, no COCO share
  changes, no resolution change.
* **Self-calibration stays the product** (2026-09-03). vfov and k1 are estimated and
  scored by the floor bench; the plate drift check (A3) is what makes a metre trustworthy
  over time rather than only on commissioning day.
* **Identity stays out** (§5 rule 5): no face, no re-ID, no cross-store tracking; the
  appearance gate is within-camera, within-gap, and splits tracks rather than joining
  people.

### 10.6 Not done, on purpose

Memory inside the network. A VLM inside the per-frame network. Re-identification. A
`stack` class before a graded miss asks for it. Any figure re-cut outside a batched scene
commit (the figure tax). The robot line (secondary since 2026-08-19; frozen until Gate C).
Buying coverage with more frames from already-annotated cameras (§5 rule 4). Any
per-frame dense scene understanding (§5 rule 6).

### 10.7 Risks, ranked

1. **No graded data arrives** — the store, the operator, or the fifteen minutes a day do
   not materialise. Then every number stays an agreement. Mitigation: Gate A is scoped so
   that one person grading one store's sheet is the whole ask.
2. **The metres are wrong on cameras where the prior is thin** (vfov guessed on 22 of 23,
   15–37 boxes). A loiter threshold in seconds does not care; an occupancy zone in metres
   does. Mitigation: every alert row carries the calibration hash and `scale_source`; the
   floor bench scores every commissioned file; A3 catches the drift.
3. **The crowded counter is where the product sells and where recall is worst** (§9.3,
   §7.11). Mitigation: B1 first among the model items; B6 only if the graded misses say so.
4. **Privacy in shadow mode**: customers of a Taiwanese store, PDPA. Mitigation: §4.6's
   retention tiers, blurred crops on the review sheet, no identity anywhere, and the
   disposition row is a pointer that expires with the clip.
5. **One shared GPU and one shared checkout across sessions**: a training unit killed by
   an edit, a run started under a stale tree. Mitigation: every long job is a systemd
   user unit; commits are atomic; nothing edits `src/` under a running job.

### 10.8 Decisions only the user can take

1. Which store is the pilot, and who at it grades (a named person, fifteen minutes a
   day).
2. The store's policy values: loiter seconds by zone kind, occupancy per zone, opening
   hours, the after-hours rule.
3. Three uniform reference photographs per store (waiting since 2026-08-27).
4. Footage for the remaining 40 cameras, and whether the fleet is 48 (§2's count) or
   what the corpus actually holds.
5. Whether the retail reading (C3) is in scope by December, or the chain buys the
   security reading alone first.
