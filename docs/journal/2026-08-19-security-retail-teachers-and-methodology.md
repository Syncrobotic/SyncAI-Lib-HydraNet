# 2026-08-19 — the person teacher swap, the first labelled clip, and what a plate can calibrate

Planner-session record. Context: the stated goal is now the **security + retail CCTV
model** (robot line continues, secondary). An external proposal arrived recommending,
among other things: exploit the fixed-camera background, ensemble SAM 3 with Grounding
DINO for detection, offline non-causal tracking for track labels, per-camera one-time
geometry from monocular depth, an operator-feedback data engine, and DINOv3 feature
distillation. This entry records what was measured today, what was decided, and what was
deliberately deferred. Three experiments ran; all numbers below have runs behind them.

Same-day related work: the doc audit (commits `b74f264`, `025d100`, `eb49467`) and the
sweep-row ↔ run-name table now in ARCHITECTURE_REVIEW.md (`coco10` = share **1.0**).

---

## 1. DECIDED: `person` teacher is Grounding DINO at 0.35, replacing SAM 3

Measured on the **exact frames** `sam3_person_boxes.py` had already labelled
(`scripts/gdino_person_boxes.py`, frame-identical, no daylight gate, floor 0.10 so both
score populations are fully visible):

| measurement | SAM 3 (its 0.50 default) | GDINO (0.35) |
|---|---|---|
| day, 4 cameras, 32 frames | 146 boxes | 135 (per camera 21→20, 21→17, 103→94, 1→1) |
| night empty IR store, 12 frames | **229 false persons** | **0** |
| day:night separation | 0.9× | 18× at 0.30; ∞ at 0.35 |
| night score maximum | 0.55–0.60 (above its threshold) | **0.326** (below 0.35 — a measured gap) |

Cross-checks, all three passed:
- The 34 night boxes in [0.25, 0.35) score **0.890 median static share** against the
  plates (all in 0.8–1.0): they are the pegboard packets, not people.
- Day boxes on Kaohsiung-cam04 score **0.133 median static share** — identical to the
  SAM 3 real-person distribution measured 2026-08-18.
- Rendered frames confirmed by eye: every ≥0.35 box on the crowded service counter is a
  person (`runs/static_gdino_day/frames/`).

Consequences:
- **0.35 sits in a measured gap** (ARCHITECTURE_DIRECTION rule 2), so **no daylight gate
  is needed** — the night hallucination population never reaches the threshold. Night
  frames go from excluded to usable, which is when a store camera is most often asked a
  security question. Empty night frames are also candidate *negatives*; note
  `CocoDetDataset` currently drops images with no annotations, so wiring negatives in
  needs a loader decision, not just data.
- **Fleet batch produced**: `datasets/retail_person_gdino01` — 23 selling-floor cameras
  × 3 open-store slots × 8 frames = 552 frames, **1,141 person boxes** at 0.35 + NMS
  0.55, zero cameras empty, all frames colour. Against the SAM 3 bootstrap
  (`retail_person_batch01`: 146 boxes, 4 cameras, daylight-gated), that is 7.8× the
  boxes at 5.8× the camera coverage with no gate.
- SAM 3 remains the teacher for masks and long-tail merchandise prompts; the swap is for
  `person` only.

**Caveats, before anyone quotes these numbers wider than they reach:** the night side is
**one empty-store clip on one camera** (Taichung-cam09) plus its static-plate check; nine
of 48 cameras returned SAM 3 night hallucinations on 2026-08-18, so the gap should be
re-measured on those before "0.35 works fleet-wide at night" is a claim. And GDINO boxes
are still a teacher's opinion — the day counts agreeing with SAM 3 is agreement between
two teachers, not accuracy. The route to human ground truth is §2 and the operator
feedback loop (§4).

## 2. NEARLY UNBLOCKED: the labelled clip, via offline non-causal tracking

`scripts/offline_tracks.py` (new): ByteTrack-style two-stage association + constant-
velocity Kalman with noise rescaled by the **measured** frame rate (avg_frame_rate; cam11
7.0, cam04 8.0 real fps) + non-causal endpoint stitching (≤4 s, crop-encoder cosine veto
thresholded on the clip's own coexisting-fragment median) + interpolation across stitched
gaps. Same detection cache for both trackers; checkpoint from the gt proposals'
provenance (`hydranet_indoor_det60`).

| clip | metric | current tracker | offline tracker |
|---|---|---|---|
| cam11 (GT = 2 identities) | fragments / ID switches | 3 / 1 | **2 / 0** (idr 1.000) |
| cam04 (no GT) | fragments | 13 | **11** |
| cam04 | median track length | 17 frames | **95 frames** |

Design points worth keeping: the cam11 IDF1 "drop" (0.976→0.900) is entirely idp — the
low band correctly followed a person through 97 occluded frames where the
detector-relative GT has no boxes; and stitching is deliberately capped at 4 s because
`track_review` can **merge but never split** — an unmade stitch costs one human decision,
a wrong one is unrecoverable. cam04's real re-appearances sit at 4.4 s and 10 s, past the
cap, on purpose.

**Human's next step (≈10–15 min total):** cam11 needs **zero merge decisions** — verify
`runs/offline_tracks01/cam11/review_01.png`, apply with an empty `merges.json`, and the
first fully-labelled site clip exists, giving `idf1`/`id_switches` their first real
denominators (RETAIL_SECURITY §4 row 2, previously "blocks all of tier 3"). cam04 is 11
review rows, roughly 4–6 merges.

## 3. MEASURED: what a background plate can and cannot calibrate

`scripts/calibrate_from_plate.py` + `runs/calib01` (DA-V2 Metric-Indoor-Large on the
temporal-median plates, floor plane by RANSAC, vfov held as input — never fitted jointly,
that is the documented residual-is-not-a-scale trap):

- **Orientation: viable.** Against the Taichung-cam01 fitted anchor (2.38 m / 50.2° /
  vfov 70.4°), pitch recovered to **−0.7°**; the fit also surfaced Kaohsiung-cam04's real
  **−12.9° mounting roll** (others ≤1.2°). One plate, one inference, no people needed.
- **Scale: not viable from DA-V2.** Height 3.84 m against the 2.38 m anchor —
  **1.61× overestimate** at ceiling-CCTV viewpoint; person-height statistics on the same
  camera give scale 0.688 (11% from the tile-grid anchor); across three cameras the
  factor spreads ~10% at fixed vfov. This is **worse than** the NYUv2 zero-shot 1.18×:
  "metric" indoor depth degrades further as the viewpoint leaves the training
  distribution, so a fleet-constant correction is *plausible but unproven* (3 cameras,
  8–36 boxes each, two independent anchors still 11% apart).
- **vfov is the dominant unknown**: 55→85° swings fitted height by ~0.9 m and scale
  0.96→0.56. k1 moves ~2° pitch, 0.1 m.

**The recipe that survives:** tile grid → lens parameters; plate + DA-V2 → plane
orientation and roll; **one known length → the metre**. DA-V2 is never the link that sets
scale. Geometry on a fixed camera is a **per-camera install-time calibration artifact**,
not a per-frame prediction — which leaves the role of a per-frame depth head on the CCTV
line (`hydranet_hm3d_cctv.yaml`, in flight in another session) an open question that
session should answer, not this entry.

## 4. DECIDED and DEFERRED, from the proposal review

Adopted, in unblock order:
1. Person teacher swap + fleet batch (§1) — next: a `site_person` dataset entry
   (`class_mask` person-only) in a `b03` retrain variant, **queued behind the running
   `hydranet-retail-xl` training**, three seeds per rule 4.
2. Offline non-causal track labelling (§2).
3. Per-camera one-time calibration (§3) — pending: one known length per store.
4. **Operator-feedback disposition log, schema before product**: every alert stores
   frame ref, model hash, event basis/value/threshold, disposition, timestamp, into the
   training lake from day one. Known limitation to design around: confirms/rejections
   label the *most confident positive* frontier — precision only; recall failures never
   alert, so the statistical-anomaly mining layer stays the recall instrument.
5. Static plates stay what the measurements say they are: false-positive *evidence* and
   static-class consensus, never a foreground detector (the p10-floor and the
   21-of-103 deletions are the precedent).

Head-design consequence (no HydraNet architecture change): teacher ensembling lives at
the **dataset level** behind `det_vocab` + per-dataset `class_mask` — GDINO arrives as
data, the FCOS head does not change. Pose goes **top-down on crops** (person boxes are
244–336 px median) as a second-stage model beside the crop encoder, keeping the exported
graph single-frame pure-conv; the `Track.keypoints` slot already waits for it. Events
stay rule-based in metres; a learned action classifier, if ever, sits over pose/track
sequences in stage 3, labelled by staged acting + operator confirmations (+ VLM triage
once the on-prem privacy question is answered).

Deferred with reasons:
- **DINOv3 distillation** — nothing DINO exists in-repo; would be a new training-only
  auxiliary head with unproven benefit. Last in line; three-seed experiment if tried.
- **Sapiens / pose** — the fall-proxy result (48 candidate spans, none a posture, the
  proxy measures camera mounting) argues pose *up* the priority list, but that reverses a
  recorded user decision (attributes before pose) and needs the user, not this entry.
- **VLM triage of mined candidates** — cheapest pilot exists (`runs/fall_mining01`'s 48
  spans) but sending site frames to an external API is a privacy decision the user has
  not made yet.
- **ReID topology self-supervision** — real, but positives from time-windows are noisy
  in a dwell-heavy store; the clean free signal is **co-occurring tracks = definite
  negatives**. Needs the camera-transition topology measured first.

---

## Addendum, later the same day

**VLM triage pilot: ran with Claude reading frames directly — verdict "worth building,
but placed differently".** All 48 fall candidates judged (report:
`runs/vlm_triage01/REPORT.md`): 0 real falls, fully consistent with the recorded
conclusion, 45/48 high confidence. The classification is the payoff — 69% sideways-mounted
cameras (at least three, not the one previously on record), 48% frame-edge truncations,
35% seated, 21% lean-over-counter — and it distils into four front-layer rules:
(1) drop `touches_frame_border` candidates outright; (2) disable or rotate the proxy on
cameras whose median person aspect is ≥1 (sideways/top-down mounts); (3) zone-mask seating
areas; (4) require a track to have shown an upright box before it may trigger. With all
four, this batch reduces to single-digit genuinely-hard cases — which is what a VLM (or a
human) should be looking at. Cost measured: ~3 frames / 2–3k tokens per candidate.

**Licence stance ruled by the user: research use for now.** CrowdHuman and RAP v2 are
usable in the current phase; the obligation is a **pre-commercial lineage re-audit**
(recorded in RETAIL_DATA §7). RAP v2 arrived and is extracted (`datasets/RAP-v2`,
84,929 crops + annotations, with the identity-disjoint `rap_zs`/`peta_zs` splits);
PoseLift is on disk complete (151 pkl, Apache-2.0) as the tier-2 shoplifting benchmark;
the same bundle carried MSP60K / WIDER / RAP v1 as attribute reserves
(`datasets/_incoming/attr_bundle`).

**Scale without a tape measure:** the user ruled out manual site measurement; the metre
is being estimated from standard-dimension priors (door widths, tile pitch, A4 signage)
read off the plates, cross-checked against the person-height scale and the Taichung-cam01
anchor — `runs/calib02_priors` when it lands. Metre-space event thresholds tolerate
5–10% scale error; occupancy density is the consumer that does not.

**Both labelled clips landed the same afternoon.** cam11: the user's crop-sheet read
("both rows are one person") was answered by physics — the two tracks coexist in 234
frames, and the rendered frame shows both bodies; ground truth applied with zero merges
(2 identities). cam04: the constraint map (pairwise coexistence + sustained same-frame
IoU) reduced 55 possible pairings to three duplicate-track merges ({2,3}, {6,7}, {9,10})
plus one appearance question — which staffer returns as #8 and #11 — settled by a
four-crop lineup (shoes, watch, lanyard, hair): {1,8} female staffer, {6,7,11} male.
11 fragments → 6 identities. The lesson worth keeping: **crop-sheet resemblance misled
both a human and an agent on identical uniforms; frame coexistence and sustained IoU are
the evidence that cannot lie, so the review tool should surface them first.**

**The metre without a tape measure (runs/calib02_priors).** Standard-dimension priors
read off the plates lock scale to ±5–8% (±3–6% where a grout grid pins vfov):
Taichung-cam01's 45 cm tile prior lands +1% from the 2.38 m anchor; Kaohsiung-cam04's
tile and door cross-cut within 1.3%. The rule that carries it: menu-free references
(door height, standard mats) or two independent references that cross-cut — tile pitch
alone has a size menu (45/50/60/80) that residuals cannot break. Good enough for
line-cross/social-distance; occupancy density squares the error, so a measured length
per store stays the upgrade path there. Details now in GROUND_PROJECTION.md.

**Production serving target set by the user: 96 streams x 15 fps (= 1,440 frames/s) on
the RTX PRO 6000 Blackwell Max-Q that trains today.** Feasibility arithmetic: ~20-25
GFLOPs/frame at 640x1120 puts the compute need near 30-36 TFLOPS -- well inside the
card -- so the risks are everything except FLOPs: host-side decode (96 h.264 streams
need NVDEC; the system ffmpeg has no cuvid build), PCIe traffic (in-graph argmax cuts
D2H 17 MB -> 0.7 MB per frame; at target rate that is 25 GB/s vs 1 GB/s), per-stream
postprocessing, and Python in the loop. Fixed-batch engines + CUDA graphs are the
serving shape; the static-composite direction is a capacity multiplier on top, and the
fleet's real cameras emit 3-8 fps today, so 96x15 is the design ceiling, not the
current load. ONNX exported at batch 1/16/32/64 with embedded preprocessing and
in-graph argmax (`exports/pro6000/`); `scripts/bench_pro6000.sh` sweeps
fp16/best-precision TensorRT engines -- to be run ONLY on the idle GPU after the seed
chain finishes, because a throughput number taken on a shared card is not a measurement.

**Pose pilot (runs/pose_pilot01): the tier-2 slot powered on for the first time.**
ViTPose-base on the human-verified track boxes is usable at ceiling-CCTV viewpoint, and
its errors flag themselves — score tracks visibility, so events.py's 0.3 gate is
meaningful here. Failure modes, measured: counters hide lower bodies (ankle/knee <0.3 on
~60% of cam04 frames); a deep bow compresses to 33° in the image plane where the eye
says ~90°, so `fall_angle_deg`'s 55° threshold is unvalidated in this viewpoint and no
track ever crossed it; nose/eye are mid-confidence hallucinations under top-down views
(ears are the stable head points); frame-edge truncation fails honestly (red scores).
`pose_posture_events` and `reach_to_shelf_events` both ran on real input without
exception — reach_to_shelf produced 9 semantically-correct spans on the reaching child
(cam04 track 4), zero false spans on cam11. Three interface findings recorded in its
REPORT §4: a stale function name in tracker.py:66; the terrain map's pixel-space
contract is implicit (a canvas-resolution map silently yields zero events); and
wrist∈fixture fires on occlusion/pose error rather than on touch, because the wrist
pixel is usually labelled `person` by the wrist's own hand. Verdict: run ViTPose as an
offline teacher/miner (~8–11 fps full pipeline); do not distill yet — nothing measures a
distilled student, and the blockers on event quality are those two events.py semantics,
not the pose model.

**Output-stability baseline (runs/flicker_baseline01), the ruler for the optimisation
directions.** On the demo clip (b03_cw_xl, cam04, 900 frames): static-mask flip rate
1.50%/frame with **83.7% of flips oscillating back** — the flicker is vibration, not
change, which is the shape logit-EMA fixes. The largest single noise source is
**phantom `person` on static pixels** (21.7% of never-changing pixels read person at
some point; 40.9% of static-area flips) — the same family as the night packet
hallucination, and precisely what the in-domain person retrain targets: re-running this
same instrument on the seed checkpoints is the before/after. Second source: fixture's
class-boundary band flips at 5.3× its interior (wall 9.4×) — the static-composite
direction erases exactly those bands. Detection side: 9.5 boxes/frame with 40% of
IoU-matched chains lasting a single frame — rendering tracks with hysteresis, not raw
detections, is the fix. Consensus stable_share 95.4% (person-excluded eligibility 47.6%).

**First seed of the retrain, measured on the stability ruler (single seed — the
three-seed table follows).** Same instrument, same clip, only the checkpoint changed
(xl → b03_gdino seed42): static-area flip rate **1.50% → 1.00%**, oscillation share
83.7% → 78.6%, fixture boundary-band flips (in-static) 0.094 → 0.065. Person detection
found the crowd it used to miss — 3.0 → 6.8 boxes/frame with track p90 14 → 42 frames
and single-frame tracks 42% → 33% — while coco_person mAP also rose 0.210 → 0.218, so
this is not threshold-shopping. One apparent regression dissolved on inspection:
boxed_stock boxes/frame collapsed 5.5 → 0.12 *at the fixed 0.30 render threshold*, but
its mAP is unchanged (0.1174 → 0.1179) — a score-calibration shift, not a capability
loss, and the argument for per-class working thresholds at the serving layer.
site_person val lands at mAP50 0.787 (agreement with GDINO, per the standing caveat).

**The reach_to_shelf mechanism is fixed and the fix found the person the old one
missed (`2c2188d`, `9e6818c`).** The old rule read the terrain class *at* the wrist
pixel — which is usually the wrist's own hand, labelled `person` — so it fired on
occlusion and pose error rather than on touch. The new rule reads what the hand is
*over*: fixture share of a 25 px disc around the wrist, person-pixels excluded,
contact at ≥0.35 (both defaults derived from measurements recorded in the docstring;
the gap 0.41–0.55 between non-contact spikes and the weakest real contact is where the
threshold sits). Re-run on the pose-pilot data: the child (t4) still fires at the same
frames but now *because* the hand is over the display (peak share 0.93–1.0), and the
deep-bowing customer (t2) — wrist confidently on the display, zero events under the old
rule because 98–100% of the wrist pixel was `person` — now fires across every reaching
window, each verified by eye on terrain overlays. cam11 stays at zero under a full
parameter scan. The terrain map's pixel-space contract is now explicit (wrong size
raises, message names the fix), and the stale function name in the keypoints contract
is corrected. Still open from the pilot: `fall_angle_deg`'s viewpoint semantics.

**The stabilisation ablation (runs/stable01, scripts/stable_infer.py at d8542fd), same
ruler as everything above.** Per-pixel logit EMA (alpha 0.35) is the unconditional win:
static flips 1.50%→0.67%, phantom-person flips −54%, every metric improves and nothing
regresses — it enters the main inference path. Track-based rendering replaces raw
detections for display and events: 949 tracks → 145, one-frame boxes 40%→11.7%, median
life 2→18 frames (count people at the birth threshold 0.35, not the 0.20 sustain).
Static composite stays a demo/serving lever for now: its unique win is the fixture
boundary band (8.9%→2.2%) but EMA already buys most of the flip reduction, and it costs
per-camera/slot plate artefacts plus a dirty-plate check — this clip's plate carries
8.6% person pixels from median-resident counter staff, and the design treats those
regions as never-trusted (worst case = baseline, immune to the old stabiliser's
self-sealing). Its real case is serving capacity (taken-over pixels need no compute or
transfer), to be measured on the TensorRT path. Full stack: static flips 0.67%,
oscillation share 81.6%→65.1%, verified by eye across moving-person frames (no smearing;
EMA ghost 1–2 frames; boxes coast ≤5). Caveat: one camera, one slot, one clip.

**The three-seed table (rule 4 satisfied), final epochs, aligned metrics:**

| metric | seed42 | seed7 | seed13 | xl baseline |
|---|---:|---:|---:|---:|
| site_person mAP50 | 0.787 | 0.803 | 0.795 | — (no site person source) |
| site_person mAP | 0.436 | 0.446 | 0.438 | — |
| coco_person mAP | 0.2176 | 0.2158 | 0.2172 | 0.2103 |
| site_boxes mAP | 0.118 | 0.118 | 0.130 | 0.117 |
| site_boxes03 mAP | 0.121 | 0.114 | 0.137 | 0.122 |
| terrain mIoU (site_seg03) | 0.549 | 0.591 | 0.567 | 0.565 |

Reading: the new in-domain person capability is **stable across seeds** (mAP50 band
0.016 wide); coco_person — the only human-labelled person metric — is above the
baseline in **all three seeds** (+0.006..+0.007, small but unanimous); merchandise
detection and segmentation sit inside their own seed bands, so the person supervision
was added at **no measurable cost to any other head**. site_person numbers are
agreement with GDINO per the standing caveat. The teacher swap decided on this
morning's frame-identical measurement is now confirmed end-to-end: teacher → fleet
batch → retrain → three seeds → stability ruler, all on instruments built today.

**The 96-stream target is met, and the risk sits exactly where the morning's
arithmetic put it (runs/bench_pro6000, GPU verified idle).** fp16 TensorRT engines of
the shipped graph at 640x1120: batch 16 delivers **3,272 frames/s compute** against the
1,440 target (2.27x margin; batch 1 alone is 1,156 — batching is mandatory; 32/64 add
nothing). With synchronous H2D copies the same engine drops to **1,553 f/s** — barely
over target — confirming the data path, not the model, as the engineering frontier.
Three known recoveries, none exotic: uint8 input binding (4x less to copy), pinned
memory with dual-stream overlap, and the static-composite's taken-over pixels needing
no transfer at all. Engine builds are ~30 s each, so rebuild-per-release is trivial.

**The fleet calibration sweep (runs/onboard01): plug-in-to-orientation is 83%
automatic, and the failures are all mount types, not math.** 19/23 cameras get
orientation from one plate inference (regression check: Taichung-cam01 reproduces
calib01 bit-for-bit); 9 get scale automatically from person-height (median 0.665, one
self-contradicting outlier flagged); 14 wait on a visual reference — mostly person-box
famine at the 0.5 gate, not geometry. The four failures diagnose cleanly: one sideways
~90° mount (Kaohsiung-cam05 — the long-suspected one, now measured), one near-nadir
desk view, one glass storefront (depth through glass is not trustable), one camera with
no plate at all. First fleet-wide roll census: 4 tilted mounts (−8.3°..−12.9°) plus the
sideways one — the VLM pilot's "at least three" is now "four plus one". SOP gaps named
for the install checklist: a mount-type pre-check (sideways is auto-fixable by
rotation), one tile-grid shot per store per camera model to pin vfov/k1, and reading a
door frame's two endpoints as an install-day action, which zeroes the 14 unmeasured.
