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
