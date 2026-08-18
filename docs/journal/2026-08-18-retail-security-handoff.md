# 2026-08-18 — retail + security: what was built, what is running, and what to read first

Written because a session ran out of context, not because the work finished. Everything
below is on disk; nothing is only in a transcript.

**Revised 19:20 by the session that took the line over.** Section 3's restart cost and
remaining time were wrong, and **two of section 4's three headline numbers are withdrawn**
— re-derived from `metrics.jsonl` rather than re-run, so the corrections cost no GPU. Read
section 4's "Corrected 19:20" block before quoting anything from this document.

The line: **fixed store CCTV, retail analytics and security on one camera.** Not the robot
platform — that is [ARCHITECTURE_REVIEW.md](../ARCHITECTURE_REVIEW.md), and a separate
session spent today building its depth head.

---

## 1. Read these first

* [RETAIL_SECURITY.md](../RETAIL_SECURITY.md) — the design: three output layers, the shared
  detection vocabulary, behaviour split by instrument, the dataset plan.
* [ARCHITECTURE_DIRECTION.md](../ARCHITECTURE_DIRECTION.md) §4 — measurement rules,
  including **rule 5 added today**: a run whose length a metric decides is not comparable
  to one of the same config with a different selecting metric.
* `datasets/studioa_clips/cameras.json` — **new**, and it gates everything: which of the
  48 cameras are selling floor (23), back of house (19) and dead (6).

## 2. What the user decided

Asked directly, answered directly. These are settled unless the user reopens them.

| # | question | decision |
|---|---|---|
| 1 | split `fixture` into display table vs wall shelving? | **no, one class for now** |
| 2 | merchandise as count or coverage? | **coverage**, via SAM 3, no human labour |
| 3 | age granularity? | **whatever the dataset supports** (PA-100K: two binary questions) |
| 4 | annotate site `person` boxes? | **yes**, via SAM 3, no human labour |
| 5 | build pose or attributes first? | **attributes (crop encoder) first** |

## 3. What is running right now

`systemctl --user status hydranet-b03-seeds.service` — six runs, interleaved by seed so a
complete pair exists after two rather than after four:

    b03_cw s42 (done) -> b03_cw_hires s42 -> b03_cw s7 -> b03_cw_hires s7 -> s13 -> s13

Log: `/home/paul/hydranet-overnight/b03_seeds.log`.

**Corrected 19:20 — a restart does not resume, it re-runs.** This section said a kill costs
at most one epoch. It does not. The driver has no completion marker, so a restart re-enters
the loop at `S=42, CFG=b03_cw` and runs a *finished* run again from epoch 1, overwriting its
`best.pt`, `metrics.jsonl` and `train.log`. That is what happened at 18:07: `b03_cw` had
exited 0 at 17:57 and was re-run in full 18:08–18:48, and `b03_cw_hires` lost 12 epochs.
`--resume last.pt` was passed and did not take effect. So interrupting now costs the current
run's progress **plus every completed run of this queue**, not one epoch.

`Restart=on-failure` with `StartLimitBurst=3` per hour: after three failures inside an hour
systemd stops trying and the unit needs `systemctl --user reset-failed`.

**Measured rates, and the real remaining time.** 35 s/epoch for `b03_cw`, 66 s/epoch for
`b03_cw_hires`, 60 epochs each. At 19:20, with hires s42 at E22:

    b03_cw_hires s42   E22/60 running    ~42 min
    b03_cw       s7    60 ep             ~40 min
    b03_cw_hires s7    60 ep             ~66 min
    b03_cw       s13   60 ep             ~40 min
    b03_cw_hires s13   60 ep             ~66 min
                                         ~4.2 h, finishing about 23:25

The "roughly 2.5 h" this section carried was an underestimate.

**And the queue does not produce the control it is compared against.** It runs `b03_cw` and
`b03_cw_hires` at three seeds each; `b03` exists at seed 42 only.
`hydranet_retail_security_b03_seed7.yaml` and `_seed13.yaml` are on disk and in no queue.
However this finishes, the class-weight question stays three-against-one and section 4's
rule 4 is not satisfied for it.

The other session's `hydranet-nyu-depth.service` shares the GPU at Nice=5 and ~4 GB. It is
not in the way.

## 4. What was measured

**The shared detection vocabulary works and cannot be credited for its own gain.** Three
paired seeds, `detection_mAP/site_boxes`, same 36 val images:

    seed 42   surfaces 0.0704 -> security 0.1061   +0.0357
    seed  7            0.0901 ->          0.1016   +0.0115
    seed 13            0.0659 ->          0.0933   +0.0274
    paired mean +0.0248, 95% CI -0.0057 .. +0.0554

All three positive, interval includes zero. And **three things changed together** — site
box ratio 1.0 -> 3.0, COCO person/bag added, the vocabulary itself — so the supportable
claim is the negative one: putting `person` and merchandise in one head cost the
merchandise nothing, and it made a combination possible that previously could not run at
all (two detection sources both emit label 0).

### Corrected 19:20 — the annotation and class-weight numbers are withdrawn

Every number in this section was re-derived from `metrics.jsonl` when the line was taken
over. **The detection-vocabulary result above reproduces exactly** — surfaces 0.0704 /
0.0901 / 0.0659 is `detection_mAP` on runs whose only detection val set is `site_boxes`, so
it is the same quantity the security runs report per-set, and both sides select on a
detection metric. It stands as written, caveat included.

The two claims that followed it do not. Neither survives, and they failed the same way,
which is the way rule 5 predicts.

**368 frames of new annotation: the +0.0153 could not be reproduced, and the effect is
inside seed noise.** The claim was seed 42, `terrain_mIoU/site_seg` 0.6511 -> 0.6664. The
0.6664 is right — b03's argmax on that metric, at E16. The **0.6511 is in no seed-42 file**;
it is the final-epoch value of `hydranet_retail_security_seed13`, which is a different seed
and not b03's paired control. Applying one procedure to both sides (argmax `site_seg`):

    control seed 42   0.6623 @E34 (48 ep)     b03 seed 42   0.6664 @E16 (50 ep)   +0.0041
    control seed  7   0.7013 @E24 (42 ep)
    control seed 13   0.6857 @E12 (34 ep)
    control family    mean 0.6831, range 0.0390, sd 0.0196

**The control family's own spread is 9.5x the effect**, and b03 scores below two of the
three controls. Worse, the two sides have **no common valid selection set**: `site_seg03` is
batch03's val split, so the control never saw it and its argmax there lands at E1 with
`site_seg` 0.2320. Argmax on the reported set is selection on the test set, which inflates
both sides and is the only procedure available. So the annotation A/B is **not scorable from
these runs at all**, in either direction — not adverse, undecided. ADE20K unchanged at
-0.0005 stands; it was never the contested half.

**Class weights: the +0.173 was the selection metric, not the loss.** The claim was "the
highest-yield change of the day, one config line, no new data". `site_confusion.py` compares
two `best.pt` files, and these two were chosen by **different metrics**: b03's is E41,
selected on `detection_mAP/site_boxes`; `b03_cw`'s is E30, selected on
`terrain_mIoU/site_seg03`. And the runs differ in **five config leaves, not one** —
`class_weights`, `early_stop_patience` 10 -> 0, `primary_metric`, `detection_val_interval`
1 -> 5, and `detection.classes` (which training never reads). b03 early-stopped at E50,
`b03_cw` ran 60. That is precisely the comparison **rule 5 was written today to forbid**,
and it was applied to everything except this document's own headline.

Both runs logged `terrain_mIoU/site_seg03` whether or not they selected on it, so the
selection can be aligned after the fact — each run's own argmax on it, reported on
`site_seg`:

    run       epoch   site_seg   floor   wall   column   fixture   person
    b03        E16     0.6664    0.826  0.594   0.352     0.767     0.793
    b03_cw     E30     0.6636    0.859  0.595   0.267     0.742     0.855

Truncating `b03_cw` to E50 gives the same rows. The weights move site mIoU **-0.0028** and
`column` **-0.085**.

Read that lightly in both directions. `column` is **1.53% of val pixels over 36 images**
(rule 1) — a quantity that moves on noise — and it is one seed a side. And b03's E16 exists
only as a row in `metrics.jsonl`: no checkpoint on disk holds those weights, so trap 1 bit
a second time, in the analysis rather than in the training.

**The honest state: whether class weights help `column` is unmeasured.** Settling it needs a
control that changes only `class_weights` — b03 re-run with `early_stop_patience: 0` and
`primary_metric: terrain_mIoU/site_seg03` — at three seeds, about 2 h of GPU. The weights
themselves are `1/sqrt(share)` from batch03's measured shares, a standard heuristic and not
a fitted quantity; whether `1/sqrt` is the right exponent was never the first question.

**The lesson survives, sharpened.** Reading the confusion matrix was still right and still
cheap: `wall` really was taking 86.2% of `column`, and reading it before buying 368 frames
would have cost ten minutes. What did not survive is the number attached to the fix.
*Look at what the model says before buying more labels — and then measure the fix against a
run that differs only by the fix.* A day that produced two headline numbers produced neither
a valid control nor a common selection set for either of them.

**Crop encoder** (`runs/crop_encoder01`, 8 epochs on PA-100K): Female recall 0.839 /
precision 0.856 on 36,492 training crops; AgeLess18 0.639 / 0.919 on 4,152; **AgeOver60
0.119 / 0.357 on 1,127, and it degraded as training ran** — a 1.41% attribute once the
loss can afford to say no. Unasked-for result: the embedding scores **mAP 0.0543 / rank-1
0.1689** on Market-1501's 3,368-query protocol against a 0.0318 ImageNet floor, with no
identity label anywhere. A floor cleared, not association solved.

**Site behaviour, 192 clips / 48 cameras / 188,523 person boxes:** 48 `fall_candidate`
spans, 23 touching the frame edge, the rest on near-nadir or rotated cameras. **None is a
posture.** On this corpus the proxy measures camera mounting.

## 5. Artefacts

    datasets/retail_objects_batch03/     368 SAM 3 masks, 6 classes, 91/88/80% labelled
                                          (train/val/test), 23 selling-floor cameras,
                                          split inherited from batch02 so no camera crosses
    datasets/retail_person_batch01/      146 site person boxes, daylight-gated
    datasets/studioa_clips_0813/         84 clips, 2026-08-13, 42 cameras, two local times
    datasets/studioa_clips/cameras.json  camera roles
    runs/coverage01/                     shelf coverage: Tao-Hsin-cam02 35.0%,
                                          Taichung-cam04 30.9%, Kaohsiung-cam08 25.6%
    runs/fall_mining01/                  the 48 candidate spans with dumped frames
    assets/studioa_b03_cw_90s.mp4        90 s, 720p, one checkpoint for the whole stack

New scripts: `site_confusion.py`, `site_events.py`, `render_demo.py`,
`mine_fall_candidates.py`, `sam3_person_boxes.py`, `sam3_product_coverage.py`,
`train_attributes.py`.

## 6. Traps found today, each of which cost real time

Listed because every one will recur.

* **`best.pt` is selected by `primary_metric`, and that decides what survives.** b03 saved
  E31 (detection best) while its segmentation peaked at E16 — 0.043 mIoU that existed in
  `metrics.jsonl` and in no checkpoint on disk.
* **Early stopping made runs incomparable.** See rule 5. Patience 10 lives in
  `hydranet_retail_products.yaml` and inherits down the whole retail chain.
* **A log shared by two runs is not evidence about either.** Killing a python child leaves
  the bash driver alive; it continued into the next split and interleaved its errors with
  the new run's. Kill the parent first, then the child, then confirm zero survivors.
* **`pkill -f <script>` matches your own shell.** It kills the caller and reports exit 144.
  Resolve the PID with `ps`, exclude zsh, kill by number.
* **zsh does not word-split unquoted variables.** `--cameras $CAMS` passed 42 names as one;
  the dry run said "1 cameras" and downloaded nothing. Use `${=CAMS}`.
* **`build_dataset` succeeding does not mean an image can be read.** The b03 queue died six
  times at epoch 1 because `datasets/retail_objects_batch03/<split>` symlinks were missing;
  the index built fine. Verification must include `ds[0]`.
* **`_base_` merges rather than replaces.** A four-channel head inherited a two-name
  `classes` list; training never reads it, so the only symptom was a renderer naming a
  person `boxed_stock`. `_check_detection_head_classes` now refuses it.
* **Analysis must not require today's validator.** `site_confusion.py` reads a run's config
  with `yaml.safe_load`, because a run from this morning cannot be expected to satisfy a
  check added this afternoon, and refusing to analyse it discards the only evidence.
* **SAM 3 re-encoded the same image 44 times a frame** — 60% of each forward is the vision
  encoder, which does not depend on the text. `vision_features` caches it: 9.4 s -> 5.6 s.
* **A systemd restart re-runs the runs that already finished.** `b03_seeds.sh` has no
  completion marker, so restarting re-enters the loop at seed 42 and trains a completed
  config again from epoch 1, overwriting its `best.pt` and `metrics.jsonl`. `--resume
  last.pt` is passed and does not prevent it. Cost on 2026-08-18: one finished 40-minute
  run plus 12 epochs of the next. A queue is not restart-safe because it says `--resume`.
* **The clips are not 30 fps.** `probe` reads `r_frame_rate` (nominal 30); counted content
  is **3.0–8.0 fps**. Anything reading `probe`'s fps on this footage inherits the error.

## 7. What to do next

0. **Re-measure what section 4 withdrew, before building on it.** Neither headline number
   survived re-derivation, and the class-weight question needs a control that differs only
   by `class_weights`: b03 at three seeds with `early_stop_patience: 0` and
   `primary_metric: terrain_mIoU/site_seg03`, ~2 h of GPU. The annotation question needs a
   selection set both sides can see, which no run currently provides. Left undone
   deliberately — it costs GPU the running queue is using, and it is the user's call.
1. **Finish the queue and report three paired seeds** for b03_cw and b03_cw_hires. One seed
   is suggestive; the family's between-seed spread is 0.0173.
2. **Per-track attribute voting is built and unwired into anything but the renderer.**
   `analytics/track_attributes.py`, tested. Gender flips on 16.2% of consecutive frame
   pairs per frame; pooled, mean agreement 82.8% and a third of tracks below 80%.
3. **Site person boxes are annotated but in no config.** `datasets/retail_person_batch01`,
   146 boxes. Wiring them in is what makes `person` measurable on a store camera at all —
   today it is scored on COCO val and nowhere else.
4. **Product coverage masks exist and train nothing.** `runs/coverage01/masks` — a dense
   `product` class from them would move coverage from seconds of SAM 3 per frame to 6 ms.
5. **One labelled clip is still the blocker for everything temporal.** `idf1` and
   `id_switches` cannot run without ground-truth tracks, so no claim about association —
   including today's tracker fix — is verifiable.

## 8. Open questions for the user

* Three seeds per config, or stop at one when the effect is large? The queue is ordered so
  it can be cut after any pair.
* RAP v2 was never obtained. The measured alternative ranking is in RETAIL_SECURITY.md;
  PA-100K is the training source and PETA's GRID subset the only distribution-matched test
  set. Worth one licence email, blocks nothing.
* CrowdHuman for occluded person detection — not downloaded.
