# 2026-08-18 — retail + security: what was built, what is running, and what to read first

Written because a session ran out of context, not because the work finished. Everything
below is on disk; nothing is only in a transcript.

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

Log: `/home/paul/hydranet-overnight/b03_seeds.log`. Resumes from `last.pt` on restart, so
a kill costs at most one epoch. Roughly 2.5 h left at the time of writing.

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

**368 frames of new annotation bought +0.0153 site mIoU** (seed 42, `terrain_mIoU/site_seg`
0.6511 -> 0.6664), with ADE20K unchanged at -0.0005. One seed; s7 and s13 are queued.

**Class weights were the highest-yield change of the day.** One config line, no new data.
`site_confusion.py` on the same 36 images, seed 42:

    class     b03     b03_cw    delta      what changed
    column   0.094 -> 0.267    +0.173     86.2% of it was coming back as `wall`, now 66.0%
    wall     0.566 -> 0.595    +0.029     stops eating neighbours, so its own IoU rises
    person   0.830 -> 0.855    +0.024
    floor    0.847 -> 0.859    +0.012
    fixture  0.734 -> 0.742    +0.008

Nothing got worse. The weights are `1/sqrt(share)` from batch03's own measured shares —
a standard heuristic, not a fitted quantity; whether `1/sqrt` is the right exponent is
unmeasured.

**The order was wrong and that is the transferable lesson.** Data was added first and the
confusion matrix read after. Reading it first would have shown `wall` eating 86% of
`column` in ten minutes, and that is a one-line fix. *Look at what the model says before
buying more labels.*

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
* **The clips are not 30 fps.** `probe` reads `r_frame_rate` (nominal 30); counted content
  is **3.0–8.0 fps**. Anything reading `probe`'s fps on this footage inherits the error.

## 7. What to do next

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
