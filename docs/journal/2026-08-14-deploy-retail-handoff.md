# Session handoff: deployment, retail and tooling

> **Journal entry, 2026-08-14.** A record of one session, not maintained afterwards. The
> durable material is in the `docs/` files this points to; where the two disagree, believe
> those. Written by the session that owned deployment, annotation tooling, docs and the
> retail line — a second session owned training experiments and wrote its own handoff.

## Where the repo is

`dev`, **10 commits ahead of `origin/dev` and unpushed**. Not a merge problem: this
workstation rebooted at 13:26 and took the ssh-agent with it, so `git push` fails with
"correct access rights". Fix and push:

```bash
ssh-add ~/.ssh/id_ed25519 && git push origin dev
```

`stage` was promoted earlier today (PR #1, `b62ca2c`); `main` is untouched and every
acceptance gate in [RELEASE.md](../RELEASE.md) is still unmet, deliberately.

## What changed, and the one idea behind most of it

A nine-item architecture audit, then the retail line. The thread running through nearly
all of it: **the failures that cost this project time do not raise errors.** They produce
a plausible number. So most of the work was making those cases fail loudly.

| Area | Change |
|---|---|
| Tests | Smoke tests for the four CLIs, which had **0%** coverage while trainer internals had 216 lines of tests. 74% → 84%, CI floor 68 → 80 |
| Data | `sample_ratio` rounding to zero batches now raises — it silently left a head untrained while the export guard, which reads the config not the schedule, still passed the model |
| Data | `_index_pairs` keyed on bare filename stem, so `0000.png` in three session directories collided and **two thirds of a batch was dropped silently**. Now keyed by path |
| Lineage | Augmentation moved from hard-coded into `data.augment`, so it reaches `meta.json`. Defaults unchanged, pinned by test |
| Deploy | Normalisation folded into the ONNX graph; input renamed `image_rgb_255` so the contract travels into the engine, which keeps no ONNX metadata |
| Perf | Loss logging no longer syncs CUDA every step; val loaders built once; confusion matrix on GPU with a **bit-identical** equivalence test |
| Config | `_base_` inheritance; three configs 390 → 229 lines. Every merged config verified byte-identical to its predecessor before committing |
| Docs | `docs/journal/` split from evergreen docs; README 433 → 221 lines with the operational half moved to `USAGE.md` |
| Retail | `--scheme retail` for the annotation tool, a CCTV config, BEV renderers, an annotation batch builder |

## The retail line, which is where the open work is

**The deployment is fixed ceiling CCTV, not a robot camera.** Three site clips are in
`assets/archive_*.mp4` with `*_bev.mp4` renders beside them.

**What the model does and does not mark** (evidence: `assets/retail_prelabel_gap.jpg`, on
2026-08-17 deleted from the tree in `f9d4fcf` and still readable at `55ec787` — the entry
below is left as it was written, because a dated record is not refreshed):
wall-mounted shelving *is* found — a wall of shelves resembles ADE20K's `bookcase`, so the
bootstrap reaches it. The free-standing display podiums in the middle of the floor are
**not**: a waist-height island with three phones on it has no analogue in any public
dataset. No amount of public data supplies that. It is the first thing worth annotating,
and `datasets/retail_batch01` (90 frames, three sessions, model pre-labels, passing
`hydranet-annotation check --scheme retail`) is ready for it.

> **Later correction, 2026-08-17, appended rather than edited in.** "The free-standing
> display podiums are **not** found" is wrong, and re-running the same baseline on store
> CCTV shows why: they are found, and labelled `obstacle_furniture` while the wall shelving
> beside them is `display_fixture`. The bootstrap does not decline to answer, it answers
> with the wrong class — which is the worse of the two failures, because a gap is visible
> and a wrong label is not. The conclusion drawn here still holds; the evidence for it was
> misread. `docs/RETAIL_SCOPE.md` carries the measured version and the replacement figure.

### Running right now

```
runs/hydranet_retail_cctv    epoch ~9 of 12 when this was written
```

A fine-tune from `runs/hydranet_retail_base/best.pt` (E22, trav 0.6409) with two changes:
input `512x896` because a 16:9 frame in `512x640` wastes 30% of the canvas on padding, and
400 pseudo-labelled target-domain frames at `sample_ratio: 3.0`.

**When it finishes, the two things to produce, and the reason they are different in kind:**

1. **Cost side, which has numbers.** Same command, same 329 ADE20K test images:
   ```bash
   ADE='data.datasets=[{name: ade20k, type: seg_folder, root: datasets/ADE20K, split_train: train, split_val: val, split_test: test, label_map: ade20k_retail, supervises: [traversability, terrain], sample_ratio: 1.0}]'
   uv run hydranet-eval --config configs/hydranet_retail.yaml \
       --checkpoint runs/hydranet_retail_cctv/best.pt --split test --set "$ADE"
   ```
   The baseline to compare against, measured: **trav 0.6685 · terrain 0.5410 · glass 0.5461
   · display_fixture 0.3668**. COCO must be dropped from the config because it declares no
   `split_test` — so "report on test" is currently possible for segmentation and not for
   detection.

2. **Benefit side, which has no numbers and must not be given any.** The third clip
   (`archive_20260802-125220_*`) was deliberately kept out of training. Render it with both
   checkpoints and compare by eye; `scripts/bev_video.py` does the rendering.

**Do not quote this run's training-time mIoU.** Its val set includes 8 pseudo-labelled
frames, and the evaluator accumulates one confusion matrix per head across all val sets, so
that number is partly "how much does this model agree with its own predecessor" — which
always looks good. It already showed up: `display_fixture` read 0.4224 against the
baseline's 0.3612 and means nothing. The config says this where the val split is declared.

## Open threads, most time-sensitive first

1. **`runs/hydranet_fixed_coco10` — started, stopped at epoch 6 of 60, no conclusion.**
   The control that separates "the balancer reallocated capacity" from "dilution", at the
   point of maximum effect (COCO ratio 1.0). Paul needed the GPU. Resumable via
   `hydranet-overnight/resume_fixed.sh`; the three possible outcomes and their readings are
   registered in advance in [ARCHITECTURE_REVIEW.md](../ARCHITECTURE_REVIEW.md). **Epoch 6
   answers nothing** — the collapse it explains appears late.
2. **Annotation has not started.** The CVAT stack is pinned and scripted in
   `deploy/annotation/`, with backups, but **the admin account still does not exist**:
   `./cvat.sh admin <user>`. The VM was left RUNNING and idle at ~$59/month.
3. **`10.8.140.130` is powered down.** The best deployment target found: AGX Orin, TRT
   10.3, CUDA works unprivileged, RealSense D435I attached, and it runs a live robot stack
   so its camera must be *subscribed to*, never opened. `scripts/live_view_ros.py` serves
   predictions at :8090 and a metric 3D scene at :8090/3d.
4. **The BEV scene renderer has never run on a robot.** It works — `scripts/bev_demo.py`
   serves it against a synthetic frame with no hardware — but the depth path has only been
   exercised on synthetic data plus one live session before the Orin went down.

## Things measured today that should not be re-derived

- **COCO `sample_ratio` 0.1 is a floor, not an optimum.** The full curve on the test split
  is monotone: every segmentation metric falls after 0.1, and the exchange rate worsens as
  you climb (0.1→0.3 costs 0.102 of glass for 0.080 of mAP; 0.3→1.0 costs 0.375 for 0.056).
- **Indoor COCO classes are *harder* than the COCO average**, not easier: the 80-class model
  scores 0.3348 over 80 categories and 0.3246 over the 25 indoor ones. I predicted the
  opposite. The prediction was wrong and the measurement is what settled it — but the
  prediction is why the scoring was constrained at all, which is what stops a narrowed model
  reading 0.34 as progress when it is a regression.
- **On a polished lobby floor, 10.6% of walkable pixels return no depth.** Specular floor,
  IR reflected away. Depth-only range gating punches holes exactly where a robot is about to
  drive, which is why the homography is the more robust gate on retail floors — the inverse
  of the usual ordering.
- **ROS camera transport costs**: colour 41.6 MB/s, depth 27.8 MB/s, a frame is 145 ms old
  when a subscriber sees it, and a subscriber slower than the publisher silently accumulates
  lag (measured 535 ms while the loop itself took 90 ms).

## What I would do next

1. Push. Ten commits are only on this machine.
2. Finish the cost/benefit comparison above — it is 20 minutes and it decides whether
   pseudo-label adaptation is worth keeping.
3. **Start annotation.** Everything else is downstream of it, `datasets/retail_batch01` is
   staged, and the podiums cannot come from anywhere else.
4. Resume the balancer control when a GPU is free overnight.

## A note on how this session was run

Two other sessions worked the same repository concurrently. What made that work was
announcing file scope before touching anything and reporting corrections immediately —
including four occasions where a session publicly withdrew its own earlier claim. The
habit worth keeping is not "use the test split", which is knowledge; it is **checking a
number again after getting it**, which is what catches the next mistake nobody has named
yet.
