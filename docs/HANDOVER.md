# Handover: moving from Apple Silicon to a CUDA machine

The development machine (M4 Pro / MPS) is only good for development and smoke testing.
This is the runbook for moving training onto a CUDA workstation (e.g. an RTX PRO 6000
Blackwell), plus the known state of things at handover time.

> **Status: the move is done.** Training now runs on an RTX PRO 6000 Blackwell Max-Q
> (96 GB, sm_120, driver 595.84 / CUDA 13.2) on `torch 2.13.0+cu130`, and the section 4
> baseline has been reproduced there. The sections below still read as a runbook because
> they are the procedure to repeat on the next machine — but CUDA, not MPS, is the
> reference now.

For the macOS setup see [TRAIN_MACOS.md](TRAIN_MACOS.md); for Jetson deployment see
[DEPLOY_JETSON.md](DEPLOY_JETSON.md).

## 1. The code needs no changes for a different GPU

`pick_device()` in `utils/device.py` picks **CUDA → MPS → CPU** in that order, and AMP and
the DataLoader's `pin_memory` are enabled off the same decision. On a CUDA machine it is
the same commands with no extra flags.

## 2. Four things git does not bring with it

`.gitignore` excludes data, outputs and large media, so none of this exists after a clone:

| Path | Size (at handover) | What to do |
|---|---|---|
| `datasets/` | 1.0 GB | Re-download on the box; do not upload from the laptop |
| `runs/<experiment>/*.pt` | 122 MB / checkpoint | See below, usually not worth transferring |
| `assets/*.mp4` | 19 MB | **Must be transferred**, cannot be regenerated |
| COCO | Not downloaded | Fetch on the box when the detection head is needed |

`assets/VID_20260813_145154.mp4` is self-recorded indoor site footage and the one asset no
command can rebuild. Copy it across if you want inference comparisons.

**Checkpoints usually do not need to travel.** 20 epochs takes 55 minutes on the M4 Pro;
the same training is roughly 5–10 minutes on a Pro 6000 with bf16 and a larger batch.
Moving 122 MB and dealing with old-format compatibility to save that is a bad trade —
retraining from the ImageNet initialisation is cleaner, and only a fresh run carries a
`meta.json`, which is what makes it visible to `hydranet-report`.

## 3. Starting up on the CUDA machine

```bash
git clone -b dev https://github.com/Syncrobotic/SyncAI-Lib-HydraNet.git
cd SyncAI-Lib-HydraNet
uv sync --group dev --extra export

# the first gate
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`uv.lock` pins the **CUDA 13** build of torch (`nvidia-cudnn-cu13` and `nvidia-nccl-cu13`
are in the dependency tree). On too old a driver **`uv sync` succeeds and the import line
above is what fails** — run it first rather than discovering this when training starts.
Blackwell (sm_120) itself has been supported since CUDA 12.8, so the version is not a problem.

Then run the tests to confirm the environment is complete:

```bash
uv run pytest -q
```

The data (not in the repo):

```bash
curl -LO https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip -q ADEChallengeData2016.zip -d datasets/
uv run hydranet-prepare-ade20k --src datasets/ADEChallengeData2016 --dst datasets/ADE20K
```

Note the casing in that URL: the path segment is `ADEchallenge` with a lowercase c while
the filename is `ADEChallengeData2016.zip` with an uppercase C. That inconsistency is
upstream's, and the uppercase-path variant returns 404.

`hydranet-prepare-ade20k` filters on **annotation content** (enough floor, little sky and
vegetation) rather than a scene-category whitelist, selecting the ground-level indoor
viewpoint a robot sees. Both splits are filtered, and the counts are deterministic — a
re-run on the same source reproduces them exactly:

```
training   -> train:  kept 5998/20210 (29.7%)
validation -> val:    kept  614/2000  (30.7%)
```

If your numbers differ, the source archive differs; check that before anything else.

## 4. The baseline

**The CUDA machine is now the reference.** The MPS column below is kept only as the origin
the move was checked against — it is not the number to beat any more.

Both columns are ADE20K indoor subset only, at epoch 20, validated on EMA weights. They are
*not* the same training: the CUDA run uses the section 6 settings (512×640, batch 48,
bf16), the MPS run used 384×512 at batch 8. Equal epoch counts on the same data are what
makes them comparable at all.

| Metric | **CUDA @ epoch 19** | MPS @ epoch 20 (origin) |
|---|---|---|
| traversability mIoU | **0.6609** | 0.665 |
| ├ blocked | 0.9506 | 0.954 |
| ├ caution | 0.1973 | 0.200 |
| └ go | 0.8349 | 0.843 |
| terrain mIoU | 0.5444 | 0.569 |
| ├ glass | 0.4686 | 0.505 |
| └ stairs | 0.1940 | — |

The traversability head reproduces the origin almost exactly — every one of its classes is
within 0.01. The terrain head sits slightly lower at this epoch (0.5444 against 0.569, glass
0.4686 against 0.505) but was still climbing when the snapshot was taken, and had reached
0.556 and 0.503 respectively a few epochs later. That is what the move had to demonstrate:
**nothing broke in the migration.**

Note in particular that `caution` reproduces at 0.1973 against 0.200 — the data gap in
section 5 is a property of the dataset, not of the hardware, and changing GPU did nothing for
it, exactly as predicted.

Two cautions about reading this table:

- **The run is still going.** These are epoch-19/20 figures from a 60-epoch run, quoted at
  epoch 20 purely so they line up with the MPS column. The final numbers will be higher and
  should replace this table when the run completes.
- **Intermediate epochs are noisy.** The same run measured `caution` at 0.158 on epoch 14 and
  0.197 on epoch 19. Do not conclude anything from a single mid-run validation.

Measured wall-clock on the CUDA box: **36 s per epoch** including validation, so 20 epochs in
12 minutes and a full 60-epoch run in about 36 minutes — against 55 minutes for 20 epochs on
MPS, at a *lower* resolution.

### Throughput and where the ceiling is

| | MPS (M4 Pro) | CUDA (Pro 6000) |
|---|---|---|
| img/s @ 512×640 | 26.7 | **275** (median; peak 284) |
| GPU utilisation | — | 93% median |
| Board power | — | 277 W median, 299 W peak against a 300 W cap |
| VRAM at batch 48 | — | 20.7 GB of 96 GB |
| CPU load | — | 4.5–7.6 of 24 cores |

At matched resolution that is a **10× throughput gain**. The utilisation and power figures
say the GPU is the bottleneck and the card is running into its power cap, which is the
correct place to be: the data pipeline has headroom, so tuning `workers` further gains
nothing. The 20.7 GB measurement also confirms the batch-48 estimate in section 6 — there is
room for a larger batch, if a larger batch were useful.

## 5. Known issues, in priority order

### 1. `caution` is stuck at 0.200, and more epochs will not help

This is a data gap, not undertraining, and the migration proved it: the CUDA run reproduced
`caution` at 0.1973 against the MPS run's 0.200, on ten times the throughput and a different
vendor's silicon. Hardware was never the constraint.

Indoors, traversability "caution" corresponds to four terrain classes (see
`data/label_maps_indoor.py`), but ADE20K covers only `stairs` among them, at **0.3%** of all
pixels. Of `floor_metal` (grating), `wet_slippery` (standing water or a freshly mopped floor)
and `threshold_ramp` (door sills and level changes) it contains **not a single example**.

Those three decide whether the robot slips or gets stuck, and only in-house annotation can
fill them in. Annotate with `label_map: indoor_native`, where the annotation ids already are
the indoor 12 classes.

### 2. The detection head is entirely unsupervised

Only segmentation datasets are wired up right now, so the detection head sits at its initial
weights and inference produces no boxes at all. The config check warns about this explicitly
at startup:

```
config: head 'detection' is not supervised by any dataset:
        it will be built, exported and left at its initial weights
```

Getting boxes means downloading COCO and putting `sample_ratio` back to `1.0` (the indoor
config currently has `0.2`, which is there to keep a laptop usable).

### 3. `channels_last` is not implemented

`train.tf32` and `train.amp_dtype=bfloat16` both exist already, but convnets on tensor cores
generally have some headroom left in the NHWC memory format. This is the only CUDA-specific
optimisation still missing from `src/`.

## 6. Suggested starting settings

For 96 GB of VRAM:

```bash
uv run hydranet-train --config configs/hydranet_indoor.yaml --set \
    train.batch_size=48 \
    train.lr=6.0e-4 \
    train.epochs=60 \
    train.warmup_iters=2000 \
    train.amp_dtype=bfloat16 \
    train.tf32=true \
    train.cudnn_benchmark=true \
    data.workers=16 \
    'data.input_size=[512,640]'
```

The reasoning for each:

- **batch 48** — fp32 at 512×640 with batch 8 peaks at 5.6 GB on the M4 Pro. On CUDA, AMP
  roughly halves activation memory, putting 48 at about 20 GB, comfortable within 96 GB.
  Measured: **20.7 GB**, so the estimate holds. Going higher gives diminishing throughput,
  and a large batch is not necessarily better for segmentation.
- **lr 6e-4** — linear scaling from batch 8 to 48 would give 1.2e-3, which is too aggressive
  for AdamW. This uses sqrt scaling (`2e-4 × √6 ≈ 4.9e-4`) rounded up a little, with warmup
  extended to 2000 steps to match.
- **bfloat16 rather than float16** — bf16 keeps fp32's exponent range, needs no loss scaler,
  and is native on Blackwell. fp16 overflows more easily under a multi-task loss where Dice
  and Focal differ substantially in magnitude.
- **workers 16** — a fair share of the throughput on the M4 Pro was JPEG decoding. On the
  server that bottleneck is gone: measured GPU utilisation is 93% at a CPU load of 4.5–7.6
  across 24 cores, so **16 is already more than enough and raising it only costs memory**.
  Note that `data.workers` is **per dataset**, not a global total: `MultiTaskLoader` builds
  one DataLoader per entry in `data.datasets`, so 16 with both ADE20K and COCO present means
  32 worker processes. With COCO absent it is 16 — which is the case measured above, so
  re-check once COCO is added. Always set this against the img/s in the log rather than
  against the core count.
- **512×640** — the laptop drops to 384×512 for speed; on CUDA there is no reason not to use
  the config default.

## 7. The first thing to do after switching cards

**Reproduce the section 4 baseline with the same settings before changing anything.** This
has been done once already for the Pro 6000, and it is what section 4's two columns record;
repeat it on the next card rather than trusting that a working setup transfers. Once it is
through:

```bash
uv run hydranet-report runs/*          # compare runs, reading meta.json and metrics.jsonl
uv run tensorboard --logdir runs/
```

In TensorBoard, look first at the `val_pred/*` three-column comparisons and the
`IoU/<head>/<class>` per-class curves — `glass` and `stairs` occupy so few pixels that mIoU
alone lets floor and wall bury them completely.
