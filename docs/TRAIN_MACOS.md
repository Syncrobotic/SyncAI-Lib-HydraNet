# Training on an Apple Silicon Mac with uv

For local development and smoke testing; real training still belongs on a CUDA machine
(AMP is CUDA-only). The numbers below were measured on an M4 Pro / 48 GB unified memory /
PyTorch 2.13.

## 1. Setting up the environment

`uv` downloads a suitable Python itself, so there is no need for pyenv or conda first.

```bash
uv sync --group dev --extra export   # create .venv and install everything (per uv.lock)
```

- `uv sync` restores exactly the dependency versions in `uv.lock` and installs this project
  in editable mode.
- `--group dev` is a PEP 735 dependency group (pytest / ruff / pre-commit) and does not end
  up in the published wheel.

From then on every command has two forms; pick either:

```bash
uv run hydranet-train ...              # no need to activate the venv
# or
source .venv/bin/activate              # then use the command names directly
```

Verifying the environment:

```bash
uv run pytest -q                                              # 1,170 passed, 1+ skipped
uv run python -c "import torch; print(torch.backends.mps.is_available())"   # True
```

## 2. Device selection

`pick_device()` in `src/syncai_hydranet/utils/device.py` picks **CUDA → MPS → CPU** in that
order, and every CLI command shares it.
On a Mac it goes to MPS automatically, and the first line of the training log prints
`device=mps`.

To force a device:

```bash
uv run hydranet-train --config ... --set device=cpu
```

Two things switch themselves off on MPS, both decided in `device.py`, so the config needs no
manual editing:

| Item | State | Reason |
|---|---|---|
| AMP (`train.amp`) | Disabled automatically | `GradScaler` / `autocast` are only fully supported on CUDA |
| DataLoader `pin_memory` | Disabled automatically | MPS does not support pinned memory yet; setting True only produces warnings |

Losing AMP costs little on a Mac: 48 GB of unified memory is enough for batch 16 in FP32.

## 3. Measured performance

RegNetX-800MF + BiFPN×2, all three heads active, including backward and the AdamW step:

| Device | Resolution | batch | ms/step | img/s | Peak memory |
|---|---|---|---|---|---|
| CPU | 384×512 | 4 | 705 | 5.7 | — |
| MPS | 384×512 | 4 | 97 | 41.2 | — |
| MPS | 384×512 | 8 | 186 | 43.0 | 3.4 GB |
| MPS | 512×640 | 8 | 300 | 26.7 | 5.6 GB |
| MPS | 512×640 | 16 | 628 | 25.5 | 10.8 GB |

**MPS is about 7.3× faster than CPU**, so do check that the log prints `device=mps`.
Going from batch 8 to 16 gains almost no throughput (the GPU is already saturated) while
doubling memory, so locally it is worth stopping at 8.

## 4. Suggested local config overrides

```bash
uv run hydranet-train --config configs/hydranet_regnet800mf.yaml --set \
  "data.input_size=[384,512]" \
  "train.batch_size=8" \
  "data.workers=3"
```

### `data.workers` should come down

`MultiTaskLoader` builds **one DataLoader per dataset**, with `persistent_workers` on
whenever `workers > 0`. The default `workers: 8` with three datasets means **24 worker
processes resident at once**, which on a 14-core laptop over-subscribes the machine and ends
up slowing data delivery down. Locally, `3`–`4` is the better range.

### COCO's `sample_ratio` should come down

Steps per epoch is `Σ len(loader_i) × ratio_i`. COCO train2017 holds about 117,000 images,
so at `ratio: 1.0` it alone contributes ~14,600 steps: 80 minutes for one epoch, and 80
hours for 60 epochs.

Drop COCO's `sample_ratio` to `0.1` and each epoch takes a random 10%. Because
`shuffle=True` and the iterator is rebuilt every epoch, each epoch sees a *different* 10%,
so over a full run the whole dataset is still covered.

Note that `set_path` in `config.py` does **not support list indices** —
`--set data.datasets[2].sample_ratio=0.1` raises no error but silently creates a useless key
literally named `datasets[2]`. Two workable approaches:

```bash
# Approach A: copy the config (recommended, the settings persist)
cp configs/hydranet_regnet800mf.yaml configs/local_mac.yaml
# edit local_mac.yaml and change sample_ratio in the coco entry to 0.1

# Approach B: override the whole list at once (--set values go through yaml.safe_load)
--set "data.datasets=[{name: rugd, type: seg_folder, root: datasets/RUGD, \
  split_train: train, split_val: val, supervises: [traversability, terrain], sample_ratio: 1.0}, \
  {name: coco, type: coco, root: datasets/coco, split_train: train2017, \
  split_val: val2017, supervises: [detection], sample_ratio: 0.1}]"
```

After that one epoch takes about 14 minutes and 60 epochs about 14 hours — an overnight run.

### Segmentation-only is a legitimate setup

Trim `data.datasets` down to just RUGD and the detection head is still built, merely left
unsupervised.
RUGD + RELLIS together (about 13,000 images) run at roughly 8 minutes per epoch at 512×640.

## 5. EMA is now safe on short runs (historical issue)

Validation uses the EMA weights, and the EMA starts from the model's **initial random
weights**. The old fixed decay left `ema_decay^n` of that initialisation behind after n
steps, so short runs were dominated by random weights:

| Steps | Random init remaining | Validation score |
|---|---|---|
| 160 steps @ 0.995 (old) | 45% | traversability mIoU 0.16 |
| The same weights, non-EMA | 0% | traversability mIoU **0.95** |

The decay now ramps with the update count, `decay * (1 - exp(-updates / ema_warmup_steps))`
(the standard YOLOv5 / timm approach): the first few steps copy the model almost exactly,
and the smoothing only strengthens as the average accumulates history.
An 18-step smoke run now scores the same with EMA on or off, so **there is no longer any
need to disable it for short runs**.

`ema_warmup_steps` defaults to 2000. If a run really is too short to finish even the ramp,
`Trainer` still prints a warning with the residual fraction; that is the point to consider:

```bash
--set train.ema=false
```

## 6. Checking whether training is data-starved

The log prints img/s every `log_interval` steps. Compare it against the table above:

- Close to the table → the GPU is the bottleneck, which is normal.
- Clearly below the table → data delivery is not keeping up. RELLIS-3D is 1920×1200 JPEG,
  and decoding it is CPU-heavy. Try raising `data.workers` by 1–2 first; if that does not
  help, downsample the dataset offline.

TensorBoard:

```bash
uv run tensorboard --logdir runs/
```

## 7. ONNX export

`hydranet-export-onnx` does not go through device selection — it exports on CPU, so a Mac
can run it directly:

```bash
uv run hydranet-export-onnx --config configs/hydranet_regnet800mf.yaml \
    --checkpoint runs/.../best.pt --output hydranet.onnx
```

Converting to a TensorRT engine with `trtexec` afterwards has to happen on the Jetson; see
[DEPLOY_JETSON.md](DEPLOY_JETSON.md).
