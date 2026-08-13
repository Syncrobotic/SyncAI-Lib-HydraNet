# SyncAI-Lib-HydraNet

Multi-head perception network for quadruped robots: **one forward pass, one frame, three
outputs — traversable surface, terrain class, object detection.**
The architecture follows the Tesla HydraNet idea: a shared backbone and neck carry almost
all of the compute, while the task heads stay tiny and mutually independent.

```mermaid
flowchart TB
    IMG(["image · 3 × 512 × 640"])

    subgraph TRUNK ["SHARED TRUNK — 7,022,342 params · 84.4%"]
        direction TB
        BB["<b>RegNetX-800MF backbone</b> · 6.59M · 79.2%<br/>stem → stage1 → stage2 → stage3 → stage4"]
        CFEAT["C2 · 64ch · 1/4 &nbsp;&nbsp; C3 · 128ch · 1/8<br/>C4 · 288ch · 1/16 &nbsp;&nbsp; C5 · 672ch · 1/32"]
        LAT["1×1 lateral → 96 ch · C2 dropped<br/>P6, P7 added by stride-2 conv from P5"]
        NECK["<b>BiFPN × 2</b> · 435K · 5.2%<br/>weighted top-down, then weighted bottom-up"]
        PYR["P3 1/8 &nbsp; P4 1/16 &nbsp; P5 1/32 &nbsp; P6 1/64 &nbsp; P7 1/128<br/>all 96 ch"]
        BB --> CFEAT --> LAT --> NECK --> PYR
    end

    subgraph HEADS ["TASK HEADS — 1,294,089 params · 15.6% · mutually independent"]
        direction LR
        TRAV["<b>Traversability</b><br/>Semantic-FPN<br/>277K · 3.3%"]
        TERR["<b>Terrain</b><br/>Semantic-FPN<br/>278K · 3.3%"]
        DET["<b>Detection</b><br/>FCOS, anchor-free<br/>739K · 8.9%"]
    end

    OUTT["<b>3 × 512 × 640</b><br/>blocked / caution / go"]
    OUTE["<b>12 × 512 × 640</b><br/>surface class"]
    OUTD["<b>per level P3–P7</b><br/>cls 80 · reg 4 (l,t,r,b) · ctr 1<br/>NMS runs outside the graph"]

    IMG --> BB
    PYR -->|"P3–P5"| TRAV
    PYR -->|"P3–P5"| TERR
    PYR -->|"P3–P7"| DET
    TRAV --> OUTT
    TERR --> OUTE
    DET --> OUTD

    classDef trunk fill:#2d6a9f,stroke:#1b4a72,color:#fff
    classDef head fill:#b4531f,stroke:#7d3915,color:#fff
    classDef out fill:#3d7a4a,stroke:#255030,color:#fff
    classDef io fill:#555,stroke:#333,color:#fff
    class BB,CFEAT,LAT,NECK,PYR trunk
    class TRAV,TERR,DET head
    class OUTT,OUTE,OUTD out
    class IMG io
```

Shapes above are for the default `input_size: [512, 640]`; the segmentation heads always
resize their logits back to the input resolution, so the two mask outputs match the image
whatever it is set to.

The parameter split bears the design out: **84.4% shared trunk, 15.6% for all three heads
combined** (8,316,434 total). A fourth head costs roughly 3–9% more parameters and reuses
the 84% already paid for.

- **Backbone**: torchvision RegNetX, swappable to ResNet18/34/50 with one config key
- **Neck**: BiFPN (fast-normalized fusion), with plain FPN as the alternative
- **Head 1, traversability**: Semantic-FPN style segmentation, 3 classes
- **Head 2, terrain**: the same segmentation head, 12 classes
- **Head 3, detection**: FCOS, anchor-free (focal + GIoU + centerness)
- **Multi-task balancing**: Kendall learned uncertainty weighting (or fixed weights)
- **Partial supervision across datasets**: each step draws a batch from a single dataset and
  backpropagates only the losses of the heads that dataset supervises
- **One annotation, two heads**: terrain labels generate traversability labels via a policy table
- **TensorRT friendly**: the forward graph is only Conv/BN/ReLU/Resize/MaxPool/Exp/Mul; NMS
  lives in post-processing

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for environments and dependencies.

```bash
uv sync --group dev --extra export   # create .venv and install everything
uv run pytest                        # smoke tests, no dataset required
```

For Apple Silicon Macs (MPS) see [docs/TRAIN_MACOS.md](docs/TRAIN_MACOS.md).
For moving to a CUDA workstation see [docs/HANDOVER.md](docs/HANDOVER.md).
Training, evaluation and inference all pick CUDA → MPS → CPU automatically.

## Commands

Installation provides seven console scripts:

| Command | Purpose |
|---|---|
| `hydranet-train` | Training |
| `hydranet-eval` | Run validation on a checkpoint |
| `hydranet-infer-image` | Overlay inference on a single image or a folder |
| `hydranet-infer-video` | Video inference (uses the system ffmpeg, no opencv needed) |
| `hydranet-export-onnx` | Export ONNX for TensorRT |
| `hydranet-prepare-ade20k` | Filter ADE20K to its indoor subset and lay it out as `seg_folder` |
| `hydranet-report` | Summarise one run, or rank runs and diff their configs |

Run them all with `uv run <command>`, or `source .venv/bin/activate` first.

## Two deployment configs

| Config | Environment | Terrain classes | Segmentation datasets |
|---|---|---|---|
| `hydranet_regnet800mf.yaml` | Off-road | 12 outdoor (grass / gravel / tree-bush …) | RUGD, RELLIS-3D |
| `hydranet_indoor.yaml` | Indoor (lobbies / corridors / factory floors) | 12 indoor (floor / glass / stairs …) | ADE20K + your own annotations |

Model structure, losses and training mechanics are identical between the two — the only
differences are `data.terrain_classes`, `label_map` and the data sources. Both use COCO for
the detection head, unchanged.
Label schemes are defined in `SCHEMES` in
[`label_maps.py`](src/syncai_hydranet/data/label_maps.py); the indoor mapping is in
[`label_maps_indoor.py`](src/syncai_hydranet/data/label_maps_indoor.py).

If your camera's aspect ratio differs from `input_size`, turn on `data.letterbox: true`
(a portrait phone video squeezed straight into the input is compressed more than 2× horizontally).

## Preparing datasets

| Dataset | Used for | Download |
|---|---|---|
| ADE20K | Indoor terrain / traversability | <https://groups.csail.mit.edu/vision/datasets/ADE20K/> |
| RUGD | Off-road terrain / traversability | <http://rugd.vision/> |
| RELLIS-3D | Off-road terrain / traversability | <https://github.com/unmannedlab/RELLIS-3D> |
| COCO 2017 | Object detection | <https://cocodataset.org/#download> |

Expected layout:

```text
datasets/
├── ADE20K/                                # produced by hydranet-prepare-ade20k
│   ├── images/{train,val}/**.jpg
│   └── annotations/{train,val}/**.png     # single channel, integers 0..150
├── RUGD/
│   ├── images/{train,val}/**.png
│   └── annotations/{train,val}/**.png     # RGB palette
└── coco/
    ├── train2017/  val2017/
    └── annotations/instances_{train,val}2017.json
```

RUGD and RELLIS ship no official train/val split — split them by sequence, or the same
sequence leaks across both sides. The same goes for your own footage: split by **recording
session**, because adjacent frames are near-identical and a random split will badly
overstate performance.

Just want to get something running? Trim `data.datasets` down to whatever you actually
have — the heads are still built, they just go unsupervised (startup warns you which head
nobody is supervising).

### Three splits

`best.pt` is selected on **val**, which makes val an optimistic estimate of itself. For
numbers you can report, prepare a **test** split that no part of training ever reads:

```yaml
- name: ade20k
  split_train: train
  split_val: val
  split_test: test        # optional, and deliberately not defaulted to val
```

```bash
uv run hydranet-eval --config ... --checkpoint ... --split test
```

Passing `--split test` without a configured `split_test` fails loudly and tells you what to
create; it never falls back to val silently.

### Data versioning

Datasets are not in git and there is no DVC. Instead, every training run writes a
**fingerprint** of each split (file count, total bytes, digest of the path-and-size listing)
into `runs/<experiment>/meta.json`, so any checkpoint can answer "which data was I trained
on?". Re-export the annotations, add a few hundred site photos, or change a filter
threshold, and the two runs' fingerprints differ.

### The ADE20K indoor subset

ADE20K's 20k images span 1055 scene categories, most of them outdoor. The command below
filters on **annotation content** (enough floor, little sky and vegetation) to select the
ground-level indoor viewpoint a robot actually sees:

```bash
uv run hydranet-prepare-ade20k \
    --src datasets/ADEChallengeData2016 --dst datasets/ADE20K
# training -> train: kept 5998/20210 (29.7%)
```

The output is symlinks, so it costs no extra disk and re-running is cheap.

## Training

```bash
uv run hydranet-train --config configs/hydranet_indoor.yaml

# override any setting (dot-path)
uv run hydranet-train --config configs/hydranet_indoor.yaml \
    --set train.batch_size=8 model.neck.name=fpn 'data.input_size=[384,512]'

# resume: the schedule continues from where it stopped, it does not replay
uv run hydranet-train --config ... --resume runs/hydranet_indoor/last.pt
```

Training features: AMP mixed precision (`train.amp_dtype` accepts `bfloat16`), cosine +
warmup, EMA (the decay ramps up, so short runs are safe too), a lower learning rate for the
backbone, gradient accumulation, and best/last checkpoints.

Typos in `--set` are not swallowed: the config is validated as a whole after the overrides
are applied, and unknown keys, type mismatches, a `supervises` entry pointing at a
non-existent head, or a `terrain_classes` count that disagrees with the head are all listed
at once before anything aborts.

### Out of VRAM? Accumulate gradients

```bash
--set train.batch_size=4 train.grad_accum_steps=4     # effective batch 16
```

The schedule, EMA and `global_step` all advance per optimizer step, so when you move to a
different machine you only need to keep `batch_size × grad_accum_steps` constant and the
learning rate carries over unchanged.

### Which metric picks the model

`train.primary_metric` names the single number that decides `best.pt`; the default is
`traversability_mIoU`. Any key validation emits works, including per-class ones:

```yaml
primary_metric: IoU/traversability/00_blocked   # the class that actually strands a robot indoors
```

A misspelled metric name aborts and lists every available key — rather than quietly
selecting the model on some other criterion for a whole run.

### What each run leaves behind

```text
runs/<experiment>/
├── meta.json          git commit, dirty flag, environment, dataset fingerprints, full config
├── config.yaml        config snapshot after --set, ready to re-run directly
├── uncommitted.patch  only when the working tree had uncommitted changes (a commit hash
│                      alone cannot restore the code)
├── metrics.jsonl      one line per validation, machine-readable
├── train.log
└── tb/
```

If a directory of that name already holds results, the new run writes to a timestamped
sibling instead: it will not overwrite an existing `best.pt`, and it will not mix two runs'
TensorBoard events together. To continue a run, use `--resume` (which lists, item by item,
any setting that differs from the one stored in the checkpoint).

These files exist precisely so `hydranet-report` can read them:

```bash
uv run hydranet-report runs/hydranet_indoor        # detail and curves for one run
uv run hydranet-report runs/* --diff               # ranking across runs + config differences
```

```text
run                          commit    epochs  best      @epoch  metric
-----------------------------------------------------------------------
indoor-b                     eaba6b8e  40      0.6412    37      traversability_mIoU
indoor-a                     3c12224f* 40      0.5980    31      traversability_mIoU

indoor-a -> indoor-b
  train.lr: 0.0002 -> 0.0004
```

The `*` after a commit means that run's working tree was dirty.

### Watching a run

```bash
uv run tensorboard --logdir runs/
```

Beyond loss curves, TensorBoard also receives:

- **`val_pred/*` comparison images** — each validation writes an "input | prediction |
  label" triptych. Grey is letterbox padding, black is the ignore region. A curve only tells
  you the loss is going down; this is what shows you the model is calling an entire floor a wall.
- **`IoU/<head>/<class>`** — per-class IoU. Indoors, the classes that matter most (glass,
  stairs) occupy a tiny pixel fraction, and mIoU alone lets the large classes hide them.
- **`task_weight/*`** — the `exp(-s)` learned by uncertainty weighting, which shows which
  task is dominating the trunk's gradients. A head collapsing toward zero has effectively
  stopped learning.

## Evaluation and inference

```bash
uv run hydranet-eval --config ... --checkpoint runs/.../best.pt

# report final numbers on the held-out test split, as JSON for cross-run comparison
uv run hydranet-eval --config ... --checkpoint runs/.../best.pt \
    --split test --json reports/best_test.json

uv run hydranet-infer-image --config ... --checkpoint runs/.../best.pt \
    --input photo.jpg --output out/

uv run hydranet-infer-video --config ... --checkpoint runs/.../best.pt \
    --input clip.mp4 --output clip_pred.mp4 --fps 10
```

## Deploying to Jetson Orin

```bash
uv run hydranet-export-onnx --config ... --checkpoint runs/.../best.pt --output hydranet.onnx
# on the Jetson:
trtexec --onnx=hydranet.onnx --saveEngine=hydranet_fp16.engine --fp16
```

Output node definitions, C++ post-processing, INT8 quantisation and latency estimates:
see [docs/DEPLOY_JETSON.md](docs/DEPLOY_JETSON.md).

## Adding a task head (example: monocular depth)

1. Add the head module under `src/syncai_hydranet/models/heads/` (it takes the FPN feature list)
2. Register the type branch in `hydranet.py::HydraNet.__init__` and add its loss in `compute_losses`
3. Add a `model.heads` section to the config, and list the new head under the relevant
   dataset's `supervises`

Heads are mutually independent, so this does not affect training or deployment of the
existing ones.

## Development

```bash
uv run ruff check --fix .    # lint + import sorting
uv run ruff format .         # formatting
uv run pytest --cov          # tests + coverage
uv run pre-commit install    # enable pre-commit checks
```

CI runs lint plus a Python 3.10 / 3.12 test matrix on GitHub Actions, and fails below 68%
coverage. The tests need no datasets: model tests run on random tensors, dataset tests build
a fixture in `tmp_path`, and `test_overfit.py` verifies the training loop really converges by
memorising one synthetic batch to over 95% pixel accuracy (chance is 33% across three classes).

## Project layout

```text
src/syncai_hydranet/
├── config.py                 # YAML config + dot-path overrides
├── config_schema.py          # config validation: unknown keys, types, cross-field consistency
├── cli/                      # console script entry points (train/eval/infer/export/report)
├── models/
│   ├── backbone.py           # RegNet / ResNet multi-scale features
│   ├── neck.py               # BiFPN / FPN
│   ├── heads/segmentation.py # Semantic-FPN segmentation head (shared by both seg heads)
│   ├── heads/detection.py    # FCOS head + target assignment + decode/NMS
│   ├── losses.py             # CE+Dice / Focal+GIoU / uncertainty weighting
│   └── hydranet.py           # assembly + compute_losses + predict
├── data/
│   ├── label_maps.py         # off-road mappings + the SCHEMES registry
│   ├── label_maps_indoor.py  # indoor 12 classes + ADE20K mapping
│   ├── datasets.py           # SegFolderDataset / CocoDetDataset / split resolution
│   ├── transforms.py         # joint image+mask+box augmentation, letterbox, geometry inversion
│   ├── fingerprint.py        # dataset fingerprints, written into meta.json
│   └── multitask.py          # round-robin loader across datasets
├── engine/
│   ├── trainer.py            # AMP / EMA / cosine / gradient accumulation / checkpoints / TB
│   └── evaluator.py          # mIoU + COCO mAP + primary metric selection
└── utils/
    ├── device.py             # CUDA → MPS → CPU
    ├── checkpoint.py         # safe loading (weights_only) + format version
    ├── runmeta.py            # git / environment / config snapshot / metrics.jsonl
    ├── seeding.py            # global seed, worker seeds, backend flags
    └── visualize.py          # palettes, overlays, letterbox, comparison images
```

## Licence

Apache-2.0 (ADE20K / RUGD / RELLIS-3D / COCO each carry their own data licences — check them
before commercial use).
