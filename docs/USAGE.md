# Using it: datasets, training, evaluation

The operational half of the README, moved here so the front page can be read in a
minute. This is *how* to run the thing; [TRAINING_GUIDE.md](TRAINING_GUIDE.md) is why
it is shaped this way and which measurements lie, and [METHODOLOGY.md](METHODOLOGY.md)
is how a team divides the work.

Paths in this file are relative to the repository root.

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
# training   -> train: kept 5998/20210 (29.7%)
# validation -> val:   kept  614/2000  (30.7%)
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

`--layout side` (the default) writes traversability on the left and terrain on the right,
both from the same forward pass:

![traversability and terrain from one forward pass](../assets/hydranet_demo.gif)

An office corridor on the 60-epoch multi-task checkpoint. Floor, wall and partition are
solid because those are the classes ADE20K has in quantity; the same model scores
`caution` 0.33 and `stairs` 0.32 on the test split, and returns a quarter of a ceiling as
"go". Watch a clip before trusting a number — a curve cannot show you a model calling a
ceiling a floor, and this is the cheapest way to find it.

## Deploying to Jetson Orin

A run directory is not something you deploy. `scripts/release_bundle.sh` freezes one into
an immutable bundle — weights, ONNX graph, config, lineage and metrics, checksummed
together — because git versions the code and nothing in git versions the model.

```bash
# on the workstation
scripts/release_bundle.sh create runs/hydranet_indoor v1   # -> releases/v1
scripts/release_bundle.sh verify releases/v1               # re-check every sha256
scripts/release_bundle.sh publish releases/v1 gs://syncai-hydranet

scp releases/v1/model.onnx orin:~/
```

```bash
# on the Orin
./bench_orin.sh model.onnx        # builds the FP16 engine and reports GPU latency
```

The engine is deliberately **not** part of the bundle. It is tied to a GPU architecture, a
TensorRT version and a JetPack version, so it is a per-target build artefact: build it on
the board and keep it beside the bundle as
`engines/<board>_<jetpack>_<trt>_<precision>.engine`.

`bench_orin.sh` reports pure GPU inference latency, which is not the number that decides
whether the robot keeps up. For that, `scripts/bench_camera_orin.py` measures end-to-end
camera-to-output frame rate, including capture, letterboxing, and the host-side argmax and
NMS that the graph deliberately leaves out.

Bringing up a board from scratch is its own runbook:
[docs/ORIN_BRINGUP.md](ORIN_BRINGUP.md) walks a freshly flashed Orin to a live
prediction stream, with each failure written symptom-first — TensorRT is not in the base L4T
image, the user is not in the `video` group, and on our board CUDA still needs root for
reasons we did not resolve.

Output node definitions, the post-processing maths, INT8 quantisation and the measured
latencies are in [docs/DEPLOY_JETSON.md](DEPLOY_JETSON.md).
