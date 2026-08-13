# Project methodology

How a team picks this project up and runs it: who owns what, what data has to be collected,
how models are trained, and how they are evaluated and tested before anything reaches a
robot.

This is the *process* document. For the concepts behind multi-head training, read
[TRAINING_GUIDE.md](TRAINING_GUIDE.md) first — it is the shorter of the two and this one
assumes it.

---

## 0. Where the project actually stands

Read this before planning anything, because it determines the priorities below.

The model design is ahead of the data. The architecture works, exports cleanly to TensorRT,
and the parameter budget confirms the shared-trunk premise (84.4% trunk, 15.6% heads). What
limits the system today is not the network:

| Constraint | Status |
|---|---|
| `caution` cannot exceed ~0.20 | 3 of the 4 terrain classes that map to it have **zero** training examples |
| Terrain metrics are not stable across dataset changes | `terrain_mIoU` averages over the classes *present*, currently 8 of 12 |
| No defensible headline number exists | No `split_test` is configured; every score so far is val, which selected the checkpoint |
| Detection produces garbage if deployed | The head is unsupervised and still gets exported |
| Real-world accuracy is unknown | Training data is ADE20K only — web photos, not robot-height footage of our sites |

**The work that moves this project is data work.** Plan accordingly: a team that assigns four
engineers to model tuning and none to annotation will produce a better-tuned model with the
same ceiling.

---

## 1. How to divide the work

Four workstreams. They are separable — each has a defined interface with the others — so they
can run in parallel and be owned by different people.

### A. Data and annotation — *the critical path*

Owns everything from camera to a directory the training code can read.

- Capture planning: sites, lighting, camera height, sessions
- Annotation: running it, writing the spec, quality control
- Split discipline: deciding which sessions are train, val and test
- Publishing datasets into `seg_folder` layout

**Interface out:** a dataset directory plus a one-line config entry. Nothing about the model
concerns this stream.

**Staff this first and most heavily.** It is the binding constraint, it has the longest lead
time, and it is the one thing that cannot be parallelised away later.

### B. Training and evaluation

Owns the model, the runs and the numbers.

- Config discipline, run hygiene, checkpoint selection
- Reading results, deciding what is a real improvement
- Loss balancing, schedules, resolution and batch decisions
- Keeping the evaluation protocol honest

**Interface in:** dataset directories from A. **Interface out:** a `best.pt` plus a
`meta.json` that says exactly what produced it.

One person can hold this. It is the *least* parallelisable of the four — two people tuning
the same model in different directions produce results neither can interpret.

### C. Deployment

Owns everything from `best.pt` to a running engine on the robot.

- ONNX export and its validation
- TensorRT engine builds, FP16 now, INT8 when the model settles
- The C++/Python runtime and post-processing (argmax, decode, NMS)
- Pre-processing parity between training and the robot
- Latency budgets on the target board

**Interface in:** a checkpoint plus the config that made it. See
[DEPLOY_JETSON.md](DEPLOY_JETSON.md).

Can start immediately and in parallel — the ONNX contract is already stable, so this stream
does not have to wait for a good model to build out the runtime.

### D. Platform

Owns the things that make the other three repeatable.

- Environments, drivers, `uv.lock`, CI
- Run storage, and a shared view of results once training spans machines
- Release and versioning discipline

Part-time, but real. Deferring it is what produces "it works on my machine" at the worst
moment.

### Coordination rules

- **One person decides `primary_metric`.** It defines what "better" means; if two people
  disagree about it, no comparison between their runs is meaningful.
- **The annotation spec is a shared contract**, not a note in someone's head. It lives in the
  repo, and the terrain→traversability policy table is part of it.
- **Nobody edits a config value and a dataset in the same change.** When two variables move,
  a difference in results tells you nothing.
- **All four streams read `meta.json`.** It is the common language: which code, which data,
  which settings produced a result.

---

## 2. What data to collect

### Priority order

Ranked by safety impact against the size of the gap. This ordering is not negotiable — the
top three classes have *no* examples at all, so they are unlearnable today.

| Priority | Class | Why | Current examples |
|---|---|---|---|
| **1** | `wet_slippery` | The most direct fall risk. Standing water, freshly mopped floor, spills. | **0** |
| **2** | `floor_metal` | Grating aperture versus foot size is a hard constraint. Steel plate, drain covers. | **0** |
| **3** | `threshold_ramp` | Door sills and level changes are the commonest indoor snag. | **0** |
| 4 | `glass` | Has data (~0.50 IoU) but the highest cost when wrong — it reads as an open corridor. | ADE20K only |
| 5 | `stairs` | Has data, but thin at 0.3% of pixels. | ADE20K only |

Everything else — floor, wall, door, furniture, person — is adequately covered by ADE20K.
Do not spend annotation budget there.

### How much

As an initial target per priority class:

- **300–500 frames** containing the class, across **at least 5 distinct sites or sessions**
- Of those, **~100 frames held back as test**, from sessions that appear in no other split

Variety across sessions matters more than raw count. Five hundred frames from one corridor
teaches the model that corridor.

### How to capture

The training data has to look like what the robot sees, or the model learns a domain it will
never be deployed in.

- **Camera height and angle: mount it on the robot**, or hold it at the robot's camera height
  and pitch. ADE20K's weakness is exactly this — human-height, human-framed photographs.
- **Lighting: deliberately vary it.** Include the bad cases: backlit entrances, overhead glare
  on polished floors, dim corridors, mixed daylight and fluorescent.
- **Include the hard negatives.** Specular reflections on a polished floor that look like an
  open corridor, mirrors, glass walls with objects visible behind them. These are where the
  dangerous failures live.
- **Record continuous sessions and log them.** Session identity is what makes an honest split
  possible later; frames without provenance cannot be split safely.

### Annotation spec

- Annotate the **12 indoor terrain classes** directly. Use `label_map: indoor_native`, where
  annotation ids *are* the class ids — no translation layer to drift.
- **Do not annotate traversability separately.** It is derived from terrain through the policy
  table in `data/label_maps_indoor.py`. One annotation pass supervises both heads.
- **Pin the policy table in the annotation spec.** If annotators and training disagree about
  what `threshold_ramp` means, the labels are silently wrong and nothing will surface it.
- Ambiguous pixels get **255 (ignore)**, never a guess. An ignore region costs nothing; a
  wrong label is trained on.
- The three entries marked `REVIEW` in the policy table are **platform decisions, not
  annotation decisions** — whether the gait handles stairs, whether the foot fits the grating.
  Settle them with whoever owns the robot, then write the answer down.

### Split rules

**Split by session or sequence. Never randomly.** Adjacent frames are near-identical; a random
split puts the same moment on both sides and validation then measures memorisation. The score
will look excellent and mean nothing.

Three splits, with distinct jobs:

| Split | Job | Who may look at it |
|---|---|---|
| `train` | Fit the weights | Everyone, constantly |
| `val` | Select the checkpoint, tune settings | Training stream, every epoch |
| `test` | Report the number | **Once, at a release decision** |

Configure `split_test` explicitly. It is deliberately not defaulted, because a silent fallback
to val produces an official-looking number that is quietly circular.

---

## 3. How to train

### Before the first run

1. `uv sync --group dev --extra export`, then `uv run pytest -q` — 166 tests, no dataset
   needed. If these fail, stop; nothing downstream will be interpretable.
2. `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg` — both types, see
   [CONTRIBUTING.md](../CONTRIBUTING.md).
3. Confirm the device: the log's first lines print `device=`, the AMP dtype and the backend
   flags. On CUDA see [HANDOVER.md](HANDOVER.md); on Apple Silicon see
   [TRAIN_MACOS.md](TRAIN_MACOS.md).

### Config discipline

`--set` is for experiments. **Anything you intend to keep goes in a config file.**

```bash
# experiment
uv run hydranet-train --config configs/hydranet_indoor.yaml --set train.lr=4.0e-4

# keeping it: copy the config, edit, commit
cp configs/hydranet_indoor.yaml configs/indoor_site_a.yaml
```

`--set` cannot index lists — `data.datasets[2].sample_ratio=0.1` silently creates a useless
key rather than erroring. Override the whole list, or copy the file.

Typos are caught: the config is validated as a whole after overrides, and unknown keys, wrong
types, a `supervises` naming a non-existent head, or a `terrain_classes` count that disagrees
with the head all abort with the full list of problems.

### Choose `primary_metric` deliberately, before the run

It is the single number that selects `best.pt`, which makes it the run's real objective. The
default `traversability_mIoU` is a balanced choice. When a specific failure mode matters more,
say so:

```yaml
primary_metric: IoU/traversability/00_blocked
```

### The run itself

- **Start from the settings that are already known to work** for the hardware — section 6 of
  [HANDOVER.md](HANDOVER.md) for CUDA — rather than inventing a configuration.
- **Change one thing at a time.** With two variables moving, a difference tells you nothing.
- **Let it run.** Intermediate epochs are noisy: this project's own baseline run measured
  `caution` at 0.158 on epoch 14 and 0.229 by epoch 27. Judging a run mid-flight is how teams
  abandon configurations that were working.
- **Resume rather than restart** when a run is interrupted: `--resume runs/.../last.pt`
  continues the schedule and reports any config drift from the checkpoint.

### What to watch, in order of value

1. **`val_pred/*` images.** The highest-value signal available. A falling loss is entirely
   compatible with the model calling a whole floor a wall; you see that instantly in a
   picture and never in a curve.
2. **Per-class IoU** for the rare, dangerous classes — `glass`, `stairs`, the new ones.
3. **`task_weight/*`.** A head's weight collapsing toward zero means it has stopped learning
   while the total loss still looks healthy.
4. **img/s.** Well below what the hardware should do means the data pipeline is the
   bottleneck, which is a different fix entirely.

### When to stop tuning

When the failing class has no data. If `caution` will not move, the question is not "which
learning rate" — it is "how many examples of it exist?" Route that back to workstream A. This
is the single most common way time gets wasted on this project.

---

## 4. How to evaluate and test

Four levels, cheapest first. Each answers a different question, and none substitutes for
another.

### Level 1 — Does the code work? (every commit)

```bash
uv run pytest -q          # 166 tests, no dataset required
```

Includes `test_overfit.py`, which memorises one synthetic batch to >95% pixel accuracy. Shape
tests pass on a model wired backwards; this is what separates "it runs" from "it trains".
CI runs the same on Python 3.10 and 3.12 and fails below 68% coverage.

### Level 2 — Is the model learning? (every epoch)

Automatic. Validation writes to `metrics.jsonl` and TensorBoard, and `best.pt` is updated on
`primary_metric`. This is for steering, not for reporting.

**Know what mIoU is averaging.** `terrain_mIoU` is the mean over classes *present in the
validation set* — currently 8 of 12. Two consequences:

- It is **not comparable** to a published 12-class mIoU.
- It will likely **drop** when you add the missing classes, because harder classes enter the
  average. That is not a regression. Record the class count alongside the number, and compare
  per-class IoU when the dataset changes.

### Level 3 — How good is it, honestly? (release candidates)

```bash
uv run hydranet-eval --config ... --checkpoint runs/.../best.pt \
    --split test --json reports/rc1_test.json
```

Rules that make this number worth anything:

- **Run it once**, at a release decision. Every extra look leaks test into your choices, and
  it degrades into a second val split.
- **Report per-class**, not just mIoU. A mean that hides `glass` at 0.2 is not a summary.
- **Use the same `--weights`** as deployment will (EMA by default).
- **Never tune against it.** If a test result is disappointing, the response is more data or
  a different approach — not a new checkpoint chosen on that same test set.

### Level 4 — Does it work on the robot? (before deployment)

The only level that measures the actual system. The three below it can all pass on a model
that fails here.

- **Field footage inference.** Run `hydranet-infer-video` on real site recordings and watch
  the overlay. Reflections, glass and thresholds are what to look for.
- **Export parity.** Confirm the ONNX/TensorRT output matches PyTorch on the same input.
  A silent mismatch here — usually pre-processing — is the classic deployment bug: the engine
  input is `(pixel/255 - mean) / std` with ImageNet statistics in **RGB**, and a BGR camera
  feed costs 5–10 points quietly.
- **Latency on the target board**, not on the workstation.

### Acceptance gates before deployment

Every one of these is currently unmet, which is a fair summary of the project's state:

| Gate | Requirement |
|---|---|
| Every head is supervised | Nothing exports with initial weights. Detection needs COCO, or the head must be removed from the export |
| A `test` split exists | Configured, populated, split by session, never trained on |
| Per-class floors met | `glass` and each caution class above an agreed threshold — set it with the robotics team, not by whoever is training |
| Field footage reviewed | A human has watched the overlay on real site video |
| Export parity confirmed | TensorRT output matches PyTorch within tolerance |
| Provenance intact | `meta.json` identifies code, config and dataset fingerprint for the exact checkpoint being shipped |

---

## 5. Keeping runs answerable

Every run writes `meta.json` (commit, dirty flag, environment, dataset fingerprints, resolved
config), `config.yaml`, `uncommitted.patch` when the tree was dirty, `metrics.jsonl`,
`train.log` and `tb/`. A second run into an occupied directory gets a timestamped sibling
rather than overwriting `best.pt`.

```bash
uv run hydranet-report runs/*  --diff     # rank runs, and show what differed
```

Two habits make this pay off:

- **Never delete a run directory that produced a shipped model.** The dataset fingerprint in
  it is the only record of which data that model saw.
- **Do not rewrite git history that run metadata points at.** Runs record the commit they came
  from; if that hash stops existing, provenance breaks. This is why
  `backup/pre-conventional-commits` is retained.

We do not run an experiment tracking server. For one machine, files plus TensorBoard cover it,
and the provenance above is better than a tracker's defaults. That calculation changes once
training routinely spans machines and there is no single place to compare them — which is the
point at which workstream D should revisit it.

---

## 6. Suggested order for a team starting today

1. **Week 1 — capture.** Workstream A plans sites and shoots the three zero-example classes.
   Workstream C builds out the Jetson runtime against the existing ONNX contract. Workstream B
   reproduces the current baseline so there is a known-good reference.
2. **Week 2 — annotate and split.** First annotated batch lands. Configure `split_test` at the
   same time, from sessions reserved before annotation starts.
3. **Week 3 — retrain and measure.** First run including in-house data. Compare per-class, not
   mIoU, for the reason in level 2.
4. **Then, and only then**, decide whether the model needs changing. It probably does not: at
   84.4% of parameters in a shared trunk exporting cleanly to TensorRT, the architecture is
   not what is holding this back.
