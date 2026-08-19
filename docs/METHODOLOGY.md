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

> **Revised 2026-08-19.** Four rows of the table below were rewritten because the
> measurements behind them were superseded: the COCO row's collapse was an artefact of
> scoring on the split that also selected the checkpoint (see level 2), and site data
> arrived in the meantime. The shape of the conclusion did not change — the constraint is
> still data — but the numbers that argued it did.

| Constraint | Status |
|---|---|
| `caution` is capped by its sourcing, not by tuning | 3 of the 4 terrain classes that map to it have **zero** training examples. It scores **0.33 on the held-out test split** — the ~0.20 figure this row used to quote was a val number, and val also selected the checkpoint |
| Terrain metrics are not stable across dataset changes | `terrain_mIoU` averages over the classes *present*, currently 8 of 12 |
| A *field* number exists and is an **agreement**, not an accuracy | Site splits are built and reserved by camera ([RETAIL_OBJECTS_SPLIT.md](RETAIL_OBJECTS_SPLIT.md)), but every mask in them is SAM 3 output with no human pass, so a model trained on SAM 3 and scored against SAM 3 shares its errors. `split.json`'s `test_provenance` says so with the data |
| Detection is trained, and the number depends entirely on which set | `releases/v1` scores **mAP 0.335 on COCO val2017** (0.3246 on the 25 indoor categories). The retail line, which is the only one scored on site boxes at all, gets **0.07–0.11 on `site_boxes`** from a different family of runs. A web number is not evidence about a store, and the two are not comparable. Export refuses any head no dataset supervises |
| The COCO share is a monotonic trade, not a collapse | Swept at four ratios and scored on **test** (`journal/2026-08-14-experiments-and-geometry.md`): at `sample_ratio: 0.1` every segmentation metric matches or beats the segmentation-only baseline and detection arrives free; above it, segmentation falls monotonically. There is no sweet spot to find and no blocker here — only an exchange rate that gets worse as you climb |
| Real-world accuracy is still unmeasured | Site footage is now in training (`retail_objects_batch02`/`batch03`, 23 selling-floor cameras) — what is missing is a *human-corrected* test split, not the data |

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
- The C++/Python runtime and post-processing (decode, NMS — and argmax, unless the export
  folded it into the graph, which on one board halved the frame)
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

### The robot was assumed to have LiDAR, and that decided the priority order

> **⚠ Corrected 2026-08-19: the platform this section was written for does not exist.**
> The robot is a **Lite3 with one monocular camera and two ultrasound returns**, and the
> ultrasound is `{forward, backward}` — only one of the two has a camera pointed at it, and
> [RESEARCH_OCCUPANCY.md](RESEARCH_OCCUPANCY.md) measured that its echo is *lateral*: across
> 29 scored frames, **0 were comparable** to the camera's forward cone. There is no point
> cloud. "The Lite3 LiDAR variant" appears in that document as one of the sensors the
> project might **buy** if E-prep says a depth sensor is needed — which is the open decision,
> not a description of what is mounted.
>
> **Read the ranking below with that substituted, because the substitution does not change
> it uniformly:**
>
> * **Priorities 1 and 2 (`glass`, `wet_slippery`) are unaffected** — they were ranked first
>   because they are invisible to *any* range sensor and the camera is the only instrument.
>   That argument never depended on LiDAR being present.
> * **Priority 3 (`floor_metal`) is unaffected** for the same reason: aperture-versus-foot is
>   a material judgement.
> * **Priorities 4 and 5 (`threshold_ramp`, `stairs`) move back up, and the sentence "no
>   longer the camera's problem alone" is false.** Nothing else on this robot measures a
>   level change. Geometry is the camera's problem — or a depth sensor's, once E-prep
>   answers whether one is needed, and E-prep is itself blocked on walking the robot.
> * **"LiDAR as an annotation lever" and "LiDAR as free validation" below are void**, not
>   deferred. Both need a point cloud. Their replacement is the auto-labelling pipeline in
>   RESEARCH_OCCUPANCY.md, whose whole premise is that no 3D sensor exists.

The section as written assumed the platform carries LiDAR alongside the camera, so
**this model's job is the half of the problem LiDAR cannot do**. Ranking annotation effort
by "which class is most dangerous" alone gets the order wrong; rank by *danger × how blind
the range sensor is* — and with no range sensor, everything geometric comes back.

| | LiDAR | Camera (this model) |
|---|---|---|
| Range and geometry | Accurate | Unreliable from one camera |
| Stair edges, level changes | Directly measurable | Inferred |
| Floor material (hard / soft / metal) | Invisible | The only source |
| **Glass** | **Invisible — passes through or returns noise** | **The only source** |
| **Wet floor** | **Invisible, and specularity adds false returns** | **The only source** |

### Priority order

| Priority | Class | Why | Current examples |
|---|---|---|---|
| **1** | `glass` | **LiDAR cannot see it at all**, and it reads as an open corridor. Currently 0.50 IoU against a 0.97 ceiling — the single largest safety gap in the system. | ADE20K only |
| **2** | `wet_slippery` | Also invisible to LiDAR, and specular returns actively mislead it. The most direct fall risk. | **0** |
| 3 | `floor_metal` | Grating aperture versus foot size is a material judgement; LiDAR sees the holes but not whether the surface is safe. | **0** |
| 4 | `threshold_ramp` | A level change is pure geometry, so LiDAR measures it directly. Still needed for the semantic map, but it is no longer the camera's problem alone. | **0** |
| 5 | `stairs` | Same reasoning: step edges are geometric. | ADE20K only |

**One thing to verify on the robot before treating 4 and 5 as settled.** Door sills are
often only 2–5 cm high, and whether a range sensor resolves that depends on its angular
resolution and mounting height. **Answered by the correction above rather than by a
measurement: there is no such sensor, so `threshold_ramp` and `stairs` have already moved
back up the list.** The question that replaces it is E-prep's — how a monocular depth
teacher behaves below 0.7 m on this lens, which public data cannot bound (NYUv2's returns
start near 0.7 m and the robot's floor sits at 0.34 m).

Everything else — floor, wall, door, furniture, person — is adequately covered by ADE20K.
Do not spend annotation budget there.

### LiDAR as an annotation lever

Geometric classes can be **pre-labelled automatically**: detect step edges and level changes
in the point cloud, project them into the image, and let annotators correct a mask rather
than draw one. That directly reduces the bottleneck for `stairs` and `threshold_ramp`.

The material classes — `glass`, `wet_slippery`, `floor_metal` — have no such shortcut. They
are hand-drawn, which is a second reason they belong at the top of the list.

### LiDAR as free validation

Where the model predicts *go* and the point cloud shows a 20 cm drop, that is a failure case
found without anyone annotating anything. This is the cheapest possible implementation of
the level-4 field evaluation below, and it should be running continuously rather than as a
one-off study.

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

1. `uv sync --group dev --extra export`, then `uv run pytest -q` — 1,171 tests, no dataset
   needed. If these fail, stop; nothing downstream will be interpretable.
2. `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg` — both types, see
   [CONTRIBUTING.md](../CONTRIBUTING.md).
3. Confirm the device: the log's first lines print `device=`, the AMP dtype and the backend
   flags. On CUDA see [the CUDA move](journal/2026-08-12-mps-to-cuda.md); on Apple Silicon see
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
  [the CUDA move](journal/2026-08-12-mps-to-cuda.md) for CUDA — rather than inventing a configuration.
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
uv run pytest -q          # 1,171 tests, no dataset required
```

Includes `test_overfit.py`, which memorises one synthetic batch to >95% pixel accuracy. Shape
tests pass on a model wired backwards; this is what separates "it runs" from "it trains".
CI runs the same on Python 3.10 and 3.12 and fails below 80% coverage.

### Level 2 — Is the model learning? (every epoch)

Automatic. Validation writes to `metrics.jsonl` and TensorBoard, and `best.pt` is updated on
`primary_metric`. This is for steering, not for reporting.

**Know what mIoU is averaging.** `terrain_mIoU` is the mean over classes *present in the
validation set* — currently 8 of 12. Two consequences:

- It is **not comparable** to a published 12-class mIoU.
- It will likely **drop** when you add the missing classes, because harder classes enter the
  average. That is not a regression. Record the class count alongside the number, and compare
  per-class IoU when the dataset changes.

The retail-objects line is the worked example, and it is the reason to read `mIoU_classes`
before reading `mIoU`. The 60-epoch run reports **0.7668 over five classes** on ADE20K val;
the site runs report **~0.56–0.60 over six**, on a val set that also contains real store
frames. Nothing got worse: a sixth and much harder class entered the mean, and the mean is
now taken against harder data. **The number went down and the measurement got better** — at
the same epoch the newer run is ahead, 0.5973 against 0.5585.

**And a per-class IoU needs its support, which is the same argument one level down.**
`support/<head>/NN_<name>` is emitted beside every `IoU/<head>/NN_<name>` for exactly this:
`column` scored 0.40–0.51 on **0.66% of ADE20K val pixels**, 22 of 285 images, and then
predicted **0.00% of pixels** across 240 frames of four daytime store cameras. A number and a
well-evidenced number are formatted identically, and only one of them is a measurement. Below
`evaluator.THIN_SUPPORT` (1% of labelled pixels) the log says so and the honest reading is
"not measured".

**A mean can hide a class that stopped learning entirely.** This is the sharper failure.
The instance this section was written from is **withdrawn**, and both halves are kept
because the difference between them is the lesson. Adding COCO to supervise the detection
head appeared to move `traversability_mIoU` from 0.6765 to 0.6303 — a 7% drop, easy to
accept as the cost of a second task. Underneath, on **val**:

| Class | Seg-only (ships) | With COCO (ships) |
|---|---|---|
| `blocked` | 0.9547 | 0.9536 |
| `go` | 0.8455 | 0.8463 |
| **`caution`** | **0.2294** | **0.0908** |

> **Withdrawn 2026-08-14: these are val numbers, and val also selected both checkpoints.**
> Scored on the held-out **test** split, the same checkpoints are equivalent on `caution` —
> 0.3252 against 0.3346 — and the four-point ratio sweep in
> [`journal/2026-08-14-experiments-and-geometry.md`](journal/2026-08-14-experiments-and-geometry.md)
> shows the real shape: segmentation falls **monotonically** with COCO share, and at 0.1 it
> comes out slightly *ahead* of the segmentation-only baseline. There was no collapse at 0.1
> to explain. `best.pt` is chosen on val, so each run was cut at whatever epoch that rare
> class happened to spike, and the selection manufactured the variance it was then measured
> with. The comparison also changed two things at once — COCO was added *and* the run was cut
> from 60 epochs to 30 — which is the other reason it could not have attributed anything.
>
> **What survives is the reading habit, not the number.** A mean really can hide a class
> pinned at zero; the denominator never changes, so the class count would not catch it. Read
> per-class IoU as a time series. Just do not read a rare class off the split that selected
> the checkpoint — see "Report rare classes on the test split only" in
> [RETAIL_SCOPE.md](RETAIL_SCOPE.md) §6.

The starvation mechanism is still worth understanding, because a *large* COCO share does
cost segmentation and the sweep measures it: COCO has 118,287 images against ADE20K's 5,998,
so the ratio decides how many optimiser steps the segmentation heads get at all. `stairs`
occupies 0.3% of pixels, and since three of `caution`'s four constituent classes have no data
at all, `caution` is effectively `stairs` and moves with it. At `sample_ratio: 1.0` on test,
`glass` goes 0.5021 → 0.0692. **Share is the lever; 0.1 is the floor, not an optimum.**

**It recurred on 2026-08-17 as `product`, and one part of the account is established while
another is not. Both are recorded here, because this is the process reference and a reader
should be able to tell which is which.**

*Established.* `product` sat at exactly **0.000 for 22 consecutive epochs**. ADE20K was
**90.2% of segmentation steps and contains zero `product` pixels**, so in nine batches out of
ten every pixel was a labelled *negative* for that channel. That is a stronger statement than
starvation: a starved class learns slowly, and this one did not move at all. **Sign, not
share, is what explains a channel pinned at exactly zero.**

*Not established: that sign is the whole account.* Rebalancing to 41.9% site did not hold the
class. `runs/hydranet_retail_objects_site_balanced` is the control and the shape is the point
— read it rather than the claim:

| epoch | 2 | 3 | 10 | 15 | 25 | 39 |
|---|---|---|---|---|---|---|
| `IoU/terrain/05_product/site_sam3` | **0.4821** | 0.0395 | 0.1030 | 0.0300 | 0.0677 | 0.0526 |

**It spiked and collapsed.** If removing the negative evidence were sufficient, it should have
held. Rebalancing *delayed* the collapse rather than preventing it.

Selection behaved correctly here, and it is worth reading why. That run's `primary_metric` is
`IoU/terrain/05_product/site_sam3` — the class under investigation, not `terrain_mIoU` — so
`best.pt` holds **epoch 2 at 0.4821**, the peak itself. The log records exactly two `new best`
lines, both inside the first two minutes, and nothing across the remaining 37 epochs. **A
checkpoint pinned at epoch 2 of 39 is not a selection failure; it is the finding.** It says
the run's best answer for that class arrived before it had learned anything else, which is
the same information the table shows and harder to miss.

The competing account is that this is a *contradiction* rather than an absence:
`ADE20K_ID_TO_RETAIL_OBJECTS` sends **15 source ids to `fixture` and 0 to `product`**, so a
shelf of goods is one `fixture` region under ADE20K and a `fixture` + `product` split under
the site masks — the same pixels with two targets. **That account is unmeasured, not
refuted**, and the reason is worth knowing: `retail_objects_batch01` leaves floor and wall at
255 (57.5% of its pixels are ignored, `floor` and `wall` are both 0.00%), so `fixture`
over-predicting into the floor is invisible to every metric computed on that split.
`retail_objects_batch02` asserts them — floor 29.8%, wall 21.1%, 11.7% ignored — which is why
it exists.

**The fix has a direction, and the obvious one is wrong.** Lower the abundant dataset's
`sample_ratio`; do not raise the scarce one. Both reach the same balance, but raising the
scarce one gets there by showing the same few images many times an epoch, which trades a
suppressed channel for a memorised one. That direction holds whichever account is right.

`config_schema.unsourced_terrain_classes` names classes *no* dataset can produce, and
`minority_sourced_terrain_classes` names the ones only a smaller dataset can — both before a
GPU-hour is spent. Neither can see pixels, so both are candidates to confirm rather than
conclusions; see the identity-map caveat in the second one's docstring.

**So: never read a mIoU without the per-class numbers next to it**, and when a dataset mix
changes, check that the rare classes still have a pulse before accepting the mean.

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

Written when every one was unmet. **Two have since been met** (2026-08-19, cutting
`releases/v1`): export parity was confirmed at worst relative divergence 7.95e-06 across 17
outputs, and the export guard that refuses an unsupervised head is what let the bundle build
at all. The rest still stand, and [RELEASE.md](RELEASE.md) §3 carries the current list:

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
- **Know what a history rewrite does to run metadata.** Runs record the commit they came
  from, and the Conventional Commits rewrite changed every hash — so the baseline run's
  `meta.json` names `ba30fa88`, which `git` can no longer reach. **No code was lost:** the
  rewrite only edited commit messages, so the old and new commits have byte-identical
  trees (`git diff ba30fa88 aa07bbe` is empty). Recover the mapping by matching the commit
  *message* rather than the hash — the rewrite lower-cased the subject and added a type
  prefix, so `ba30fa88` is `aa07bbe`. [CONTRIBUTING.md](../CONTRIBUTING.md) carries the
  current mapping. If a future rewrite ever changes trees rather than messages, tag the
  commits a run directory references *before* doing it.

We do not run an experiment tracking server. For one machine, files plus TensorBoard cover it,
and the provenance above is better than a tracker's defaults. That calculation changes once
training routinely spans machines and there is no single place to compare them — which is the
point at which workstream D should revisit it.

### A run is not a version, and the gap between them was 54 to 0

Counted on 2026-08-19: **54 run directories, 0 releases.** Of the 49 carrying `meta.json`,
**31 were trained from a dirty tree** — their exact source exists only in
`uncommitted.patch`, so they cannot be released at all, and the check that would have said
so runs at release time, long after the eight hours were spent.

That ratio is the argument for two habits, and neither is about tooling:

- **Refuse a dirty tree at the start of training, not at release.** The warning the trainer
  prints today is read after the fact or not at all. The cost of ignoring it is a whole run.
- **Cut the release when the model ships, not when someone remembers.** `releases/v1` was
  cut months after `hydranet_joint_coco10` went onto the robot, and cutting it immediately
  surfaced that the shipped checkpoint scores 0.360 terrain mIoU where the project's own
  notes had been quoting 0.491 — a per-head maximum from a different epoch. A model that is
  running somewhere with no frozen record is a model whose numbers drift in retelling.

[RELEASE.md](RELEASE.md) has the mechanism and the three gates. The point here is only that
the mechanism existed, was correct, and was worth nothing until it was run once.

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
