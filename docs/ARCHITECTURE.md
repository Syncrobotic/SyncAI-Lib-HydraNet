# Architecture: what is measured, and where it goes next

One file for both halves of the architecture question, because they were two and the
boundary between them was the first thing every reader had to reconstruct.

* **Part I — the measured verdicts** (was `ARCHITECTURE_REVIEW.md`). Questions that kept
  being re-litigated from intuition, answered with numbers. Written for the **robot
  platform**, and two of its verdicts have since been overturned — both marked in place.
* **Part II — the direction** (was `ARCHITECTURE_DIRECTION.md`). Where the **retail and
  security CCTV line** goes, the rules that decide when a number may be believed, and what
  deliberately does not get built.

They share a trunk, an export path and a repository. They do not share a consumer, and
several stalls have come from a decision correct for one being applied to the other. Where
Part II says something about a Part I verdict, it says which.

**Summary: the architecture is not what limits this model.** One head is measurably
redundant, one obvious optimisation is a dead end, and everything else that is wrong is
data or measurement.

---

# Part I — measured verdicts

> **Written for the quadruped, which is now the secondary line.** The verdicts stand where
> they are about the architecture rather than the platform — section 1's resolution ceiling
> and section 2's head redundancy are properties of the labels and the taxonomy, and both
> apply to the CCTV line unchanged. Two verdicts were overturned and say so in place. Read
> the platform-specific reasoning in section 5 with the LiDAR correction beside it.

Every number in this part was produced on this checkout against the shipped checkpoint and
the held-out test split — the methods are described well enough to re-run when the data
changes.

## 1. Higher segmentation resolution would not help

**Question.** The segmentation heads fuse at P3 (stride 8) and bilinearly upsample the
logits to full resolution, so the model can only express structure that survives a round
trip through 1/8 scale. Thin, safety-critical things — a door sill, a stair edge, a glass
frame — are exactly what that should destroy. Should the heads read P2 as well, or should
the neck keep C2?

**Method.** Take the validation annotations, one-hot them, downsample by the stride with
area interpolation, upsample back bilinearly, argmax. The IoU of that round trip against
the original is the **ceiling**: the best any model with this output resolution could
score, before any question of learning.

**Result.** Ceilings from 200 ADE20K val annotations; "current" from the shipped checkpoint
on the 329-image test split. The ceiling is a property of the label geometry rather than of
which split it was measured on, so the two are comparable.

> **Which checkpoint "shipped" means, re-checked 2026-08-19.** When this was written it was
> `runs/hydranet_indoor/best.pt` — epoch 27, **segmentation only, no detection head** (its
> test traversability mIoU 0.7055 is the "segmentation only" row of the ratio sweep at the
> foot of this file; that section's run-to-column table says which run is which, and warns
> that `coco10` means share **1.0**). Since `releases/v1` was cut, the model that is actually deployed is
> `runs/hydranet_joint_coco10` at **epoch 55, selected on `detection_mAP`**, and it is a
> different and much weaker segmenter: on its validation row `caution` and `stairs` are
> **exactly 0.0**, `glass` is 0.066, `door` 0.003, terrain mIoU 0.360 against the 0.5694
> that `hydranet_indoor` reached. **The "Current" column below has not been re-derived for
> it** — and it must not be, on this split, without re-running `hydranet-eval --split test`,
> because those v1 figures are validation and these are test.
>
> **Neither conclusion in this section moves.** The ceiling is a property of the labels, so
> it is checkpoint-independent, and every "headroom used" figure below becomes an *upper*
> bound for the deployed model rather than an estimate of it. "Resolution is not the
> constraint" is if anything better supported: the shipped model sits further from the same
> ceiling than the checkpoint measured here did.

| Class | Ceiling @ stride 8 | Ceiling @ stride 4 | Current | Headroom used |
|---|---|---|---|---|
| `wall` | 0.986 | 0.995 | 0.815 | 83% |
| `floor_hard` | 0.979 | 0.992 | 0.814 | 83% |
| `door` | 0.979 | 0.994 | 0.244 | 25% |
| `stairs` | 0.974 | 0.991 | 0.325 | 33% |
| `floor_soft` | 0.973 | 0.990 | 0.496 | 51% |
| `obstacle_furniture` | 0.971 | 0.989 | 0.700 | 72% |
| `glass` | 0.969 | 0.988 | 0.502 | 52% |
| **`person`** | **0.799** | 0.891 | 0.679 | **85%** |
| mean | 0.954 | 0.978 | — | |

**Conclusion. Resolution is not the constraint.** `glass` sits at 0.50 against a ceiling of
0.97; `stairs` at 0.33 against 0.97. Those gaps are learning and data, not geometry.
Moving to stride 4 raises the ceiling by 0.024 — headroom that is nowhere near being used —
and costs roughly 4× in segmentation-head compute on a Jetson budget.

The single exception is `person`, which is genuinely resolution-limited at 85% of its
ceiling because people are thin and vertical. It maps to `blocked`, which already scores
0.95, so the practical impact is nil.

**Do not spend effort here.** Revisit only if a class ever approaches its ceiling.

---

## 2. The traversability head is measurably redundant

**Question.** The traversability target is a deterministic function of the terrain target
via the policy table in `data/label_maps_indoor.py`. So traversability could be derived
from the terrain head at inference for zero parameters. Does the dedicated head earn its
277K parameters — 3.3% of the model?

**Method.** Run the shipped checkpoint on the 329-image test split and score three ways
against the same ground truth.

**Result** (80,041,239 labelled pixels):

| Method | Params | blocked | caution | go | mIoU |
|---|---|---|---|---|---|
| Dedicated head | **277K** | 0.9531 | 0.3252 | 0.8383 | **0.7055** |
| `argmax(terrain)` → policy table | 0 | 0.9514 | 0.3249 | 0.8341 | 0.7035 |
| Marginalise `P(terrain)` over groups | 0 | 0.9520 | 0.3223 | 0.8351 | 0.7031 |

**277K parameters buy 0.002 mIoU.** For scale, the val/test difference on this same
checkpoint is 0.03 — the head's contribution is an order of magnitude inside the noise.
Marginalising probabilities, which is the mathematically correct derivation, does not beat
plain argmax either, so the result is not an artefact of how the derivation was done.

**Three arguments for removing it:**

1. 3.3% of parameters and a head's worth of compute, for nothing measurable.
2. It removes a real inconsistency. The two heads currently contradict each other on
   **0.88%** of labelled pixels — the model saying "glass" and "walkable" about the same
   pixel — and nothing in the code prevents it. A derived output is consistent by
   construction. (`head_disagreement` now reports this every validation, and **it carries
   over to the deployed model**: `releases/v1/metrics.json` records 0.009083, reproducing
   this 0.88–0.91% on a different run entirely.)

> **Re-checked 2026-08-19.** The three-way table above was measured on the segmentation-only
> checkpoint (see §1's note) and **has not been re-run on `releases/v1`**, so "277K
> parameters buy 0.002 mIoU" is a within-checkpoint result that has not been reconfirmed on
> the shipped weights. `head_disagreement` is the one quantity that did carry over, and it
> is the one the argument leans on: the two heads remain near-deterministically related, so
> the redundancy verdict stands while its exact price tag is one checkpoint old.
3. **Policy changes become free.** The table is applied to *targets* during training, so
   changing it today means retraining. Derived, it is applied at inference: changing it is
   a config edit. Three entries in that table are marked `REVIEW` because they are
   *platform* decisions — whether the gait handles stairs, whether the foot fits through
   grating — and those change when the robot changes.

**One argument for keeping it:** if a future dataset ever labels traversability
*independently* rather than deriving it, the head would carry signal the terrain head
cannot. Under the annotation spec in [METHODOLOGY.md](METHODOLOGY.md) that does
not happen — traversability is always derived — so this is a hypothetical, not a current
requirement.

**Recommendation.** Strong candidate for removal, but it changes the ONNX output contract
and therefore the Jetson post-processing, so decide it with whoever owns deployment rather
than unilaterally. If it stays, add a consistency term to the loss so that 0.88% shrinks
instead of being merely observed.

---

## 3. What was checked and found sound

- **Partial supervision and loss balancing.** A head with no labels in the current batch is
  absent from the loss dict, so its uncertainty weight `s` receives no gradient. Verified
  in `UncertaintyWeighting.forward` and `HydraNet.compute_losses`. The 60-epoch run
  confirms it empirically: `task_weight/detection` sat at exactly 1.000000 across every
  sample while detection was unsupervised.
- **Shared-trunk premise.** 84.4% of parameters in the backbone and neck, 15.6% across
  three heads. The design claim holds and a fourth head really would cost 3–9%.
- **Operator set.** The forward graph remains Conv/BN/ReLU/Resize/MaxPool/Exp/Mul, with
  target assignment and NMS outside it. The TensorRT constraint has been respected
  consistently.
- **`channels_last`.** Built (`train.channels_last`, default off) and worth about **−21%**
  per training step: 153.2 → 121.1 ms at batch 48, 512×640, bf16 autocast on an RTX PRO
  6000, reproducible with `scripts/bench_channels_last.py`. Under fp32 it buys nothing
  (219.6 → 220.9 ms), because those convolutions run on CUDA cores, which are indifferent
  to layout; the config schema warns when the flag is set without AMP.

  **This entry previously argued the opposite and was wrong twice**, which is why it is
  kept rather than quietly rewritten. It said the GPU was already saturated at 93%
  utilisation against a 300 W cap, so there was nothing to win — but implicit transposes
  count as utilisation, and the saving is in work not done. It then said the layout
  changes no arithmetic: true of the strides, false of the run. NHWC selects different
  convolution kernels, which accumulate in a different order, and under the TF32 every run
  enables the two layouts disagree by ~1e-2 of scale (1e-5 with TF32 off — both pinned in
  `tests/test_channels_last.py`).

  That divergence is why the default is off rather than on. A run whose flag is flipped
  partway through is not bit-comparable to itself, and a control that exists to isolate one
  variable must not pick up a second one from a restart.

---

## 4. The finding that is not about architecture

The AMP crash in the FCOS loss (`0394057`) is worth recording here because of what it
implies rather than what it was.

The bug needed three conditions at once: AMP enabled, a CUDA-class device, and **the
detection head actually supervised**. Until COCO arrived, no run had ever called that loss.
So a line of code sat in the repository, through 166 passing tests and a complete 60-epoch
training run, having never once executed.

The lesson is that an unsupervised head is not merely a head with random weights — **its
entire loss path is unverified code**. That is a stronger reason for the export guard than
the one it was originally added for, and it is why
`tests/test_amp_detection_loss.py` forces autocast explicitly rather than trusting the
device to enable it: a regression test that silently no-ops in CI would repeat the same
failure.

---

## 5. Which heads should exist

The platform carries LiDAR, which settles the question that would otherwise dominate this
section.

> **⚠ Corrected 2026-08-19: it does not, and this is the premise two verdicts below rest
> on.** The robot is a **Lite3 with one monocular camera and two ultrasound returns**
> (`{forward, backward}`), whose forward echo was measured *lateral* — 0 of 29 frames
> comparable to the camera's cone. There is no point cloud, and "the Lite3 LiDAR variant"
> is one of the sensors [RESEARCH_OCCUPANCY.md](RESEARCH_OCCUPANCY.md) would justify
> **buying** if E-prep says so. Row by row:
>
> * **Terrain — unchanged.** Material semantics were never LiDAR's to provide, and with no
>   range sensor at all the case for making this head authoritative is stronger.
> * **Detection, Traversability — unchanged.** Neither argument mentions LiDAR.
> * **Depth — OVERTURNED.** `models/heads/depth.py` was built (`0f8f8ea`),
>   `runs/hydranet_nyu_depth` trained it, and it is head ⑤ of the occupancy direction. The
>   reasoning here ("LiDAR already measures it") had no referent.
> * **Surface normal / slope — the reasoning is void**, though nothing has been built. It
>   would now derive from the depth head's output rather than from a point cloud, which is a
>   different and unmeasured proposition; treat the verdict as unexamined, not as standing.
> * **Confidence, instance/panoptic, tracking — unchanged.** All three are argued from the
>   exported graph and from cost, not from the sensor suite. In particular the tracking
>   verdict, which Part II §2 and
>   [RETAIL_DATA.md](RETAIL_DATA.md) both lean on, stands exactly as written.

| Head | Verdict | Reasoning |
|---|---|---|
| Terrain segmentation, 12 classes | **Keep, and make authoritative** | Material semantics are what LiDAR cannot provide. This is the model's irreplaceable output. |
| Detection | **Keep; narrow the class set at deployment** | Dynamic obstacles are a separate decision. See below. |
| Traversability | **Remove, derive instead** | Measured redundant in section 2. |
| Depth | ~~**Do not build**~~ — **built** | Said "LiDAR already measures it, more accurately than a monocular head ever would". There is no LiDAR; the head exists. |
| Surface normal / slope | **Do not build** — reasoning void | Said "derived from the point cloud". There is no point cloud. Unexamined rather than settled. |
| Confidence / uncertainty | **Do not build a head** | Temperature-calibrate the existing softmax instead. A new head is the wrong tool for a calibration problem. |
| Instance or panoptic segmentation | **Do not build** | Detection plus semantics already answers the decisions, at far lower cost. |
| Tracking / re-identification | **Do not build** | Cross-frame post-processing. Putting it in the graph would break the no-dynamic-control-flow property that makes TensorRT conversion work first time. |

**Fuse outside the network.** Project the point cloud into the image, take the semantic
label at each point, and return to robot frame. That is a deterministic operation and does
not need to be learned. Mid-level fusion (LiDAR as an input channel) and a BEV output head
are both plausible in principle and both wrong here: each demands calibration-aware
training data, and annotation throughput is already the binding constraint. Do not solve a
data shortage with an architecture that needs more data.

> **2026-08-19: the first half has nothing to fuse.** With no point cloud, the cheap
> deterministic route does not exist. The *warning* in the second half is the part that
> survives and it is now load-bearing rather than theoretical — a BEV output head is exactly
> what [RESEARCH_OCCUPANCY.md](RESEARCH_OCCUPANCY.md) proposes (step E1), and that document
> agrees the binding constraint is supervision: its whole first milestone is auto-labelling,
> because it accepts that hand-drawn 3D data is unaffordable. The two documents disagree
> about the conclusion and agree about the constraint, which is the honest state of it.

**On narrowing detection.** COCO's 80 classes cost more than parameters at inference: at
512×640 the class logits are 80 × 6,825 ≈ 546,000 values per frame to move and decode, and
roughly five of those classes change robot behaviour. Narrowing to ~8 is a 10× reduction in
post-processing traffic on a Jetson. Train on the full 80 — free auxiliary supervision for
the trunk — and narrow at export, rather than choosing one or the other.

---

## Open questions, in the order worth answering

1. **Remove the traversability head?** Needs the deployment owner. Measured above.
2. **What is the field accuracy?** Still unknown *for the robot*. Everything here is
   ADE20K, which is web photography, not robot-height footage. The retail line has since
   built site splits and found the next problem behind this one — a number scored against
   SAM 3 pre-labels is an *agreement*, not an accuracy
   ([RETAIL_DATA.md](RETAIL_DATA.md) R3) — and the robot line has no site
   data at all until `robot_capture.py`'s clips are annotated. Still the largest
   unquantified risk, and still not an architecture question.
3. **Do the rare classes need loss reweighting rather than more data?** Currently
   unanswerable: three of them have zero examples, so there is nothing to reweight. Ask
   again after the first in-house annotation batch.
4. ~~**Can LiDAR resolve a 2–5 cm door sill on this platform?**~~ **Void — there is no
   LiDAR** (2026-08-19). It has been replaced by a harder question with the same
   consequence: **how a monocular depth teacher behaves below 0.7 m on this lens.** Public
   data cannot bound it — NYUv2's returns start near 0.7 m and the robot's forward cone puts
   the floor at 0.34 m — so only the robot can answer, and only once it walks.
   `threshold_ramp` and `stairs` have **already moved back up** the
   [METHODOLOGY.md](METHODOLOGY.md) list in the meantime, because nothing else on the
   platform measures a level change.

---

## The COCO ratio sweep has two variables in it

Sweeping `sample_ratio` produced a sharp result on the held-out test split:

| | segmentation only | COCO 0.1 | COCO 1.0 |
|---|---|---|---|
| `glass` IoU | 0.5021 | **0.5462** | 0.0692 |
| terrain mIoU | 0.5718 | **0.5968** | 0.3591 |
| detection mAP | — | 0.1985 | **0.3348** |

### Which run is which column, and the naming trap in between

Recorded 2026-08-19 because reading a run's ratio off its *name* nearly manufactured a
contradiction that does not exist. Every figure below is `sample_ratio` on the coco dataset
block of that run's own `config.yaml`:

| sweep column | run | COCO `sample_ratio` | `primary_metric` |
|---|---|---|---|
| segmentation only | `runs/hydranet_indoor` | — (no coco block) | `traversability_mIoU` |
| **COCO 0.1** | `runs/hydranet_indoor_det60` | **0.1** | `traversability_mIoU` |
| COCO 0.3 (in the four-point sweep) | `runs/hydranet_joint_coco03` | **0.3** | `detection_mAP` |
| **COCO 1.0** | `runs/hydranet_joint_coco10` | **1.0** | `detection_mAP` |

> **⚠ `coco10` means share 1.0, not 0.10** — and `coco03` means 0.3, so the two names are
> not even on the same scale as each other. Reading them as decimals puts the strongest
> segmenter and the weakest one in each other's places, which is the exact shape of error
> this repository keeps paying for: a record consulted about something it does not encode.
> **Read `config.yaml`, never the directory name.**

The apparent contradiction it resolves: `hydranet_joint_coco10`'s validation `glass` never
reaches the segmentation-only run's 0.4981 at *any* epoch, peaking at 0.4221. Under the
misreading that looks like the "COCO 0.1 costs nothing" result failing to reproduce. It is
not — that run **is** the 1.0 corner, the sweep's worst segmenter by construction. Two
independent cross-checks confirm the mapping: `releases/v1` (= this run, epoch 55) reports
`detection_mAP` **0.33478**, which is the 1.0 column's 0.3348, and its `glass` 0.066 sits on
the 1.0 column's 0.0692 rather than anywhere near 0.5462.

**Which means `releases/v1` ships the 1.0 corner** — the best detector and the weakest
segmenter this sweep produced — while the sweep's own recommendation for retail is to stay
at 0.1. That is not a mistake: v1's `primary_metric` is `detection_mAP`, so it selected
exactly what it was asked to. It is a trade-off a reader should be able to see, and
[RELEASE.md](RELEASE.md) now names it beside the release table.

The natural reading is dilution: at ratio 1.0 the segmentation heads receive about a
quarter of the optimiser steps and the rare classes starve. **That reading is not
established, because the sweep moved two things at once.**

With Kendall uncertainty weighting the model learns a `log_var` per head, and each head's
loss is scaled by it. Change the sampling ratio and every head's `log_var` converges
somewhere else — so the ratio changes the *effective learning rate per head* as well as
the step counts. Whether glass fell from 0.502 to 0.069 because the segmentation heads saw
fewer steps, or because the balancer handed the trunk's gradients to detection, cannot be
read off these numbers.

### The control, and what each outcome would mean

Rerun ratio 1.0 with `model.loss_balancing: fixed` and equal weights, changing nothing
else. One run, at the point of maximum effect, rather than a fixed-weight sweep across
every ratio:

| Outcome at ratio 1.0, fixed weights | Reading |
|---|---|
| glass still collapses | **Dilution.** Step count is the mechanism; the balancer is a bystander. The fix is sampling, and the current 0.1 is right. |
| glass survives | **The balancer.** Uncertainty weighting is amplifying the imbalance, and the fix is a floor on the segmentation weights — which would recover detection mAP *and* glass, rather than trading them. |
| glass partly recovers | Both, and the split is now quantified. |

Cost is about 8 hours on an RTX PRO 6000 (measured: the ratio-1.0 run took 22:00→06:08
exclusive, and a second run on the same card costs the pair roughly 55% each).

> **Status: started, stopped at epoch 6 of 60, no conclusion.** `runs/hydranet_fixed_coco10`
> holds a resumable checkpoint (optimizer and schedule included) and
> `hydranet-overnight/resume_fixed.sh` restarts it. **Its numbers at epoch 6 answer
> nothing** — glass was 0.4199 there, and the collapse this control exists to explain only
> became visible late in the ratio-1.0 run. Read that directory as an experiment in
> progress, not a result; a half-finished run left on disk without a note is worse than no
> run at all, because the next person finds numbers and no warning.

The result matters beyond this project's tuning: if it is the balancer, then every
multi-task config here — indoor, retail, and whatever comes next — is carrying a knob that
silently reallocates capacity whenever a dataset's size changes.

---

# Part II — direction: the retail and security CCTV programme

**Scope.** This part covers the line the project's stated goal now names first: fixed store
CCTV, no LiDAR, and a product ask ("零售分析需要年齡、性別、動作") that Part I's robot review
never considered. Nothing here overturns a Part I verdict; where one has been overturned,
Part I says so itself.

## 1. The diagnosis: the bottleneck is the instruments, not the models

On 2026-08-17 the session board counted **five** occasions where a conclusion turned out
to be a property of the *instrument* rather than of the thing measured. A sixth was found
the same evening. The list is worth reading as one thing rather than six:

| what looked true | what it actually was |
|---|---|
| `column` scores 0.51 on val | 0.51 over **0.66%** of val pixels, 22 of 285 images — and **0.00%** on site |
| `product box` finds 0 instances on accessory walls | a property of 352×240 and **one prompt of eight**; at 1920×1080 it is 33% of pixels |
| `export_onnx.py` coverage regressed 90% → 54% | the file **grew 205 → 209 statements mid-run** as another session committed |
| `column` is on 4 cameras | **4-of-14 measured, reported as 4** — the unmeasured ten held a 15.11% |
| `best.pt` held a value on the way down | the run selected on a different key than assumed |
| a class-collision check at cosine ≥ 0.95 | unrelated CLIP text scores **0.72–0.74, not 0** — the check would never once have fired |

Every one is the same shape: **the record consulted was produced by the thing it was being
used to check.** A `git status` is not evidence about your own past. An mtime you can
rewrite is not authorship. A fixture built from the tuples it validates resolves a typo
against its own reflection. A threshold measured in the wrong space never fires.

> **The rule, and it belongs here rather than in a journal file that gets deleted:**
> before attributing a number to a mechanism, check what produced the number. Not whether
> it is right — what produced it.

This is a *process* gap, not a modelling one, and no architecture change fixes it. Section
4 says what does.

---

## 2. HydraNet should stop growing, and the second stage needs a name

`HydraNet.forward(images)` is a **single-frame pure convolution graph**, and its docstring
says so: "This is exactly what gets exported to ONNX." That property is why TensorRT
conversion works first time, and it is worth more than any feature that would break it.

Everything the analytics roadmap wants is therefore host-side: attributes, action, track
association, re-identification, temporal smoothing. **None of them can be a HydraNet head**,
for two independent reasons:

* **ROI breaks head independence.** `hydranet.py` states the invariant — heads read only
  the neck's feature list, never another head's output — and the `Head` protocol's
  `forward_into(out, feats, size)` enforces it in the signature. An attribute or action
  head needs the *detection head's decoded boxes*. There is no way to express that.
* **Time is not in the signature at all.** `forward` takes one `[B,3,H,W]`. "Reach to
  shelf" and "retract from shelf" are the same posture in two directions; no single frame
  separates them.

**The problem is that the second stage already exists and has no name.** Four components
of one system, in three directories, sharing no contract:

| component | lives in | what it actually is |
|---|---|---|
| `analytics/tracker.py`, `analytics/dwell.py` | `analytics/` | second stage |
| `utils/temporal.py` | `utils/` | second stage |
| `scripts/retail_flow.py` | `scripts/` | second stage |
| the crop encoder, when it is built | — | second stage |

The costs are already visible and were paid today: `temporal.py` reached 17% coverage
owned by nobody, and `retail_flow.py`'s output — the only independent answer to "was this
pixel occupied at frame N" — lives in a **session scratchpad** that disappears when the
session ends.

**Give it a contract**: frames plus HydraNet outputs in, tracks plus per-track attributes
out; its own small ONNX export for the crop encoder; its own tests; its own metrics. Until
that exists, every new capability re-invents the interface and none of them agree.

> This does **not** contradict Part I's "Tracking / re-identification: do not build". That
> verdict is about the **exported graph**, where cross-frame state would break the
> no-dynamic-control-flow property. A host-side stage is a different object and the
> verdict's reasoning does not reach it.

---

## 3. Split the taxonomy by instrument, not by concept

`terrain` currently carries `void, floor, wall, column, fixture, product, person` — room
geometry, furniture, merchandise and agents in one head, at wildly different pixel scales,
from different sources, with different consumers.

`hydranet_retail_products.yaml` already made this argument for one class and made it
well: at stride 8 a boxed handset is **2–3 feature cells across**, so the segmentation head
"cannot draw its outline no matter how long it trains — there are not enough cells to put
an edge in." That is why `product` IoU sits near 0.1 while `fixture` reaches 0.72 on the
same frames. Part I §1 measured the same limit from the other side: the round-trip ceiling
at stride 8, which no class here is anywhere near using.

> ### ⚠ CORRECTED, and the first draft of this section would have caused a safety
> ### regression if anyone had acted on it
>
> The first version of this section extended that reasoning to `person` as well —
> "the detection head is already finding people", so move it out of segmentation. **That
> is wrong, and it is wrong in the dangerous direction.** Checked before acting, which is
> the only reason it is a correction rather than an incident.
>
> `cli/scene.py:161` **derives** free space from the terrain map through
> `terrain_to_traversability`, because the object configs drop the traversability head on
> purpose (it was a lookup, not a second signal). `RETAIL_OBJECTS_TO_TRAV` maps
> `person → blocked`. So the segmentation `person` channel is not decoration — it is what
> puts a person into free space at all.
>
> The right question is not "which instrument draws this class better". It is:
> **what does free space become if this channel disappears?**
>
> | class | pixels fall back to | free space becomes | verdict |
> |---|---|---|---|
> | `product` | `fixture` — merchandise sits on the thing that holds it | `blocked` either way, **unchanged** | safe to move to detection |
> | `person` | `floor` — a shopper stands on it | a person-shaped hole of **walkable floor** | **must stay in segmentation** |
>
> The asymmetry is the whole answer, and it is a property of what each class sits *on*.
> `tests/test_scene_derived_trav.py` states the consequence in its own docstring: a wrong
> guess here "draws a floor polygon over a wall — confidently, in metres, with no error."
> A floor polygon drawn over a shopper is the same failure aimed at a person.

**Ask which instrument answers the question before asking what class it is — and then ask
what reads the channel:**

| class | instrument | why |
|---|---|---|
| `floor`, `wall`, `column`, `fixture` | segmentation | large, edge-defined, mostly static |
| `product` | detection | measured unrenderable at stride 8; free space unaffected |
| `person` | **both** | detection feeds the second stage; segmentation feeds free space |

`person` in both heads is not redundancy to be optimised away. The two answer different
questions — *where can the robot drive* and *which shopper is this* — and only one of them
is allowed to be wrong. `head_disagreement` was the right check for terrain versus
traversability, where one **was** a deterministic function of the other; it is not the
right check here, because neither of these is derivable from the other.

What does still hold: classes keep coming back empty, and `unsourced_terrain_classes` now
names four such channels across the shipped configs. That is a sourcing problem, and
moving a class to a head that has no data for it does not fix it.

---

## 4. Make the measurement apparatus a reviewed artefact

Half of this landed on 2026-08-17 (`support/` in `metrics.jsonl`,
`unsourced_terrain_classes` wired into the validator). Three rules make it a discipline
rather than a good day:

1. **Any number that selects a model or leaves the building carries its support.**
   `column`'s 0.51 stood on 22 images and 0.66% of pixels and was formatted identically to
   `wall`'s 283 images and 61.86%.
2. **Any threshold is relative to a measured baseline, never an absolute.** The CLIP
   collision check is the worked example: unrelated words score 0.72–0.74 in that space,
   so an 0.95 absolute cut admits a 0.9458 collapse and fires on nothing, ever. Pin the
   *relationship*, as `tests/test_temporal.py` pins `diff_thr * plate_alpha` rather than
   0.24.
3. **Any site claim names the camera count it was measured over.** A partial sweep reads
   exactly like a complete one.
4. **A detection comparison runs three seeds, or it reports nothing.** Measured below, and
   it is the most expensive rule here because it triples the GPU cost of an answer. It is
   also the one that would have saved the most wasted argument.

   **Amended 2026-08-18: it says *detection* for a reason, and the reason was found by
   generalising it anyway.** The rule is affordable only because seed-pairing works on
   `detection_mAP` — paired sd **0.0034** against an unpaired **0.0098**, five times
   tighter, which is what lets n=3 resolve anything at all. That structure is
   metric-specific and it does **not** hold on `terrain_mIoU/site_seg`. Same three seeds,
   the surfaces -> security contrast:

       seed 42  0.7545 -> 0.6623   -0.0922        unpaired sd   0.0214 / 0.0196
       seed  7  0.7228 -> 0.7013   -0.0216        paired sd     0.0373
       seed 13  0.7637 -> 0.6857   -0.0780

   **Pairing makes it worse**, so n=3 buys a minimum detectable effect of ~0.061 unpaired
   and ~0.111 paired, against segmentation effects this project actually argues about of
   0.003 to 0.085. Six 60-epoch runs were queued on 2026-08-18 under this rule and killed
   once that was computed: at ~5 h of GPU they could not have resolved any of the three
   questions they were asked.

   The likely mechanism is worth naming because it is fixable and seeds are not: a run's
   reported score is `max` over 60 epoch validations, which is an **order statistic**, and
   the epoch it lands on differs per arm (E16, E30, E11 on three runs here). Whatever the
   seed contributes in common mode is scattered by taking the max, so there is nothing left
   for pairing to cancel. Detection pairs because both arms share most of their trajectory;
   these two share neither dataset nor loss.

   So the rule generalises as: **measure the metric's own noise floor, paired and unpaired,
   before buying seeds to beat it.** Three seeds is not a standard, it is an answer that was
   correct once for one metric. And when the floor turns out to be above the effect, seeds
   are the wrong purchase — a larger val set, or a selection rule that is not an argmax over
   epochs, changes what any future run can resolve; more seeds only re-measure the same
   ceiling.
5. **A run whose *length* is decided by a metric is not comparable to a run of the same
   config with a different selecting metric.** Measured 2026-08-18 and it cost a whole
   comparison. `early_stop_patience: 10` is set in `hydranet_retail_products.yaml` and
   inherits down the entire retail chain, so it applies to configs written months later
   by people not reading that file. Under `primary_metric: detection_mAP/site_boxes` the
   control runs stopped at **E48** and **E50**; changing selection to
   `terrain_mIoU/site_seg03` -- a 48-image val set, where ten non-improving validations is
   a low bar -- stopped the very next run at **E19** of 60. Two runs trained to different
   lengths, and nothing in either log says the lengths differ for a reason unrelated to
   the thing being compared.

   The general shape is worse than early stopping, which is only its most visible
   instance: **any mechanism that reads the selection metric can silently change what a
   run *is*.** Selection decides which checkpoint survives, which is intended; it must not
   also decide how long the run trains, which epochs get validated, or which val sets are
   scored. `_check_detection_val_interval` refuses one such coupling -- thinning detection
   validation while selecting on a detection metric kills the run at the end of epoch 1 --
   and that check exists because the coupling was found the expensive way.

   The rule for a comparison: fix the length in the config, disable early stopping, and
   let selection choose only the checkpoint.

### The variance nobody had measured, and what it invalidates

`configs/hydranet_retail_products.yaml`, three runs, **identical in every respect except
`seed`**:

| seed | best `detection_mAP` |
|---|---|
| 42 | 0.0655 |
| 7 | **0.0821** |
| 13 | 0.0648 |

mean **0.0708**, range **0.0648–0.0821**, spread **0.0173**, sd 0.0098.

**Two numbers describe this and they are not the same number.** Relative standard deviation
is `0.0098 / 0.0708` = **14%**; range over minimum is `0.0173 / 0.0648` = **27%**. Both are
correct and other documents quote the second. Named here because a reader meeting 14% in
one file and 27% in another will reasonably suspect one of them is wrong, and the cheapest
way to lose a measurement is to make it look contested. Use the sd for "how much does a run
wobble" and the range for "how far apart can two runs land", which is the question a
single-run comparison is actually asking.

**What it was run to settle, and did.** The open-vocabulary control scored 0.0572 against
the linear head's 0.0655, and that 0.0083 gap was reported here as a ~13% cost. The
control's own seed spread is **1.3× that gap**. The comparison is therefore *undecided*,
not adverse, and the earlier reading is withdrawn. Deciding it needs three seeds on the
open-vocabulary side too.

**What it invalidates is much wider than one comparison.** Every detection number this
project has produced is a single run — the towers-at-half-depth control, the export
narrowing evaluation, every "which checkpoint is better" choice. **All of them now have to
be read against ±0.017**, and a difference under about a tenth of the mean is not readable
at all.

The uncomfortable part is that this was cheap. Two extra runs, 26 minutes, on a GPU that
was idle. The reason it had never been done is not cost; it is that a single run produces a
number, and a number does not look like it is missing anything.

### The rule's first use, and it changed the claim

Section 3's taxonomy split was measured this way rather than on one run — seeds **matched**
to the control's, so the same data order and init apply and the taxonomy is the only
difference:

| seed | products | surfaces | Δ |
|---|---|---|---|
| 42 | 0.0655 | 0.0704 | +0.0050 |
| 7 | 0.0821 | 0.0901 | +0.0080 |
| 13 | 0.0648 | 0.0659 | +0.0011 |

Paired mean **+0.0047**, paired sd 0.0034, **95% CI −0.0038 .. +0.0132** on 2 df. All three
deltas are positive, which is suggestive and is not a result: the interval includes zero.

**The supportable claim is the negative one — removing `product` from the dense head did
not cost detection mAP.** That is enough to justify the split, because the split was never
argued from mAP; it was argued from `product` being unrenderable at stride 8 and from free
space being unchanged. What the seeds do is remove the objection that it might have cost
something elsewhere.

Note also what pairing bought. The between-seed spread is 0.0173 and the paired sd is
**0.0034** — five times tighter, because matching the seed cancels the run-to-run term
instead of averaging over it. Three paired runs answer a question three unpaired runs
cannot.

**And one measurement that looked like the strongest evidence and is not.** `fixture` IoU
rose about +0.20. It is unusable: the migration merges `product` into `fixture`, making it
a **1.37× larger class**, so most of that is definitional rather than a better model. An
earlier version of this section promised `fixture` as the within-run comparison that would
be readable from a single run. It is not readable at all without a class-size correction,
and the size change is the first thing to check whenever a taxonomy edit is followed by a
per-class improvement.

---

## 5. Build the re-identification *metric* before the re-identification *model*

[RETAIL_DATA.md](RETAIL_DATA.md) gets the ordering right — association before
attributes, because a 4.6-minute clip fragments into 1234 tracks and attributes computed
over that produce a demographic report whose denominator is fragments, biased in exactly
the direction dwell analytics care about.

One step goes before even that: **this repository has no ID-switch or track-length
metric.** Without one there is no way to know whether a crop encoder helped. Attribute
accuracy is not a proxy — under 9–16-frame fragmentation the two are unrelated.

This is the same shape as `unsourced_classes()` shipping with a docstring, tests, and no
caller: the mechanism was built and the thing that would show whether it worked was not.

**Landed in `078f0e9` as `analytics/reid_metrics.py`, and the half that is live has a
reading rather than only tests.** An ImageNet resnet18 with no re-identification training
scores **mAP 0.0318 / rank-1 0.0962 / rank-5 0.1986 / rank-10 0.2699** over Market-1501's
standard 3,368-query, 19,732-gallery protocol. That is the floor any association encoder
has to clear, and a metric with a number on the board is a different object from a metric
with a green test.

**The tracking half is armed and blocked, and the blocker is worth naming rather than
leaving implied: `idf1` and `id_switches` need ground-truth tracks, and no site clip is
labelled.** So the next thing on this line is not a model and not a metric — it is **one
labelled clip**. Until it exists the tracker's 1234 fragments cannot be scored at all, only
described.

Two things that follow, both from having been inside `tracker.py` rather than reasoning
about it from outside:

* **Part of §2's contract already exists.** `Track` carries `frames` and `boxes`, so a
  metric consumes it without touching the tracker. The contract is *undocumented*, not
  absent — which is a smaller job than §2 implies and a different one: write down what is
  already true before adding to it.
* **`iou_threshold=0.5` in `idf1`/`id_switches` is an absolute threshold**, of exactly the
  kind rule 2 above warns about. It is defensible where the CLIP cutoff was not — 0.5 is
  the published MOT convention, so a number computed at it is comparable to the
  literature, and it is a keyword argument a caller can sweep. What cannot be said is that
  0.5 is right *for these cameras*, because measuring the IoU distribution needs the same
  labelled clip. That assumption is recorded in the docstring rather than left silent,
  which is the most this rule can ask for while the measurement is impossible.

---

## 6. Open-vocabulary detection is the highest-leverage change on the table

Not because it fixes `book`-for-handset, but because it **decouples the vocabulary from
the weights**. Every later line inherits that: per-store product lists, attribute prompts,
export narrowing as a row slice rather than a retrain.

**One ordering constraint, which nothing in the code enforces and which will otherwise
cost a run.** `text_classifier.py` promises "per-store vocabularies without retraining".
That is a property of the **training run**, not of the head. The `text_embeddings` buffer
ships as a random *orthogonal placeholder*, and `embed_pred` learns alignment with
whatever matrix was installed while the gradient flowed:

```
train with placeholder -> swap in real embeddings  = noise; the projection points at
                                                     random directions
train with a real matrix -> swap for another one   = works; both live in one space
```

So the matrix is built **before** training and named in the config that trains the head. A
matrix installed at export time onto a placeholder-trained head produces confident,
meaningless boxes — no error, plausible output, the failure mode this project ranks worst.

---

## 7. Stop sharing one checkout

Measured cost on a single day, all from one working tree shared by six sessions: a tracked
file deleted out from under a running test suite, **four** authorship errors (the fourth
made by a session that had read the first three), and twenty minutes chasing a coverage
regression that was a file growing mid-run.

`git worktree add` per session is the fix and the board named it before any of that
happened. Moving a session *mid-flight* is itself risky — untracked work has to move with
it — so the practical rule is: **each new session starts in its own worktree**, and the
shared tree is for whoever holds the branch.

**And one mechanism that makes the authorship problem worse than it looks: mtime in this
tree is a record of the last commit that ran, not of who wrote what.** The pre-commit hook
stashes unstaged work and restores it a second later, and the restore rewrites the
modification time. So *every commit by anyone re-stamps everything anyone else has in
flight*, and two files edited fourteen hours apart can carry mtimes one second apart.

This nearly produced a fifth authorship error: `label_maps.py` carried a 09:32:42 mtime
against its author's own configs at 20:09, which reads as a different session's work. 09:32:41
was another session's commit, and the pre-commit patch files hold exactly those two files.

The board's standard already said an mtime is not evidence. This is *why*, and the why
matters because "not evidence" invites a reader to weigh it a little anyway. It carries no
information about authorship at all.

---

## What not to build

* **No ROI or temporal head in HydraNet.** Section 2. Host-side is cheaper and does not
  cost the single-frame graph.
* **No fine-grained age.** The head region measures **31–42 px**. Any pipeline routing age
  through a face detector reports confidently on noise. Specify age as coarse bands.
* **No MERL-style action annotation before confirming those actions occur here.** MERL
  Shopping is a grocery aisle of closed shelving; these stores are open display tables.
  "Reach to shelf" may not be the dominant behaviour, and finding that out after the
  annotation spend is the `column` failure repeated at a higher price.
* **Do not raise the `ty_ratchet.sh` baseline from 15 to 19.** Its docstring justifies the
  loose number by claiming the debt is "almost all torch and pycocotools stub gaps".
  Measured: 10 of 17 are in `data/transforms.py` and `data/datasets.py`, which is this
  repository's own code. Raising it blesses debt the comment says is not there.

---

## Sequence

| phase | work | why here |
|---|---|---|
| **0** | worktree isolation; settle `utils/temporal.py`; open-vocab wiring **and one training run** | first two are bleeding; the third is the precondition for everything downstream |
| **1** | split the taxonomy by instrument; name the second stage and give it a contract | decides where every later model lives |
| **2** | the re-ID **metric**, then track association | metric before model |
| **3** | attributes on the same encoder; confirm RAP v2's licence and re-ID subset first | the denominator has to be people |
| **4** | action | needs 2 and 3 to hold |

**Reversing 2 and 3 produces a demographic report whose denominator is fragments, and it
will look finished.** That is the single most expensive mistake available on this roadmap,
because unlike a wrong number it does not read as wrong.
