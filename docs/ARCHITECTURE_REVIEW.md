# Architecture review

Measured answers to the architecture questions that keep coming up, so they do not get
re-litigated from intuition. Every number here was produced on this checkout against the
shipped checkpoint and the held-out test split — the scripts are described well enough to
re-run when the data changes.

Summary: **the architecture is not what limits this model.** One head is measurably
redundant, one obvious optimisation is a dead end, and everything else that is wrong is
data or measurement.

---

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
cannot. Under the annotation spec in [ANNOTATION_SETUP.md](ANNOTATION_SETUP.md) that does
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
>   verdict, which [ARCHITECTURE_DIRECTION.md](ARCHITECTURE_DIRECTION.md) §2 and
>   [PERSON_ATTRIBUTES.md](PERSON_ATTRIBUTES.md) both lean on, stands exactly as written.

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
   ([RETAIL_OBJECTS_SPLIT.md](RETAIL_OBJECTS_SPLIT.md) R3) — and the robot line has no site
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
