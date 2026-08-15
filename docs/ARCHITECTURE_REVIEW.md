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
   construction. (`head_disagreement` now reports this every validation.)
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

| Head | Verdict | Reasoning |
|---|---|---|
| Terrain segmentation, 12 classes | **Keep, and make authoritative** | Material semantics are what LiDAR cannot provide. This is the model's irreplaceable output. |
| Detection | **Keep; narrow the class set at deployment** | Dynamic obstacles are a separate decision. See below. |
| Traversability | **Remove, derive instead** | Measured redundant in section 2. |
| Depth | **Do not build** | LiDAR already measures it, more accurately than a monocular head ever would. |
| Surface normal / slope | **Do not build** | Derived from the point cloud. |
| Confidence / uncertainty | **Do not build a head** | Temperature-calibrate the existing softmax instead. A new head is the wrong tool for a calibration problem. |
| Instance or panoptic segmentation | **Do not build** | Detection plus semantics already answers the decisions, at far lower cost. |
| Tracking / re-identification | **Do not build** | Cross-frame post-processing. Putting it in the graph would break the no-dynamic-control-flow property that makes TensorRT conversion work first time. |

**Fuse outside the network.** Project the point cloud into the image, take the semantic
label at each point, and return to robot frame. That is a deterministic operation and does
not need to be learned. Mid-level fusion (LiDAR as an input channel) and a BEV output head
are both plausible in principle and both wrong here: each demands calibration-aware
training data, and annotation throughput is already the binding constraint. Do not solve a
data shortage with an architecture that needs more data.

**On narrowing detection.** COCO's 80 classes cost more than parameters at inference: at
512×640 the class logits are 80 × 6,825 ≈ 546,000 values per frame to move and decode, and
roughly five of those classes change robot behaviour. Narrowing to ~8 is a 10× reduction in
post-processing traffic on a Jetson. Train on the full 80 — free auxiliary supervision for
the trunk — and narrow at export, rather than choosing one or the other.

---

## Open questions, in the order worth answering

1. **Remove the traversability head?** Needs the deployment owner. Measured above.
2. **What is the field accuracy?** Unknown. Everything here is ADE20K, which is web
   photography, not robot-height footage of our sites. This is the largest unquantified
   risk in the project and no architecture change addresses it.
3. **Do the rare classes need loss reweighting rather than more data?** Currently
   unanswerable: three of them have zero examples, so there is nothing to reweight. Ask
   again after the first in-house annotation batch.
4. **Can LiDAR resolve a 2–5 cm door sill on this platform?** It decides whether
   `threshold_ramp` and `stairs` stay low on the annotation priority list or move back up.
   One measurement on the real robot settles it; until then the ordering in
   [METHODOLOGY.md](METHODOLOGY.md) carries an assumption rather than a result.

---

## The COCO ratio sweep has two variables in it

Sweeping `sample_ratio` produced a sharp result on the held-out test split:

| | segmentation only | COCO 0.1 | COCO 1.0 |
|---|---|---|---|
| `glass` IoU | 0.5021 | **0.5462** | 0.0692 |
| terrain mIoU | 0.5718 | **0.5968** | 0.3591 |
| detection mAP | — | 0.1985 | **0.3348** |

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
