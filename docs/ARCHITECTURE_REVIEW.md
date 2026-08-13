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
- **`channels_last`.** Still unimplemented, and worth less than it looks: the training GPU
  runs at 93% utilisation against a 300 W power cap, so it is already saturated. Evaluate
  this on the Jetson, where the memory format actually matters, not on the workstation.

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

## Open questions, in the order worth answering

1. **Remove the traversability head?** Needs the deployment owner. Measured above.
2. **What is the field accuracy?** Unknown. Everything here is ADE20K, which is web
   photography, not robot-height footage of our sites. This is the largest unquantified
   risk in the project and no architecture change addresses it.
3. **Do the rare classes need loss reweighting rather than more data?** Currently
   unanswerable: three of them have zero examples, so there is nothing to reweight. Ask
   again after the first in-house annotation batch.
