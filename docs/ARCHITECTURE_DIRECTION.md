# Architecture direction: the retail analytics programme

**Scope, and the boundary that makes this document safe to read.**
[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) reviews the **robot platform** — a
LiDAR-carrying machine whose model answers "can I drive here". Its verdicts stand as
written and nothing here overturns them. This document covers the **retail analytics
programme**: fixed store CCTV, no LiDAR, and a product ask ("零售分析需要年齡、性別、動作")
that the robot review never considered.

The two share a trunk, an export path and a repository. They do not share a consumer, and
several of today's stalls come from a decision correct for one being applied to the other.

---

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

> This does **not** contradict ARCHITECTURE_REVIEW.md's "Tracking / re-identification: do
> not build". That verdict is about the **exported graph**, where cross-frame state would
> break the no-dynamic-control-flow property. A host-side stage is a different object and
> the review's reasoning does not reach it.

---

## 3. Split the taxonomy by instrument, not by concept

`terrain` currently carries `void, floor, wall, column, fixture, product, person` — room
geometry, furniture, merchandise and agents in one head, at wildly different pixel scales,
from different sources, with different consumers.

`hydranet_retail_products.yaml` already made this argument for one class and made it
well: at stride 8 a boxed handset is **2–3 feature cells across**, so the segmentation head
"cannot draw its outline no matter how long it trains — there are not enough cells to put
an edge in." That is why `product` IoU sits near 0.1 while `fixture` reaches 0.72 on the
same frames.

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

[PERSON_ATTRIBUTES.md](PERSON_ATTRIBUTES.md) gets the ordering right — association before
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
