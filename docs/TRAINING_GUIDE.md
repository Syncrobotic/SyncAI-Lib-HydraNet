# How to train a vision multi-head network

*An internal walkthrough, using SyncAI-Lib-HydraNet as the worked example. Written for
everyone who touches the perception stack — you do not need an ML background to follow it,
but you do need to care about why the robot walks into a glass door.*

Reading time: about 10 minutes.

---

## 1. What we are actually building

A quadruped walking through a lobby needs three different questions answered about the same
camera frame, at the same instant:

| Question | Output shape | Called |
|---|---|---|
| Can I put a foot here? | one label per pixel, 3 classes | traversability |
| What is this surface made of? | one label per pixel, 12 classes | terrain |
| Where are the people and objects? | a list of boxes | detection |

The first two are **segmentation**: the answer is a picture the same size as the input, where
every pixel carries a class. The third is **detection**: the answer is a variable-length list
of rectangles, each with a class and a confidence.

That difference matters more than it looks. Segmentation answers "what is *here*" for every
location, which is what you need to plan a footstep. Detection answers "how many of *those*
are in frame and where", which is what you need to avoid a person. Neither substitutes for
the other: a segmentation map does not tell you there are three people rather than one blob
of person-coloured pixels, and a box around a person tells you nothing about whether the
floor under them is wet.

### How to read the scores

Two numbers show up constantly, and it is worth knowing what they mean before they appear in
a review:

- **IoU** (intersection over union), per class: of all the pixels that are *either* predicted
  as `glass` or actually *are* `glass`, what fraction were both? 1.0 is perfect, 0.0 means no
  overlap at all. **mIoU** is the mean of that across classes.
- **mAP** for detection: roughly, how well the boxes are ranked and placed. Same direction —
  higher is better.

The trap in mIoU is that it is a *mean over classes*, not over pixels. In an indoor frame,
floor and wall are most of the image and `glass` might be 1% of it. A model that gets glass
completely wrong loses very little mIoU, because glass is one class out of twelve regardless
of how few pixels it holds. **That is exactly backwards from what we care about** — glass is
the most dangerous class we have, because it reads as an open corridor. This is why we look
at per-class IoU curves, not just the mean, and why `primary_metric` can be pointed at a
single class.

---

## 2. Why one network instead of three

The naive approach is three models: one per task. It works, and we did not do it.

The reason is the shape of the cost. Almost all of the computation in a vision network goes
into extracting general-purpose features — edges, textures, materials, shapes. Those features
are not task-specific. The parts that *are* task-specific are small.

Measured on our current model:

| Component | Role | Parameters | Share |
|---|---|---|---|
| RegNetX-800MF backbone | shared | 6,586,656 | 79.2% |
| BiFPN ×2 neck | shared | 435,686 | 5.2% |
| Detection head (FCOS) | task | 738,618 | 8.9% |
| Terrain head | task | 278,028 | 3.3% |
| Traversability head | task | 277,443 | 3.3% |
| **Total** | | **8,316,434** | |

**84.4% of the network is shared; the three heads together are 15.6%.** Three separate models
would mean paying that 84.4% three times, and running it three times per frame on a Jetson
with a fixed power budget.

The architecture that follows from this is the "HydraNet" shape — one trunk, many heads:

- **Backbone** (RegNetX): consumes the image, emits features at four resolutions — 1/4, 1/8,
  1/16, 1/32 of the input, getting deeper and more abstract as they get smaller.
- **Neck** (BiFPN): mixes information *across* those resolutions, so the fine-detail level
  knows what the coarse level saw, and vice versa. Out come five levels, P3 (1/8) through
  P7 (1/128), all 96 channels wide.
- **Heads**: each reads the levels it needs and nothing else. The segmentation heads take
  P3–P5 (the high-resolution ones — they must produce a full-size mask). The detection head
  takes all five, because objects come in wildly different sizes and each level specialises
  in a size range.

The rule that keeps this maintainable: **heads never read each other's output.** Only the
neck's features. That means a head can be added, removed or retrained without touching the
others, and a fourth head (depth, say) costs 3–9% more parameters rather than another 100%.

---

## 3. The hard part is not the architecture

Architectures are the easy half. You can copy one from a paper in an afternoon. What actually
consumes the schedule is that **no single dataset labels all three tasks.**

We have segmentation labels for indoor scenes (ADE20K), segmentation labels for off-road
(RUGD, RELLIS-3D), and detection labels for everything (COCO). Not one of them labels both.
There is no dataset where the same image has a traversability mask *and* boxes.

The standard wrong answer is to only train on data labelled for everything, which here means
training on nothing. The approach that works is **partial supervision**:

> Each training step draws a batch from *one* dataset, runs the full network, and
> backpropagates only the losses for the heads that dataset supervises.

A COCO batch produces a detection loss; the segmentation heads produce output that is simply
not scored, so no gradient flows into them. An ADE20K batch does the reverse. The shared
trunk receives gradient from *both*, which is the entire point — it gets to learn from all
the data, while each head only learns from data that actually answers its question.

In the config this is one line per dataset:

```yaml
- name: coco
  supervises: [detection]
- name: ade20k
  supervises: [traversability, terrain]
```

Two consequences to keep in mind:

1. **Sampling ratio is a real knob.** Steps per epoch is `Σ len(loader_i) × ratio_i`. COCO has
   ~117,000 images; at ratio 1.0 it will dominate every epoch and the segmentation heads will
   see comparatively little. We run COCO at a reduced ratio, and each epoch takes a different
   random slice, so nothing is permanently discarded.
2. **A head with no dataset is silently useless.** If nothing supervises detection, the head
   still gets built, still gets exported to ONNX, and still outputs numbers — they are just
   the initial random weights. The config check warns about this at startup, and you should
   read that warning rather than scroll past it. Right now, on the indoor config, this is
   genuinely the case until COCO is downloaded.

### One annotation, two heads

Here is a trick worth stealing. We do not annotate traversability separately. Terrain labels
already contain the information — if a pixel is `wet_slippery`, whether it is walkable is a
*policy decision*, not a new labelling job. So we keep an explicit mapping table:

```python
INDOOR_TERRAIN_TO_TRAV = {
    1: 2,   # floor_hard      -> go
    4: 1,   # wet_slippery    -> caution
    8: 0,   # glass           -> blocked
    ...
}
```

One annotation pass, two supervised heads. And when the platform changes — a gait that
handles stairs, a foot that fits through grating — you change a policy table and retrain, not
your entire label set. Put that table in the annotation spec so annotators and training
cannot drift apart.

---

## 4. Balancing three losses

Three heads means three loss terms, and they are not on the same scale. A segmentation
cross-entropy and a detection focal loss produce numbers of different magnitudes, and if you
simply add them, whichever is numerically larger silently becomes the network's real
objective.

You have two options:

- **Fixed weights.** Simple, predictable, and you tune them by hand. Fine when you know what
  you want.
- **Learned uncertainty weighting** (Kendall et al.). The network learns one scalar per task
  representing how noisy that task is, and down-weights tasks it cannot predict reliably.
  Three extra parameters total.

We default to the learned version, with one important operational caveat: **watch the weights
during training.** They are logged as `task_weight/*`. If one task's weight collapses toward
zero, that head has effectively stopped learning and the trunk is being optimised for the
other two. That is a real failure mode that looks fine in the total loss curve.

---

## 5. The traps that actually cost us time

Every one of these is a real thing that happened, not a hypothetical.

### Validation weights are not the weights you trained

We keep an EMA — an exponential moving average of the weights — because averaging over
training reduces noise and usually validates better. But the average *starts from the model's
random initialisation*, and with a fixed decay it takes thousands of steps to forget it.

On a short run we measured traversability mIoU of **0.16** with EMA and **0.95** from the
exact same training without it. Nothing was wrong with the training; we were validating a
weighted blend of a trained model and random noise.

The fix is to ramp the decay with the update count, so the early average is nearly a straight
copy of the model and smoothing strengthens as history accumulates. That is in place now. The
general lesson is broader than EMA: **know which weights your validation number came from.**

### Whichever metric picks the model *is* your objective

`best.pt` is whichever epoch scored highest on one number. If nobody chooses that number
deliberately, it defaults to something reasonable-sounding, and the entire run silently
optimises for it.

Make it explicit and make it match the mission. If the thing that strands a robot is walking
into glass, then select on glass:

```yaml
primary_metric: IoU/traversability/00_blocked
```

A typo here aborts the run and lists the valid keys — deliberately, because the alternative
is discovering after 14 hours that you selected on the wrong quantity.

### Random train/val splits will lie to you

RUGD and RELLIS ship no official split, and your own footage certainly does not. If you split
frames randomly, frame 1041 lands in train and frame 1042 in val — and they are nearly the
same picture. Your validation score then measures memorisation, and it will look excellent.

**Split by sequence, or by recording session.** Always. The score will drop. That drop is not
a regression; it is the first honest number you have had.

### val is optimistic about itself

`best.pt` is the epoch that scored best *on val*. Reporting that val score as the model's
performance is circular — you picked the winner using that exact measurement. For a number
you would defend to a customer, you need a third split that no part of training ever reads:

```bash
hydranet-eval --config ... --checkpoint ... --split test
```

We deliberately do not default `split_test` to val. If it is not configured, asking for it
fails loudly, because a silent fallback here produces an official-looking number that is
quietly wrong.

### Aspect ratio

If the camera is 9:16 and the network input is 4:5, naively resizing squeezes the image more
than 2× horizontally. Every learned notion of shape breaks. Turn on `letterbox: true` and the
padding gets labelled *ignore* so it contributes no loss.

---

## 6. What to look at while it trains

The loss curve tells you almost nothing beyond "it has not diverged". Three things are worth
more:

1. **The prediction images.** Every validation writes an `input | prediction | label`
   triptych to TensorBoard. This is the single highest-value signal available: a falling loss
   is compatible with the model calling an entire floor a wall, and you will see that
   instantly in a picture and never in a curve.
2. **Per-class IoU**, not just mIoU — for the reason in section 1. Watch the rare, dangerous
   classes specifically.
3. **Throughput (img/s).** If it is well below what the hardware should do, you are not
   compute-bound, you are waiting on data loading. That is a different fix (more workers,
   smaller images offline) and no amount of model tuning will help.

And know when to stop tuning. If `caution` sits at 0.200 and will not move — as ours does,
reproducing at 0.197 when we moved to a GPU ten times faster — the question to ask is not
"which learning rate" — it is "how many examples of `caution` are actually in the
training data?" In our case: of the four terrain classes that map to caution, ADE20K contains
exactly one, at 0.3% of pixels. **That is a data acquisition problem wearing a training
problem's clothes,** and no hyperparameter fixes it.

---

## 7. Make every run answerable

Six weeks after a run, someone will ask what produced a checkpoint. If the answer requires
memory, the answer is lost.

Every run here writes:

```text
runs/<experiment>/
├── meta.json          git commit, dirty flag, environment, dataset fingerprints, full config
├── config.yaml        the config after --set overrides, ready to re-run
├── uncommitted.patch  the working-tree diff, when there was one
├── metrics.jsonl      one line per validation
├── train.log
└── tb/
```

Two of those are less obvious and both earn their place. **`uncommitted.patch`** exists
because a commit hash does not restore code that was never committed, and real training runs
happen on dirty working trees. **Dataset fingerprints** — file counts, sizes, a digest of the
path listing — exist because the data changes more often than the code, and "same config,
different result" is almost always the data.

Then `hydranet-report` reads it back:

```bash
hydranet-report runs/*  --diff     # rank the runs, and show what differed between them
```

We do not run an experiment tracking server (MLflow, W&B). For one machine, files plus
TensorBoard cover it, and the provenance above is better than a tracker's defaults. The point
at which that stops being true is when training spans two machines and there is no single
place to compare them — which is roughly where we are now, so expect this to be revisited.

---

## 8. The order to do things in

If you are picking this up and training a multi-head network from scratch:

1. **Get one head working end to end first.** Data → training → a validation number → a
   picture of a prediction. Do not add head two until head one is honestly scoring.
2. **Prove the loop, not the architecture.** Overfit a single batch deliberately. If the
   network cannot drive four fixed images to near-zero loss, something is disconnected, and
   no amount of data will reveal which. Our test suite does this on every commit.
3. **Add heads one at a time**, and after each, check the task weights and confirm the
   existing head did not regress.
4. **Fix the splits before trusting any number.** Sequence-level, with a test split you do
   not touch.
5. **Choose `primary_metric` deliberately** before the first long run.
6. **Then** tune hyperparameters — and only for things the data can actually support.

The recurring theme: most of what looks like a modelling problem is a data or measurement
problem. The architecture is 15.6% of the parameters and considerably less than that of the
difficulty.

---

*Related: [TRAIN_MACOS.md](TRAIN_MACOS.md) for local development, [HANDOVER.md](HANDOVER.md)
for moving to a CUDA machine, [DEPLOY_JETSON.md](DEPLOY_JETSON.md) for deployment. The
architecture diagram and per-component parameter counts are in the
[README](../README.md).*
