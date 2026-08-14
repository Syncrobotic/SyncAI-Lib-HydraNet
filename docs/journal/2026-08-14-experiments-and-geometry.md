# 2026-08-14 — the ratio sweep, the measurement rule, and ground projection

Handoff from the session that ran the experiments. Three other sessions worked the same
repo today on deployment, annotation tooling and the retail variant; this covers only the
training/measurement line and the geometry module.

## The one thing to read first

**A validation split cannot measure what it also selects.** Three checkpoints scored
0.1241 / 0.0844 / 0.0057 on `caution` on val, and 0.3252 / 0.3281 / 0.3346 on test — the
same three models, equivalent all along. `best.pt` is chosen on val, so each run was cut
at whatever epoch that class happened to spike, and the selection manufactured the
variance it was then measured with.

I reached four wrong conclusions from val today before finding this. Every correction was
a comparison whose basis did not hold: final-vs-best, then two different val sets, then
selection variance, then "ratio 0.3 looks like it gets both" — which test flatly denied.
The rule is now in `docs/RETAIL_SCOPE.md` §6: **low-frequency classes are reported on the
test split only; validation selects and nothing else.**

## The ratio sweep, on the held-out test split

How much of each epoch's steps come from COCO, against what it costs the segmentation
heads. Both joint runs gave the segmentation heads the same 7,440 steps as the baseline,
so this is not a step-count artefact.

| COCO share | detection mAP | glass | caution | traversability mIoU |
|---|---|---|---|---|
| 0 (seg only) | — | 0.5021 | 0.3252 | 0.7055 |
| **0.1** | 0.1988 | **0.5462** | **0.3346** | **0.7093** |
| 0.3 | 0.2790 | 0.4438 | 0.2649 | 0.6783 |
| 1.0 | 0.3348 | 0.0692 | 0.1064 | 0.6055 |

**Monotonic. There is no sweet spot** — 0.1 is a floor, not an optimum. The exchange rate
gets worse as you climb: 0.1 → 0.3 buys 0.080 mAP for 0.102 of glass; 0.3 → 1.0 buys 0.056
for 0.375. At 0.1 detection is better than free: segmentation came out slightly ahead of
the segmentation-only baseline.

Retail stays at 0.1 (`configs/hydranet_retail.yaml`). Glass is what a store is made of.

**Caveat that is still open:** with `loss_balancing: uncertainty`, changing `sample_ratio`
also moves each head's learned weight, so dilution and balancer effects are not separated.
The control that would separate them is `runs/hydranet_fixed_coco10`, stopped at epoch 6
of 60 — see `docs/ARCHITECTURE_REVIEW.md` for the three readings, written before it ran.

## Detection: a subset, and how to score it

`classes:` on a coco dataset block trains the head on named categories instead of all 80.
`score_classes:` restricts COCOeval without touching the head or the label numbering —
they are separate keys because they are separate jobs, and using `classes` for scoring
would renumber labels under a checkpoint trained with the old numbering and report a
confident wrong number. The config guard catches that; it caught me.

Pre-registered baseline, `runs/hydranet_joint_coco10/best.pt` on val2017:

```
all 80 categories              mAP 0.3348
the 25 indoor categories only  mAP 0.3246
```

**The indoor subset is harder than COCO's average, not easier.** A narrowed head has to
beat 0.3246. The experiment itself has not been run.

## channels_last is worth 33%, and only under autocast

> Later the same day: re-measured at **-21%**, and "it changes no arithmetic" turned
> out to be too strong. Both corrections are in the continuation section below; the
> table here is left as written.

Measured on this box, batch 48, 512×640, full training step:

```
bf16 autocast   180.6 ms -> 120.6 ms   -33.2%
fp32            219.6 ms -> 220.9 ms    +0.6%
```

NHWC is what tensor cores want; fp32 goes through CUDA cores and does not care. Not
implemented — it needs a config flag and two `.to(memory_format=...)` calls in the
trainer. It changes no arithmetic.

I had argued against trying it, reasoning that 91% GPU utilisation and a 300 W power cap
meant there was nothing to win. That reasoning was wrong: implicit transposes count as
utilisation, and the saving is in work not done.

## geometry/

`src/syncai_hydranet/geometry/` projects a mask onto the floor in metres and places
detections where their boxes meet the ground. The camera pose is fitted to the depth
return per frame rather than measured once, because a walking quadruped pitches and rolls
with every step. `scene()` emits plain data — metres and class ids, no colours — for
renderers, RViz and a costmap publisher to share. `docs/GROUND_PROJECTION.md` has the
reasoning.

Two things it still needs: a CLI entry point (the module is importable but not runnable),
and one session recorded on the robot, which is also what the annotation pipeline and the
3D work are waiting on.

## Open, in the order I would take them

1. ~~**Resume `hydranet_fixed_coco10`**~~ — running since 14:55, see below.
2. ~~**channels_last**~~ — built, tested, benchmarked; see below.
3. **The narrowed detection head** — the run is not done, and the baseline was not as
   ready as this list claimed; see below. It is now. Wants the GPU, so it queues behind
   the control above.
4. ~~**Config defaults are fiction.**~~ — two live defects found and fixed; see below.
5. ~~**`prepare_ade20k.py` has 21% coverage**~~ — now 98%, and the coverage found a
   live bug; see below.

---

# Continuation, same day, 14:52 onwards

## The control died twice before it ran

`hydranet_fixed_coco10` was not running at the start of this session. It had reached
`E7 [400/2567]` and stopped at 14:44:53 with no traceback and no `=== fixed control
exited ===` line, which the script prints on any exit — so the whole process group took
a SIGKILL. Not memory: 29 GB of 503 in use, nothing in the kernel log. `runs/` was
intact at epoch 6, `detection_mAP=0.2875`.

Between the 13:26 reboot and this, the run has now been killed twice in one day without
either event being anybody's decision. So it is a systemd user unit now,
`~/.config/systemd/user/hydranet-fixed-coco10.service`, with lingering enabled — which
`loginctl enable-linger` sets without root, so the missing piece was never permissions:

- survives a session ending *and* a reboot, which is what `setsid` could not do
- `Restart=on-failure` with `StartLimitBurst=3` per hour: a crash or a reboot recovers,
  a person killing it three times wins. `ExecStart` resumes from `last.pt`, so a
  restart costs at most one epoch
- `Nice=5`, matching the retail run sharing the card, so neither starves the other's
  dataloader workers

Also: the bracket trick against `pgrep` self-matching has a second failure mode that
was not in the note above. Under zsh, `pgrep -f "[h]ydranet-train"` does not run at all
— the shell tries to glob `[h]...`, matches no file, and aborts the command with
`no matches found` before `pgrep` is reached. It bit me on the first command of this
session. Quote the pattern, or keep taking PIDs and reading `/proc/<pid>/cmdline`.

## channels_last, and the thing the measurement did not say

Built as `train.channels_last`, defaulting to **false**. Model conversion happens
before the EMA copy is taken — `ModelEMA` deep-copies, so converting afterwards would
leave validation running NCHW while training ran NHWC. The evaluator reads the layout
off the model's own weights rather than the config, because a checkpoint does not
record its memory format and `hydranet-eval` has no flag to consult.

The default is false for a reason the earlier note gets wrong. "It changes no
arithmetic" is true of the layout and false of the run: NHWC selects different
convolution kernels, which accumulate in a different order.

| | max divergence, relative to each output's scale |
|---|---|
| TF32 off | ~1e-5 |
| TF32 on — what every run sets | ~1e-2 |

TF32 carries about ten mantissa bits, so a reordered sum shows up immediately. It is
not a bias and not a defect, but it means a run with the flag flipped partway through
is not bit-comparable to itself. A control that exists to isolate one variable must not
pick up a second one from a restart — which is exactly what the systemd unit above
makes possible. Both facts are in `tests/test_channels_last.py`.

Re-benchmarked with `scripts/bench_channels_last.py`, interleaved A/B, twice — once
with two other trainings on the card and once after the retail run finished:

```
batch 48 @ 512x640, bf16 autocast, 45 steps each arm
                       median        min
two competitors  NCHW  181.4 ms    158.7      NHWC  163.3 ms   122.4     -10.0%
one competitor   NCHW  153.2 ms    150.8      NHWC  121.1 ms   116.6     -20.9%
```

In the second run median and min agree to within 3%, which is the sign that the
measurement is no longer mostly about the neighbours. **So the number to carry forward
is about -21%, not -33%.**

The gap is entirely in the NCHW arm: NHWC measures 121.1 ms against the earlier 120.6,
while NCHW measures 153.2 against 180.6. Which of the two is right is not something
this session can settle, because **the original benchmark script was never committed**
— the figure exists, the code that produced it does not. That is the same failure as an
unrun config default: a plausible number with nothing behind it that anyone can
re-execute. Mine is in `scripts/` for that reason, and its docstring is explicit that
its synthetic loss makes the absolute milliseconds non-comparable to a timing taken
around a real training step.

## Config defaults: two live defects, both invisible to everyone who has run this

Sharper than "never validated". Both were load-bearing.

**The inherited learning rate.** Every from-scratch run sits exactly on
`lr = batch_size x 1.25e-5` — 48 → 6.0e-4, 32 → 4.0e-4, base 16 → 2.0e-4. A convention
nobody wrote down. `hydranet_indoor.yaml` and `hydranet_retail.yaml` both lower
`batch_size` to 8 and inherit the base's `lr: 2.0e-4`, which was set for its batch of
16 — so both shipped configs asked for **twice the effective rate of every run this
project has produced**. Nobody saw it because every run passed `batch_size=48
lr=6.0e-4` on the command line and never read those two lines. Both now carry
`lr: 1.0e-4`.

**The mixed-precision dtype nothing was trained in.** No config set `amp_dtype`. The
trainer's fallback was `float16`. All nine runs in `runs/` used bfloat16, every one of
them supplying it via `--set`. So `hydranet-train --config` trained in fp16 — a
precision this project has never used, on a detection loss with a logged AMP-dtype
crash that only autocast reaches (`tests/test_amp_detection_loss.py` exists because of
it). The base config now states `amp_dtype: bfloat16` and the code fallback agrees.
bf16 needs Ampere or newer, so the trainer refuses with an explanation on older cards
rather than downgrading silently — a silent downgrade here is precisely what would
explain a run's numbers to nobody, months later.

`tests/test_config_defaults.py` checks both across every shipped config, with a
`DECLARED_DEVIATIONS` table so a config off the lr line has to say why. It has one
entry: `hydranet_retail_cctv.yaml` fine-tunes at a quarter of the from-scratch rate,
which is what `runs/hydranet_retail_cctv` actually used.

The general shape is worth keeping: **a default that is always overridden is not a
default, it is an untested code path with a plausible-looking value sitting in front of
it.** Item 5 on the list above is the same species — 21% coverage on the script that
decides every downstream number.

## The 0.3246 baseline had outlived its own definition

Item 3 was described as "code and baseline are in; the run is not." The code is in. The
baseline was not: **no config, no module and no log anywhere in this repo recorded which
25 COCO categories 0.3246 was scored over.** `score_classes` appears in `datasets.py`,
in the schema and in a test with a two-name toy list — never with the 25. The
measurement's own JSON records the checkpoint, the epoch and the git commit, and not one
word about what was scored.

That is not a small gap. mAP over a subset is the mean of its per-category APs, so the
category list *is* the denominator. A narrowed head measured against a
reconstructed-from-memory list would have produced a confident comparison between two
different quantities — the precise failure `score_classes` was built to prevent, one
level up.

Recovered from a previous session's transcript, and then **proved rather than assumed**
by re-running the evaluation:

```
                       original                reproduced
detection_mAP          0.3245647505782422      0.3245647505782422   exact
detection_mAP50        0.49341168993493867     0.49341168993493867  exact
traversability_mIoU    0.5838602503643852      0.5838602503643852   exact
terrain_mIoU           0.3595365932237891      0.3595365932237891   exact
```

Sixteen digits on four metrics is not a coincidence; a single category added or dropped
moves mAP in the third decimal. `sample_ratio: 0.1` turned out to be part of the
definition too, not just the category list.

Now in three places that a test forces to agree — `data/coco_subsets.py:INDOOR_25`,
`configs/eval_indoor25.yaml` (runnable in one command, with the expected numbers in the
file you would run), and `tests/test_indoor25_baseline.py`. One of its tests also
asserts no *training* config sets `score_classes`, which would change every reported mAP
without changing what the model learned.

And the reason it could go missing at all is fixed: `hydranet-eval --json` now records
`config`, `set` and the resolved `datasets` block alongside the metrics. A file saying
`0.3246` with no trace of which 25 categories cost most of a day to reconstruct.

## prepare_ade20k: 21% → 98%, and the missing 79% was hiding a real one

`_is_test` had eight thorough tests. Everything around it — the filter that chooses the
frames, the split writer, the re-run path — had none, which is how this survived:

**Re-running without `--test-fraction` put the entire test split back into val.** The
flag defaults to 0, so omitting it on a re-run is the easy mistake, not an exotic one.
The code cleared only the directories it was about to write; with `test_fraction=0` the
test directory is not one of them, so the old test split sat there untouched while val
was rebuilt over *every* kept frame. On a 20-image fixture, 9 of 9 held-out images came
back in val. Nothing printed a warning.

This is the same failure as `docs/RETAIL_SCOPE.md` §6 — the split that is supposed to be
uncontaminated quietly stops being uncontaminated — arriving through the filesystem
instead of through the code. And §6 is the rule the rare-class numbers depend on, so a
`caution` or `glass` score read off a contaminated test split would have looked exactly
like a real one.

`datasets/ADE20K` is clean: val 285, test 329, zero overlap. The last real invocation
passed `--test-fraction 0.5` and nobody re-ran it. This was a live hazard, not a past
contamination — worth being precise about, because "we might have been reporting
contaminated numbers all along" and "we would have, on the next re-run" call for very
different responses.

The fix clears a stale test split whenever the val split is rebuilt without one, and
says what it removed. 18 new tests; the two lines still uncovered are `_score_one`
(executes in a worker process, so coverage cannot see it) and the `__main__` guard.

## Operational notes

- The machine rebooted at 13:26 and killed two runs. `setsid` survives a session ending,
  **not** a reboot; that needs a systemd unit, which nobody has set up.
- `pgrep`/`pkill` self-matching bit me five times. The bracket trick (`coco0[3]`) only
  works when the escaped character does not appear in matching form elsewhere on the same
  command line — a full path mentioned later defeats it. Take PIDs, check
  `/proc/<pid>/cmdline`, then kill by PID.
- The GPU driver survived the reboot because it was installed via apt rather than
  `modprobe`. A shortcut there would have charged its bill at the next boot, with nobody
  connecting the two events.
