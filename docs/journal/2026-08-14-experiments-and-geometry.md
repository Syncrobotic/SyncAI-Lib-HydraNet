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

1. **Resume `hydranet_fixed_coco10`** — `/home/paul/hydranet-overnight/resume_fixed.sh`,
   ~7 h. It is the only open question about the ratio curve's mechanism.
2. **channels_last** — measured, not built. Two lines and a flag.
3. **The narrowed detection head** — code and baseline are in; the run is not.
4. **Config defaults are fiction.** `batch_size: 16`, `workers: 8` — every real run
   overrides them, so the defaults have never been validated and the next person to run
   `hydranet-train --config` gets a combination nobody has tried.
5. **`prepare_ade20k.py` has 21% coverage** and decides every downstream number.

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
