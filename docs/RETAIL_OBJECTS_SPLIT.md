# The site test split for `retail_objects`

This file exists because of one measurement. The 60-epoch run
`runs/hydranet_retail_objects` scored **terrain_mIoU 0.7668** and, on daytime site
footage from the stores it is meant to serve, predicted **`column` 0.00% and `product`
0.00%** — the two classes the taxonomy was created for. Both numbers are true at the same
time and neither is wrong. The first is ADE20K, the second is the shop.

So the split below is not a tidiness exercise. It is the instrument that would have caught
that, and the rules are written down because most of them are only enforceable by someone
choosing to honour them later.

## The rules

**R1 — Split by camera, never by frame.** These are fixed cameras. Two frames from one
clip are the same shelf, the same floor tiles and the same pixels, a second apart. A
frame-level split puts the answer in the training set and the test set measures memory.
This is the only rule here whose violation invalidates every number that follows it.

**R2 — A test camera never supplies a training frame. Ever.** Not in batch01, not in
batch02, not "just the night clips from it". The moment it does, every comparison across
batches needs a footnote explaining which model saw what, and in practice nobody writes
that footnote. Enforced structurally: test cameras live under the `test` split, and
`configs/*.yaml` names `split_train` separately.

**R3 — Test masks are human-corrected. A pre-label is disqualifying.** The training data is
SAM 3's opinion. If the test data is also SAM 3's opinion, the metric reports whether the
model reproduced SAM 3, not whether it is right — a record produced by the thing it is
being used to check. That is the failure the session board of 2026-08-17 spent a day
naming, and it is not less circular for being about pixels.

**R4 — Every rare class appears on at least two test cameras.** One fixed camera is one
scene measured N times, not N samples. A `column` IoU resting on a single camera is that
camera's IoU, and it will move for reasons that have nothing to do with the model.

**R5 — The richest camera for each rare class stays in training.** Training needs the
strongest available signal; the test only needs enough pixels for a stable number. Where
those compete, training wins — the point is a model that works, not a comfortable metric.

**R6 — All three stores, and both viewpoint families.** Wall-mounted wide-angle and
near-nadir over a desk are different problems; the survey found both across the fleet.

**R7 — A per-class number computed over fewer than two test cameras is not reported.**
Write "not measured" and the camera count, not a figure. This is the rule that would have
caught the state this file opens with: `column` scored 0.51 on a source-domain split and
0.00% in deployment, and a single-camera site IoU is capable of reproducing exactly that
kind of confident wrong number in the other direction. A blank and a `0.42` are read very
differently by the next person, and only one of them is honest about what was behind it.
Applies to reports, dashboards and commit messages, not just to this dataset.

## Why R7 exists, in pixels rather than cameras

The same concern one level down, measured on the ADE20K splits this run actually evaluated,
mapped through `ade20k_retail_objects`:

| class | val images | % of val images | % of labelled px |
|---|---|---|---|
| floor | 285 | 100.0% | 24.60% |
| wall | 283 | 99.3% | 61.86% |
| **column** | **22** | **7.7%** | **0.66%** |
| fixture | 195 | 68.4% | 11.18% |
| product | 0 | 0.0% | 0.00% |
| person | 46 | 16.1% | 1.70% |

**`column`'s entire val IoU of 0.40–0.51 stands on 22 images and 0.66% of labelled
pixels**, and `metrics.jsonl` reports it in the same format, to the same precision, as
`wall`'s 283 images and 61.86%. The ep25→ep60 decay from 0.510 to 0.400 that looked like
the run's headline regression is a swing across 22 images.

That is a second, independent mechanism behind the site 0.00%: `column` is not only out of
distribution, it is nearly absent from the data that scored it. A metric carries no record
of what stood behind it, which is the whole reason R7 is a rule and not a habit.

Neither the domain gap nor its cause was discovered by measurement, incidentally —
`configs/hydranet_retail_objects.yaml:84-86` says it in a comment written before the run:
"`column` gets only ADE20K's id 43, which is architectural columns in atriums and lobbies
rather than a shop's." A true sentence with nothing wired to it, exactly like the
`unsourced_classes` gap. Writing it down is not the same as measuring it, and neither is
the same as being told about it at the point it matters.

## The cameras

Chosen from measured class share in the SAM 3 pre-labels, not from the contact sheets.
Only 24 of the fleet's 48 cameras are shop floor — the rest are back office, stockroom,
classroom, stairwell, street or effectively black — and only those 24 are in scope.

| camera | column | product | person | why |
|---|---|---|---|---|
| Kaohsiung-cam12 | 8.95% | 3.02% | 0.58% | `column` coverage ① |
| Kaohsiung-cam03 | 6.87% | 2.26% | 2.35% | `column` coverage ② |
| Tao-Hsin-cam03 | 5.47% | 1.64% | 4.67% | `column` coverage ③, third store |
| Taichung-cam06 | 0.00% | 13.82% | 2.83% | merchandise wall |
| Tao-Hsin-cam01 | 1.39% | 14.90% | 5.31% | merchandise wall ②, wood floor, own lighting |
| Taichung-cam02 | 0.00% | 1.78% | 11.38% | people and floor |

Two cameras per store, and every rare class on at least two test cameras: `column` on three,
`product` on two, `person` on two. Held in training under R5, and worth stating so nobody
"balances" them into the test set later: **Kaohsiung-cam08** (`column` 17.52%, richest),
**Kaohsiung-cam07** (`product` 33.39%), **Kaohsiung-cam04** (`person` 11.56%).

An earlier version of this table listed **Taichung-cam08** as the merchandise wall on a
measured `product` share of 33.33%. That number was an artefact: the bucket names a clip by
its timestamp, two cameras can start recording in the same second, and the pre-label script
names one output directory per clip *basename* — so Taichung-cam08's frames and
Kaohsiung-cam07's wrote into the same directory and one silently replaced the other. Three
such pairs existed in this pull. Re-run through camera-prefixed names, Taichung-cam08's real
share is `product` 2.49% and `person` 9.61%: an ordinary counter camera, not a wall. The
selection is unchanged in method and different in outcome, which is the argument for
deriving it from measurement in the first place.

## The `column` supply, and a scare that did not survive measurement

At 14 of 24 cameras measured, `column` appeared on four, and this section argued about how
to live with four. Finishing the measurement dissolved most of it: **ten of the
twenty-four cameras carry `column` at 3% or more**, led by Kaohsiung-cam08 at 17.52% and
Taichung-cam11 at 15.11%, and every store contributes some. Three go to the test side and
seven stay in training. That is a workable supply, and no argument about more stores is
needed.

Two things from the scare are worth keeping anyway, because both are true independently of
how the count landed.

**Do not buy more `column` with more clips from the same cameras.** A column is a static
structural object on a camera that never moves, so the 20:00 clip from Kaohsiung-cam08
contains the *same column from the same angle* as the 14:30 one. It adds frames, not
instances: the same pillars at an inflated pixel share, and more opportunity to memorise
them. `person` and `product` do vary across time-slots, which is what makes the temporal
axis worth sampling for them and not for this. The pull already states the principle — a
fixed camera's 189th clip of a day tells you almost nothing its 1st did not — and `column`
is the class it binds hardest on.

**Do not conclude a class's supply from a partial sweep.** Four-of-fourteen was reported as
four, and the missing ten held Taichung-cam11's 15.11%. A partial measurement reads exactly
like a complete one; nothing about the number announces which it is.

One time-axis exception, on its own terms: the **48 night IR clips** are a genuinely
different *appearance* of the same columns, shuttered and unoccupied, and this class fails
specifically on a white-clad pillar against a white wall. A lighting change is not nothing
for that. Take them as a second view, not as more columns.

## What the split is for

Every claim of the form "the model got better" about this taxonomy is measured here or it
is not measured. The source-domain split is ADE20K, and
[RETAIL_SCOPE.md](RETAIL_SCOPE.md) already recorded what happens when it is used to judge
a target-domain change: pseudo-labels scored approximately zero on the source split while
moving `display_fixture` by −0.0096 on the target, and without a control the run would
have been read as a success.

Judge domain adaptation on the target domain or not at all.
