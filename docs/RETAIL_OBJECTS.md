# Retail object segmentation

Segmenting what is *in* a shop — person, floor, wall, column, fixture, product — rather
than whether a robot can walk on it. This document is the argument for a second retail
taxonomy, the audit that produced it, and what is still missing.

The traversability line is unchanged and stays shipped. [RETAIL_SCOPE.md](RETAIL_SCOPE.md)
is still the reference for it.

---

## The audit

18 fixed-camera site clips across three stores, 1,620 frames sampled at 3 fps, scored
with the 60-epoch checkpoint `runs/hydranet_retail/best.pt`. Raw output in
`runs/review_20260816/`.

The camera does not move, so a pixel that changes class between frames is one the model
was unsure about — the same error signal the SAM 3 consensus pass uses, turned on the
model itself. `agree` below is the share of frames on which a pixel keeps its modal
class.

| class | share of frame | agree | test IoU |
|---|---|---|---|
| `wall` | 39.8% | 93.7% | 0.835 |
| `floor_hard` | 26.2% | 93.5% | 0.826 |
| `obstacle_furniture` | 17.6% | 86.8% | 0.707 |
| `display_fixture` | 10.7% | 79.3% | **0.336** |
| `door` | 3.5% | 71.0% | 0.380 |
| `person` | 1.4% | 76.3% | 0.760 |
| `glass` | 0.6% | 62.8% | 0.493 |
| `stairs` | 0.3% | 55.6% | 0.108 |
| `floor_metal`, `wet_slippery`, `threshold_ramp` | 0.00% | — | 0.000 |

Per-camera terrain stability runs from 92.9% (Kaohsiung-cam12) to 61.6%
(Tao-Hsin-cam05); the four worst are the cameras with polished near-field tiles or a
large glazed frontage.

### Three findings

**1. Fixtures are split across two classes.** In a single Kaohsiung-cam08 frame the
round MacBook podium is `obstacle_furniture` and the wall shelving three metres away is
`display_fixture`. Both are shop fixtures. The cause is in
[`label_maps_retail.py`](../src/syncai_hydranet/data/label_maps_retail.py): ADE20K's
`table` (16) is deliberately withheld from `display_fixture` so that a dining table does
not teach the model it is shop furniture. Under traversability that is correct and the
cost is only semantic — both classes are `blocked`. Under object segmentation the
semantics *are* the output, and the cost is `display_fixture`'s 0.336.

**2. Columns are inside `wall`, by construction.** `ADE20K_ID_TO_INDOOR` maps `column`
(43) to `wall`, and the class comment has read "wall / ceiling / column" since the first
commit. `wall` is 39.8% of every frame, so the largest and most stable class in the model
is also the one hiding a structure the store cares about. RETAIL_SCOPE.md §5 already
tracked a column through 97.4% of one camera's frames — as a false `caution`.

**3. Merchandise has no class, and the detection head is already finding it.** Sweeping
the score threshold over 40 frames each:

| threshold | Taichung-cam01 | Kaohsiung-cam08 |
|---|---|---|
| 0.05 | 100.0 boxes/frame — `book` 914, `bottle` 783, `refrigerator` 476 | 84.7 — `book` 1683, `refrigerator` 384 |
| 0.15 | 11.8 — `person` 112, `chair` 108 | 10.0 — `book` 311, `mouse` 19 |
| 0.25 | 3.4 — `person` 72 | **0.0** |
| 0.35 | 1.6 — `person` 62 | **0.0** |

Kaohsiung-cam08 is an Apple store. There is no `laptop` in that column at any threshold
and there are 1,683 `book`: the head is finding the merchandise and naming it with the
nearest shape word COCO owns. **The localisation already works; the vocabulary and the
calibration do not.** At the shipped viewing threshold that camera returns no detections
at all.

---

## The taxonomy

[`label_maps_retail_objects.py`](../src/syncai_hydranet/data/label_maps_retail_objects.py)

| id | class | notes |
|---|---|---|
| 0 | `void` | ignore; never annotated |
| 1 | `floor` | hard and soft merged |
| 2 | `wall` | absorbs ceiling, glazing and doors |
| 3 | `column` | **split out of wall** |
| 4 | `fixture` | **`obstacle_furniture` + `display_fixture` merged** |
| 5 | `product` | **new; no public source** |
| 6 | `person` | |

### Why a second scheme rather than two more ids

`tests/test_retail_scheme.py` pins retail ids 0–11 to `INDOOR_TERRAIN` so a retail run
warm-starts from an indoor checkpoint and indoor masks stay readable. This taxonomy has
to merge two of those ids and split a third, neither of which is expressible under that
invariant — extending would mean deleting the tests that exist to stop the extension.
The robot's taxonomy is untouched.

The reversal on ADE20K's `table` is the clearest illustration:
`tests/test_retail_scheme.py` asserts it is **not** a display fixture and
`tests/test_retail_objects_scheme.py` asserts it **is** a fixture. Both are right, which
is the argument for two taxonomies.

### The traversability head is gone

On segmentation data the traversability target is `terrain` run through a lookup table —
no new information. The 60-epoch run measured the consequence: `head_disagreement`
0.0091, the two heads agreeing on 99.1% of pixels because one is a deterministic function
of the other. Under the traversability goal that redundancy still bought a head that was
exactly the deployment question. Here it buys nothing.

`model.heads.traversability: null` in a config removes an inherited head. The deletion
happens once, before validation, so the schema, `unsupervised_heads`, the exporter and
`meta.json` all agree it does not exist.

---

## Migrating the masks already drawn

Everything under `datasets/retail_sam3_consensus*` is annotated as the retail 13. The
`retail_objects_migrated` label map reads it under the new taxonomy with no re-export.
Verified on all 216 consensus frames:

```
retail_native              retail_objects_migrated
  floor_hard    42.28%       floor      42.71%
  wall          26.43%       wall       31.34%   = wall + glass + door, pixel-exact
  door           4.60%
  obstacle_f.    1.14%       fixture    20.89%   = obstacle_furniture + display_fixture,
  display_fix.  19.55%                            pixel-exact (5,558,232 px)
  person         5.01%       person      5.06%
  stairs         1.00%       (-> ignore)
  ignore        62.04%       ignore     62.41%
                             column      0 px
                             product     0 px
```

**`column` does not survive the migration and cannot.** Those masks put columns inside
`wall`; there is no signal left to separate them. `get_scheme("retail_objects_migrated")`
warns, `columns_from_migration_only()` reports it to the config validator, and
`hydranet-annotation check` shows it as 0.00% flagged `<- priority`. Three places,
because the failure it prevents — a permanently empty output channel reporting IoU 0.000
after sixty epochs with no error anywhere — is the one this project has already shipped
three times (`floor_metal`, `wet_slippery`, `threshold_ramp`).

---

## What is still missing, in order

**1. Columns.** ADE20K supplies id 43, which is atrium and lobby columns rather than a
shop's. The cheap source is the site: the cameras are fixed, so a column is **one polygon
per camera**, correct for every frame that camera will ever produce. Same argument the
floor already gets in `sam3_prompts.py`. 41 cameras, one polygon each.

```bash
hydranet-annotation labels --scheme retail_objects --out cvat_objects.json
```

**2. Products.** No public segmentation dataset labels merchandise, and nothing has yet
asked SAM 3 for it — the prompts in
[`sam3_prompts_objects.py`](../src/syncai_hydranet/data/sam3_prompts_objects.py) are
unmeasured, so the first batch is an experiment and its coverage figure is the result.
The script prints `FOUND NOTHING:` for any class that came back empty rather than
letting it vanish from the composition table.

```bash
python3 scripts/sam3_prelabel.py --scheme retail_objects --consensus 0.9 \
    --out datasets/retail_objects_batch01 --frames 12 --upscale 1.0 /path/to/cam*/clip.mp4
hydranet-annotation check datasets/retail_objects_batch01 --scheme retail_objects
```

The consensus pass removes only *random* error — a fixture SAM 3 consistently misses
survives untouched — and it cannot produce a test split, because scoring a model against
SAM 3's masks measures agreement with SAM 3.

**3. A hand-drawn test split.** Still the critical path, and still absent. Every mask in
`datasets/retail_cctv_pilot*` and `datasets/retail_sam3_consensus*` is SAM 3 output with
no human pass. RETAIL_SCOPE.md §6 argues why this has to come first; nothing here changes
that.

**4. Detection calibration.** `SCORE_THR_RETAIL = 0.20` restores boxes on the cameras
that return none at 0.30, and it is a stopgap, not a fix: scores are not comparable
between these cameras, so one global cut cannot be right for both. Per-class thresholds
fitted to hand-labelled site boxes are the fix, and they need item 3 first.

`--vocab retail` reads the existing head's existing output as shop nouns
(`product/book`, `fixture/refrigerator`). It **adds no knowledge and trains nothing** —
it is a rename, and it inherits every mistake the head makes.

---

## Running it

```bash
# train (ADE20K bootstrap only until the site data above exists)
hydranet-train --config configs/hydranet_retail_objects.yaml

# inference; there is no traversability panel, so the terrain overlay carries the boxes
hydranet-infer-video --config configs/hydranet_retail_objects.yaml \
    --checkpoint runs/hydranet_retail_objects/best.pt \
    --input clip.mp4 --output pred.mp4 --fps 6 \
    --vocab retail --score-thr 0.20
```

`hydranet-scene` refuses this config by design: every panel it draws starts from free
space, which is the head that was removed.
