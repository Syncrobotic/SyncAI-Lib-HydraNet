# site30k annotation campaign — the plan, the scope, and what to check

Written 2026-08-20 01:40 local, at the user's request, before an unattended overnight run.
The user reviewed the recipe through four eye passes on 2026-08-19 and accepted v6.0; the
speed work after it (v6.1, v6.2) was verified not to change a pixel. This is the plan that
takes that recipe from 30 frames to the campaign.

## What runs

    .venv/bin/python tools/site30k/plan.py --target 30000
    .venv/bin/python tools/site30k/run_campaign.py \
        --plan runs/site30k_qa/campaign_plan.json --out datasets/site30k_v1 --workers 11

Running as `hydra-site30k-campaign.service` (systemd --user, linger on, so closing the
editor session does not touch it).

`tools/site30k/` is where the reviewed recipe now lives. It was in a session scratchpad
under /tmp until tonight, which was a standing risk recorded in the 2026-08-19 handoff:
the reviewed floor recipe existed in exactly one place and that place is cleared when the
session ends. `recipe.py` is v6.2 plus the campaign hardening below; `plan.py` selects the
work; `run_campaign.py` runs it; `boxes.py` writes the detection boxes; `compare_masks.py`
is the pixel-level diff used to prove a speed change changed nothing.

## Scope: 29,835 frames, and why it is nine cameras rather than forty-three

    calibrated cameras (runs/onboard01/*.calib.json)          23
    of those, METRIC (zones units = m AND calib.scale set)     9   <- the campaign
    day clips on those 9 cameras, 1920x1080                  765   (64 more are 352x240)
    camera-date units                                        261
    frames per clip                                           39
    planned frames                                        29,835

The floor recipe is built on a metric ground plane: the on-plane test, the tolerance
tol(r) = 0.06 + 0.035r, and the BEV vote in 0.05 m cells all need a scale. Ten of the
remaining cameras carry `dav2_raw` zones (relative depth, no scale) and four have no zones
file at all. Without a scale the recipe degrades to b03 alone — and b03 is the teacher
this recipe exists to correct: it reads a polished floor's reflection as `fixture`
(measured 100% / 96.8% / 99.4% on Taichung-cam01's three plates) and does not cover white
or reflective furniture. Annotating 14 cameras that way would put exactly the errors the
user spent the evening rejecting into the training set, silently.

**So those 14 cameras wait for a calibration pass. That is a decision for the user**, not
something to slip into an unattended run. Once they are metric, `plan.py` picks them up
with no other change and the campaign continues to 30k+ by re-running `run_campaign.py`,
which resumes rather than redoes.

Night clips (54) are also out: night is a person-only tranche by decision, with all-IGNORE
segmentation masks, and the plate teachers have no measurement on IR.

Split shares of the planned frames: train 55.6%, val 33.3%, test 11.1%. The assignment is
inherited per camera from `datasets/site30k/split.json` and was not touched. Test is thin
because only one of the 9 metric cameras is a test camera — worth fixing with the
calibration pass, since a thin test split is the one that cannot be repaired later.

## The unit of work is a camera-DATE

The floor and the immovable objects are decided once per camera over the day plates of one
date, then every frame of that date subtracts what stands on them. Per camera is what the
30-frame batch measured as necessary: decided per clip, `column` was accepted in one slot
per camera and never on Taichung-cam04, and a class that flickers between tranches is a
class the student cannot learn. Per date is the bound on that: a fixture that moved in the
three weeks the pull spans must not be painted where it used to stand.

## What protects an unattended run

* **Process isolation per unit.** `data.video.frames` raises on a short decode, which is
  correct — silent truncation of customer footage is the failure this project cannot audit
  after the fact. A unit that dies loses only its own frames; the message lands in
  `failures.jsonl` and the next unit starts.
* **Resume.** A clip whose masks already exist is skipped, so a crash, a reboot or a
  deliberate stop costs only the units in flight. Re-running the same command continues.
* **Plates made on demand**, with the same definition as `scripts/static_plates.py`
  (0.5 Hz, half resolution, temporal median), into
  `datasets/studioa_static_site30k/` so a campaign plate never lands in the pilot's
  directory.
* **Geometry cached per camera** in `runs/site30k_qa/geometry_cache/`. It is a function of
  the calibration and one plate's depth, identical for every clip and every worker, and it
  costs ~40 s to build.
* **Audit trail.** `clips.jsonl` (one line per clip: frames, seconds, plate used),
  `progress.jsonl` (one line per unit), `logs/<camera>_<date>.log` (full output),
  `failures.jsonl` (unit, return code, last 800 bytes of the log).
* Previews are written for one frame in 20 — enough to look at, small enough to keep.

## Cost, measured rather than assumed

Per frame 21.0 s at 3.3 frames/clip fell to 10.0 s at 10 frames/clip, and the daylight
verdict at stride 8 (bit-identical output, verified over 30 frames) takes another ~2.4 s
off. At 39 frames/clip the fixed per-clip cost is nearly gone; the launch measured 1.38 s per
frame across 11 workers, ≈ 11 h for 29,835 frames. The two measurements that redirected this work:

* **The decoder was never the problem.** ffmpeg decodes a 304-frame clip in 1.0 s. The
  22.6 s attributed to "video decode" was `is_daylight` at full resolution — 23.7 s on a
  day clip, 47.5 s on an IR night clip. At stride 8 it is 0.35 s with the same verdict on
  every frame of both.
* **NVDEC does not help and is not equivalent.** With a CUDA ffmpeg build installed at
  `~/.local/opt/ffmpeg-cuda` (h264_cuvid present), the same clip decoded in 1.5 s against
  the CPU's 1.4 s — 0.94x — and not one of the 304 frames came back identical (mean 1.05
  levels, worst pixel 19). It would re-colour the campaign for no speed. Not adopted; the
  build is parked, not deleted, in case a future job is genuinely decode-bound.

## What to check tomorrow

1. `runs/site30k_qa/campaign_run.log` — the per-unit line, the ETA, and the final count.
2. `datasets/site30k_v1/failures.jsonl` — should be short; every line names a clip and
   ffmpeg's own message.
3. The review page (published artifact) — a per-camera sample of the finished frames.
   **Six of the nine cameras have never been looked at by anyone**; a smoke pass of one
   frame per camera was inspected before the run started, but a sample per camera after it
   is the check that matters.
4. `clips.jsonl` — per-clip seconds, to see whether any camera is an outlier.

## Open decisions waiting for the user

* The 14 non-metric cameras: calibrate (and lift the campaign to 30k+), or accept 9.
* The thin test split that follows from those 9.
* The human-judged baseline over the reviewed 30 frames — approved, not yet started; it is
  the only route from teacher-agreement to accuracy.
* Products decided once on the plate (approved as item 4) — not in this run. It changes
  the product boxes, and it was not going to be reviewed before the run started.

## Addendum, 01:30 — what the launch itself measured

**The pull is mixed resolution and that had to be caught before the run, not after.** The
first units died on `operands could not be broadcast together with shapes (240,352,3)
(1080,1920,3)`. A probe of all 883 clips on the nine metric cameras: **810 are 1920x1080
and 73 are 352x240**, the small ones concentrated in the earliest dates of the window
(2026-07-19/20). The store's archive changed profile mid-window. `plan.py` now probes
every candidate clip and drops anything narrower than 1920 — 64 day clips — because SAM 3
returns nothing usable at 352x240 (the campaign's own recorded `product` failure) and the
entire recipe was reviewed at 1080p. Upscaling instead would be a decision to review, not
a default.

**Worker ceiling on this machine: 11.** Each worker holds SAM 3 + b03 + Depth-Anything,
about 6.5 GB of the card. At 14 workers the run put 91.5 GB of 97 GB on the GPU and the
first unit died of CUDA OOM (process-isolated, one unit lost, resumable — which is what
that design bought). At 11 workers with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` it sits at ~52 GB with a load average
of 12 on 24 cores. Thread pools are pinned to one per worker (`OMP_NUM_THREADS=1` and
friends): at 8 unpinned workers the load was 18.5, which is the pools fighting each other
rather than frames being made.

Steady-state after the settling: **1.38 s per frame across 11 workers**, ~11 h for the
remaining 29.7k frames. Sampled worker output shows 90.7-94.7% labelled, and
`column` on Kaohsiung-cam04 holds 3.02-3.04% across four different dates — the
per-camera pooling doing exactly what it was added for.

## Outcome, checked 2026-08-20 13:00–13:50

The run finished 12:03:15 local, 10.39 h wall clock, 261/261 units, **29,211 masks**
(planned 29,835; the 624-frame shortfall is one failed unit plus per-clip rounding).

    failures.jsonl        1 line -- Kaohsiung-cam04 20260729, CUDA OOM at 11 workers,
                          117 frames. Resumable: re-running run_campaign.py picks up
                          only that unit.
    integrity gate        600-mask sample: no problems. Pairing, class ids, emptiness,
                          duplicates all clean. labelled mean 91.1%, min 40.5%.
    per camera            3198-3315 masks each, nine cameras, even.
    class ids present     0..10 and 255 only. Products ARE painted into the mask
                          (laptop 1.2%, tablet 0.1%, phone 0.3%, boxed_stock 5.6%).

### Blocking defect: Tao-Hsin-cam03 paints white counters and the right-hand floor as `wall`

Checked by eye on three frames from three dates (20260722, 20260728, 20260811): the white
display counters down the right side, and the grey floor they stand on, are labelled
`wall`. `shelf` is 0.0% on every one of them and `display_table` only ever covers the
yellow bench in the middle. Sampled over 40 frames the camera reads **wall 51.1%,
shelf 0.4%** against a nine-camera mean of 25.6% / 7.8% -- the largest outlier in the set.

This is the failure already recorded for STUDIO A's white tile and white podiums: a white
surface against a white wall has no photometric edge, and the recipe folds one into the
other. Tao-Hsin-cam03 is the store where that is the whole right half of the frame.

**Tao-Hsin-cam03 is also the only test camera of the nine.** So the test split is 3,237
frames of exactly this error and cannot measure anything: a model that learns the mistake
scores well on it, and a model that gets the counters right is penalised.

### What is still missing before training

1. **No detection labels at all.** `recipe.py` writes masks, images, previews and
   per-clip reports -- there is no COCO output anywhere in `datasets/site30k_v1`.
   `boxes.py` was written for the 30-frame batch and was never wired into the campaign.
   The `person`/`product` heads get nothing from this dataset until a box pass runs.
2. **No split layout.** Images and masks are flat; `seg_folder` wants
   `root/{train,val,test}`. A materialise step keyed on `datasets/site30k/split.json`
   (train 16,146 / val 9,828 / test 3,237 frames) has not been written.
3. **No label_map for this taxonomy.** Existing configs use
   `retail_surfaces_from_objects`; these masks are the 11-id campaign taxonomy.
4. Full integrity gate has only been run on a 600-mask sample.

Review page (12 frames per camera, spread over dates) published as an artifact.

## Two more defects the user found in the review page, checked 2026-08-20 14:30

### Glass is labelled `wall`, and so is everything behind it

On Tao-Hsin-cam04 the whole shopfront glazing is `wall` -- and with it the street behind
the pane: parked scooters, cars, the pavement, and people standing outside. wall is
40.8% of that frame, 36.9% over 40 sampled frames.

This is a KNOWN failure the campaign inherited, not a new one:

    scripts/sam3_prelabel.py:433  "glass  SAM 3 segments what is *behind* the pane
                                   along with the pane"
    scripts/sam3_prelabel.py:444  "wall   absorbs glass here, and a pane still segments
                                   the pavement behind it -- still wrong pixels"
    scripts/column_camera_sweep.py:24  Tao-Hsin-cam11 rejected: "street bollards seen
                                   through a glass door -- glass failure mode 2"

For a model that counts people INSIDE a store this is worse than a mislabelled surface:
a person outside the window is being taught as `wall`, which suppresses `person`.
Confirmed on Tao-Hsin-cam04 and Tao-Hsin-cam03; Taichung-cam11 shows outdoor through
glass at the back. The other six cameras have not been checked for glass yet.

### The "broken wall" is one contiguous abstention, and morphology cannot repair it

The holes the user saw are IGNORE(255), not damage. Measured over 12 cam04 masks:

    < 100 px speckle          82% of holes but  0.9% of ignore AREA
    > 10,000 px components    42 holes       78.1% of ignore area
    fillable under a safe rule (border >=95% one class)   8.3% of ignore area
    median hole 7 px, largest single hole 290,577 px

So hole-filling buys almost nothing. On the 20260801 frame the ignore region is the whole
left third of the image -- left wall, white pillar, left counter, back room -- and it
swallows a customer standing at the counter, so she contributes nothing to `person`.
Repairing it means changing the b03 veto (recipe.py:320) or a human pass, not pixel surgery.

### Proposed fix for all three defects: a per-camera static polygon layer

Glass, cam03's white counters and cam04's left wall are all IMMOVABLE and the cameras are
fixed -- the same premise the recipe already uses to decide structure once per camera-date,
except the teachers vote wrong in exactly these places. Draw once per camera, stamp
deterministically over the finished masks:

* write a new id 11 `glass` rather than 255 directly, and let the label_map decide whether
  this training round treats it as a class or as IGNORE -- changing that decision later
  then costs nothing.
* the same glass line fixes detection: a person box whose feet fall beyond it is a person
  outside the store.
* minutes of CPU, no teacher re-run, verifiable per pixel with tools/site30k/compare_masks.py.

Open question for the user: is `glass` a real class or just IGNORE? Only 4 of 9 cameras
see glass, and a minority-sourced class gets suppressed by the dominant data.

Evidence page published as an artifact (defects_glass_ignore.html).

## Glass, measured rather than drawn — and two pipeline gaps closed, 2026-08-20 15:00

### The day/night differential does NOT find glass. The night plate is worth having anyway

Built the first night plate the project has (Tao-Hsin-cam04, 20260725-152924 = 23:33 local,
150 frames, same 0.5 Hz / half-res / temporal-median definition). At 23:33 the store is
closed: **every product is packed away and the shutter is down**. So a day-vs-night
correlation collapses over the whole shop, not over the glazing, and cannot isolate it.

The plate itself is a find: bare white counters, clean floor, no stock, no people. That is
the view the recipe's "decide the immovable structure once" step actually wants, and it is
the untested lever against the white-on-white failure that cost the test split.

### What DOES find glass: cross-date deviation between plates of the same time slot

Glass is the only surface in the frame whose geometry never moves and whose content changes
every day. 26 plates of Tao-Hsin-cam04's 14:57-15:02 family, per-pixel median absolute
deviation across DATES:

    MAD > 12   2 components   6.7% of frame   glass + merchandise on the centre counter
    MAD > 14   1 component    3.0% of frame   the street through the glazing, nothing indoors
    MAD > 16   1 component    2.4% of frame   same, tighter

At 14 there are zero indoor false positives. The honest limit: it finds *what changes
behind the pane*, not the pane. The lower glazing (pavement, door mat) is as static as a
wall and is not caught, so this is a SEED for the region, not the region. Seed -> grow to
the glazed opening -> one human confirmation per camera is still far less work than drawing
nine cameras from nothing, and it starts from a measurement.

### Two of the four blockers are closed

* **label_map.** `src/syncai_hydranet/data/label_maps_site30k.py` (new), registered as two
  schemes. `site30k_to_surfaces` reads the campaign masks as `RETAIL_SURFACES` --
  display_table+shelf -> fixture, all four product ids -> fixture, following
  `RETAIL_OBJECTS_ID_TO_SURFACES` -- so site30k_v1 mixes with batch02/batch03 under the
  configs that already exist, and the dense head never sees a product class.
  `site30k_native` keeps all eleven. Verified end to end: `SegFolderDataset` over all three
  splits returns {1,2,3,4,5,255} and {1..10,255}.
* **split layout.** `tools/site30k/materialize_splits.py` builds BOTH layouts by relative
  symlink into the flat store -- `images/<split>` + `annotations/<split>` for
  `SegFolderDataset`, and `root/<split>` for `CocoDetDataset`, which disagree about where
  images live. train 16,146 / val 9,828 / test 3,237; `splits.json` records the camera
  identities per split, and the tool warns on a test split of fewer than two cameras.

### Still open

* **Detection labels.** Nothing exists yet. `boxes.py` shows the pass is cheaper than
  feared: `product` boxes come from the mask's own ids 7-10 by connected component, so only
  `person` needs a GPU teacher (GDINO @0.35 over 29,211 jpgs -- no video decode).
  DECIDED HERE: the COCO will be written in the SHIPPING `retail_security` vocab, not the
  campaign's, by mapping the per-box family -- boxed_stock -> boxed_stock, laptop/tablet/
  phone -> device, person -> person. Otherwise a retrain silently empties
  `analytics/events/zones.py:346`, whose default class list is ("boxed_stock", "device");
  `wanted` becomes empty, `zone_stock_counts` returns 0 for every frame, and the
  stock-removal alarm stops firing without an exception. Found by the peer session.
* **The campaign contradicts itself about products in the dense mask.**
  `scripts/campaign_site30k.py` declares "Segmentation, 7 classes: void/floor/wall/column/
  display_table/shelf/person" and puts product in detection; `tools/site30k/recipe.py:79-80`
  writes laptop=7, tablet=8, phone=9, boxed_stock=10 and they are 7.2% of the finished
  pixels. That reverses `analytics/events/_types.py:24` ("moved `product` out of the dense
  head and into detection") and no decision is recorded anywhere. The two schemes above make
  it a config choice rather than a data commitment, but THE DECISION IS STILL THE USER'S.
* `analytics/events/pose.py:300-305` takes `fixture_id: int`, one int, which cannot express
  display_table AND shelf. Left to the analytics session; flagged, not changed.

## Glass: four automatic routes tried, all fail. It is TWO cameras, and it gets drawn

The cross-date MAD seed generalises badly. Run over all nine cameras (largest time-of-day
plate family per camera, 16-20 plates each, threshold 14, components >= 8000 px):

    Tao-Hsin-cam04     3.10%  TRUE  POSITIVE -- the street through the shopfront
    Kaohsiung-cam04   15.29%  FALSE POSITIVE -- the service counter and its daily merchandise
    Taichung-cam01     2.11%  FALSE POSITIVE -- a rolling display cart that gets moved
    the other six      0.00%  NOTHING -- including Tao-Hsin-cam03, which HAS glass doors

One true positive out of three fires, and it misses the other glazed camera because the
mall corridor behind cam03's doors is as static as a wall. It is not a detector.

Everything else was tried or was already closed:

* **SAM 3 glass prompts.** Closed before this session: `sam3_prompts.py:203-233` records
  four distinct failure modes across two sessions and 112 frames -- finds nothing (12
  phrasings x 7 thresholds, <=0.12%), finds what is behind the pane, finds glossy surfaces
  (a whole teal workbench), finds people reflected in it. "It stays a human class", and
  METHODOLOGY.md already ranks glass the highest-priority ANNOTATION target.
* **Day/night plate differential.** Fails because the store empties at night: stock packed
  away, shutter down, so the whole interior changes, not the glazing.
* **Day/night restricted to `wall` pixels.** Measured, no separation: local NCC inside wall
  p25/p50/p75 = -0.06 / 0.29 / 0.80 against the glass seed's p10/p50/p90 = -0.22 / 0.07 /
  0.44. Night lighting changes a real wall as much as glazing changes.
* **Metric depth / ground projection.** Measured, no separation: gz p50 13.5 m inside
  wall AND seed against 9.6 m inside wall AND NOT seed.

### The scope was wrong, and that is the useful finding

Checked every one of the nine plates by eye. **Only two cameras see glazing at all**:
Tao-Hsin-cam03 (glass doors onto the mall corridor) and Tao-Hsin-cam04 (the street-facing
shopfront). The other seven are interior views -- Taichung-cam10's and cam11's bright
patches are interior aisles, not windows. So this was never a nine-camera drawing job.

Proposed polygons, in plate (960x540) coordinates, scaled x2 when stamped:

    Tao-Hsin-cam03  (196,0) (460,0) (458,214) (198,222)   11.1% of frame
    Tao-Hsin-cam04  (272,0) (580,0) (575,150) (278,157)    9.0% of frame

**Stamped only where the mask currently says `wall`.** That guard is what makes a
slightly-too-large polygon safe: an interior counter standing in front of the glazing keeps
`display_table`, and only the pixels the defect is actually about change. Measured effect:
27.0% of cam03's wall and 23.0% of cam04's wall would become `glass`.

Awaiting the user's confirmation of the two polygons before stamping.

## The night pass is dead for the white-on-white defect: the store shutters its walls

Built Tao-Hsin-cam03's first night plates (20260725-152759 and 20260812-153008, 150 and 152
frames, the same plate definition). At 23:39 local **both display walls are behind rolling
shutters** -- corrugated panels from ceiling to floor, left and right. The white counter
banks that read as `wall` in daylight are not visible at night at all.

So the hypothesis that IR's directional illumination would put an edge where daylight puts
none cannot be tested on this camera, and the camera is the one that motivated it. Recorded
as a closed route, not a pending one. (Tao-Hsin-cam04's night plate does show its counters,
bare -- but cam04 is not the camera with the defect.)

That leaves the polygon assertion as the fix for cam03, which is the same mechanism the
glass fix needs, and `tools/site30k/stamp_zones.py` now does both.

## Ready for the 18:00 review

* **Full integrity gate, all 29,211 masks: no problems found.** labelled share mean 91.2%,
  min 35.1%, max 99.6%; per-camera counts even (3,198-3,315).
* `tools/site30k/stamp_zones.py` + `tools/site30k/zones.json` -- four polygons over two
  cameras, dry-run only. Every zone names the classes it may overwrite (`from`), so a
  counter standing in front of the glazing keeps `display_table` and the floor between two
  counters keeps `floor`. Dry run: cam03 3,237 masks (203k px -> glass, 359k px ->
  display_table per frame), cam04 3,198 masks (144k px -> glass). Originals are backed up
  to `masks_prestamp/` on `--apply`, so `compare_masks.py` can diff it.
* `tools/site30k/box_pass.py` running: GDINO person + product boxes from the mask's own
  ids, written in the SHIPPING `retail_security` vocab. 6.6 fps, ~1.2 h for the set.
* `configs/hydranet_site30k.yaml` -- validated by `load_config`/`check_config`. Eleven
  terrain classes (the user's decision that products go in the dense head), detection
  unchanged at four. The sibling sets keep supervising terrain through two new schemes,
  `ade20k_site30k` and `retail_objects_to_site30k`, which read `fixture` as IGNORE because
  those masks predate the display_table/shelf split and cannot say which they meant.
* Five schemes now registered in `label_maps.py`: `site30k_native`, `site30k_to_surfaces`,
  `surfaces_to_site30k`, `ade20k_site30k`, `retail_objects_to_site30k`.

Nothing is committed and nothing is trained.

## An operational hazard, hit today: editing the package under a running campaign

At 15:00 a new label-map table was added with a `KeyError: 255` in it -- `_through()` did
not exist yet and IGNORE was not passed through. The package was broken for about six
minutes. The campaign resume was running at the time, and `run_campaign.py` spawns ONE
PROCESS PER UNIT, so every unit that started inside that window died at import:

    20 spurious lines in failures.jsonl, dates 20260729-20260731 across all nine cameras,
    rc=1, 2.0 s each, tail = the KeyError traceback.

No frames were lost -- those units were already complete and would have skipped their
clips anyway -- but the one genuinely missing unit (Kaohsiung-cam04 20260729, the original
CUDA OOM) failed again for this new reason and is still missing. Resume relaunched at
15:26 after verifying the import.

`failures.jsonl` is left intact rather than pruned: it is the audit trail, and a file that
gets edited when it says something inconvenient is not one. The 21 lines mean 1 real
failure, 20 caused by this edit.

**The rule this earns**: the campaign's per-unit process isolation, which protects it from
a truncated clip, gives NO protection against the source tree changing underneath it -- it
converts that into one dead unit per process instead. Sessions share this checkout. Do not
touch `src/syncai_hydranet/` while a campaign or a box pass is running; stage the edit and
land it between runs.
