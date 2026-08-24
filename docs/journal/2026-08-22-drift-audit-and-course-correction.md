# Drift audit against the business plan, and the corrected order of work

Written 2026-08-22 at the user's request, after they read the site30k campaign results and
said the project had gone the wrong way. The reference is
[`docs/hydranet-decisions.md`](../hydranet-decisions.md) (dated 2026-08-22, "架構與資料管線
已定案,待執行") — the business plan. This entry measures the distance between that plan and
what was actually built on 2026-08-19/20, and states what to do instead.

The short version: **the campaign executed priority #4's pipeline before priority #4's
blocking decision was made, and before priorities #1, #2 and #3 were started.** It also
did so by a method the plan's most important decision forbids.

---

## 1. The drift, item by item

| plan | what it says | what was built | verdict |
|---|---|---|---|
| **§4 zero manual annotation** — called "本次最重要決議" | humans do accept/reject sampling only; total budget one person ~3 days; the *only* one-time human input is 3 uniform reference photos | `tools/site30k/zones.json` — four polygons drawn by hand; four eye-passes over the recipe on 08-19; a click tool for 9 cameras proposed on 08-22 | **violated** |
| **§6 #1 homography calibration** — "所有速度規則的地基" | 4 clicked ground points per camera, ~10 min each | not done. The repo instead runs DA-V2 depth → RANSAC floor → person-height scale, which is **blocked on 14 of 23 cameras** | **priority #1 skipped, and a harder route taken** |
| **§6 #2 Pipeline B** — "量最大、lead time 最長" | GDINO ensemble + **CoTracker3 temporal consistency filter** + Gold/Silver/Gray confidence tiering, Gray → ignore region | `tools/site30k/box_pass.py`: a single GDINO pass, no temporal filter, no tiering, every box written | **core mechanism absent** |
| **§6 #3 overhead person retrain** | third priority; ReID only after, "不同時上兩個變因" | not started | — |
| **§6 #4 seg dual-weights decision → then Pipeline A** | the decision gates the pipeline | decision still open; Pipeline A ran to completion — 29,211 masks, 10.4 GPU-hours | **order inverted** |
| **§5 #7 training order** | detection converges → freeze first three backbone stages → then train the seg head | the whole of 08-19/20 was seg-side; detection was bolted on at the end | **order inverted** |
| **§1 seg output** | 10 classes incl. `floor_carpet`, `ramp`, `stairs`, `movable_obstacle`, `glass_door` | site30k_v1 is 11 ids: floor/wall/column/display_table/shelf/person + 4 product classes | **taxonomy mismatch — 5 of the plan's 10 classes have zero pixels in site30k_v1** |
| **§1 detection** | 6 classes: person / iphone / ipad / macbook / boxed_stock / stack | shipping vocab is 4 (person/bag/boxed_stock/device); box_pass wrote 3 | **mismatch** |
| **§4 Pipeline A glass** | "玻璃門用 IR 夜間幀補" | measured 08-20: Tao-Hsin-cam03 rolls **shutters** over both display walls at night; the day/night differential collapses because the store empties | **the plan's stated mechanism is disproved on the camera that needs it** |
| **§7 #2 data scope** | run the teacher pipeline over the existing 48 streams | 9 cameras — and all 9 were already annotated in `retail_objects_batch02`/`batch03`. Union unchanged at 24 | **zero new cameras for 10.4 GPU-hours** |

### Two numbers in the plan that are themselves stale

* **"80 路"** appears in §1 and §5 #1. The recorded production ceiling is
  **96 streams at 15 fps (1,440 frames/s)**, set in `cc80fc3` on 2026-08-19. The ReID
  128-vs-256 sizing argument rests on the wrong number.
* **§2 keeps the seg head "室內可能架設不同地形(機器狗需求)"** — but the quadruped line
  was deleted on 2026-08-19 in that same commit `cc80fc3` (24 files, −4,118 lines). Four of
  the plan's ten seg classes (`floor_carpet`, `ramp`, `stairs`, `movable_obstacle`) exist
  only to serve a line that is currently history. **This contradiction gates §7 #1.**

---

## 2. What survives

Not all of 08-19/20 is waste, and it is worth being exact about which part.

* **`instances_all_<split>.json` keeps every box down to score 0.10 with no NMS.**
  `box_pass.py`'s own docstring says why: "the score population stays visible and a future
  threshold question can be answered without GPU time". So **Gold/Silver/Gray tiering can be
  built from what is already on disk, with no GPU** — the plan's §4 confidence layering is
  one analysis pass away, not one campaign away.
* **Four of the plan's ten seg classes are already supplied** by site30k_v1:
  floor→`floor_hard`, wall→`wall`, display_table→`display_table`, shelf→`display_shelf`.
* **The IR-glass finding is a real correction to Pipeline A**, not a defect report. It cost
  two night plates and it closes a route the plan would otherwise have spent a campaign on.
* **The onboarding calibration** gives pitch/roll/raw height with vfov bands for 19 of 23
  cameras. That is a free cross-check against the homography, once the homography exists.
* **The per-date structure flip** (Tao-Hsin cam03/cam04 lose their counters to `wall` on
  10/29 and 6/29 dates) is a measured property of the teacher on white-on-white. It belongs
  in Pipeline A's acceptance criteria whatever taxonomy is chosen.

---

## 3. The corrected order

**Stop drawing polygons.** `tools/site30k/zones.json` and `stamp_zones.py` stay unapplied
(nothing was stamped; `masks_prestamp/` does not exist). They are a record of a route the
plan forbids, not work in progress.

0. **Answer §7 #1 first.** Is the robot line alive? If it is dead, the seg head drops from
   10 classes to 6 (`background`, `floor`, `wall`, `display_table`, `display_shelf`,
   `glass_door`) and the dual-weight question disappears with it. **Pipeline A must not
   restart before this is answered** — that is the mistake 08-19/20 made.
1. **Homography calibration, §6 #1.** 4 ground points per camera on the undistorted plate
   (k1 = −0.225 is known fleet-wide), ~10 min each, all 48 streams. It is exact for a fixed
   camera, needs no depth model, and is not blocked by the GDINO score problem that stalls
   14 of 23 cameras on the current route. Every speed and dwell rule in `analytics/events`
   is in metres and rests on this.
2. **Pipeline B tiering, §6 #2, from existing files.** Build Gold/Silver/Gray over
   `instances_all_*.json`, sample 300 frames of each per §4's acceptance rule
   (Gold precision ≥ 95%, Silver ≥ 85%), and only then decide whether CoTracker3 is needed.
   Zero GPU until that measurement says otherwise.
3. **Overhead person retrain, §6 #3.** One variable at a time; ReID after, not with.
4. **Pipeline A last**, with the taxonomy settled by step 0, and with frames per camera set
   by the plan's own arithmetic — the design asks for 5–10 static frames per camera, and the
   campaign took 3,246.

---

## 4. Open decisions this entry does not make

1. ~~**§7 #1 — seg dual-weights / is the robot line alive?**~~ — **answered the same day; see §5.**
   What replaced it: the three contract differences the new 6-class taxonomy opens.
2. **Detection vocabulary** — the plan's 6 brand-named classes vs the shipping 4. A retrain
   in the wrong vocab silently empties `analytics/events/zones.py:346`, whose default class
   list is `("boxed_stock", "device")`.
3. **Night.** The plan has no night handling anywhere, yet `after-hours person` is the first
   entry on its VLM trigger whitelist, and `person` is measured firing on hanging packets on
   an empty IR frame. Either night enters scope with a measurement, or the trigger comes off
   the list.
4. **§7 #3 — which of (a) Pipeline B spec / (b) Pipeline C VLM prompt / (c) night GPU
   window** is the next deliverable.

---

## 5. Step 0 answered, same day: the robot line is removed

The user ruled on §7 #1 immediately after reading this audit: **the quadruped task is
dropped; only security-retail remains.** `docs/hydranet-decisions.md` §2 is rewritten from
10 seg classes to 6 — `background`, `floor`, `wall`, `display_table`, `display_shelf`,
`glass_door` — and the dual-weight question closes with it. `floor_carpet`, `ramp`,
`stairs` and `movable_obstacle` existed only to serve gait and obstacle avoidance and are
gone.

That is consistent with the code: `cc80fc3` (2026-08-19) had already deleted the quadruped
line itself, 24 files and −4,118 lines, in the same commit that set the 96-stream/15 fps
serving ceiling. The plan was the last place still carrying it.

### What this does NOT settle — three contract differences, now the gate on Pipeline A

The new 6 are not the shipping taxonomy. `hydranet_retail_security_b03_cw_xl.yaml` trains
`[void, floor, wall, column, fixture, person]`, and the differences each carry a cost:

* **`person` leaves the seg head.** `RETAIL.md` §1 and `ARCHITECTURE.md` §3 measured why it
  was there: without it, a person standing on the floor leaves a person-shaped hole of
  walkable floor. §4's "空景幀自動選取" answers this for *training*. **Inference needs the
  same gate** — the seg pass is scheduled and cached, so whoever is standing in the frame
  when it runs is baked into the zone mask until the next pass.
* **`column` is dropped.** Defensible on its record (0.86–0.88 on trained cameras, 0.00–0.51
  on unseen ones), but `RETAIL_DATA.md`'s R4, R5 and R7 are written with `column` as the
  worked example. Those rules need restating against whichever class is now the rare one.
* **`fixture` splits into `display_table` + `display_shelf`.** Every previous `fixture`
  number becomes incomparable. No checkpoint's score survives as a baseline across this
  change, and the journal for 2026-08-19 already records one case where a +0.20 on
  `fixture` was `product` merged in rather than learning.

### Robot-line remnants still in the tree (not deleted — awaiting the user's call)

Removing the task from the plan is not the same as deleting code, and two of these are not
obviously robot-only:

| file | what it is | note |
|---|---|---|
| `configs/hydranet_regnet800mf.yaml` | RUGD + RELLIS-3D off-road terrain, 12 classes | off-road only; nothing in the retail line reads it |
| `configs/hydranet_indoor.yaml` | indoor traversability, 12 classes incl. `stairs`, `threshold_ramp`, `wet_slippery` | `METHODOLOGY.md` §0 already records its `caution` class as capped by sourcing |
| `configs/hydranet_retail_cctv.yaml` | `primary_metric: traversability_mIoU`, `supervises: [traversability, terrain]` | **a retail config** that happens to score on traversability — read before deleting |
| `src/syncai_hydranet/data/label_maps_indoor.py` | the indoor traversability schemes | imported by the two configs above |

`utils/temporal.py`'s docstring also cites a traversability measurement; that is a comment
about provenance, not a dependency.
