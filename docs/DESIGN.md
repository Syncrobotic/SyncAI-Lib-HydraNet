> **Superseded (2026-08-22) by [PLAN.md](PLAN.md).** This draft kept the terrain head in
> the per-frame path; walking the placement rule over every shipped output showed nothing
> requires per-frame dense segmentation, and the second head became pose. Kept for the audit
> trail. **Do not build from this file.**

# Design: one vision system for security and retail, and how it grows

Written 2026-08-22, replacing the architecture sketch in `hydranet-decisions.md` §1. That
sketch described *what to build*; it did not say *where a new capability goes*, and the
2026-08-19/20 work went wrong in exactly that gap — glass became a trained class when it is
a drawn polygon, products moved into the dense head with no decision recorded, and a
campaign ran before the decision that gated it.

So this document leads with the placement rule, not the network. The network is four
paragraphs; the rule is what keeps the next six months from repeating last week.

---

## 1. The rule: the cheapest instrument that can answer the question

For any new capability, walk this ladder from the top and stop at the first rung that can
answer it. **Never start at the bottom.**

| # | instrument | what it costs | what it needs |
|---|---|---|---|
| 1 | **a config value** — a polygon, a threshold, a schedule | nothing | someone to write it once per camera |
| 2 | **a geometric derivation** from existing outputs, in metres | nothing | the ground plane (§3, L1) |
| 3 | **a rule over track time series** — CPU | nothing | tracks in metres |
| 4 | **a branch on the person-crop encoder** — 2nd stage, ~1/45 of frame rate | a small model, crop labels | detections that are already there |
| 5 | **a track-level temporal model** — CPU, <100K params | clip labels (2–3 s each) | tracks |
| 6 | **a new head on the shared trunk** | **+0.27M params, +3% latency (measured)** | dense or box labels |
| 7 | **a new class in an existing head** | invalidates every checkpoint's comparability | labels *and* a re-baseline |

Rung 7 is the most expensive thing in this project and it is the one that looks cheapest
from a config file. `column` is the worked example: it scores 0.86–0.88 on cameras a run
trained on and 0.00–0.51 on cameras it never saw, and `fixture` is the other — a measured
+0.20 on it turned out to be `product` merged in rather than learning.

**The test that catches a misplacement:** if the answer is a constant on a fixed camera, it
belongs on rung 1. Glass, zone boundaries, "which shelf is the premium shelf", store
opening hours, the till's location — all constants. None of them are classes.

---

## 2. The five layers

```
L0  pixels -> boxes + terrain      one shared-trunk net, every frame, GPU
L1  boxes -> tracks in METRES      tracker + ground plane, every frame, CPU
L2  tracks -> per-person facts     crop heads, every ~3 s per track, GPU
L3  facts -> events                rules in metres and seconds, CPU
L4  events -> judgement            VLM on trigger, GPU queue
```

Two invariants, both already enforced in the tree:

* **Nothing crosses from L3/L4 down into L0.** A threshold a store manager changes on a
  Tuesday is a function argument. `RETAIL.md` §5 argues this at length and
  `analytics/events/zones.py` is built that way — every event there reads only floor
  polygons in metres and track positions, and touches no model output.
* **Heads read only the neck, never each other** (`models/hydranet.py` design rule 2), and
  `forward` is pure convolution so the ONNX/TensorRT export stays clean. This is what makes
  rung 6 cost 3% instead of a rewrite.

---

## 3. L0, frozen

| part | value | why this and not something else |
|---|---|---|
| backbone | RegNetX-800MF | `DEPLOY.md` swept the alternatives: `regnet_x_400mf` inside noise |
| neck | BiFPN ×2, 96 ch, 5 levels (P3–P7) | 64 ch measured *slower* (2.45 vs 2.31 ms); `num_repeats: 1` inside noise |
| input | **640 × 1120** FP16 | the ladder below. Do not reduce it |
| head: detection | FCOS, 4 classes — `person`, `bag`, `boxed_stock`, `device` | one vocabulary, ids assigned once; `data/label_maps_retail_security.py` |
| head: terrain | semantic FPN over P3–P5, 6 classes — `void`, `floor`, `wall`, `display_table`, `display_shelf`, `person` | 3% of latency; runs every frame, no scheduling |

**Measured tonight** (RTX PRO 6000, idle, FP16 eager, 640×1120, random weights, batch 16):

| heads | params | trunk share | frames/s |
|---|---:|---:|---:|
| terrain + detection | 7.97 M | 88.2% | **940** |
| detection only | 7.70 M | 91.3% | 970 |
| terrain only | 7.30 M | 96.2% | 1221 |

So: trunk 7.03 M, terrain head 0.27 M, detection head 0.67 M. **Terrain costs 3% of
throughput; detection costs 23%.** Batch saturates at 8–16 — batch 32 is slower than 16.
`channels_last` is slower than contiguous (865 vs 938). Eager is a floor, not a ceiling:
TensorRT and CUDA graphs are not in this number.

### Why the input stays at 640×1120

The site objects are small and FCOS's finest level is stride 8, so a box whose short side
is under 8 px does not span one feature cell — not hard to learn, unlearnable.

| input | canvas content | median object short side | boxes under 8 px |
|---|---|---|---|
| 512 × 640 | 70.3% | 17.3 px | 8.0–12.5% |
| 512 × 896 | 98.4% | 24.3 px | 1.6–2.0% |
| **640 × 1120** | 98.4% | **~30 px** | — |

And the jump was measured on a pair of configs differing in nothing else:
`detection_mAP/site_boxes` **0.1210 → 0.1469, +21% relative** — a larger gain than
batch03's extra annotation (+0.013) and class weights (+0.002) combined.

### Two classes that are deliberately absent from L0

* **`glass_door` is not a class.** Measured unteachable across two sessions and 112 frames
  (12 phrasings × 7 thresholds, ≤0.12%), then four more routes killed on 2026-08-20:
  day/night differential, NCC inside `wall`, cross-date plate deviation, metric depth. It
  is also a constant on a fixed camera and only 2 of 9 cameras see it. **Rung 1.**
* **product classes are not in terrain.** `laptop`/`tablet`/`phone`/`boxed_stock` are box
  classes; they move, they occlude each other, and a dense head cannot count them.

---

## 4. Where the next capabilities land

This is the extensibility claim, and it is checkable: for each item, the rung is named and
nothing lands on rung 6 or 7.

### Human behaviour analysis — one capability per rung, not one model

| behaviour | rung | needs |
|---|---|---|
| stand / walk / run | 3 — floor speed in m/s | calibration only. **Already implemented** |
| loiter, dwell, queue time | 3 — dwell inside a polygon | calibration + a polygon |
| line crossing, zone intrusion, tailgating, crowd | 3 | **already implemented** in `events/zones.py` |
| reach to shelf | 3 + pose | pose keypoints (pilot passed 2026-08-19) + the cached fixture mask |
| fall | 3 + a handful of staged clips to verify | bbox aspect and height over time |
| sit / crouch / bend | **5** — track-level 1D-CNN, <100K params, CPU | clip labels, a few hundred |
| product handling, concealment, intent | **4** — VLM on trigger | the event schema, and nothing else |

**Nothing in this table touches L0.** That is the point: behaviour is a time-series
question, and a per-frame dense head is the wrong instrument for it — `run` and `walk` are
indistinguishable in one frame.

### Spatial analysis — no new model at all

Heatmaps, path flow, dwell maps, zone occupancy, shelf attention, queue length, entry
funnels: every one of them is L1 output — a track's floor position in metres — accumulated
over time. **Rung 2 or 3 for all of it.**

The single prerequisite is the metric ground plane per camera. That is why calibration is
step 3 below and not step 6: it is not "the thing speed rules need", it is the denominator
of every spatial and behavioural number the product will ever sell.

### Later, if they are ever needed

* cross-camera ReID → rung 4, a branch on the crop encoder that already exists
  (`models/crop_encoder.py`)
* depth / free space → a head already exists (`models/heads/depth.py`), rung 6
* open-vocabulary product classes → `models/heads/text_classifier.py`, rung 6, and it
  avoids rung 7 by construction — new product names need no new channel

---

## 5. The step plan: one artefact and one gate per step

The failure of the last cycle was a long chain with no gates. Each step here produces one
thing that can be looked at, and states what makes it fail.

| # | step | artefact | gate |
|---|---|---|---|
| 1 | freeze §1–§3 | this document | you agree with the placement rule and the L0 table |
| 2 | throughput ceiling | one number: frames/s under TensorRT + CUDA graphs vs 1,440 | ≥1,440 → resolution stays. Below → we know exactly what to trade, measured |
| 3 | metric calibration, all cameras | per-camera homography + **a 1 m floor grid rendered on a real frame, one image per camera** | the grid looks right to your eye on every camera |
| 4 | detection first | `site_boxes` mAP, plus Gold/Silver/Gray tiers built from the existing `instances_all_*.json` at zero GPU cost | Gold precision ≥95%, Silver ≥85% on a 300-frame sample |
| 5 | terrain, 6 classes | per-class IoU with **at least 2 test cameras** per class | `RETAIL_DATA.md` R4/R7 satisfiable — otherwise the number is not reported |
| 6 | L3 end to end, one camera | an event log readable against the video it came from | the events match what you see when you watch the clip |

Steps 2 and 3 are independent of each other and of everything below; 4 before 5 is the
plan's own §5 #7 (detection converges, then the seg head). Nothing after 6 starts until 6
passes: crop heads, the behaviour model and the VLM all consume L3's output, so a wrong L3
makes all three unmeasurable.

**Data discipline that applies to every step:** more *cameras*, not more frames from the
same ones. `METHODOLOGY.md` §0 measured this — `column` at 0.86–0.88 on trained cameras and
0.00–0.51 on unseen ones — and the 2026-08-20 campaign is the counter-example: 29,211
frames, 10.4 GPU-hours, 9 cameras, **all 9 already annotated**, union unchanged at 24.

---

## 6. Open, and each one blocks a specific step

1. **The 1,440 frames/s target has never been measured** (step 2). Eager is 940. `trtexec`
   is not installed on this machine — only TensorRT's Python bindings — so
   `scripts/bench_pro6000.sh` cannot run as written.
2. **`fixture` splitting into `display_table` + `display_shelf`** (step 5) makes every
   previous `fixture` number incomparable. No checkpoint survives as a baseline.
3. **`column` leaving the taxonomy** (step 5) removes the worked example that
   `RETAIL_DATA.md` R4/R5/R7 are written around. Those rules need restating.
4. **Night is unscoped.** `person` is measured firing on hanging packets on an empty IR
   frame, and `after-hours person` is the first entry on the VLM trigger whitelist. Either
   night enters with a measurement, or the trigger comes off the list.
5. **No site number is an accuracy** — everything is agreement with SAM 3 / GDINO. §4 of the
   plan forbids manual annotation; `METHODOLOGY.md` §0 says the binding constraint is a
   human-corrected test split. These two cannot both hold, and the resolution decides what
   "done" means for every step above.
