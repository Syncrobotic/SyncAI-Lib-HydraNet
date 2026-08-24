# Vision: one camera, two questions

## 1. What this is

A vision system that reads **fixed, already-installed store CCTV** — no LiDAR, no new
hardware, no fisheye retrofit — and answers two families of question from the same frames:

* **Security / loss prevention.** Who entered where, how many, how long, what did they do,
  and did stock leave without being paid for.
* **Retail analytics.** What merchandise draws attention, where do people walk, where do
  they stop, how long do they queue.

These are not two products. They are two readings of one camera, one export, one latency
budget. Every design decision below follows from refusing to build them twice.

The first vertical is an **Apple-reseller store chain** — white fixtures, glass shopfronts,
small high-value devices on open display tables, boxed stock behind the counter. That is a
hard case on purpose: it is the environment where generic web-trained models fail hardest,
and solving it generalises downward to easier stores.

## 2. Who it is for

| user | what they open the system for |
|---|---|
| **loss-prevention operator** | a short queue of *actionable* alerts, each with the clip and the reason |
| **store manager** | yesterday's footfall, dwell, queue times, which display was touched |
| **regional / HQ** | the same numbers across stores, comparable |
| **installer** | a camera onboarded in under 20 minutes without a data scientist |

The installer is the user most systems forget, and onboarding cost is what decides whether
this scales past one chain. §5 of [PLAN.md](PLAN.md) treats commissioning as a first-class
product surface, not a setup chore.

## 3. What success is — one number, and it is not mAP

> **Actionable alerts per camera per day, and the incidents missed.**

An operator who receives more than a handful of alerts per camera per day stops reading
them, and a system nobody reads has zero value regardless of its mAP. Every threshold,
every confidence tier and the whole VLM triggering layer exist to serve this number.

Model metrics (mAP, IoU, MOTA) are **instruments**, not goals. They are how we debug a
regression; they are never what we report as success.

The second number, and the harder one:

> **No site figure is an accuracy until a human has graded it.**

Today every site number in this project is an *agreement* with SAM 3 or Grounding DINO —
a model trained on a teacher and scored against the same teacher shares its errors. The
resolution is in [PLAN.md](PLAN.md) §6: operators grading real alerts is the labelling
mechanism, and it is free because they were going to look at the alerts anyway.

## 4. What it must never do

These are the rules that keep the system coherent as it grows. Each one is here because
breaking it has already cost this project time.

1. **Never put an answer in the weights that belongs in a config.** A zone boundary, a
   dwell threshold, opening hours, which shelf is premium, where the glass is — all of them
   are constants on a fixed camera. A constant is stored, not learned. Glass cost two
   sessions and 112 frames of measurement before this rule was applied to it.
2. **Never let the rule layer reach into the model.** "Four minutes is loitering" is a
   function argument a manager changes on a Tuesday. It is not a class.
3. **Never report a per-class number resting on one camera.** One fixed camera is one scene
   measured N times, not N samples. Measured: `column` scores 0.86–0.88 on cameras a run
   trained on and 0.00–0.51 on cameras it never saw.
4. **Never buy coverage with more frames from the same cameras.** More *cameras* fixes
   generalisation; more frames from the same ones does not. The 2026-08-20 campaign is the
   counter-example: 29,211 frames, 10.4 GPU-hours, nine cameras, all nine already annotated.
5. **Never infer a customer attribute we would not defend in court.** Age and gender
   inference on customers is regulated (BIPA, GDPR, CCPA) and is off by default.
   `staff / customer` is not the same thing — it is a uniform, it is the highest-value
   attribute in the system, and it carries none of that exposure.

## 5. Non-goals, stated so they stop being re-litigated

| not doing | why |
|---|---|
| quadruped / robot autonomy | removed 2026-08-22. The line was deleted in code on 2026-08-19 (`cc80fc3`, −4,118 lines) |
| face recognition, face-based identity | legal exposure out of proportion to value; nothing in §1 needs it |
| cross-store customer tracking | privacy landmine; single-store re-linking covers the events we sell |
| SKU-level checkout ("just walk out") | the industry's graveyard. We detect *change and attribution*, not baskets |
| new camera hardware | the premise is the CCTV already on the ceiling |
| per-frame dense scene understanding | static structure is a per-camera constant; see rule 1 |

## 6. The extensibility promise, stated so it can be tested

The two capabilities asked for next are **human behaviour analysis** and **spatial
analysis**. The promise is specific:

> **Neither requires a change to the per-frame network.**

* **Spatial analysis** — heatmaps, path flow, dwell maps, zone occupancy, shelf attention,
  queue length, entry funnels — is all accumulated floor positions in metres. No new model.
  Its single prerequisite is the metric ground plane per camera, which is why calibration is
  the first build step and not the sixth.
* **Behaviour analysis** decomposes by instrument, not into one model: walk/run/loiter are
  speed and dwell rules; sit/crouch/fall is a <100K-parameter temporal model on the CPU;
  intent and concealment are a VLM on trigger. Only pose keypoints touch the GPU per frame,
  and pose is already the next head in the plan.

If a future capability *does* force a change to L0, that is the signal to re-read §4 rule 1
before writing code.

## 7. The constraints that shape everything

| constraint | value | status |
|---|---|---|
| streams per card, real time | 96 × 15 fps = 1,440 frames/s | **target, never measured.** Eager PyTorch reaches 941 |
| cameras available today | 48 in the fleet, 24 annotated, part of the fleet is back-office | measured |
| manual pixel annotation | **none.** Humans do accept/reject, plus 4 calibration clicks per camera | policy |
| new hardware | none | policy |
| model size | ~8 M parameters, FP16 | measured: 7.97 M |

## 8. Where this document sits

* **This file** — why the system exists and what success means. Change it rarely.
* [PLAN.md](PLAN.md) — the model design, the data strategy, the build order and the gates.
  Change it whenever a measurement says to.
* [METHODOLOGY.md](METHODOLOGY.md), [RETAIL.md](RETAIL.md),
  [RETAIL_DATA.md](RETAIL_DATA.md), [DEPLOY.md](DEPLOY.md) — the standing reference for
  process, output contract, split discipline and deployment.
* `hydranet-decisions.md` and `DESIGN.md` are **superseded by PLAN.md**; they are kept for
  the audit trail in `journal/2026-08-22-drift-audit-and-course-correction.md`.
