# Research direction: a BEV occupancy network for the quadruped, and the one thing it turns on

**Scope.** Product A, the Lite3 robot line ([`deploy/robot/`](../deploy/robot/)). The
north star is the modern occupancy-network perception stack — multi-camera → shared
backbone → geometry-conditioned BEV cross-attention → temporal transformer → a fan of
dense heads — adapted for a legged robot. This document does not argue the architecture;
it is the state of the art and it is settled. It argues about the **one prerequisite that
decides whether we can build it at all**, and lays out the research program around that.

## The target

```
each camera → shared backbone (RegNetX, already ours)
            → BEV cross-attention (learned projection conditioned on intrinsics/extrinsics;
                                   LiDAR point features injected when present)
            → temporal transformer (aggregate past N frames, ego-motion compensated)
            → heads:
                ① Occupancy (3D)
                ② Occupancy flow (dynamic/static + velocity)   ← the moving-obstacle answer
                ③ Semantics (person / vehicle / door / glass / …)
                ④ Traversability (legged: grass, gravel, stair walkability)
                ⑤ Ground height (elevation map for foothold planning)
```

Heads ④ and ⑤ are the legged-specific ones and they are the reason this is not just a
car's occupancy net; a body-level free-space map is not a foothold plan.

## The thesis: the research is supervision, not architecture

The transformer is a solved shape. What we do not have is **3D ground truth on a monocular
robot**, and ①②⑤ cannot be trained without it. So the whole program reduces to one
question: *can we label occupancy, flow and height without a 3D sensor?* Everything else
is engineering that follows once that answer is yes.

Three weak supervisors were wired up in the same week this document was written, and they
are exactly the anchors that make the answer plausible:

- **odometry** — `pos_world` / `vel_world` / `vel_body`, from the `0x0901` state packet
  (see the handoff journal). Legged-inertial pose we do not have to build.
- **ultrasound ×2** — `ultrasound[2]` in the same packet. Two metric range points per
  frame; weak, but real, and free.
- **HydraNet's own heads** — `terrain` and `detection` already produce ③④'s labels.

## The auto-labeling pipeline (the crux)

Turn "no 3D sensor" into "no hand-drawn 3D labels":

```
robot video + odometry + ultrasound[2] + HydraNet semantics
  ├─ (A) monocular-depth teacher (Depth-Anything V2, offline)  → per-frame relative depth
  ├─ (B) temporal SfM / multi-view, odometry as the pose prior → metric geometry + scale
  ├─ (C) ultrasound[2] + odometry z                            → weak metric anchors that
  │                                                              scale (A)'s relative depth
  └─ (D) HydraNet terrain / detection                          → semantic labels for ①③④
        ↓ offline fusion
   pseudo-3D labels: occupancy voxels / ground-height / dynamic-static flow
        ↓
   train a full-size model (server / Orin) → distill to a small student (RK3588)
```

The teacher runs **once, offline, on the server** — never on the robot. The robot ever
only runs a distilled student. That is the Tesla/Wayve pattern and it is what keeps a
6-TOPS NPU in the conversation at all.

## Model evolution: grow HydraNet, do not rebuild it

One head at a time, each with a stop condition. The RegNetX backbone stays; the
`models/heads/registry.py` abstraction was built (its own docstring says so) for exactly
this — "a third head family: depth, keypoints, whatever comes."

| step | add | supervision | success gate |
|---|---|---|---|
| E1 | BEV lift (geometric projection first, then learned cross-attn) | geometry (`geometry/bev.py`, ours) | BEV-space seg mIoU ≥ the current 2D-then-project baseline |
| E2 | **⑤ ground-height head** | (A)+(B)+(C) pseudo-depth | height MAE vs held-out ultrasound / SfM |
| E3 | **① occupancy head** | pseudo-depth, voxelized | occupancy IoU vs SfM reconstruction |
| E4 | temporal transformer + **② flow** | odometry ego-motion + photometric consistency | dynamic/static F1; static-scene stability |
| E5 | **④ material traversability** (grass/gravel/stair) | collect + annotate ([METHODOLOGY.md](METHODOLOGY.md) ranks it) | per-class IoU; lift `stairs` off 0.32 |
| E6 | distill → RK3588 student | the full model as teacher | student ≥ 10 fps with metrics retained |

## First milestone — E-prep, and it is a decision gate not a demo

Before any head, answer the biggest unknown with a number: **is the monocular-depth
teacher good enough on this camera to bootstrap the labels?**

- Run Depth-Anything V2 over existing robot video (offline; RKNN not required for the
  teacher). Calibrate its relative depth against the two ultrasound returns and odometry
  `z`, and measure the residual.
- **Hypothesis:** teacher depth + our weak anchors give a metric residual below some
  threshold the campaign sets.
- **If it holds:** the auto-labeling floor is solid; E2/E3 have labels to train on.
- **If it does not:** this is the quantified case for buying a depth sensor
  (RealSense D435i / Orbbec / the Lite3 LiDAR variant). A number, not a hunch — and the
  same sensor then supervises ①②⑤ directly and closes the stairs/glass safety gap.

E-prep touches no hardware and buys nothing; it consumes footage we already have and
answers the question the whole direction turns on.

## Preconditions carried in from the platform work (do not rediscover these)

- **Per-frame ground plane, never a fixed pose.** A walking camera pitches *and rolls*
  every step; roll error rotates the floor and puts obstacles on the wrong side. The BEV
  now builds `GroundPlane` per frame from live attitude; `geometry/fit_ground_plane`
  (written, tested, unused) is the drop-in the day depth lands. See
  [GROUND_PROJECTION.md](GROUND_PROJECTION.md).
- **Traversability's `caution`/`stairs` are near-untrained** (0.33 / 0.32; three of four
  caution classes have zero examples). ~A quarter of a ceiling comes back `go`. Avoidance
  cannot key on ④ until those classes have data — a data precondition, not a planner one.
- **Fork: who owns locomotion.** The factory `jy_exe` has an official *autonomous mode*
  (`0x21010C03`) that follows velocity from a perception host, so body-level autonomy needs
  no policy replacement. ⑤'s foothold plan does — it requires a locomotion policy that
  *consumes* an elevation map (e.g. `Lite3_rl_deploy`), which the velocity interface does
  not expose. Decide this before ⑤, because it changes the data campaign.
- **Two compute tiers, one training pipeline.** RK3588 for the robot student, Orin for the
  retail/security tier ([DEPLOY_JETSON.md](DEPLOY_JETSON.md)). The auto-labeler and the
  full model are shared; only the distilled student differs. This is the same one-core /
  two-products split the repo is organised around.

## Why this unifies the two products

A patrolling quadruped running this stack is a **mobile instance of product B**: the same
person/detection/terrain perception and the same L1/L2 events ([RETAIL_SECURITY.md](RETAIL_SECURITY.md)),
now covering the aisles a fixed camera cannot see. The occupancy net is where product A's
navigation and product B's analytics stop being two problems.
