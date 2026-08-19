# Ground projection

> **This file serves both lines and the split runs down the middle of it.** The
> per-frame plane fit below is the *quadruped's* need — a walking camera pitches and
> rolls every step — and the quadruped is the secondary line. The **fixed-camera**
> half, under *Running it*, is the main CCTV line: on a camera that never moves the
> whole thing is an install-time calibration artifact. Neither half is deprecated;
> they are answers to the same question for two platforms.

A traversability mask answers "is this pixel walkable". A navigation stack asks something
else: "is there a path three metres ahead". Those are the same information in different
coordinates, and the conversion is geometry, not learning.

The right-hand panel of `assets/retail_cctv_scene.gif` shows the result — the free space
and the detections placed on the floor in metres, drawn as a room rather than as a plot.

There used to be a second figure here, `assets/bev_ground_projection.gif`, showing the
flat top-down version of the same projection. That renderer was removed once `bev3d`
became the default panel, so the figure showed output the code can no longer produce and
went with it. The source footage for it — a handheld pass through a building lobby — was
never in the repository, so it could not be re-rendered.

## How it works

For a flat floor, every ground cell has exactly one image pixel. The projection maps
**backwards**, from each BEV cell to the pixel that sees it, because the forward direction
leaves holes in the far field where one pixel covers many cells.

Each cell needs the camera's intrinsics and its pose above the floor. Intrinsics come from
the session's `*_calibration.json`. The pose is meant not to come from a tape measure:

> Fit a plane to the lower half of the depth return, every frame. Its normal is pitch and
> roll; its distance is camera height.

**That is the design, and it is not what runs today.** `fit_ground_plane` is implemented
and tested (`geometry/ground.py`, six tests including a recovery check against synthetic
floors at three known poses) and **has no caller outside those tests**. Everything that
executes — `hydranet-scene`, and the archived clips rendered with it — takes the pose from
`--camera-height` and `--pitch` and prints on every frame that it did. Read the rest of
this file as what the module is for; read the Status section for what has been run.

That matters on a quadruped. A wheeled robot's camera pose is a constant you can measure
once; a walking one pitches and rolls with every step, so a fixed homography smears in a
way that gets worse the further out you look. Fitting per frame is immune to it, and to
someone re-mounting the camera. It beats using the IMU, which gives angles but not height —
and height changes with gait too.

## Objects

A box gives no range. Where its **bottom edge meets the floor** does, because that point is
on the plane we just fitted. That is the only ground position a single camera recovers, and
it is wrong for anything not standing on the floor — a wall-mounted screen, a mug on a
table. The projection places objects; it does not measure them.

## The residual is the second output

Points that no plane explains are as useful as the plane. A polished floor scatters the
projector's infrared pattern away, so it returns no depth at all: on `10.8.140.130`,
**10.6% of the pixels the model called walkable came back with no depth**. Those are the
pixels that fool the camera and the depth sensor at the same time, which is the failure
this deployment should fear most.

The residual is continuous, not a binary "had depth or not", so it should separate two
cases that look alike in the mask: a reflective floor still lies **on** the ground plane,
while glass does not. Whether it actually separates them is untested and worth one
walk-through to find out.

## Running it

```bash
hydranet-scene --config configs/hydranet_indoor.yaml --checkpoint runs/.../best.pt \
    --input clip.mp4 --output clip_bev.mp4          # camera view + the floor panel

hydranet-scene --config ... --checkpoint ... --input frame.jpg --json scene.json
```

`--json` is the payload the module exists to produce: metres and class ids, no colours,
for a renderer, an RViz overlay or a costmap publisher to share. One JSON object for an
image, one per line for a clip.

The pose flags are `--camera-height`, `--pitch` and `--vfov`, and their values are printed
on every rendered frame. `--pose-note` replaces that caption, for the case where the pose
came from a fit rather than a guess — `scripts/fit_camera_from_people.py` recovers a fixed
camera's height and pitch from detected people.

**A second fitted pose landed 2026-08-19, and it splits the problem in two.**
`scripts/calibrate_from_plate.py` runs a metric monocular depth model over a camera's
temporal-median background plate and fits the floor by RANSAC — one plate, one inference,
no people needed. What comes back divides cleanly:

| | verdict | evidence (`runs/calib01`) |
|---|---|---|
| **orientation** | **viable** | pitch recovered to **−0.7°** against the Taichung-cam01 fitted anchor, and it surfaced Kaohsiung-cam04's real **−12.9° mounting roll** that nothing else had caught |
| **scale** | **not viable** | 3.84 m against a 2.38 m anchor — a **1.61×** overestimate at ceiling-CCTV viewpoint, worse than the same model's 1.18× on NYUv2 |
| **vfov** | **the dominant unknown** | 55→85° swings fitted height by ~0.9 m and scale 0.96→0.56 — which is why it is held as an input and never fitted jointly |

**So the metre never comes from the depth model.** The recipe that survives is: tile grid →
lens parameters; plate + depth → plane orientation and roll; **one known length → the
metre.** And on a fixed camera the whole thing is an **install-time calibration artifact**,
not a per-frame prediction — which is the opposite of the per-frame fit this document argues
for on a walking robot, and both are right for their own platform.

**The known length does not need a tape measure (measured 2026-08-19,
`runs/calib02_priors`).** Standard-dimension priors read off the plates lock scale to
**±5–8%** — ±3–6% where a grout grid pins vfov: Taichung-cam01 at the 45 cm tile prior
gives H = 2.40 m against the 2.38 m anchor (+1%); Kaohsiung-cam04's tile@60 and door@2.0 m
agree within 1.3%. Two rules carry the result: prefer **menu-free references** (door
height, standard mats) or two independent references that cross-cut, because tile pitch has
a size menu (45/50/60/80 cm — pick wrong and the error is +35% while every residual still
looks perfect); and measurement precision is not the bottleneck (same-camera references
agree ≤3%), the prior is. Good enough for line-cross and social-distance rules; occupancy
density squares the error (±10–16%), so a real measured length per store remains the
upgrade path there.

## Status

The projection runs offline against recorded clips. It is not wired into ROS; the live ROS
view (`scripts/robot/live_view_ros.py`) builds its scene from registered metric depth by
back-projection instead, and deliberately fits no plane. The figures published so far
assume the camera rather than measuring it, which the captions say. What unblocks the
measured version — and gives `fit_ground_plane` its first caller — is a session recorded on
the robot, the same capture that the annotation pipeline and the 3D work are both waiting
on.
