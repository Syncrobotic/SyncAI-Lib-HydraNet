# Ground projection

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
camera's height and pitch from detected people, which is the only pose here that is fitted
to data today.

## Status

The projection runs offline against recorded clips. It is not wired into ROS; the live ROS
view (`scripts/live_view_ros.py`) builds its scene from registered metric depth by
back-projection instead, and deliberately fits no plane. The figures published so far
assume the camera rather than measuring it, which the captions say. What unblocks the
measured version — and gives `fit_ground_plane` its first caller — is a session recorded on
the robot, the same capture that the annotation pipeline and the 3D work are both waiting
on.
