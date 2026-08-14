# Ground projection

A traversability mask answers "is this pixel walkable". A navigation stack asks something
else: "is there a path three metres ahead". Those are the same information in different
coordinates, and the conversion is geometry, not learning.

`assets/bev_ground_projection.gif` shows the pair — the mask, and the mask projected onto
the floor in metres.

## How it works

For a flat floor, every ground cell has exactly one image pixel. The projection maps
**backwards**, from each BEV cell to the pixel that sees it, because the forward direction
leaves holes in the far field where one pixel covers many cells.

Each cell needs the camera's intrinsics and its pose above the floor. Intrinsics come from
the session's `*_calibration.json`. The pose does **not** come from a tape measure:

> Fit a plane to the lower half of the depth return, every frame. Its normal is pitch and
> roll; its distance is camera height.

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

## Status

The projection runs offline against recorded clips. It is not wired into ROS, and the
figures published so far assume the camera rather than measuring it, which the captions
say. What unblocks the measured version is a session recorded on the robot — the same
capture that the annotation pipeline and the 3D work are both waiting on.
