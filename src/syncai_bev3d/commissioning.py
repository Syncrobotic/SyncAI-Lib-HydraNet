"""Onboard calibration -> `camera.json`, and the metre grid that proves it.

Build-order step 2's gate is visual: **a 1 m floor grid rendered on a real frame, one
per camera, judged by eye** (docs/PLAN.md §6). This module is both halves of that step:
the converter that turns a `runs/onboard01/<camera>.calib.json` (the 2026-08-19 fleet
scan) into the `camera.json` contract, and the renderer that draws the grid the human
judges.

The converter **refuses a camera without metres rather than writing plausible NaN**: a
calib whose `scale_source` is unmeasured has a shape (DA-V2's relative plane) but no
scale, and a `camera.json` written from it would put confident wrong metres under every
downstream event. Those cameras wait for their visual reference (door height / tile
pitch, the calib02 method) -- the refusal message says exactly that.

The grid is drawn on the **undistorted** frame, because that is the frame the
`Camera` + `GroundPlane` model lives in: `pixel_to_ground` knows nothing about the lens,
and drawing the grid on the raw frame would bake the lens error into the picture the
human is asked to approve.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from syncai_hydranet.geometry.camera_json import CameraFile, Lens
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def _teachers_of(raw: dict) -> dict[str, str] | None:
    """`{model_id: revision}` from a calib scan's provenance block, or None if it has none.

    The geometry in a `camera.json` is whatever Depth-Anything V2 said about one plate,
    so which DA-V2 is part of what the numbers mean. `onboard_camera.py` already recorded
    the model id; it records the revision beside it now, and this carries the pair into
    the file the event layer actually reads.

    None rather than `{}` when the scan predates this: a file that did not record its
    teachers is not a file that used none, and the difference decides whether a mismatch
    later is a finding or an unknown.
    """
    prov = raw.get("provenance") or {}
    model, revision = prov.get("depth_model"), prov.get("depth_model_revision")
    return {str(model): str(revision)} if model and revision else None


def from_onboard_calib(path: str | Path) -> CameraFile:
    """One `<camera>.calib.json` from the onboard scan -> the `camera.json` contract.

    Carries only what the scan measured. Zones, shelf ROIs, masks and the
    false-positive polygons are later commissioning passes that append to the file;
    this function establishes the geometry they will all be drawn in.
    """
    raw = json.loads(Path(path).read_text())
    camera_id = raw["camera"]

    problems = []
    if raw.get("height_m") is None:
        problems.append("no fitted floor (height_m is null)")
    if str(raw.get("scale_source", "unmeasured")).startswith("unmeasured"):
        problems.append(
            "scale is unmeasured -- DA-V2's plane is a shape, not metres; this camera "
            "waits for its visual reference (door height / tile pitch) before it gets a "
            "camera.json"
        )
    if problems:
        raise ValueError(f"{camera_id}: refusing to write camera.json: " + "; ".join(problems))

    h, w = raw["frame_hw_px"]
    out = CameraFile(
        camera_id=camera_id,
        image_size_px=(w, h),
        camera=Camera.from_vfov(h, w, raw["vfov_assumed_deg"]),
        plane=GroundPlane(
            height=float(raw["height_m"]),
            pitch=math.radians(float(raw["pitch_deg"])),
            roll=math.radians(float(raw["roll_deg"])),
        ),
        # The half-diagonal convention is `plate_calibration.undistort_image`'s, which is
        # what makes this k1 the same k1 the tile fit measured.
        lens=Lens(
            k1=float(raw["k1_division_model"]),
            centre_px=(w / 2.0, h / 2.0),
            radius_px=math.hypot(h, w) / 2.0,
        ),
        plate_file=raw.get("plate_used"),
        commissioned_at=raw.get("generated"),
        teachers=_teachers_of(raw),
    )
    out.validate()
    return out


def regeometry_from_calib(camera_json: str | Path, calib: str | Path) -> CameraFile:
    """Push a corrected calibration into a camera.json without losing what came after it.

    **The pipeline was one-directional and this is the missing return leg.**
    :func:`from_onboard_calib` writes a file carrying only what the scan measured, so
    re-running it on a camera that has since been commissioned discards the later passes:
    on Taichung-cam10 that is 13 mask files, 14 zones, 5 shelf ROIs and 3 false-positive
    polygons. There was therefore no supported way to correct the geometry of a
    commissioned camera, and on 2026-09-01 seven of the eight shipped cameras had drifted
    without anything saying so -- Taichung-cam10 by 0.30 m, its camera.json still reading
    2.87 m against a calib.json that had been re-fitted to 2.57 m, both stamped with the
    same `commissioned_at`.

    **Zones are in metres and therefore move with the geometry.** They are rescaled rather
    than preserved: with pitch and roll unchanged, `pixel_to_ground` puts a floor pixel at
    a distance proportional to the plane height, so multiplying every zone point by
    `height_new / height_old` is exact, not an approximation. Everything in pixel space --
    mask files, shelf ROIs, false-positive polygons -- is unaffected by a height change and
    is carried across untouched.

    **A pitch or roll change is refused.** That linearity is what makes the rescale exact,
    and it does not hold when the plane's orientation moves: the zones would have to be
    re-derived from the masks rather than scaled. Refusing is the honest answer, because a
    scaled zone under a changed pitch is wrong in a way nothing downstream would notice.
    """
    existing = CameraFile.load(camera_json)
    fresh = from_onboard_calib(calib)

    d_pitch = abs(math.degrees(fresh.plane.pitch - existing.plane.pitch))
    d_roll = abs(math.degrees(fresh.plane.roll - existing.plane.roll))
    if max(d_pitch, d_roll) > 0.05:
        raise ValueError(
            f"{existing.camera_id}: pitch moved {d_pitch:.2f} deg and roll {d_roll:.2f} deg. "
            "The metre zones can only be rescaled while the plane's orientation holds; "
            "re-derive them from the masks instead of scaling them."
        )

    ratio = fresh.plane.height / existing.plane.height
    zones = tuple(
        replace(z, points_m=tuple((x * ratio, z_m * ratio) for x, z_m in z.points_m))
        for z in existing.zones
    )
    return replace(
        existing,
        camera=fresh.camera,
        plane=fresh.plane,
        lens=fresh.lens,
        plate_file=fresh.plate_file,
        plate_sha256=fresh.plate_sha256,
        commissioned_at=fresh.commissioned_at,
        teachers=fresh.teachers,
        zones=zones,
    )
