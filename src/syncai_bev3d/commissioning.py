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
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_hydranet.geometry.camera_json import CameraFile, Lens
from syncai_hydranet.geometry.ground import Camera, GroundPlane, ground_to_pixel

from .plate_calibration import undistort_image


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


def render_metre_grid(
    camera_file: CameraFile,
    frame: Image.Image,
    *,
    extent_m: float = 8.0,
    step_m: float = 1.0,
    z_near_m: float = 0.3,
) -> Image.Image:
    """Draw the floor grid this calibration believes in, for a human to disbelieve.

    Grid lines every `step_m` in floor metres, the x = 0 axis (straight ahead under the
    camera) emphasised, and a range label where each metre line crosses it. What the
    judge checks: do the lines land on the tile joints, is a known ~1 m object about one
    cell, does the grid stay parallel to the fixtures it runs along.
    """
    w, h = camera_file.image_size_px
    if frame.size != (w, h):
        frame = frame.resize((w, h), Image.Resampling.LANCZOS)
    art = np.asarray(frame.convert("RGB"))
    if camera_file.lens is not None:
        art = undistort_image(art, camera_file.lens.k1)
    img = Image.fromarray(art)
    draw = ImageDraw.Draw(img, "RGBA")
    cam, plane = camera_file.camera, camera_file.plane

    def polyline(xs: np.ndarray, zs: np.ndarray, colour, width):
        u, v, depth = ground_to_pixel(xs, zs, cam, plane)
        ok = np.isfinite(u) & np.isfinite(v) & (depth > 0)
        # Split at every invisible sample so a line never bridges the horizon.
        run: list[tuple[float, float]] = []
        for ui, vi, oki in zip(u, v, ok, strict=True):
            if oki and -w <= ui <= 2 * w and -h <= vi <= 2 * h:
                run.append((float(ui), float(vi)))
            else:
                if len(run) > 1:
                    draw.line(run, fill=colour, width=width)
                run = []
        if len(run) > 1:
            draw.line(run, fill=colour, width=width)

    n = int(extent_m / step_m)
    zs_dense = np.linspace(z_near_m, extent_m, 240)
    xs_dense = np.linspace(-extent_m, extent_m, 480)
    minor = (90, 220, 120, 160)
    axis = (255, 210, 60, 220)
    for i in range(-n, n + 1):
        x = i * step_m
        polyline(
            np.full_like(zs_dense, x), zs_dense, axis if i == 0 else minor, 3 if i == 0 else 1
        )
    for j in range(1, n + 1):
        z = j * step_m
        polyline(xs_dense, np.full_like(xs_dense, z), minor, 1)
        u, v, depth = ground_to_pixel(np.array([0.0]), np.array([z]), cam, plane)
        if depth[0] > 0 and 0 <= u[0] < w and 0 <= v[0] < h:
            draw.text(
                (float(u[0]) + 4, float(v[0]) - 12), f"{z:g} m", fill=(255, 255, 255, 230)
            )

    stamp = (
        f"{camera_file.camera_id}  h={plane.height:.2f} m  "
        f"pitch={math.degrees(plane.pitch):.1f}\N{DEGREE SIGN}  "
        f"roll={math.degrees(plane.roll):.1f}\N{DEGREE SIGN}  grid={step_m:g} m  "
        f"(undistorted frame; scale from person-height prior -- judge, don't trust)"
    )
    draw.rectangle([0, h - 18, w, h], fill=(0, 0, 0, 170))
    draw.text((6, h - 15), stamp, fill=(255, 255, 255, 240))
    return img
