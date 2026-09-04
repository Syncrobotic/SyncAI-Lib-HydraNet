"""Score a depth source against the floor it is supposed to reproduce -- no labels.

Every metre in this repo rests on the 1.70 m person prior and a vfov copied from one
camera's tile grid (PLAN 7.19). That makes "is this depth model better" a question
nobody could answer, so it was never asked: `Depth-Anything-V2-Metric-Indoor-Large` has
been the world model's depth since commissioning began, and the one time an alternative
was tried (MapAnything, 2026-08-30) it read *higher* on every fixture and looked like a
fix for the white-surface collapse.

It was not. **The floor is the witness.** A commissioned camera carries a ground plane
and a `walkable` mask, so the height of the floor above the floor is known to be zero
without anyone measuring anything. Unprojecting MapAnything's depth through the same
plane put the floor 0.245 m underground against DA-V2's 0.057 m -- the whole rise was a
scale offset, and refitting the scale on the floor left it worse on the table (1.06 m
against DA-V2's 0.82 m) while better on the shelf. A systematic bias, not a measurement.

**Two halves that no degenerate source can pass at once.** Flatness alone is won by a
model that returns the ground plane and nothing else, which is why `--control flat` is
run on every invocation rather than offered as an option: it scores perfectly on the
floor and reports no fixture relief at all, and seeing it in the table is what stops the
floor numbers being read as a verdict on their own.

The two halves are never summed. A single number would have to weight "the floor is
0.06 m off" against "the shelf reads 1.2 m", and the exchange rate between those is
exactly the judgement this tool exists to leave to a reader.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_hydranet.geometry.ground import (
    Camera as GCamera,
)
from syncai_hydranet.geometry.ground import (
    GroundPlane,
    undistort_points,
    unproject,
)

ROOT = Path(__file__).resolve().parent.parent.parent


FRAME_H, FRAME_W = 1080, 1920

# The percentile each class's height is read at, copied from `scene_mesh.HEIGHT_PCT` so
# the numbers here are the numbers the renderer draws. Importing it would drag the whole
# mesh builder (and scipy) in for four integers, and drift is caught by the test.
HEIGHT_PCT = {"wall": 99, "column": 99, "display_table": 85, "display_shelf": 99}


@dataclass
class Reading:
    """One depth source on one camera. `abstained` is a first-class outcome."""

    camera: str
    source: str
    floor_px: int = 0
    floor_offset_m: float = float("nan")  # |median height| over walkable; 0 is correct
    floor_spread_m: float = float("nan")  # p95 - p05 over walkable; flatness
    relief_m: dict[str, float] = field(default_factory=dict)
    abstained: str = ""


def _geometry(camera: str, root: Path):
    """The undistort/sample lattice `GeomTeacher` builds, and nothing else.

    Rebuilt rather than read from `runs/site30k_qa/geometry_cache`: the cache stores the
    *result* of one depth source, and a bench that read it could only ever score that one.
    """
    calib = json.loads((root / f"runs/onboard01/{camera}.calib.json").read_text())
    vfov = float(calib["vfov_assumed_deg"])
    k1 = float(calib.get("k1_division_model") or 0.0)
    plane = GroundPlane(
        height=float(calib["height_m"]),
        pitch=math.radians(calib["pitch_deg"]),
        roll=math.radians(calib["roll_deg"]),
    )
    plate = np.asarray(Image.open(calib["plate_used"]).convert("RGB"))
    ph, pw = plate.shape[:2]
    vv, uu = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float64)
    und = undistort_points(
        np.stack([uu.ravel(), vv.ravel()], 1),
        k1,
        (FRAME_W / 2.0, FRAME_H / 2.0),
        math.hypot(FRAME_H, FRAME_W) / 2.0,
    )
    uu_u = und[:, 0].reshape(FRAME_H, FRAME_W)
    vv_u = und[:, 1].reshape(FRAME_H, FRAME_W)
    su, sv = uu_u * (pw / FRAME_W), vv_u * (ph / FRAME_H)
    oob = (su < 0) | (su > pw - 1) | (sv < 0) | (sv > ph - 1)
    sui = np.clip(np.round(su).astype(int), 0, pw - 1)
    svi = np.clip(np.round(sv).astype(int), 0, ph - 1)
    return {
        "calib": calib,
        "plate": plate,
        "plane": plane,
        "cam_plate": GCamera.from_vfov(ph, pw, vfov),
        "sui": sui,
        "svi": svi,
        "oob": oob,
        "vfov": vfov,
    }


def height_map(depth_plate: np.ndarray, geo: dict) -> np.ndarray:
    """Depth on the undistorted plate -> height above the floor, in frame space.

    The three lines are `GeomTeacher.__init__`'s, kept identical on purpose: a bench that
    unprojected differently from the pipeline would score a geometry the renderer never
    draws. The test asserts this reproduces the shipped cache.
    """
    level = unproject(depth_plate, geo["cam_plate"]) @ geo["plane"].rotation
    h = geo["plane"].height - level[..., 1]
    return h[geo["svi"], geo["sui"]].astype(np.float32)


def _mask(root: Path, name: str, cf: dict) -> np.ndarray | None:
    rel = cf.get("mask_files", {}).get(name)
    if not rel or not (root / "runs/commission01" / rel).exists():
        return None
    img = Image.open(root / "runs/commission01" / rel).resize(
        (FRAME_W, FRAME_H), Image.Resampling.NEAREST
    )
    return np.asarray(img) > 127


def score(camera: str, source: str, depth_plate: np.ndarray | None, geo: dict, root: Path):
    """One reading. Abstains rather than returning a number it cannot stand behind."""
    r = Reading(camera=camera, source=source)
    cf = json.loads((root / f"runs/commission01/{camera}.camera.json").read_text())
    if depth_plate is None:
        r.abstained = "the source returned no depth"
        return r
    walk = _mask(root, "walkable", cf)
    if walk is None:
        r.abstained = "no walkable mask; the floor cannot witness anything"
        return r
    h = height_map(depth_plate, geo)
    ok = ~geo["oob"] & np.isfinite(h)
    fl = h[walk & ok]
    r.floor_px = int(fl.size)
    # 2000 px is ~0.1% of the frame. Below that the percentiles are a handful of pixels
    # and a source could win the bench on noise.
    if fl.size < 2000:
        r.abstained = f"only {fl.size} usable floor pixels"
        return r
    r.floor_offset_m = float(abs(np.median(fl)))
    r.floor_spread_m = float(np.percentile(fl, 95) - np.percentile(fl, 5))
    for name, pct in HEIGHT_PCT.items():
        m = _mask(root, name, cf)
        if m is None:
            continue
        v = h[m & ok]
        v = v[v > 0.05]
        if v.size > 200:
            r.relief_m[name] = float(np.percentile(v, pct))
    return r


# ---------------------------------------------------------------------------
# depth sources


def source_dav2(geo: dict):
    """The shipped world model: DA-V2 metric indoor, undistorted plate, x calib scale."""
    from syncai_bev3d.plate_calibration import run_depth, undistort_image

    k1 = float(geo["calib"].get("k1_division_model") or 0.0)
    scale = geo["calib"].get("scale")
    if scale is None:
        return None  # uncommissioned: no person-prior scale, so no metres
    return run_depth(undistort_image(geo["plate"], k1)) * float(scale)


def source_flat(geo: dict):
    """The degenerate control: the ground plane itself, rendered as a depth map.

    Runs on every invocation. It is what a model that had learned nothing but "floors are
    flat" would produce, and it wins the floor half outright -- offset 0.00, spread 0.00 --
    while reporting no fixture relief. Its row in the table is the sentence "flatness is
    not the verdict", written in the same units as the candidates.
    """
    ph, pw = geo["plate"].shape[:2]
    cam = geo["cam_plate"]
    vv, uu = np.mgrid[0:ph, 0:pw].astype(np.float64)
    # Ray through each pixel, then the depth at which that ray meets y = plane.height.
    rays = np.stack([(uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fy, np.ones_like(uu)], axis=-1)
    level_dir = rays @ geo["plane"].rotation
    with np.errstate(invalid="ignore", divide="ignore"):
        t = geo["plane"].height / level_dir[..., 1]
    # `t` is already the z-depth `unproject` expects, so it is returned as-is.
    return np.where((t > 0) & np.isfinite(t), t, np.nan)


def source_npy(path: Path):
    """A depth map computed elsewhere -- another environment, another model.

    MoGe-2 and MapAnything need conflicting torch pins, so they run in their own venv and
    hand a `.npy` over. Resized to the plate here so the caller cannot silently score a
    depth map against the wrong lattice.
    """

    def _load(geo: dict):
        ph, pw = geo["plate"].shape[:2]
        a = np.load(path).astype(np.float32).squeeze()
        if a.shape != (ph, pw):
            a = np.asarray(
                Image.fromarray(a).resize((pw, ph), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        return np.where(a > 0, a, np.nan)

    return _load


SOURCES = {"dav2": source_dav2, "flat": source_flat}


# ---------------------------------------------------------------------------


def commissioned(root: Path) -> list[str]:
    return sorted(
        p.name.split(".camera.json")[0]
        for p in (root / "runs/commission01").glob("*.camera.json")
    )


def render(readings: list[Reading]) -> str:
    classes = sorted({k for r in readings for k in r.relief_m})
    head = f"{'camera':<17}{'source':<10}{'floor|med|':>11}{'spread':>9}"
    head += "".join(f"{c[:13]:>14}" for c in classes)
    out = [head, "-" * len(head)]
    for r in readings:
        if r.abstained:
            out.append(f"{r.camera:<17}{r.source:<10}  ABSTAINED: {r.abstained}")
            continue
        line = f"{r.camera:<17}{r.source:<10}{r.floor_offset_m:>11.3f}{r.floor_spread_m:>9.3f}"
        for c in classes:
            v = r.relief_m.get(c)
            line += f"{v:>14.2f}" if v is not None else f"{'--':>14}"
        out.append(line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cameras", nargs="*", help="default: every commissioned camera")
    ap.add_argument("--source", default="dav2", choices=sorted(SOURCES), help="depth source")
    ap.add_argument("--npy", type=Path, help="score a depth map from another environment")
    ap.add_argument("--npy-label", default="npy", help="what to call the --npy source")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--json", type=Path, help="also write the readings here")
    a = ap.parse_args()

    cams = a.cameras or commissioned(a.root)
    if not cams:
        print("no commissioned cameras; nothing to score", file=sys.stderr)
        return 2
    readings: list[Reading] = []
    for cam in cams:
        try:
            geo = _geometry(cam, a.root)
        except (FileNotFoundError, KeyError) as e:
            readings.append(Reading(cam, a.source, abstained=f"no calibration ({e})"))
            continue
        jobs = [(a.source, SOURCES[a.source])]
        if a.npy:
            jobs.append((a.npy_label, source_npy(a.npy)))
        # The control is not optional. See `source_flat`.
        jobs.append(("flat(ctl)", source_flat))
        for label, fn in jobs:
            try:
                depth = fn(geo)
            except Exception as e:
                readings.append(Reading(cam, label, abstained=f"{type(e).__name__}: {e}"))
                continue
            readings.append(score(cam, label, depth, geo, a.root))

    print(render(readings))
    print(
        "\nfloor|med| and spread: metres, 0 is correct -- the floor is at floor level by\n"
        "definition. Relief: the height each class is drawn at, per scene_mesh.HEIGHT_PCT.\n"
        "flat(ctl) is the ground plane itself: it wins the floor half and has no relief,\n"
        "which is why the two halves are reported side by side and never summed."
    )
    scored = [r for r in readings if not r.abstained and r.source != "flat(ctl)"]
    if not scored:
        print("\nnothing was scored; every source abstained", file=sys.stderr)
        return 1
    if a.json:
        a.json.write_text(
            json.dumps([r.__dict__ for r in readings], indent=2, sort_keys=True) + "\n"
        )
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
