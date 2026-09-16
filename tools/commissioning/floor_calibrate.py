"""The floor's own right angles as the calibration witness: the vfov per camera.

Every onboarded camera but one carries `vfov_assumed_deg: 70.4` (PLAN 9.5), and on
Tao-Hsin-cam15 (2026-09-10) the floor's two line families -- planks and their joints,
laid square -- came out 100 deg apart in that assumption. A wrong vfov skews the ground:
straight, square things on the floor stop being square in the metres the scene is
fitted in, and every box turns with them. So this sweeps the vfov, refits the floor
plane at each (the onboarding's own RANSAC, `plate_calibration.floor_candidates`), reads
the angle between the two families through that calibration, and takes the vfov where
they are 90 deg apart. No person, no tile size, no operator: the floor is square because
floors are laid square.

Per camera it prints the sweep and writes `runs/commission_review/floor_calib/<cam>.json`;
`--pins-out` collects the cameras with a clean zero crossing into a pins file the
onboarding takes (`scripts/onboard_camera.py --vfov-pins`), so the person prior and the
rulers are recomputed at the measured vfov rather than around it.

  uv run python tools/commissioning/floor_calibrate.py Tao-Hsin-cam15 Taichung-cam10
  uv run python tools/commissioning/floor_calibrate.py --pins-out runs/floor_calib/pins.json
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_bev3d import floor_axis
from syncai_bev3d import plate_calibration as pc
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, pixel_to_ground, undistort_points

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/commission_review/floor_calib"
# The sweep. 45-95 covers every lens the fleet census found (PLAN 7.12).
VFOV_MIN, VFOV_MAX, VFOV_STEP = 45.0, 95.0, 2.5
# A crossing counts only if both families are read sharply there.
SHARPNESS_MAX = floor_axis.SHARPNESS_MAX
INLIER_M = 0.03


def raw_depth(camera: str, plate: np.ndarray, k1: float) -> np.ndarray:
    cache = ROOT / "runs/site30k_qa/geometry_cache"
    for name in (f"{camera}.depth.npy", f"{camera}.rebuilt.depth.npy"):
        if (cache / name).exists():
            d = np.load(cache / name)
            if d.shape == plate.shape[:2]:
                return d
    depth = pc.run_depth(pc.undistort_image(plate, k1))
    np.save(cache / f"{camera}.depth.npy", depth)
    return depth


def skew_at(vfov: float, plate, walk, depth, k1, lens) -> dict:
    h, w = plate.shape[:2]
    cam = Camera.from_vfov(h, w, vfov)
    plane, _residual, _rows = pc.choose_floor(pc.floor_candidates(depth, cam, INLIER_M))
    if plane is None:
        return {"vfov": vfov, "plane": None}

    def ground(px):
        pts = px if lens is None else undistort_points(px, k1, lens[0], lens[1])
        x, z = pixel_to_ground(pts[:, 0], pts[:, 1], cam, plane)
        return np.stack([x, z], axis=1)

    ys, xs = np.mgrid[0:h, 0:w]
    g = ground(np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1))
    gz = g[:, 1].reshape(h, w)
    ok = np.isfinite(gz)
    a1, a2, second = floor_axis.floor_line_axes(plate, walk, np.where(ok, gz, 99.0), ok, ground)
    # The families are ordered by angle, not by strength: the stronger one changes
    # between vfovs and would flip the sign of the skew with it.
    if a1 is not None and a2 is not None:
        lo, hi = sorted((a1 % 180, a2 % 180))
        skew = (hi - lo) - 90
    else:
        skew = None
    return {
        "vfov": vfov,
        "pitch_deg": math.degrees(plane.pitch),
        "roll_deg": math.degrees(plane.roll),
        "family_1": a1,
        "family_2": a2,
        "skew_deg": skew,
        "second_peak": second,
    }


def crossing(rows: list[dict]) -> dict | None:
    """The vfov where the skew crosses zero, interpolated between sharp neighbours; the
    crossing nearest the fleet assumption when there are several."""
    usable = [
        r for r in rows if r.get("skew_deg") is not None and r["second_peak"] <= SHARPNESS_MAX
    ]
    found = []
    for a, b in itertools.pairwise(usable):
        if b["vfov"] - a["vfov"] > VFOV_STEP * 1.5:
            continue  # a gap in the sharp readings: no interpolation across it
        sa, sb = a["skew_deg"], b["skew_deg"]
        if sa == 0:
            found.append({"vfov_deg": a["vfov"], "between": (a["vfov"], b["vfov"])})
        elif sa * sb < 0:
            v = a["vfov"] + (b["vfov"] - a["vfov"]) * (-sa) / (sb - sa)
            found.append({"vfov_deg": round(v, 2), "between": (a["vfov"], b["vfov"])})
    if not found:
        return None
    return min(found, key=lambda f: abs(f["vfov_deg"] - 70.4))


def run(camera: str) -> dict:
    calib = json.loads((ROOT / f"runs/onboard01/{camera}.calib.json").read_text())
    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    plate = np.asarray(Image.open(ROOT / calib["plate_used"]).convert("RGB"))
    h, w = plate.shape[:2]
    k1 = float(calib["k1_division_model"])
    lens = None if cf.lens is None else (cf.lens.centre_px, cf.lens.radius_px)
    walk = (
        np.asarray(
            Image.open(ROOT / "runs/commission01" / cf.mask_files["walkable"]).resize(
                (w, h), Image.Resampling.NEAREST
            )
        )
        > 127
    )
    depth = raw_depth(camera, plate, k1)
    rows = [
        skew_at(v, plate, walk, depth, k1, lens)
        for v in np.arange(VFOV_MIN, VFOV_MAX + 1e-9, VFOV_STEP)
    ]
    hit = crossing(rows)
    at_fleet = next((r for r in rows if abs(r["vfov"] - 70.0) < 1.3), None)
    out = {
        "camera": camera,
        "assumed_vfov_deg": calib["vfov_assumed_deg"],
        "vfov_source_before": calib["vfov_source"],
        "sweep": rows,
        "crossing": hit,
        "skew_near_assumed_deg": None if at_fleet is None else at_fleet.get("skew_deg"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{camera}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cameras", nargs="*")
    ap.add_argument("--pins-out", type=Path, default=None)
    a = ap.parse_args()
    cameras = a.cameras or sorted(
        p.name.replace(".camera.json", "")
        for p in (ROOT / "runs/commission01").glob("*.camera.json")
        if not p.name.startswith("test")
    )
    pins = {}
    for camera in cameras:
        out = run(camera)
        line = f"{camera:16s}"
        for r in out["sweep"]:
            if r.get("skew_deg") is None:
                continue
            mark = "*" if r["second_peak"] <= SHARPNESS_MAX else " "
            line += f"  {r['vfov']:.0f}:{r['skew_deg']:+5.1f}{mark}"
        print(line)
        hit = out["crossing"]
        if hit is None:
            print(f"    no sharp zero crossing; assumed {out['assumed_vfov_deg']} stays")
            continue
        print(
            f"    vfov {hit['vfov_deg']:.1f} (between {hit['between'][0]:.0f} and "
            f"{hit['between'][1]:.0f}); skew near the assumed vfov "
            f"{out['skew_near_assumed_deg']:+.1f} deg"
        )
        pins[camera] = {
            "vfov_deg": hit["vfov_deg"],
            "source": "floor_orthogonality",
            "skew_at_assumed_deg": out["skew_near_assumed_deg"],
        }
    if a.pins_out is not None:
        a.pins_out.parent.mkdir(parents=True, exist_ok=True)
        a.pins_out.write_text(json.dumps(pins, indent=1))
        print(f"wrote {a.pins_out} ({len(pins)} cameras)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
