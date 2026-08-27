#!/usr/bin/env python3
"""Step 5's first gate: how many metres wrong is a floor position, against a truth.

    python3 scripts/wildtrack_ground_eval.py check --root datasets/wildtrack
    python3 scripts/wildtrack_ground_eval.py eval  --root datasets/wildtrack

Every metre this project reports -- the 71 service zones, journey path lengths, the
`height_above_floor_m` that the `fall` fix turns on -- comes out of one chain: a person
box, its bottom-centre, `undistort_points`, `pixel_to_ground`. **That chain has never
been compared against a ground truth.** WILDTRACK is public, needs no store visit, and
carries synchronised multi-view boxes with each person's floor position on a known grid.

---------------------------------------------------------------------------
WHAT IS BEING MEASURED, AND WHAT WOULD MAKE THE ANSWER MEANINGLESS

**The projection code is not on trial; the calibration is.** `tests/test_geometry.py`
round-trips `ground_to_pixel` / `pixel_to_ground` at the fleet's own poses to 1.3e-14 m,
and `tests/test_calibrate.py` round-trips the lens contract to 1e-8 px. So a metre of
error found here is a metre of *calibration* error, and that attribution is the reason
those tests were widened before this file was written.

**Ground-truth boxes, not detections.** Feeding this the detector would measure detection
and projection together and report the sum as geometry.

**WILDTRACK is a harder regime than the fleet, and the number does not transfer as-is.**
Its seven cameras sit at 8.7-20.1 degrees of pitch (measured, see `selfcheck`'s sibling
walk-through in the commit); the shipped fleet sits at 38.8-52.3. A shallower pitch grazes
the floor, so the same pitch error buys more metres of position error. The gate is
therefore conservative, which is the right direction, but "our cameras are X metres out"
is not a sentence this file's output supports.

**The scale must not be fitted, and this is the decision the whole gate turns on.**
`GroundPlane` carries height, pitch and roll and no yaw, so `pixel_to_ground` answers in
the camera's own level frame: a floor position is (lateral, forward) *from this camera*,
not a world coordinate. Comparing that against WILDTRACK's world frame therefore needs a
rigid 2D fit -- one rotation, one translation -- to absorb the yaw and the origin.
Allowing a **scale** into that fit would absorb exactly the error the fleet has already
been bitten by: Taichung-cam10's metres are 1.21x too large (PLAN section 7.10), and a
similarity fit would have reported that camera as excellent. So `eval` locks the scale at
1 and reports the free-scale residual **beside** it as a diagnostic, never instead: if
freeing the scale collapses the error, the fault is a scale fault and wants a different
fix from a pitch fault.

---------------------------------------------------------------------------
THE GRID CONVENTION IS MEASURED HERE, NOT REMEMBERED

WILDTRACK encodes a person's floor position as a single `positionID` into a grid, and the
mapping from that integer to metres is a convention with an origin, a step and a row
order. Getting it wrong produces positions that are plausible and wrong, which is the
failure mode this whole file exists to catch, so it is not taken on trust: `check`
reprojects each ground-truth position through **WILDTRACK's own calibration** into every
view that annotated it, and measures the distance to the bottom-centre of that view's
annotated box. A wrong convention cannot survive that -- it puts people metres from their
own boxes across seven cameras at once. `--grid` selects a candidate convention and the
one that passes is the one used, which is a measurement rather than a memory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from syncai_hydranet.geometry.ground import (  # noqa: E402
    Camera,
    GroundPlane,
    pixel_to_ground,
    undistort_points,
)

# WILDTRACK's published layout: 480 x 1440 cells of 2.5 cm, origin at (-300, -900) cm.
#
# **The dataset's world frame is centimetres**, which is not a detail: `extr_*.xml` gives
# tvec as (-525.89, 45.41, 986.72) for CVLab1, i.e. a camera 9.87 m up. Everything here
# converts to metres at the boundary and stays in metres, because a hundredfold unit slip
# would be absorbed by a similarity fit and would show up under the rigid fit this gate
# uses as a huge residual -- readable as "the calibration is terrible" when it is a units
# bug. Candidates rather than a constant, because `check` decides between them.
# Each entry is (n_cols, step_m, origin_x_m, origin_y_m, row_major).
GRID_CANDIDATES = {
    "wildtrack": (480, 0.025, -3.0, -9.0, False),
    "wildtrack_rowmajor": (480, 0.025, -3.0, -9.0, True),
    "wildtrack_origin0": (480, 0.025, 0.0, 0.0, False),
}
# `extr_*.xml` tvec is in centimetres; every consumer here wants metres.
CM_TO_M = 0.01


@dataclass(frozen=True)
class WtCamera:
    """WILDTRACK's own calibration for one view, in OpenCV's convention."""

    name: str
    k: np.ndarray  # 3x3 intrinsics
    dist: np.ndarray  # distortion coefficients as published
    rvec: np.ndarray  # world -> camera, Rodrigues
    tvec: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        return _rodrigues(self.rvec)

    @property
    def centre_world(self) -> np.ndarray:
        """Camera centre in world coordinates: C = -R^T t."""
        return -(self.rotation.T @ self.tvec)


def _rodrigues(r: np.ndarray) -> np.ndarray:
    """Rodrigues vector -> rotation matrix, without pulling in cv2 for three lines."""
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = (r / theta).reshape(3)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * (kx @ kx)


def grid_to_metres(position_id: np.ndarray, convention: str) -> np.ndarray:
    """A WILDTRACK `positionID` -> (x, y) metres in its world frame."""
    n_cols, step, ox, oy, row_major = GRID_CANDIDATES[convention]
    pid = np.asarray(position_id, dtype=np.int64)
    if row_major:
        col, row = pid % n_cols, pid // n_cols
    else:
        row, col = pid % n_cols, pid // n_cols
    return np.stack([ox + step * col, oy + step * row], axis=-1)


def project_world(points_xy: np.ndarray, cam: WtCamera, z: float = 0.0) -> np.ndarray:
    """World floor points (metres) -> pixels, through WILDTRACK's own calibration.

    Used only by `check`. `eval` deliberately does not touch this: validating our chain
    with their projection would validate theirs.

    **The distortion is applied, and leaving it out was a real bug in this file.** The
    annotated boxes are drawn on the *original* frames, so a pinhole reprojection lands
    tens of pixels away at the edges -- which reads as "the grid convention is wrong",
    which is the one thing `check` exists to decide. OpenCV's 5-coefficient model, the
    same one `intr_*.xml` publishes.
    """
    pts = np.concatenate(
        [np.asarray(points_xy, float), np.full((len(points_xy), 1), z)], axis=1
    )
    cam_pts = pts @ cam.rotation.T + cam.tvec.reshape(1, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        xn = cam_pts[:, 0] / cam_pts[:, 2]
        yn = cam_pts[:, 1] / cam_pts[:, 2]
    d = np.zeros(5)
    d[: min(5, len(cam.dist))] = cam.dist[: min(5, len(cam.dist))]
    k1, k2, p1, p2, k3 = d
    r2 = xn**2 + yn**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
    yd = yn * radial + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn
    u = cam.k[0, 0] * xd + cam.k[0, 2]
    v = cam.k[1, 1] * yd + cam.k[1, 2]
    return np.stack([u, v], axis=1)


def to_our_model(cam: WtCamera, floor_z: float = 0.0) -> tuple[Camera, GroundPlane]:
    """WILDTRACK calibration -> the (Camera, GroundPlane) this project ships.

    `GroundPlane` is height, pitch and roll and carries **no yaw**, because a single fixed
    camera cannot observe its own heading and nothing downstream needs it. The conversion
    is therefore lossy by design and the lost quantity is recovered by the rigid fit in
    `eval`, not smuggled in here.

    Closed form. Our level frame is x right, y **down**, z forward, and the plane's
    rotation is R_roll @ R_pitch, so the floor's downward normal expressed in camera
    coordinates is d = (-sin r cos p, cos r cos p, sin p). Reading pitch and roll back off
    that vector is one asin and one atan2; height is the camera centre's distance to the
    plane.
    """
    k = cam.k
    intr = Camera(fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]))
    # world "down" is -z when the floor is the z = floor_z plane
    down_world = np.array([0.0, 0.0, -1.0])
    d = cam.rotation @ down_world
    d = d / np.linalg.norm(d)
    pitch = math.asin(float(np.clip(d[2], -1.0, 1.0)))
    cp = math.cos(pitch)
    roll = math.atan2(-float(d[0]), float(d[1])) if abs(cp) > 1e-9 else 0.0
    height = float(cam.centre_world[2] - floor_z)
    return intr, GroundPlane(height=height, pitch=pitch, roll=roll)


def rigid_fit(
    src: np.ndarray, dst: np.ndarray, *, allow_scale: bool
) -> tuple[np.ndarray, float]:
    """Best 2D rotation + translation (optionally scale) taking `src` onto `dst`.

    Umeyama, restricted to the plane. With `allow_scale=False` this is the honest fit for
    this gate: a metre is a metre, and a fit that may rescale cannot report a scale error.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, d]) @ vt
    scale = 1.0
    if allow_scale:
        var = (a**2).sum() / len(src)
        scale = float((s * np.array([1.0, d])).sum() / var) if var > 0 else 1.0
    out = (scale * (rot @ src.T)).T + (mu_d - scale * (rot @ mu_s))
    return out, scale


# --------------------------------------------------------------------- loading


def load_calibration(root: Path, intrinsics: str = "intrinsic_original") -> dict[str, WtCamera]:
    """Read WILDTRACK's intrinsic and extrinsic XML for every view.

    Two shapes in one format, and assuming the wrong one is a crash rather than a wrong
    number, which is the good case. `camera_matrix` and `distortion_coefficients` are
    `opencv-matrix` nodes with rows/cols/data; **`rvec` and `tvec` are plain text nodes**
    with the numbers and nothing else.

    `intrinsic_original` is the default because the annotated boxes are drawn on the
    original frames. `intrinsic_zero` describes the undistorted ones and pairing it with
    those boxes would put every reprojection a few tens of pixels out at the edges.
    """
    import xml.etree.ElementTree as ET

    def numbers(node, name: str) -> np.ndarray:
        if node is None:
            raise ValueError(f"{name}: node missing")
        data = node.findtext("data")
        if data is None:  # a plain text node, which is how rvec and tvec are written
            return np.array([float(x) for x in (node.text or "").split()])
        rows, cols = int(node.findtext("rows")), int(node.findtext("cols"))
        return np.array([float(x) for x in data.split()]).reshape(rows, cols)

    intr_dir = root / "calibrations" / intrinsics
    extr_dir = root / "calibrations" / "extrinsic"
    if not extr_dir.is_dir() or not intr_dir.is_dir():
        raise FileNotFoundError(f"expected {extr_dir} and {intr_dir}")
    cams: dict[str, WtCamera] = {}
    for extr in sorted(extr_dir.glob("*.xml")):
        name = extr.stem
        tree = ET.parse(extr)
        rvec = numbers(tree.find(".//rvec"), f"{name}:rvec").reshape(3)
        # centimetres in the file, metres everywhere here
        tvec = numbers(tree.find(".//tvec"), f"{name}:tvec").reshape(3) * CM_TO_M
        view = name.split("_")[-1]
        matches = sorted(intr_dir.glob(f"*{view}*.xml"))
        if not matches:
            raise FileNotFoundError(f"no intrinsic file for {view} in {intr_dir}")
        itree = ET.parse(matches[0])
        k = numbers(itree.find(".//camera_matrix"), f"{view}:camera_matrix")
        dist = numbers(itree.find(".//distortion_coefficients"), f"{view}:dist").reshape(-1)
        cams[name] = WtCamera(name=name, k=k, dist=dist, rvec=rvec, tvec=tvec)
    return cams


def load_annotations(root: Path, limit: int = 0) -> list[dict]:
    """Read `annotations_positions/*.json`: per frame, people with a positionID + views."""
    files = sorted((root / "annotations_positions").glob("*.json"))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        out.append({"frame": f.stem, "people": json.loads(f.read_text())})
    return out


# ---------------------------------------------------------------------- modes


def run_unpack(a) -> None:
    zf = Path(a.zip)
    dest = Path(a.root)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zf) as z:
        z.extractall(dest)
    print(f"unpacked {zf} -> {dest}")
    for p in sorted(dest.rglob("*"))[:20]:
        print("  ", p.relative_to(dest))


def run_check(a) -> None:
    """Decide the grid convention by reprojection, and say how well it agrees."""
    root = Path(a.root)
    cams = load_calibration(root, a.intrinsics)
    frames = load_annotations(root, limit=a.frames)
    print(f"cameras {len(cams)}  frames {len(frames)}")
    names = sorted(cams)
    for convention in a.grid:
        errs: list[float] = []
        for fr in frames:
            for person in fr["people"]:
                world = grid_to_metres(np.array([person["positionID"]]), convention)
                for view in person.get("views", []):
                    vn = int(view["viewNum"])
                    if vn >= len(names) or view["xmax"] < 0:
                        continue
                    cam = cams[names[vn]]
                    uv = project_world(world, cam)[0]
                    foot = np.array([(view["xmin"] + view["xmax"]) / 2.0, view["ymax"]])
                    errs.append(float(np.hypot(*(uv - foot))))
        e = np.array(errs)
        if not len(e):
            print(f"{convention:22s}  no annotated views found")
            continue
        print(
            f"{convention:22s}  n {len(e):6d}  reprojection to the box foot: "
            f"p50 {np.median(e):7.1f} px   p90 {np.percentile(e, 90):7.1f} px"
        )
    print(
        "\nThe convention to use is the one with a small p50. A wrong one puts people "
        "hundreds of pixels from their own boxes on every view at once."
    )


def run_eval(a) -> None:
    root = Path(a.root)
    cams = load_calibration(root, a.intrinsics)
    frames = load_annotations(root, limit=a.frames)
    names = sorted(cams)
    left = f"{'view':>10s} {'n':>6s} {'rigid p50':>10s} {'p90':>8s}"
    print(f"{left} {'free-scale p50':>15s} {'scale':>7s}")
    rows = {}
    for vn, name in enumerate(names):
        cam = cams[name]
        intr, plane = to_our_model(cam, floor_z=a.floor_z)
        ours, truth = [], []
        for fr in frames:
            for person in fr["people"]:
                for view in person.get("views", []):
                    if int(view["viewNum"]) != vn or view["xmax"] < 0:
                        continue
                    foot = np.array(
                        [[(view["xmin"] + view["xmax"]) / 2.0, float(view["ymax"])]]
                    )
                    if a.k1:
                        foot = undistort_points(
                            foot, a.k1, (intr.cx, intr.cy), float(np.hypot(intr.cx, intr.cy))
                        )
                    x, z = pixel_to_ground(foot[:, 0], foot[:, 1], intr, plane)
                    if not (np.isfinite(x).all() and np.isfinite(z).all()):
                        continue
                    ours.append([float(x[0]), float(z[0])])
                    truth.append(grid_to_metres(np.array([person["positionID"]]), a.grid[0])[0])
        if len(ours) < 8:
            print(f"{name:>10s} {len(ours):6d}   too few placeable feet to fit")
            continue
        o, t = np.array(ours), np.array(truth)
        fit_r, _ = rigid_fit(o, t, allow_scale=False)
        fit_s, scale = rigid_fit(o, t, allow_scale=True)
        er = np.hypot(*(fit_r - t).T)
        es = np.hypot(*(fit_s - t).T)
        rows[name] = {
            "n": len(o),
            "rigid_p50": float(np.median(er)),
            "rigid_p90": float(np.percentile(er, 90)),
            "free_p50": float(np.median(es)),
            "scale": scale,
            "height": plane.height,
            "pitch_deg": math.degrees(plane.pitch),
            "roll_deg": math.degrees(plane.roll),
        }
        print(
            f"{name:>10s} {len(o):6d} {np.median(er):10.3f} {np.percentile(er, 90):8.3f} "
            f"{np.median(es):15.3f} {scale:7.3f}"
        )
    if rows:
        print(
            "\nThe rigid column is the gate. The free-scale column is a diagnostic: if it "
            "collapses while the rigid one does not, the fault is a scale fault -- the "
            "shape of PLAN section 7.10 -- and wants a different fix from a pitch fault."
        )
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))


def run_selfcheck(_a) -> None:
    """Verify this file's own two derivations, before any data exists to hide them in.

    Both are places where a plausible wrong answer is indistinguishable from a right one
    once real numbers are flowing, so they are checked against constructions whose answer
    is known rather than against WILDTRACK.
    """
    from scipy.spatial.transform import Rotation

    print("to_our_model recovers a pose it was given (27 fleet-range poses)")
    bad = 0
    for h in (2.17, 2.5, 2.91):
        for pd in (38.8, 45.0, 52.3):
            for rd in (-12.9, 0.0, 8.3):
                plane = GroundPlane(h, math.radians(pd), math.radians(rd))
                # world (x, y, z up) -> our level frame (x right, y down, z forward),
                # at an arbitrary yaw, which to_our_model is entitled to lose
                yaw = math.radians(40.0)
                cy, sy = math.cos(yaw), math.sin(yaw)
                r_wl = np.array([[cy, sy, 0.0], [0.0, 0.0, -1.0], [-sy, cy, 0.0]])
                r_wc = plane.rotation @ r_wl
                centre = np.array([1.3, -2.4, h])
                cam = WtCamera(
                    name="synthetic",
                    k=np.array([[400.0, 0, 480.0], [0, 400.0, 270.0], [0, 0, 1.0]]),
                    dist=np.zeros(5),
                    rvec=Rotation.from_matrix(r_wc).as_rotvec(),
                    tvec=-r_wc @ centre,
                )
                _intr, got = to_our_model(cam)
                off = max(
                    abs(got.height - h),
                    abs(math.degrees(got.pitch) - pd),
                    abs(math.degrees(got.roll) - rd),
                )
                if off > 1e-6:
                    bad += 1
                    print(
                        f"  MISMATCH h={h} pitch={pd} roll={rd} -> "
                        f"{got.height:.3f} {math.degrees(got.pitch):.3f} "
                        f"{math.degrees(got.roll):.3f}"
                    )
    print(f"  mismatches: {bad} of 27")

    print("\nrigid_fit refuses to absorb a scale error, which is the whole gate")
    rng = np.random.default_rng(0)
    src = rng.uniform(-5, 5, (500, 2))
    th = math.radians(33.0)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    # 1.21 and 0.8824 are Taichung-cam10's measured error and its correction (PLAN 7.10)
    for true_scale in (1.0, 1.21, 0.8824):
        dst = (true_scale * (rot @ src.T)).T + np.array([7.0, -2.0])
        fit_r, _ = rigid_fit(src, dst, allow_scale=False)
        fit_s, scale = rigid_fit(src, dst, allow_scale=True)
        er = np.median(np.hypot(*(fit_r - dst).T))
        es = np.median(np.hypot(*(fit_s - dst).T))
        print(
            f"  true scale {true_scale:6.4f}   rigid p50 {er:8.4f} m   "
            f"free p50 {es:.1e} m   recovered scale {scale:6.4f}"
        )
    print(
        "  A rigid residual that grows with the scale error is the property this gate "
        "needs; a free fit driving it to zero is what would have hidden cam10."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default="datasets/wildtrack")
    common.add_argument(
        "--intrinsics",
        default="intrinsic_original",
        choices=("intrinsic_original", "intrinsic_zero"),
        help="the annotated boxes are on the original frames",
    )
    common.add_argument("--frames", type=int, default=0, help="0 = every annotated frame")
    common.add_argument("--out")
    sub = ap.add_subparsers(dest="mode", required=True)

    sc = sub.add_parser(
        "selfcheck", parents=[common], help="verify this file's own derivations"
    )
    sc.set_defaults(fn=run_selfcheck)

    u = sub.add_parser("unpack", parents=[common], help="extract the archive")
    u.add_argument("--zip", default="datasets/_incoming/wildtrack/Wildtrack_dataset_full.zip")
    u.set_defaults(fn=run_unpack)

    c = sub.add_parser("check", parents=[common], help="decide the grid convention")
    c.add_argument("--grid", nargs="+", default=sorted(GRID_CANDIDATES))
    c.set_defaults(fn=run_check)

    e = sub.add_parser("eval", parents=[common], help="metres of error in our own chain")
    # Measured by `check`, not chosen: over 60 frames and seven views the row-major
    # reading reprojects to the foot of the annotated box at a median 24.1 px, against
    # 1,399.9 px and 1,507.4 px for the other two. The default was one of the losers
    # until that run, which is the whole argument for `check` existing.
    e.add_argument("--grid", nargs="+", default=["wildtrack_rowmajor"])
    e.add_argument("--floor-z", type=float, default=0.0)
    e.add_argument(
        "--k1", type=float, default=0.0, help="apply our lens model before projecting"
    )
    e.set_defaults(fn=run_eval)

    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
