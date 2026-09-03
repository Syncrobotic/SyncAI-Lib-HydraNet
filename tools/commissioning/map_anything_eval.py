#!/usr/bin/env python3
"""MapAnything against this fleet's own commissioned geometry, with a negative control.

    tools/commissioning/map_anything_eval.py intrinsics --out runs/mapany01
    tools/commissioning/map_anything_eval.py register   --out runs/mapany01

Two questions, and the second one only means anything because of the control.

**Intrinsics.** Every metre this product reports rests on a vertical field of view of
70.4 deg, and PLAN 7.19 records that as `vfov_source: fleet_hardware_assumed` on 22 of 23
cameras -- pinned by tile grid on Taichung-cam01 alone. So that camera is the only anchor
there is, and this asks an independent metric model what it thinks. Measured 2026-08-30:
it answers 38.26 deg, a **2.03x disagreement in focal length**, and three explanations were
ruled out rather than assumed -- `load_images` does not crop (294x518, aspect 1.762
against the source's 1.778), undistorting first moves the answer by 0.03 deg, and the
undistortion is not a no-op (corner mean delta 44.9/255, centre 0.7, 62.8% of pixels).
Feeding the same person boxes through `fit_pose_from_people` at each vfov cannot separate
them either: spreads of 2.66e-02 and 2.80e-02. That is the measurement
`test_plate_calibration.py` already records -- the residual is blind to a wrong vfov.

**Registration.** `WorldFrame.space` is `camera_floor(<camera_id>)` and there is no store
frame; PLAN 2.3.1 says the missing piece is a 2D similarity per camera against a store
plan, and no store plan exists. Cameras that overlap can be registered into one metric
frame instead -- but only if the method knows when they do not.

---------------------------------------------------------------------------
THE NEGATIVE CONTROL IS THE INSTRUMENT, NOT A FOOTNOTE

Three cameras in three different buildings, in three different cities, came back placed
2.13 to 5.91 m apart -- **numbers that look entirely reasonable**. A pipeline reading the
poses would build a store frame out of cameras in different towns and nothing would look
wrong.

What separates them is confidence: 1.0 for all three of the control (the floor value,
meaning no evidence) against 4.33 / 2.03 / 2.04 for three cameras in one room. So the
poses are unusable alone and the confidence is what makes them usable, and
:func:`registration_verdict` refuses to return a usable answer unless the control is
actually separated. `tests/test_map_anything_eval.py` holds that refusal.

Three groups is not enough to place a threshold, and the tool says so rather than
inventing one: `separation` is reported, and the caller decides.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# PLAN 7.19: pinned by tile grid on this camera and assumed on every other.
ANCHOR_CAMERA = "Taichung-cam01"
ANCHOR_VFOV_DEG = 70.4

# The Apache-licensed weights, deliberately. The CC-BY-NC variant lives at a different
# repo id, so "may this ship" is answered by which one is downloaded rather than by
# reading a model card -- unlike SAM 3, which is gated, or NTU/PoseLift, which are
# research-only and are kept out of every training config for that reason.
MODEL_ID = "facebook/map-anything-apache"
# The snapshot every number in this file's docstring was measured against, taken from
# `huggingface_hub.scan_cache_dir()` rather than from whatever upstream `main` points at
# today. Without it the readings would silently re-base on a model none of them was made
# against -- and 38.26 deg is only worth reporting if it can be reproduced.
#
# `tests/test_teacher_revisions_are_pinned.py` caught this after the file was committed
# and not before, because that guard enumerates *tracked* files: the full suite ran clean
# while the file was still untracked. The lesson is the ordering, not the pin.
MODEL_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"


@dataclass(frozen=True)
class GroupReading:
    """One set of cameras handed to the model together."""

    label: str
    cameras: tuple[str, ...]
    confidences: tuple[float, ...]
    pair_distances_m: dict[str, float] = field(default_factory=dict)
    # True for the set that physically cannot overlap. Exactly one group must carry it.
    is_control: bool = False


@dataclass(frozen=True)
class RegistrationVerdict:
    usable: bool
    reason: str
    control_max: float
    overlapping_min: float
    separation: float


def registration_verdict(readings: list[GroupReading]) -> RegistrationVerdict:
    """Is the confidence signal usable as a gate -- decided by the control, not by hope.

    Refuses rather than scores when the control is missing. A registration run without a
    set that *cannot* register has measured nothing: every group would come back with
    plausible distances, which is exactly what the control showed they do.
    """
    controls = [r for r in readings if r.is_control]
    others = [r for r in readings if not r.is_control]
    if not controls:
        return RegistrationVerdict(False, "no negative control in the run", 0.0, 0.0, 0.0)
    if not others:
        return RegistrationVerdict(
            False, "nothing to compare the control against", 0.0, 0.0, 0.0
        )

    control_max = max(max(r.confidences) for r in controls)
    overlapping_min = min(min(r.confidences) for r in others)
    separation = overlapping_min - control_max
    if separation <= 0:
        return RegistrationVerdict(
            False,
            "cameras in different buildings scored as high as cameras in one room; "
            "confidence cannot gate a store frame on this evidence",
            control_max,
            overlapping_min,
            separation,
        )
    return RegistrationVerdict(
        True,
        f"the control tops out at {control_max:.2f} and every other group stays at or "
        f"above {overlapping_min:.2f}; a gate belongs between them",
        control_max,
        overlapping_min,
        separation,
    )


def focal_disagreement(measured_vfov_deg: float, predicted_vfov_deg: float) -> float:
    """Ratio of focal lengths implied by two vfov claims about the same frame.

    Reported as focal rather than as a difference of angles because that is the quantity
    every metre downstream is divided by: `Camera.from_vfov` turns the angle into `fy`
    immediately, and a reader comparing "70.4 against 38.3" underestimates what it costs.
    """
    fm = 1.0 / math.tan(math.radians(measured_vfov_deg) / 2.0)
    fp = 1.0 / math.tan(math.radians(predicted_vfov_deg) / 2.0)
    return fp / fm


#: A shipped camera's plate lives under this root. `runs/commission01/` is a working
#: directory, not a manifest, so a camera commissioned there for an experiment would
#: otherwise enter this baseline: `dingpu-1f/test1` was commissioned on 2026-09-01 and
#: became a ninth row in a comparison of the Studio A fleet against itself. The plate
#: path is the discriminator because it is written into the artefact by the commissioning
#: that produced it -- no second list to keep in step.
SHIPPED_PLATE_ROOT = "datasets/studioa_static/"


def commissioned() -> dict[str, dict]:
    """The shipped `camera.json` -- the Studio A fleet -- as the baseline to compare against.

    Every `height` here is fitted from the 1.70 m person prior (PLAN 7.19), not measured
    with a tape, so a comparison against them is two estimates meeting -- which the
    reporting says in words rather than leaving to be inferred.

    Filtered by plate root rather than counted: see :data:`SHIPPED_PLATE_ROOT`.
    """
    out = {}
    for f in sorted((REPO / "runs/commission01").glob("*.camera.json")):
        d = json.loads(f.read_text())
        if not str(d.get("plate_file", "")).startswith(SHIPPED_PLATE_ROOT):
            continue
        out[d["camera_id"]] = {
            "height_m": d["plane"]["height"],
            "pitch_deg": round(math.degrees(d["plane"]["pitch"]), 2),
            "plate": d["plate_file"],
            "k1": d.get("lens", {}).get("k1"),
        }
    return out


def _load_model():
    """Imported here, not at module scope, so the analysis above is testable without it.

    MapAnything is not a dependency of this repository and installing it is a decision
    nobody has taken. Note for whoever does: the README's `pip install -e ".[all]"` does
    not resolve -- `mapanything` requires `rerun-sdk <0.25` and the extra pulls
    `vggt-omega`, which conflicts with it -- while the plain `pip install -e .` loads the
    Apache weights and runs inference, fetching its DINOv2 backbone from torch hub.
    """
    import torch
    from mapanything.models import MapAnything

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return MapAnything.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(dev).eval(), dev


def undistorted_plates(dest: Path) -> dict[str, Path]:
    """Undistort every commissioned plate once. k1 = -0.225 is a fleet constant.

    Not optional. `plate_calibration`'s pipeline undistorts as step 1, so the 70.4 deg the
    anchor is compared against is a property of the undistorted frame; handing the raw
    plate to a pinhole estimator compares two different images.
    """
    import numpy as np
    from PIL import Image

    from syncai_bev3d.plate_calibration import undistort_image

    dest.mkdir(parents=True, exist_ok=True)
    base = commissioned()
    out = {}
    # Commissioned cameras first: their own plate, their own k1.
    for cam, meta in base.items():
        src = REPO / meta["plate"]
        if src.is_file():
            out[cam] = (src, meta["k1"])
    # Then any camera with a plate but no camera.json. Registration is the reason: the
    # most informative group is three cameras in one room, and two of them
    # (Taichung-cam05, -cam06) are not commissioned -- cam05 was withdrawn when two
    # furniture checks agreed its cells were over-scaled. Excluding them would leave this
    # tool unable to reproduce its own docstring's numbers. `k1` is a fleet constant
    # (-0.225 on every shipped camera.json), which is what makes the fallback honest
    # rather than a guess.
    fleet_k1 = next((m["k1"] for m in base.values() if m["k1"] is not None), None)
    for d in sorted((REPO / "datasets/studioa_static").glob("*/")):
        cam = d.name
        if cam in out or fleet_k1 is None:
            continue
        plates = sorted(d.glob("plate_*.png"))
        if plates:
            out[cam] = (plates[0], fleet_k1)

    made = {}
    for cam, (src, k1) in out.items():
        dst = dest / f"{cam}.png"
        if not dst.is_file():
            img = np.asarray(Image.open(src).convert("RGB"))
            Image.fromarray(undistort_image(img, k1)).save(dst)
        made[cam] = dst
    return made


def _vfov_of(k_matrix, height_px: int) -> float:
    return math.degrees(2 * math.atan((height_px / 2) / float(k_matrix[1, 1])))


def run_intrinsics(plates: dict[str, Path]) -> dict[str, dict]:
    """One reading per camera: what the model thinks the lens is."""
    import numpy as np
    import torch
    from mapanything.utils.image import load_images

    model, _dev = _load_model()
    base = commissioned()
    readings = {}
    # Only the commissioned cameras. `undistorted_plates` deliberately returns more than
    # these so `register` can use the uncommissioned pair in the same-room group, but a
    # camera with no `camera.json` has no baseline to disagree with, and reporting a vfov
    # beside nothing is a number that invites a comparison it cannot support.
    for cam, path in sorted(plates.items()):
        if cam not in base:
            continue
        views = load_images([str(path)])
        h = int(views[0]["img"].shape[-2])
        with torch.no_grad():
            pred = model.infer(
                views,
                memory_efficient_inference=True,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
            )
        v = pred[0] if isinstance(pred, list) else pred
        k_matrix = np.asarray(v["intrinsics"][0].float().cpu())
        vf = round(_vfov_of(k_matrix, h), 2)
        readings[cam] = {
            "vfov_deg": vf,
            "commissioned_height_m": base[cam]["height_m"],
            "commissioned_pitch_deg": base[cam]["pitch_deg"],
        }
        if cam == ANCHOR_CAMERA:
            readings[cam]["focal_ratio_vs_tile_grid"] = round(
                focal_disagreement(ANCHOR_VFOV_DEG, vf), 3
            )
        print(f"  {cam:<18} vfov {vf:6.2f}", flush=True)
    return readings


def run_register(plates: dict[str, Path], groups: dict[str, tuple]) -> list[GroupReading]:
    """One reading per group. The control is a group like any other, on purpose."""
    import itertools

    import numpy as np
    import torch
    from mapanything.utils.image import load_images

    model, _dev = _load_model()
    out = []
    for label, (cams, is_control) in groups.items():
        paths = [str(plates[c]) for c in cams if c in plates]
        if len(paths) != len(cams):
            print(f"  {label}: skipped, a plate is missing")
            continue
        views = load_images(paths)
        with torch.no_grad():
            pred = model.infer(
                views,
                memory_efficient_inference=True,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
            )
        preds = pred if isinstance(pred, list) else [pred]
        centres, confs = [], []
        for v in preds:
            pose = np.asarray(v["camera_poses"][0].float().cpu())
            centres.append(pose[:3, 3])
            confs.append(round(float(np.nanmean(np.asarray(v["conf"][0].float().cpu()))), 3))
        dists = {
            f"{cams[i]}~{cams[j]}": round(float(np.linalg.norm(centres[i] - centres[j])), 3)
            for i, j in itertools.combinations(range(len(cams)), 2)
        }
        out.append(GroupReading(label, tuple(cams), tuple(confs), dists, is_control))
        print(f"  {label:<44} conf {confs}", flush=True)
    return out


# (cameras, is_control). The control is three cameras in three different cities.
DEFAULT_GROUPS = {
    "same room": (("Taichung-cam04", "Taichung-cam05", "Taichung-cam06"), False),
    "same store, far apart": (("Taichung-cam01", "Taichung-cam07", "Taichung-cam10"), False),
    "CONTROL: three different stores": (
        ("Taichung-cam04", "Kaohsiung-cam04", "Tao-Hsin-cam03"),
        True,
    ),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("intrinsics", "register"))
    ap.add_argument("--out", required=True, help="run directory for the readings")
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "note: this compares two estimates. Every commissioned height is fitted from the "
        "1.70 m person prior, and vfov is a fleet assumption on 22 of 23 cameras."
    )
    plates = undistorted_plates(out_dir / "undistorted")
    if not plates:
        print("::error::no commissioned plates found; nothing to measure")
        return 1

    if a.mode == "intrinsics":
        readings = run_intrinsics(plates)
        (out_dir / "intrinsics.json").write_text(json.dumps(readings, indent=2) + "\n")
        anchor = readings.get(ANCHOR_CAMERA, {})
        if "focal_ratio_vs_tile_grid" in anchor:
            print(
                f"\n{ANCHOR_CAMERA}: tile grid {ANCHOR_VFOV_DEG} deg against "
                f"{anchor['vfov_deg']} deg -- focal ratio "
                f"{anchor['focal_ratio_vs_tile_grid']}x. Neither is a tape measure."
            )
        return 0

    readings = run_register(plates, DEFAULT_GROUPS)
    verdict = registration_verdict(readings)
    payload = {
        "groups": [
            {
                "label": r.label,
                "cameras": list(r.cameras),
                "confidences": list(r.confidences),
                "pair_distances_m": r.pair_distances_m,
                "is_control": r.is_control,
            }
            for r in readings
        ],
        "verdict": {
            "usable": verdict.usable,
            "reason": verdict.reason,
            "control_max": verdict.control_max,
            "overlapping_min": verdict.overlapping_min,
            "separation": round(verdict.separation, 3),
        },
    }
    (out_dir / "register.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nverdict: {'USABLE' if verdict.usable else 'NOT USABLE'} -- {verdict.reason}")
    return 0 if verdict.usable else 1


if __name__ == "__main__":
    sys.exit(main())
