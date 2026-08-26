"""What is actually inside `NTU60_CS.npz`, and whether PLAN's projection route survives it.

    uv run python tools/temporal/ntu_survey.py \\
        --npz datasets/_incoming/ntu60/NTU60_CS.npz --out runs/ntu_survey01

The plan (docs/PLAN.md 2.3, 4.3) replaces staged in-store clips with public 3D action data
projected through our own measured camera parameters. That route makes three assumptions
about the file, and a redistribution of NTU RGB+D is exactly the kind of artefact where
they can quietly fail -- most published NTU tensors have been through a preprocessing
pipeline that centres, rotates and rescales the skeletons, and a view-normalised skeleton
has had the very quantity we mean to re-impose already removed from it.

So this measures, rather than assumes:

1. **Shape and labels** -- how many sequences, how many bodies, which of the 60 classes
   are present and in what numbers.
2. **What the coordinates still are.** Centred? Rotated to a canonical pose? In real
   metres? Answered from the data: the spread of the first frame's spine direction and
   origin across samples, and whether torso length and shoulder width carry between-subject
   variation or a single normalised value.
3. **Whether the classes separate the way our detector assumes.** `events/pose.py` fires
   `fall` on shoulder-to-hip angle from vertical, sustained. NTU has ground-truth 3D for
   both `A43 falling down` and `A06 pick up`, so the question "does the torso angle tell
   them apart" has an answer here that costs nothing to get, and it is the answer that
   decides whether a temporal model trained on this data can be trusted in a shop.

The output is one JSON of numbers, so a later reader argues with the measurement rather
than with a recollection of it.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

# Kinect v2's 25-joint layout, only the joints this survey touches. Named rather than
# written as integers at the use site: the 25-joint order is a property of the sensor and
# a bare `b[:, 8]` in an angle calculation is unreviewable.
J = {
    "spine_base": 0,
    "spine_mid": 1,
    "neck": 2,
    "head": 3,
    "l_shoulder": 4,
    "r_shoulder": 8,
    "l_hip": 12,
    "r_hip": 16,
    "l_foot": 14,
    "r_foot": 18,
}
# The classes this project has a use for, and the two controls. `A06 pick up` is here
# because a shopper reaching a bottom shelf is what our `crouch` is trying to be, and
# because it is the nearest thing to a fall that must NOT alarm.
CLASSES = {
    42: "A43 falling down",
    41: "A42 staggering",
    5: "A06 pick up",
    7: "A08 sit down",
    26: "A27 jump up",
    0: "A01 drink water",
}
FALL_ANGLE_DEG = 55.0  # events/pose.py's default, so the comparison is the shipped one


def bodies(sample: np.ndarray) -> np.ndarray:
    """(300, 150) -> (T, 2, 25, 3) with the all-zero padding frames dropped."""
    s = sample.reshape(-1, 2, 25, 3)
    return s[np.abs(s).sum(axis=(1, 2, 3)) > 0]


def torso_from_vertical(b: np.ndarray) -> np.ndarray:
    """Per frame: the angle of the shoulder-to-hip line from vertical, in degrees.

    The same quantity `events.pose._torso` measures in image space, computed here in 3D
    where it is not a projection of anything -- which is the point of asking NTU.
    """
    sh = (b[:, J["l_shoulder"]] + b[:, J["r_shoulder"]]) / 2
    hp = (b[:, J["l_hip"]] + b[:, J["r_hip"]]) / 2
    d = sh - hp
    return np.degrees(np.arctan2(np.linalg.norm(d[:, [0, 2]], axis=1), d[:, 1]))


def read_member(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        return np.lib.format.read_array(f)


def header(zf: zipfile.ZipFile, name: str) -> tuple:
    with zf.open(name) as f:
        ver = np.lib.format.read_magic(f)
        shape, _, dtype = np.lib.format._read_array_header(f, ver)
    return shape, dtype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/ntu_survey01"))
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(args.npz)
    report: dict = {"npz": args.npz, "split": args.split, "members": {}}
    for name in zf.namelist():
        shape, dtype = header(zf, name)
        report["members"][name] = {"shape": list(shape), "dtype": str(dtype)}
    print(json.dumps(report["members"], indent=1))

    y = read_member(zf, f"y_{args.split}.npy")
    labels = y.argmax(1)
    counts = Counter(labels.tolist())
    report["samples"] = len(labels)
    report["classes_present"] = len(counts)
    report["per_class_count"] = {"min": min(counts.values()), "max": max(counts.values())}

    # The array is read once, streaming, because it is ~3 GB for test and ~7 GB for train.
    # Only the sampled indices are kept.
    wanted: dict[int, list[int]] = {
        c: list(np.where(labels == c)[0][: args.per_class]) for c in CLASSES
    }
    keep = sorted({int(i) for v in wanted.values() for i in v})
    shape, dtype = header(zf, f"x_{args.split}.npy")
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    held: dict[int, np.ndarray] = {}
    two_body = 0
    with zf.open(f"x_{args.split}.npy") as f:
        np.lib.format.read_magic(f)
        np.lib.format._read_array_header(f, (1, 0))
        pos = 0
        for i in keep:
            while pos < i:
                f.read(row_bytes)
                pos += 1
            row = np.frombuffer(f.read(row_bytes), dtype=dtype).reshape(shape[1:])
            pos += 1
            held[i] = row.copy()
            if np.abs(row.reshape(-1, 2, 25, 3)[:, 1]).max() > 0:
                two_body += 1
    report["two_body_share_of_sample"] = round(two_body / max(len(held), 1), 3)

    # --- what the coordinates still are -------------------------------------------
    spine_dirs, origins, torso_len, shoulder_w, foot_std = [], [], [], [], []
    for i, row in held.items():
        b = bodies(row)[:, 0]
        if not len(b):
            continue
        sp = b[0, J["spine_mid"]] - b[0, J["spine_base"]]
        n = np.linalg.norm(sp)
        if n > 0:
            spine_dirs.append(sp / n)
        origins.append(b[0, J["spine_base"]])
        sh = (b[:, J["l_shoulder"]] + b[:, J["r_shoulder"]]) / 2
        hp = (b[:, J["l_hip"]] + b[:, J["r_hip"]]) / 2
        torso_len.append(float(np.median(np.linalg.norm(sh - hp, axis=1))))
        shoulder_w.append(
            float(
                np.median(np.linalg.norm(b[:, J["l_shoulder"]] - b[:, J["r_shoulder"]], axis=1))
            )
        )
        if labels[i] == 0:  # a standing action: are the feet planted?
            foot = np.minimum(b[:, J["l_foot"], 1], b[:, J["r_foot"], 1])
            foot_std.append(float(foot.std()))
    report["frame0_spine_direction"] = {
        "median": np.median(np.stack(spine_dirs), axis=0).round(3).tolist(),
        "std": np.stack(spine_dirs).std(axis=0).round(3).tolist(),
    }
    report["frame0_spine_base_origin"] = {
        "median": np.median(np.stack(origins), axis=0).round(3).tolist(),
        "std": np.stack(origins).std(axis=0).round(3).tolist(),
    }
    for nm, vals in (("torso_length_m", torso_len), ("shoulder_width_m", shoulder_w)):
        a = np.array(vals)
        report[nm] = {
            "p10": round(float(np.percentile(a, 10)), 3),
            "median": round(float(np.median(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3),
            "std": round(float(a.std()), 3),
        }
    if foot_std:
        report["standing_foot_height_std_m"] = round(float(np.median(foot_std)), 4)

    # --- does the torso angle separate a fall from a bend? ------------------------
    per_class = {}
    for c, nm in CLASSES.items():
        peaks = []
        for i in wanted[c]:
            b = bodies(held[int(i)])[:, 0]
            if len(b):
                peaks.append(float(torso_from_vertical(b).max()))
        a = np.array(peaks)
        per_class[nm] = {
            "n": len(a),
            "peak_torso_deg_median": round(float(np.median(a)), 1),
            "peak_torso_deg_p10": round(float(np.percentile(a, 10)), 1),
            "over_fall_threshold": int((a >= FALL_ANGLE_DEG).sum()),
        }
        print(
            f"{nm:<20} peak torso median {per_class[nm]['peak_torso_deg_median']:5.1f} deg  "
            f"over {FALL_ANGLE_DEG:.0f} deg in {per_class[nm]['over_fall_threshold']}/{len(a)}"
        )
    report["fall_angle_deg"] = FALL_ANGLE_DEG
    report["peak_torso_by_class"] = per_class

    out = args.out / f"ntu60_{args.split}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
