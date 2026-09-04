#!/usr/bin/env python3
"""Check a site dataset against RETAIL_DATA.md's rules, from the record.

    python3 scripts/site_split_report.py --dataset datasets/retail_objects_batch01

That document opens by saying most of its rules "are only enforceable by someone choosing
to honour them later". This is the part that does not have to be. R1, R2, R4, R6 and R7
are all decidable from the manifests and the masks; R3 and R5 are judgements and are
reported as unchecked rather than guessed at.

**Why a tool rather than a look.** batch01's clips are named `archive_<start>_<end>` with
no camera in the name, so R1 -- split by camera, the one rule whose violation invalidates
every number after it -- could not be read off the directory at all. The camera is
recoverable, from `manifest_*.json`, and the recovery turned up something a glance would
not: **8 of 184 clip stems are claimed by more than one camera.** Two cameras in different
stores can start a recording in the same second, and then their files have the same name.
One of those collisions is in batch01's test split.

So the flat naming is not merely inconvenient, it is lossy, and no amount of care at
labelling time recovers it. `--write-cameras` writes the attribution back as a
`cameras.json` beside the dataset rather than renaming anything, because a rename would
invalidate every `meta.json` that has already recorded a path.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np
from PIL import Image

from syncai_hydranet.data.label_maps_retail_objects import RETAIL_OBJECTS

HERE = pathlib.Path(__file__).resolve().parent


AMBIGUOUS = "AMBIGUOUS"
# Under this share of a split's labelled pixels a class is "rare" for R4/R7 purposes.
# 1% is `evaluator.THIN_SUPPORT`, set there on `column` at 0.66% -- the one class known to
# have failed this way. Same threshold, same reason, stated rather than re-derived.
RARE_SHARE = 0.01


def camera_index(clip_root: pathlib.Path) -> dict[str, set[str]]:
    """clip stem -> the camera(s) any manifest attributes it to."""
    index: dict[str, set[str]] = collections.defaultdict(set)
    for path in sorted(clip_root.glob("manifest_*.json")):
        for entry in json.loads(path.read_text()).get("pulled", []):
            camera = entry.get("camera")
            for clip in entry.get("clips", []):
                index[pathlib.Path(clip["uri"]).stem].add(camera)
    return index


def resolve(name: str, index: dict[str, set[str]]) -> str:
    """The camera a clip directory belongs to.

    A `Camera__stem` prefix wins over the manifest: it was written at export time by
    something that knew, and it is the form that cannot collide.
    """
    if "__" in name:
        return name.split("__", 1)[0]
    cameras = index.get(name, set())
    if len(cameras) == 1:
        return next(iter(cameras))
    return AMBIGUOUS


def class_pixels(ann_dir: pathlib.Path, n: int) -> dict[str, np.ndarray]:
    """Per-clip class pixel counts, from the masks rather than from a model."""
    out = {}
    for clip in sorted(p for p in ann_dir.iterdir() if p.is_dir()):
        total = np.zeros(n, np.int64)
        for mask in sorted(clip.glob("*.png")):
            counts = np.bincount(np.asarray(Image.open(mask)).ravel(), minlength=256)
            total += counts[:n]
        out[clip.name] = total
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="site-split-report", description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clips", default="datasets/studioa_clips")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument(
        "--write-cameras",
        action="store_true",
        help="write <dataset>/cameras.json with the resolved attribution",
    )
    args = ap.parse_args(argv)

    root = pathlib.Path(args.dataset)
    names = list(RETAIL_OBJECTS)
    index = camera_index(pathlib.Path(args.clips))
    collisions = sum(1 for v in index.values() if len(v) > 1)
    print(f"manifests attribute {len(index)} clips; {collisions} stems claimed by >1 camera\n")

    per_split: dict[str, dict[str, np.ndarray]] = {}
    cameras: dict[str, dict[str, list[str]]] = {}
    failures: list[str] = []

    for split in args.splits:
        ann = root / "annotations" / split
        if not ann.is_dir():
            continue
        pixels = class_pixels(ann, len(names))
        by_camera: dict[str, np.ndarray] = collections.defaultdict(
            lambda: np.zeros(len(names), np.int64)
        )
        members: dict[str, list[str]] = collections.defaultdict(list)
        for clip, counts in pixels.items():
            cam = resolve(clip, index)
            by_camera[cam] += counts
            members[cam].append(clip)
        per_split[split] = dict(by_camera)
        cameras[split] = {k: sorted(v) for k, v in members.items()}
        print(f"{split}: {len(pixels)} clips over {len(by_camera)} cameras")
        if AMBIGUOUS in by_camera:
            failures.append(
                f"{split} has {len(members[AMBIGUOUS])} clip(s) no manifest can attribute "
                f"to one camera: {', '.join(members[AMBIGUOUS])}"
            )

    train, test = per_split.get("train", {}), per_split.get("test", {})

    print("\n--- R1 / R2: split by camera, and a test camera supplies no training frame")
    shared = (set(train) & set(test)) - {AMBIGUOUS}
    if shared:
        failures.append(f"R2 VIOLATED: {sorted(shared)} appear in both train and test")
        print(f"  FAIL  shared cameras: {sorted(shared)}")
    else:
        print(f"  pass  {len(set(train) - {AMBIGUOUS})} train cameras, "
              f"{len(set(test) - {AMBIGUOUS})} test cameras, no overlap")  # fmt: skip

    print("\n--- R6: all three stores on the test side")
    stores = {c.split("-cam")[0] for c in test if c != AMBIGUOUS}
    print(f"  {'pass' if len(stores) >= 3 else 'FAIL'}  test stores: {sorted(stores)}")
    if len(stores) < 3:
        failures.append(f"R6: test covers {len(stores)} store(s), not 3")

    print("\n--- R4 / R7: a rare class needs at least two test cameras")
    labelled = sum(v.sum() for v in test.values())
    print(f"  {'class':10s} {'% of test px':>13s} {'cameras':>9s}")
    for i, name in enumerate(names):
        share = sum(v[i] for v in test.values()) / max(labelled, 1)
        n_cams = sum(1 for c, v in test.items() if c != AMBIGUOUS and v[i] > 0)
        flag = ""
        if share == 0:
            flag = "  not present -- not measurable on this split"
        elif n_cams < 2:
            flag = "  R7: report as 'not measured', not as a number"
            failures.append(f"R4/R7: {name} stands on {n_cams} test camera(s)")
        elif share < RARE_SHARE:
            flag = f"  thin ({share:.2%} < {RARE_SHARE:.0%}); quote the support with it"
        print(f"  {name:10s} {100 * share:12.2f}% {n_cams:9d}{flag}")

    print("\n--- R5: the richest camera for each rare class stays in training")
    for i, name in enumerate(names):
        if not any(v[i] for v in test.values()):
            continue
        best_tr = max(
            ((c, v[i]) for c, v in train.items()), key=lambda kv: kv[1], default=("", 0)
        )
        best_te = max(
            ((c, v[i]) for c, v in test.items()), key=lambda kv: kv[1], default=("", 0)
        )
        verdict = "pass" if best_tr[1] >= best_te[1] else "REVIEW"
        print(f"  {verdict:6s} {name:10s} richest train {best_tr[0]} ({best_tr[1]:,}) "
              f"vs test {best_te[0]} ({best_te[1]:,})")  # fmt: skip

    print("\n--- R3: test masks must be human-corrected")
    print("  NOT CHECKABLE HERE. Nothing in a mask file records who drew it. If these came "
          "from SAM 3 and were not corrected, every number above measures agreement with "
          "SAM 3 rather than correctness -- see RETAIL_DATA.md R3.")  # fmt: skip

    if args.write_cameras:
        out = root / "cameras.json"
        out.write_text(json.dumps(cameras, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {out}")

    print("\n" + ("=" * 72))
    if failures:
        print(f"{len(failures)} finding(s):")
        for f in failures:
            print(f"  * {f}")
    else:
        print("no findings against the checkable rules (R3 and R5 still need a human)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
