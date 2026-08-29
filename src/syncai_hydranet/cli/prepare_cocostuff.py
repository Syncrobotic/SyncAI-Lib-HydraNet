"""Build a `seg_folder` split from COCO-Stuff, filtered to the classes we are short of.

    hydranet-prepare-cocostuff --coco datasets/coco --stuff datasets/cocostuff \
        --out datasets/cocostuff_seg

COCO-Stuff labels the images already in ``datasets/coco``, so nothing is copied: the
split is symlinks into both trees. What it costs is disk inodes, not 39 GB.

**Why a filter rather than all 118,287.** ``cocostuff_retail`` deliberately maps only
what it is confident about, and COCO is mostly outdoors -- sky, sea, grass, tree all fall
through to ignore. Feeding the whole split in would spend most of its compute on frames
that produce no gradient. So an image is kept only when it carries a real amount of a
class this project is actually short of; ``--classes`` and ``--min-fraction`` say what
that means and both are recorded in the manifest.

**Why there is no val split.** COCO-Stuff arrives here as a *second labelled domain*, and
its value is that nothing selects on it. Training keeps choosing checkpoints on ADE20K
val exactly as before, so runs stay comparable with everything measured to date, and
``val2017`` becomes a test split that no part of training reads. This follows
`docs/PLAN.md` §4.4, which carries the rule now: the split that answers the question is
never the split that picked the answer.

The tool refuses to write into a non-empty split rather than merging into it. Re-running
``hydranet-prepare-ade20k`` with different flags left a stale test split beside a rebuilt
val one and nothing warned; that cost a day of trusting contaminated numbers.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

# Every list here is **names only**, resolved against the `labels.txt` that ships beside
# the annotations. Nothing is transcribed by hand.
#
# The first draft of this file did hand-transcribe the ids, and every entry in two of the
# three tables came out one too high -- the exact off-by-one this dataset is notorious
# for, reintroduced by the person writing the warning about it. A wrong id here does not
# raise: it silently filters on the neighbouring class, and the split still looks
# reasonable. Names cannot drift, so names are what the file stores.
SCARCE = (
    "stairs",
    "door-stuff",
    "mirror-stuff",
    "window-other",
    "carpet",
    "rug",
    "mat",
)
DEFAULT_CLASSES = SCARCE

# Class presence is not enough to make an image useful here, and the gap is not visible
# in any summary statistic -- it was caught by looking at four sampled frames. Selecting
# on `window-other` alone pulls in **car interiors**, where the side window becomes
# `glass`; selecting on `stairs` pulls in outdoor steps and street facades, where the
# building front becomes `wall` and the pavement becomes `floor_hard`. Training a shop
# robot on those teaches it the wrong thing about its two highest-consequence classes.
#
# A sky-and-grass test does not catch it: a photo taken close to a building facade has no
# sky in it, and a car interior has no outdoor class at all. So the filter is positive --
# the frame must show indoor *finishes* (ceiling, floor covering, finished wall) -- with
# outdoor and vehicle pixels as vetoes on top.
#
# `label_maps_indoor.ADE20K_ID_TO_INDOOR` carries the same warning for ADE20K, which
# ships sceneCategories.txt to solve it. COCO has no scene labels, hence this.
INDOOR_EVIDENCE = (
    "ceiling-other",
    "ceiling-tile",
    "floor-marble",
    "floor-other",
    "floor-stone",
    "floor-tile",
    "floor-wood",
    "carpet",
    "rug",
    "wall-other",
    "wall-panel",
    "wall-tile",
    "wall-wood",
)
OUTDOOR_VETO = (
    "sky-other",
    "tree",
    "grass",
    "sea",
    "clouds",
    "mountain",
    "snow",
    "sand",
    "road",
    "pavement",
    "building-other",
    "bush",
    "river",
    "roof",
)
# A frame with a vehicle in it is a street, or the inside of one, not a floor this robot
# drives on. `window-other` inside a car is what made this necessary.
VEHICLE_VETO = ("car", "motorcycle", "airplane", "bus", "train", "truck", "boat")


def _load_png_values(stuff: Path) -> dict[str, int]:
    """name -> the value the PNGs actually store, read from the dataset's own labels.txt.

    The file is 1-based against the annotations: `labels.txt` id N is stored as N-1, and
    `person` (1) is therefore value 0. Established by counting pixels; pinned by
    tests/test_cocostuff_scheme.py.
    """
    path = stuff / "labels.txt"
    if not path.is_file():
        raise SystemExit(f"{path} not found; it ships with the dataset and is required")
    out = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        idx, name = line.split(":", 1)
        out[name.strip()] = int(idx) - 1
    return out


def _resolve_named(png: dict[str, int], names) -> list[str]:
    missing = [n for n in names if n not in png]
    if missing:
        raise SystemExit(f"not COCO-Stuff class names: {', '.join(missing)}")
    return list(names)


def _resolve(png: dict[str, int], names) -> set[int]:
    missing = [n for n in names if n not in png]
    if missing:
        raise SystemExit(f"not COCO-Stuff class names: {', '.join(missing)}")
    return {png[n] for n in names}


def _scan(args):
    """Return (name, {png_value: fraction}, indoor_ok) for one annotation."""
    path, wanted, sets = args
    a = np.asarray(Image.open(path))
    total = a.size
    vals, counts = np.unique(a, return_counts=True)
    frac = {int(v): int(c) / total for v, c in zip(vals, counts, strict=True)}
    hit = {v: f for v, f in frac.items() if v in wanted}
    if sets is None:
        return path.name, hit, True
    ind_v, out_v, veh_v = sets
    ind = sum(frac.get(v, 0.0) for v in ind_v)
    out = sum(frac.get(v, 0.0) for v in out_v)
    veh = sum(frac.get(v, 0.0) for v in veh_v)
    return path.name, hit, (ind >= 0.05 and out < 0.02 and veh < 0.05)


def _select(ann_dir: Path, wanted: set[int], min_fraction: float, workers: int, sets):
    files = sorted(p for p in ann_dir.iterdir() if p.suffix.lower() == ".png")
    keep, per_class, dropped = [], dict.fromkeys(wanted, 0), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, hit, indoor_ok in ex.map(
            _scan, ((f, wanted, sets) for f in files), chunksize=256
        ):
            passing = [v for v, f in hit.items() if f >= min_fraction]
            if not passing:
                continue
            if not indoor_ok:
                dropped += 1
                continue
            keep.append(name)
            for v in passing:
                per_class[v] += 1
    return files, keep, per_class, dropped


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hydranet-prepare-cocostuff",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--coco", type=Path, default=Path("datasets/coco"))
    ap.add_argument("--stuff", type=Path, default=Path("datasets/cocostuff"))
    ap.add_argument("--out", type=Path, default=Path("datasets/cocostuff_seg"))
    ap.add_argument(
        "--classes",
        nargs="*",
        default=list(DEFAULT_CLASSES),
        help="labels.txt names an image must carry to be kept "
        f"(default: {' '.join(DEFAULT_CLASSES)})",
    )
    ap.add_argument(
        "--min-fraction",
        type=float,
        default=0.01,
        help="minimum share of the frame one of those classes must cover (default 0.01)",
    )
    ap.add_argument(
        "--no-indoor-filter",
        action="store_true",
        help="keep frames that carry the classes but show no indoor finishes; street "
        "facades and car interiors come with it, see INDOOR_EVIDENCE",
    )
    ap.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)))
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty split")
    return ap


def _refuse_existing_splits(out: Path, plan, force: bool) -> None:
    """Refuse before any work, rather than merging into what a previous run left.

    A half-rebuilt split looks exactly like a good one from the outside: the right
    directories, plausible counts, and no way to tell which frames came from which
    filter settings.
    """
    for _src_split, dst_split in plan:
        for kind in ("images", "annotations"):
            d = out / kind / dst_split
            if d.is_dir() and any(d.iterdir()) and not force:
                raise SystemExit(
                    f"{d} already has files. Refusing to merge into an existing split -- "
                    "a half-rebuilt split is indistinguishable from a good one. "
                    "Delete it or pass --force."
                )


def _link_split(keep, img_dir: Path, ann_dir: Path, out_img: Path, out_ann: Path):
    """Symlink the kept pairs. Returns `(linked, missing)`.

    An annotation whose image is absent is collected rather than skipped quietly: it
    means the two archives were unpacked from different releases, and the count is the
    only signal of that.
    """
    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)
    linked, missing = 0, []
    for name in keep:
        jpg = img_dir / (Path(name).stem + ".jpg")
        if not jpg.exists():
            missing.append(name)
            continue
        _link(jpg, out_img / jpg.name)
        _link(ann_dir / name, out_ann / name)
        linked += 1
    return linked, missing


def _report_split(src_split, dst_split, files, keep, linked, dropped, missing, img_dir, names):
    """What this split scanned, kept and linked -- never a filtered count as the whole."""
    print(
        f"[{src_split} -> {dst_split}] scanned {len(files):,}, "
        f"kept {len(keep):,}, linked {linked:,}"
    )
    if dropped:
        print(f"    dropped {dropped:,} that carried the classes but were not indoor scenes")
    for v, n in sorted(names.items(), key=lambda kv: -kv[1]):
        print(f"    {v:14} {n:>7,}")
    if missing:
        print(f"    !! {len(missing)} annotations had no image in {img_dir} and were skipped")


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)

    png = _load_png_values(a.stuff)
    values = {n: png[n] for n in _resolve_named(png, a.classes)}
    wanted = set(values.values())
    by_value = {v: n for n, v in values.items()}
    sets = (
        None
        if a.no_indoor_filter
        else (
            _resolve(png, INDOOR_EVIDENCE),
            _resolve(png, OUTDOOR_VETO),
            _resolve(png, VEHICLE_VETO),
        )
    )

    # train2017 -> train,  val2017 -> test.  Deliberately no val; see the module docstring.
    plan = [("train2017", "train"), ("val2017", "test")]
    _refuse_existing_splits(a.out, plan, a.force)

    manifest = {
        "classes": list(a.classes),
        "min_fraction": a.min_fraction,
        "indoor_filter": sets is not None,
        "splits": {},
    }
    for src_split, dst_split in plan:
        ann_dir = a.stuff / src_split
        img_dir = a.coco / src_split
        for d in (ann_dir, img_dir):
            if not d.is_dir():
                raise SystemExit(f"{d} does not exist")

        files, keep, per_class, dropped = _select(
            ann_dir, wanted, a.min_fraction, a.workers, sets
        )
        named = {by_value[v]: n for v, n in per_class.items()}
        linked, missing = _link_split(
            keep,
            img_dir,
            ann_dir,
            a.out / "images" / dst_split,
            a.out / "annotations" / dst_split,
        )
        _report_split(
            src_split, dst_split, files, keep, linked, dropped, missing, img_dir, named
        )
        manifest["splits"][dst_split] = {
            "source": src_split,
            "scanned": len(files),
            "kept": len(keep),
            "linked": linked,
            "missing_images": len(missing),
            "dropped_not_indoor": dropped,
            "per_class": named,
        }

    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {a.out}/manifest.json")
    print("no val split by design: selection stays on ADE20K, so `test` here stays clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
