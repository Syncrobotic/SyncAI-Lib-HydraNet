"""The annotation spec, as something a machine checks.

    hydranet-annotation labels --out cvat_labels.json    # the schema CVAT imports
    hydranet-annotation check datasets/site_a            # the gate before training

[docs/ANNOTATION_SETUP.md](../../../docs/ANNOTATION_SETUP.md) states the contract between
annotators and training in prose, and prose drifts: a class renamed in the tool, an id
typed by hand, a background exported as 0 where the loss expects 255, one session
appearing in both train and val. None of that raises an error. Training accepts the
masks, the loss falls, every metric stays plausible -- and the model learns something
other than what the labels meant.

These two commands cover the half of the spec a machine can verify. ``labels`` emits the
class list so nobody types it twice; ``check`` reads a finished dataset and refuses it if
it violates the contract. What is left to human review is the part that matters most --
whether the pixels are labelled *correctly* -- which is why the checks here try to be
exhaustive about everything else.

``check`` deliberately reuses training's own pairing and label-map code rather than
reimplementing the layout rules, so it cannot pass a dataset that training would read
differently.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.datasets import _index_pairs
from ..data.label_maps_indoor import (
    INDOOR_NATIVE_ID,
    INDOOR_TERRAIN,
    INDOOR_TERRAIN_TO_TRAV,
)
from ..utils.visualize import TERRAIN_COLORS

IGNORE = 255
TRAV_NAMES = {0: "blocked", 1: "caution", 2: "go", IGNORE: "ignore"}
ID_TO_NAME = {v: k for k, v in INDOOR_TERRAIN.items()}

# Annotation priority, from ANNOTATION_SETUP.md: rank by danger x how blind LiDAR is.
# Confirmed annotate-only -- the full-vocabulary ADE20K does not supply these three.
PRIORITY = ("glass", "wet_slippery", "floor_metal", "threshold_ramp", "stairs")


# --------------------------------------------------------------------- labels


def label_spec() -> list[dict]:
    """The CVAT label list for the 12-class indoor scheme.

    Colours come from the palette the inference overlays use, so a mask drawn in CVAT
    and a prediction rendered by ``hydranet-infer-video`` are the same colour for the
    same class. Reviewing a model against its labels is hard enough without a
    translation step in the middle.

    ``void`` is omitted on purpose: it is what unlabelled pixels become, not something
    an annotator draws. See ``check`` for why that distinction has teeth.
    """
    spec = []
    for name, tid in INDOOR_TERRAIN.items():
        if name == "void":
            continue
        r, g, b = (int(c) for c in TERRAIN_COLORS[tid])
        trav = TRAV_NAMES.get(INDOOR_TERRAIN_TO_TRAV[tid], "?")
        spec.append(
            {
                "name": name,
                "color": f"#{r:02x}{g:02x}{b:02x}",
                "type": "mask",
                "attributes": [],
                # CVAT ignores unknown keys on import; these two ride along so the
                # exported project carries the contract with it.
                "terrain_id": tid,
                "traversability": trav,
            }
        )
    return spec


def cmd_labels(args) -> int:
    spec = label_spec()
    text = json.dumps(spec, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({len(spec)} labels)")
        print("Import it in CVAT: project -> Raw labels -> paste -> Done.\n")
    else:
        print(text)
        return 0

    print(f"{'id':>3}  {'class':<20} {'colour':<9} traversability")
    for entry in spec:
        print(
            f"{entry['terrain_id']:>3}  {entry['name']:<20} {entry['color']:<9} "
            f"{entry['traversability']}"
        )
    print(
        "\nid 0 (void) is not a label: leave a pixel unlabelled and export it as 255.\n"
        "Traversability is derived from terrain, never annotated directly."
    )
    return 0


# ---------------------------------------------------------------------- check


class Report:
    """Findings, split into what blocks training and what merely deserves a look."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _session_of(img_path: Path, img_dir: Path) -> str:
    """A session is the directory a session's frames were written into.

    Adjacent frames of one walk-through are near-identical, so a random split measures
    memorisation. The layout is what carries session identity into training; frames
    dumped flat into one directory cannot be split honestly.
    """
    rel = img_path.relative_to(img_dir).parent
    return rel.as_posix() if rel.as_posix() != "." else "(flat)"


def _read_mask(path: Path) -> np.ndarray | None:
    """Load one annotation as class ids, or None if it is not single-channel.

    Mirrors what ``SegFolderDataset._decode_ann`` does for an id-format scheme: mode P
    is fine (the palette indices *are* the ids), mode RGB is not -- training would read
    the red channel and call it a class.
    """
    ann = Image.open(path)
    if ann.mode not in ("L", "P", "I", "I;16"):
        return None
    return np.asarray(ann)


def check_split(root: Path, split: str, rep: Report, allow_void: bool) -> dict:
    """Validate one split and return its per-class pixel and image counts."""
    img_dir, ann_dir = root / "images" / split, root / "annotations" / split
    stats = {
        "pixels": Counter(),
        "images": Counter(),
        "sessions": defaultdict(int),
        "n_pairs": 0,
    }
    if not img_dir.is_dir() or not ann_dir.is_dir():
        rep.error(f"{split}: expected {img_dir} and {ann_dir}")
        return stats

    pairs = _index_pairs(img_dir, ann_dir)
    stats["n_pairs"] = len(pairs)
    if not pairs:
        rep.error(f"{split}: no image/annotation pairs -- training would refuse this split")
        return stats

    # Anything unpaired is dropped silently by the loader, which is how an annotation
    # run "finishes" with a third of its work not being trained on.
    paired_imgs = {p.resolve() for p, _ in pairs}
    paired_anns = {a.resolve() for _, a in pairs}
    lonely_imgs = [
        p
        for p in img_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        and p.resolve() not in paired_imgs
    ]
    lonely_anns = [
        p
        for p in ann_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        and p.resolve() not in paired_anns
    ]
    for label, lonely in (("image", lonely_imgs), ("annotation", lonely_anns)):
        if lonely:
            shown = ", ".join(p.name for p in sorted(lonely)[:3])
            rep.error(
                f"{split}: {len(lonely)} {label}(s) with no counterpart, silently "
                f"dropped by the loader (e.g. {shown})"
            )

    allowed = set(INDOOR_NATIVE_ID) | {IGNORE}
    void_is_trained = INDOOR_NATIVE_ID.get(0) == 0
    void_pixels = 0

    for img_path, ann_path in pairs:
        mask = _read_mask(ann_path)
        if mask is None:
            rep.error(
                f"{ann_path.name}: {Image.open(ann_path).mode} mode, not single channel. "
                "Export the mask as a single-channel PNG whose pixel value is the class id."
            )
            continue
        if mask.ndim != 2:
            rep.error(f"{ann_path.name}: {mask.ndim}-d annotation, expected 2-d")
            continue

        with Image.open(img_path) as im:
            if im.size != (mask.shape[1], mask.shape[0]):
                rep.error(
                    f"{ann_path.name}: mask {mask.shape[1]}x{mask.shape[0]} but image "
                    f"{im.size[0]}x{im.size[1]} -- the two are resized independently, so "
                    "the labels would be geometrically wrong, not merely misaligned"
                )
                continue

        present = np.unique(mask)
        unknown = sorted(int(v) for v in present if int(v) not in allowed)
        if unknown:
            rep.error(
                f"{ann_path.name}: ids {unknown} are not in the 12-class scheme; the "
                "loader maps them to ignore, so those pixels train on nothing"
            )
        counts = np.bincount(mask.ravel(), minlength=256)
        for tid in present:
            tid = int(tid)
            if tid in allowed:
                stats["pixels"][tid] += int(counts[tid])
                stats["images"][tid] += 1
        void_pixels += int(counts[0])
        stats["sessions"][_session_of(img_path, img_dir)] += 1

    total_px = sum(stats["pixels"].values())
    if void_pixels and void_is_trained and not allow_void:
        share = 100 * void_pixels / max(total_px, 1)
        rep.error(
            f"{split}: {share:.1f}% of pixels are id 0 (void), and the indoor_native map "
            "sends 0 -> 0, so the loss trains them as a class rather than ignoring them. "
            "Most tools export unlabelled background as 0: remap it to 255 on export, or "
            "pass --allow-void if this dataset really does mean void"
        )
    return stats


def check_sessions(per_split: dict[str, dict], rep: Report) -> None:
    """One session must not appear in two splits.

    This is the leak that looks like success: adjacent frames are near-identical, so a
    session spanning train and val produces a val score that measures memorisation. It
    is also invisible afterwards -- the number is simply too good, and nothing says why.
    """
    owners: dict[str, list[str]] = defaultdict(list)
    for split, stats in per_split.items():
        for session in stats["sessions"]:
            owners[session].append(split)
    for session, splits in sorted(owners.items()):
        if session == "(flat)":
            continue
        if len(splits) > 1:
            rep.error(
                f"session {session!r} appears in {', '.join(sorted(splits))} -- split by "
                "session, never by frame"
            )
    if any("(flat)" in stats["sessions"] for stats in per_split.values()):
        rep.warn(
            "frames sit directly in images/<split>/ with no session directory, so session "
            "discipline cannot be checked. Write each capture session to its own "
            "subdirectory, e.g. images/train/lobby-2026-08-20-a/"
        )
    for split, stats in per_split.items():
        sessions = [s for s in stats["sessions"] if s != "(flat)"]
        if split in ("val", "test") and len(sessions) == 1:
            rep.warn(
                f"{split} comes from a single session ({sessions[0]}), so it measures one "
                "room's lighting and geometry rather than the site"
            )


def _ignored_ids() -> set[int]:
    """The ids the loss never sees, whatever the label map calls them.

    255 always, plus anything ``indoor_native`` sends to 255 -- id 0 does, now that
    unlabelled background is mapped to ignore rather than to a `void` class. Counting
    those pixels in a coverage percentage would quietly shrink every class in the table
    in proportion to how much of the frame nobody labelled.
    """
    return {IGNORE} | {src for src, dst in INDOOR_NATIVE_ID.items() if dst == IGNORE}


def _print_coverage(per_split: dict[str, dict]) -> None:
    splits = list(per_split)
    ignored = _ignored_ids()
    total_px = {
        s: max(sum(px for tid, px in per_split[s]["pixels"].items() if tid not in ignored), 1)
        for s in splits
    }

    print("\nterrain coverage (% of labelled pixels / images containing the class)")
    header = "".join(f"{s:>22}" for s in splits)
    print(f"{'id':>3}  {'class':<20}{header}")
    for tid, name in sorted(ID_TO_NAME.items()):
        if tid in ignored:
            continue
        cells = ""
        for s in splits:
            px = per_split[s]["pixels"][tid]
            n = per_split[s]["images"][tid]
            cells += f"{100 * px / total_px[s]:>15.2f}% /{n:>5}"
        flag = (
            "  <- priority"
            if name in PRIORITY and all(per_split[s]["pixels"][tid] == 0 for s in splits)
            else ""
        )
        print(f"{tid:>3}  {name:<20}{cells}{flag}")

    # The traversability head is trained through the policy table, so this is the
    # distribution that decides whether `caution` can ever be learned.
    print("\nderived traversability (% of labelled pixels)")
    for trav_id in (0, 1, 2):
        cells = ""
        for s in splits:
            px = sum(
                per_split[s]["pixels"][tid]
                for tid, t in INDOOR_TERRAIN_TO_TRAV.items()
                if t == trav_id
            )
            cells += f"{100 * px / total_px[s]:>21.2f}%"
        print(f"     {TRAV_NAMES[trav_id]:<20}{cells}")

    # How much of the frame nobody labelled. An annotation batch that comes back 95%
    # ignore is a batch that mostly did not happen, and the class percentages above --
    # which are shares of what *is* labelled -- will not show it.
    print("\nunlabelled (ignored by the loss, % of all pixels)")
    cells = ""
    for s in splits:
        all_px = max(sum(per_split[s]["pixels"].values()), 1)
        ignored_px = sum(per_split[s]["pixels"][tid] for tid in ignored)
        cells += f"{100 * ignored_px / all_px:>21.2f}%"
    print(f"     {'ignore':<20}{cells}")


def cmd_check(args) -> int:
    root = Path(args.root)
    if not (root / "images").is_dir():
        raise SystemExit(f"{root}/images not found; see README for the expected layout")

    splits = args.split or sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
    rep = Report()
    per_split = {s: check_split(root, s, rep, args.allow_void) for s in splits}
    check_sessions(per_split, rep)

    print(f"{root}  splits: {', '.join(splits)}")
    for s in splits:
        n_sessions = len([x for x in per_split[s]["sessions"] if x != "(flat)"])
        print(f"  {s:<8} {per_split[s]['n_pairs']:>6} frames  {n_sessions:>3} sessions")
    _print_coverage(per_split)

    for msg in rep.warnings:
        print(f"\nWARN  {msg}")
    for msg in rep.errors:
        print(f"\nFAIL  {msg}")
    if rep.errors:
        print(f"\n{len(rep.errors)} contract violation(s). Do not train on this yet.")
        return 1
    print("\nContract checks passed. What is labelled is labelled consistently;")
    print("whether it is labelled *correctly* is still a human question.")
    return 0


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-annotation", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_labels = sub.add_parser("labels", help="emit the CVAT label schema for the 12 classes")
    p_labels.add_argument("--out", help="write JSON here instead of stdout")
    p_labels.set_defaults(func=cmd_labels)

    p_check = sub.add_parser("check", help="verify an annotated dataset against the contract")
    p_check.add_argument("root", help="dataset root, containing images/ and annotations/")
    p_check.add_argument("--split", action="append", help="check only this split (repeatable)")
    p_check.add_argument(
        "--allow-void",
        action="store_true",
        help="accept id 0 as a real class rather than an unremapped background",
    )
    p_check.set_defaults(func=cmd_check)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
