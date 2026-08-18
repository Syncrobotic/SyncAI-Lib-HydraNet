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
from ..data.label_maps_retail import (
    RETAIL_NATIVE_ID,
    RETAIL_TERRAIN,
    RETAIL_TERRAIN_TO_TRAV,
)
from ..data.label_maps_retail_objects import (
    RETAIL_OBJECTS,
    RETAIL_OBJECTS_NATIVE_ID,
    RETAIL_OBJECTS_TO_TRAV,
)
from ..utils.visualize import terrain_palette

IGNORE = 255
TRAV_NAMES = {0: "blocked", 1: "caution", 2: "go", IGNORE: "ignore"}

# Annotation priority, from ANNOTATION_SETUP.md: rank by danger x how blind LiDAR is.
# Confirmed annotate-only -- the full-vocabulary ADE20K does not supply these three.
# `display_fixture` is retail-only and joins the list for the same reason: no public
# dataset has shop fixtures, and ADE20K's tables are domestic.
#
# `column` and `product` join for the object taxonomy, and they are the two the coverage
# table must flag loudest. `product` has no public source at all. `column` has one in
# ADE20K (id 43) but **not** in any mask this project has already drawn: every site mask
# so far was annotated under the retail 13, where a column is `wall`. Migrated data
# therefore reports 0.00% for it while looking entirely healthy, which is exactly the
# shape of the `wet_slippery` failure this list exists to make visible.
PRIORITY = (
    "glass",
    "wet_slippery",
    "floor_metal",
    "threshold_ramp",
    "stairs",
    "display_fixture",
    "column",
    "product",
)


class Scheme:
    """One annotation taxonomy: its classes, its policy table and its native id map.

    Retail annotation needs the 13-class list -- a store's annotators drawing the indoor
    12 would produce data with no `display_fixture` in it at all, and nothing downstream
    would say so: the masks would validate, train, and silently teach the model that
    shelving is furniture.
    """

    def __init__(self, name, terrain, trav, native):
        self.name = name
        self.terrain = terrain
        self.trav = trav
        self.native = native
        self.id_to_name = {v: k for k, v in terrain.items()}


SCHEMES = {
    "indoor": Scheme("indoor", INDOOR_TERRAIN, INDOOR_TERRAIN_TO_TRAV, INDOOR_NATIVE_ID),
    "retail": Scheme("retail", RETAIL_TERRAIN, RETAIL_TERRAIN_TO_TRAV, RETAIL_NATIVE_ID),
    # The object taxonomy: seven classes, for "what is this" rather than "can the robot
    # step here". An annotator drawing a shop under `retail` produces no `column` and no
    # `product` at all -- the same failure the class docstring above describes for
    # indoor-vs-retail, one taxonomy further on.
    "retail_objects": Scheme(
        "retail_objects", RETAIL_OBJECTS, RETAIL_OBJECTS_TO_TRAV, RETAIL_OBJECTS_NATIVE_ID
    ),
}
DEFAULT_SCHEME = "indoor"


# --------------------------------------------------------------------- labels


def label_spec(scheme: Scheme | None = None) -> list[dict]:
    """The CVAT label list for one scheme.

    Colours come from the palette the inference overlays use, so a mask drawn in CVAT
    and a prediction rendered by ``hydranet-infer-video`` are the same colour for the
    same class. Reviewing a model against its labels is hard enough without a
    translation step in the middle.

    ``void`` is omitted on purpose: it is what unlabelled pixels become, not something
    an annotator draws. See ``check`` for why that distinction has teeth.
    """
    scheme = scheme or SCHEMES[DEFAULT_SCHEME]
    # The palette for *this* scheme, so retail's display_fixture gets the colour the
    # renderers give it. It used to index a single 12-entry palette and fall back to a
    # separate EXTRA_COLORS tuple for anything past the end -- two sources for one
    # question, and the fallback's colour matched nothing that ever drew a prediction.
    palette = terrain_palette(list(scheme.terrain))
    spec = []
    for name, tid in scheme.terrain.items():
        if name == "void":
            continue
        r, g, b = (int(c) for c in palette[tid])
        trav = TRAV_NAMES.get(scheme.trav[tid], "?")
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
    scheme = SCHEMES[args.scheme]
    spec = label_spec(scheme)
    text = json.dumps(spec, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({len(spec)} labels, {scheme.name} scheme)")
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
    # `with`, and `asarray` *inside* it: PIL opens lazily, so the array has to be
    # materialised before the handle closes. The refusal path is the one that leaked --
    # a dataset of RGB masks is exactly the case that hits it on every file, and this
    # function is called once per mask across a whole split.
    with Image.open(path) as ann:
        if ann.mode not in ("L", "P", "I", "I;16"):
            return None
        return np.asarray(ann)


def check_split(root: Path, split: str, rep: Report, allow_void: bool, scheme: Scheme) -> dict:
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

    _report_unpaired(split, img_dir, ann_dir, pairs, rep)
    void_pixels = _scan_pairs(pairs, img_dir, scheme, rep, stats)
    _report_unlabelled(split, stats, void_pixels, scheme, allow_void, rep)
    return stats


def _report_unpaired(split: str, img_dir: Path, ann_dir: Path, pairs, rep: Report) -> None:
    """Anything unpaired is dropped silently by the loader.

    Which is how an annotation run "finishes" with a third of its work not being trained
    on, and why this is an error rather than a note.
    """
    paired_imgs = {p.resolve() for p, _ in pairs}
    paired_anns = {a.resolve() for _, a in pairs}
    for label, directory, paired in (
        ("image", img_dir, paired_imgs),
        ("annotation", ann_dir, paired_anns),
    ):
        lonely = [
            p
            for p in directory.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            and p.resolve() not in paired
        ]
        if lonely:
            shown = ", ".join(p.name for p in sorted(lonely)[:3])
            rep.error(
                f"{split}: {len(lonely)} {label}(s) with no counterpart, silently "
                f"dropped by the loader (e.g. {shown})"
            )


def _check_one_mask(img_path: Path, ann_path: Path, allowed: set, scheme: Scheme, rep: Report):
    """Read and validate one mask. Returns the array, or None if it cannot be counted.

    Each refusal is separate because each has a different fix, and a mask that fails any
    of them must not reach the counters -- a wrong shape counted anyway is worse than an
    uncounted one, because it makes the class shares below quietly wrong.
    """
    mask = _read_mask(ann_path)
    if mask is None:
        # `with`, because this is the error path of a checker that walks whole datasets:
        # a leaked handle per bad mask is a file-descriptor exhaustion on exactly the run
        # that has the most of them. PIL opens lazily, so `.mode` inside the block is
        # enough to read the header without decoding the image.
        with Image.open(ann_path) as im:
            mode = im.mode
        rep.error(
            f"{ann_path.name}: {mode} mode, not single channel. "
            "Export the mask as a single-channel PNG whose pixel value is the class id."
        )
        return None
    if mask.ndim != 2:
        rep.error(f"{ann_path.name}: {mask.ndim}-d annotation, expected 2-d")
        return None
    with Image.open(img_path) as im:
        if im.size != (mask.shape[1], mask.shape[0]):
            rep.error(
                f"{ann_path.name}: mask {mask.shape[1]}x{mask.shape[0]} but image "
                f"{im.size[0]}x{im.size[1]} -- the two are resized independently, so "
                "the labels would be geometrically wrong, not merely misaligned"
            )
            return None
    unknown = sorted(int(v) for v in np.unique(mask) if int(v) not in allowed)
    if unknown:
        rep.error(
            f"{ann_path.name}: ids {unknown} are not in the {scheme.name} scheme; the "
            "loader maps them to ignore, so those pixels train on nothing"
        )
    return mask


def _scan_pairs(pairs, img_dir: Path, scheme: Scheme, rep: Report, stats: dict) -> int:
    """Fold every valid pair into `stats`. Returns the void (id 0) pixel count."""
    allowed = set(scheme.native) | {IGNORE}
    void_pixels = 0
    for img_path, ann_path in pairs:
        mask = _check_one_mask(img_path, ann_path, allowed, scheme, rep)
        if mask is None:
            continue
        counts = np.bincount(mask.ravel(), minlength=256)
        for tid in np.unique(mask):
            tid = int(tid)
            if tid in allowed:
                stats["pixels"][tid] += int(counts[tid])
                stats["images"][tid] += 1
        void_pixels += int(counts[0])
        stats["sessions"][_session_of(img_path, img_dir)] += 1
    return void_pixels


def _report_unlabelled(
    split: str, stats: dict, void_pixels: int, scheme: Scheme, allow_void: bool, rep: Report
) -> None:
    """The two ways a split can be mostly unlabelled without looking like it."""
    total_px = sum(stats["pixels"].values())
    if void_pixels and scheme.native.get(0) == 0 and not allow_void:
        share = 100 * void_pixels / max(total_px, 1)
        rep.error(
            f"{split}: {share:.1f}% of pixels are id 0 (void), and the indoor_native map "
            "sends 0 -> 0, so the loss trains them as a class rather than ignoring them. "
            "Most tools export unlabelled background as 0: remap it to 255 on export, or "
            "pass --allow-void if this dataset really does mean void"
        )

    # A forgotten background and a deliberately blank one are the same pixels, so this
    # cannot be an error. It can be said out loud, which is enough: nobody sets out to
    # ship a batch that is four-fifths ignore, and the per-class shares reported later
    # are shares of what *was* labelled, so they stay reassuring while it happens.
    ignored_px = sum(stats["pixels"][tid] for tid in _ignored_ids(scheme))
    if total_px and ignored_px / total_px > 0.6:
        rep.warn(
            f"{split}: {100 * ignored_px / total_px:.0f}% of pixels are ignore. If that is "
            "deliberate -- only the rare classes drawn, everything else left blank -- it is "
            "fine and the loss simply skips them. If it is not, most of this batch will "
            "train on nothing"
        )


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


def _ignored_ids(scheme: Scheme) -> set[int]:
    """The ids the loss never sees, whatever the label map calls them.

    255 always, plus anything ``indoor_native`` sends to 255 -- id 0 does, now that
    unlabelled background is mapped to ignore rather than to a `void` class. Counting
    those pixels in a coverage percentage would quietly shrink every class in the table
    in proportion to how much of the frame nobody labelled.
    """
    return {IGNORE} | {src for src, dst in scheme.native.items() if dst == IGNORE}


def _print_coverage(per_split: dict[str, dict], scheme: Scheme) -> None:
    splits = list(per_split)
    ignored = _ignored_ids(scheme)
    total_px = {
        s: max(sum(px for tid, px in per_split[s]["pixels"].items() if tid not in ignored), 1)
        for s in splits
    }

    print("\nterrain coverage (% of labelled pixels / images containing the class)")
    header = "".join(f"{s:>22}" for s in splits)
    print(f"{'id':>3}  {'class':<20}{header}")
    for tid, name in sorted(scheme.id_to_name.items()):
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
                per_split[s]["pixels"][tid] for tid, t in scheme.trav.items() if t == trav_id
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

    scheme = SCHEMES[args.scheme]
    splits = args.split or sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
    rep = Report()
    per_split = {s: check_split(root, s, rep, args.allow_void, scheme) for s in splits}
    check_sessions(per_split, rep)

    print(f"{root}  scheme: {scheme.name}  splits: {', '.join(splits)}")
    for s in splits:
        n_sessions = len([x for x in per_split[s]["sessions"] if x != "(flat)"])
        print(f"  {s:<8} {per_split[s]['n_pairs']:>6} frames  {n_sessions:>3} sessions")
    _print_coverage(per_split, scheme)

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


def _add_scheme(parser) -> None:
    parser.add_argument(
        "--scheme",
        choices=sorted(SCHEMES),
        default=DEFAULT_SCHEME,
        help="which taxonomy to use. retail is the indoor 12 plus display_fixture, for "
        "'can the robot step here'; retail_objects is the 7-class object taxonomy, for "
        "'what is this'. Annotating a shop under the wrong one yields data missing a "
        "whole class -- fixtures under indoor, columns and products under retail -- and "
        "nothing downstream would report that",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-annotation", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_labels = sub.add_parser("labels", help="emit the CVAT label schema for the 12 classes")
    p_labels.add_argument("--out", help="write JSON here instead of stdout")
    _add_scheme(p_labels)
    p_labels.set_defaults(func=cmd_labels)

    p_check = sub.add_parser("check", help="verify an annotated dataset against the contract")
    p_check.add_argument("root", help="dataset root, containing images/ and annotations/")
    p_check.add_argument("--split", action="append", help="check only this split (repeatable)")
    _add_scheme(p_check)
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
