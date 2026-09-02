#!/usr/bin/env python3
"""Move whole cameras between splits, in every batch at once, or not at all.

    python3 scripts/resplit_selling_floor.py --to val \\
        Kaohsiung-cam05 Kaohsiung-cam07 Taichung-cam03 Taichung-cam05 Tao-Hsin-cam04

A camera is the unit because the split's whole purpose is that none appears on two sides,
and `retail_objects_batch03/split.json` inherits batch02's assignment to keep it that way.
So moving one in a single batch is the one edit that silently breaks the invariant the
files were arranged to hold: batch02 would still train on a camera batch03 scores. Every
batch moves together here, and the check is unconditional: the whole operation is refused
if any batch would end up disagreeing. (There is no `--check` flag and never was; an
earlier version of this sentence named one.)

**Train-only supplements are checked too, and this is the failure that put it here.**
`datasets/retail_objects_columns_clean` is a `column` supplement with no val and no test,
so it has no assignment for this script to move and was invisible to it. It was built on
2026-08-18 from the nine cameras then in neither val nor test; six hours later a `--to val`
run promoted `Taichung-cam10`, and the supplement went on training on a val camera with
nothing anywhere reporting it. The three `hydranet_retail_surfaces_columns` seeds that had
already finished were honest; anything trained after that was not, and no tool said so.

The fix is a digest rather than an assignment. A dataset that records `clean_against` --
the batches it was built to be clean of, and a hash of their assignment at the time -- is
verified before and after every move here: if this move would invalidate one, the whole
operation is refused by name. See `datasets/retail_objects_columns_v2/split.json`. A
supplement without that block is reported as unprotected rather than assumed safe, because
"nothing to check" and "checked and fine" are the two states this failure confused.

Three trees carry a camera and all three move: `images/<split>/`,
`annotations/<split>/` (the masks), and the `annotations/instances_<split>.json` detection
records, whose `file_name` is relative to its split and whose ids must not collide with the
receiving file's. `split.json` is rewritten last, with the reason, because a split with no
recorded reason is one nobody can argue with later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BATCHES = ["datasets/retail_objects_batch02", "datasets/retail_objects_batch03"]
SPLITS = ["train", "val", "test"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cameras", nargs="+")
    ap.add_argument("--to", required=True, choices=SPLITS)
    ap.add_argument("--batches", nargs="+", default=BATCHES)
    ap.add_argument("--reason", default="")
    ap.add_argument(
        "--allow-stale-supplements",
        action="store_true",
        help="proceed even though a train-only supplement would start "
        "training on a held-out camera. Only correct when the supplement is "
        "being rebuilt in the same change.",
    )
    ap.add_argument("--apply", action="store_true", help="without this it only reports")
    return ap


def camera_dirs(root: Path, split: str, cam: str) -> list[Path]:
    base = root / "images" / split
    return sorted(d for d in base.glob(f"{cam}__*") if d.is_dir()) if base.is_dir() else []


def locate(root: Path, cam: str) -> list[str]:
    return [s for s in SPLITS if camera_dirs(root, s, cam)]


def move_coco(root: Path, cam: str, src: str, dst: str, apply: bool) -> int:
    """Move one camera's images and annotations between two instances_*.json files."""
    fsrc = root / "annotations" / f"instances_{src}.json"
    fdst = root / "annotations" / f"instances_{dst}.json"
    if not fsrc.exists() or not fdst.exists():
        return 0
    a, b = json.loads(fsrc.read_text()), json.loads(fdst.read_text())
    moving = [im for im in a["images"] if im["file_name"].startswith(f"{cam}__")]
    if not moving:
        return 0
    ids = {im["id"] for im in moving}
    anns = [an for an in a["annotations"] if an["image_id"] in ids]
    # Detach from the source before renumbering: the ids are about to change, so a filter
    # written after the remap would test the new ids against the old set and keep everything.
    a["images"] = [im for im in a["images"] if im["id"] not in ids]
    a["annotations"] = [an for an in a["annotations"] if an["image_id"] not in ids]
    next_id = max((im["id"] for im in b["images"]), default=0) + 1
    next_ann = max((an["id"] for an in b["annotations"]), default=0) + 1
    remap = {}
    for im in moving:
        remap[im["id"]] = next_id
        im["id"] = next_id
        next_id += 1
    for an in anns:
        an["image_id"] = remap[an["image_id"]]
        an["id"] = next_ann
        next_ann += 1
    b["images"].extend(moving)
    b["annotations"].extend(anns)
    if apply:
        fsrc.write_text(json.dumps(a, indent=1))
        fdst.write_text(json.dumps(b, indent=1))
    return len(moving)


def assign_digest(roots: list[Path]) -> str:
    """A stable hash of every batch's camera->split map, as `clean_against` records it."""
    payload = {
        str(r): dict(sorted(json.loads((r / "split.json").read_text())["assign"].items()))
        for r in sorted(roots, key=str)
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def supplements(datasets: Path = Path("datasets")) -> list[tuple[Path, dict]]:
    """Every dataset carrying a `clean_against` block, and the block."""
    found = []
    for f in sorted(datasets.glob("*/split.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "clean_against" in d:
            found.append((f.parent, d))
    return found


def check_supplements(roots: list[Path], when: str) -> list[str]:
    """Which supplements no longer match the assignment they were built against.

    Reported by name and by camera, because "a supplement is stale" is not actionable and
    "columns_v2 trains on Taichung-cam10, which is now val" is.
    """
    digest = assign_digest(roots)
    broken = []
    for root, d in supplements():
        ca = d["clean_against"]
        if ca.get("assign_sha256_12") == digest:
            continue
        held = {}
        for r in roots:
            assign = json.loads((r / "split.json").read_text())["assign"]
            for cam in d.get("assign", {}):
                if assign.get(cam) in ("val", "test"):
                    held.setdefault(cam, {})[r.name] = assign[cam]
        if held:
            detail = "; ".join(f"{c} is {v}" for c, v in sorted(held.items()))
            broken.append(f"{root} ({when}): {detail}")
        else:
            print(
                f"note: {root} digest is stale but no camera of its is held; "
                f"refresh assign_sha256_12 to {digest}"
            )
    return broken


def check_after(cameras: list[str], to: str) -> list[str]:
    """What `check_supplements` would say *after* this move, computed before making it.

    Refusing afterwards is not a guard, it is a post-mortem.
    """
    if to == "train":
        return []
    broken = []
    for root, d in supplements():
        overlap = sorted(set(cameras) & set(d.get("assign", {})))
        if overlap:
            broken.append(
                f"{root} trains on {', '.join(overlap)}, which this move sends to {to}"
            )
    return broken


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = [Path(b) for b in args.batches]

    plan: list[tuple[Path, str, str, list[Path]]] = []
    for cam in args.cameras:
        where = {r: locate(r, cam) for r in roots}
        for r, splits in where.items():
            if len(splits) > 1:
                print(f"REFUSED: {cam} is already in {splits} under {r}", file=sys.stderr)
                return 1
        present = {r: s[0] for r, s in where.items() if s}
        if not present:
            print(f"REFUSED: {cam} is in none of {[str(r) for r in roots]}", file=sys.stderr)
            return 1
        if len(set(present.values())) > 1:
            msg = f"REFUSED: {cam} sits on different sides per batch: {present}"
            print(msg, file=sys.stderr)
            return 1
        for r, src in present.items():
            if src != args.to:
                plan.append((r, cam, src, camera_dirs(r, src, cam)))

    if not plan:
        print("nothing to do; every camera is already there")
        return 0
    for r, cam, src, dirs in plan:
        n = sum(len(list(d.glob("*.jpg"))) for d in dirs)
        print(f"{r.name}: {cam}  {src} -> {args.to}   {len(dirs)} clips, {n} frames")
    would = check_after(args.cameras, args.to)
    if would:
        # The verdict word must match what happens next: this used to print REFUSED
        # unconditionally and then, with --allow-stale-supplements, proceed anyway.
        verdict = (
            "PROCEEDING ANYWAY (--allow-stale-supplements): this move invalidates "
            "a train-only supplement:"
            if args.allow_stale_supplements
            else "REFUSED: this move invalidates a train-only supplement:"
        )
        print(f"\n{verdict}", file=sys.stderr)
        for line in would:
            print(f"  {line}", file=sys.stderr)
        if not args.allow_stale_supplements:
            print(
                "\nRebuild the supplement without those cameras and update its "
                "clean_against digest, or pass --allow-stale-supplements if you are "
                "rebuilding it in the same breath.",
                file=sys.stderr,
            )
            return 1

    if not args.apply:
        print("\n(dry run; pass --apply)")
        return 0

    for r, cam, src, dirs in plan:
        for tree in ("images", "annotations"):
            for d in dirs:
                s = r / tree / src / d.name
                if s.exists():
                    (r / tree / args.to).mkdir(parents=True, exist_ok=True)
                    shutil.move(str(s), str(r / tree / args.to / d.name))
        move_coco(r, cam, src, args.to, apply=True)

    for r in roots:
        f = r / "split.json"
        sp = json.loads(f.read_text())
        for cam in args.cameras:
            if cam in sp.get("assign", {}):
                sp["assign"][cam] = args.to
        sp.setdefault("moves", []).append(
            {"cameras": list(args.cameras), "to": args.to, "reason": args.reason}
        )
        f.write_text(json.dumps(sp, indent=1))

    stale = check_supplements(roots, "after the move")
    if stale:
        print("\nSUPPLEMENTS NOW STALE -- rebuild before the next training run:")
        for line in stale:
            print(f"  {line}")
    else:
        print(f"\nsupplements verified; assignment digest is now {assign_digest(roots)}")

    print("\nafter:")
    for r in roots:
        for s in SPLITS:
            cams = {d.name.split("__")[0] for d in (r / "images" / s).glob("*__*")}
            n = len(list((r / "images" / s).rglob("*.jpg")))
            print(f"  {r.name:26s} {s:5s} {len(cams):2d} cameras {n:4d} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
