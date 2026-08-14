"""Identify the data a run was trained on.

Datasets live outside git, so "which data produced this checkpoint" is otherwise
unanswerable: annotations get re-exported, a filter threshold changes, someone adds a
hundred site photos, and two runs that look identical in the config are not comparable.

A fingerprint is taken per split and stored in the run's meta.json, so a difference
between two runs is visible without needing the data to still exist:

    {"files": 5998, "bytes": 812374915, "digest": "sha256:1f3c..."}

The digest covers each file's path and size, not its contents -- hashing a gigabyte at
the start of every run is not worth it. That catches added, removed, renamed and
resized files, which is how datasets actually change. It will not catch an edit that
leaves a file exactly the same size, so it is a provenance record, not a security
control.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

# Cheap insurance against pointing the fingerprint at a home directory by accident.
MAX_FILES = 500_000


def fingerprint_dir(root: str | Path, suffixes: set[str] | None = None) -> dict[str, Any]:
    """Summarise a directory tree as file count, total size and a stable digest."""
    root = Path(root)
    if not root.is_dir():
        return {"present": False, "path": str(root)}

    entries: list[tuple[str, int]] = []
    total = 0
    truncated = False
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        if len(entries) >= MAX_FILES:
            truncated = True
            break
        size = path.stat().st_size
        entries.append((path.relative_to(root).as_posix(), size))
        total += size

    h = hashlib.sha256()
    for name, size in entries:
        h.update(f"{name}\0{size}\n".encode())
    out: dict[str, Any] = {
        "present": True,
        "path": str(root),
        "files": len(entries),
        "bytes": total,
        "digest": f"sha256:{h.hexdigest()[:32]}",
    }
    if truncated:
        out["truncated_at"] = MAX_FILES
    return out


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """Path and size of a single file, for COCO's one-big-json layout."""
    path = Path(path)
    if not path.is_file():
        return {"present": False, "path": str(path)}
    return {"present": True, "path": str(path), "bytes": path.stat().st_size}


def _first_present(candidates: list[dict]) -> dict | None:
    return next((c for c in candidates if c.get("present")), None)


def fingerprint_dataset(dcfg: dict, splits: tuple[str, ...] = ("train", "val")) -> dict:
    """Fingerprint the splits of one dataset config entry.

    Handles both layouts: seg_folder keeps ``images/<split>`` beside
    ``annotations/<split>``, while COCO puts images in ``<split>/`` and all labels in a
    single ``annotations/instances_<split>.json``.
    """
    root = Path(dcfg["root"])
    out: dict[str, Any] = {"name": dcfg.get("name"), "root": str(root), "splits": {}}
    for split in splits:
        folder = dcfg.get(f"split_{split}")
        if folder is None:
            continue
        images = _first_present(
            [fingerprint_dir(root / "images" / folder), fingerprint_dir(root / folder)]
        )
        labels = _first_present(
            [
                fingerprint_dir(root / "annotations" / folder),
                fingerprint_file(root / "annotations" / f"instances_{folder}.json"),
            ]
        )
        entry = {}
        if images:
            entry["images"] = images
        if labels:
            entry["labels"] = labels
        if entry:
            out["splits"][split] = entry
    return out
