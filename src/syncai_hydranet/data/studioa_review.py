"""Prepare source-bound annotation work, never manufacture a human acceptance set."""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .store_split import STORES, camera_store, fold_split, validate_fold
from .studioa_contract import (
    ENTITY_IDS,
    SOURCE_CLASSES,
    VERSION,
    VIEWS,
    contract,
    contract_sha256,
    entity_views,
    migrate_mask,
)

REVIEW_SCHEMA = "studioa.review.v1"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    """Publish complete JSON in one replacement; readers never see a partial write."""
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def index_source(root: Path, scheme: str) -> list[dict]:
    """Use exact relative image/mask paths; reject missing or ambiguous supervision."""
    if scheme not in SOURCE_CLASSES:
        raise ValueError(f"unsupported source taxonomy: {scheme}")
    rows = []
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        mask_dir = root / "annotations" / split
        masks = {}
        for path in sorted(mask_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                key = path.relative_to(mask_dir).with_suffix("")
                if key in masks:
                    raise ValueError(f"ambiguous mask path: {key}")
                masks[key] = path
        for image in sorted(image_dir.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
                continue
            relative = image.relative_to(image_dir)
            mask = masks.get(relative.with_suffix(""))
            if mask is None:
                raise ValueError(f"missing paired mask: {image}")
            camera, store = camera_store(str(relative))
            rows.append(
                {
                    "source_root": str(root.resolve()),
                    "source_scheme": scheme,
                    "source_image": str(image.absolute()),
                    "source_mask": str(mask.absolute()),
                    "camera": camera,
                    "store": store,
                    "original_split": split,
                    "session": str(relative.parent)
                    if len(relative.parts) > 1
                    else image.stem.rsplit("__", 1)[0],
                }
            )
    if not rows:
        raise ValueError(f"no paired split images: {root}")
    return rows


def select_frames(rows: list[dict], per_camera: int) -> list[dict]:
    """Spread sessions, then select each session's middle frame; no score-based selection."""
    if per_camera < 1:
        raise ValueError("per_camera must be positive")
    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row["source_root"], row["camera"])][row["session"]].append(row)
    selected = []
    for key in sorted(groups):
        sessions = groups[key]
        names = sorted(sessions)
        n = min(per_camera, len(names))
        indices = (
            [len(names) // 2] if n == 1 else [i * (len(names) - 1) // (n - 1) for i in range(n)]
        )
        for index in indices:
            frames = sorted(sessions[names[index]], key=lambda row: row["source_image"])
            selected.append(frames[len(frames) // 2])
    return selected


def review_template(frame: dict) -> dict:
    return {
        "schema": REVIEW_SCHEMA,
        "contract_version": VERSION,
        "contract_sha256": contract_sha256(),
        "frame_id": frame["id"],
        "image_sha256": frame["files"][frame["image"]],
        "image_size_px": frame["image_size_px"],
        "pixel_space": "source_image",
        "scope": "visible_pixels_only",
        "status": "pending",
        "reviewer": None,
        "reviewed_at": None,
        "coverage": dict.fromkeys(VIEWS, "unreviewed"),
        "entities": [],
        "unresolved_regions": [],
        "ignore_regions": [],
        "notes": "Legacy proposals only; unlabelled pixels are not negatives.",
    }


def _polygon(points, size: list[int]) -> None:
    array = np.asarray(points, dtype=float)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or len(array) < 3
        or not np.isfinite(array).all()
        or (array < 0).any()
        or (array > np.asarray(size) - 1).any()
    ):
        raise ValueError("polygon needs finite source-image pixel coordinates")
    x, y = array.T
    if abs(float(x @ np.roll(y, 1) - y @ np.roll(x, 1))) < 1e-6:
        raise ValueError("polygon has no area")


def validate_review(review: dict, frame: dict) -> dict[str, int]:
    """Validate a review's provenance and completeness, not the truth of human judgement."""
    expected = review_template(frame)
    for key in (
        "schema",
        "contract_version",
        "contract_sha256",
        "frame_id",
        "image_sha256",
        "image_size_px",
        "pixel_space",
        "scope",
    ):
        if review.get(key) != expected[key]:
            raise ValueError(f"review source/contract mismatch: {key}")
    if review.get("status") != "reviewed":
        raise ValueError("human review is pending")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("reviewer is required")
    try:
        reviewed_at = datetime.fromisoformat(review["reviewed_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reviewed_at requires an ISO timestamp with timezone") from exc
    if reviewed_at.tzinfo is None:
        raise ValueError("reviewed_at requires a timezone")
    coverage = review.get("coverage", {})
    if set(coverage) != set(VIEWS) or any(
        state not in {"present", "absent", "unobservable", "unresolved"}
        for state in coverage.values()
    ):
        raise ValueError("every requested view needs explicit reviewed coverage")
    counts = dict.fromkeys(VIEWS, 0)
    ids = set()
    for entity in review.get("entities", []):
        identity = entity.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError("entities require unique nonempty instance IDs")
        ids.add(identity)
        polygons = entity.get("polygons_px")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError("entity requires visible polygons_px")
        for polygon in polygons:
            _polygon(polygon, frame["image_size_px"])
        if entity.get("visibility") not in {"visible", "occluded", "truncated"}:
            raise ValueError("entity visibility is required")
        for view in entity_views(entity):
            counts[view] += 1
        uncertain_view = None
        if entity["entity"] == "door" and entity.get("glazed") is None:
            uncertain_view = "glass_door"
        if entity["entity"] == "counter" and entity.get("role", "unknown") == "unknown":
            uncertain_view = "checkout_counter"
        if uncertain_view and coverage[uncertain_view] == "absent":
            raise ValueError(f"unknown attribute cannot establish absence: {uncertain_view}")
    for view, state in coverage.items():
        if (state == "present") != (counts[view] > 0):
            raise ValueError(f"coverage and visible entities disagree: {view}")
    for region in review.get("unresolved_regions", []):
        _polygon(region.get("polygon_px"), frame["image_size_px"])
        reason = region.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("unresolved region requires a reason")
        possible = set(region.get("possible_entities", []))
        if not possible <= set(ENTITY_IDS):
            raise ValueError("unknown unresolved entity candidate")
        for view, rule in VIEWS.items():
            if (not possible or rule["entity"] in possible) and coverage[view] == "absent":
                raise ValueError(f"unresolved region cannot establish absence: {view}")
    for region in review.get("ignore_regions", []):
        _polygon(region.get("polygon_px"), frame["image_size_px"])
        reason = region.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("ignore region requires a reason")
    return counts


def _package_path(out: Path, name: str) -> Path:
    path = (out / name).resolve()
    if not path.is_relative_to(out.resolve()):
        raise ValueError("package path escapes review directory")
    return path


def check_package(out: Path) -> dict:
    """Check frozen inputs even before review; incomplete reviews cannot become gold labels."""
    manifest = json.loads((out / "manifest.json").read_text())
    if (
        manifest.get("schema") != REVIEW_SCHEMA
        or manifest.get("contract_sha256") != contract_sha256()
        or json.loads((out / "contract.json").read_text()) != contract()
    ):
        raise ValueError("review package contract mismatch")
    counts = dict.fromkeys(VIEWS, 0)
    reviewed = 0
    seen_ids = set()
    coverage = {view: Counter() for view in VIEWS}
    for frame in manifest["frames"]:
        if frame["id"] in seen_ids:
            raise ValueError("duplicate review frame ID")
        seen_ids.add(frame["id"])
        for name, expected in frame["files"].items():
            if digest(_package_path(out, name)) != expected:
                raise ValueError(f"changed frozen input: {name}")
        with Image.open(_package_path(out, frame["image"])) as image:
            if list(image.size) != frame["image_size_px"]:
                raise ValueError("source image dimensions changed")
        review = json.loads(_package_path(out, frame["review"]).read_text())
        for key, value in review_template(frame).items():
            if (
                key
                in {
                    "schema",
                    "contract_version",
                    "contract_sha256",
                    "frame_id",
                    "image_sha256",
                    "image_size_px",
                    "pixel_space",
                    "scope",
                }
                and review.get(key) != value
            ):
                raise ValueError(f"review source/contract mismatch: {key}")
        if review.get("status") == "pending":
            continue
        result = validate_review(review, frame)
        reviewed += 1
        for view, count in result.items():
            counts[view] += count
            coverage[view][review["coverage"][view]] += 1
    return {
        "status": "review_complete"
        if reviewed == len(seen_ids) and seen_ids
        else "pending_review",
        "frames": len(seen_ids),
        "reviewed_frames": reviewed,
        "reviewed_instances_by_view": counts,
        "reviewed_coverage": {view: dict(states) for view, states in coverage.items()},
        "acceptance": "not established; requires independent split, coverage and metric gates",
    }


def prepare_package(sources: list[tuple[Path, str]], out: Path, per_camera: int = 3) -> dict:
    """Freeze candidate images and masks, produce empty human tasks and prospective folds."""
    roots = [str(path.resolve()) for path, _ in sources]
    if len(roots) != len(set(roots)):
        raise ValueError("duplicate dataset source")
    rows = [row for path, scheme in sources for row in index_source(path, scheme)]
    selected = select_frames(rows, per_camera)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "contract.json", contract())
    frames: list[dict] = []
    for index, row in enumerate(selected):
        identity = f"{index:04d}-{row['camera']}"
        folder = out / "frames" / identity
        folder.mkdir(parents=True)
        source_image, source_mask = Path(row["source_image"]), Path(row["source_mask"])
        image_path, mask_path = (
            folder / ("image" + source_image.suffix),
            folder / "source_mask.png",
        )
        original_hashes = {"image": digest(source_image), "mask": digest(source_mask)}
        shutil.copyfile(source_image, image_path)
        shutil.copyfile(source_mask, mask_path)
        if (
            digest(image_path) != original_hashes["image"]
            or digest(mask_path) != original_hashes["mask"]
        ):
            raise ValueError("source changed during copy")
        with Image.open(image_path) as image:
            size = list(image.size)
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(mask_path) as image:
            raw = np.asarray(image)
        if raw.shape != tuple(size[::-1]):
            raise ValueError("source image and mask have different dimensions")
        proposal, report = migrate_mask(raw, row["source_scheme"])
        proposal_path = folder / "proposal_entity_ids.png"
        Image.fromarray(proposal).save(proposal_path)
        frame = {
            **row,
            "id": identity,
            "image_size_px": size,
            "image_sha256": hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest(),
            "image": str(image_path.relative_to(out)),
            "proposal": str(proposal_path.relative_to(out)),
            "review": str((folder / "review.json").relative_to(out)),
            "files": {
                str(path.relative_to(out)): digest(path)
                for path in (image_path, mask_path, proposal_path)
            },
            "migration": report,
            "exposure": "legacy data; previous training/tuning exposure possible",
        }
        write_json(folder / "review.json", review_template(frame))
        frames.append(frame)
    inventory = Counter(
        (row["source_root"], row["camera"], row["original_split"]) for row in rows
    )
    folds = {}
    for store in STORES:
        try:
            counts = validate_fold(frames, store)
            status, reason = "structurally_valid", None
        except ValueError as exc:
            counts, status, reason = None, "blocked", str(exc)
        folds[store] = {
            "status": status,
            "reason": reason,
            "counts": counts,
            "assignments": {
                frame["id"]: fold_split(frame["store"], frame["original_split"], store)
                for frame in frames
            },
            "scope": "selected frames only; prospective, not blind for existing models",
        }
    manifest = {
        "schema": REVIEW_SCHEMA,
        "contract_sha256": contract_sha256(),
        "purpose": "annotation bootstrap; no training or independent acceptance export",
        "selection": "per source/camera: spread sessions, middle frames; no model scores",
        "requested_frames_per_source_camera": per_camera,
        "indexed_pairs": len(rows),
        "inventory": [
            {"source_root": root, "camera": camera, "original_split": split, "pairs": n}
            for (root, camera, split), n in sorted(inventory.items())
        ],
        "frames": frames,
        "prospective_folds": folds,
        "producer_files": {
            Path(__file__).name: digest(Path(__file__)),
            "studioa_contract.py": digest(Path(__file__).with_name("studioa_contract.py")),
        },
    }
    write_json(out / "manifest.json", manifest)
    summary = check_package(out)
    write_json(out / "review_status.json", summary)
    cards = []
    for frame in frames:
        camera, split = html.escape(frame["camera"]), html.escape(frame["original_split"])
        image_url = html.escape(frame["image"], quote=True)
        review_url = html.escape(frame["review"], quote=True)
        cards.append(
            f'<article><h2>{camera} / {split}</h2><img loading="lazy" src="{image_url}">'
            f'<p><a href="{review_url}">人工標註 JSON</a> — 初始狀態: 待審核</p></article>'
        )
    (out / "index.html").write_text(
        '<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
        "<title>StudioA 標註準備</title>"
        "<style>body{font-family:sans-serif;margin:2rem}main{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(min(400px,100%),1fr));gap:1rem}"
        "img{width:100%}article{border:1px solid #ddd;padding:1rem}</style>"
        "<h1>StudioA 分類標註準備</h1><p>這是既有資料的待標註包, 不是模型成果或盲測真值。"
        "原標籤與遷移遮罩僅供另行參考; 請依原圖標註。完成後執行 check 更新審核狀態。</p>"
        "<main>" + "".join(cards) + "</main></html>\n"
    )
    return {
        "indexed_pairs": len(rows),
        **summary,
        "folds": {store: fold["status"] for store, fold in folds.items()},
    }
