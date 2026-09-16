#!/usr/bin/env python3
"""Freeze/resume a bounded SAM3 AI annotation job; no human annotation is required."""

from __future__ import annotations

import argparse
import fcntl
import html
import importlib.metadata
import json
import os
import shutil
import signal
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d.teachers import sam3
from syncai_hydranet.data.studioa_autolabel import (
    MIN_SCORE,
    PROMPTS,
    Candidate,
    annotate,
    apply_ai_review,
    coco_annotations,
    decode,
    policy,
    validate_annotation,
)
from syncai_hydranet.data.studioa_contract import ENTITY_IDS, contract
from syncai_hydranet.data.studioa_relabel import (
    EXTRA_PROMPTS,
    FIXTURE_PROMPT,
    LocalReviewer,
    ReviewCache,
    apply_visual_decision,
    candidate_groups,
    constrain_decision,
    reannotate,
    relabel_policy,
    review_panel,
    revise_group_decisions,
    validate_visual_decisions,
)
from syncai_hydranet.data.studioa_relabel import PROMPT as REVIEW_PROMPT
from syncai_hydranet.data.studioa_review import check_package, digest

ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare(bundle: Path, out: Path, limit: int | None) -> None:
    check_package(bundle)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    source = json.loads((bundle / "manifest.json").read_text())
    frames = source["frames"][:limit]
    freeze_frames(bundle, out, frames)


def prepare_media(source: Path, selection: Path, out: Path) -> None:
    """AI-selected raw media needs no fake semantic masks or human acceptance labels."""
    report = json.loads((source / "report.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    review = json.loads(selection.read_text())
    if report["status"] != "completed" or manifest["schema"] != "studioa.gcs-intake.v1":
        raise ValueError("requires completed GCS intake")
    for name, expected in report["outputs"].items():
        if digest(source / name) != expected:
            raise ValueError("GCS intake file changed")
    if review.get("reviewer_kind") != "ai" or review.get("source_manifest_sha256") != digest(
        source / "manifest.json"
    ):
        raise ValueError("media selection requires bound AI review")
    selected, seen = [], set()
    candidates = {frame["id"]: frame for frame in manifest["frames"]}
    for decision in review["frames"]:
        frame = candidates[decision["id"]]
        if (
            frame["id"] in seen
            or frame["original_split"] != "train"
            or not decision.get("camera_match_confirmed")
            or not decision.get("reason")
        ):
            raise ValueError("selection needs unique train frames and camera confirmation")
        seen.add(frame["id"])
        selected.append({**frame, "files": {frame["image"]: frame["image_sha256"]}})
    if not selected:
        raise ValueError("empty AI media selection")
    freeze_frames(
        source,
        out,
        selected,
        {"media_selection_sha256": digest(selection), "media_selection": review},
    )


def freeze_frames(
    bundle: Path, out: Path, frames: list[dict], extra: dict | None = None
) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    (out / "frames").mkdir()
    snapshot = out / "snapshot"
    shutil.copytree(
        ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (snapshot / "tools/annotation").mkdir(parents=True)
    shutil.copyfile(__file__, snapshot / "tools/annotation/studioa_autolabel.py")
    selected = []
    for frame in frames:
        original = bundle / frame["image"]
        dest = out / "images" / (frame["id"] + original.suffix)
        shutil.copyfile(original, dest)
        if digest(dest) != frame["files"][frame["image"]]:
            raise ValueError("source image changed during freeze")
        selected.append(
            {
                "id": frame["id"],
                "image": str(dest.relative_to(out)),
                "image_sha256": digest(dest),
                "image_size_px": frame["image_size_px"],
                "camera": frame["camera"],
                "store": frame["store"],
                "original_split": frame["original_split"],
            }
        )
    write(
        out / "job.json",
        {
            "schema": "studioa.ai.job.v1",
            "source_bundle": str(bundle.resolve()),
            "source_manifest_sha256": digest(bundle / "manifest.json"),
            "teacher": {"model": sam3.MODEL_ID, "revision": sam3.MODEL_REVISION},
            "policy": policy(),
            "contract": contract(),
            "frames": selected,
            **(extra or {}),
            "code": {
                str(p.relative_to(out)): digest(p) for p in sorted(snapshot.rglob("*.py"))
            },
        },
    )
    write(out / "status.json", {"status": "prepared", "completed": 0, "total": len(selected)})
    print(f"Prepared {len(selected)} images at {out}")


def overlay(image: Image.Image, annotation: dict, dest: Path) -> None:
    rgb = np.asarray(image).copy()
    for row in annotation["entities"]:
        mask = decode(row["segmentation"])
        cid = ENTITY_IDS[row["entity"]]
        color = np.array(
            [(cid * 67) % 205 + 50, (cid * 103) % 205 + 50, (cid * 149) % 205 + 50]
        )
        rgb[mask] = (0.65 * rgb[mask] + 0.35 * color).astype(np.uint8)
    preview = Image.fromarray(rgb)
    draw = ImageDraw.Draw(preview)
    for row in annotation["entities"]:
        x, y, w, h = row["bbox_xywh"]
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline="yellow", width=2)
        draw.text((x, y), row["entity"], fill="yellow", stroke_fill="black", stroke_width=1)
    preview.thumbnail((1280, 720))
    preview.save(dest, quality=85)


SCENE_LAYERS = (
    ("Floor", ("floor",), (0, 255, 80)),
    ("Round / long tables", ("display_table",), (0, 200, 255)),
    ("Cabinets / upright racks", ("display_cabinet", "other_shelf"), (255, 0, 210)),
    ("Boxed stock", ("boxed_stock",), (255, 220, 0)),
)


def scene_overlay(rgb: np.ndarray, annotation: dict, include_uncertain: bool) -> Image.Image:
    """Combine requested display fixtures; hatching does not resolve their classes."""
    shape = rgb.shape[:2]
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    stripes = (xx + yy) % 16 < 7
    result = rgb.copy()
    for _label, entities, color in SCENE_LAYERS:
        positive = np.zeros(shape, dtype=bool)
        uncertain = np.zeros(shape, dtype=bool)
        for key, mask in (("entities", positive), ("unresolved_candidates", uncertain)):
            for row in annotation[key]:
                if row["entity"] in entities:
                    mask |= decode(row["segmentation"])
        if include_uncertain:
            hatched = uncertain & ~positive & stripes
            result[hatched] = (0.2 * rgb[hatched] + 0.8 * np.array(color)).astype(np.uint8)
        result[positive] = (0.35 * rgb[positive] + 0.65 * np.array(color)).astype(np.uint8)
    preview = Image.fromarray(result)
    preview.thumbnail((1280, 720))
    canvas = Image.new("RGB", (max(640, preview.width), preview.height + 78), "#171717")
    canvas.paste(preview, (0, 78))
    draw = ImageDraw.Draw(canvas)
    for index, (label, _entities, color) in enumerate(SCENE_LAYERS):
        x, y = 8 + (index % 2) * (canvas.width // 2), 6 + (index // 2) * 22
        draw.rectangle((x, y, x + 14, y + 14), fill=color)
        draw.text((x + 21, y), label, fill="white")
    draw.text(
        (8, 55), "Solid: retained AI labels | Hatched: unresolved candidates", fill="white"
    )
    return canvas


def focus_preview(source: Path, out: Path, combined: bool = False) -> None:
    """Render floor/table/cabinet masks without changing annotation decisions."""
    parent = json.loads((source / "report.json").read_text())
    job = json.loads((source / "job.json").read_text())
    if parent["status"] != "completed" or digest(source / "job.json") != parent["job_sha256"]:
        raise ValueError("preview requires a completed bound inference run")
    for name, expected in parent["outputs"].items():
        if digest(source / name) != expected:
            raise ValueError("source annotation changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    (out / "images").mkdir()
    cards = []
    for frame in job["frames"]:
        original = source / frame["image"]
        if digest(original) != frame["image_sha256"]:
            raise ValueError("source image changed")
        shutil.copyfile(original, out / frame["image"])
        data = json.loads((source / "frames" / (frame["id"] + ".json")).read_text())
        with Image.open(original) as image:
            rgb = np.asarray(image.convert("RGB"))
        if combined:
            name = html.escape(frame["id"])
            for mode in ("positive", "uncertain"):
                scene_overlay(rgb, data, mode == "uncertain").save(
                    out / "frames" / f"{frame['id']}-scene-{mode}.jpg", quality=92
                )
            cards.append(
                f'<article><h2>{name}</h2><a href="{html.escape(frame["image"])}">原圖</a>'
                + "".join(
                    f'<a class="{mode}" href="frames/{name}-scene-{mode}.jpg">'
                    f'<img loading="lazy" src="frames/{name}-scene-{mode}.jpg"></a>'
                    for mode in ("positive", "uncertain")
                )
                + "</article>"
            )
            continue
        shape = rgb.shape[:2]
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        stripes = (xx + yy) % 16 < 7
        panels = []
        for entity, label, color in (
            ("floor", "地板", (0, 255, 80)),
            ("display_table", "展示桌 (含圓桌、長桌)", (0, 200, 255)),
            ("display_cabinet", "展示櫃", (255, 0, 210)),
        ):
            positive = np.zeros(shape, dtype=bool)
            uncertain = np.zeros(shape, dtype=bool)
            counts = []
            for key, mask in (("entities", positive), ("unresolved_candidates", uncertain)):
                rows = [row for row in data[key] if row["entity"] == entity]
                counts.append(len(rows))
                for row in rows:
                    mask |= decode(row["segmentation"])
            result = rgb.copy()
            result[positive] = (0.35 * rgb[positive] + 0.65 * np.array(color)).astype(np.uint8)
            uncertain_only = uncertain & ~positive & stripes
            with_uncertain = result.copy()
            with_uncertain[uncertain_only] = (
                0.2 * rgb[uncertain_only] + 0.8 * np.array((255, 170, 0))
            ).astype(np.uint8)
            for mode, pixels in (("positive", result), ("uncertain", with_uncertain)):
                preview = Image.fromarray(pixels)
                preview.thumbnail((1280, 720))
                preview.save(out / "frames" / f"{frame['id']}-{entity}-{mode}.jpg", quality=92)
            name = html.escape(frame["id"])
            panels.append(
                f"<section><h3>{label}: 保留 {counts[0]} 筆 / 未決 {counts[1]} 筆</h3>"
                + "".join(
                    f'<img class="{mode}" loading="lazy" '
                    f'src="frames/{name}-{entity}-{mode}.jpg">'
                    for mode in ("positive", "uncertain")
                )
                + "</section>"
            )
        cards.append(
            f'<article><h2>{name}</h2><a href="{html.escape(frame["image"])}">原圖</a>'
            '<div class="panels">' + "".join(panels) + "</div></article>"
        )
    (out / "index.html").write_text(
        '<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>StudioA 展示設備遮罩檢查</title><style>"
        "body{font-family:sans-serif;margin:1rem;background:#171717;color:#eee}"
        "a{color:#8cd5ff}.panels{display:grid;grid-template-columns:"
        "repeat(auto-fit,minmax(min(550px,100%),1fr));gap:1rem}img{width:100%}"
        ".positive{display:none}body.clean .positive{display:block}"
        "body.clean .uncertain{display:none}article{border-top:1px solid #555}"
        "</style><body><h1>地板 / 展示桌 / 展示櫃與直立架 / 盒裝商品</h1>"
        "<p>圓桌、長桌屬於展示桌。本頁一起呈現展示桌與展示櫃。</p>"
        "<p>綠色 = 地板保留標籤; 藍色 = 展示桌保留標籤; "
        + (
            "桃紅色 = 展示櫃與直立架; 黃色 = boxed-stock。各類同色斜線 = 未決候選。"
            "直立架包含既有 other_shelf 候選, 此顯示分組不會修改原始標籤。"
            "重疊區域依地板、桌、櫃架、商品順序顯示, 不代表已解決分類衝突。"
            if combined
            else "桃紅色 = 展示櫃保留標籤; 橘色斜線 = 未決候選。"
        )
        + "未上色不代表不存在。未決候選尚未確認類別, 可能和其他物件衝突。"
        "本頁只改善顯示, 沒有新增、補全或確認標註。</p>"
        '<label><input type="checkbox" checked '
        "onchange=\"document.body.classList.toggle('clean',!this.checked)\">顯示未決候選</label>"
        + "".join(cards)
        + "</body></html>\n"
    )
    shutil.copyfile(__file__, out / "preview_worker.py")
    write(
        out / "report.json",
        {
            "kind": "visualization_only",
            "frames": len(job["frames"]),
            "source": str(source.resolve()),
            "source_report_sha256": digest(source / "report.json"),
            "annotations_changed": False,
            "combined": combined,
            "display_layers": SCENE_LAYERS if combined else None,
            "outputs": {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        },
    )
    print(f"Rendered {len(job['frames'])} previews at {out}")


def summarise(out: Path, job: dict) -> dict:
    counts, unresolved = Counter(), Counter()
    coco = {
        "info": {
            "annotation_kind": "ai_generated",
            "training_ready": False,
            "scope": "partial positives; unresolved-region/per-class supervision required",
        },
        "categories": [{"id": cid, "name": name} for name, cid in ENTITY_IDS.items()],
        "images": [],
        "annotations": [],
    }
    cards = []
    for image_id, frame in enumerate(job["frames"], 1):
        path = out / "frames" / (frame["id"] + ".json")
        data = json.loads(path.read_text())
        if (
            data.get("job_sha256") != digest(out / "job.json")
            or data.get("frame_id") != frame["id"]
        ):
            raise ValueError("annotation belongs to another job/frame")
        shape = tuple(frame["image_size_px"][::-1])
        validate_annotation(data, frame["image_sha256"], shape)
        if digest(out / frame["image"]) != frame["image_sha256"]:
            raise ValueError("frozen image changed")
        counts.update(row["entity"] for row in data["entities"])
        unresolved.update(row["entity"] for row in data["unresolved_candidates"])
        coco["images"].append(
            {
                "id": image_id,
                "file_name": frame["image"],
                "width": shape[1],
                "height": shape[0],
                "frame_id": frame["id"],
                "companion_annotations": str(path.relative_to(out)),
            }
        )
        coco["annotations"].extend(
            coco_annotations(data, image_id, len(coco["annotations"]) + 1)
        )
        name = html.escape(frame["id"])
        cards.append(
            f'<article><h2>{name}</h2><a href="{frame["image"]}">Source</a> | '
            f'<a href="frames/{name}.json">AI labels</a>'
            f'<img loading="lazy" src="frames/{name}.jpg"></article>'
        )
    write(out / "instances_ai.json", coco)
    report = {
        "status": "completed",
        "frames": len(job["frames"]),
        "instances": sum(counts.values()),
        "instances_by_entity": dict(counts),
        "unresolved_candidates": sum(unresolved.values()),
        "unresolved_by_entity": dict(unresolved),
        "annotation_kind": "ai_generated",
        "human_reviewed": False,
        "teacher": job["teacher"],
        "job_sha256": digest(out / "job.json"),
        "outputs": {
            str(p.relative_to(out)): digest(p) for p in sorted((out / "frames").glob("*.json"))
        },
        "limits": "AI pseudo-labels; no independent accuracy, absence or metric geometry claim",
    }
    report["outputs"]["instances_ai.json"] = digest(out / "instances_ai.json")
    write(out / "report.json", report)
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>StudioA AI annotations</title>'
        "<style>body{font-family:sans-serif;margin:2rem}main{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(min(600px,100%),1fr));gap:1rem}img{width:100%}</style>"
        "<h1>StudioA AI annotations</h1><p>AI-generated masks. "
        "Uncertain candidates remain in each JSON. "
        "Not independently measured accuracy.</p><main>" + "".join(cards) + "</main>\n"
    )
    return report


def refine(source: Path, out: Path) -> None:
    """Reclassify only source-bound mask conflicts; preserve the original inference run."""
    parent = json.loads((source / "report.json").read_text())
    parent_job = json.loads((source / "job.json").read_text())
    if parent["status"] != "completed" or digest(source / "job.json") != parent["job_sha256"]:
        raise ValueError("refinement requires a completed bound inference run")
    for name, expected in parent["outputs"].items():
        if digest(source / name) != expected:
            raise ValueError("parent annotation changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    shutil.copytree(source / "images", out / "images")
    (out / "snapshot").mkdir()
    for name in ("studioa_autolabel.py", "studioa_contract.py"):
        shutil.copyfile(ROOT / "src/syncai_hydranet/data" / name, out / "snapshot" / name)
    shutil.copyfile(__file__, out / "snapshot/worker.py")
    job = {
        "schema": "studioa.ai.refinement.v1",
        "frames": parent_job["frames"],
        "teacher": parent_job["teacher"],
        "policy": policy(),
        "contract": contract(),
        "parent_report_sha256": digest(source / "report.json"),
        "parent_run": str(source.resolve()),
        "code": {
            str(p.relative_to(out)): digest(p) for p in sorted((out / "snapshot").glob("*.py"))
        },
    }
    write(out / "job.json", job)
    completed = 0
    started = time.monotonic()
    try:
        for frame in job["frames"]:
            write(
                out / "status.json",
                {"status": "refining", "completed": completed, "total": len(job["frames"])},
            )
            source_path = source / "frames" / (frame["id"] + ".json")
            previous = json.loads(source_path.read_text())
            shape = tuple(frame["image_size_px"][::-1])
            validate_annotation(previous, frame["image_sha256"], shape)
            if (
                previous.get("job_sha256") != parent["job_sha256"]
                or previous.get("frame_id") != frame["id"]
            ):
                raise ValueError("parent frame identity mismatch")
            candidates = [
                Candidate(
                    row["entity"], decode(row["segmentation"]), row["score"], row["prompts"]
                )
                for row in previous["entities"] + previous["unresolved_candidates"]
            ]
            data = annotate(candidates, shape)
            data.update(
                frame_id=frame["id"],
                image_sha256=frame["image_sha256"],
                teacher=job["teacher"],
                job_sha256=digest(out / "job.json"),
                parent_annotation_sha256=digest(source_path),
            )
            validate_annotation(data, frame["image_sha256"], shape)
            with Image.open(out / frame["image"]) as image:
                overlay(image.convert("RGB"), data, out / "frames" / (frame["id"] + ".jpg"))
            write(out / "frames" / (frame["id"] + ".json"), data)
            completed += 1
        report = summarise(out, job)
        write(
            out / "status.json",
            {
                "status": "completed",
                "completed": completed,
                "total": len(job["frames"]),
                "elapsed_s": time.monotonic() - started,
                "instances": report["instances"],
            },
        )
    except BaseException:
        write(
            out / "status.json",
            {"status": "failed", "completed": completed, "error": traceback.format_exc()},
        )
        raise


def visual_review(source: Path, decisions_path: Path, out: Path) -> None:
    """Preserve the base inference job identity and attach a separate AI review layer."""
    parent = json.loads((source / "report.json").read_text())
    decisions = json.loads(decisions_path.read_text())
    if parent["status"] != "completed" or decisions.get("reviewer_kind") != "ai":
        raise ValueError("AI review requires completed labels and AI provenance")
    for name, expected in parent["outputs"].items():
        if digest(source / name) != expected:
            raise ValueError("AI review source changed")
    updates = {}
    for item in decisions["frames"]:
        identity = item["frame_id"]
        if Path(identity).name != identity or identity in updates:
            raise ValueError("invalid or duplicate AI review frame ID")
        path = source / "frames" / (identity + ".json")
        if digest(path) != item["annotation_sha256"]:
            raise ValueError("AI review annotation changed")
        original = json.loads(path.read_text())
        if original["image_sha256"] != item["image_sha256"]:
            raise ValueError("AI review image changed")
        updated = apply_ai_review(original, item["decisions"])
        validate_annotation(
            updated, item["image_sha256"], tuple(original["image_size_px"][::-1])
        )
        updated["ai_visual_review"] = {
            "reviewer_kind": "ai",
            "source_annotation_sha256": digest(path),
            "decisions_sha256": digest(decisions_path),
        }
        updates[identity] = updated
    shutil.copytree(source, out)
    write(
        out / "status.json", {"status": "applying_ai_review", "reviewed_frames": len(updates)}
    )
    shutil.copyfile(decisions_path, out / "ai_visual_decisions.json")
    shutil.copyfile(__file__, out / "ai_review_worker.py")
    shutil.copyfile(
        ROOT / "src/syncai_hydranet/data/studioa_autolabel.py", out / "ai_review_rules.py"
    )
    try:
        job = json.loads((out / "job.json").read_text())
        for frame in job["frames"]:
            if frame["id"] not in updates:
                continue
            data = updates[frame["id"]]
            write(out / "frames" / (frame["id"] + ".json"), data)
            with Image.open(out / frame["image"]) as image:
                overlay(image.convert("RGB"), data, out / "frames" / (frame["id"] + ".jpg"))
        report = summarise(out, job)
        report["ai_visual_review"] = {
            "parent_run": str(source.resolve()),
            "parent_report_sha256": digest(source / "report.json"),
            "frames_checked": len(updates),
            "decision_count": sum(len(item["decisions"]) for item in decisions["frames"]),
            "reviewer_kind": "ai",
            "decisions_sha256": digest(decisions_path),
            "worker_sha256": digest(out / "ai_review_worker.py"),
            "rules_sha256": digest(out / "ai_review_rules.py"),
        }
        write(out / "report.json", report)
        write(
            out / "status.json",
            {
                "status": "completed",
                "completed": len(job["frames"]),
                "total": len(job["frames"]),
                "instances": report["instances"],
            },
        )
    except BaseException:
        write(out / "status.json", {"status": "failed", "error": traceback.format_exc()})
        raise


def run(out: Path, device: str) -> None:
    job = json.loads((out / "job.json").read_text())
    if job["policy"] != policy() or job["teacher"] != {
        "model": sam3.MODEL_ID,
        "revision": sam3.MODEL_REVISION,
    }:
        raise ValueError("job policy/model differs from worker")
    for name, expected in job["code"].items():
        if digest(out / name) != expected:
            raise ValueError(f"frozen worker changed: {name}")
    lock = (out / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = 0
    started = time.monotonic()

    def status(state: str, **extra):
        write(
            out / "status.json",
            {
                "status": state,
                "completed": completed,
                "total": len(job["frames"]),
                "updated_at": datetime.now(UTC).isoformat(),
                "elapsed_s": time.monotonic() - started,
                **extra,
            },
        )

    def stopped(_signum, _frame):
        raise InterruptedError(
            "worker terminated; completed image checkpoints remain resumable"
        )

    signal.signal(signal.SIGTERM, stopped)
    try:
        status("loading_teacher")
        write(
            out / "environment.json",
            {
                "python": sys.version,
                "device": device,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "numpy", "pillow", "pycocotools")
                },
            },
        )
        proc, model = sam3.load_sam3(sam3.MODEL_ID, device)
        for frame in job["frames"]:
            image_path = out / frame["image"]
            if digest(image_path) != frame["image_sha256"]:
                raise ValueError("frozen source image changed")
            target = out / "frames" / (frame["id"] + ".json")
            shape = tuple(frame["image_size_px"][::-1])
            if target.exists():
                previous = json.loads(target.read_text())
                if (
                    previous.get("job_sha256") != digest(out / "job.json")
                    or previous.get("frame_id") != frame["id"]
                ):
                    raise ValueError("checkpoint belongs to another job/frame")
                validate_annotation(previous, frame["image_sha256"], shape)
                completed += 1
                continue
            status("running", frame_id=frame["id"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            embeds = sam3.vision_features(proc, model, image, device)
            candidates = []
            for entity, prompts in PROMPTS.items():
                for prompt in prompts:
                    found = sam3.segment(proc, model, image, prompt, MIN_SCORE, device, embeds)
                    candidates.extend(
                        Candidate(entity, mask, score, [prompt]) for mask, score in found
                    )
            data = annotate(candidates, shape)
            data.update(
                frame_id=frame["id"],
                image_sha256=frame["image_sha256"],
                teacher=job["teacher"],
                job_sha256=digest(out / "job.json"),
            )
            validate_annotation(data, frame["image_sha256"], shape)
            overlay(image, data, target.with_suffix(".jpg"))
            write(target, data)
            completed += 1
            print(
                f"{completed}/{len(job['frames'])} {frame['id']}: "
                f"{len(data['entities'])} labels, "
                f"{len(data['unresolved_candidates'])} uncertain",
                flush=True,
            )
            status("running", frame_id=frame["id"])
        report = summarise(out, job)
        status(
            "completed",
            instances=report["instances"],
            unresolved_candidates=report["unresolved_candidates"],
        )
    except BaseException:
        status("failed", error=traceback.format_exc())
        raise
    finally:
        lock.close()


def relabel_prepare(source: Path, out: Path, decisions_path: Path | None = None) -> None:
    parent = json.loads((source / "report.json").read_text())
    original = json.loads((source / "job.json").read_text())
    if parent["status"] != "completed" or digest(source / "job.json") != parent["job_sha256"]:
        raise ValueError("relabel requires a completed bound source run")
    for name, expected in parent["outputs"].items():
        if digest(source / name) != expected:
            raise ValueError("source annotation changed")
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("frames", "sources", "raw"):
        (out / folder).mkdir()
    shutil.copytree(source / "images", out / "images")
    snapshot = out / "snapshot"
    shutil.copytree(
        ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (snapshot / "tools/annotation").mkdir(parents=True)
    shutil.copyfile(__file__, snapshot / "tools/annotation/studioa_autolabel.py")
    frames = []
    for frame in original["frames"]:
        if digest(out / frame["image"]) != frame["image_sha256"]:
            raise ValueError("source image changed")
        name = f"sources/{frame['id']}.json"
        shutil.copyfile(source / "frames" / (frame["id"] + ".json"), out / name)
        frames.append({**frame, "annotation": name, "annotation_sha256": digest(out / name)})
    visual = None
    if decisions_path is not None:
        document = json.loads(decisions_path.read_text())
        validate_visual_decisions(
            document,
            frames,
            {f["id"]: json.loads((out / f["annotation"]).read_text()) for f in frames},
        )
        shutil.copyfile(decisions_path, out / "ai_visual_decisions.json")
        visual = {
            "path": "ai_visual_decisions.json",
            "sha256": digest(out / "ai_visual_decisions.json"),
        }
    write(
        out / "job.json",
        {
            "schema": "studioa.ai.relabel.v1",
            "visual_decisions": visual,
            "frames": frames,
            "teacher": original["teacher"],
            "parent_run": str(source.resolve()),
            "parent_report_sha256": digest(source / "report.json"),
            "relabel_policy": relabel_policy(),
            "code": {
                str(p.relative_to(out)): digest(p) for p in sorted(snapshot.rglob("*.py"))
            },
        },
    )
    write(out / "status.json", {"status": "prepared", "completed": 0, "total": len(frames)})


def relabel_run(out: Path, device: str) -> None:
    job = json.loads((out / "job.json").read_text())
    job_hash = digest(out / "job.json")
    if job["relabel_policy"] != relabel_policy():
        raise ValueError("relabel policy changed")
    for name, expected in job["code"].items():
        if digest(out / name) != expected:
            raise ValueError("frozen relabel code changed")
    overrides = {}
    if visual := job.get("visual_decisions"):
        if digest(out / visual["path"]) != visual["sha256"]:
            raise ValueError("frozen visual decisions changed")
        overrides = validate_visual_decisions(
            json.loads((out / visual["path"]).read_text()),
            job["frames"],
            {f["id"]: json.loads((out / f["annotation"]).read_text()) for f in job["frames"]},
        )
    lock = (out / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = 0
    started = time.monotonic()

    def status(state, **kw):
        write(
            out / "status.json",
            {
                "status": state,
                "completed": completed,
                "total": len(job["frames"]),
                "elapsed_s": time.monotonic() - started,
                **kw,
            },
        )

    def stopped(_signum, _frame):
        raise InterruptedError("relabel stopped; source-bound batch decisions remain resumable")

    signal.signal(signal.SIGTERM, stopped)
    try:
        status("loading_teachers")
        write(
            out / "environment.json",
            {
                "python": sys.version,
                "device": device,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "numpy", "pillow", "pycocotools")
                },
            },
        )
        reviewer = LocalReviewer(device)
        proc, model = sam3.load_sam3(sam3.MODEL_ID, device)
        for frame in job["frames"]:
            identity = frame["id"]
            target = out / "frames" / (identity + ".json")
            raw_path = out / "raw" / (identity + ".json")
            shape = tuple(frame["image_size_px"][::-1])
            if (
                digest(out / frame["image"]) != frame["image_sha256"]
                or digest(out / frame["annotation"]) != frame["annotation_sha256"]
            ):
                raise ValueError("frozen relabel input changed")
            if target.exists():
                data = json.loads(target.read_text())
                if (
                    data.get("job_sha256") != job_hash
                    or data.get("frame_id") != identity
                    or data.get("decisions_sha256") != digest(raw_path)
                ):
                    raise ValueError("foreign or changed relabel checkpoint")
                validate_annotation(data, frame["image_sha256"], shape)
                completed += 1
                continue
            with Image.open(out / frame["image"]) as source:
                image = source.convert("RGB")
            status("supplementing", frame_id=identity)
            if raw_path.exists():
                raw: ReviewCache = json.loads(raw_path.read_text())
                if raw.get("job_sha256") != job_hash or raw.get("frame_id") != identity:
                    raise ValueError("foreign relabel decision cache")
            else:
                previous = json.loads((out / frame["annotation"]).read_text())
                validate_annotation(previous, frame["image_sha256"], shape)
                embeds = sam3.vision_features(proc, model, image, device)
                extras = []
                for entity, prompts in EXTRA_PROMPTS.items():
                    for prompt in prompts:
                        extras.extend(
                            Candidate(entity, mask, score, [prompt])
                            for mask, score in sam3.segment(
                                proc, model, image, prompt, MIN_SCORE, device, embeds
                            )
                        )
                extra = annotate(extras, shape)
                extra_rows = extra["entities"] + extra["unresolved_candidates"]
                for row in extra_rows:
                    row["id"] = "extra-" + row["id"]
                groups = candidate_groups(
                    previous["entities"] + previous["unresolved_candidates"] + extra_rows
                )
                raw = ReviewCache(
                    job_sha256=job_hash,
                    frame_id=identity,
                    groups=groups,
                    decisions=[],
                    source_annotation_sha256=frame["annotation_sha256"],
                    extra_candidates=len(extra_rows),
                )
                write(raw_path, raw)
            groups = raw["groups"]
            frame_overrides = overrides.get(identity, {})
            if not set(frame_overrides) <= {row["id"] for group in groups for row in group}:
                raise ValueError("visual decision target missing from review groups")
            batch_size = job["relabel_policy"]["batch_size"]
            for first in range(len(raw["decisions"]), len(groups), batch_size):
                status(
                    "classifying",
                    frame_id=identity,
                    reviewed_groups=first,
                    total_groups=len(groups),
                )
                batch_groups = groups[first : first + batch_size]
                panels = [review_panel(image, group[0]) for group in batch_groups]
                prompts = [
                    FIXTURE_PROMPT
                    if group[0]["entity"]
                    in {"display_table", "display_cabinet", "counter", "other_shelf"}
                    else REVIEW_PROMPT
                    for group in batch_groups
                ]
                decisions = reviewer.classify(panels, prompts)
                if len(decisions) != len(panels):
                    raise ValueError("review batch length mismatch")
                raw["decisions"].extend(
                    apply_visual_decision(
                        group, constrain_decision(group, decision), frame_overrides
                    )
                    for group, decision in zip(batch_groups, decisions, strict=True)
                )
                write(raw_path, raw)
            data = reannotate(groups, raw["decisions"], shape)
            data.update(
                frame_id=identity,
                image_sha256=frame["image_sha256"],
                teacher=job["teacher"],
                job_sha256=job_hash,
                decisions_sha256=digest(raw_path),
                parent_annotation_sha256=frame["annotation_sha256"],
            )
            validate_annotation(data, frame["image_sha256"], shape)
            write(target, data)
            overlay(image, data, target.with_suffix(".jpg"))
            completed += 1
            print(
                f"{completed}/{len(job['frames'])} {identity}: {len(data['entities'])} labels, "
                f"{len(data['unresolved_candidates'])} unresolved, "
                f"{len(groups)} reviewed groups",
                flush=True,
            )
            status("running", frame_id=identity)
        status("summarising")
        report = summarise(out, job)
        report["outputs"].update(
            {str(p.relative_to(out)): digest(p) for p in sorted((out / "raw").glob("*.json"))}
        )
        report["ai_reclassification"] = {
            "policy": relabel_policy(),
            "parent_report_sha256": job["parent_report_sha256"],
        }
        write(out / "report.json", report)
        status(
            "completed",
            instances=report["instances"],
            unresolved_candidates=report["unresolved_candidates"],
        )
    except BaseException:
        status("failed", error=traceback.format_exc())
        raise
    finally:
        lock.close()


def relabel_review(source: Path, decisions_path: Path, out: Path) -> None:
    """Rebuild masks after source-bound AI inspection of full candidate groups."""
    parent = json.loads((source / "report.json").read_text())
    job = json.loads((source / "job.json").read_text())
    document = json.loads(decisions_path.read_text())
    if parent["status"] != "completed" or document.get("reviewer_kind") != "ai":
        raise ValueError("group review requires complete inference and AI provenance")
    if parent["job_sha256"] != digest(source / "job.json"):
        raise ValueError("group review source job changed")
    for name, expected in parent["outputs"].items():
        if digest(source / name) != expected:
            raise ValueError("group review source output changed")
    frames = {frame["id"]: frame for frame in job["frames"]}
    updates = {}
    for item in document["frames"]:
        identity = item["frame_id"]
        if identity not in frames or identity in updates:
            raise ValueError("invalid or duplicate group review frame")
        frame = frames[identity]
        raw_path = source / "raw" / (identity + ".json")
        if (
            item["raw_sha256"] != digest(raw_path)
            or item["image_sha256"] != frame["image_sha256"]
            or digest(source / frame["image"]) != frame["image_sha256"]
        ):
            raise ValueError("group review source hash mismatch")
        raw = json.loads(raw_path.read_text())
        revised = revise_group_decisions(raw, item["decisions"])
        data = reannotate(raw["groups"], revised, tuple(frame["image_size_px"][::-1]))
        original = json.loads((source / "frames" / (identity + ".json")).read_text())
        data.update(
            {
                key: original[key]
                for key in (
                    "frame_id",
                    "image_sha256",
                    "teacher",
                    "job_sha256",
                    "decisions_sha256",
                    "parent_annotation_sha256",
                )
            }
        )
        data["ai_group_review"] = {
            "reviewer_kind": "ai",
            "decisions_sha256": digest(decisions_path),
            "source_annotation_sha256": digest(source / "frames" / (identity + ".json")),
            "decisions": item["decisions"],
        }
        validate_annotation(data, frame["image_sha256"], tuple(frame["image_size_px"][::-1]))
        updates[identity] = data
    shutil.copytree(source, out)
    write(out / "status.json", {"status": "applying_group_review"})
    shutil.copyfile(decisions_path, out / "ai_group_decisions.json")
    shutil.copyfile(__file__, out / "ai_group_worker.py")
    shutil.copyfile(
        ROOT / "src/syncai_hydranet/data/studioa_relabel.py", out / "ai_group_rules.py"
    )
    try:
        for identity, data in updates.items():
            target = out / "frames" / (identity + ".json")
            write(target, data)
            with Image.open(out / frames[identity]["image"]) as image:
                overlay(image.convert("RGB"), data, target.with_suffix(".jpg"))
        report = summarise(out, job)
        report["outputs"].update(
            {str(p.relative_to(out)): digest(p) for p in (out / "raw").glob("*.json")}
        )
        for name in ("ai_group_decisions.json", "ai_group_worker.py", "ai_group_rules.py"):
            report["outputs"][name] = digest(out / name)
        report["ai_group_review"] = {
            "parent_run": str(source.resolve()),
            "parent_report_sha256": digest(source / "report.json"),
            "frames": len(updates),
            "decisions": sum(len(item["decisions"]) for item in document["frames"]),
            "reviewer_kind": "ai",
        }
        write(out / "report.json", report)
        write(
            out / "status.json",
            {
                "status": "completed",
                "completed": len(frames),
                "total": len(frames),
                "instances": report["instances"],
            },
        )
    except BaseException:
        write(out / "status.json", {"status": "failed", "error": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("prepare-media")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p = sub.add_parser("refine")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("visual-review")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("focus-preview")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--combined", action="store_true")
    p = sub.add_parser("relabel-prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--decisions", type=Path)
    p = sub.add_parser("relabel-review")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p = sub.add_parser("relabel-run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.bundle, args.out, args.limit)
    elif args.action == "prepare-media":
        prepare_media(args.source, args.selection, args.out)
    elif args.action == "refine":
        refine(args.source, args.out)
    elif args.action == "visual-review":
        visual_review(args.source, args.decisions, args.out)
    elif args.action == "focus-preview":
        focus_preview(args.source, args.out, args.combined)
    elif args.action == "relabel-prepare":
        relabel_prepare(args.source, args.out, args.decisions)
    elif args.action == "relabel-review":
        relabel_review(args.source, args.decisions, args.out)
    else:
        expected = args.out.resolve() / "snapshot/tools/annotation/studioa_autolabel.py"
        if Path(__file__).resolve() != expected:
            env = dict(os.environ, PYTHONPATH=str(args.out.resolve() / "snapshot/src"))
            os.execve(
                sys.executable,
                [
                    sys.executable,
                    str(expected),
                    args.action,
                    "--out",
                    str(args.out.resolve()),
                    "--device",
                    args.device,
                ],
                env,
            )
        if args.action == "relabel-run":
            relabel_run(args.out, args.device)
        else:
            run(args.out, args.device)


if __name__ == "__main__":
    main()
