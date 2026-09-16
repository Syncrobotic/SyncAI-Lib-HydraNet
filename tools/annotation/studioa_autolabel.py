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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
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
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.bundle, args.out, args.limit)
    elif args.action == "refine":
        refine(args.source, args.out)
    elif args.action == "visual-review":
        visual_review(args.source, args.decisions, args.out)
    else:
        expected = args.out.resolve() / "snapshot/tools/annotation/studioa_autolabel.py"
        if Path(__file__).resolve() != expected:
            env = dict(os.environ, PYTHONPATH=str(args.out.resolve() / "snapshot/src"))
            os.execve(
                sys.executable,
                [
                    sys.executable,
                    str(expected),
                    "run",
                    "--out",
                    str(args.out.resolve()),
                    "--device",
                    args.device,
                ],
                env,
            )
        run(args.out, args.device)


if __name__ == "__main__":
    main()
