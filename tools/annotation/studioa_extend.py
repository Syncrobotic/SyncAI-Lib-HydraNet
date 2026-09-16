#!/usr/bin/env python3
"""Append explicitly AI-reviewed train positives while preserving every prior target."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path

from syncai_hydranet.data.studioa_autolabel import coverage, validate_annotation
from syncai_hydranet.data.studioa_contract import ENTITY_IDS, entity_views
from syncai_hydranet.data.studioa_review import digest, write_json
from syncai_hydranet.data.studioa_supervision import check_supervision, export_supervision


def retain_reviewed(data, decisions):
    result = copy.deepcopy(data)
    candidates = {r["id"]: r for r in result["entities"] + result["unresolved_candidates"]}
    accepted, seen = [], set()
    for decision in decisions:
        identity, entity = decision["id"], decision["entity"]
        row = candidates[identity]
        if identity in seen or not decision.get("reason") or entity not in ENTITY_IDS:
            raise ValueError("positive review needs unique supported class and reason")
        seen.add(identity)
        if entity != row["entity"]:
            row["original_proposal_entity"] = row["entity"]
            row["entity"] = entity
        row["assistant_review"] = decision["reason"]
        accepted.append(row)
    if not accepted:
        raise ValueError("new train frame needs at least one reviewed positive")
    result["entities"] = accepted
    result["unresolved_candidates"] = []
    # Unselected teacher proposals establish neither positive nor negative truth.
    # Preserve them for audit outside supervision; unclaimed pixels remain ignore.
    result["unreviewed_proposals"] = [r for ident, r in candidates.items() if ident not in seen]
    result["coverage"] = coverage(accepted, [])
    result["counts_by_view"] = dict(Counter(v for r in accepted for v in entity_views(r)))
    result["supervision_scope"] = "AI-reviewed positive subset; all other pixels unknown"
    return result


def extend(base: Path, media: Path, decisions: Path, out: Path):
    original = check_supervision(base)
    report = json.loads((media / "report.json").read_text())
    job = json.loads((media / "job.json").read_text())
    review = json.loads(decisions.read_text())
    if (
        report["status"] != "completed"
        or report["job_sha256"] != digest(media / "job.json")
        or review.get("reviewer_kind") != "ai"
        or review.get("source_job_sha256") != report["job_sha256"]
    ):
        raise ValueError("requires completed source-bound AI review")
    for name, expected in report["outputs"].items():
        if digest(media / name) != expected:
            raise ValueError("source AI annotation changed")
    allowed = {
        f["camera"]
        for f in original["frames"]
        if original["folds"]["Tao-Hsin"]["assignments"][f["id"]] == "train"
    }
    additions = {f["id"]: f for f in job["frames"]}
    chosen, seen = [], {f["id"] for f in original["frames"]}
    for decision in review["frames"]:
        frame = additions[decision["frame_id"]]
        if (
            frame["id"] in seen
            or frame["camera"] not in allowed
            or frame["original_split"] != "train"
        ):
            raise ValueError("new frames must be unique and belong to existing train cameras")
        seen.add(frame["id"])
        annotation = media / "frames" / (frame["id"] + ".json")
        if decision["annotation_sha256"] != digest(annotation):
            raise ValueError("AI decision annotation changed")
        data = retain_reviewed(json.loads(annotation.read_text()), decision["positives"])
        validate_annotation(data, frame["image_sha256"], tuple(frame["image_size_px"][::-1]))
        chosen.append((frame, data))
    out.mkdir(parents=True, exist_ok=False)
    merged = out / "ai_source"
    (merged / "images").mkdir(parents=True)
    (merged / "frames").mkdir()
    shutil.copyfile(__file__, merged / "producer.py")
    shutil.copyfile(decisions, merged / "decisions.json")
    fields = (
        "id",
        "camera",
        "store",
        "original_split",
        "image_size_px",
        "image",
        "image_sha256",
    )
    all_frames = [
        {k: f[k] for k in fields} for f in original["frames"] + [f for f, _ in chosen]
    ]
    write_json(
        merged / "job.json",
        {
            "schema": "studioa.reviewed-extension.v1",
            "frames": all_frames,
            "base_manifest_sha256": digest(base / "manifest.json"),
            "media_job_sha256": digest(media / "job.json"),
            "decisions_sha256": digest(decisions),
            "policy": "append reviewed train positives; preserve old labels; unknown is ignore",
        },
    )
    job_hash = digest(merged / "job.json")
    for frame in original["frames"]:
        shutil.copyfile(base / frame["image"], merged / frame["image"])
        data = json.loads((base / frame["companion"]).read_text())
        data["parent_job_sha256"] = data["job_sha256"]
        data["job_sha256"] = job_hash
        write_json(merged / "frames" / (frame["id"] + ".json"), data)
    for frame, data in chosen:
        if digest(media / frame["image"]) != frame["image_sha256"]:
            raise ValueError("new source image changed")
        shutil.copyfile(media / frame["image"], merged / frame["image"])
        data["parent_job_sha256"] = data["job_sha256"]
        data["job_sha256"] = job_hash
        write_json(merged / "frames" / (frame["id"] + ".json"), data)
    write_json(
        merged / "report.json",
        {
            "status": "completed",
            "job_sha256": job_hash,
            "outputs": {
                str(p.relative_to(merged)): digest(p)
                for p in sorted(merged.rglob("*"))
                if p.is_file()
            },
        },
    )
    export_supervision(merged, out / "semantic")
    extended = check_supervision(out / "semantic")
    for frame in original["frames"]:
        if digest(base / frame["mask"]) != digest(out / "semantic" / frame["mask"]):
            raise ValueError("prior semantic target changed")
    for store, fold in original["folds"].items():
        if any(
            extended["folds"][store]["assignments"][fid] != role
            for fid, role in fold["assignments"].items()
        ):
            raise ValueError("prior split assignment changed")
    write_json(
        out / "extension_report.json",
        {
            "status": "completed",
            "added_train_frames": len(chosen),
            "prior_masks_preserved": len(original["frames"]),
            "prior_assignments_preserved": True,
            "semantic_manifest_sha256": digest(out / "semantic/manifest.json"),
            "independent_accuracy": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "media", "decisions", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    extend(args.base, args.media, args.decisions, args.out)


if __name__ == "__main__":
    main()
