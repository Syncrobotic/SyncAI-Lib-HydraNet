#!/usr/bin/env python3
"""Recheck mixed-family groups without fixture context, retaining source-bound answers."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

from syncai_hydranet.data.studioa_autolabel import validate_annotation
from syncai_hydranet.data.studioa_relabel import (
    MODEL,
    PROMPT,
    REVISION,
    LocalReviewer,
    constrain_isolated_recheck,
    needs_isolated_recheck,
    review_panel,
)
from syncai_hydranet.data.studioa_review import digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def prepare(source: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        ROOT / "src",
        out / "snapshot/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (out / "snapshot/tools/annotation").mkdir(parents=True)
    shutil.copyfile(__file__, out / "snapshot/tools/annotation/studioa_isolated_recheck.py")
    (out / "frames").mkdir()
    write_json(
        out / "job.json",
        {
            "source": str(source.resolve()),
            "source_job_sha256": digest(source / "job.json"),
            "model": MODEL,
            "revision": REVISION,
            "prompt": PROMPT,
            "policy": "mixed families predicted as furniture; isolated target",
            "code": {
                str(p.relative_to(out)): digest(p)
                for p in sorted((out / "snapshot").rglob("*.py"))
            },
        },
    )
    write_json(out / "status.json", {"status": "prepared"})


def run(out: Path, device: str) -> None:
    spec = json.loads((out / "job.json").read_text())
    for name, expected in spec["code"].items():
        if digest(out / name) != expected:
            raise ValueError("recheck frozen code changed")
    source = Path(spec["source"])
    if digest(source / "job.json") != spec["source_job_sha256"]:
        raise ValueError("recheck source job changed")
    job = json.loads((source / "job.json").read_text())
    lock = (out / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job_hash = digest(out / "job.json")
    completed = 0
    started = time.monotonic()

    def status(state, **extra):
        write_json(
            out / "status.json",
            {
                "status": state,
                "completed": completed,
                "total": len(job["frames"]),
                "elapsed_s": time.monotonic() - started,
                **extra,
            },
        )

    def stopped(_signum, _frame):
        raise InterruptedError("isolated recheck stopped; completed frames are resumable")

    signal.signal(signal.SIGTERM, stopped)
    try:
        status("loading_model")
        reviewer = LocalReviewer(device)
        results = []
        for frame in job["frames"]:
            identity = frame["id"]
            annotation = source / "frames" / (identity + ".json")
            raw_path = source / "raw" / (identity + ".json")
            target = out / "frames" / (identity + ".json")
            while not annotation.exists():
                state = json.loads((source / "status.json").read_text())
                if state["status"] in ("failed", "completed"):
                    raise ValueError("source stopped before required frame")
                status("waiting_source", frame_id=identity)
                time.sleep(5)
            data = json.loads(annotation.read_text())
            raw_hash = digest(raw_path)
            if (
                data["job_sha256"] != spec["source_job_sha256"]
                or data["decisions_sha256"] != raw_hash
                or digest(source / frame["image"]) != frame["image_sha256"]
            ):
                raise ValueError("unbound source frame/raw/image")
            validate_annotation(
                data, frame["image_sha256"], tuple(frame["image_size_px"][::-1])
            )
            raw = json.loads(raw_path.read_text())
            selected = [
                i
                for i, (g, d) in enumerate(zip(raw["groups"], raw["decisions"], strict=True))
                if needs_isolated_recheck(g, d)
            ]
            if target.exists():
                result = json.loads(target.read_text())
                if (
                    result.get("job_sha256") != job_hash
                    or result["raw_sha256"] != raw_hash
                    or result["image_sha256"] != frame["image_sha256"]
                    or [d["group"] for d in result["decisions"]] != selected
                ):
                    raise ValueError("foreign recheck checkpoint")
            else:
                result = {
                    "job_sha256": job_hash,
                    "frame_id": identity,
                    "image_sha256": frame["image_sha256"],
                    "raw_sha256": raw_hash,
                    "decisions": [],
                }
                with Image.open(source / frame["image"]) as original:
                    image = original.convert("RGB")
                for offset in range(0, len(selected), 16):
                    indices = selected[offset : offset + 16]
                    groups = [raw["groups"][i] for i in indices]
                    panels = [
                        review_panel(image, {**g[0], "entity": "unknown"}) for g in groups
                    ]
                    status(
                        "classifying", frame_id=identity, groups=len(selected), offset=offset
                    )
                    answers = reviewer.classify(panels)
                    for i, group, answer in zip(indices, groups, answers, strict=True):
                        answer = constrain_isolated_recheck(group, answer)
                        result["decisions"].append(
                            {
                                **answer,
                                "group": i,
                                "review_source": "local_vlm_isolated_target",
                                "reason": answer["reason"],
                            }
                        )
                write_json(target, result)
            if result["decisions"]:
                results.append(result)
            completed += 1
            status("running", frame_id=identity)
            print(
                f"{completed}/{len(job['frames'])} {identity}: {len(selected)} groups",
                flush=True,
            )
        while not (source / "report.json").exists():
            if json.loads((source / "status.json").read_text())["status"] == "failed":
                raise ValueError("source inference failed")
            status("waiting_source_report")
            time.sleep(5)
        parent = json.loads((source / "report.json").read_text())
        if parent["status"] != "completed":
            raise ValueError("source report incomplete")
        for name, expected in parent["outputs"].items():
            if digest(source / name) != expected:
                raise ValueError("source output changed")
        write_json(
            out / "decisions.json",
            {
                "reviewer_kind": "ai",
                "review_method": "Local VLM isolated-target recheck; no human annotations",
                "source_report_sha256": digest(source / "report.json"),
                "frames": results,
            },
        )
        write_json(
            out / "report.json",
            {
                "status": "completed",
                "frames": completed,
                "reviewed_groups": sum(len(r["decisions"]) for r in results),
                "job_sha256": digest(out / "job.json"),
                "outputs": {
                    str(p.relative_to(out)): digest(p)
                    for p in [out / "decisions.json", *sorted((out / "frames").glob("*.json"))]
                },
            },
        )
        status("completed")
    except BaseException:
        status("failed", error=traceback.format_exc())
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(args.source, args.out)
    else:
        expected = args.out.resolve() / "snapshot/tools/annotation/studioa_isolated_recheck.py"
        if Path(__file__).resolve() != expected:
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
                dict(os.environ, PYTHONPATH=str(args.out.resolve() / "snapshot/src")),
            )
        run(args.out, args.device)


if __name__ == "__main__":
    main()
