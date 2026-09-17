#!/usr/bin/env python3
"""Freeze a finite overnight training matrix and publish model checkpoints."""

from __future__ import annotations

import argparse
import fcntl
import html
import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from syncai_hydranet.data.studioa_review import digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def child(command, log, seconds, *, env=None):
    """Bound only our own process group; let workers save a failure record on timeout."""
    with log.open("a") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        try:
            code = process.wait(timeout=max(1, seconds))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"worker exceeded {seconds:.0f}s; see {log}") from None
    if code:
        raise RuntimeError(f"worker exit {code}; see {log}")


def choose_model(completed):
    """Only completed new runs enter selection; all use the same frozen validation."""
    if not completed:
        raise ValueError("no completed new model")
    return max(completed, key=lambda row: (row["miou"], row["name"]))


def prepare(out, source, weights, deadline):
    if (out / "plan.json").exists():
        raise ValueError("overnight plan already exists")
    worker = runpy.run_path(str(ROOT / "tools/annotation/studioa_train.py"))
    jobs = [{"name": "bootstrap", "seed": 42, "crop": True, "updates": 495}]
    jobs += [
        {"name": f"{mode}_seed{seed}", "seed": seed, "crop": mode == "focus", "updates": 4950}
        for seed in (42, 43, 44)
        for mode in ("uniform", "focus")
    ]
    for job in jobs:
        destination = out / "jobs" / job["name"]
        worker["prepare"](
            source,
            destination,
            "Tao-Hsin",
            scene_comparison_updates=job["updates"],
            scene_validation_updates=495,
            scene_class_weights_source=weights,
            scene_validation_source=source,
            scene_focus_crop=job["crop"],
            scene_seed=job["seed"],
        )
        worker["verify"](destination)
        job["job_sha256"] = digest(destination / "job.json")
        print("Prepared", job["name"], flush=True)
    runner = out / "runner"
    runner.mkdir()
    shutil.copyfile(__file__, runner / "studioa_overnight.py")
    plan = {
        "schema": "studioa.overnight.v2",
        "deadline": deadline,
        "training_stop": datetime.fromisoformat(deadline).timestamp() - 300,
        "jobs": jobs,
        "runner_files": {p.name: digest(p) for p in runner.iterdir()},
        "git_commit": read(out / "jobs/bootstrap/job.json")["git_commit"],
        "selection": (
            "maximum completed checkpoint source-val 19-class mIoU; "
            "AI partial labels, not independent accuracy"
        ),
        "test_evaluated": False,
        "automatic_deployment": False,
    }
    write_json(out / "plan.json", plan)
    write_json(
        out / "status.json", {"status": "prepared", "deadline": deadline, "completed_jobs": []}
    )


def publish(out, _plan, selected, deadline):
    """Publish training artifacts only; object worlds use the scene_mesh CLI."""
    run = out / "jobs" / selected["name"]
    destination = out / "deliveries" / selected["name"]
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "model/best.pt", destination / "model.pt")
    shutil.copyfile(run / "config.json", destination / "config.json")
    delivery = {
        "selected": selected,
        "checkpoint_sha256": digest(run / "model/best.pt"),
        "model": str((destination / "model.pt").relative_to(out)),
        "preview_root": str(destination.relative_to(out)),
        "cameras": [],
        "render_errors": [],
        "scene_status": "not_built; use tools/commissioning/scene_mesh.py --model-run",
        "completed_at": datetime.now(UTC).isoformat(),
        "before_deadline": time.time() <= deadline,
        "test_evaluated": False,
        "automatic_deployment": False,
    }
    write_json(destination / "delivery.json", delivery)
    write_json(out / "delivery.json", delivery)
    gallery(out, delivery, "模型訓練完成;物件式 3D 場景需另行建置")
    return delivery


def gallery(out, delivery, status):
    page = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>StudioA 模型訓練</title><h1>StudioA 模型訓練</h1>
<p>{html.escape(status)}</p>
<p>模型: {html.escape(delivery["selected"]["name"])};
AI val mIoU: {delivery["selected"]["miou"] * 100:.2f}% (非獨立準確率)</p>
<a href="{delivery["model"]}">下載模型</a>
<p>物件式場景使用既有 scene_mesh.py --model-run 建置與驗證。</p></html>'''
    atomic_text(out / "index.html", page)


def run(out):
    lock = (out / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(out / "plan.json")
    deadline = datetime.fromisoformat(plan["deadline"]).timestamp()
    for name, sha in plan["runner_files"].items():
        if digest(out / "runner" / name) != sha:
            raise ValueError("frozen runner changed")
    worker = runpy.run_path(
        str(out / "jobs/bootstrap/snapshot/tools/annotation/studioa_train.py")
    )
    completed = []
    errors = []
    for job in plan["jobs"]:
        name = job["name"]
        directory = out / "jobs" / name
        if time.time() >= plan["training_stop"] and not (directory / "report.json").exists():
            errors.append(
                {"job": name, "error": "training cutoff reached; final publication reserved"}
            )
            continue
        state = {
            "status": "training",
            "current_job": name,
            "deadline": plan["deadline"],
            "completed_jobs": completed,
            "errors": errors,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        write_json(out / "status.json", state)
        try:
            if digest(directory / "job.json") != job["job_sha256"]:
                raise ValueError("frozen training job changed")
            worker["verify"](directory)
            if not (directory / "report.json").exists():
                child(
                    [
                        sys.executable,
                        str(directory / "snapshot/tools/annotation/studioa_train.py"),
                        "run",
                        "--out",
                        str(directory),
                    ],
                    directory / "service.log",
                    min(1800, max(1, plan["training_stop"] - time.time())),
                    env=dict(
                        os.environ,
                        PYTHONPATH=str(directory / "snapshot/src"),
                        TORCH_HOME=str(directory / "torch"),
                    ),
                )
            report = worker["verify_completed"](directory, read(directory / "job.json"))
            from syncai_hydranet.utils.checkpoint import load_checkpoint

            last = load_checkpoint(directory / "model/last.pt")
            if last["global_step"] != job["updates"]:
                raise ValueError("completed run has the wrong optimizer update budget")
            del last
            result = {
                "name": name,
                "seed": job["seed"],
                "crop": job["crop"],
                "updates": job["updates"],
                "miou": report["validation_teacher_miou"],
                "best_epoch": report["best_epoch"],
                "checkpoint_sha256": digest(directory / "model/best.pt"),
                "validation": read(directory / "validation.json"),
            }
            completed.append(result)
            for partner in completed[:-1]:
                if partner["seed"] == job["seed"] and partner["updates"] == job["updates"]:
                    import torch

                    initial = torch.load(
                        directory / "initial_state.pt", map_location="cpu", weights_only=True
                    )
                    reference = torch.load(
                        out / "jobs" / partner["name"] / "initial_state.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    if initial.keys() != reference.keys() or not all(
                        torch.equal(value, reference[key]) for key, value in initial.items()
                    ):
                        completed.pop()
                        raise ValueError("paired seed initial weights differ")
                    result["paired_initial_tensors_equal"] = True
                    partner["paired_initial_tensors_equal"] = True
                    del initial, reference
            write_json(
                out / "summary.json",
                {
                    "completed_jobs": completed,
                    "errors": errors,
                    "plan_sha256": digest(out / "plan.json"),
                },
            )
            if name == "bootstrap" or not (out / "delivery.json").exists():
                publish(out, plan, result, deadline)
        except Exception as error:
            errors.append({"job": name, "error": str(error)})
            write_json(out / "errors.json", {"errors": errors})
    selected = choose_model(completed)
    try:
        delivery = publish(out, plan, selected, deadline)
    except Exception as error:
        errors.append({"job": "final_publish", "error": str(error)})
        if not (out / "delivery.json").exists():
            raise
        delivery = read(out / "delivery.json")
    state = {
        "status": "completed"
        if len(completed) == len(plan["jobs"]) and not errors
        else "completed_with_issues",
        "deadline": plan["deadline"],
        "completed_jobs": completed,
        "errors": errors,
        "delivery": delivery,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    write_json(out / "summary.json", state)
    write_json(out / "status.json", state)
    gallery(
        out,
        delivery,
        "夜間批次完成" if not errors else "已有模型交付;部分工作有錯誤,詳見工作狀態",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--deadline")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "prepare":
        if args.source is None or args.weights is None or args.deadline is None:
            parser.error("prepare requires source, weights and deadline")
        prepare(out, args.source, args.weights, args.deadline)
    else:
        if read(out / "plan.json").get("schema") != "studioa.overnight.v2":
            parser.error(
                "legacy surface-preview jobs are retired; preserve their results as history"
            )
        frozen = out / "runner/studioa_overnight.py"
        if Path(__file__).resolve() != frozen:
            os.execve(
                sys.executable,
                [sys.executable, str(frozen), "run", "--out", str(out)],
                dict(os.environ, PYTHONPATH=str(out / "jobs/bootstrap/snapshot/src")),
            )
        try:
            run(out)
        except Exception as error:
            write_json(
                out / "failure.json",
                {"error": str(error), "time": datetime.now(UTC).isoformat()},
            )
            raise


if __name__ == "__main__":
    main()
