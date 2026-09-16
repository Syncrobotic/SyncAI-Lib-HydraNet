#!/usr/bin/env python3
"""Freeze and run a finite StudioA partial-semantic pilot using the existing Trainer."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from syncai_hydranet.config_schema import check_config
from syncai_hydranet.data.studioa_review import digest, write_json
from syncai_hydranet.data.studioa_supervision import CLASSES, check_supervision

ROOT = Path(__file__).resolve().parents[2]


def pilot_config(data: Path, out: Path, manifest: dict, held_out: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs/_base/hydranet.yaml").read_text())
    counts = manifest["folds"][held_out]["pixels_by_split"]["train"]
    pixels = np.array([counts[name] for name in CLASSES], dtype=float)
    weights = np.clip(np.sqrt(np.median(pixels[pixels > 0]) / np.maximum(pixels, 1)), 0.25, 4)
    cfg.update(
        experiment="studioa_scene_taohsin_pilot",
        output_dir=str(out / "model"),
        device="cuda",
        seed=42,
    )
    cfg["model"] = {
        "backbone": {"name": "resnet18", "pretrained": True, "frozen_stages": 1},
        "neck": {"name": "fpn", "out_channels": 64, "num_levels": 5},
        "heads": {
            "scene": {
                "type": "semantic_fpn",
                "num_classes": len(CLASSES),
                "channels": 64,
                "dropout": 0.1,
                "loss": {
                    "ce_weight": 1.0,
                    "dice_weight": 0.5,
                    "ignore_index": 255,
                    "class_weights": weights.tolist(),
                },
            }
        },
        "loss_balancing": "fixed",
        "fixed_weights": {"scene": 1.0},
    }
    cfg["data"] = {
        "input_size": [512, 896],
        "workers": 2,
        "letterbox": True,
        "terrain_classes": list(CLASSES),
        "augment": {
            "scale_range": [0.9, 1.1],
            "flip_p": 0.5,
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.1,
        },
        "datasets": [
            {
                "name": "studioa",
                "type": "studioa_partial",
                "root": str(data),
                "held_out": held_out,
                "split_train": "train",
                "split_val": "val",
                "split_test": "test",
                "supervises": ["scene"],
                "sample_ratio": 1.0,
            }
        ],
    }
    cfg["train"].update(
        epochs=40,
        batch_size=4,
        lr=2e-4,
        backbone_lr_mult=0.1,
        warmup_iters=22,
        ema=False,
        early_stop_patience=10,
        primary_metric="scene_mIoU",
        log_interval=5,
        val_interval=1,
        amp=True,
        amp_dtype="bfloat16",
        compile=False,
    )
    check_config(cfg)
    return cfg


def prepare(source: Path, out: Path, held_out: str) -> None:
    manifest = check_supervision(source)
    if manifest["folds"][held_out]["missing_train_classes"]:
        raise ValueError("pilot requires positive train support for every class")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, out / "data")
    check_supervision(out / "data")
    for folder in ("src", "configs"):
        shutil.copytree(
            ROOT / folder,
            out / "snapshot" / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, out / "snapshot" / name)
    dest = out / "snapshot/tools/annotation/studioa_train.py"
    dest.parent.mkdir(parents=True)
    shutil.copyfile(__file__, dest)
    weights = out / "torch/hub/checkpoints/resnet18-f37072fd.pth"
    weights.parent.mkdir(parents=True)
    shutil.copyfile(Path.home() / ".cache/torch/hub/checkpoints" / weights.name, weights)
    config = pilot_config(out / "data", out, manifest, held_out)
    write_json(out / "config.json", config)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "diff", "HEAD", "--", "src", "configs", str(Path(__file__).relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if tracked:
        raise ValueError("commit weight-bearing changes before preparing a pilot")
    files = [p for p in out.rglob("*") if p.is_file()]
    write_json(
        out / "job.json",
        {
            "schema": "studioa.semantic-pilot.v1",
            "git_commit": git,
            "held_out": held_out,
            "counts": manifest["folds"][held_out]["counts"],
            "source_manifest_sha256": digest(source / "manifest.json"),
            "classes": CLASSES,
            "class_weights": "sqrt median/train frequency clipped 0.25..4",
            "selection": "source-camera val mIoU on retained AI pixels only",
            "evaluation": "held-out store once, after selection; no independent accuracy claim",
            "resume": "last complete epoch optimizer/scheduler; not bitwise RNG replay",
            "files": {str(p.relative_to(out)): digest(p) for p in sorted(files)},
        },
    )
    write_json(out / "status.json", {"status": "prepared", "epoch": 0})


def verify(out: Path) -> dict:
    job = json.loads((out / "job.json").read_text())
    for name, expected in job["files"].items():
        if digest(out / name) != expected:
            raise ValueError(f"frozen pilot input changed: {name}")
    return job


def run(out: Path) -> None:
    import torch
    from PIL import Image

    from syncai_hydranet.data.datasets import build_dataset
    from syncai_hydranet.engine.evaluator import evaluate
    from syncai_hydranet.engine.trainer import Trainer
    from syncai_hydranet.utils.checkpoint import load_checkpoint
    from syncai_hydranet.utils.visualize import prediction_grid, terrain_palette

    lock = (out / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    state: dict = {"status": "verifying", "epoch": 0, "pid": os.getpid()}

    def status(kind, **extra):
        state.update(status=kind, elapsed_s=time.monotonic() - started, **extra)
        write_json(out / "status.json", state)

    def stopped(_signal, _frame):
        raise InterruptedError("pilot stopped; last complete epoch can resume")

    signal.signal(signal.SIGTERM, stopped)
    trainer = None
    try:
        status("verifying")
        job = verify(out)
        job_hash = digest(out / "job.json")
        if (out / "report.json").exists():
            status("completed", report="report.json")
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing accidental CPU pilot training")
        torch.set_num_threads(4)
        cfg = json.loads((out / "config.json").read_text())

        class PilotTrainer(Trainer):
            def state_dict(self, epoch):
                result = super().state_dict(epoch)
                result["studioa_job_sha256"] = job_hash
                return result

            def record_epoch(self, epoch, metrics):
                improved = super().record_epoch(epoch, metrics)
                status(
                    "training",
                    epoch=epoch,
                    global_step=self.global_step,
                    validation_teacher_miou=metrics["scene_mIoU"],
                    best_validation_teacher_miou=self.best_metric,
                )
                return improved

        last = out / "model/last.pt"
        trainer = PilotTrainer(cfg, resuming=last.exists())
        if last.exists():
            if load_checkpoint(last).get("studioa_job_sha256") != job_hash:
                raise ValueError("foreign pilot checkpoint")
            trainer.load(str(last), resume=True)
        elif not (out / "baseline_val.json").exists():
            status("baseline_validation", gpu=torch.cuda.get_device_name(0))
            baseline = evaluate(
                trainer.model,
                trainer.val_sets,
                cfg,
                trainer.device,
                trainer.logger,
                loaders=trainer.val_loaders,
            )
            write_json(out / "baseline_val.json", baseline)
        if not (out / "training_finished.json").exists():
            status("training")
            trainer.train()
            write_json(
                out / "training_finished.json",
                {"job_sha256": job_hash, "best_sha256": digest(out / "model/best.pt")},
            )
        checkpoint = out / "model/best.pt"
        best = load_checkpoint(checkpoint)
        if best.get("studioa_job_sha256") != job_hash:
            raise ValueError("foreign best checkpoint")
        trainer.model.load_state_dict(best["model"])
        status("held_out_evaluation", best_epoch=best["epoch"])
        samples = {}
        dataset = build_dataset(
            cfg["data"]["datasets"][0], cfg["data"]["input_size"], "test", letterbox=True
        )
        test = evaluate(
            trainer.model,
            [("studioa", dataset)],
            cfg,
            trainer.device,
            trainer.logger,
            samples=samples,
        )
        write_json(out / "test.json", test)
        for head, (images, predictions, targets) in samples.items():
            grid = prediction_grid(images, predictions, targets, terrain_palette(list(CLASSES)))
            Image.fromarray(grid).save(out / f"test_{head}_preview.jpg")
        verify(out)
        report = {
            "status": "completed",
            "kind": "AI partial-semantic pilot",
            "job_sha256": job_hash,
            "git_commit": job["git_commit"],
            "counts": job["counts"],
            "best_epoch": best["epoch"],
            "last_epoch": load_checkpoint(last)["epoch"],
            "best_validation_teacher_miou": best["best_metric"],
            "test_teacher_miou": test["scene_mIoU"],
            "independent_accuracy": False,
            "deployment_ready": False,
            "outputs": {
                str(p.relative_to(out)): digest(p)
                for p in [
                    checkpoint,
                    last,
                    out / "test.json",
                    out / "baseline_val.json",
                    out / "model/metrics.jsonl",
                    *out.glob("test_*_preview.jpg"),
                ]
            },
        }
        write_json(out / "report.json", report)
        status("completed", best_epoch=best["epoch"], test_teacher_miou=test["scene_mIoU"])
    except BaseException:
        status("failed", error=traceback.format_exc())
        raise
    finally:
        if trainer is not None and trainer.tb:
            trainer.tb.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--held-out", default="Tao-Hsin", choices=("Tao-Hsin", "Taichung", "Kaohsiung")
    )
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(args.source, args.out, args.held_out)
    else:
        out = args.out.resolve()
        target = out / "snapshot/tools/annotation/studioa_train.py"
        if Path(__file__).resolve() != target:
            os.execve(
                sys.executable,
                [sys.executable, str(target), "run", "--out", str(out)],
                dict(
                    os.environ,
                    PYTHONPATH=str(out / "snapshot/src"),
                    TORCH_HOME=str(out / "torch"),
                ),
            )
        run(out)


if __name__ == "__main__":
    main()
