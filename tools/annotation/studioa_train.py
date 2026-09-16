#!/usr/bin/env python3
"""Freeze and run a finite StudioA partial-semantic pilot using the existing Trainer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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
DET_METRIC = "partial_det/studioa_instances/recall50_at_020"
EMPTY_FP = "partial_det/studioa_instances/empty_fp"


def detector_warmup_config(config: dict) -> dict:
    config["experiment"] = "studioa_detector_warmup"
    config["model"]["detection_only_training"] = True
    config["model"]["fixed_weights"]["detection"] = 1.0
    config["data"]["datasets"][0]["validation_only"] = True
    config["train"].update(
        epochs=60,
        lr=2e-4,
        warmup_iters=26,
        early_stop_patience=20,
        primary_metric=DET_METRIC,
        deterministic=True,
        cudnn_benchmark=False,
        tf32=False,
    )
    check_config(config)
    return config


def frozen_state(model) -> dict:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("det_head.")
    }


def assert_frozen(model, reference: dict) -> None:
    import torch

    current = frozen_state(model)
    if current.keys() != reference.keys():
        raise ValueError("frozen scene state keys changed")
    for name, value in reference.items():
        if not torch.equal(current[name], value):
            raise ValueError(f"frozen scene state changed: {name}")
    for name, module in model.named_children():
        if name != "det_head" and any(m.training for m in module.modules()):
            raise ValueError(f"frozen scene module in training mode: {name}")


def scene_signature(model, loaders, device) -> dict:
    """Hash every source-validation scene logit, not just a preview or argmax."""
    import torch

    from syncai_hydranet.engine.evaluator import model_memory_format

    was_training = model.training
    model.eval()
    signature = hashlib.sha256()
    frames = 0
    try:
        with torch.inference_mode():
            for name, loader in loaders:
                if name != "studioa":
                    continue
                for batch in loader:
                    images = (
                        batch["image"]
                        .to(device)
                        .contiguous(memory_format=model_memory_format(model))
                    )
                    logits = model(images)["scene"].contiguous().cpu()
                    signature.update(logits.numpy().tobytes())
                    frames += len(images)
    finally:
        model.train(was_training)
    if not frames:
        raise ValueError("scene invariance requires source validation images")
    return {"frames": frames, "float32_logits_sha256": signature.hexdigest()}


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


def joint_config(config: dict, instances: Path, held_out: str) -> dict:
    from syncai_hydranet.data.studioa_instances import CLASSES as DET_CLASSES

    config["experiment"] = "studioa_joint_source_pilot"
    config["model"]["backbone"]["pretrained"] = False
    config["model"]["heads"]["detection"] = {
        "type": "fcos",
        "num_classes": len(DET_CLASSES),
        "classes": list(DET_CLASSES),
        "channels": 64,
        "num_convs": 2,
    }
    config["model"]["fixed_weights"]["detection"] = 0.5
    config["data"]["datasets"][0].pop("split_test")
    config["data"]["datasets"].append(
        {
            "name": "studioa_instances",
            "type": "studioa_instances",
            "root": str(instances),
            "held_out": held_out,
            "split_train": "train",
            "split_val": "val",
            "partial_eval": "reviewed_regions_v1",
            "supervises": ["detection"],
            "sample_ratio": 1.0,
        }
    )
    config["train"].update(
        epochs=30, batch_size=2, lr=1e-4, warmup_iters=56, detection_val_interval=1
    )
    check_config(config)
    return config


def warm_start(model, checkpoint: dict, cfg: dict) -> None:
    """Transfer only a taxonomy-compatible scene model; detection starts fresh."""
    prior = checkpoint["cfg"]
    if prior["data"]["terrain_classes"] != cfg["data"]["terrain_classes"] or set(
        prior["model"]["heads"]
    ) != {"scene"}:
        raise ValueError("warm start requires a compatible scene-only checkpoint")
    result = model.load_state_dict(checkpoint["model"], strict=False)
    if result.unexpected_keys or any(
        not k.startswith("det_head.") for k in result.missing_keys
    ):
        raise ValueError(f"incompatible warm start: {result}")


def scene_comparison_config(
    config: dict, train_frames: int, updates: int, interval: int, weights: list[float]
) -> dict:
    """Equal update/selection budgets across unequal datasets; never read test images."""
    if config["train"].get("grad_accum_steps", 1) != 1:
        raise ValueError("scene comparison requires one batch per optimizer update")
    steps = train_frames // config["train"]["batch_size"]
    if (
        steps < 1
        or updates < 1
        or interval < 1
        or updates % steps
        or interval % steps
        or updates % interval
    ):
        raise ValueError("scene updates/validation interval must align with full epochs")
    if len(weights) != len(CLASSES) or not all(np.isfinite(w) and w > 0 for w in weights):
        raise ValueError("scene comparison requires positive finite class weights")
    config["experiment"] = "studioa_scene_data_comparison"
    config["model"]["heads"]["scene"]["loss"]["class_weights"] = weights
    config["train"].update(
        epochs=updates // steps,
        val_interval=interval // steps,
        early_stop_patience=0,
        deterministic=True,
        cudnn_benchmark=False,
        tf32=False,
    )
    for ds in config["data"]["datasets"]:
        ds.pop("split_test", None)
    check_config(config)
    return config


def prepare(
    source: Path,
    out: Path,
    held_out: str,
    instances: Path | None = None,
    initial_checkpoint: Path | None = None,
    detector_warmup: bool = False,
    class_negative_normalization: str = "sum",
    small_object_crop: bool = False,
    fixed_epochs: int | None = None,
    positive_classification: str = "positive_only",
    regression_normalization: str = "positive_point_mean",
    scene_comparison_updates: int | None = None,
    scene_validation_updates: int = 165,
    scene_class_weights_source: Path | None = None,
) -> None:
    from syncai_hydranet.utils.visualize import terrain_palette

    terrain_palette(list(CLASSES), len(CLASSES))
    manifest = check_supervision(source)
    if manifest["folds"][held_out]["missing_train_classes"]:
        raise ValueError("pilot requires positive train support for every class")
    if (instances is None) != (initial_checkpoint is None):
        raise ValueError("joint pilot requires both instances and initial checkpoint")
    if detector_warmup and instances is None:
        raise ValueError("detector warmup requires instances and initial checkpoint")
    if (small_object_crop or fixed_epochs is not None) and not detector_warmup:
        raise ValueError("crop/fixed-epoch comparison requires detector warmup")
    if fixed_epochs is not None and not 1 <= fixed_epochs <= 60:
        raise ValueError("fixed epochs must be between 1 and 60")
    if positive_classification not in ("positive_only", "assigned_object"):
        raise ValueError("unsupported positive classification")
    if positive_classification != "positive_only" and not detector_warmup:
        raise ValueError("positive classification comparison requires detector warmup")
    if regression_normalization not in ("positive_point_mean", "assigned_object_mean"):
        raise ValueError("unsupported regression normalization")
    if regression_normalization != "positive_point_mean" and not detector_warmup:
        raise ValueError("regression normalization comparison requires detector warmup")
    if class_negative_normalization not in ("sum", "positive_budget"):
        raise ValueError("unsupported class negative normalization")
    if class_negative_normalization != "sum" and not detector_warmup:
        raise ValueError("class negative normalization comparison requires detector warmup")
    scene_weights_manifest = None
    if scene_comparison_updates is not None:
        if instances is not None or detector_warmup or scene_class_weights_source is None:
            raise ValueError("scene comparison requires semantic-only data and weight source")
        scene_weights_manifest = check_supervision(scene_class_weights_source)
        if scene_weights_manifest["folds"][held_out]["missing_train_classes"]:
            raise ValueError("scene comparison weight source is missing train classes")
    elif scene_class_weights_source is not None:
        raise ValueError("scene weight source requires scene comparison updates")
    instance_counts = {}
    if instances is not None:
        from collections import Counter

        from syncai_hydranet.data.studioa_instances import check_instances

        im = check_instances(instances)
        if im["held_out"] != held_out or im["source_manifest_sha256"] != digest(
            source / "manifest.json"
        ):
            raise ValueError("joint instance source/fold mismatch")
        for split in ("train", "val"):
            counts: Counter = Counter()
            for frame in im["frames"]:
                if im["folds"][held_out]["assignments"][frame["id"]] == split:
                    counts.update(
                        json.loads((instances / frame["targets"]).read_text())["labels"]
                    )
            if set(counts) != set(range(len(im["classes"]))):
                raise ValueError(f"joint pilot requires all detection classes in {split}")
            instance_counts[split] = {name: counts[i] for i, name in enumerate(im["classes"])}
        assert initial_checkpoint is not None
        parent = initial_checkpoint.parent.parent
        prior_report = json.loads((parent / "report.json").read_text())
        if prior_report["status"] != "completed" or prior_report["outputs"].get(
            str(initial_checkpoint.relative_to(parent))
        ) != digest(initial_checkpoint):
            raise ValueError("initial checkpoint is not bound to a completed pilot report")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, out / "data")
    check_supervision(out / "data")
    if instances is not None:
        shutil.copytree(instances, out / "instances")
        assert initial_checkpoint is not None
        shutil.copyfile(initial_checkpoint, out / "initial.pt")
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
    if scene_weights_manifest is not None:
        weight_config = pilot_config(out / "data", out, scene_weights_manifest, held_out)
        weights = weight_config["model"]["heads"]["scene"]["loss"]["class_weights"]
        assert scene_comparison_updates is not None
        config = scene_comparison_config(
            config,
            manifest["folds"][held_out]["counts"]["train"],
            scene_comparison_updates,
            scene_validation_updates,
            weights,
        )
        write_json(out / "class_weights_source_manifest.json", scene_weights_manifest)
    if instances is not None:
        config = joint_config(config, out / "instances", held_out)
    if detector_warmup:
        config = detector_warmup_config(config)
        config["model"]["heads"]["detection"]["loss"] = {
            "class_negative_normalization": class_negative_normalization
        }
        if positive_classification != "positive_only":
            config["model"]["heads"]["detection"]["loss"]["positive_classification"] = (
                positive_classification
            )
        if small_object_crop:
            config["data"]["datasets"][1]["small_object_crop"] = True
        if regression_normalization != "positive_point_mean":
            config["model"]["heads"]["detection"]["loss"]["regression_normalization"] = (
                regression_normalization
            )
        if fixed_epochs is not None:
            config["train"].update(epochs=fixed_epochs, early_stop_patience=0)
        check_config(config)
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
            "schema": "studioa.joint-pilot.v1"
            if instances is not None
            else "studioa.semantic-pilot.v1",
            "joint": instances is not None,
            "detector_warmup": detector_warmup,
            "class_negative_normalization": class_negative_normalization,
            "small_object_crop": small_object_crop,
            "fixed_epochs": fixed_epochs,
            "positive_classification": positive_classification,
            "regression_normalization": regression_normalization,
            "scene_comparison": scene_comparison_updates is not None,
            "scene_comparison_updates": scene_comparison_updates,
            "scene_validation_updates": scene_validation_updates
            if scene_comparison_updates is not None
            else None,
            "scene_class_weights_source_sha256": digest(
                scene_class_weights_source / "manifest.json"
            )
            if scene_class_weights_source is not None
            else None,
            "instance_counts": instance_counts,
            "git_commit": git,
            "held_out": held_out,
            "counts": manifest["folds"][held_out]["counts"],
            "source_manifest_sha256": digest(source / "manifest.json"),
            "classes": CLASSES,
            "class_weights": "shared train reference; sqrt median/frequency clipped 0.25..4"
            if scene_comparison_updates is not None
            else "sqrt median/train frequency clipped 0.25..4",
            "selection": (
                "maximum reviewed-positive source-val recall at fixed score >0.20 / IoU >=0.50"
                if detector_warmup
                else "source-camera val mIoU on retained AI pixels only"
            ),
            "acceptance": (
                "strict recall improvement; no increase in reviewed-empty FP; frozen scene "
                "parameters/buffers and all source-val float32 logits byte-identical"
                if detector_warmup
                else None
            ),
            "evaluation": (
                "source val only; retained AI semantic pixels; no test evaluation"
                if scene_comparison_updates is not None
                else "source val only; partial recall and empty-region alarms; no test"
                if instances is not None
                else "held-out store once, after selection; no independent accuracy claim"
            ),
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
        warmup = job.get("detector_warmup", False)
        reference = None

        class PilotTrainer(Trainer):
            def state_dict(self, epoch):
                result = super().state_dict(epoch)
                result["studioa_job_sha256"] = job_hash
                return result

            def record_epoch(self, epoch, metrics):
                if warmup:
                    assert reference is not None
                    assert_frozen(self.model, reference)
                improved = super().record_epoch(epoch, metrics)
                status(
                    "training",
                    epoch=epoch,
                    global_step=self.global_step,
                    validation_teacher_miou=metrics["scene_mIoU"],
                    primary_metric=self.primary_metric,
                    best_primary_metric=self.best_metric,
                )
                return improved

        last = out / "model/last.pt"
        trainer = PilotTrainer(cfg, resuming=last.exists())
        if job.get("scene_comparison") and not last.exists():
            torch.save(trainer.model.state_dict(), out / "initial_state.pt")
        if warmup:
            warm_start(trainer.model, load_checkpoint(out / "initial.pt"), cfg)
            reference = frozen_state(trainer.model)
            initial_signature = scene_signature(
                trainer.model, trainer.val_loaders, trainer.device
            )
            before_path = out / "scene_before.json"
            if (
                before_path.exists()
                and json.loads(before_path.read_text()) != initial_signature
            ):
                raise ValueError("resumed scene reference differs from original run")
            write_json(before_path, initial_signature)
        if last.exists():
            if load_checkpoint(last).get("studioa_job_sha256") != job_hash:
                raise ValueError("foreign pilot checkpoint")
            trainer.load(str(last), resume=True)
            if warmup:
                assert reference is not None
                assert_frozen(trainer.model, reference)
        else:
            if job.get("joint") and not warmup:
                warm_start(trainer.model, load_checkpoint(out / "initial.pt"), cfg)
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
            if job.get("joint") or job.get("scene_comparison"):
                # Keep epoch zero eligible: adding a head must not silently promote
                # a checkpoint whose scene agreement regressed below its warm start.
                trainer.record_epoch(0, baseline)
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
        warmup_evidence = {}
        if warmup:
            assert reference is not None
            assert_frozen(trainer.model, reference)
            final_signature = scene_signature(
                trainer.model, trainer.val_loaders, trainer.device
            )
            if final_signature != initial_signature:
                raise ValueError(
                    "source validation scene logits changed during detector warmup"
                )
            warmup_evidence = {
                "frozen_state_tensors": len(reference),
                "frozen_state_equal": True,
                "scene_before": initial_signature,
                "scene_after": final_signature,
                "scene_logits_equal": True,
            }
        joint = job.get("joint", False)
        source_validation_only = joint or job.get("scene_comparison", False)
        status(
            "source_validation" if source_validation_only else "held_out_evaluation",
            best_epoch=best["epoch"],
        )
        samples = {}
        if source_validation_only:
            evaluation_sets = trainer.val_sets
        else:
            dataset = build_dataset(
                cfg["data"]["datasets"][0], cfg["data"]["input_size"], "test", letterbox=True
            )
            evaluation_sets = [("studioa", dataset)]
        test = evaluate(
            trainer.model,
            evaluation_sets,
            cfg,
            trainer.device,
            trainer.logger,
            samples=samples,
        )
        evaluation_name = "validation" if source_validation_only else "test"
        write_json(out / f"{evaluation_name}.json", test)
        if warmup:
            baseline = json.loads((out / "baseline_val.json").read_text())
            warmup_evidence.update(
                baseline_recall=baseline[DET_METRIC],
                selected_recall=test[DET_METRIC],
                baseline_empty_fp=baseline[EMPTY_FP],
                selected_empty_fp=test[EMPTY_FP],
                accepted=(
                    test[DET_METRIC] > baseline[DET_METRIC]
                    and test[EMPTY_FP] <= baseline[EMPTY_FP]
                ),
            )
            write_json(out / "warmup_evidence.json", warmup_evidence)
        for head, (images, predictions, targets) in samples.items():
            grid = prediction_grid(images, predictions, targets, terrain_palette(list(CLASSES)))
            Image.fromarray(grid).save(out / f"{evaluation_name}_{head}_preview.jpg")
        verify(out)
        report = {
            "status": "completed",
            "kind": (
                "AI partial detector warmup"
                if warmup
                else "AI partial joint pilot"
                if joint
                else "AI partial-semantic pilot"
            ),
            "job_sha256": job_hash,
            "git_commit": job["git_commit"],
            "counts": job["counts"],
            "best_epoch": best["epoch"],
            "last_epoch": load_checkpoint(last)["epoch"],
            "primary_metric": trainer.primary_metric,
            "best_primary_metric": best["best_metric"],
            "best_validation_teacher_miou": test["scene_mIoU"]
            if source_validation_only
            else best["best_metric"],
            "warmup": warmup_evidence,
            "test_evaluated": not source_validation_only,
            f"{evaluation_name}_teacher_miou": test["scene_mIoU"],
            "instance_counts": job.get("instance_counts", {}),
            "independent_accuracy": False,
            "deployment_ready": False,
            "outputs": {
                str(p.relative_to(out)): digest(p)
                for p in [
                    checkpoint,
                    last,
                    out / f"{evaluation_name}.json",
                    out / "baseline_val.json",
                    out / "model/metrics.jsonl",
                    *out.glob(f"{evaluation_name}_*_preview.jpg"),
                    *out.glob("scene_before.json"),
                    *out.glob("warmup_evidence.json"),
                    *out.glob("initial_state.pt"),
                ]
            },
        }
        write_json(out / "report.json", report)
        status(
            "completed",
            best_epoch=best["epoch"],
            **{f"{evaluation_name}_teacher_miou": test["scene_mIoU"]},
        )
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
    parser.add_argument("--instances", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--detector-warmup", action="store_true")
    parser.add_argument("--small-object-crop", action="store_true")
    parser.add_argument("--fixed-epochs", type=int)
    parser.add_argument("--scene-comparison-updates", type=int)
    parser.add_argument("--scene-validation-updates", type=int, default=165)
    parser.add_argument("--scene-class-weights-source", type=Path)
    parser.add_argument(
        "--positive-classification",
        choices=("positive_only", "assigned_object"),
        default="positive_only",
    )
    parser.add_argument(
        "--class-negative-normalization", choices=("sum", "positive_budget"), default="sum"
    )
    parser.add_argument(
        "--regression-normalization",
        choices=("positive_point_mean", "assigned_object_mean"),
        default="positive_point_mean",
    )
    parser.add_argument(
        "--held-out", default="Tao-Hsin", choices=("Tao-Hsin", "Taichung", "Kaohsiung")
    )
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(
            args.source,
            args.out,
            args.held_out,
            args.instances,
            args.initial_checkpoint,
            args.detector_warmup,
            args.class_negative_normalization,
            args.small_object_crop,
            args.fixed_epochs,
            args.positive_classification,
            args.regression_normalization,
            args.scene_comparison_updates,
            args.scene_validation_updates,
            args.scene_class_weights_source,
        )
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
