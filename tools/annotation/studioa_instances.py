#!/usr/bin/env python3
"""Export reviewed partial instances and run a frozen GPU loss-wiring check."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import traceback
from pathlib import Path

from syncai_hydranet.data.studioa_instances import CLASSES, check_instances, export_instances
from syncai_hydranet.data.studioa_review import digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def prepare(data: Path, out: Path) -> None:
    check_instances(data)
    dirty = subprocess.run(
        ["git", "diff", "HEAD", "--", "src", "tools/annotation/studioa_instances.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if dirty:
        raise ValueError("commit source changes before freezing smoke")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(data, out / "data")
    shutil.copytree(
        ROOT / "src",
        out / "snapshot/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    worker = out / "snapshot/tools/annotation/studioa_instances.py"
    worker.parent.mkdir(parents=True)
    shutil.copyfile(__file__, worker)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, out / "snapshot" / name)
    write_json(
        out / "job.json",
        {
            "kind": "partial-instance-loss-wiring-only",
            "device": "cuda",
            "input_size": [512, 896],
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "files": {
                str(p.relative_to(out)): digest(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        },
    )
    write_json(out / "status.json", {"status": "prepared"})


def run(out: Path) -> None:
    import torch

    from syncai_hydranet.data.datasets import build_dataset
    from syncai_hydranet.data.multitask import collate
    from syncai_hydranet.models.hydranet import build_model

    def stop(_signum, _frame):
        raise InterruptedError("smoke terminated")

    signal.signal(signal.SIGTERM, stop)
    with (out / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "report.json").exists():
            raise ValueError("completed smoke cannot be overwritten")
        try:
            write_json(out / "status.json", {"status": "running"})
            job = json.loads((out / "job.json").read_text())
            for name, expected in job["files"].items():
                if digest(out / name) != expected:
                    raise ValueError(f"frozen smoke input changed: {name}")
            if not torch.cuda.is_available():
                raise RuntimeError("GPU smoke requires CUDA")
            torch.manual_seed(42)
            torch.set_num_threads(4)
            manifest = check_instances(out / "data")
            cfg = {
                "model": {
                    "backbone": {"name": "resnet18", "pretrained": False},
                    "neck": {"name": "fpn", "out_channels": 32, "num_levels": 5},
                    "heads": {
                        "detection": {
                            "type": "fcos",
                            "num_classes": len(CLASSES),
                            "classes": list(CLASSES),
                            "channels": 32,
                            "num_convs": 2,
                        }
                    },
                    "loss_balancing": "fixed",
                    "fixed_weights": {"detection": 1.0},
                }
            }
            model = build_model(cfg).cuda().train()
            det = model._detection()
            assert det is not None
            dataset = build_dataset(
                {
                    "type": "studioa_instances",
                    "root": str(out / "data"),
                    "held_out": manifest["held_out"],
                    "supervises": ["detection"],
                    "split_train": "train",
                },
                job["input_size"],
                "train",
                letterbox=True,
                augment={
                    "scale_range": [1.0, 1.0],
                    "flip_p": 0.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "saturation": 0.0,
                },
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            classifier = dict(model.named_parameters())["det_head.cls_pred.weight"]
            before = classifier.detach().clone()
            steps = []
            for offset in range(0, len(dataset), 2):
                batch = collate(
                    [dataset[i] for i in range(offset, min(offset + 2, len(dataset)))]
                )
                targets: dict = {
                    k: [v.cuda() for v in values] if isinstance(values, list) else values.cuda()
                    for k, values in batch["targets"].items()
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(batch["image"].cuda())
                    for logits in outputs["det_cls"]:
                        logits.retain_grad()
                    loss, logs = model.compute_losses(outputs, targets, ["detection"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite partial detection loss")
                loss.backward()
                if not all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                ):
                    raise ValueError("nonfinite partial detection gradient")
                points, _ = det.head._grid_points(
                    [x.shape[-2:] for x in outputs["det_cls"]], loss.device
                )
                x, y = points.long().unbind(1)
                h, w = targets["det_negative_mask"].shape[-2:]
                safe_x, safe_y = x.clamp(max=w - 1), y.clamp(max=h - 1)
                grads = torch.cat(
                    [
                        t.grad.permute(0, 2, 3, 1).reshape(
                            len(batch["image"]), -1, len(CLASSES)
                        )
                        for t in outputs["det_cls"]
                    ],
                    dim=1,
                )
                unknown_channels = 0
                for b, boxes in enumerate(targets["boxes"]):
                    known_empty = (
                        (targets["det_negative_mask"][b, safe_y, safe_x] == 1)
                        & (x < w)
                        & (y < h)
                    )
                    in_any_box = (
                        (points[:, None, 0] >= boxes[None, :, 0])
                        & (points[:, None, 0] <= boxes[None, :, 2])
                        & (points[:, None, 1] >= boxes[None, :, 1])
                        & (points[:, None, 1] <= boxes[None, :, 3])
                    ).any(dim=1)
                    unknown = (
                        (~known_empty & ~in_any_box)[:, None].expand(-1, len(CLASSES)).clone()
                    )
                    class_negative = targets.get("det_class_negative_mask")
                    if class_negative is not None:
                        reviewed = (class_negative[b, :, safe_y, safe_x] == 1).T
                        reviewed &= ((x < w) & (y < h))[:, None]
                        unknown &= ~reviewed
                    if torch.count_nonzero(grads[b, unknown]):
                        raise ValueError("unreviewed region produced background gradient")
                    unknown_channels += int(unknown.sum())
                optimizer.step()
                steps.append(
                    {
                        "loss": loss.item(),
                        **{k: float(v) for k, v in logs.items()},
                        "unknown_logit_channels_checked": unknown_channels,
                    }
                )
            delta = float((classifier.detach() - before).norm())
            if not delta > 0:
                raise ValueError("classifier was not updated")
            for name, expected in job["files"].items():
                if digest(out / name) != expected:
                    raise ValueError("frozen smoke input changed during run")
            write_json(
                out / "report.json",
                {
                    "status": "passed",
                    "kind": "wiring_check_not_trained_detector",
                    "job_sha256": digest(out / "job.json"),
                    "git_commit": job["git_commit"],
                    "frames": [f["id"] for f in dataset.frames],
                    "partition": "train",
                    "input_size": job["input_size"],
                    "classes": list(CLASSES),
                    "model_config": cfg,
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0),
                    "amp": "bfloat16",
                    "seed": 42,
                    "steps": steps,
                    "classifier_update_norm": delta,
                    "unknown_gradients_nonzero": 0,
                    "checkpoint_written": False,
                    "accuracy_claim": False,
                },
            )
            write_json(out / "status.json", {"status": "completed", "steps": len(steps)})
        except BaseException:
            write_json(
                out / "status.json", {"status": "failed", "error": traceback.format_exc()}
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "prepare", "run", "check"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        if not all((args.source, args.reviews, args.out)):
            parser.error("export requires --source --reviews --out")
        export_instances(args.source, args.reviews, args.out)
    elif args.action == "check":
        if args.data is None:
            parser.error("check requires --data")
        print(json.dumps(check_instances(args.data)["instances_by_class"]))
    elif args.action == "prepare":
        if args.data is None or args.out is None:
            parser.error("prepare requires --data --out")
        prepare(args.data, args.out)
    else:
        if args.out is None:
            parser.error("run requires --out")
        out = args.out.resolve()
        worker = out / "snapshot/tools/annotation/studioa_instances.py"
        if Path(__file__).resolve() != worker:
            os.execve(
                sys.executable,
                [sys.executable, str(worker), "run", "--out", str(out)],
                dict(os.environ, PYTHONPATH=str(out / "snapshot/src")),
            )
        run(out)


if __name__ == "__main__":
    main()
