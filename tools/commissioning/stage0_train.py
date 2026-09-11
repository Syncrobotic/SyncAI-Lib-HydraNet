#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Prepare and run a finite, frozen Stage 0 whole-store segmentation experiment.

prepare copies selected data, remaps masks using the existing taxonomy, and fixes all
six configurations before any target-store evaluation. run executes those configs in a
frozen source tree, with timeouts, atomic progress and explicit incomplete outcomes.
These are experimental candidates, not deployment bundles or human accuracy measures.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from syncai_hydranet.config import load_config
from syncai_hydranet.data.datasets import _index_pairs
from syncai_hydranet.data.label_maps import get_scheme
from syncai_hydranet.data.store_split import STORES, camera_store, fold_split, validate_fold

ROOT = Path(os.environ.get("SYNCAI_ROOT", Path(__file__).resolve().parents[2]))
SOURCES = {
    "retail_objects_batch02": "retail_surfaces_from_objects",
    "retail_objects_batch03": "retail_surfaces_from_objects",
    "site30k_v1": "site30k_to_surfaces",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def spread(pairs: list, limit: int) -> list:
    if len(pairs) <= limit:
        return pairs
    return [pairs[i * (len(pairs) - 1) // (limit - 1)] for i in range(limit)]


def freeze_pair(out: Path, image: Path, mask: Path, scheme: str, key: str) -> dict:
    dest = out / "pool" / key
    dest.mkdir(parents=True)
    target_image = dest / ("image" + image.suffix)
    digest = sha256(image)
    mask_digest = sha256(mask)
    shutil.copy2(image, target_image)
    if sha256(target_image) != digest:
        raise RuntimeError(f"image changed during copy: {image}")
    raw = np.asarray(Image.open(mask))
    if raw.ndim != 2:
        raise ValueError(f"expected integer mask: {mask}")
    mapping = get_scheme(scheme).mapping
    # ADE is a partial reading of a known 150-class taxonomy; its unrelated classes
    # deliberately become ignore. Site taxonomies must cover every authored ID.
    allowed = set(range(151)) if scheme == "ade20k_retail_surfaces" else set(mapping)
    unknown = set(np.unique(raw)) - allowed - {255}
    if unknown:
        raise ValueError(f"unmapped labels {unknown}: {mask}")
    lut = np.full(256, 255, dtype=np.uint8)
    for label, value in mapping.items():
        lut[label] = value
    mapped = lut[raw]
    target_mask = dest / "mask.png"
    Image.fromarray(mapped).save(target_mask)
    if sha256(mask) != mask_digest:
        raise RuntimeError(f"mask changed during read: {mask}")
    # Content identity also catches differently compressed copies of the same frame.
    with Image.open(target_image) as img:
        rgb = np.asarray(img.convert("RGB"))
        pixel_hash = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
    return {
        "key": key,
        "image": str(target_image.relative_to(out)),
        "mask": str(target_mask.relative_to(out)),
        "source_image": str(image),
        "source_mask": str(mask),
        "source_image_sha256": digest,
        "source_mask_sha256": mask_digest,
        "image_sha256": pixel_hash,
        "frozen_image_sha256": digest,
        "frozen_mask_sha256": sha256(target_mask),
        "source_label_map": scheme,
        "class_pixels": np.bincount(mapped[mapped != 255], minlength=6).tolist(),
    }


def link_pair(out: Path, dataset: Path, split: str, row: dict) -> None:
    for kind, key in (("images", "image"), ("annotations", "mask")):
        source = out / row[key]
        dest = dataset / kind / split / (row["key"] + source.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Links point only to the frozen pool, never the live source dataset.
        dest.symlink_to(os.path.relpath(source, dest.parent))


def prepare(out: Path, deadline: str) -> None:
    out.mkdir(parents=True, exist_ok=False)
    if datetime.fromisoformat(deadline).tzinfo is None:
        raise ValueError("deadline must include a timezone")
    snapshot = out / "snapshot"
    snapshot.mkdir()
    for name in ("src", "configs"):
        shutil.copytree(
            ROOT / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, snapshot / name)
    for name in (
        "tools/commissioning/stage0_train.py",
        "tools/commissioning/staff_store_probe.py",
        "scripts/staff_probe.py",
    ):
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in ("regnet_x_800mf-94a99ebd.pth", "resnet18-f37072fd.pth"):
        source = Path.home() / ".cache/torch/hub/checkpoints" / name
        target = out / "torch/hub/checkpoints" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError("pretrained weights changed during copy")
    rows = []
    for source, scheme in SOURCES.items():
        for split in ("train", "val", "test"):
            root = ROOT / "datasets" / source
            pairs = _index_pairs(root / "images" / split, root / "annotations" / split)
            if not pairs:
                raise ValueError(f"missing source partition: {source}/{split}")
            if source == "site30k_v1":
                groups = defaultdict(list)
                for image, mask in pairs:
                    camera, _ = camera_store(str(image))
                    date = re.search(r"__(\d{8})-", image.name)
                    if date is None:
                        raise ValueError(f"missing session date: {image}")
                    groups[camera, date[1]].append((image, mask))
                pairs = [p for key in sorted(groups) for p in spread(groups[key], 4)]
            for image, mask in pairs:
                camera, store = camera_store(str(image))
                row = freeze_pair(out, image, mask, scheme, f"site_{len(rows):05}")
                row.update(camera=camera, store=store, original_split=split, source=source)
                rows.append(row)
    # Keep every retained source in the manifest; duplicated contents cannot straddle
    # train/val/test even when source filenames or image compression differ.
    counts = {store: validate_fold(rows, store) for store in STORES}
    public = []
    ade = ROOT / "datasets/ADE20K"
    for split, limit in (("train", 512), ("val", 64)):
        pairs = _index_pairs(ade / "images" / split, ade / "annotations" / split)
        pairs.sort(key=lambda p: hashlib.sha256(str(p[0].relative_to(ade)).encode()).digest())
        if len(pairs) < limit:
            raise ValueError("public dataset subset is incomplete")
        for image, mask in pairs[:limit]:
            row = freeze_pair(
                out, image, mask, "ade20k_retail_surfaces", f"public_{len(public):05}"
            )
            row["original_split"] = split
            public.append(row)
            link_pair(out, out / "datasets/public", split, row)
    for store in STORES:
        for row in rows:
            split = fold_split(row["store"], row["original_split"], store)
            if split:
                link_pair(out, out / "datasets" / store, split, row)
    staff_source = ROOT / "datasets/staff_customer_batch01"
    for label in ("customer", "staff"):
        shutil.copytree(staff_source / label, out / "staff_data" / label)
    cfg = load_config(ROOT / "configs/hydranet_retail_surfaces.yaml")
    cfg["model"]["heads"].pop("detection")
    cfg["model"].update(loss_balancing="fixed", fixed_weights={"terrain": 1.0})
    cfg["data"].update(input_size=[512, 896], workers=4)
    cfg["train"].update(
        epochs=40,
        batch_size=24,
        warmup_iters=150,
        ema_decay=0.99,
        ema_warmup_steps=100,
        log_interval=10,
        early_stop_patience=8,
        primary_metric="terrain_mIoU/site_seg",
    )
    jobs = []
    (out / "configs").mkdir()
    for arm in ("standard", "photometric"):
        for store in STORES:
            name = f"{arm}_{store}"
            run_cfg = copy.deepcopy(dict(cfg))
            run_cfg.update(
                experiment=f"stage0_store_{name}", output_dir=str(out / "models" / name)
            )
            if arm == "photometric":
                run_cfg["data"]["augment"].update(brightness=0.6, contrast=0.5, saturation=0.4)

            def dataset(name, root, ratio):
                return {
                    "name": name,
                    "type": "seg_folder",
                    "root": str(root),
                    "split_train": "train",
                    "split_val": "val",
                    "split_test": "test",
                    "label_map": "retail_surfaces_native",
                    "sample_ratio": ratio,
                    "supervises": ["terrain"],
                }

            run_cfg["data"]["datasets"] = [
                dataset("public", out / "datasets/public", 1.0),
                dataset("site_seg", out / "datasets" / store, 2.0),
            ]
            path = out / "configs" / f"{name}.yaml"
            path.write_text(yaml.safe_dump(run_cfg, sort_keys=False))
            load_config(path)
            test_cfg = copy.deepcopy(run_cfg)
            test_cfg["data"]["datasets"] = [test_cfg["data"]["datasets"][1]]
            test_cfg["train"]["primary_metric"] = "terrain_mIoU"
            test_path = out / "configs" / f"{name}_test.yaml"
            test_path.write_text(yaml.safe_dump(test_cfg, sort_keys=False))
            load_config(test_path)
            jobs.append(
                {
                    "name": name,
                    "arm": arm,
                    "held_out": store,
                    "config": str(path),
                    "test_config": str(test_path),
                }
            )
    frozen = {}
    for folder in ("snapshot", "configs", "torch", "staff_data"):
        for path in sorted((out / folder).rglob("*")):
            if path.is_file():
                frozen[str(path.relative_to(out))] = sha256(path)
    for row in rows + public:
        frozen[row["image"]] = row["frozen_image_sha256"]
        frozen[row["mask"]] = row["frozen_mask_sha256"]
    plan = {
        "version": 1,
        "deadline": deadline,
        "jobs": jobs,
        "fold_counts": counts,
        "source_counts": dict(Counter(r["source"] for r in rows)),
        "site_rows": rows,
        "public_rows": public,
        "frozen_files": frozen,
        "dataset_links": {
            str(path.relative_to(out)): str(path.readlink())
            for path in sorted((out / "datasets").rglob("*"))
            if path.is_symlink()
        },
        "train_timeout_seconds": 2400,
        "eval_timeout_seconds": 600,
        "protocol": "Whole-store holdout; source train/val cameras remain disjoint; "
        "source test cameras excluded. ImageNet-only initialization. Seed 42. "
        "Two predeclared photometric arms; one seed is exploratory, not evidence "
        "of a stable gain. Teacher masks measure agreement, not human accuracy. "
        "No glass/door distinction or metric-scale supervision; no auto-deployment.",
    }
    write_json(out / "plan.json", plan)
    print(
        json.dumps({k: plan[k] for k in ("deadline", "fold_counts", "source_counts")}, indent=2)
    )


class Worker:
    def __init__(self, out: Path):
        self.out = out
        self.plan: dict[str, Any] = json.loads((out / "plan.json").read_text())
        self.deadline = datetime.fromisoformat(self.plan["deadline"]).timestamp()
        self.child: subprocess.Popen | None = None
        self.env = dict(
            os.environ,
            PYTHONPATH=str(out / "snapshot/src"),
            TORCH_HOME=str(out / "torch"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            CUDA_VISIBLE_DEVICES="0",
        )
        self.state: dict[str, Any] = {
            "status": "starting",
            "pid": os.getpid(),
            "deadline": self.plan["deadline"],
            "jobs": [dict(j, status="queued") for j in self.plan["jobs"]],
        }

    def save(self):
        self.state["updated_at"] = datetime.now(UTC).isoformat()
        write_json(self.out / "status.json", self.state)
        lines = [
            "# Stage 0 新店泛化背景訓練",
            "",
            "狀態：" + self.state["status"],
            "截止時間：" + self.plan["deadline"],
            "",
            "整店留出；ImageNet 初始化；來源店 test 鏡頭不參與訓練或選模。",
            "分割數字是教師標註一致性，不能當作人工準確率或空間公尺誤差。",
            "單種子對照屬探索結果；玻璃門、實測尺度及即時端到端驗收尚未完成。",
            "目前 checkout 有未提交變更；凍結副本及雜湊保留重現依據，候選不自動部署。",
            "",
            "| 組別 | 留出店 | 狀態 | test mIoU |",
            "|---|---|---|---|",
        ]
        scores = defaultdict(dict)
        for row in self.state["jobs"]:
            metric = row.get("test_mIoU")
            value = f"{metric:.4f}" if metric is not None else "未完成"
            lines.append(f"| {row['arm']} | {row['held_out']} | {row['status']} | {value} |")
            if metric is not None and row["status"] == "completed":
                scores[row["arm"]][row["held_out"]] = metric
        for arm, results in scores.items():
            if len(results) == len(STORES):
                lines += [
                    "",
                    f"{arm} 三店等權平均：{sum(results.values()) / len(STORES):.4f}；"
                    f"最差店：{min(results.values()):.4f}。",
                ]
        lines += [
            "",
            "逐類別 IoU 與像素支援量：evaluations/*/test.json。",
            "員工／顧客整店留出探針：staff_probe/report.json（若完成）。",
            "不解除既有 unknown 門檻。",
            "資料、設定、權重來源：plan.json；各步驟日誌：logs/。",
        ]
        if self.state.get("error"):
            lines += ["", "失敗原因：", "```", self.state["error"], "```"]
        temp = self.out / "REPORT.tmp"
        temp.write_text("\n".join(lines) + "\n")
        temp.replace(self.out / "REPORT.zh-TW.md")

    def verify(self):
        for name, digest in self.plan["frozen_files"].items():
            if sha256(self.out / name) != digest:
                raise RuntimeError(f"frozen input changed: {name}")
        if "dataset_links" in self.plan:
            links = {
                str(path.relative_to(self.out)): str(path.readlink())
                for path in sorted((self.out / "datasets").rglob("*"))
                if path.is_symlink()
            }
            if links != self.plan["dataset_links"]:
                raise RuntimeError("frozen dataset membership changed")

    def stop_child(self):
        child = self.child
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def step(self, command: list[str], name: str, limit: int) -> tuple[int, bool]:
        log = self.out / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        end = min(time.time() + limit, self.deadline - 60)
        self.state.update(active_step=name, active_log=str(log), step_deadline=end)
        self.save()
        timed_out = False
        with log.open("w") as stream:
            self.child = subprocess.Popen(
                command,
                cwd=self.out / "snapshot",
                env=self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.state["child_pid"] = self.child.pid
            while self.child.poll() is None:
                if time.time() >= end:
                    timed_out = True
                    self.stop_child()
                    break
                self.save()
                time.sleep(10)
            code = self.child.wait()
        self.child = None
        self.state.pop("child_pid", None)
        self.save()
        return code, timed_out

    def run(self):
        lock = (self.out / ".worker.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.out / "status.json").exists():
            raise RuntimeError(
                "worker already has a status; preserve results and use a new output"
            )
        self.save()

        def terminated(_sig, _frame):
            raise RuntimeError("worker received termination signal")

        signal.signal(signal.SIGTERM, terminated)
        try:
            self.verify()
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; refusing an accidental CPU training run")
            self.state.update(status="running", gpu=torch.cuda.get_device_name(0))
            self.save()
            for row in self.state["jobs"]:
                if time.time() > self.deadline - 900:
                    row["status"] = "skipped: budget exhausted"
                    self.save()
                    continue
                row["status"] = "training"
                code, timeout = self.step(
                    [
                        sys.executable,
                        "-m",
                        "syncai_hydranet.cli.train",
                        "--config",
                        row["config"],
                        "--allow-dirty",
                    ],
                    row["name"] + "_train",
                    self.plan["train_timeout_seconds"],
                )
                row.update(train_exit_code=code, train_timed_out=timeout)
                checkpoint = self.out / "models" / row["name"] / "best.pt"
                if not checkpoint.is_file():
                    row["status"] = "failed: no checkpoint"
                    self.save()
                    continue
                row.update(
                    status="evaluating held-out store", checkpoint_sha256=sha256(checkpoint)
                )
                dest = self.out / "evaluations" / row["name"] / "test.json"
                result, eval_timeout = self.step(
                    [
                        sys.executable,
                        "-m",
                        "syncai_hydranet.cli.evaluate",
                        "--config",
                        row["test_config"],
                        "--checkpoint",
                        str(checkpoint),
                        "--weights",
                        "ema",
                        "--split",
                        "test",
                        "--json",
                        str(dest),
                    ],
                    row["name"] + "_test",
                    self.plan["eval_timeout_seconds"],
                )
                row.update(eval_exit_code=result, eval_timed_out=eval_timeout)
                if not result and not eval_timeout and dest.is_file():
                    row["test_mIoU"] = json.loads(dest.read_text())["terrain_mIoU"]
                    row["test_sha256"] = sha256(dest)
                    row["status"] = (
                        "completed"
                        if not code and not timeout
                        else "partial training; evaluated"
                    )
                else:
                    row["status"] = "failed: held-out evaluation"
                self.save()
            if time.time() < self.deadline - 600:
                code, timeout = self.step(
                    [
                        sys.executable,
                        "tools/commissioning/staff_store_probe.py",
                        "--batch",
                        str(self.out / "staff_data"),
                        "--out",
                        str(self.out / "staff_probe"),
                    ],
                    "staff_store_probe",
                    600,
                )
                self.state["staff_probe"] = (
                    "completed" if not code and not timeout else "failed"
                )
            self.verify()
            self.state["status"] = (
                "completed; review required"
                if all(j["status"] == "completed" for j in self.state["jobs"])
                else "finished with incomplete jobs; review required"
            )
        except BaseException:
            self.stop_child()
            self.state.update(status="failed", error=traceback.format_exc())
        finally:
            self.save()
            lock.close()
        return 1 if self.state["status"] == "failed" else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "run"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--deadline", help="absolute ISO time with timezone, required for prepare")
    args = ap.parse_args()
    if args.mode == "prepare":
        if not args.deadline:
            ap.error("prepare requires --deadline")
        prepare(args.out.resolve(), args.deadline)
        return 0
    return Worker(args.out.resolve()).run()


if __name__ == "__main__":
    raise SystemExit(main())
