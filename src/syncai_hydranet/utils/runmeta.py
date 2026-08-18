"""Provenance for a training run.

A checkpoint six months from now is only useful if you can still answer: which commit
produced it, on what data, with which config, and what did it score. TensorBoard event
files answer none of those, so every run also writes plain files next to them:

    runs/<experiment>/
        meta.json         git commit, dirty flag, environment, resolved config
        config.yaml       the config after --set overrides, ready to re-run
        uncommitted.patch only when the working tree was dirty
        metrics.jsonl     one line per validation, machine-readable
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

MAX_PATCH_BYTES = 2_000_000


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # Only the trailing newline: --porcelain encodes the status in the first two
    # columns, so stripping leading whitespace would eat part of the first filename.
    return out.stdout.rstrip("\n")


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    """Commit, branch and dirtiness of the tree the run was launched from."""
    cwd = Path(cwd or Path.cwd())
    commit = _git("rev-parse", "HEAD", cwd=cwd)
    if commit is None:
        return {"available": False}
    status = _git("status", "--porcelain", cwd=cwd) or ""
    return {
        "available": True,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd),
        "dirty": bool(status),
        # porcelain v1: "XY <path>", so the path starts at column 3.
        "dirty_files": sorted(line[3:] for line in status.splitlines() if line)[:50],
    }


def environment(device: Any = None) -> dict[str, Any]:
    try:
        from importlib.metadata import version

        pkg = version("syncai-hydranet")
    except Exception:
        pkg = None
    return {
        "package_version": pkg,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device) if device is not None else None,
    }


def resolve_out_dir(out_dir: Path, resuming: bool = False) -> Path:
    """Never write a second run on top of a first one.

    Overwriting ``best.pt`` and mixing two runs' TensorBoard events into one directory
    is silent and unrecoverable, so a fresh run into an occupied directory gets a
    timestamped sibling instead.
    """
    out_dir = Path(out_dir)
    if resuming or not out_dir.exists():
        return out_dir
    occupied = any(out_dir.glob("*.pt")) or (out_dir / "meta.json").exists()
    if not occupied:
        return out_dir
    return out_dir.with_name(f"{out_dir.name}-{time.strftime('%Y%m%d-%H%M%S')}")


def write_run_meta(out_dir: Path, cfg: dict, device: Any = None, **extra: Any) -> dict:
    """Write ``meta.json``, the resolved ``config.yaml`` and any uncommitted diff."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    git = git_state()
    meta = {
        "experiment": cfg.get("experiment"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git,
        "environment": environment(device),
        "config": json.loads(json.dumps(dict(cfg), default=str)),
        **extra,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(
            json.loads(json.dumps(dict(cfg), default=str)), sort_keys=False, allow_unicode=True
        )
    )
    if git.get("dirty"):
        # A commit hash does not identify the code that ran when the tree was dirty.
        patch = _git("diff", "HEAD", cwd=Path.cwd()) or ""
        if 0 < len(patch) <= MAX_PATCH_BYTES:
            (out_dir / "uncommitted.patch").write_text(patch)
    return meta


def append_metrics(out_dir: Path, record: dict) -> None:
    """Append one validation result to ``metrics.jsonl``."""
    line = json.dumps(record, ensure_ascii=False, default=float)
    with (Path(out_dir) / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# The metrics a multi-head run could reasonably have been selected on. Head-level
# aggregates only: per-class IoUs are far noisier epoch to epoch -- a class present in 22
# of 285 val images swings on a handful of frames -- and warning on each would train
# people to skim past the whole block.
# `depth_delta1` joins these because a depth head is otherwise invisible to the
# selection report: a checkpoint chosen on segmentation could have given up half its
# depth accuracy and nothing would say so. Scaled like mIoU (0-1, higher better), so the
# report's 0.02 regression threshold means the same thing for it.
HEAD_METRICS = ("terrain_mIoU", "traversability_mIoU", "detection_mAP", "depth_delta1")


def selection_report(rows: list[dict], primary_metric: str) -> tuple[dict, list[str]]:
    """What the chosen checkpoint gave up on the heads nobody was selecting for.

    `best.pt` is picked by one scalar. A three-head model therefore has two heads whose
    quality nothing checks, and the cost is not hypothetical: measured across this
    project's own runs, selecting on `detection_mAP` gives away 0.01-0.08 terrain mIoU,
    and selecting on a segmentation metric has produced checkpoints whose detection mAP
    is 0.0024 against the 0.2007 the same run reached later.

    The worst case found was `hydranet_retail_objects_site_balanced`, whose
    `primary_metric` was a single rare class's IoU. It peaked at **epoch 2** on noise, so
    `best.pt` is an epoch-2 checkpoint of a 39-epoch run -- terrain and detection both
    near their starting values -- and nothing said so.

    Returns `(per-metric summary, warnings)`. It does not change the selection: which
    metric should pick a checkpoint is a research decision, and a silent change would
    invalidate comparison with every run already on disk. It makes the cost visible.
    """
    scored = [r for r in rows if primary_metric in r]
    if not scored:
        return {}, []
    chosen = max(scored, key=lambda r: r[primary_metric])
    summary: dict[str, dict] = {}
    warnings: list[str] = []

    for m in HEAD_METRICS:
        have = [r for r in rows if m in r]
        if not have:
            continue
        top = max(have, key=lambda r: r[m])
        at_sel = chosen.get(m)
        summary[m] = {
            "best": float(top[m]),
            "best_epoch": int(top["epoch"]),
            "at_selected": None if at_sel is None else float(at_sel),
            "selected_epoch": int(chosen["epoch"]),
        }
        if m == primary_metric or at_sel is None:
            continue
        gap = float(top[m]) - float(at_sel)
        if gap > 0.02:
            warnings.append(
                f"{m} is {at_sel:.4f} in the selected checkpoint (epoch "
                f"{chosen['epoch']}) but reached {top[m]:.4f} at epoch {top['epoch']}: "
                f"selecting on {primary_metric} gave up {gap:.4f} on this head"
            )

    # A best epoch near the start of a long run is the signature of selection latching
    # onto early noise, which is what a low-support per-class primary metric produces.
    if len(scored) >= 10:
        rank = sorted(r["epoch"] for r in scored).index(chosen["epoch"])
        if rank < 0.2 * len(scored):
            warnings.append(
                f"the selected checkpoint is epoch {chosen['epoch']}, in the first fifth "
                f"of {len(scored)} validations. A {primary_metric} that peaks that early "
                "is usually noise rather than a model, and best.pt then holds weights "
                "from before the other heads had trained"
            )
    return summary, warnings
