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
import os
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


RUN_LOCK = ".run.lock"


def _lock_holder(lock: Path) -> int | None:
    """The pid recorded in ``lock`` if that process is still alive, else None.

    ``os.kill(pid, 0)`` is the liveness test, which is exact for this deployment -- one
    box, one user, runs started by that user's systemd units -- and would need replacing
    if runs ever moved to a shared filesystem. A recycled pid reads as alive; the cost of
    that is one unnecessary timestamped sibling, which is the safe direction to be wrong
    in.
    """
    try:
        pid = int(json.loads(lock.read_text())["pid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None  # unreadable or truncated: treat it as abandoned
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid  # alive and owned by someone else, which is still alive
    return pid


def _claim(out_dir: Path) -> bool:
    """Create ``out_dir`` and take its lock. False if a live run already holds it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / RUN_LOCK
    payload = json.dumps(
        {"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    )
    for _ in range(2):
        try:
            with lock.open("x", encoding="utf-8") as fh:
                fh.write(payload)
            return True
        except FileExistsError:
            if _lock_holder(lock) is not None:
                return False
            # The holder is gone. Drop the stale lock and try once more; if another
            # process is doing the same thing, one of us wins the O_EXCL and the other
            # gets a sibling, which is the outcome this function exists to produce.
            lock.unlink(missing_ok=True)
    return False


def resolve_out_dir(out_dir: Path, resuming: bool = False) -> Path:
    """Never write a second run on top of a first one.

    Overwriting ``best.pt`` and mixing two runs' TensorBoard events into one directory
    is silent and unrecoverable, so a fresh run into an occupied directory gets a
    timestamped sibling instead.

    **Occupancy used to be decided by artefacts, and artefacts arrive late.** The test
    was ``any(*.pt) or meta.json``, and ``meta.json`` is written at the *end* of
    ``Trainer.__init__`` -- after the model is built, after the datasets are scanned,
    after every split is fingerprinted. A second run starting anywhere inside that window
    saw a directory that existed and was not yet occupied, and was handed it. Both runs
    then shared ``train.log``, ``tb/``, ``metrics.jsonl``, ``last.pt`` and ``best.pt``,
    which is precisely the outcome the docstring above promises cannot happen. Several
    sessions share this checkout and one GPU, so two runs starting minutes apart is the
    normal case rather than the pathological one.

    So the claim is now made at the moment of the decision, with an exclusive create.
    Artefacts still count -- a finished run's directory is occupied whether or not anyone
    holds its lock -- and a lock whose process is gone is treated as abandoned, so a
    killed run does not poison its own directory forever. Nothing releases the lock on
    the way out: a preempted run never reaches a release, so liveness has to be the test
    that works, and if liveness is the test then a release is decoration.
    """
    out_dir = Path(out_dir)
    if resuming:
        # A resume owns the directory by definition; the lock is refreshed so a
        # concurrent fresh run into the same path is sent to a sibling.
        (out_dir / RUN_LOCK).unlink(missing_ok=True)
        _claim(out_dir)
        return out_dir

    finished = out_dir.exists() and (
        any(out_dir.glob("*.pt")) or (out_dir / "meta.json").exists()
    )
    if not finished and _claim(out_dir):
        return out_dir

    stamp = time.strftime("%Y%m%d-%H%M%S")
    for suffix in ("", *(f"-{i}" for i in range(2, 100))):
        sibling = out_dir.with_name(f"{out_dir.name}-{stamp}{suffix}")
        if not sibling.exists() and _claim(sibling):
            return sibling
    raise RuntimeError(
        f"could not claim an output directory beside {out_dir}: 99 siblings for "
        f"{stamp} are already taken. Something is starting runs in a loop."
    )


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
# report's 0.02 regression threshold means the same thing for it. `pose_PCK@0.2h` joins
# them for the same reason and is scaled the same way.
HEAD_METRICS = (
    "terrain_mIoU",
    "traversability_mIoU",
    "detection_mAP",
    "depth_delta1",
    "pose_PCK@0.2h",
)


def head_metric_keys(rows: list[dict]) -> list[str]:
    """Every head-level key present in `rows`, qualified by dataset or not.

    These are names, not keys. `detection_mAP` is what a run with ONE detection val set
    writes; a run with two or more writes `detection_mAP/<dataset>` for each and no
    unqualified key at all, because `evaluator._det_metrics` sets its suffix from
    `len(det_results)`. Matching the bare name therefore matched nothing on every
    multi-detection-dataset run this project has, which is all of the retail ones:
    `hydranet_retail_pose02` and `...b03_cw_xl-20260825-162131` both have a
    `selection.json` whose `heads` block holds `terrain_mIoU` alone, while the second
    one's `best.pt` scores 0.1447 on `detection_mAP/coco_person` against the 0.2022 its
    `last.pt` reaches. A 40% worse person detector is exactly what this report exists to
    print, and it was found by hand on 2026-08-31 instead.

    The same blindness misreports the head that *is* shown. That run's `selection.json`
    names `terrain_mIoU/site_seg03` as its primary metric and records 0.7030 beside it,
    which is its best plain `terrain_mIoU`; its `site_seg03` best is 0.6681. The two keys
    now appear under their own names, so neither is read as the other.

    Split on the first `/` rather than testing a prefix: `detection_mAP50/coco_person`
    and `terrain_mIoU_classes` both begin with a head metric's name and neither is one --
    the first is a second IoU threshold on the same head, the second is a list of class
    names that `float()` would refuse.
    """
    present = {k for r in rows for k in r}
    return [k for m in HEAD_METRICS for k in sorted(present) if k.partition("/")[0] == m]


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

    for m in head_metric_keys(rows):
        have = [r for r in rows if m in r]
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
