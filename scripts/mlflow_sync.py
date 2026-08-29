#!/usr/bin/env python3
"""Mirror runs/*/{meta.json,metrics.jsonl} into a local MLflow store. Files stay the truth.

    python3 scripts/mlflow_sync.py                 # sync every run under runs/
    python3 scripts/mlflow_sync.py runs/hydranet_retail_security_b03_cw
    python3 scripts/mlflow_sync.py --serve-hint    # print the systemd/server line

Why a sync layer instead of a trainer hook: the trainer already writes meta.json and
metrics.jsonl, `hydranet-report` already reads them, and every number in docs/ cites
those files. MLflow here is an index and a UI over that record -- it must never become a
second source of truth, so nothing in training depends on it and deleting runs/mlflow/
loses nothing that cannot be rebuilt by re-running this script. The immediate reason it
exists at all is 2026-08-19's `coco10` incident: the COCO share was in every config.yaml
and still nearly produced a false contradiction, because nothing put the params of many
runs side by side unless someone thought to ask. A passive comparison surface is the fix.

Idempotent: a run is identified by its `run_dir` tag; it is re-imported only when its
metrics row count or commit changed, so cron/systemd can call this freely. The store is
sqlite under runs/mlflow/ (gitignored with everything else in runs/).

Scale note: this logs per-epoch metric rows. MLflow's sqlite backend handles the current
run count (~40 runs x <=60 epochs) without effort; if the fleet grows 100x, move the
backend, not the trainer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
STORE = REPO / "runs" / "mlflow"
EXPERIMENT = "hydranet"
# Non-numeric row fields that are per-run facts, not metrics.
ROW_TAGS = ("primary_metric", "weights")


def flatten(cfg: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            out.update(flatten(v, f"{prefix}{k}." if prefix else f"{k}."))
    elif isinstance(cfg, list):
        # datasets and class lists: index-stable, short
        for i, v in enumerate(cfg):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        key = prefix.rstrip(".")
        out[key] = str(cfg)[:490]
    return out


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sync_run(client, exp_id: str, run_dir: Path) -> str:
    meta = json.loads((run_dir / "meta.json").read_text())
    rows = read_rows(run_dir / "metrics.jsonl")
    git = meta.get("git", {}) or {}
    commit = git.get("commit") or ""
    fingerprint = f"{len(rows)}@{commit[:12]}"

    existing = client.search_runs([exp_id], filter_string=f"tags.run_dir = '{run_dir.name}'")
    if existing:
        if existing[0].data.tags.get("sync_fingerprint") == fingerprint:
            return "unchanged"
        for r in existing:  # changed: rebuild rather than diff metric histories
            client.delete_run(r.info.run_id)
        verdict = "reimported"
    else:
        verdict = "imported"

    start_ms = None
    if meta.get("started_at"):
        with contextlib.suppress(ValueError):
            start_ms = int(datetime.fromisoformat(meta["started_at"]).timestamp() * 1000)

    run = client.create_run(exp_id, run_name=run_dir.name, start_time=start_ms)
    rid = run.info.run_id
    tags = {
        "run_dir": run_dir.name,
        "sync_fingerprint": fingerprint,
        "git.commit": commit,
        "git.dirty": str(git.get("dirty")),
        "datasets": ",".join(
            str(d.get("name")) for d in (meta.get("datasets") or []) if isinstance(d, dict)
        ),
    }
    if rows:
        for t in ROW_TAGS:
            if t in rows[-1]:
                tags[t] = str(rows[-1][t])
        tags["epochs_run"] = str(rows[-1].get("epoch"))
    for k, v in tags.items():
        client.set_tag(rid, k, v)

    params = flatten(meta.get("config") or {})
    params["parameters_total"] = str(meta.get("parameters"))
    params["steps_per_epoch"] = str(meta.get("steps_per_epoch"))
    for k, v in list(params.items()):
        client.log_param(rid, k, v)

    for row in rows:
        epoch = int(row.get("epoch", 0))
        for k, v in row.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            client.log_metric(rid, k, float(v), step=epoch)

    for artifact in ("config.yaml", "meta.json"):
        p = run_dir / artifact
        if p.is_file():
            client.log_artifact(rid, str(p))
    client.set_terminated(rid, "FINISHED")
    return verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="*", help="run dirs; default: every runs/* with meta.json")
    ap.add_argument("--serve-hint", action="store_true")
    args = ap.parse_args(argv)

    uri = f"sqlite:///{STORE / 'mlflow.db'}"
    if args.serve_hint:
        print(
            f".venv/bin/mlflow server --backend-store-uri {uri} "
            f"--default-artifact-root {STORE / 'artifacts'} --host 127.0.0.1 --port 5000"
        )
        return 0

    STORE.mkdir(parents=True, exist_ok=True)
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = (
        exp.experiment_id
        if exp
        else client.create_experiment(EXPERIMENT, artifact_location=str(STORE / "artifacts"))
    )

    targets = (
        [Path(r) for r in args.runs]
        if args.runs
        else sorted(p for p in (REPO / "runs").iterdir() if (p / "meta.json").is_file())
    )
    counts: dict[str, int] = {}
    for run_dir in targets:
        if not (run_dir / "meta.json").is_file():
            print(f"{run_dir}: no meta.json, skipped", file=sys.stderr)
            continue
        verdict = sync_run(client, exp_id, run_dir)
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"{run_dir.name:50s} {verdict}")
    print(f"\n{sum(counts.values())} runs: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
