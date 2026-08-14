"""Summarise finished runs, and compare them.

    hydranet-report runs/hydranet_indoor
    hydranet-report runs/*                       # a table across every run
    hydranet-report runs/a runs/b --diff         # what actually differed

Training writes meta.json and metrics.jsonl precisely so this question has an answer,
but reading them by hand across a dozen directories is how "which run was that again"
becomes guesswork. This reads them and says: what scored best, at which epoch, from
which commit, on which data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config import diff_config


def read_run(run_dir: str | Path) -> dict[str, Any]:
    """Load one run directory into a flat summary."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found: not a run directory")

    meta = json.loads(meta_path.read_text())
    rows: list[dict] = []
    if metrics_path.is_file():
        rows = [json.loads(x) for x in metrics_path.read_text().splitlines() if x.strip()]

    primary = rows[-1].get("primary_metric") if rows else None
    best = None
    if primary and rows:
        scored = [r for r in rows if primary in r]
        best = max(scored, key=lambda r: r[primary]) if scored else None

    git = meta.get("git", {})
    return {
        "dir": str(run_dir),
        "experiment": meta.get("experiment"),
        "started_at": meta.get("started_at"),
        "commit": (git.get("commit") or "")[:8] or None,
        "dirty": git.get("dirty"),
        "primary_metric": primary,
        "best_score": best.get(primary) if best else None,
        "best_epoch": best.get("epoch") if best else None,
        "epochs_run": rows[-1]["epoch"] if rows else 0,
        "datasets": {
            d.get("name"): d.get("train_size") for d in meta.get("datasets", []) or []
        },
        "meta": meta,
        "rows": rows,
    }


def _fmt(value: Any, width: int) -> str:
    if value is None:
        return "-".ljust(width)
    if isinstance(value, float):
        return f"{value:.4f}".ljust(width)
    return str(value)[:width].ljust(width)


def format_table(runs: list[dict]) -> str:
    """One line per run, newest scores first."""
    header = f"{'run':<28} {'commit':<9} {'epochs':<7} {'best':<9} {'@epoch':<7} metric"
    lines = [header, "-" * len(header)]
    for r in sorted(runs, key=lambda r: r["best_score"] or -1, reverse=True):
        name = Path(r["dir"]).name
        commit = (r["commit"] or "-") + ("*" if r["dirty"] else "")
        lines.append(
            f"{name[:28]:<28} {commit:<9} {_fmt(r['epochs_run'], 7)} "
            f"{_fmt(r['best_score'], 9)} {_fmt(r['best_epoch'], 7)} "
            f"{r['primary_metric'] or '-'}"
        )
    return "\n".join(lines)


def format_run(run: dict) -> str:
    """The long form for a single run."""
    meta = run["meta"]
    env = meta.get("environment", {})
    out = [
        f"run:        {run['dir']}",
        f"experiment: {run['experiment']}",
        f"started:    {run['started_at']}",
        f"code:       {run['commit']}{' (dirty)' if run['dirty'] else ''}",
        f"env:        torch {env.get('torch')}, python {env.get('python')}, "
        f"{env.get('device')}",
        f"epochs run: {run['epochs_run']}",
    ]
    if run["best_score"] is not None:
        out.append(
            f"best:       {run['primary_metric']} = {run['best_score']:.4f} "
            f"at epoch {run['best_epoch']}"
        )
    else:
        out.append("best:       no validation recorded")

    for ds in meta.get("datasets", []) or []:
        splits = ds.get("splits", {})
        parts = [
            f"{name}={info.get('images', {}).get('files', '?')} imgs"
            for name, info in splits.items()
        ]
        digest = splits.get("train", {}).get("images", {}).get("digest", "")
        out.append(f"data:       {ds.get('name')}: {', '.join(parts)}  {digest}")

    if run["rows"]:
        metric = run["primary_metric"]
        curve = [f"{r['epoch']}:{r.get(metric, float('nan')):.3f}" for r in run["rows"]]
        out.append("curve:      " + " ".join(curve))
    return "\n".join(out)


def format_diff(a: dict, b: dict) -> str:
    """Config differences between two runs, which is usually the question being asked."""
    drift = diff_config(a["meta"].get("config", {}), b["meta"].get("config", {}))
    head = f"{Path(a['dir']).name} -> {Path(b['dir']).name}"
    if not drift:
        return f"{head}\n  configs are identical"
    lines = [head]
    lines += [f"  {where}: {was} -> {now}" for where, was, now in drift]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-report", description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories, e.g. runs/*")
    ap.add_argument(
        "--diff",
        action="store_true",
        help="show the config differences between the first run and each other one",
    )
    ap.add_argument("--json", default=None, metavar="PATH", help="write the summary as JSON")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runs = []
    for path in args.runs:
        try:
            runs.append(read_run(path))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"skipping {path}: {e}")
    if not runs:
        raise SystemExit("no readable run directories")

    if len(runs) == 1:
        print(format_run(runs[0]))
    else:
        print(format_table(runs))
    if args.diff and len(runs) > 1:
        print()
        for other in runs[1:]:
            print(format_diff(runs[0], other))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = [{k: v for k, v in r.items() if k not in ("meta", "rows")} for r in runs]
        out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
