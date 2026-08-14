"""hydranet-report: reading run directories back.

pytest tests/test_report.py -v
"""

import json

import pytest

from syncai_hydranet.cli.report import (
    build_parser,
    format_diff,
    format_run,
    format_table,
    main,
    read_run,
)


def _run(root, name, *, scores, lr=2e-4, commit="a" * 40, dirty=False):
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "experiment": name,
                "started_at": "2026-08-13T10:00:00+0800",
                "git": {"available": True, "commit": commit, "dirty": dirty},
                "environment": {"torch": "2.13.0", "python": "3.12.0", "device": "cuda"},
                "config": {"train": {"lr": lr, "epochs": len(scores)}},
                "datasets": [
                    {
                        "name": "ade20k",
                        "train_size": 5998,
                        "splits": {"train": {"images": {"files": 5998, "digest": "sha256:xy"}}},
                    }
                ],
            }
        )
    )
    (d / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "epoch": i + 1,
                    "primary_metric": "traversability_mIoU",
                    "traversability_mIoU": s,
                }
            )
            for i, s in enumerate(scores)
        )
        + "\n"
    )
    return d


# ------------------------------------------------------------------ reading


def test_best_epoch_is_the_best_one_not_the_last(tmp_path):
    """Training keeps best.pt from the peak, so the report has to agree with it."""
    run = read_run(_run(tmp_path, "a", scores=[0.1, 0.42, 0.30]))
    assert run["best_score"] == 0.42
    assert run["best_epoch"] == 2
    assert run["epochs_run"] == 3


def test_reads_provenance(tmp_path):
    run = read_run(_run(tmp_path, "a", scores=[0.1], commit="b" * 40, dirty=True))
    assert run["commit"] == "bbbbbbbb"
    assert run["dirty"] is True
    assert run["datasets"] == {"ade20k": 5998}


def test_a_run_with_no_validation_yet(tmp_path):
    d = tmp_path / "fresh"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"experiment": "fresh"}))
    run = read_run(d)
    assert run["best_score"] is None and run["epochs_run"] == 0
    assert "no validation recorded" in format_run(run)


def test_a_directory_that_is_not_a_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a run directory"):
        read_run(tmp_path)


# ---------------------------------------------------------------- rendering


def test_table_ranks_by_best_score(tmp_path):
    runs = [
        read_run(_run(tmp_path, "worse", scores=[0.2])),
        read_run(_run(tmp_path, "better", scores=[0.9])),
    ]
    body = format_table(runs).splitlines()[2:]
    assert body[0].startswith("better")
    assert body[1].startswith("worse")


def test_dirty_runs_are_marked(tmp_path):
    runs = [read_run(_run(tmp_path, "a", scores=[0.2], dirty=True))]
    assert "*" in format_table(runs).splitlines()[2]


def test_single_run_view_includes_the_curve_and_the_data_digest(tmp_path):
    text = format_run(read_run(_run(tmp_path, "a", scores=[0.1, 0.2])))
    assert "1:0.100 2:0.200" in text
    assert "sha256:xy" in text


def test_diff_names_the_settings_that_changed(tmp_path):
    a = read_run(_run(tmp_path, "a", scores=[0.1], lr=1e-4))
    b = read_run(_run(tmp_path, "b", scores=[0.1], lr=5e-4))
    text = format_diff(a, b)
    assert "train.lr: 0.0001 -> 0.0005" in text


def test_diff_says_so_when_nothing_changed(tmp_path):
    a = read_run(_run(tmp_path, "a", scores=[0.1]))
    b = read_run(_run(tmp_path, "b", scores=[0.1]))
    assert "identical" in format_diff(a, b)


# ---------------------------------------------------------------------- cli


def test_cli_writes_json_and_skips_unreadable_dirs(tmp_path, capsys):
    _run(tmp_path, "a", scores=[0.3])
    out = tmp_path / "summary.json"
    main([str(tmp_path / "a"), str(tmp_path / "missing"), "--json", str(out)])

    captured = capsys.readouterr().out
    assert "skipping" in captured
    summary = json.loads(out.read_text())
    assert summary[0]["best_score"] == 0.3
    assert "meta" not in summary[0]  # the summary stays small


def test_cli_refuses_when_nothing_is_readable(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "nope")])


def test_parser_accepts_a_glob_expansion():
    args = build_parser().parse_args(["runs/a", "runs/b", "--diff"])
    assert args.runs == ["runs/a", "runs/b"] and args.diff
