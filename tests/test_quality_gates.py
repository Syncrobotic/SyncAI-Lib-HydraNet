"""Quality gates must fail at their real threshold and count unexecuted tools."""

import os
import subprocess
import sys
from pathlib import Path

import coverage
import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_gate(tmp_path, hits, *, missing=False):
    for folder in ("src", "tools"):
        p = tmp_path / folder / "module.py"
        p.parent.mkdir()
        p.write_text("x = 1\n" * 1000)
    data = coverage.CoverageData(basename=str(tmp_path / ".coverage"))
    if not missing:
        data.add_lines(
            {
                str(tmp_path / "src/module.py"): list(range(1, hits + 1)),
                str(tmp_path / "tools/module.py"): list(range(1, 501)),
            }
        )
        data.write()
    # Test the shell gate against real coverage data without resolving dependencies.
    shim = tmp_path / "uv"
    shim.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "assert sys.argv[1:3] == ['run', '--frozen']\n"
        f'os.execv({sys.executable!r}, [{sys.executable!r}, "-m", *sys.argv[3:]])\n'
    )
    shim.chmod(0o755)
    return subprocess.run(
        ["bash", str(ROOT / "scripts/coverage_ratchet.sh")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "COV_SKIP_RUN": "1",
            "COV_SRC_FLOOR": "85",
            "COV_DEV_FLOOR": "11",
            "COVERAGE_FILE": str(tmp_path / ".coverage"),
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("hits,expected", [(846, 1), (850, 0)])
def test_gate_does_not_round_subthreshold_coverage_up(tmp_path, hits, expected):
    result = run_gate(tmp_path, hits)
    assert result.returncode == expected, result.stdout + result.stderr


def test_missing_coverage_is_not_a_pass(tmp_path):
    result = run_gate(tmp_path, 0, missing=True)
    assert result.returncode != 0


def test_unexecuted_nested_tools_are_in_the_coverage_denominator(tmp_path):
    tool = tmp_path / "tools/nested/never_imported.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("uncovered = 1\n")
    script = tmp_path / "tools/executed.py"
    script.write_text("covered = 1\n")
    cov = coverage.Coverage(
        config_file=str(ROOT / "pyproject.toml"),
        data_file=str(tmp_path / ".coverage"),
        source=[str(tmp_path / "tools")],
    )
    cov.start()
    exec(compile(script.read_text(), str(script), "exec"), {})
    cov.stop()
    cov.save()
    assert str(tool) in cov.get_data().measured_files()
    assert cov.analysis2(str(tool))[3] == [1]
