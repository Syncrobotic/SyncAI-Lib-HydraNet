"""`get_logger` is a process-wide singleton, and a second run must not inherit the first.

The bug this pins: `logging.getLogger(name)` returns the same object every time, so a
guard of `if logger.handlers: return logger` made the second `Trainer` in one process go
on writing into the first run's `train.log`. Its own `output_dir` got none.

It surfaced as a test that passed alone and failed after `tests/test_trainer.py` --
`test_cli_smoke.py::test_train_writes_everything_a_run_needs`, asserting on a missing
`train.log`. Order-dependence was the symptom. The defect is a run whose log went to
another run's directory, which no test was looking for and which nothing in the log
itself would show, since the lines are there and look ordinary.
"""

from __future__ import annotations

import logging

import pytest

from syncai_hydranet.utils.logger import get_logger


@pytest.fixture
def fresh():
    """A logger name nothing else in the suite touches, torn down after."""
    name = "hydranet_test_logger"
    yield name
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def _files(name: str) -> list[str]:
    return [
        h.baseFilename
        for h in logging.getLogger(name).handlers
        if isinstance(h, logging.FileHandler)
    ]


def test_a_second_run_writes_to_its_own_file(fresh, tmp_path):
    """The regression, stated as the two runs that produced it."""
    first, second = tmp_path / "run_a.log", tmp_path / "run_b.log"
    get_logger(fresh, first).info("run a")
    get_logger(fresh, second).info("run b")

    assert second.read_text(encoding="utf-8").count("run b") == 1
    assert "run b" not in first.read_text(encoding="utf-8"), (
        "the second run's lines landed in the first run's log, under its name"
    )


def test_only_one_file_handler_survives_a_move(fresh, tmp_path):
    """Replacing, not adding: keeping both writes run B into run A's log as well."""
    get_logger(fresh, tmp_path / "run_a.log")
    get_logger(fresh, tmp_path / "run_b.log")
    assert len(_files(fresh)) == 1


def test_asking_again_for_the_same_file_is_a_no_op(fresh, tmp_path):
    """A resume path calls this more than once; stacking handlers doubles every line."""
    path = tmp_path / "train.log"
    get_logger(fresh, path)
    get_logger(fresh, path)
    logger = get_logger(fresh, path)
    assert len(_files(fresh)) == 1

    logger.info("once")
    assert path.read_text(encoding="utf-8").count("once") == 1


def test_a_relative_and_absolute_path_to_one_file_are_one_handler(fresh, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    get_logger(fresh, tmp_path / "train.log")
    get_logger(fresh, "train.log")
    assert len(_files(fresh)) == 1


def test_the_console_handler_is_added_once_and_kept(fresh, tmp_path):
    streams = lambda: [  # noqa: E731
        h
        for h in logging.getLogger(fresh).handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    get_logger(fresh)
    assert len(streams()) == 1
    get_logger(fresh, tmp_path / "a.log")
    get_logger(fresh, tmp_path / "b.log")
    assert len(streams()) == 1, "a run that moves its file must not double its console output"


def test_no_file_argument_leaves_an_attached_file_alone(fresh, tmp_path):
    """`cli/evaluate.py` calls `get_logger("eval")` with no file. That must not detach one."""
    path = tmp_path / "train.log"
    get_logger(fresh, path)
    get_logger(fresh)
    assert _files(fresh) == [str(path)]


def test_the_file_gets_the_same_format_as_the_console(fresh, tmp_path):
    path = tmp_path / "train.log"
    get_logger(fresh, path).info("hello")
    line = path.read_text(encoding="utf-8").strip()
    assert line.startswith("[") and line.endswith("] hello")
