"""Console and file logging with a compact timestamp format."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """A named logger writing to stdout, and to ``log_file`` when one is given.

    **A second call with a different file moves the file handler**, and that is the
    whole reason this is not four lines. `logging.getLogger(name)` is a process-wide
    singleton, so the previous version's `if logger.handlers: return logger` meant the
    *second* run in one process silently kept writing into the *first* run's file: its
    own `output_dir` ended up with no `train.log` at all, and its lines were appended to
    a finished run's log under that run's name.

    Nothing in production hit it, because `hydranet-train` builds one `Trainer` and
    exits. It is reachable from anything that trains twice in one process -- a sweep
    driver, a notebook, and the test suite, where it shows up as
    `tests/test_cli_smoke.py::test_train_writes_everything_a_run_needs` failing on a
    missing `train.log` when it runs after `tests/test_trainer.py` and passing alone.
    A test that passes or fails on what ran before it is the symptom; a run whose log
    went into another run's directory is the defect.

    Replacing rather than adding, because a second run means the first one is over.
    Keeping both would put run B's lines in run A's log, which is the same failure
    written more slowly.

    Asking for the file already attached is a no-op, so the repeated calls a resume path
    makes do not stack duplicate handlers and double every line.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(_FORMATTER)
        logger.addHandler(stream)
    if log_file is None:
        return logger

    target = Path(log_file).resolve()
    existing = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    if any(Path(h.baseFilename) == target for h in existing):
        return logger
    for handler in existing:
        logger.removeHandler(handler)
        handler.close()
    file = logging.FileHandler(target, encoding="utf-8")
    file.setFormatter(_FORMATTER)
    logger.addHandler(file)
    return logger


_FORMATTER = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
