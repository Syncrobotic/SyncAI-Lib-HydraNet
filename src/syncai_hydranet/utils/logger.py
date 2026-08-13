"""Console and file logging with a compact timestamp format."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
