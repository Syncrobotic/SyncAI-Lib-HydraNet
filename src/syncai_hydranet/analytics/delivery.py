"""What may appear in a report that leaves this building.

`site_events.py`, `retail_flow.py` and `mine_fall_candidates.py` each write a JSON report
and each opened it with ``{"settings": vars(args)}``. That block is worth having -- every
row below it is a threshold crossing, and the thresholds *are* the argument list -- but
``vars(args)`` is the whole namespace, and four of its entries are filesystem paths:

    "clips": ["/home/paul/SyncAI-Lib-HydraNet/datasets/studioa_clips/Taichung-cam01/..."]
    "checkpoint": "/home/paul/SyncAI-Lib-HydraNet/runs/hydranet_retail_security_b03/best.pt"

which names an operator, a home directory, a repository checkout and a dataset root. None
of that is needed to read the report, and it is the kind of disclosure nobody notices,
because the file it lands in looks like output rather than like a document.

**The last two components survive: enough to identify, not enough to locate.** A basename
alone was tried first and it is worse than the leak in one specific way -- `best.pt` names
nothing, there are forty of them under `runs/`, and a report that cannot say which weights
produced it is not auditable. Two components keep `hydranet_retail_security_b03/best.pt`
and `Taichung-cam01/archive_20260816-113012_....mp4`, both of which the report already
states elsewhere as `camera` and `session`, and drop everything above them.

**Reduction is by value shape, not by argument name.** A list of keys to redact is a list
someone has to remember to extend, and the next `--zones-file` would be added by a person
thinking about zones rather than about disclosure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

KEEP_COMPONENTS = 2

_SEPARATORS = tuple(sep for sep in (os.sep, os.altsep) if sep)


def _looks_like_a_path(value: Any) -> bool:
    """A string carrying a separator and no whitespace.

    The whitespace test is what keeps a prose argument -- a `--note`, or a basis reading
    "5.2 m / 4.0 m" -- from being truncated by a rule meant for paths.
    """
    if not isinstance(value, str):
        return False
    return any(sep in value for sep in _SEPARATORS) and not any(c.isspace() for c in value)


def tail(value: Any) -> Any:
    """A path reduced to its last ``KEEP_COMPONENTS`` parts. Lists reduce elementwise."""
    if isinstance(value, (list, tuple)):
        return [tail(v) for v in value]
    if not isinstance(value, Path) and not _looks_like_a_path(value):
        return value
    parts = Path(value).parts
    return str(Path(*parts[-KEEP_COMPONENTS:])) if parts else str(value)


def report_settings(args: argparse.Namespace, **extra: Any) -> dict:
    """The settings block of a report: every argument, no absolute paths.

    ``extra`` appends facts the namespace does not carry -- `retail_flow` passes the
    tracker's simplifications, which bound every row as firmly as any threshold does.
    """
    return {**{k: tail(v) for k, v in vars(args).items()}, **extra}
