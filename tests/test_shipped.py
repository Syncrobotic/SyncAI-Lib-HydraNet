"""`shipped.py`: which run the tools ship, and which checkpoint answers which question.

Untested until 2026-09-04, on the module that exists *because* this project shipped the
wrong checkpoint: `best.pt` is selected on one head's metric, and on the run before the
current one it was a 40% worse person detector than `last.pt` while every tool pointed
at it. The module's answer is two functions rather than one path, and six call sites
were consolidated into it so that promoting a run is one edit.

What is worth holding here is what a caller depends on and cannot see: that both
questions resolve to a file that exists inside the run this module names, that the
config sits beside the checkpoint rather than in `configs/` (a config in `configs/` is
what the NEXT run will use), and that the paths are absolute so a tool's working
directory cannot change which weights it loads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from syncai_hydranet import shipped


def test_the_run_is_named_absolutely_so_a_working_directory_cannot_change_it():
    """Every consumer is a script or a tool, run from wherever the operator happens to
    be. A relative run path would make the shipped weights a property of the shell."""
    assert shipped.SHIPPED_RUN.is_absolute()
    assert shipped.SHIPPED_CONFIG.is_absolute()


def test_the_config_comes_from_the_run_and_not_from_configs():
    """Stated in the module and load-bearing: `configs/<name>.yaml` is what the next run
    will use and may already have moved, so a tool reading it would preprocess with one
    config and load weights trained under another."""
    assert shipped.SHIPPED_CONFIG.parent == shipped.SHIPPED_RUN
    assert shipped.SHIPPED_CONFIG.name == "config.yaml"
    assert Path("configs") not in shipped.SHIPPED_CONFIG.parents


@pytest.mark.parametrize("resolve", [shipped.for_terrain, shipped.for_detection])
def test_both_questions_resolve_inside_the_run_they_name(resolve):
    """The two-answers doctrine: a caller names the head it is judged on and gets the
    checkpoint that run won on. Whatever they resolve to must live in the shipped run --
    a checkpoint from a different directory is the six-copies-of-a-string bug returning.
    """
    ckpt = resolve()
    assert ckpt.is_absolute()
    assert ckpt.parent == shipped.SHIPPED_RUN
    assert ckpt.suffix == ".pt"


@pytest.mark.skipif(
    not shipped.SHIPPED_RUN.is_dir(),
    reason="runs/ is gitignored; this asserts against the real run when it is present",
)
def test_the_run_on_disk_actually_holds_what_this_module_promises():
    """The half a path check cannot make. Skipped on a clean checkout by design -- but
    on the box that trains and renders, a promoted run whose files are missing is
    exactly the failure this module was written after."""
    assert shipped.SHIPPED_CONFIG.is_file(), f"{shipped.SHIPPED_CONFIG} is missing"
    for resolve in (shipped.for_terrain, shipped.for_detection):
        assert resolve().is_file(), f"{resolve.__name__}() -> {resolve()} is missing"
