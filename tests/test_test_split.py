"""The held-out split has to stay held out.

`best.pt` is selected on val, which makes val circular as a reported number. A test split
fixes that only if an image's membership never changes -- otherwise an image can migrate
from test into val between runs and quietly contaminate the one number that is supposed to
be uncontaminated. That is why assignment hashes the filename instead of slicing a sorted
list.

pytest tests/test_test_split.py -v
"""

from syncai_hydranet.cli.prepare_ade20k import _is_test

NAMES = [f"ADE_val_{i:08d}" for i in range(1, 2001)]


def _members(names, fraction):
    return {n for n in names if _is_test(n, fraction)}


# ------------------------------------------------------- disabled by default


def test_zero_fraction_assigns_nothing():
    assert _members(NAMES, 0.0) == set()


def test_negative_fraction_assigns_nothing():
    assert _members(NAMES, -0.5) == set()


# ------------------------------------------------------- stability


def test_assignment_is_deterministic():
    """Pinned to values computed once, not to a same-process re-run.

    `_members(x) == _members(x)` held even for the builtin salted `hash()`, which is
    stable within one process and different across processes -- exactly the migration
    this file exists to prevent. The golden set below was computed from the sha1
    implementation on 2026-09-02; if membership ever changes across versions or
    processes, this is the test that says so.
    """
    first_ten = [f"ADE_val_{i:08d}" for i in range(1, 11)]
    assert _members(first_ten, 0.5) == {
        "ADE_val_00000003",
        "ADE_val_00000007",
        "ADE_val_00000008",
        "ADE_val_00000009",
    }


def test_assignment_does_not_depend_on_iteration_order():
    assert _members(NAMES, 0.5) == _members(list(reversed(NAMES)), 0.5)


def test_adding_images_does_not_reassign_existing_ones():
    """The property that makes the split trustworthy across dataset revisions."""
    before = _members(NAMES, 0.5)
    grown = NAMES + [f"ADE_val_{i:08d}" for i in range(2001, 2500)]
    after = _members(grown, 0.5)
    assert before == {n for n in after if n in set(NAMES)}


def test_raising_the_fraction_only_adds():
    """A larger test split must be a superset, never a reshuffle."""
    assert _members(NAMES, 0.3) <= _members(NAMES, 0.6)


# ------------------------------------------------------- proportion


def test_fraction_is_roughly_honoured():
    n = len(_members(NAMES, 0.5))
    assert 0.4 < n / len(NAMES) < 0.6, f"got {n}/{len(NAMES)}"


def test_full_fraction_takes_everything():
    assert _members(NAMES, 1.0) == set(NAMES)
