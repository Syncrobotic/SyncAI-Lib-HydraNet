"""A test module that imports another test module is a helper in the wrong place.

This is `test_scripts_are_not_libraries.py`'s rule for a third directory, and the cost
here is different in kind rather than in degree. Under `scripts/`, the shared code merely
sits outside the wheel. Here, **the import does not resolve at all under the command this
project documents**:

    uv run pytest -q          ->  ModuleNotFoundError: No module named 'tests'
    python -m pytest -q       ->  2049 passed

Same tree, same second. `tests/` has no `__init__.py`, so `tests` is not a package and
pytest puts `tests/` on `sys.path`, not the repository root; only the `-m` form adds the
working directory. `tests/test_track_support.py` reached `tests/test_security_events.py`
that way on 2026-08-26 (`e2dfb3d`) and the suite was red by the documented command for a
day while being reported green by the other one.

**Two spellings, both banned, and the bare one is the one that would slip through.**
`from tests.test_x import y` announces itself by failing. `from test_x import y` *works*,
because pytest already put this directory first on `sys.path` -- so it would pass CI, pass
locally, and quietly make the two modules one module with two names the day someone runs a
single file. The fix for both is the same and `tests/_posture.py` is the worked example: a
non-`test_` helper module, which every test may import and pytest does not collect.

**A hard zero, not a ratchet.** `test_scripts_are_not_libraries.py` ratchets because its
existing pairs are deliberate and its directory is in flux. This set is empty as of
2026-08-27 and there is no legitimate member of it: anything two test modules share is a
helper, and naming it `test_*.py` is what makes it unreachable.

`CI cannot be relied on to catch the next one`: `ci.yml` triggers on `pull_request` and
this project's work lands straight on `dev`, which is why `e2dfb3d` reached `dev` red.
This test runs wherever the suite runs, which is the only place that is true of.

pytest tests/test_tests_do_not_import_each_other.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _test_modules() -> set[str]:
    """Every collectable module name in this directory -- what may not be imported."""
    return {p.stem for p in HERE.glob("test_*.py")}


def _cross_imports() -> list[tuple[str, str]]:
    """Every `tests/a.py` importing a `test_*` sibling, by either spelling.

    An `ast.walk` rather than a scan of the top of the file, because both offending
    imports were *inside a function body* -- which is exactly where an import that only
    works under one runner tends to hide, since a module-level one fails at collection
    and gets noticed.
    """
    modules = _test_modules()
    found: list[tuple[str, str]] = []
    for path in sorted(HERE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                head, _, tail = name.partition(".")
                target = tail if head == "tests" else name
                if target in modules and target != path.stem:
                    found.append((path.name, name))
    return found


def test_no_test_module_imports_another():
    pairs = _cross_imports()
    assert not pairs, (
        "a test module imports another test module:\n"
        + "\n".join(f"  {a} -> {b}" for a, b in pairs)
        + "\n\nWhatever they share belongs in a helper module whose name does not start "
        "with `test_` -- `tests/_posture.py` is the pattern. The `tests.` spelling does "
        "not resolve under `uv run pytest` at all (there is no `tests/__init__.py`), and "
        "the bare spelling resolves only because pytest put this directory on `sys.path`."
    )


def test_the_scan_can_still_see_an_import():
    """A guard whose search returns nothing passes forever.

    Both banned spellings are parsed here rather than pinned to a real pair, because there
    is no real pair left and inventing one to keep a witness would be the offence itself.
    The AST forms are what the scan can lose: `ast.Import` for `import test_x` and
    `ast.ImportFrom` for `from tests.test_x import y`.
    """
    modules = _test_modules()
    assert "test_security_events" in modules, "has tests/ moved?"

    source = "def f():\n    from tests.test_security_events import posed\n    import test_analytics\n"
    seen = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            seen |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            seen.add(node.module)
    assert seen == {"tests.test_security_events", "test_analytics"}
    assert all(n.rpartition(".")[2] in modules for n in seen)
