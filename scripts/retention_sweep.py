#!/usr/bin/env python3
"""Delete customer imagery past its retention age. PLAN §4.6 is the policy; this is it running.

    scripts/retention_sweep.py            # report only -- the default; deletes nothing
    scripts/retention_sweep.py --apply         # actually delete
    scripts/retention_sweep.py --json          # machine-readable, for a cron log

Until 2026-08-30 nothing in this repository said how long a frame of a customer's shop
floor is kept, and the answer was "forever". §4.6 sets the tiers; this enforces them.

---------------------------------------------------------------------------
IT REPORTS WHAT IT EXAMINED, NOT ONLY WHAT IT DELETED

**On the day it was written this sweep deletes nothing, and that is the hazard.** Every
file under the swept roots is younger than the shortest tier -- 247 clips, 379 plates,
4,404 files under `runs/`, and not one of them 30 days old, because the corpus is weeks
old. A sweep that reports "deleted 0" on a tree where nothing is due looks exactly like a
sweep whose glob matches nothing, whose root moved, or that a refusal is silently
short-circuiting. This repository has more notes about that shape than any other.

So every run reports three numbers per tier -- **examined, over age, deleted** -- and
`examined` is the one that catches a broken sweep. A tier that examines zero files has
stopped looking at anything, whatever its deletion count says, and the exit code says so.

---------------------------------------------------------------------------
THREE REFUSALS, AND EACH IS A DIFFERENT WAY TO DELETE THE WRONG THING

1. **Nothing tracked by git is ever deleted.** `assets/` holds published figures and they
   are in the history, so removing the working copy would achieve nothing except a dirty
   tree and a confusing diff. The check is `git ls-files`, not a path rule, because the
   allowlist that decides what is tracked lives in `.gitignore` and may change.
2. **Nothing outside the named roots is touched**, resolved after following symlinks. A
   root that is a link into a dataset share would otherwise let a glob walk out of the
   tree entirely.
3. **Measurements are not imagery.** `runs/` holds both, and the tiers separate them by
   suffix rather than by directory, because the layout under `runs/` is per-experiment and
   no convention has ever held across all 146 of them. Deleting the JSON would delete the
   apparatus behind every number in PLAN.

`--apply` is required to delete. The default is a report, because the first thing anyone
runs a deletion tool with is no arguments at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# PLAN §4.6's table, as data. `tests/test_retention_policy.py` holds these to the document:
# a tier here that the table does not name, or a number that disagrees with it, fails.
IMAGERY = (".jpg", ".jpeg", ".png", ".gif", ".mp4")
MEASUREMENTS = (".json", ".jsonl", ".npz", ".log", ".yaml", ".yml", ".pt", ".csv")


@dataclass(frozen=True)
class Tier:
    root: str
    days: int
    suffixes: tuple[str, ...]
    what: str


TIERS = (
    Tier("datasets/studioa_clips", 30, (), "raw store clips"),
    Tier("runs", 90, IMAGERY, "imagery derived from them"),
)

# Named so the report can say what it is deliberately not sweeping, rather than leaving a
# reader to infer it from an absence.
KEPT = (
    ("datasets/studioa_static", "static plates -- the temporal median removes every person"),
    ("runs/**" + "/*{json,jsonl,npz,log,yaml}", "measurements -- numbers, not pictures"),
    ("assets/", "published figures -- tracked, so in the history and not erasable here"),
)


@dataclass
class TierResult:
    tier: Tier
    examined: int = 0
    over_age: int = 0
    deleted: int = 0
    bytes_freed: int = 0
    refused_tracked: int = 0
    errors: list[str] = field(default_factory=list)


def _tracked_files() -> set[Path]:
    """Absolute paths of everything git tracks. Empty set outside a checkout."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return set()
    return {REPO / p for p in out.stdout.split("\0") if p}


def sweep(tier: Tier, now: float, tracked: set[Path], apply: bool) -> TierResult:
    result = TierResult(tier=tier)
    root = (REPO / tier.root).resolve()
    if not root.is_dir():
        result.errors.append(f"{tier.root} does not exist")
        return result

    cutoff = now - tier.days * 86400
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if tier.suffixes and path.suffix.lower() not in tier.suffixes:
            continue
        # Refusal 2: a symlinked subtree could otherwise put `resolve()` outside the root.
        if root not in path.resolve().parents:
            continue
        result.examined += 1
        if path.stat().st_mtime >= cutoff:
            continue
        result.over_age += 1
        # Refusal 1: tracked files are in the history; deleting the working copy is noise.
        if path in tracked:
            result.refused_tracked += 1
            continue
        if not apply:
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            result.deleted += 1
            result.bytes_freed += size
        except OSError as exc:
            result.errors.append(f"{path.relative_to(REPO)}: {exc}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--apply",
        action="store_true",
        help="actually delete; without it this reports and changes nothing",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)

    now = time.time()
    tracked = _tracked_files()
    results = [sweep(t, now, tracked, a.apply) for t in TIERS]

    # A tier that examined nothing has stopped looking, whatever it deleted. That is the
    # failure this sweep is most likely to have and least likely to show.
    blind = [r for r in results if r.examined == 0]
    errors = [e for r in results for e in r.errors]

    if a.json:
        print(
            json.dumps(
                {
                    "applied": a.apply,
                    "tiers": [
                        {
                            "root": r.tier.root,
                            "days": r.tier.days,
                            "what": r.tier.what,
                            "examined": r.examined,
                            "over_age": r.over_age,
                            "deleted": r.deleted,
                            "bytes_freed": r.bytes_freed,
                            "refused_tracked": r.refused_tracked,
                        }
                        for r in results
                    ],
                    "blind_tiers": [r.tier.root for r in blind],
                    "errors": errors,
                },
                indent=2,
            )
        )
    else:
        print(f"retention sweep ({'APPLY' if a.apply else 'report only'}) -- PLAN §4.6")
        for r in results:
            print(
                f"  {r.tier.root:<26} keep {r.tier.days:>3}d  "
                f"examined {r.examined:>6}  over age {r.over_age:>5}  "
                f"deleted {r.deleted:>5}"
                + (f"  ({r.bytes_freed / 1e9:.2f} GB)" if r.bytes_freed else "")
                + (f"  refused {r.refused_tracked} tracked" if r.refused_tracked else "")
            )
        for root, why in KEPT:
            print(f"  {root:<26} kept      {why}")
        if not a.apply and any(r.over_age for r in results):
            print("\n  --apply would delete the 'over age' files above.")

    for r in blind:
        print(f"::error::{r.tier.root} examined 0 files; the sweep is not looking at anything")
    for e in errors:
        print(f"::error::{e}")
    return 1 if blind or errors else 0


if __name__ == "__main__":
    sys.exit(main())
