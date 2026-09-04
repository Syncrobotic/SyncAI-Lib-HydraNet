#!/usr/bin/env bash
# Hold TWO coverage floors against ONE test run: what ships, and the dev-side trees.
#
#   scripts/coverage_ratchet.sh          # runs the suite, then both floors
#   COV_SKIP_RUN=1 scripts/coverage_ratchet.sh   # reuse an existing .coverage
#
# **Why two numbers rather than one.** `--cov-fail-under` takes a single figure, and the
# combined one would be meaningless here: `src/` is 87% over 10,029 statements and
# `scripts/` + `tools/` are 9% over 7,453, so a combined floor is dominated by whichever
# tree grows faster and says nothing about either. Worse, it fails in the direction nobody
# is watching -- adding an untested script would lower the combined number and the fix
# would be to lower the floor.
#
# **Why the dev-side trees get a floor at all.** They are 7,453 statements, more than half
# the Python in this repository, and until 2026-09-04 nothing measured them. Most of it is
# CLI entry points that need artefacts a clean checkout does not have, so 9% is the honest
# reading and not a target -- what the ratchet buys is that it cannot quietly become 4%
# as new scripts land untested. Same discipline as `scripts/ty_ratchet.sh`: a bound that
# may fall and must not rise.
#
# One pytest run feeds both. Running the suite twice to measure two subsets would cost
# ~80 s per matrix row for a number `coverage report --include` already has.
set -euo pipefail

SRC_FLOOR="${COV_SRC_FLOOR:-85}"
# 7, against 9% measured on a full box on 2026-09-04, and the gap is deliberate for the
# reason `ci-promote.yml` gives about the src floor: the number CI reads is not the number
# a laptop reads. Roughly a dozen tests here guard on `runs/` and `datasets/`, which are
# gitignored, so on a clean checkout they skip and whatever `scripts/` coverage they
# carried goes with them. Two points of headroom rather than a guess at how much.
#
# **Tighten this to CI's own figure once CI has printed one** -- it prints on every run,
# in the line above the error. A floor far below the true value is decoration, which is
# the argument `ci.yml` makes about its own 80 -> 83 move.
DEV_FLOOR="${COV_DEV_FLOOR:-7}"

if [ -z "${COV_SKIP_RUN:-}" ]; then
  uv run pytest -q --cov=src --cov=scripts --cov=tools --cov-report=term-missing
fi

report() {  # $1 = include glob, $2 = floor, $3 = label
  local out
  out=$(uv run coverage report --include="$1" 2>&1) || {
    echo "::error::coverage report failed for $3; the percentage below is not meaningful"
    printf '%s\n' "$out"
    return 1
  }
  local pct
  pct=$(printf '%s\n' "$out" | awk '/^TOTAL/ {gsub("%","",$NF); print $NF}')
  # An empty percentage means the include matched nothing -- a moved directory, a typo --
  # which `awk` would otherwise turn into a silent pass at "0 >= 0".
  if [ -z "$pct" ]; then
    echo "::error::$3: no TOTAL line; the include pattern '$1' matched no files"
    printf '%s\n' "$out"
    return 1
  fi
  echo "$3 coverage: ${pct}% (floor ${2}%)"
  if [ "$pct" -lt "$2" ]; then
    echo "::error::$3 coverage fell to ${pct}% (floor ${2}%)"
    return 1
  fi
}

rc=0
report "src/*" "$SRC_FLOOR" "src/" || rc=1
report "scripts/*,tools/*" "$DEV_FLOOR" "scripts/ + tools/" || rc=1
exit "$rc"
