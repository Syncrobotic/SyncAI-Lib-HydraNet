#!/usr/bin/env bash
# Hold TWO coverage floors against ONE test run: what ships, and the dev-side trees.
#
#   scripts/coverage_ratchet.sh          # runs the suite, then both floors
#   COV_SKIP_RUN=1 scripts/coverage_ratchet.sh   # reuse an existing .coverage
#
# Separate floors prevent high runtime coverage from hiding untested CLI code.
# Count nested tools without __init__.py too (include_namespace_packages in
# pyproject.toml), including files never imported by the test suite.
#
# The dev floor began at 7%, rose to 9%, then 11% in September 2026. Those
# historical measurements used a smaller tree and are not current coverage.
# Keep both floors fixed while measuring the full current tree. If an environment
# falls below a floor, investigate missing tests instead of lowering the threshold.
# One test run supplies both reports; precise coverage enforcement avoids rounding
# a subthreshold result up to a pass.
set -euo pipefail

SRC_FLOOR="${COV_SRC_FLOOR:-85}"
DEV_FLOOR="${COV_DEV_FLOOR:-11}"

if [ -z "${COV_SKIP_RUN:-}" ]; then
  uv run --frozen pytest -q --cov=src --cov=scripts --cov=tools --cov-report=term-missing
fi

report() {  # $1 = include glob, $2 = floor, $3 = label
  local out
  # Let coverage enforce its own threshold with two decimal places. Parsing its
  # default integer display admitted 84.6% through an 85% floor. Missing/empty
  # data and tool errors must also fail, rather than being interpreted as zero.
  out=$(uv run --frozen coverage report --include="$1" --precision=2 \
    --format=total --fail-under="$2" 2>&1) || {
    echo "::error::$3 coverage gate failed (floor ${2}%)"
    printf '%s\n' "$out"
    return 1
  }
  echo "$3 coverage: ${out}% (floor ${2}%)"
}

rc=0
report "src/*" "$SRC_FLOOR" "src/" || rc=1
report "scripts/*,tools/*" "$DEV_FLOOR" "scripts/ + tools/" || rc=1
exit "$rc"
