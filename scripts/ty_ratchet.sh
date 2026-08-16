#!/usr/bin/env bash
# Hold `ty` on src/ to a baseline count that may fall and never rise.
#
#   scripts/ty_ratchet.sh            # check against BASELINE below
#   TY_BASELINE=12 scripts/ty_ratchet.sh
#
# A ratchet rather than a pass/fail gate. What is left in src/ is almost all torch and
# pycocotools stub gaps rather than anything this repo wrote, and turning a gate red on
# all of it at once is how a checker gets deleted a week later. Blocking the *increase*
# is the part worth having: new code type-checks, old debt is paid down as those files
# are touched anyway.
#
# Lives in a script rather than inline in the workflows because two of them run it
# (ci.yml at feat→dev, ci-promote.yml on the combined dev→stage state). A copy in each
# would put the baseline in two places, and the copy nobody edits is the one that
# silently stops meaning anything.
#
# Scoped to src/: that is what ships to the robot. scripts/ and tests/ carry another
# ~23 and are dev-side; widen when their count is worth defending.
set -euo pipefail

# Lower this when you fix things. Raising it needs a reason in the PR body.
BASELINE="${TY_BASELINE:-15}"

TARGET="${1:-src/}"
RUNNER=(uv run ty check "$TARGET" --output-format=concise)

# ty exits 0 clean, 1 with diagnostics, 2 when it could not run at all -- a broken
# [tool.ty] table, an unreadable tree. Only 0 and 1 mean the count below is real, so 2
# fails here rather than counting zero errors in output that was never produced.
# `--exit-zero` would flatten exactly that distinction and report a checker that never
# started as a clean tree; this repo has already had a [tool.ty] key that silently
# stopped it, so the difference is not hypothetical.
out=$("${RUNNER[@]}" 2>&1) && rc=0 || rc=$?
printf '%s\n' "$out"

if [ "$rc" -gt 1 ]; then
  echo "::error::ty could not run (exit $rc); the diagnostic count is not meaningful"
  exit 1
fi

count=$(printf '%s\n' "$out" | grep -c 'error\[' || true)
echo "diagnostics: $count (baseline $BASELINE)"

if [ "$count" -gt "$BASELINE" ]; then
  echo "::error::type diagnostics rose to $count (baseline $BASELINE); see the log above"
  exit 1
fi

# A fall is a notice, not a failure, on purpose. Making a PR that happens to fix an
# unrelated annotation also edit this file to stay green is exactly the friction that
# gets a check muted. The cost is that the baseline drifts above the true count until
# someone lowers it, which this asks for by name on every run that earns it.
if [ "$count" -lt "$BASELINE" ]; then
  echo "::notice::type diagnostics fell to $count -- lower BASELINE in scripts/ty_ratchet.sh"
fi
