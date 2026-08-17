#!/usr/bin/env bash
# Hold `ty` on src/ to a baseline count that may fall and never rise.
#
#   scripts/ty_ratchet.sh            # check against BASELINE below
#   TY_BASELINE=12 scripts/ty_ratchet.sh
#
# A ratchet rather than a pass/fail gate: turning a gate red on all of the debt at once
# is how a checker gets deleted a week later. Blocking the *increase* is the part worth
# having -- new code type-checks, old debt is paid down as those files are touched.
#
# This comment used to say the debt was "almost all torch and pycocotools stub gaps
# rather than anything this repo wrote", and used that to justify a loose baseline. It
# was measurably backwards. At 17 diagnostics only 7 mentioned a torch type at all, and
# 10 of them sat in two files -- `data/transforms.py`, where AUGMENT_DEFAULTS mixed
# `tuple[float, float]` and `float` so every read of it was ambiguous, and
# `data/datasets.py`, where `LabelScheme` stores `fmt` and `mapping` as independent
# fields when the mapping's key type *is* the fmt. Fixing the first took the count to
# 11. The second is a label-decode refactor (ColorScheme/IdScheme) and wants its own
# reviewed change rather than being smuggled into a green-CI commit.
#
# The general lesson, since it cost a day: a comment that excuses debt is itself a claim,
# and nothing was checking this one.
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
#
# 11 is the measured count after `data/transforms.py` was fixed, so the ratchet actually
# ratchets: a baseline left above the real number is a ceiling, and it lets the next five
# regressions through without a word. The remaining 11 are 5 in `data/datasets.py` and
# one each in export_onnx, seeding, detection, backbone, bev and trainer.
BASELINE="${TY_BASELINE:-11}"

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
