#!/usr/bin/env bash
# Hold `ty` on a tree to a baseline count that may fall and never rise.
#
#   scripts/ty_ratchet.sh            # src/, its own baseline
#   scripts/ty_ratchet.sh scripts/   # scripts/, its own baseline
#   scripts/ty_ratchet.sh tools/     # tools/, its own baseline
#   TY_BASELINE=12 scripts/ty_ratchet.sh scripts/   # override, for a one-off
#   TY_PYTHON=3.11 scripts/ty_ratchet.sh            # override the pinned interpreter
#
# A ratchet rather than a pass/fail gate: turning a gate red on all of the debt at once is
# how a checker gets deleted a week later. Blocking the *increase* is the part worth
# having -- new code type-checks, old debt is paid down as those files are touched.
#
# Three baselines, not one sum. `src/` is what ships, `scripts/` is dev-side and `tools/`
# is one tree further out; a single number would let what ships regress behind an
# improvement somewhere else. They ratchet independently.
#
# ===========================================================================
# TWO WARNINGS BEFORE EDITING THIS FILE.
#
# 1. **Never invoke the checker in a way that can write to the caller's `.venv`.**
#    `uv run --python 3.12` does not merely select an interpreter -- it re-syncs the
#    project environment to it. One run switched this repo's venv to a different Python
#    on a checkout several people share, and it surfaced as an unrelated test skipping on
#    a missing `onnx`. `--isolated` below is what makes this safe to run twice.
#
# 2. **The baselines are meaningless unless the interpreter is pinned**, because the count
#    reports what the installed *stubs* say and `uv.lock` resolves a different numpy per
#    Python version. One tree, one commit, three environments:
#
#        python 3.10 / numpy 2.2.6    src/  8    scripts/ 27
#        python 3.12 / numpy 2.5.2    src/ 12    scripts/ 18
#        python 3.12, pinned here     src/ 12    scripts/ 18
#
#    Nothing in the tree differed between those rows.
# ===========================================================================
set -euo pipefail

TARGET="${1:-src/}"

# Lower these when you fix things. Raising one needs a reason, and the reason goes here
# rather than only in a PR body, because this file is what the next person reads.
#
# **A baseline above the true count is not a safe margin, it is a hole**: at 18 against a
# real 17, a change introducing one new diagnostic passes green, which is the exact
# failure this exists to catch. So a fall is worth spending -- and note that a baseline
# falling because a file was *deleted* and one falling because something was *fixed* look
# identical in the number and mean opposite things about the tree.
#
# `tools/` is armed at its own count rather than paid down first, and that is not a claim
# that it is clean. Much of its debt is one pattern: a tool that loads another module
# through `importlib.util.spec_from_file_location` gets an untyped module object, and
# every attribute read off it is a diagnostic. That comes down by a packaging change, not
# by annotating.
case "$TARGET" in
  scripts/|scripts) DEFAULT_BASELINE=14 ;;
  tools/|tools)     DEFAULT_BASELINE=73 ;;
  *)                DEFAULT_BASELINE=8 ;;
esac
BASELINE="${TY_BASELINE:-$DEFAULT_BASELINE}"

# 3.12 rather than the 3.11 floor because it is what a fresh `uv sync` resolves today and
# what the test matrix's upper row uses, so the gate measures the environment contributors
# actually get.
TY_PYTHON="${TY_PYTHON:-3.12}"
# `--isolated` is load-bearing -- warning 1 says why. Do not drop it.
RUNNER=(uv run --isolated --python "$TY_PYTHON" ty check "$TARGET" --output-format=concise)

# ty exits 0 clean, 1 with diagnostics, 2 when it could not run at all -- a broken
# [tool.ty] table, an unreadable tree. Only 0 and 1 mean the count below is real, so 2
# fails here rather than counting zero errors in output that was never produced.
# `--exit-zero` would flatten that distinction and report a checker that never started as
# a clean tree; this repo has already had a [tool.ty] key that silently stopped it.
out=$("${RUNNER[@]}" 2>&1) && rc=0 || rc=$?
printf '%s\n' "$out"

if [ "$rc" -gt 1 ]; then
  echo "::error::ty could not run (exit $rc); the diagnostic count is not meaningful"
  exit 1
fi

# `error[` lines, not ty's own trailing "Found N diagnostics", which counts warnings too --
# measured on `tools/`, ty said 65 and this says 63. A reader who runs the checker by hand
# and reads its summary is comparing the larger quantity against these baselines, which
# fails in the direction that looks like a regression that is not there.
count=$(printf '%s\n' "$out" | grep -c 'error\[' || true)
echo "$TARGET diagnostics: $count (baseline $BASELINE)"

if [ "$count" -gt "$BASELINE" ]; then
  echo "::error::$TARGET type diagnostics rose to $count (baseline $BASELINE); see the log above"
  exit 1
fi

# A fall is a notice, not a failure, on purpose. Making a PR that happens to fix an
# unrelated annotation also edit this file to stay green is exactly the friction that gets
# a check muted. The cost is that the baseline drifts above the true count until someone
# lowers it, which this asks for by name on every run that earns it.
if [ "$count" -lt "$BASELINE" ]; then
  echo "::notice::$TARGET diagnostics fell to $count -- lower its baseline in scripts/ty_ratchet.sh"
fi
