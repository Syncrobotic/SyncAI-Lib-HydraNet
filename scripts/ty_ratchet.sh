#!/usr/bin/env bash
# Hold `ty` on a tree to a baseline count that may fall and never rise.
#
#   scripts/ty_ratchet.sh            # src/, its own baseline
#   scripts/ty_ratchet.sh scripts/   # scripts/, its own baseline
#   TY_BASELINE=12 scripts/ty_ratchet.sh scripts/   # override, for a one-off
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
# **Two baselines, not one sum.** src/ is what ships to the robot and scripts/ is
# dev-side, and a single number would let src/ debt hide behind a scripts/ improvement --
# the total can fall while the thing that ships gets worse. They ratchet independently.
#
# scripts/ was outside this gate until 2026-08-18, on the reasoning quoted in the line
# this replaces: dev-side, widen when the count is worth defending. It became worth
# defending. scripts/ is 5,392 lines -- larger than cli/ or data/ -- and it is where the
# only two genuine code duplications in the repo grew, one of them with a behavioural
# divergence between the copies (`track_clip` undistorts boxes in site_events.py and does
# not in mine_fall_candidates.py). An unwatched tree that size does not stay still.
set -euo pipefail

TARGET="${1:-src/}"

# Lower these when you fix things. Raising one needs a reason, and the reason goes here
# rather than only in a PR body, because this file is what the next person reads.
#
# src/ 13. It was 11 -- 5 in `data/datasets.py` and one each in export_onnx, seeding,
# detection, backbone, bev and trainer -- and rose to 14 with the crop encoder in
# `e79421d`. One of those three was `Image.BILINEAR`, a Pillow alias the stubs no longer
# declare, and it is fixed rather than absorbed (four sites across src/ and scripts/, now
# `Image.Resampling.*`). The other two are named because a baseline that rises without
# names is just a ceiling: `data/attributes.py` overrides `Dataset.__getitem__`
# incompatibly, which is the same shape as the two already accepted in `data/datasets.py`,
# and `models/crop_encoder.py` passes a `Tensor | Module` where a `Tensor` is wanted,
# which is the same shape as the one already accepted in `trainer.py`. Both are instances
# of categories in the baseline, not new kinds of debt.
#
# scripts/ 19, measured on the day the gate was extended. No breakdown here on purpose:
# unlike src/'s, this number has never been paid down, so naming its members would imply
# a review that has not happened.
case "$TARGET" in
  scripts/|scripts) DEFAULT_BASELINE=19 ;;
  *)                DEFAULT_BASELINE=13 ;;
esac
BASELINE="${TY_BASELINE:-$DEFAULT_BASELINE}"
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
echo "$TARGET diagnostics: $count (baseline $BASELINE)"

if [ "$count" -gt "$BASELINE" ]; then
  echo "::error::$TARGET type diagnostics rose to $count (baseline $BASELINE); see the log above"
  exit 1
fi

# A fall is a notice, not a failure, on purpose. Making a PR that happens to fix an
# unrelated annotation also edit this file to stay green is exactly the friction that
# gets a check muted. The cost is that the baseline drifts above the true count until
# someone lowers it, which this asks for by name on every run that earns it.
if [ "$count" -lt "$BASELINE" ]; then
  echo "::notice::$TARGET diagnostics fell to $count -- lower its baseline in scripts/ty_ratchet.sh"
fi
