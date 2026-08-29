#!/usr/bin/env bash
# Hold `ty` on a tree to a baseline count that may fall and never rise.
#
#   scripts/ty_ratchet.sh            # src/, its own baseline
#   scripts/ty_ratchet.sh scripts/   # scripts/, its own baseline
#   TY_BASELINE=12 scripts/ty_ratchet.sh scripts/   # override, for a one-off
#   TY_PYTHON=3.10 scripts/ty_ratchet.sh            # override the pinned interpreter
#
# A ratchet rather than a pass/fail gate: turning a gate red on all of the debt at once
# is how a checker gets deleted a week later. Blocking the *increase* is the part worth
# having -- new code type-checks, old debt is paid down as those files are touched.
#
# ===========================================================================
# TWO WARNINGS BEFORE EDITING THIS FILE. Both were paid for on 2026-08-19.
#
# 1. **Never invoke the checker in a way that can write to the caller's `.venv`.**
#    `uv run --python 3.12` does not merely select an interpreter -- it re-syncs the
#    project environment to it. One run switched this repo's venv from 3.10.12 to
#    3.12.13 and dropped the `export` extra, on a checkout several people share, and it
#    surfaced as an unrelated test skipping on a missing `onnx`. A measuring tool that
#    rewrites the environment it measures is more dangerous than whatever it measures.
#    `--isolated` below is not a detail; it is what makes this safe to run twice.
#
# 2. **The baselines are meaningless unless the interpreter is pinned.** See the block
#    above `TY_PYTHON` for the three-environment table. The short version: the count
#    reports what the installed *stubs* say, `uv.lock` resolves a different numpy per
#    Python version, and the same tree measured 8 and 12 on two of them.
# ===========================================================================
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
# **Two baselines, not one sum.** src/ is what ships and scripts/ is
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
# src/ 11, and this paragraph said 8 until 2026-08-27 while the case statement below
# enforced 11 and the tree measured 11. The 8 was real: `e468209` paid the debt back past
# where it started. Then `d9e3cca` raised the baseline to 11 when the pose head landed and
# left this prose at 8 -- so the file that closes with "a comment that excuses debt is
# itself a claim, and nothing was checking this one" had the failure in its own header,
# three revisions deep. Nothing broke, which is why it survived: the gate was right the
# whole time and only its description was wrong.
#
# The history below is kept because it is how the number got to 8, and how it left.
# src/ 8, down from 11. It rose to 14 with the crop encoder in `e79421d` and was paid
# back past where it started rather than absorbed. What was fixed, by category:
#
#   Image.BILINEAR              the Pillow stubs no longer declare it; four sites across
#                               src/ and scripts/ moved to `Image.Resampling.*`. Still
#                               resolves at runtime on Pillow 12, so this was lint-level
#                               today -- and it is the deprecated family, and the Lite3
#                               copy of geometry/ already had to move to Pillow 10.4.
#   __getitem__ override x3     `PA100K`, `SegFolderDataset`, `CocoDetDataset`. Not a
#                               torch stub gap: `Dataset.__getitem__` names its parameter
#                               `index` and can be called with it as a keyword, so `i` and
#                               `idx` were genuine Liskov violations. Renamed.
#   Tensor | Module x2          `register_buffer` and `ModuleDict.__getitem__` both read
#                               back wider than what they hold. Fixed with a class-level
#                               annotation and a `cast` respectively.
#
# The remaining 8 are older and in files this change did not otherwise touch -- 3 in
# `data/datasets.py`, one each in seeding, detection, backbone, bev and export_onnx. They
# are left alone on the principle in the header: debt is paid down as those files are
# touched, and editing six unrelated modules to chase a number is how a fix becomes a
# regression.
#
# scripts/ 19 on the day the gate was extended, then 18 when the interpreter was pinned
# (a7d8410, 08-19). It sat at 18 until 2026-08-26 and the tree had long since moved: nothing
# in `scripts/` was ever *fixed* for this number, but twelve scripts left in `500cdd2` and
# the plate-calibration, SAM 3 and photometry code moved into the packages. The count went
# with them and the baseline did not follow, because a fall is a notice here rather than a
# failure and nobody spent the notice.
#
# **A baseline above the true count is not a safe margin, it is a hole.** At 18 against a
# real 17, a change introducing one new diagnostic passes green -- which is the exact
# failure this ratchet exists to catch, arrived at from the loose side rather than the
# red side.
#
# scripts/ 17, measured 2026-08-26. The breakdown, which was previously withheld on the
# grounds that naming the members of a never-reviewed number would imply a review:
#
#   campaign_site30k.py   7
#   site_events.py        3
#   track_review.py       2
#   bench_e2e.py          2
#
# scripts/ 14, measured 2026-08-28: the Orin scripts left with the board they ran on, and
# `live_view_orin.py`'s one diagnostic went with them. A fall is a notice rather than a
# failure here, and this is the notice being spent.
#
# It is here now because a number being lowered has been looked at, which is the whole
# difference. Still not a backlog: none of them paid down, and the header's rule applies
# -- they come down as those files are touched.
#
# scripts/ 15 on 2026-08-28, and this one is not a payment either: `render_demo.py` was
# deleted (superseded by `tools/commissioning/demo_video.py`) and took its 2 with it.
# Recorded because a baseline that falls for a deletion and a baseline that falls for a
# fix look identical in the number and mean opposite things about the tree.
#
# tools/ armed 2026-08-28 at **73**, which is where it already was. Arming a third tree at
# its own count rather than paying it down first is this file's whole argument applied
# once more -- 73 red diagnostics on a first run is how a checker gets switched off, and
# blocking the increase is the part worth having. What it is not is a claim that tools/ is
# clean. The debt is concentrated: `site30k/review_page.py` 15, `site30k/recipe.py` 7,
# `temporal/ntu_project.py` 6, `commissioning/service_zones.py` 5, and the remaining 40
# are 1-4 apiece across sixteen files. A large share of those ones and threes is one
# pattern rather than one mistake -- a tool that loads `recipe.py` through
# `importlib.util.spec_from_file_location` gets an untyped module object, and every
# attribute read off it is a diagnostic. Fixing that is a packaging change, not a type
# annotation, so it does not come down by touching the files.
case "$TARGET" in
  scripts/|scripts) DEFAULT_BASELINE=14 ;;
  tools/|tools)     DEFAULT_BASELINE=73 ;;
  *)                DEFAULT_BASELINE=11 ;;
esac
BASELINE="${TY_BASELINE:-$DEFAULT_BASELINE}"

# **The interpreter is pinned, and without this the baselines mean nothing.**
#
# `uv.lock` resolves a different dependency set per Python version, and the count moves
# with it -- the checker reports what the installed *stubs* say, not only what this code
# says. Measured on one tree, one commit, three environments:
#
#     python 3.10 / numpy 2.2.6    src/  8    scripts/ 27
#     python 3.12 / numpy 2.5.2    src/ 12    scripts/ 18   (fresh `uv sync --locked`)
#     python 3.12, pinned here     src/ 12    scripts/ 18   (reproducible everywhere)
#
# Nothing in the tree differed between those rows. A ratchet whose number depends on which
# interpreter `uv` happened to pick cannot ratchet: it fails for a contributor whose venv
# is a minor version behind CI's, and it passes for one whose venv is ahead. This was found
# when a peer attributed an 8-diagnostic overage to a file move that could not have caused
# it -- moving files inside `scripts/` cannot change a recursive check of `scripts/`.
#
# 3.12 rather than the 3.10 floor because it is what a fresh `uv sync` resolves today and
# what the test matrix's upper row uses, so the gate measures the environment contributors
# actually get. Override for a one-off with TY_PYTHON=3.10.
#
# **A second reason a hand-run count does not compare, independent of the interpreter.**
# This script counts `error[` lines; `ty`'s own trailing "Found N diagnostics" counts
# warnings too. Measured on `tools/` on 2026-08-27: ty said 65, this says 63. So a reader
# who runs the checker directly and reads its summary line is comparing a different
# quantity against the baselines below, and it is the larger one -- which fails in the
# direction that looks like a regression that is not there.
TY_PYTHON="${TY_PYTHON:-3.12}"
# `--isolated` is load-bearing -- warning 1 in the header says why. Do not drop it.
RUNNER=(uv run --isolated --python "$TY_PYTHON" ty check "$TARGET" --output-format=concise)

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
