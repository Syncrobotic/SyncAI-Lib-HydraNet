# Contributing

## Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/).
Every commit subject must start with a type and a colon:

```
<type>: <description>
```

The description is lower case, imperative mood, and says what the commit does
rather than what it touched — `fix: write last.pt after updating the best score`,
not `fix: trainer.py changes`.

### Types

| Type | Use for |
|---|---|
| `feat` | a new capability |
| `fix` | a defect corrected |
| `docs` | documentation only |
| `style` | formatting with no behaviour change |
| `refactor` | restructuring with no behaviour change |
| `perf` | a change made for speed or memory |
| `test` | tests only |
| `build` | dependencies, packaging, `uv.lock` |
| `ci` | workflows and CI configuration |
| `chore` | tooling and housekeeping |
| `revert` | reverting an earlier commit |

A breaking change takes a `!` before the colon (`feat!: drop the fixed_weights
option`) and explains the migration in the body.

### This is enforced

`.pre-commit-config.yaml` runs
[conventional-pre-commit](https://github.com/compilerla/conventional-pre-commit)
at the `commit-msg` stage. That stage is **not** installed by a plain
`pre-commit install`, so install it explicitly once per clone:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Without that second hook type the check silently does nothing, which is how the
convention was missed in the first place.

### The history was rewritten

Commits up to and including `docs: make CUDA the reference platform in the
handover doc` originally used a different style (imperative sentence case, no
type prefix) and were rewritten in place to conform. Only the subject lines
changed: every tree hash and every `Co-Authored-By` trailer is byte-identical to
the original.

The pre-rewrite commits are gone; nothing is kept on the side, and nothing needs
to be. Because only subject lines changed, each old commit has a counterpart on
`dev` with an identical tree — the code is all still here under a different hash.

That matters because `runs/*/meta.json` records the commit each run was trained
from, so a run made before the rewrite points at a hash that no longer resolves.
To identify one, match on the subject rather than the hash: the rewritten
counterpart is the same sentence, lower-cased, behind a type prefix. The baseline
run `runs/hydranet_indoor` (traversability mIoU 0.6765) records `ba30fa88`, which
is `aa07bbe docs: translate the three docs to English, and fix a 404 download URL`.

## Code

Lint and format with ruff, and run the tests, before opening a pull request:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest -q
```

## Figures under `assets/`

`assets/` is an **allowlist**, not a denylist: `.gitignore` ignores `assets/*` and
names the figures back in one by one. Adding a new one therefore takes two steps,
and the second one is the point:

```bash
git add -f assets/my_new_figure.png     # 1. override the ignore
#          then add `!assets/my_new_figure.png` to .gitignore   # 2. keep it addable
```

The friction is deliberate. Most images in this project are rendered from customer
CCTV — `datasets/studioa_clips/<City>-cam<NN>/` — and a single frame of a shop floor
carries identifiable shoppers and staff. A denylist of formats cannot protect that,
because the format a frame lands in is whichever one the renderer chose. It was
`assets/*.mp4`, `*.mov` and `*.pdf` for a while, and a 1920x380 PNG cut from
Taichung-cam01 walked straight past all three.

So before step 1, look at the image and ask whether anyone in it is identifiable.
A figure that only shows masks, meshes or plots is fine. A frame with people in it
needs their consent or needs to not be here — the history is not editable after the
fact, and `tests/test_assets_allowlist.py` is only guarding the mechanism, not the
judgement.
