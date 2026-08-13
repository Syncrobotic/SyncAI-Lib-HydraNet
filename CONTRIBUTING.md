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

The pre-rewrite history is kept at the branch and tag
`backup/pre-conventional-commits`. It is not dead weight — `runs/*/meta.json`
records the commit each run was trained from, and runs made before the rewrite
point at hashes that only exist there. Do not delete it while those runs matter.

## Code

Lint and format with ruff, and run the tests, before opening a pull request:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest -q
```
