# Release process

Two things get released and they version differently. **Code** flows `dev → stage → main`
through branches. **Models** do not live in git at all — `runs/` is gitignored — so a
checkpoint has no version until something freezes it into an immutable bundle.

Conflating the two is the usual mistake. A commit on `main` says which code is released; it
says nothing about which weights are running on a robot.

---

## 1. The branches

| Branch | Means | Who merges | Gate |
|---|---|---|---|
| `dev` | Integration. Work lands here. | Anyone | CI: lint, format, tests, coverage ≥ 80% |
| `stage` | Release candidate. Deployed to test robots. | Release owner | Everything above, plus the export gate and the acceptance checklist |
| `main` | Released. What a robot in the field is running. | Release owner | Fast-forward from `stage` only |

Rules that make this worth having:

- **`main` and `stage` are never committed to directly.** They only ever advance by merge
  or fast-forward from the branch below.
- **Promotion is fast-forward.** If `stage` cannot fast-forward onto `dev`, rebase or fix
  it on `dev` first. A merge commit created during promotion is code that was never tested
  on `dev`.
- **No history rewriting on `stage` or `main`, ever.** Run metadata records the commit it
  trained from, and a rewrite makes those hashes unresolvable. This has already happened
  once on `dev`; see the recovery note in [CONTRIBUTING.md](../CONTRIBUTING.md).
- **Hotfixes go to `dev` first,** then get promoted, unless the robot is actively unsafe.
  A fix applied straight to `main` is a fix that will be lost at the next promotion.

### Promoting

```bash
# dev -> stage
git checkout stage && git merge --ff-only origin/dev && git push origin stage

# stage -> main, after the acceptance checklist passes on a test robot
git checkout main && git merge --ff-only origin/stage && git push origin main
```

### Where the version number comes in

Nothing above changes when a release is cut, and that is the point.
[`.github/workflows/release-please.yml`](../.github/workflows/release-please.yml) splits a
release into the two halves that belong at opposite ends of this flow:

| | Runs on | Does | Merged how |
|---|---|---|---|
| `release-pr` | `dev` | opens a PR bumping `__version__` and writing `CHANGELOG.md` | an ordinary PR into `dev`, like any `feat/*` |
| `release` | `main` | sees a version with no tag, tags it, cuts the GitHub Release, attaches the wheel and sdist | nothing — it only reads |

So the bump enters at the bottom and rides the same fast-forward promotions as the code
it describes. **No branch ever gains a commit that did not come from the branch below
it**, and the rules above stay exactly as written.

The obvious alternative — pointing release-please at `main`, as the sibling Gateway repo
does — was rejected because it cannot work here. A release PR merges *into* its target, so
`main` would gain a commit `stage` does not have and the next promotion would be refused:

```text
$ git checkout main && git merge --ff-only origin/stage
fatal: Not possible to fast-forward, aborting.
```

Cutting the tag on `dev` instead was rejected for the opposite reason: a tag is a claim
that code is released, and this table defines `main` as what a robot in the field is
running. Tagging on `dev` would label code that has not been through `stage`'s acceptance
checklist.

Because promotion is fast-forward, the commit the tag names is the same git object that
was reviewed on `dev` and accepted on `stage` — not a copy of it.

The version bump lands on `dev` before the code is released, so `dev` and `stage` will
report a version whose tag does not exist yet. That is expected: the version is the *next*
release's number from the moment the PR merges. `pip install` from a tag is unaffected —
it can only resolve tags that exist.

## 2. Model releases

A model version is the weights, the graph, the config, the lineage and the numbers, frozen
together and checksummed. `scripts/release_bundle.sh` builds one:

```bash
scripts/release_bundle.sh create runs/hydranet_indoor_det60 v1
scripts/release_bundle.sh verify releases/v1
scripts/release_bundle.sh publish releases/v1 gs://syncai-hydranet
```

```text
releases/v1/
├── model.pt        weights
├── model.onnx      the portable graph — this is what is version-controlled
├── config.yaml     exact config, including any --set overrides
├── meta.json       lineage: commit, dirty flag, dataset fingerprint, environment
├── metrics.json    the selected epoch and its scores
└── MANIFEST.json   sha256 of every file above
```

### It refuses to build a bundle it cannot stand behind

Three gates, all of which have fired on real runs here:

1. **Dirty tree.** If `meta.json` says the working tree was dirty, the exact source exists
   only in `uncommitted.patch`. That is a recovery route, not a release.
2. **Unreachable commit.** If the recorded commit is not in the repository, the lineage is
   a string, not a reference.
3. **Export parity.** The ONNX must reproduce the checkpoint's outputs. Without this,
   `model.pt` and `model.onnx` in one directory is a claim rather than a fact. This gate is
   also what stops a model whose head was never supervised — the export guard refuses it.

### Versions are immutable

`create` refuses an existing version directory; `publish` refuses an existing GCS prefix.
To change anything, cut a new version. Promote by moving an **alias** — in Model Registry,
or as a small pointer file — never by overwriting bytes.

### Engines are not in the bundle

A TensorRT engine is tied to a **GPU architecture, TensorRT version and JetPack version**.
The engine built for the AGX Orin on JetPack 6.1 / TRT 10.3 will not load on an Orin NX, or
after a JetPack upgrade. So the engine is a per-target build artefact derived from
`model.onnx`, not the model itself:

```text
releases/v1/engines/agx-orin_jp6.1_trt10.3_fp16.engine
```

Rebuild on the board; keep the ONNX as the thing you version.

## 3. What has to be true before `main`

Every item is currently unmet, which is an accurate summary of where the project is.

| Gate | How to check |
|---|---|
| CI green on `stage` | Actions |
| Every head supervised | `hydranet-export-onnx` refuses otherwise |
| Numbers from `test`, not `val` | `hydranet-eval --split test`; val selected the checkpoint, so quoting it is circular |
| Per-class floors met | Not just mIoU. A mean stays respectable while one class goes to noise — that has happened here |
| Field footage reviewed | A human has watched the overlay on real site video |
| Export parity | `--check-parity` |
| Latency measured on the target board | `scripts/bench_orin.sh`, and `scripts/bench_camera_orin.py` for the end-to-end number |
| Bundle published and verified | `release_bundle.sh publish` |

## 4. Which version is on which robot

The bundle answers "what is this model". It does not answer "what is running where".

Until a registry is in place, keep a single file in the bucket —
`gs://syncai-hydranet/deployments.json` — mapping fleet or robot id to a released version
and the engine build it uses. One file, appended to, never edited in place. When Model
Registry aliases replace it, they answer the same question with an audit trail.

The question this exists to make answerable: *a robot behaved badly on Tuesday — which
weights were on it, which data trained them, and which commit produced that data?*
`deployments.json` → `MANIFEST.json` → `meta.json` → dataset fingerprint answers it in
four hops, all of them immutable.
