# Session board, 2026-08-17

> **Live coordination file, not a durable document.** Three sessions have worked this
> repository concurrently before (see [2026-08-14](2026-08-14-deploy-retail-handoff.md)),
> and what made that work was announcing file scope before touching anything and
> reporting corrections immediately. `SendMessage` does that between live sessions; this
> file does it for sessions that start later or die mid-run, which messages cannot reach.
>
> **Append your own section. Do not edit another session's.** The board is in git, so two
> sessions editing the same lines is the exact conflict it exists to prevent.
>
> Delete the whole file when the day's work lands.

## The hazard that path claims do not cover

**Every session is in the same working tree** — `/home/paul/SyncAI-Lib-HydraNet` on `dev`,
one checkout, one index. Not separate clones. Established 2026-08-17 by session `-d61906`,
which saw another session's brand-new untracked file in its own `git status`.

Disjoint file lists therefore do **not** isolate anyone, because git operates on the whole
tree rather than on the paths a session has claimed:

| command | what it does to the other session |
|---|---|
| `git add -A`, `git add .`, `git commit -a` | sweeps their uncommitted work into your commit |
| `git stash`, `git checkout -- .`, `git restore .` | **destroys** their unstaged work outright |
| `git checkout <branch>` | same, and moves the branch under them |

**Rules, agreed between the live sessions:**

1. `git add <explicit paths you claimed>` only. Never `-A`, never `.`, never `commit -a`.
2. **No `stash` / `checkout` / `restore` / `clean` / branch switch on this tree without
   announcing first and getting a reply.** This is the one that loses work irrecoverably.
3. Announce before committing — a commit on `dev` moves the branch under everyone.

The index is shared state and already holds two staged deletions
(`assets/retail_cctv_clip3_ab.png`, `assets/retail_cctv_clip3_column.png`) that
**predate every live session**. Owner unknown; do not sweep them into a commit.

> This line was wrong twice before it was right, and the way it failed is the part worth
> keeping. It first attributed the deletions to one session, then to another, on the
> strength of `git status` and file mtimes. Neither is evidence. **In a shared checkout,
> finding uncommitted work is not evidence of having written it, and an mtime is not
> evidence of authorship** — `utils/temporal.py`'s 09:15:39 is a `stash pop` restoring
> the file 13 seconds after the rebase finished, not somebody typing.
>
> What settles it is the stash, and anyone can re-run it: `stash@{0}` was created
> 09:05:20, *before* the rebase and before any live session's work, and
> `git stash show --include-untracked --name-status stash@{0}` lists both deletions plus
> `A src/syncai_hydranet/utils/temporal.py` and `M scripts/bev_video.py`. Independently,
> session `4f65b69c`'s own start-of-session `git status` snapshot already contained all
> three, which puts them earlier still.
>
> **`src/syncai_hydranet/utils/temporal.py` is therefore unowned by any live session.**
> It is untracked, it changes model output, and its failure mode is a *cleaner-looking*
> panel rather than a crash. An unowned file is what gets lost when a day's sessions end.
>
> How that conclusion was reached — and the general standard for settling an authorship
> question in a shared tree — is under session `3988c4fc`'s heading below. One copy, there.

### Left here for whoever picks up `utils/temporal.py`, since nobody currently holds it

**The 48 night clips are the best test material that file will get.** Pulled by accident —
a timezone mistake, see below — `datasets/studioa_clips/<camera>/` holds one clip per
camera at local ~00:00: greyscale IR, shutters down, empty store. That is the only footage
where `FixedCameraStabiliser`'s change-gate can be measured **in isolation from the thing
it exists to protect**, because nothing is moving through the plate. A gate that is never
exercised and a gate that works look identical on footage with people in it.

**Do not take the camera list from the bucket's directory listing.** Of the 48 cameras,
only **24 are shop floor**; ten are back office or stockroom under IR, eight are
classrooms, stairwells or the street outside, and **six return an essentially black
frame**. A background plate converging on a black frame converges beautifully and means
nothing. Contact sheets, one labelled tile per camera:
`datasets/studioa_clips/_survey/sheet_{Kaohsiung,Taichung,Tao-Hsin}.jpg`.

**Selecting those night clips by an hour-prefix glob silently returns half of them.** The
bucket straddles the UTC hour — the clips run `20260816-155732` to `20260816-160231`, so
`archive_20260816-16*` matches 23 and the other 25 are under `-1557`. Verified: 23 + 25 =
48, one per camera. Third member of today's family of failures that return a plausible
number instead of an error. Select from `manifest_*.json`, not from a glob.

If real isolation is wanted rather than discipline, `git worktree add` for one session is
the clean fix.

The venv is an **editable install** pointing at this same `src/`, so a training run does
not hold a private copy of the code — it imports the tree everyone is editing. Modules
already loaded stay loaded, but nothing about that is guaranteed for a dataloader worker
respawned later in the run. Treat the label-decode path as frozen while a trainer is up.

### Two traps found today that are not about git

**`uv sync --locked --group dev --extra export` prunes `transformers`.** It lives only
under the `annotate` extra, so reproducing a CI job in the project venv removes
`transformers`, `huggingface-hub`, `tokenizers` and `safetensors` — and
`scripts/sam3_prelabel.py` stops working with no other symptom. Use
`--extra export --extra annotate`, or point `UV_PROJECT_ENVIRONMENT` at a throwaway venv
and leave `.venv` alone. (Found by session `-4421a4` with `--dry-run`.)

**CI is red before anyone commits anything.** `ty_ratchet.sh` reports 19 diagnostics
against `BASELINE=15`, and `origin/dev` reports the same 19 — so the next PR into `dev`
is red whatever it contains. **Four of the nineteen are in
`src/syncai_hydranet/analytics/dwell.py:103-104`** (`ndarray.min`/`max` under the numpy
2.5 stubs the lock pins; this `.venv` has 2.2.6, which is why it looks clean locally).
Anyone extending `analytics/` hits this first, and those four lines are the cheap fix.

**`scripts/bev_video.py` is gone**, deleted upstream in `9ec0c23` and rewritten as
`src/syncai_hydranet/cli/scene.py` (`hydranet-scene`). Anything shelling out to it broke
at the rebase.

`dev` was rebased onto `origin/dev` at ~09:20 CST today; two commits were replayed with
new SHAs (`b34a2c2`→`ce8445f`, `2b593ef`→`4840034`). **Any SHA recorded before then is
stale** — including in run metadata.

## Long-running jobs — check here before touching the GPU

| unit / process | owner | started | state | output |
|---|---|---|---|---|
| `hydranet-retail-objects.service` | `4f65b69c` | 09:32 CST | **done 10:57, clean exit, 60 epochs — GPU IS FREE** | `runs/hydranet_retail_objects/` |
| `studioa-pull.service` | `4f65b69c` | 09:41 CST | done, clean exit — 96 clips | `datasets/studioa_clips/` |
| `studioa-pull2.service` | `4f65b69c` | 09:52 CST | done, clean exit — 96 more, 3.41 GB total | `datasets/studioa_clips/` |

Both are **systemd user units**, not background shells. `Linger=yes` is set for `paul`,
so they survive an SSH disconnect and a logout; `WantedBy=default.target` and
`Restart=on-failure` mean they also survive a reboot and a transient crash. Check state
with `systemctl --user status <unit>` and follow with
`tail -f ~/hydranet-overnight/retail_objects.log`.

Also on disk and **not** running: `runs/hydranet_fixed_coco10` is the balancer control,
stopped at epoch 6 of 60, resumable via `hydranet-overnight/resume_fixed.sh`. Its epoch-6
numbers answer nothing — see [ARCHITECTURE_REVIEW.md](../ARCHITECTURE_REVIEW.md).

---

## Session `-4f65b6` — the retail analytics line

**Working on.** Extending retail toward analytics: person attributes (age / gender),
Apple-product detection, people flow. Driven by the product ask "零售分析需要年齡、性別、動作".

**Paths claimed.**

```
scripts/pull_studioa.py                     new — bounded GCS clip sampler
datasets/studioa_clips/                     new — sampled site footage from the bucket
docs/journal/2026-08-17-session-board.md    this file
~/hydranet-overnight/retail_objects.sh      new
~/.config/systemd/user/hydranet-retail-objects.service   new
~/.config/systemd/user/studioa-pull.service              new
```

Likely next, announced in advance: `datasets/retail_objects_batch01/` via
`scripts/sam3_prelabel.py`, and possibly `docs/RETAIL_ANALYTICS.md`.

**Not touched, and will not be without asking:** anything under `src/syncai_hydranet/`,
`configs/`, `tests/`.

**GCS.** `gs://studioa-recording` is now readable — 48 cameras (Kaohsiung / Taichung /
Tao-Hsin, 16 each), 2026-07-19 to 2026-08-17, 24 h/day, ~189 clips per camera-day at
~11 MB. **That is roughly 3 TB; never pull the bucket.** `scripts/pull_studioa.py` takes
one clip per camera per target time, which is the axis with information in it — a fixed
camera's 189th clip of a day tells you almost nothing its 1st did not.

The `paul_jiang@syncrobotic.com` token expires and needs an interactive
`gcloud auth login`. The service account `syncai-hydranet@syncrobotic-aisw` still lacks
`storage.objects.list` on that bucket; granting it `roles/storage.objectViewer` would
make unattended runs work and is worth doing before anything is scheduled.

**Correction, mine: the bucket's object names are UTC and the stores are UTC+8.** A pull
targeting "11:30 and 16:00, opening and the busy hour" returned local 19:30 and local
**00:02** — greyscale IR, shutters down, nobody in frame. Nothing errored; the sample was
simply of the wrong thing. Caught by tiling one still per camera and reading the burnt-in
timestamps, not by reading the manifest. `scripts/pull_studioa.py` now takes store-local
`--times` with `--utc-offset` (default 8.0) and records both in the manifest. The night
clips are kept: an empty IR-lit store is the *best* pass for `column` and `fixture`
footprints, because nobody is standing in front of them, and on a fixed camera the
geometry is what the polygon is for.

**Half the fleet is not shop floor.** From the contact sheets in
`datasets/studioa_clips/_survey/` (one still per camera, per store):

| store | retail floor | back office / stockroom | classroom, stairwell, street | dark or offline |
|---|---|---|---|---|
| Kaohsiung | 8 | 2 | 4 | 2 |
| Taichung | 11 | 3 | 2 | 0 |
| Tao-Hsin | 5 | 5 | 2 | 4 |
| **total** | **24 / 48** | 10 | 8 | 6 |

Scope anything per-camera against 24, not 48. Also visible and worth carrying: viewpoints
range from wall-mounted wide-angle to near-nadir over a desk (`Tao-Hsin-cam05`, `cam07`),
every camera has heavy barrel distortion, and the accessory walls in `Kaohsiung-cam07`,
`Taichung-cam06/10/11` and `Tao-Hsin-cam01/09` are hanging packets rather than boxes —
which is exactly the case `sam3_prompts_objects.py` measured at **0** instances for the
`product box` prompt. One vocabulary will not cover both.

**Measured today, so it is not re-derived.**

| finding | how |
|---|---|
| Person boxes are **244–336 px tall** (median) on `assets/archive_*.mp4`, 85–98% above 128 px | shipped detection head at score 0.20, boxes mapped back through the letterbox |
| The head region is **31–42 px** | body height / 8, an upper bound under a 50° down-pitch |
| Tracks last a median of **9–16 frames** at 5 fps; one 4.6-min clip fragments into **1234 tracks** | `scripts/retail_flow.py` on the three site clips |

What they decide: body-crop attribute models are feasible and face-based age is not;
and **track continuity, not model capacity, is the binding constraint** on anything
temporal. The second-stage crop encoder should therefore be built to serve track
association first, attributes second.

Convergent with `sam3_prompts_objects.py`'s own measurement — brand prompts are silent
(`iphone` <2 instances, `macbook` 4 at 0.478 against `laptop` 0.800), so iPad / MacBook /
iPhone is also a second-stage crop classifier. One crop model, four jobs: attributes,
track association, action, product sub-class.

---

## Session `-4421a4` — the rebase, CI, and an architecture review

> **Identifier note.** `-4421a4` is the ref other sessions see for me; my own session
> directory is `11a858b0-…`. The two traps in the hazard section above are credited to
> `-4421a4`, which is this session — no separate write-up needed here.

**Working on.** Rebasing local `dev` onto `origin/dev`, reproducing CI locally, then a
read-only architecture review for PSheon. Started before this board existed, which is why
the items below are reported rather than announced in advance.

**Git state — the part other sessions must not step on.**

I rebased `dev` onto `origin/dev` at ~09:20 CST (53 commits behind). Two commits were
replayed with new SHAs, as recorded in the hazard section. `dev` is now `ce8445f`, ahead
2 / behind 0. `git reflog dev` confirms `dev@{1}`/`dev@{2}` were both PSheon's own commits
and both were replayed — no other session's work was orphaned.

`stash@{0}` ("claude: pre-rebase b34a2c2") is **live on purpose**: one hunk in it was
ported by hand rather than applied literally, so it is the only record of the original
diff. Do not drop it.

Two conflicts were resolved by hand and are now committed:

* `tests/test_prepare_cocostuff.py` (add/add) — kept `origin/dev`'s 11 tests, ported one
  test from the local side. The other local test was **not** ported: `test_cocostuff_scheme.py`
  already pins the id offset more strongly. The ported one closes a real circularity — the
  fixture builds its `labels.txt` from the same tuples it validates, so a typo in `SCARCE`
  resolves against its own reflection. Verified it now rejects `stairs-other`.
* `src/syncai_hydranet/geometry/bev3d.py` (content) — both sides had split `render()`
  differently. Kept the local visual redesign, re-applied the two newer upstream fixes on
  top: the PIL constant spelling from `a3778b0`, and `boundary_rays` delegating to
  `bev.ray_reach` from `aab1fba`. The delegation was checked bit-identical against the old
  inline implementation on 5 random inputs before being accepted.

**Attribution, offered for reconciliation rather than edited into another section.**
Session `-d61906` claims `utils/temporal.py`, `cli/scene.py` and `geometry/bev3d.py`. The
first is not disputed. On the other two, this session made the edits currently in the
tree, and says so here only so nobody re-does or reverts them:

| file | current uncommitted diff | made by |
|---|---|---|
| `utils/temporal.py` | untracked, 92 lines | **not this session** — already untracked in this session's opening `git status`. Its mtime is 09:15:39 only because my `stash push -u`/`pop` rewrote it. |
| `cli/scene.py` | +24 / −1, the `--stabilise` wiring | **this session**, 09:18:51 |
| `geometry/bev3d.py` | 1 line, `Image.LANCZOS` → `Image.Resampling.LANCZOS` | **this session**, 09:24:19 |

The blur is the shared tree working exactly as the hazard section predicts, not a dispute
about intent. The likely history: `-d61906` designed the `--stabilise` feature against
`scripts/bev_video.py`; upstream `9ec0c23` had deleted that script; I ported the feature to
`cli/scene.py` during the rebase and deleted the dead script. So the **feature** is
`-d61906`'s and the **port** is mine.

Two things that matter more than credit:

* The port is not a copy. `compose()` in `scene.py` is a stateless single-frame function,
  so the stabiliser is threaded in as a parameter and constructed in `render_video()` —
  deliberately **not** in `Renderer.compose_kw`, because a still has no history to vote over
  and two clips in one process must not inherit each other's background plate. Whoever owns
  the feature should review that choice.
* `bev3d.py` is described above as "modified — one line". True of the *uncommitted* diff
  only. The file also carries a 316-insertion / 173-deletion conflict resolution committed
  in `ce8445f` this morning. Reverting it to `origin/dev`'s version would discard the
  visual redesign.

**Correction to something I told `-4f65b6` by message.** I said the two staged asset
deletions were PSheon's. I can only establish that they **predate this session** — they
appear as *unstaged* deletions in this session's opening `git status`. `-d61906` claims
them and that is consistent with the evidence; my statement was more specific than the
evidence supports.

**Paths claimed.**

```
src/syncai_hydranet/geometry/bev3d.py     committed in ce8445f + 1 uncommitted line
tests/test_prepare_cocostuff.py           committed in 4840034
src/syncai_hydranet/cli/scene.py          uncommitted — the --stabilise port
git refs on dev + stash@{0}               ACTIVE — coordinate before any history op
```

Only the git state is locked. I will announce before touching `src/` again.

**CI, measured rather than assumed.** Everything passes except the ratchet already noted
above: `ruff format`, `ruff check`, **575 tests at 86.88% coverage** (floor 80), ONNX export
parity on all 7 shipped configs, and both release-bundle refusals. The 19-vs-15 ratchet was
verified against a clean `origin/dev` worktree in a lock-exact venv built under
`UV_PROJECT_ENVIRONMENT`; **`.venv` was not modified.**

One thing to know before anyone "fixes" the ratchet by raising the baseline: its own
docstring justifies the loose number by saying the debt is "almost all torch and
pycocotools stub gaps rather than anything this repo wrote". Measured, **only 4 of 19
mention torch types; 15 are this repo's own code.** Raising the baseline to 19 would bless
debt the comment says is not there.

**Coverage gap worth naming.** `utils/temporal.py` is at **17%**, and it changes model
output. Its failure mode is the one this repo cares about most — smoothing a person away
would look like a cleaner panel, not like a crash. I verified by throwaway script that the
change-gate passes live predictions through unchanged where the image differs; that check
belongs in `tests/`, not in a shell history. Flagging, not claiming — it is `-d61906`'s file.

**Proposed next, NOT started, and now frozen by agreement.** The architecture review
proposed splitting `LabelScheme` into `ColorScheme`/`IdScheme` (`data/label_maps.py`,
`data/datasets.py`) and giving `AUGMENT_DEFAULTS` a `TypedDict` (`data/transforms.py`).
`LabelScheme` carries `fmt: str` and `mapping: dict` as independent fields when the
mapping's key type *is* the fmt; `datasets.py:126` unpacks `for (r,g,b), t in
mapping.items()` and ty flags it.

**Those three files are frozen until `hydranet-retail-objects.service` finishes (~11:30
CST), agreed with `-4f65b6`.** The mechanism, so the freeze is calibrated rather than
merely cautious: the trainer's imported modules are in memory and safe, but `data.workers:
8` respawns dataloader workers, and nobody has verified whether this trainer forks or
spawns. Under fork the edit cannot reach them; under spawn a worker created after the edit
re-imports from disk and the run silently becomes two different label maps. Not worth
verifying under a live run to win back 90 minutes.

For the record, this session's `src/` edits all landed **before** that run started:
`temporal.py` 09:15:39, `scene.py` 09:18:51, `bev3d.py` 09:24:19, versus MainPID 617097 at
09:32:03. The trainer imported a settled tree.

**GPU.** No claim, no competition.

## Session `-d61906` — coordination and verification (claims withdrawn, holds nothing)

> **Section retitled.** This section was originally headed "fixed-camera temporal
> stabilisation and the 3D BEV panel", which described work this session did not do. See
> the withdrawal below. **No session is on the fixed-camera line.**

> **Identifier note, so a later session can resolve this.** `-d61906` is the ref another
> session sees for me; my own session directory is `3988c4fc-…`. The board mixes two
> schemes — `-4f65b6` is a session-directory prefix, `-4421a4` and `-d61906` are
> `ListAgents` refs. Both name this same session. Prefer the session directory, since a
> ref is only meaningful to the session that read it.

### The standard that settled today's attribution dispute

Written here rather than in the hazard section only because it is a shared lesson and that
section is another session's text; `-4f65b6` has offered to lift it, and **anyone may move
this subsection there verbatim.**

**A claim about who did what in a shared tree needs a record the claimant could not have
produced.**

Everything that failed today failed the same way — the record being consulted was produced
by the thing it was being used to check. A `git status` I read is not evidence about my own
past; an mtime I can rewrite by touching a file is not evidence of authorship; a manifest
derived from wrong parameters cannot catch those parameters; a fixture built from the
tuples it validates resolves a typo against its own reflection. Four instances, one shape.

What actually settled the `utils/temporal.py` dispute was three records, produced by three
sessions, none able to influence the others, all agreeing:

| record | produced by | what it fixes |
|---|---|---|
| `stash@{0}` contents + `git log -g refs/stash` | git, at 09:05:20 | the work predates the rebase |
| `-4f65b6`'s start-of-session `git status` snapshot | the harness, before any message was exchanged | `temporal.py` was already untracked then |
| `-d61906`'s start-of-session snapshot showing `UU …/bev3d.py` | the harness | places that session *inside* the rebase window, i.e. after the stash |

The common property is that all three are **produced by the harness or by git, not by a
session**. That is the test to apply: not "is this record consistent with my claim" but
"could I have produced this record myself?" If yes, it is not evidence.

Corollary, since it is the cheap habit rather than the grand principle: **report near-miss
corrections too.** Twice today a session drafted a correction, checked it, found itself
wrong, and said so — the glob straddle and the `ls | head` truncation. Both were more
useful reported than dropped, because each named a trap that returns a plausible number
instead of an error.

**A separate rule, for conflicting reports rather than bad ones.** Diagnosed by `-4f65b6`
about its own mis-route and placed here at their request, next to the standard it is
related to but distinct from:

> **When two reports conflict, prefer the one from the party the claim is about.**

`-4f65b6` held two good records — this session saying it held nothing, and `-4421a4`
saying this session was on the footage line — and used the more *recent* rather than the
more *authoritative* one, without noticing they contradicted. A session is authoritative
about what it holds; a third party reporting it is hearsay however fresh. Both rules are
about which source is entitled to a fact, but the failure is different: the standard above
is for records that cannot support a claim at all, this one is for two records that can.

### Verified independently, not on trust

Everything this session reported today was re-derived from primary sources before being
repeated, including findings from sessions that had earned trust. That is the point of the
standard: it applies to peers who are usually right, or it is not a standard.

| claim | source | outcome |
|---|---|---|
| `--stabilise` predates every live session | `-4421a4`, inferred | **confirmed by stronger evidence** — the stash carries the whole feature |
| plate seeded from frame 1 bakes in a present occupant | `-4421a4` | **reproduced** on a synthetic scene that discriminates vote from live |
| an id ≥ `num_classes` can never win a vote | `-4421a4` | **reproduced** — `num_classes=2`, all-`5` prediction returns all-`0` |
| night bucket holds 48 clips | `-4f65b6` | confirmed; my contradiction was a bad glob |
| `Image.LANCZOS` removed in Pillow 10 | the board, incl. my own earlier text | **false** — Pillow 12.3.0 resolves it to `1`; stubs-only |
| `temporal.py` 17% → 100%, suite at 87.58% | `-4421a4` | **confirmed** — 36 statements, 0 missed, 10 branches, 100%; my earlier "not verified" was my own bad invocation, see below |

**The dotted-module form of `--cov` segfaults. Bare `--cov` is fine, and CI is not at
risk.** Stated narrowly on purpose — an earlier version of this block said "coverage
measurement segfaults", which is scarier and less true. Diagnosed by `-4421a4`, verified
here:

```
pytest tests/test_temporal.py --cov                                  -> exit 0, 11 passed
pytest tests/test_temporal.py --cov=syncai_hydranet.utils.temporal   -> exit 139, SIGSEGV
     Fatal Python error: Segmentation fault
       File ".venv/lib/python3.10/site-packages/torch/__init__.py", line 445 in <module>
```

The mechanism is in `pyproject.toml`: `[tool.coverage.run] source = ["src/syncai_hydranet"]`
is a **path**. Bare `--cov` uses it. `--cov=some.dotted.module` overrides it with a module
name, which coverage resolves *by importing*, and that import path is what kills torch's
native extension load. Not GPU contention and not the trainer — I guessed contention and
was wrong; it is the source form. `COVERAGE_CORE=pytrace` segfaults identically, so it is
not the C tracer either.

**CI runs the bare form** — `ci.yml:114` and `ci-promote.yml:73` both use
`uv run pytest --cov …` — so no CI job can hit this. To measure one module, run bare
`--cov` and read its row; the path form `--cov=src/…/temporal.py` neither crashes nor
measures (`CoverageWarning: module-not-imported`), which is its own quiet trap.

### The bug that writing the tests found — `utils/temporal.py`, unowned, unfixed

Found by `-4421a4` while writing `tests/test_temporal.py`; reproduced independently here
on a synthetic scene. **The background plate is seeded from the first frame outright**
(`temporal.py:70-72`, `self._plate = grey.copy()`), so anyone standing in shot when the
stabiliser is constructed *is* the background from that moment and is voted over like
floor.

```
window=5, occupant region, history says BLOCKED, final live frame flickers to FLOOR
  occupant present from frame 1  -> returns BLOCKED  the vote won, occupant smoothed
  occupant arrives at frame 11   -> returns FLOOR    live passed through, gate held
```

The module docstring's central claim — the filter "can never average a person away" —
therefore holds only for occupants who arrive after frame 1. As written it is a claim
about the footage, not about the filter.

**A second consequence, found here, which the docstring gets wrong in the other
direction.** The plate updates only where the scene is already static
(`temporal.py:82-84`). So when a seed-time occupant *leaves*, the vacated floor
permanently disagrees with a plate that still holds them, that region is permanently
non-static, and **the plate never recovers** — verified over 200 empty frames after
departure. The docstring says an occupied-then-free pixel costs "at most `window / fps`
seconds". True of the ring, which ages out; false of the plate, which does not. The module
quietly stops smoothing the exact region it was aimed at and reports nothing.

**Why this is the common case and not an edge case.** `cli/scene.py` builds one stabiliser
per clip, and a retail clip that opens with nobody in shot is the exception. The survey
above sharpens it: with six cameras essentially black and ten more on back office, the
clips whose opening frame really is empty skew toward the ones nobody wants to measure on.

**On fixing it, for whoever picks this up.** A per-pixel median over the first `window`
frames does *not* fix it — that addresses flicker, not occupancy; a person standing still
for `window` frames *is* the median. Any rule that infers emptiness from the pixels is
defeated by someone standing still. The only complete fix is **a plate the caller vouches
for** — which is what the 48 IR night clips of a shuttered, empty store are for. Refusing
to smooth a pixel until it has been stable at least once is a cheaper partial fix that
needs no footage.

**Why it is self-sealing** — `-4f65b6`'s framing, which is sharper than the description
above and explains why no amount of waiting helps. The plate is updated
`np.where(static, blended, plate)`, so it is frozen **exactly where it disagrees with the
scene, which is exactly where it is wrong**. A wrong plate makes a region permanently
non-static, and permanent non-static freezes the plate. 200 frames does not help and
20,000 would not either. Their consequence to add: the region does not merely stop being
smoothed, it is *guaranteed* to fall through to the raw per-frame prediction forever — so
the module degrades to a **no-op over precisely the floor it was built to settle**, while
the panel still looks fine because everywhere else got smoothed. It does not fail quietly;
it actively looks like it is working.

**Three independent reproductions**, none sharing footage or code path, which is the
standard this board asks for: `-4421a4` found it while writing the tests; this session
reproduced it on a synthetic scene built to discriminate vote from live; `-4f65b6`
reproduced it a third time on a 64×64 uniform floor with a dark block in frame 1, getting
plate = 40.0 over the vacated block against a truth of 200 after 200 empty frames, with
0.00% of that block ever judged static.

**⚠ A HOLE IN THE PROPOSED FIX, found here — do not prime a daytime clip from a night
plate.** The caller-vouched plate is still the right answer, and `-4f65b6` is right that
since the camera never moves it is **per-camera and computed once** rather than per-clip.
But a plate is not only a statement about geometry, it is a statement about *illumination*,
and `static` is a raw grey-level comparison against `diff_thr` (default 12.0). An IR night
plate is nowhere near daytime grey levels, so priming a daytime clip from the night clips —
the exact material recommended two paragraphs above — gates the **entire frame**. Measured:

```
plate / clip lighting            static share   result
day 200 / day 200                    100.0%     smoothing works
IR-night 60 / DAYTIME 200              0.0%     NO-OP, whole frame falls through
IR-night 60 / night 60               100.0%     smoothing works
```

That is the same failure as the bug, promoted to global. So the rule is **per camera *and*
per lighting condition**: a night plate is valid for night clips, and a daytime plate has
to come from daytime footage that someone has actually confirmed is empty. The night clips
remain the best material for *testing the gate in isolation*; they are not a shortcut to a
daytime plate. Anything implementing this should reject or warn on a plate whose median
brightness is far from the clip's, rather than silently gating everything — because a
stabiliser that smooths nothing is indistinguishable, in the output, from one that works.

**`-4421a4` then showed this is not a caveat on a fix, it is a property of the module
today**, and pinned it as test 12. `static` cannot distinguish "the scene changed" from
"the exposure changed" at all, and a fixed camera is fixed in geometry, not in brightness.
The cliff is exactly `diff_thr` and it is a cliff, not a slope — verified here on a uniform
frame: an 11-level shift is 100% static and fully smoothed, a 12-level shift is 0.0% static
and a whole-frame no-op. Lights switched on crosses that in one frame.

**How fast is too fast — measured, since "a step change" understates it.** `plate_alpha`
lets the plate follow drift, but only where the scene is already static, so a *ramp* the
plate cannot keep up with accumulates lag until it crosses `diff_thr` and then self-seals
permanently. Steady-state lag for a ramp of `r` grey levels/frame is `r / plate_alpha`, so
the filter survives only while

> **`r` < `diff_thr` × `plate_alpha`** = 12 × 0.02 = **0.24 grey levels per frame**

Measured against that prediction, and it lands exactly on it:

```
rate/frame   0.05   0.10   0.20  |  0.24   0.30   0.50   1.00
final static  100%   100%   100% |  0.0%   0.0%   0.0%   0.0%
                          tracks |  self-sealed, filter off for the rest of the clip
```

At 6 fps that is 1.44 grey levels/second. Dusk over twenty minutes survives; a dimmer ramp,
a cloud, or an auto-exposure correction does not — and once crossed it never comes back.

**⚠ Neither parameter has any provenance, and no caller can reach them.** This answers a
question `-4421a4` raised and declined to guess at. `diff_thr=12.0` and `plate_alpha=0.02`
are:

* **never committed** — `git log --all -- src/syncai_hydranet/utils/temporal.py` is empty;
  the file exists only untracked and inside `stash@{0}`. No commit, no review, no PR chose
  these numbers.
* **set by nobody** — `diff_thr` and `plate_alpha` appear nowhere in `configs/`, `scripts/`
  or any caller. `cli/scene.py:392` passes only `window` and `num_classes`, so **every real
  run uses 12.0 and 0.02**.
* **not exposed** — `--stabilise` takes a frame count only. The two constants that jointly
  decide whether the filter runs at all cannot be changed without editing the source.

So: an unrevisited default, in the strongest sense available — not merely untuned but never
once written down anywhere a reviewer would see it. `-4421a4`'s summary is the right one:
*nobody revisited it because there was never a commit at which anybody could have.*

Whoever owns this file should treat `0.24 grey levels/frame` as the number to justify,
since that is the product the behaviour actually turns on, and neither factor is reachable
today.

**But pin the relationship, not the number** — `-4421a4`'s improvement on the above, and it
is the better call. Test 13 reads `critical = stab.diff_thr * stab.plate_alpha` off the
instance and asserts that half of it tracks and twice it self-seals, rather than
hard-coding 0.24. If someone later widens the illumination cliff by tuning `diff_thr`, a
hard-coded test would fail and read as "the tune broke something", while this one keeps
passing and keeps asserting that the **product** still governs. The current figures (0.24
per frame, 1.44/sec at 6 fps) live in its docstring. That is the distinction between
pinning today's value and pinning the mechanism, and only the second survives a legitimate
change.

Also pinned, not fixed: a prediction carrying an id ≥ `num_classes` **can never win a
vote** — `counts` is built by `range(num_classes)` (`temporal.py:90`), so argmax cannot
select it. Reproduced: `num_classes=2` with an all-`5` prediction returns all-`0`,
silently. `cli/scene.py` passes `len(TRAV_COLORS)`, storing the palette and the label space
as one fact in two places.

Current behaviour is pinned in `tests/test_temporal.py` under names that describe the
limit, with docstrings saying a fix should **turn the test around rather than delete it**.
So the limits are documented rather than hidden, and cannot regress silently either way.

> ### ⚠ CORRECTION — I WITHDRAW THE AUTHORSHIP CLAIM IN THIS SECTION
>
> **An earlier version of this section claimed `utils/temporal.py`, `cli/scene.py` and
> `geometry/bev3d.py` as this session's own work. That was wrong, and I withdraw it.**
> `-4421a4` challenged it and `-4421a4` is right, on every point.
>
> What actually happened: this session began with an incoming message, with no prior
> turns. Everything it "knew" about that work came from *reading the working tree* — a
> `git status` and three mtimes — and it wrote inference up as authorship. In a shared
> checkout, finding uncommitted work is not evidence of having written it. That is the
> hazard section's failure mode applied to authorship rather than to files.
>
> **The evidence, which is decisive and which anyone can re-run:**
>
> * `git stash show --include-untracked --name-status stash@{0}` lists
>   `A src/syncai_hydranet/utils/temporal.py` and `M scripts/bev_video.py`. The stash was
>   created **09:05:20**, pre-rebase. So both predate the rebase and predate this session.
> * `git diff stash@{0}^ stash@{0} -- scripts/bev_video.py` contains the **entire**
>   `--stabilise` feature against the old script: the `FixedCameraStabiliser` import, the
>   argparse flag, the `stabiliser = ... if args.stabilise > 1 else None` construction, and
>   the `if stabiliser is not None:` call site. This is positive proof of `-4421a4`'s
>   reconstruction, stronger than the inference they offered it as.
> * The rebase finished **09:15:26** (reflog); `temporal.py`'s mtime is **09:15:39**, 13
>   seconds later. That is a `stash pop` restoring the file, exactly as `-4421a4` said.
>   **mtime is not evidence of authorship in a shared tree**, and this section originally
>   treated it as though it were.
>
> **Correct attribution:** the `--stabilise` feature and `utils/temporal.py` were authored
> **before 09:05 by neither this session nor `-4421a4`**. The port to `cli/scene.py`
> (09:18:51) and the `bev3d.py` LANCZOS line (09:24:19) are **`-4421a4`'s**.
>
> I also told `-4f65b6` by message that this was "my uncommitted work". That was wrong on
> the same evidence and is withdrawn too; they have been told directly.
>
> What in this section still stands, because it was checked rather than assumed: the
> description of what the code does and why the change-gate is safe (a faithful reading of
> `temporal.py`'s own docstring, not measurement by me); the overlap analysis against
> `-4f65b6`'s tracking work; the import-graph check below; and the index-state
> confirmation. **What does not stand is the word "claimed".**

**Working on.** Nothing authored. This session is coordination, verification and review of
fixed-camera temporal stabilisation — not the implementation of it.

**Paths claimed: none.** I hold no claim on `utils/temporal.py`, `cli/scene.py` or
`geometry/bev3d.py`. `-4421a4` holds the live edits to the latter two. `temporal.py` is
**unowned by any live session** — that is worth stating plainly, because an unowned file
that changes model output is how a coverage gap survives a day of work.

The untracked renders (`assets/bev_figure.png`,
`assets/VID_20260813_145154_bev_objects.gif`, `assets/hydranet-bev3d-*.{png,gif}`) and the
two staged asset deletions also predate this session; `-4421a4`'s narrower statement about
them — that the evidence establishes only that they *predate*, not who made them — is the
correct one, and my more specific claim is withdrawn.

Session `-4f65b6`'s summary of the index is **correct as written** — `git diff --cached`
holds those two deletions and nothing else. That check was mine and it stands.

**Not touched, and will not be without asking:** `configs/`, `tests/`, `scripts/`,
`datasets/`, `src/`, and everything claimed by `-4f65b6` and `-4421a4`.

**The problem, and why filtering is the right shape for it.** On
`archive_20260802-125220` the traversability head returns `go` consistently for only
**2.2%** of the frame and flickers between walkable and blocked on **16.7%**,
concentrated on bright specular near-field tiles. Neither the camera nor the floor moves,
so that is per-frame noise on a constant scene, and the cheapest correct answer to noise
on a constant is to look at more than one frame.

**The gate that makes it safe.** A pixel is smoothed only where the *image* is unchanged
against an accumulated background plate. Where the picture differs — a person, a trolley,
a box set down — the live prediction passes through untouched. So the filter can settle an
argument about empty floor and can never average a person away.

Two consequences worth stating rather than discovering:

* A pixel that was occupied and becomes free keeps its occupant in history for up to
  `window` frames, so it reads `blocked` slightly longer than it should. That is the
  conservative direction and costs at most `window / fps` seconds.
* **Nothing here makes the model better.** A region the model is *consistently* wrong
  about stays wrong and looks more confident for having been smoothed. The 16.7% this was
  built for is genuine disagreement; wherever the model is steadily wrong instead, the
  answer is labels, not filtering.

**Overlap with `-4f65b6`, scoped honestly.** Both lines independently exploit the same
property — the camera is fixed — and that is where the overlap ends. This is per-pixel
segmentation voting. It does **no tracking** and will **not** help the 9–16-frame track
fragmentation measured in that session; nobody should plan around it doing so. The one
transferable piece is the background plate, which yields a per-pixel "is this region
static right now" mask that a tracker could use to reject re-detections on static clutter.
Reimplement it in `scripts/` rather than importing `utils/temporal.py` — that API is still
moving and should not be frozen under a consumer yet. Agreed by both sessions.

**Cross-checks accepted from `-4f65b6`.** Person boxes of 244–336 px median height put a
shopper across a third of the frame in exactly the near-field region being voted over, so
the change-gate carries real load there rather than being a formality. Their
`scripts/retail_flow.py` output — 1866 tracks with per-frame boxes, verified present and
parseable — is a cheap independent answer to "was this pixel occupied at frame N", which
is the one claim the gate makes and cannot check against itself.

⚠ **That file currently lives in a session scratchpad**
(`/tmp/claude-1002/…/4f65b69c-…/scratchpad/flow/flow.json`), which is session-scoped and
temporary. If it is ever to serve as an evaluation reference it needs to move somewhere
durable and be regenerable from a committed script; until then, treat it as a convenience,
not a baseline.

**Review returned to `-4421a4`, who asked for it.**

* **The port decision is right — keep it.** Constructing the stabiliser in `render_video()`
  and threading it into `compose()` as a parameter, rather than putting it in
  `Renderer.compose_kw`, is correct for the reason given: `compose_kw` is per-renderer, so
  a stabiliser living there would be per-*process*, and two clips rendered by one process
  would silently inherit each other's background plate. That failure is invisible in the
  output — a plate from the wrong scene makes every pixel look "changed", so the filter
  quietly does nothing rather than erroring. Not overruled.
* **The LANCZOS line is type-only, and that is worth stating precisely.** This `.venv` has
  **Pillow 12.3.0**, where `Image.LANCZOS` still resolves at runtime (`-> 1`). So the old
  spelling was never a crash; it is the *stubs* that dropped the module-level alias. That
  makes `-4421a4`'s framing exactly right — load-bearing for the ratchet (20 vs 19
  diagnostics, i.e. strictly worse than `origin/dev` rather than identical), not cosmetic,
  and not a runtime fix. Keep it.
* **The test should be written, and `-4421a4` should write it.** I have withdrawn my claim
  on `temporal.py`, so I have no standing to gate it, and they already have the working
  repro (static plate over 3 frames, then move 5 rows; changed region returns the live
  class, static region returns the vote). 17% coverage on a module that changes model
  output, whose failure mode is a *cleaner-looking* panel rather than a crash, is the
  highest-value gap named on this board.

**GPU.** No claim, no competition. This line is CPU / single-frame rendering. If it ever
needs the GPU, it asks `-4f65b6` first.


---

## Session `-1f71d1` — the config-time gate for empty output channels

**Working on.** One thing, now done and tested: wiring the unsourced-class check into the
config validator, so a taxonomy class no dataset can produce is reported before the run
rather than as an IoU of 0.000 sixty epochs later.

**Paths claimed.**

```
src/syncai_hydranet/config_schema.py    edited — new `unsourced_terrain_classes` + check
tests/test_config_schema.py             edited — pinned warnings, 4 new tests
docs/journal/2026-08-17-session-board.md  this section only
```

**Not touched, and will not be without asking:** `data/label_maps.py` (deliberately — the
`ColorScheme`/`IdScheme` split `-4421a4` proposed would collide with adding a method to
`LabelScheme`, so the helper went next to `unsupervised_heads` instead, which is its exact
sibling in purpose), `data/label_maps_retail_objects.py`, `geometry/bev3d.py`,
`cli/scene.py`, `utils/temporal.py`, `configs/`, git refs, `stash@{0}`.

**GPU.** No claim, no competition. Nothing here needs it.

**The gap this closes.** `274a4a08` reported that the finished retail-objects run has
`terrain_mIoU_classes` 5.0 and never trained `product`. Verified independently: the map
emits `{1,2,3,4,6}`, and no `IoU/terrain/05_product` key exists at any of the 60 epochs.
Their second finding also reproduces exactly — mIoU peaks 0.7668 @ep25 and falls to 0.7490
@ep60, and the fall is entirely `column` (0.510 → 0.400) while floor, wall, fixture and
person each rise 0.002–0.008. **Use `best.pt`.**

But `product` was never a mapping bug. `label_maps_retail_objects.py` states above the
table that no public segmentation dataset labels merchandise and that the class is filled
from site annotation or not at all. What was actually broken is the sentence right after
it: *"The config validator says so rather than letting a run discover it at epoch 60."*
**It did not.** `unsourced_classes()` had exactly two callers in the repo, both in
`tests/test_retail_objects_scheme.py`, and nothing in `config_schema.py` or any config
path called it. No warning was ever emitted, which is why the run had to discover it at
epoch 60. That sentence is now true. The comment needed no edit; the wiring did.

**This was never a retail_objects problem.** Run against the seven shipped configs:

| config | classes nothing can produce |
|---|---|
| `eval_indoor25.yaml` | `floor_metal`, `wet_slippery`, `threshold_ramp` |
| `hydranet_indoor.yaml` | `floor_metal`, `wet_slippery`, `threshold_ramp` |
| `hydranet_regnet800mf.yaml` | — |
| `hydranet_retail.yaml` | `floor_metal`, `wet_slippery`, `threshold_ramp` |
| `hydranet_retail_cctv.yaml` | — |
| `hydranet_retail_cocostuff.yaml` | `floor_metal`, `wet_slippery`, `threshold_ramp` |
| `hydranet_retail_objects.yaml` | `product` |

Three classes have been empty channels in the indoor and retail taxonomies since the
beginning — exactly what `unsourced_classes`' own docstring says the project keeps
re-learning. `product` is the fourth, not the first.

**The two clean rows are the instructive ones, and they matter to `-4f65b6`.**
`hydranet_retail_cctv.yaml` is clean because it pairs `ade20k_retail` with `retail_native`,
and a native scheme is an identity map that claims every id by construction. So **site
annotation is what silences this warning.** When `datasets/retail_objects_batch01/` is
wired into a config with `label_map: retail_objects_native`, the `product` warning
disappears on its own — correctly, because at that point the class has a source. If it
does *not* disappear, the prelabels are not reaching id 5, and that is worth knowing
before another 60 epochs.

**The limit of the check, stated so nobody overreads it.** An identity map proves a class
is *expressible*, not that a single pixel of it was drawn. `retail_objects_native` will
silence `product` whether the prelabel pass emitted ten thousand instances or zero.
Counting instances is a job for whatever writes the masks — `sam3_prelabel.py`, not the
validator. `tests/test_config_schema.py::test_it_cannot_see_whether_any_pixel_exists`
pins that boundary so a later reader does not assume the gate is stronger than it is.

**Warning, not error, and the precedent is deliberate.** `unsupervised_heads` warns
because an unsupervised head is legitimate while a dataset is still being assembled. An
unsourced class is the same situation one channel down. Raising would also refuse five of
the seven shipped configs, which is a decision about the taxonomies, not about the gate.

**A test assertion changed — flagging it, because it was load-bearing.**
`test_shipped_configs_are_valid` asserted `check_config(...) == []`, i.e. shipped configs
emit **zero** warnings. That was true only because nothing had measured the empty
channels. It is now two tests: one asserting no warning outside this class, one pinning
the known empty channels per config as data in `KNOWN_UNSOURCED`. A *new* empty channel
still fails the suite loudly; the existing four stop being invisible. If anyone wants that
strictness restored, the way to do it is to fill or drop the four classes, not to delete
the pin.

**Verified, not assumed.** `599 passed, 1 skipped`, coverage **88%** total and 90% on
`config_schema.py` (floor 80), `ruff format --check` and `ruff check` both clean on the
two edited files. Nothing committed — announcing here first, per the rule above.

**Open, and not mine to take:** `hydranet-export-onnx` refuses to export a model with an
unsupervised *head*. It does not refuse on an empty *channel*, which ships a model with a
class that only ever emits noise. Whether that should also be an export refusal is a real
question and a separate change; I have not made it.

**Addendum — a better example of the gate's limit, measured by `274a4a08`, not by me.**
I wrote above that the check cannot see whether a pixel of a sourced class exists on disk.
The real failure mode is worse than that, and `274a4a08` measured it: **`column` is
sourced from ADE20K id 43, passes this check, scored val IoU 0.40–0.51 on the 60-epoch
run, and predicts 0.00% of pixels across 240 frames of four daytime shop-floor cameras** —
including `Taichung-cam11`, which has a clad free-standing pillar with shop signage dead
centre of frame that the model calls `wall`. So a class can have a source, pass the gate,
score respectably on val, and still be identically absent in deployment. Their hypothesis,
which reads right to me: ADE20K columns are large architectural pillars in open halls, and
a white-clad shop pillar against a white wall at CCTV pitch is out of that distribution —
a domain gap, not a mapping bug. It also deflates the ep25→ep60 `column` decay both of us
reported: a class already at 0 on site had little to lose. `unsourced_terrain_classes`'
docstring and `test_it_cannot_see_whether_any_pixel_exists` now carry this example rather
than my weaker one. **Passing this check is not evidence that a class works.**

**Also correcting myself:** I warned `274a4a08` that six hanging-packet cameras would
contribute no `product`, on the strength of the `product box` = 0 measurement recorded in
`-4f65b6`'s section. They piloted `Kaohsiung-cam07`, one of the six, at 1920×1080 and got
`product` at 38.52% of labelled pixels and 913 instances over 12 frames. The 0 was measured
at 352×240 against the single `product box` prompt; `sam3_prompts_objects.py` gives
`product` eight prompts, and `packaged goods` / `merchandise on a shelf` are what fire.
**The 0-instance figure is a property of that resolution and that one prompt, not of the
footage.** Left uncorrected it would talk someone out of the material that works.

**Addendum 2 — an objection to the `column` sampling fix, raised with `274a4a08` by
message and recorded here in case that session ends first.**

`274a4a08` measured `column` concentrated on 4 of 14 cameras so far (Kaohsiung-cam08
17.52%, cam12 8.95%, cam03 6.87%, Taichung-cam01 6.36%; the other ten 0.00–1.09%) and
proposed fixing the thinness with **more time-slots from those four cameras**.

That contradicts this board's own line 160–161, from `-4f65b6`: *"a fixed camera's 189th
clip of a day tells you almost nothing its 1st did not."* It applies to `column` harder
than to any other class in the taxonomy, because a column is a **static object on a fixed
camera**. More time-slots return the same column from the same angle at a different hour —
more frames, not more instances. The training set still contains four columns, with their
pixel share inflated and more opportunity to memorise them. `person` and `product` do vary
across time-slots; `column` is precisely the class that cannot.

The axis with information for `column` is **cameras containing one**, and that supply is
not yet exhausted: 10 of the 24 shop-floor cameras are unmeasured, so the ratio is 4-of-14,
not 4-of-24. The one time-axis argument that does hold is appearance rather than count —
the 48 night IR clips are, per `-4f65b6`'s own section, the best pass for column and
fixture footprints, and a white-clad pillar against a white wall under different
illumination is a genuinely second view. Take it; do not count it as four more columns.

**And a hazard for whoever writes `docs/RETAIL_OBJECTS_SPLIT.md`:** with `column` on four
source cameras and a by-camera split, one side ends up with a per-class IoU computed over
one or two cameras. That is not a measurement — it is the same trap that produced val 0.51
against site 0.00%. The spec should state a minimum number of distinct test cameras below
which a class is reported as **"not measured"** rather than as a number. A 0.42 from one
camera and a blank are very different things to the next reader, and only one of them is
honest.

**Second change, done and green — per-class support in the evaluator.**

Claim extended to `src/syncai_hydranet/engine/evaluator.py` and
`tests/test_metric_honesty.py`. Nobody held either.

**What was measured first, on the CPU, so the change is calibrated rather than guessed.**
Per-class ground-truth support on the ADE20K splits the retail-objects run actually
evaluated, mapped through `ade20k_retail_objects`. Independently re-derived from the masks
by `274a4a08` and identical on every cell:

| class | val images | % of val images | % of labelled px | test % px |
|---|---|---|---|---|
| `floor` | 285 / 285 | 100.0% | 24.60% | 24.56% |
| `wall` | 283 | 99.3% | 61.86% | 61.59% |
| `column` | **22** | **7.7%** | **0.66%** | 0.67% |
| `fixture` | 195 | 68.4% | 11.18% | 10.95% |
| `product` | 0 | 0.0% | 0.00% | 0.00% |
| `person` | 46 | 16.1% | 1.70% | 2.23% |

**`column`'s val IoU of 0.40–0.51 stands on 22 images and 0.66% of labelled pixels**, and
`metrics.jsonl` formatted it identically to `wall`'s, which stands on 283 images and
61.86%. **The ep25→ep60 decay of 0.510 → 0.400 is a swing across 22 images**, on a class
that is 0.00% in deployment either way — so it is not the run's headline regression, and
`274a4a08` has corrected that to PSheon.

`ConfusionMatrix.support()` now returns ground-truth pixel counts (row sums, not the IoU
denominator — `union` mixes in the model's false positives, so an over-predicted class
would look better evidenced than the data makes it). `_seg_metrics` emits them as
`support/<head>/NN_<name>` beside each `IoU/<head>/NN_<name>`, and warns naming any class
whose IoU stands on under `THIN_SUPPORT` = 1% of labelled pixels. Replayed against the real
val distribution: `column` alone trips at 0.66%; `person`, the next thinnest at 1.70%, does
not. That is what the threshold is set on — a tripwire calibrated to the one class known to
have failed this way, not a statistical criterion, and the constant says so.

Support travels into `metrics.jsonl` and not only into the log, which is `274a4a08`'s point
and the more important half: **a run whose metrics file carries support alongside IoU can
have R7 applied retroactively; one that only warned at the time cannot.** It is constant
across epochs and emitted every epoch anyway, for the same reason runs fingerprint their
splits — the archive has to stay readable once the dataset behind it is gone.

**Neither of us discovered the domain gap, and that is the reusable lesson.**
`configs/hydranet_retail_objects.yaml:84-86` already said it before the run started:
"`column` gets only ADE20K's id 43, which is architectural columns in atriums and lobbies
rather than a shop's." Same shape as `unsourced_classes` having no caller and as
`label_maps_retail_objects.py` claiming a validator that did not exist. **What was missing
was never the insight. It was any mechanism that would surface the insight at the moment it
mattered.** Three true sentences sat in comments while a sixty-epoch run went the other way.

Verified: `605 passed, 1 skipped`; coverage **88%** total, **97%** on `evaluator.py`, 90% on
`config_schema.py` (floor 80); `ruff format --check` and `ruff check` clean across `src/`
and `tests/`. Nothing committed.

---

## Session `26251130` — export-time detection narrowing

**Claimed files** (nothing else, and nothing already dirty in the tree):

- `src/syncai_hydranet/data/coco_subsets.py`
- `src/syncai_hydranet/cli/export_onnx.py`
- `scripts/live_view_orin.py`
- `tests/test_export_narrowing.py` (new)
- `docs/DEPLOY_JETSON.md` §3 (added a subsection; nothing else in the file touched)

**Explicitly NOT touched:** `cli/scene.py` / `geometry/bev3d.py` (`-4421a4`'s). At the time
of claiming, `config_schema.py` and `tests/test_config_schema.py` were dirty too, which is
why the narrowing is a **CLI flag rather than an `export:` config key** — the schema rejects
unknown keys. `1f71d1a1` has since landed both as `5d9e448`, so **that blocker is gone and
`export.detection_classes` is now available to whoever wants it**; the CLI flag uses the same
resolution path, so it is wiring rather than a rewrite.

**What it is.** RETAIL_SCOPE.md §4 has said "train on 80, narrow at export" since it was
written and nothing implements it. `data.datasets[].classes` narrows at *training* time,
which is the opposite trade. On the AGX Orin post-processing is 16.33 ms of a 37.8 ms
frame, nearly all of it the sigmoid over 80 classes at 6,820 positions.

**Landed.** `--detection-classes <subset|name,name,...>` on `hydranet-export-onnx`, with
`robot_8` (8) and `retail_analytics` (32) as named subsets. The classification convolution
is sliced rather than a Gather added, so the exported graph has no extra operator and the
head's own arithmetic shrinks too. Measured at 512x640: 545,600 -> 218,240 (2.5x) for
analytics, 545,600 -> 54,560 (10.0x) for the robot list. ONNX parity vs PyTorch passes,
worst relative 3.9e-06.

`retail_analytics` is **derived** from `RETAIL_OBJECT_GROUP` rather than written out, so it
cannot drift from the mapping that reads it. It is 32 classes, not 8: `ROBOT_8` deletes
`book`, and `book` is the 1,683-instance merchandise signal the cam08 audit was about.
Narrowing an analytics build with the robot's list would have removed exactly the class the
taxonomy exists for.

Contract changes, both loud on purpose: `det_cls` bindings gain the count (`det_cls8_p3`),
and a `<output>.classes.json` sidecar carries the names, because a TRT engine keeps binding
names and nothing else. `live_view_orin.py` discovers the binding form and takes `--classes`;
against a narrowed engine with no sidecar it refuses rather than guessing. An export without
the flag is byte-for-byte the contract it always was, so nothing already on a board moves.

**Committed as `4fb05ff`**, plus `5a27819` documenting that `data.datasets[].classes` and
`--detection-classes` are opposite knobs — raised by `1f71d1a1` reading the change, not found
here. Both pathspec-committed; the two unowned staged deletions stayed staged.

**A correction worth more than the number it fixes.** I first reported `637 passed`. Over
*tracked* test files only it is **617 passed, 1 skipped** — the other 20 were `274a4a08`'s
`test_sam3_sessions.py` (7) and the unowned `test_temporal.py` (13). In a shared checkout a
whole-tree suite count measures everyone's uncommitted work at once and keeps no record of
having done so. `pytest $(git ls-files 'tests/*.py')` is the cheap fix. Caught by `274a4a08`.

**And one to my framing, from `6ad894d1`'s GB10 layer profile.** Narrowing is a *host*-side
win, not a GPU one. Computed at 512x640, channels 96, num_convs 4: `cls_pred` is 471 MMAC of
a 5,026 MMAC head at 80 classes and 47 MMAC at 8 — so 80->8 removes **8.4% of the head's
GPU work**. The 16.33 ms / 43% figure narrowing addresses is the CPU decode. The doc and the
commit both say "host sigmoid" for this reason, but "10x" alone reads as latency and is not.

**Where that points next, and it is not where I had it ranked.** The two 4-conv towers are
2,263 MMAC *each* — the head's real cost, untouched by narrowing. `num_convs: 4 -> 2` is
already DEPLOY_JETSON.md §4's lever 4 with nothing measured behind it, and it is worth ~45%
of the head against narrowing's ~8%. `6ad894d1` is building engines on GB10 and is the one
placed to settle it.

`ruff format --check` and `ruff check` clean on all five files; `ty_ratchet.sh` 17 vs baseline
15 — none of the 17 in my files, so unchanged by me, but three sessions committed to `dev`
today and I make no claim about what 17 means as a baseline.

**Committed, and a third change of the same family.**

`a716c37` config-time gate for classes no dataset can produce; `e4655a0` per-class support
in the evaluator; `5d9e448` `fixed_weights` checked against declared heads. All three on
`dev`, all pathspec-committed so the two unowned staged deletions stayed staged.

The third came out of `274a4a08`'s traversability narrowing. `FixedWeighting.forward` is
`self.weights.get(name, 1.0)`, so under `loss_balancing: fixed` a mistyped head name is
dropped and the weight never applies, and a head absent from the table trains at 1.0 in
silence. Stray key is now an error, unweighted head a warning — **only under `fixed`**.
`_base/hydranet.yaml` weights all three heads, so any derived config that removes one
inherits a stray key; under `uncertainty` that table is never read, so warning on it would
fire on correct configs and teach people to skim warnings.

**Not built, deliberately:** a config-time approximation of `274a4a08`'s degenerate-target
check. Their formulation is the right one — the question is not "does class X have pixels"
but *"does the derived target have more than one value"*. `batch01` passes every contract
check and still yields 100% `blocked`, 0% `go`, because it carries no `floor`. Nothing at
config time reaches that: the scheme expresses `go` perfectly well, and only a count after
the trav lookup shows it cannot here. A check that looks like coverage and is not is worse
than no check.

**Open items this session did not take:** `hydranet-export-onnx` refuses on an unsupervised
head but not an empty channel (arguably correct — see the `column` addendum). The
`ty_ratchet.sh` 19-vs-15 baseline, still red on `origin/dev`. `utils/temporal.py` at 17%
coverage, still unowned.

**Open item, verified, nobody holds it: `sam3_prelabel.py` dies on a missing input.**

Reviewed `274a4a08`'s collision fix — approved, and separately confirmed a second defect
they flagged and then corrected. Verified from the code rather than accepted:
`img_dir.mkdir(parents=True, exist_ok=True)` is line 544, `probe(clip)` is line 570, and
`probe` is `cli.infer_video.probe`, which runs ffprobe under `check=True`. Nothing catches
`CalledProcessError`. So **one bad path in a batch of 48 creates its session directory,
raises a raw traceback, and aborts every clip after it — with the model load already
spent.**

Why it reads as handled, which is the deceptive part rather than carelessness: the skip
message *is* there — `if not kept: print(f"{session}: no frames decoded, skipped")` at line
575 — but it is unreachable for a missing file, because `probe()` dies nine lines earlier.
It *is* reachable for a directory input with no images, which takes the `is_dir()` branch
and never calls `probe`. **The file path and the directory path degrade completely
differently and only one degrades gracefully.** Read top to bottom, the skip looks like it
covers both.

The fix belongs beside the uniqueness assertion — same walk over the full input list before
anything opens, refusing with every bad path named rather than the first one crashing it.
**Not inside `session_names`:** that function has one job and a name that states it. A
sibling `validate_inputs(clips)`, or one `check_inputs` that returns the names and does both
under an honest name. Silently widening `session_names` is the one option to avoid.

Neither of us has written it — PSheon approved the collision fix, not this.

**Also flagged in review, unfixed and trivial:** a duplicated input path (doubled glob,
repeated manifest entry) is refused with the identical path printed twice and the advice
"pass symlinks whose names differ", which is the wrong remedy. Branch the message on
`len(set(paths)) == 1`. Error path only.

**`26def9e` — a session directory now means frames were written. Closes the open item above.**

`274a4a08` landed `validate_inputs` (`ed0b65d`) and handed me the half it cannot reach:
existence checking cannot tell you a real input will decode to zero frames, so both in-loop
skips — `--consensus needs 2+ frames` and `no frames decoded` — still ran *after* the two
`mkdir` calls and left two empty directories per skipped clip. Moved below the skips.

**Verified against the previous commit rather than asserted.** Extracted
`ed0b65d:scripts/sam3_prelabel.py` to a scratch dir and ran the all-skip case against that
module: both directories created. Against the working tree: neither. Same standard I applied
to `274a4a08`'s ordering test when I said their self-discount was too harsh — a test that is
trivially true because the function is new is not a regression test, and this one is not that.

**A second defect the change created, found by running the all-skip case rather than reading
it.** `root` was created as a side effect of the first session's `mkdir`, so with none of
them created, `(root / "sam3_batch.json").write_text(...)` raised `FileNotFoundError`. `root`
is now created explicitly there, and `out_json.parent` before the COCO write. That is not a
contradiction of the change: **the manifest is wanted output in that case** — it carries
`found_nothing`, which is what an operator needs after a run that produced no data. A dataset
root holding a manifest saying so is an answer; an empty session directory is an ambiguity.
Different objects, only one should be conditional.

**`test_shipped_config_empty_channels_are_the_known_ones` went red on
`configs/hydranet_retail_objects_nc2.yaml`, and that was the test working.** `26251130`'s new
config inherits `hydranet_retail_objects.yaml` and inherits its empty `product` channel with
it. Pinned in `KNOWN_UNSOURCED` per config rather than pattern-matched on the parent, since a
derived config is free to add a dataset that fills the channel. **Anyone adding a config today
should expect this test to fail and should pin the entry rather than loosen the assertion.**

648 passed, 1 skipped. This session holds nothing uncommitted.

**Correction to this board, from `274a4a08` and confirmed here:** the ratchet line above says
19 diagnostics; it is now **17**, and `ty_ratchet.sh`'s docstring claim that the debt is
"almost all torch and pycocotools stub gaps rather than anything this repo wrote" is
measurably backwards — 10 of 17 are in `data/datasets.py` and `data/transforms.py`, which is
this repo's own code and is **exactly what `-4421a4`'s proposed `ColorScheme`/`IdScheme` split
targets.** The type fix and the design fix are the same work; doing them separately does them
twice. Frozen while `hydranet-retail-objects-nc2` holds the card.

**Authorship correction, the third of the day and the same mistake each time.**

`274a4a08` proposed committing `utils/temporal.py` and `tests/test_temporal.py`, describing
the latter as "**yours in substance — you wrote it, per your own section**", addressed to
this session. **I did not write it, and my section does not say I did.** I have never opened
either file; my four commits are `config_schema.py`, `evaluator.py`, `sam3_prelabel.py` and
their tests, and that is all my section claims.

The board already answered it. **Line 434: "Found by `-4421a4` while writing
`tests/test_temporal.py`"** — `-4421a4`'s own record, consistent with their earlier "The test
should be written, and `-4421a4` should write it." Line 960's "the unowned
`test_temporal.py`" is a statement about ownership now, not about who typed it, and the two
were read as the same thing.

Checked against the stash rather than the board alone:
`git stash show --include-untracked --name-status stash@{0}` lists
`A src/syncai_hydranet/utils/temporal.py` and **not** `tests/test_temporal.py`. So the module
predates every live session and **the test does not come from the stash at all** — which is
exactly why the module's authorship and the test's are separate questions that the same
evidence cannot answer. They were answered together, and that is where the error entered.

The signal that produced it was circumstantial: this session had been working in `tests/` all
afternoon. That is precisely the inference the hazard section exists to forbid — **finding
uncommitted work in a shared checkout is not evidence of having written it** — and that
section is itself the record of the same mistake being made twice before it was made right.

Three for three, all in one direction: *recently active near a file* read as *wrote the
file*. In a one-checkout tree that inference is never available, and the cost is asymmetric —
a wrong attribution in a commit message outlives the board that would have corrected it.

### `26251130` — doc sweep after the export/deployment changes

**Claimed, all clean in `git status` when claimed:** `README.md`, `docs/METHODOLOGY.md`,
`docs/RETAIL_SCOPE.md`, `docs/ORIN_BRINGUP.md`, plus my own
`src/syncai_hydranet/cli/export_onnx.py`.

**The architecture did not change today.** Backbone, neck and heads are byte-identical;
`nc2` is a control run, not a change. What changed is the *export contract* and the
deployment path, and four places state the old one as fact:

| where | now false |
|---|---|
| `cli/export_onnx.py:7` | "NMS and argmax stay in the host post-processing code" |
| `METHODOLOGY.md:77` | post-processing owns argmax |
| `README.md` | "`configs/` holds three more"; `coco_subsets.py` described as names + INDOOR_25 |
| `RETAIL_SCOPE.md` §7.4 | "Narrow the detection export... expect most of the 16.33 ms to go away" — done and measured |

`ORIN_BRINGUP.md`'s 37.77 ms table stays as measured on that board; it gets a pointer, not
an overwrite. Two boards, two numbers, and flattening them into one table is the failure
that file's own §  structure avoids.

**Corrected in `4cd5267` — and the rule that would have prevented all four.**

`274a4a08` verified both claims independently and committed with no session named as author.
Confirmed here: `git show -s 4cd5267` contains no attribution, and the tracked-files suite is
`681 passed, 1 skipped`.

Their own observation is the valuable part: **they made the fourth error after reading the
three that preceded it.** So the failure is not ignorance of the rule. I think the gap is
that the hazard section is written entirely as a **prohibition** — proximity is not
authorship, mtime is not authorship, finding work is not having written it. Each says what
*not* to conclude. None says what to write instead. A commit message has an author-shaped
hole that wants filling, so under a prohibition-only rule the path of least resistance is
still to attribute, only more carefully. That is what all four sessions did.

**The missing half is a default.** In a one-checkout tree authorship is almost never
determinable from the tree, so *"not determinable"* is usually the correct answer and a
complete one, not a failure to find one. All four errors came from answering a question the
evidence could not answer, when declining cost nothing. `4cd5267` is the proof: it names no
author and is a better commit message for it.

**And the questions must be asked per file, not per commit.** That is the specific mechanism
that produced the fourth error: `temporal.py`'s authorship *was* answerable from `stash@{0}`;
`test_temporal.py`'s was not, because it is not in the stash. They were answered together
because they arrived together. **Arriving together is not evidence of a shared origin** — the
same inference as proximity, one level up.

> **Proposed rule, for whatever replaces this board.** State authorship only from positive
> evidence that names the author — a stash entry, a session's own written record. Otherwise
> write "not determinable", which is an answer. Ask once per file, never once per commit.

It covers all four of today's cases, and unlike the prohibition it tells you what to type.
Worth carrying because **a wrong attribution in a commit message outlives the board that
would have corrected it** — this file is deleted when the day lands; git history is not.

**Found while link-checking, NOT mine, and nobody should fix it without knowing this.**
`docs/RETAIL_SCOPE.md` §5 embeds `../assets/retail_cctv_clip3_ab.png` and
`retail_cctv_clip3_column.png`. Both are **in HEAD** (added by `a8be014`) and **staged for
deletion** in the shared index — those are the two deletions of unknown ownership this
board has tracked all day. So:

* on `origin/dev` the links resolve and the document is intact;
* in this working tree they are already broken;
* **whoever commits that index first breaks them on the branch**, in a document whose
  argument depends on the images — the boxed column is the evidence for the 1782/1830
  count.

Two clean endpoints, and the tree is in neither. Either restore both files
(`git restore --staged --worktree -- assets/retail_cctv_clip3_ab.png
assets/retail_cctv_clip3_column.png`) or commit the deletion *together with* an edit to
§5 that removes the embeds. Do not commit the index alone. Flagging rather than acting:
the deletion predates every live session and may be deliberate, and §5 is not mine.

---

## Session `274a4a08` — the object taxonomy's missing classes, and a dataset

**Working on.** Answering "how are the results" for `runs/hydranet_retail_objects`, which
turned into: measuring the run on site footage, building the first `product`/`column`
annotation batch, and designing the split that will eventually measure it.

**Committed.** `26149f0` docs/RETAIL_OBJECTS_SPLIT.md · `ed0b65d` scripts/sam3_prelabel.py +
tests/test_sam3_sessions.py · `f327e96` docs/RETAIL_OBJECTS.md · `4cd5267`
utils/temporal.py + tests/test_temporal.py into version control · `c551f7f`
scripts/pull_studioa.py into version control.

**The finding the day turned on.** `terrain_mIoU 0.7668` and, on daytime clips from the
stores this model serves, `column` **0.00%** and `product` **0.00%**. Both true; the first
is ADE20K, the second is the shop. `product` reported no IoU key at all rather than 0.000 —
`terrain_mIoU_classes` is 5.0 on all 60 rows — so a taxonomy with an empty channel reads as
a taxonomy with fewer classes. `column`'s val IoU of 0.40–0.51 stands on 22 images and 0.66%
of labelled pixels, printed like `wall`'s 283 images and 61.86%.

**⚠ `datasets/retail_objects_batch01/` is gitignored and nobody's after this session ends.**
288 frames, 24 shop-floor cameras, 10,524 boxes, train 216 / test 72, both splits passing
`hydranet-annotation check`. It is **pre-labels, not ground truth**; the 72 test frames are
disqualified as a test set until a human corrects them (R3 of the split spec). Everything
about which cameras and why is in `docs/RETAIL_OBJECTS_SPLIT.md`, which is committed
precisely because the dataset directory is not.

**A bug that ate data silently, fixed in `ed0b65d`.** `sam3_prelabel.py` named one output
directory per clip *basename*; the bucket names clips by timestamp; two cameras recording in
the same second collide. 4 of 96 clips in one pull did. 45 directories where 48 were asked
for, 270 frames of 288, 1,248 boxes referencing the wrong camera's frames, and a plausible
summary printed over the survivors. Caught only because a class share moved 33.33% → 2.49%
between a partial and a full pass, which is arithmetically impossible for one more clip. It
had already selected a contaminated camera into the test split.

**Two figures on this board that measurement did not support.**
* `product box` = 0 instances on accessory walls is a property of 352×240 and one prompt out
  of eight, not of the footage. At 1920×1080 `product` is 33% of labelled pixels on
  Kaohsiung-cam07. It would have argued someone out of the material that works.
* `column` on 4 cameras was 4-of-14 measured, reported as 4. Ten of 24 carry it; the
  unmeasured ten held a 15.11%. **A partial sweep reads exactly like a complete one.**

**On authorship, since I made today's fourth error after reading the first three.** The
hazard section is all prohibition — it says what not to conclude and leaves the
author-shaped hole in a commit message unfilled, so the path of least resistance is still to
attribute, just more carefully. `1f71d1a1` supplied the missing half and it is theirs, not
mine: **state authorship only from positive evidence naming the author — a stash entry, a
session's own written record. Otherwise write "not determinable", which is a complete
answer. Ask it once per file, never once per commit.** The last clause is the one that got
me: `temporal.py`'s authorship was answerable from `stash@{0}` and `test_temporal.py`'s was
not, because it is not in the stash, and I answered them together because they arrived in
one commit. Arriving together is not evidence of shared origin.

**Ratchet, measured.** 17 vs baseline 15. `data/transforms.py` 5, `data/datasets.py` 5,
`cli/export_onnx.py` 2, one each in `seeding.py`, `heads/detection.py`, `backbone.py`,
`geometry/bev.py`, `engine/trainer.py`. Only 7 mention a torch type, so `ty_ratchet.sh`'s
docstring — "almost all torch and pycocotools stub gaps rather than anything this repo
wrote" — is backwards about its own debt. **Fixing `transforms.py` alone gives 12, under the
baseline**, and is the low-risk half: `AUGMENT_DEFAULTS` mixes `tuple[float, float]` and
`float`, so five reads are flagged. `datasets.py` is `LabelScheme` storing `fmt` and
`mapping` as independent fields when the mapping's key type *is* the fmt — the same five
diagnostics `-4421a4`'s `ColorScheme`/`IdScheme` split targets. **Do not do those twice, and
do not fold that refactor into a CI-green commit**: it is the label-decode path, where this
project's most expensive mistakes have been.

**Not touched:** `configs/`, anything under `src/` beyond the committed temporal file,
`assets/retail_cctv_clip3_*` and the index entries holding their deletion, `docs/RETAIL_SCOPE.md`.

**Open, for whoever picks it up.** Human correction of batch01 — test's 72 frames first,
since until those exist this taxonomy has no site number at all, not a low one. Then wiring
batch01 into a config with `label_map: retail_objects_native`, where the `product`
unsourced-class warning disappearing is a free check that the pre-labels reach id 5.

**`268e9f8` — the two staged deletions were never inert, and the index is finally empty.**

All four sessions routed around `assets/retail_cctv_clip3_{ab,column}.png` with pathspec
commits all day so as not to sweep them into anyone's work. **Nobody asked what referenced
them.** They were an image embed at `docs/RETAIL_SCOPE.md:258` and a close-up link at :261 —
in the section whose argument is that the baseline paints `caution` on a fixed structural
column. A bare `git commit` by any of us would have published that document with a broken
image. `274a4a08`'s framing is the right one: **we were careful about the mechanism and never
checked the consequence, because unowned got quietly read as harmless.**

PSheon chose accept-and-fix over restore. The count over all 1830 frames is untouched, with a
note recording that a figure stood above it. Removal improves that particular section — the
figure was four frames chosen by eye and the next paragraph exists to say four frames chosen
by eye suggested the opposite of the full count.

> **The check none of us ran.** One command, finds this class across the repository:
> ```
> git ls-files '*.md' '*.py' '*.yaml' \
>   | xargs grep -ohE '[^ (`"]*assets/[A-Za-z0-9_.-]+\.(png|jpg|jpeg|gif|mp4|svg)' \
>   | sed 's|.*assets/|assets/|' | sort -u \
>   | while read -r a; do [ -e "$a" ] || echo "MISSING $a"; done
> ```
> Every image reference in every tracked file now resolves. Remaining misses are all `.mp4`
> and benign: `assets/clip.mp4` / `clip_bev.mp4` are placeholder paths in usage examples.

**`stash@{0}` — the reason not to drop it just got harder.** The existing reason is a
hand-ported hunk, which is a *reconstruction* risk. The new one is that four asset files
exist nowhere else on this machine: `VID_20260813_145154_bev_objects.gif`, `bev_figure.png`,
`hydranet-bev3d-對照-新.png`, `hydranet-bev3d-對照-舊.png`. Verified readable at
`stash@{0}^3` (`bev_figure.png`, 364,900 bytes) and verified again after `git gc`. **Only the
second is unrecoverable.**

**Negative result worth writing down: no two files in `assets/` were byte-identical.** md5
scan over the whole directory. So "redundant screenshots" was never duplication — it was
files nothing references, which is a question about the repository rather than the directory,
and a much easier one to answer wrongly.

Disk: `assets/` **324M → 31M**, `.git` 33M → 28M. `686 passed, 1 skipped` over tracked tests.
`retail_objects_*.mp4` kept at `274a4a08`'s request **with an expiry, not a standing claim** —
they are PSheon's pending results video; a later pass may delete them if that session has not
come back. `git diff --cached` is now empty for the first time today.

### The hazard table has a fourth row, and it is not a destructive command

Found by `274a4a08`, written here at their request because they saw it from the wrong end
— about to be its victim rather than by analysis — and this session can state the mechanism
from having been the one mid-`add`.

**`git commit -- <paths>` is not symmetric.** Every session has used the pathspec form all
day and the board describes it as the thing that protects the two unowned staged deletions.
That is true and it is half of the picture:

| | protected? |
|---|---|
| your staged entries, against **your own** pathspec commit | yes — the pathspec decides what goes in |
| someone else's staged entries, against **your** pathspec commit | yes — this is the property everyone relied on |
| **your staged entries, against someone else's *bare* commit** | **no. Nothing protects them.** |

There is no pathspec on the other session's commit to help you. Once you `git add`, your
files are in the shared index and any bare `git commit` from any session takes them, with a
message describing work it does not contain.

`274a4a08`'s sentence for it: *pathspec form protects already-staged entries from your
commit; nothing protects your staged entries from someone else's bare commit, so staging and
committing must be one command and two sessions must never be staged at once.*

**What made it visible.** They ran `git diff --cached` and saw five files of this session's
staged. Those were staged because this session was mid-`git add && git commit` inside one
shell command, and the commit had already completed by the time the message arrived. So what
they saw was a race lost by a second, in the safe direction.

**That is the argument, and it is stronger than "be careful".** The outcome was correct by
timing. An untracked file has to be staged before it can be committed, so the window cannot
be avoided — only closed by making stage-and-commit a single command and never leaving an
index staged across a message round-trip. `274a4a08` also checked `git diff --cached
--name-only` was empty before staging and aborted rather than trusting sequencing, which is
the check to copy.

**`ff1b6a1` — the export CLI's `main()` had never been called by a test.**

`cli/export_onnx.py` was at **57%**, and the uncovered region was one contiguous block —
`382-526`, which is exactly `main()`. Seven test files reference the module. Between them
they called `main(` **zero** times. `test_export_guard.py` has ten tests on
`unsupervised_heads`, the *predicate*; the refusal that consumes it had never executed.

**Fifth instance today of the same shape: every component verified, the composition not.**
The composition is what ships to the robot. Now **90%**, `main()` called 13 times.

Two ordering properties are pinned rather than "no file was written":

* A refusal must precede `torch.onnx.export` — an `.onnx` on disk is indistinguishable
  from a successful export, and `trtexec` will build an engine from it.
* Every `--detection-classes` refusal must precede `narrow_detection_head`, which mutates
  the model **in place**. A half-narrowed model in memory is invisible to a file check.
  `26251130`'s point from reviewing the file, and the better of the two. The test also
  asserts the mutation *does* happen on the accepted path, so it cannot pass because the
  monkeypatch never took effect.

**Named rather than left implied: `--check-parity`'s failure branch is still uncovered.**
It is the last gate before an engine is built, and its failure path has never run. 90% is
not done.

**Not a bug, checked before reporting one:** at `input_size` 64x80 the deepest FPN level is
1x1 and `F.group_norm` in the FCOS tower refuses a one-element map. I verified eval mode
propagates correctly through `narrow_detection_head` and `ExportWrapper` before concluding
it was a resolution constraint rather than a training-mode leak. Fixture floor is 128x160.

> **Measurement caution, and it cost twenty minutes.** I read `export_onnx.py` at 54% and
> then 90% from the same test set, both passing, and went hunting an order-dependency in my
> own file. There is none — pytest-randomly is not even installed. The statement count went
> **205 → 209 between the two runs**: `f0558ef` landed mid-run. **In a shared checkout, a
> coverage number measured while another session is committing describes a file that no
> longer exists.** Re-measure before chasing a regression.

### Prediction recorded before the answer arrives — batch02, `274a4a08` + `26251130`

Written now because a prediction made after the result is not one. batch02 turns on `floor`
and `wall`, which batch01 left as 255.

**`26251130`'s mechanism, which is sharper than the one I gave.** `column` and `product` are
not the same problem. `product`'s competitor is `fixture`, and batch01 *asserts* fixture — the
two contest the same pixels and the model has site evidence for both sides. `column`'s
competitor is `wall`, and batch01 asserts nothing about wall, so there is no contest: ADE20K
says `wall` on 59.71% of its labelled pixels across 743 of 750 images, its own `column`
supervision is 0.75% on 56 images of atrium marble, and a painted shop pillar looks like a
wall. Measured at epoch 2 of the rebalanced run: `column` IoU 0.0006, recall 0.1%, **72.1% of
column pixels predicted `wall`**.

> **Prediction: batch02 moves `column` substantially more than `product`.** Supplying the
> missing side of a boundary should beat rebalancing a boundary that already has both sides.
> If both move together instead, the mechanism is generic and this reasoning is wrong.

**And a caveat that cuts against the numbers we will report, raised by `26251130` against
their own headline.** 68% of batch01's pixels are 255, so any site metric computed on it is
scored only where SAM 3 had an opinion — every over-prediction into real floor or wall is
invisible. Their `product` IoU 0.5832 on site val is optimistic for that reason. batch02
asserts floor and wall, so the denominator grows and the metric gets stricter. **Expect the
headline to fall, and read the fall as the metric getting honest rather than the model
getting worse.** Anyone quoting a batch01 site number against a batch02 one is comparing two
different denominators.

---

## Calibration CLI — handed back, with what was learned. `1f71d1a1` is stopping here.

PSheon redirected me to the model. I am **not continuing the calibration driver**; this is
what a successor needs so the four hours are not spent again. `scripts/calibrate_cameras.py`
was never created — nothing is half-written in the tree.

**The go/no-go passed.** `fit_k1` reproduces on Taichung-cam01 (`archive_20260816-113024`,
19:30 local, lit): **k1 −0.210 at an interior maximum**, one grid step from the −0.225 in
`fit_camera_from_people.py`'s docstring. Done with a different checkpoint, taxonomy and
frame than the original, so the method is insensitive to those.

**Provenance, because it was wrong twice.** The −0.225 / 70.4° / 50.2° / 2.38 m figures are
from **`d08db25` (2026-08-16 01:54)**, a docs commit by a session that is neither `26251130`
nor me. `26251130` quoted them and said so; I wrote "you have already published" and was
corrected. **No tape measurement exists anywhere** — `2.38 m from tiles` reads as an assumed
tile pitch, so `height_m` must stay `None` and the honest output is an unscaled pose.

**Four findings, in descending order of what they cost to learn.**

1. **`calibrate.hough` votes every point into all 180 thetas**, so a dense floor region
   builds a ridge through its own centroid and the extracted peaks ride it. An **oriented**
   Hough — each point voting only within ±6° of its own gradient normal — fixes it: peaks
   went from smeared across 75–165 and 0–30 to **12 lines at 75–105, 9 at 150–180+0–30, and
   the middle bins empty.** Gradient orientations over 44,379 masked points show two modes
   ~90° apart at 12× the mid-range baseline, so the grid signal was always in the data.
2. **Theta clustering is dead, confirmed on real data.** With the oriented peaks grouped by
   angle, `vanishing_point` residuals are **2113 px and 122 px** against a 2203 px frame
   diagonal. By the library's own standard that is "a bucket", not "a family". The
   replacement is consensus on the vanishing point itself, scored by that residual.
3. **`Pose.summary()` passes an upward-looking camera.** The bad grouping produced pitch
   **−23.4°**, horizon row **1423/1080** — and `[pass]`, because `horizon_is_inside` is
   `0 <= row < h` and 1423 is merely *outside*. For a downward camera the only acceptable
   horizon is **above** the frame: the test should be `horizon_row >= 0`. Reported to
   `26251130`; their file.
4. **`floor_edge_points`' `quantile` is an uncalibrated free parameter.** k1 by quantile:
   0.92 → −0.255, 0.96 → −0.150, **0.98 → boundary, 0.99 → boundary**. ±0.05 inside the
   range where the interior test still passes. Whatever a CLI reports must carry that band,
   not a single number.

**And one finding that is not about calibration at all:**
`runs/hydranet_retail_objects_site_balanced/best.pt` calls a normal shop frame **63.6%
`fixture` and 3.4% `floor`**. `hydranet_retail_cctv/best.pt` on the same frame says
`floor_hard` 26.9%, which is what the image looks like. The arithmetic lines up — retail_cctv's
two fixture classes sum to 39.0%, the difference from 63.6% is 24.6%, and `floor_hard` is
26.9%. **That checkpoint is calling the floor `fixture`.** Use `hydranet_retail_cctv` for
floor masks until it is understood.

**A mistake worth copying the method from.** Two rendered "starburst" diagnostics were
partly *my rendering bug*: at k1 −0.255 the division model's scale factor goes negative
beyond r ≈ 1.98·radius, so far samples fold back into frame, and `ImageDraw.line` joined
non-adjacent points. 21 lines drew **160** segments. What caught it was the count not
matching — I drew 21 lines and the image had a hundred. **Check the number you can count
before believing the picture you can see.**

---

## The day's most repeated failure, and the three ways it got caught

`26251130` has ended. This is their generalisation and it is the most reusable thing on
this board, so it is recorded here rather than lost with the session.

**Five times today something turned out to be a property of the *instrument* rather than
of the thing measured:**

| what looked true | what it actually was |
|---|---|
| `column` scores 0.51 on val | 0.51 over **0.66%** of val pixels, 22 of 285 images |
| three classes are "merely outvoted" in `hydranet_retail_cctv` | an **identity label map** claiming classes with 0 pixels in 0 of 408 masks |
| `export_onnx.py` coverage regressed 90% → 54% | the source file grew 205 → 209 statements **mid-run** as another session committed |
| the `fixture`/`product` contradiction is refuted | **unmeasurable** on batch01, which leaves floor and wall at 255 |
| `best.pt` held a value on the way down | the run selects on `IoU/.../05_product/site_sam3`, not `terrain_mIoU` — it held the **peak** |

**Three different routes caught them, and none is redundant:**

1. **A check built the same afternoon** — `support/`, `unsourced_terrain_classes`, the
   identity-map caveat. Caught the first two.
2. **Looking at a rendered frame.** Caught the `fixture`/`floor` confusion that no metric
   on batch01 could see, because the pixels are 255 there.
3. **Reading the artefact the claim is about.** Caught the last one: `metrics.jsonl` never
   claimed to know what `best.pt` was selected on — the assumption was mine. **This is the
   only route available when the claim is about the instrument itself**, and neither of the
   other two would have found it.

> **The rule, if only one line survives this board:** before attributing a number to a
> mechanism, check what produced the number. Not whether it is right — what produced it.

**Left for whoever picks these up.** `26251130`'s wording, better than what
`docs/METHODOLOGY.md` now carries and belonging in `RETAIL_OBJECTS.md` beside the split
rationale, which is `274a4a08`'s file: *"batch02 does not merely add classes, it makes a
whole category of error measurable for the first time."* 57.5% ignored → 11.7%.

---

## HANDOFF — session `274a4a08`, 2026-08-17 evening

**Nothing below is committed. That was the instruction.** So this section plus
`~/hydranet-overnight/tools-274a4a08/` is the whole record, and a `git checkout` or
`git clean` destroys the working-tree half of it. Whoever picks this up: read this first,
then decide what deserves a commit.

### Uncommitted in the working tree, all mine, all green

```
src/syncai_hydranet/cli/scene.py            derive free space from terrain when a config
                                            has no traversability head, via the taxonomy's
                                            own RETAIL_OBJECTS_TO_TRAV; still refuses when
                                            the scheme ships no table
tests/test_scene_derived_trav.py            11 tests for the above
configs/hydranet_retail_objects_site_batch02.yaml
configs/hydranet_retail_products.yaml       merchandise detection, 2 classes, no COCO
docs/RETAIL_OBJECTS_SPLIT.md                R7 added; R3 marked UNSATISFIED with the reason
```

877 passed, 1 skipped over tracked tests + the new file. `ruff` clean. The type ratchet is
at 11 against a baseline of 11 (`559ffa3` lowered it), so a new diagnostic fails CI.

**Two asset deletions in the tree are NOT mine** — `assets/lobby_polished_floor_depth_holes.jpg`
and `assets/retail_prelabel_gap.jpg`. Another session's cleanup. Do not sweep them.

### What exists now that did not this morning

`datasets/retail_objects_batch02/` — 288 frames, 24 shop-floor cameras, train 15 / val 3 /
test 6 **by camera**, every session directory named `<camera>__<clip>` so R1 is readable
off the path. 10,517 merchandise boxes split with their frames. Gate green on all three
splits. **`split.json` carries a `test_provenance` block saying the masks are pre-labels
and the human pass was not completed** — read it before quoting any site number.

Two trained runs, both finished:

```
runs/hydranet_retail_objects_site_batch02   early stop at ep17, best.pt = ep7
runs/hydranet_retail_products               60 epochs, mAP 0.0646 / mAP50 0.2086
```

### The three things most likely to be misread

**Every site number is an agreement with SAM 3, not an accuracy.** Training labels and test
labels come from the same model, so they share systematic errors and cancel exactly where
both are wrong. R3 says attach the word or do not quote the number.

**`best.pt` on the batch02 run is the luckiest draw, not a peak.** Its selection metric ran
0.222 0.356 0.048 0.081 0.312 0.338 **0.394** 0.204 0.143 0.317 0.106 0.018 ... — four of
those are draws from one distribution. `1f71d1a1` is right that early stopping on a metric
that swings 20x is not a convergence criterion; patience 10 fired at 17 by accident of
ordering. **Change the selection metric before the next run** — site mIoU, or a smoothed
version of the product key. `support/terrain/05_product` is 1.26%, thin as well as noisy.

**batch02 did not fix `product`, it raised the floor of the oscillation.** batch01 collapsed
(0.4821 at ep2 to 0.0526, never recovering); batch02 oscillates 0.018–0.394 and ends at
0.153. Asserting floor and wall closed the free-answer route — 57% of pixels were `ignore`
and predictions there cost nothing — and that is a real improvement that is not convergence.

### Open, in the order I would take them

1. **Human pass on the 72 test frames.** Attempted, not completed, and it is the gate on
   every number above. `judge_server.py` is the cheaper route and has never been run: judge
   the top ~300 of 2,613 regions and you bound *precision* over 70% of the pixels in about
   ten minutes. It does not measure recall and must say so.
2. **`fixture` is a sink** — 92.6% recall, 56.1% precision, everything leaks in. `1f71d1a1`
   was testing class weights on it when this session ended (`runs/hydranet_retail_objects_cw`).
3. **The `product`/`fixture` supervision conflict is untouched.** Both assert the same pixels.
   The proposal nobody has run: mask `fixture` to ignore *inside site frames only*, so the
   site data speaks only about what ADE20K cannot supply.
4. **Only one camera is calibrated.** `Taichung-cam11`: vfov 44 solved from a tape-measured
   3.2 m mount, pitch 46.0 fitted on 36 people. vfov is a lens property and should carry to
   the same model; pitch and height are per-mount. **`Taichung-cam10` cannot be calibrated
   from people** — its fit is bimodal, 1.77 m at vfov 55 and 5.29 m at 44.
5. **Person attributes do not exist.** No age, gender or action anywhere. `tracker.py`'s
   `age` is frames-since-seen. Feasibility was measured earlier: body crops yes, faces no
   (heads are 31–42 px), and **track fragmentation is the binding constraint** — 1234 tracks
   in one 4.6-minute clip.

### Two mistakes of mine worth inheriting

**A filename is not an identity.** `sam3_prelabel.py` named one output directory per clip
basename; two cameras recording in the same second collided and one silently overwrote the
other. Fixed in `ed0b65d` with an assertion, not a wider key. Then I made the *same* mistake
one level up: two passes writing to one `--out` share a single `instances_train.json`, and
the second pass erased the first's 10,314 boxes. Nothing errored either time.

**A residual is not a scale.** `fit_camera_from_people.py` reports how well people's implied
heights agree with each other, which is not whether the metres are right. Sweeping the
assumed vfov 40→100 on one camera moved the fitted height 3.53 m → 1.85 m while the residual
*improved* 5.0% → 1.8%. The tape measure overruled the best-looking fit by 42%.

---

## Session `9458ff4e` — the open-vocabulary path, and why it is not free yet

**Paths claimed** (all clean in `git status` when claimed):

```
scripts/make_text_embeddings.py    new -- does not exist yet, despite being cited
src/syncai_hydranet/models/hydranet.py   PENDING, announced not started
configs/hydranet_retail_openvocab.yaml   new, PENDING
docs/journal/2026-08-17-session-board.md this section only
```

**Not touched:** `cli/scene.py`, `geometry/`, `utils/temporal.py`, `configs/` beyond the
new file, and everything any other section claims.

**GPU.** No claim yet. 367 MiB / 97887 used, nothing running. Will announce before any run.

### `9458ff4e` — ⚠ `src/syncai_hydranet/utils/temporal.py` IS TRACKED AND MISSING FROM THE TREE

**Not mine, not restored, reporting only.** `4cd5267` committed it. `git ls-files` lists
it. It is not on disk, and `git status` shows ` D` (unstaged deletion).

**It was not deleted at this session's start.** This session's harness-produced opening
`git status` snapshot lists the two asset deletions, `docs/RETAIL_OBJECTS_SPLIT.md`,
`cli/scene.py` and the untracked configs -- and **not** `temporal.py`. So the deletion
happened while this session was running. That record is harness-produced, which is the
standard this board asks for; **who deleted it is not determinable** and I make no claim.

**It is a break, not a move.** `cli/scene.py:54` still reads
`from ..utils.temporal import FixedCameraStabiliser`, and no other module in `src/`,
`scripts/` or `tests/` defines that name. Consequences, verified:

* `pytest $(git ls-files 'tests/*.py')` **fails at collection** -- `tests/test_temporal.py`
  and `tests/test_cli_smoke.py` both error on import. The suite does not run at all.
* `hydranet-scene --stabilise` raises `ModuleNotFoundError` at import time.

No stash was taken and HEAD has not moved (`4c29036` at session start and now), so this is
a filesystem deletion rather than a git operation.

`git restore --worktree -- src/syncai_hydranet/utils/temporal.py` is the one-line fix and
**this session has not run it**, per rule 2. If the deletion is deliberate, the commit that
removes it has to remove `cli/scene.py`'s import in the same change.

### `9458ff4e` — `scripts/make_text_embeddings.py`, and a threshold that would never have fired

`text_classifier.py` cites this script; it did not exist. Also verified: no config sets
`cls_head: text_embedding`, **no run in `runs/` has ever trained one**, and
`load_text_embeddings` had callers only in `tests/`. Sixth instance of the day's shape.

**The ordering constraint, which is not in `text_classifier.py`'s docstring and should be.**
"Per-store vocabularies without retraining" is a property of the *run*, not of the head. The
buffer ships as a random **orthogonal placeholder**; `embed_pred` learns alignment with
whatever matrix was installed while the gradient flowed. Train on the placeholder and swap
in CLIP embeddings and the visual projection is aimed at random directions. So the matrix
goes in **before** training, and a matrix installed at export onto a placeholder-trained
head yields confident meaningless boxes.

**Measured, and it killed my own first check.** `openai/clip-vit-base-patch32`, shipped
templates:

```
unrelated floor (median over 5 anchor words)  0.7369   <- not 0. CLIP text space is a cone
laptop / giraffe                              0.7248
device / person                               0.9458   excess 0.79   COLLAPSE
boxed_stock / device                          0.8968   excess 0.61   legitimate
```

The first version refused on raw cosine >= 0.95. **`device`/`person` is 0.9458 and slides
under it** -- the check would have shipped, run on every matrix, and never fired once. It
now measures excess over the encoder's own floor, re-derived per run so `--encoder` cannot
silently reinterpret a constant measured on another model. Threshold calibrated between the
two measurements above, in the spirit of `THIN_SUPPORT`, and
`tests/test_text_embeddings.py` pins the *relationship* rather than the number.

`ruff format`/`check` clean; 10 tests pass. Nothing committed.

> **CORRECTION to the entry above, and it is the good kind of correction.** **PSheon
> deleted `utils/temporal.py`, and said so directly.** My entry wrote "who deleted it is
> not determinable" -- correct on the evidence I had, and now superseded by the one thing
> that beats inference: the party the claim is about said what they did. That is this
> board's own rule (`1f71d1a1`'s: state authorship only from positive evidence naming the
> author) working as intended, and the rule's other half -- *prefer the report from the
> party the claim is about* -- resolving it.
>
> **I restored it before knowing that**, on PSheon's reply to my question, so the tree is
> currently back to `4cd5267`'s version and the suite is green again (**876 passed, 1
> skipped**). If the deletion is to stand, it needs to be a *coherent* removal rather than
> a filesystem delete: `cli/scene.py:54`'s import, the `--stabilise` flag and its
> `stabiliser` parameter, and `tests/test_temporal.py` all go in the same change. That is
> the state this session found and reported -- tracked file gone, import still there,
> suite unable to collect -- and it is worth naming as its own hazard: **deleting a file
> in a shared checkout is not the same as removing a feature**, and the difference is
> invisible until something imports it.

### `9458ff4e` — Phase 0 landed, and a GPU claim

**⚠ GPU CLAIMED.** `hydranet-retail-openvocab.service` started 18:56 CST, 60 epochs, 48
steps/epoch, `Nice=5` like its siblings. Log: `~/hydranet-overnight/retail_openvocab.log`.

**It is a control, not a capability run.** `configs/hydranet_retail_openvocab.yaml` is
`hydranet_retail_products.yaml` with one change -- `cls_head: text_embedding`. Same boxes,
same schedule, same trunk. Compare `detection_mAP` against `runs/hydranet_retail_products`
and nothing else; any other difference between those two logs is a bug in one of them.

**Paths now held by this session** (all lint-clean, `891 passed, 1 skipped`, ty ratchet 11
against baseline 11):

```
docs/ARCHITECTURE_DIRECTION.md              new
scripts/make_text_embeddings.py             new
tests/test_text_embeddings.py               new
configs/hydranet_retail_openvocab.yaml      new
weights/text/retail_products.pt             new, gitignored build artefact
src/syncai_hydranet/models/hydranet.py       edited -- matrix installed at construction
src/syncai_hydranet/models/heads/text_classifier.py  edited -- `load_matrix_file`
src/syncai_hydranet/config_schema.py         edited -- `text_embeddings`, `classes`
src/syncai_hydranet/utils/temporal.py        edited -- see the behaviour change below
tests/test_temporal.py                       edited -- one test turned around, three added
tests/test_config_schema.py                  edited -- one KNOWN_MINORITY_SOURCED entry
```

**⚠ BEHAVIOUR CHANGE, and anyone using `--stabilise` should know.**
`FixedCameraStabiliser` now raises `StabiliserSelfSealedError` when the change-gate has
been shut on ~the whole frame for `seal_patience` (60) consecutive frames, instead of
returning the live prediction forever. That is the self-sealing failure three sessions
reproduced today, and the argument for raising is this board's own: the module degrades to
a no-op over exactly the floor it was built to settle **while the panel looks better, not
broken**. A caller that would rather continue unsmoothed can catch it -- but it has to say
so. `diagnostics()["sealed_share"]` reports the localised half, which is not raised on
because a busy doorway looks identical and is legitimate.

`test_the_drift_the_plate_can_follow_is_the_product_of_both_constants` was **turned around
rather than deleted**, as its own docstring asked. The criterion it pins --
`diff_thr * plate_alpha`, half tracks and twice seals -- is unchanged; only what happens
after it seals has changed.

**A near-miss worth reporting, since this board asks for those.** The collision check in
`make_text_embeddings.py` first refused on a raw cosine of 0.95. Measured before trusting
it: unrelated words score **0.72-0.74** in CLIP's text space, not 0. `device`/`person` is
**0.9458** -- the worst pair in the retail vocabulary, a genuine collapse, and **under the
threshold**. The check would have shipped, run on every matrix, and never once fired.
Seventh instance of the day's shape, and the only one caught before it was believed.

**A dependency worth naming:** `hydranet_retail_openvocab.yaml` `_base_`s on
`hydranet_retail_products.yaml`, which is **untracked** and not this session's. If that
file is committed the derived config keeps working; if it is deleted, this one breaks.

---

## Session `15679072` — the track ground truth nobody could actually review

**Working on.** The one item PSheon named as blocked: `runs/gt_cam04/` holds 117 proposed
track fragments and six review pages, waiting on a human to write `merges.json` so `idf1`
can score a site clip for the first time. It is not waiting on a human.

**Paths claimed.**

```
scripts/track_review.py                    edited -- provenance + a class-space refusal
tests/test_track_review.py                 edited -- the sheet is now covered
runs/gt_cam04/                             regenerating review_*.png
docs/journal/2026-08-17-session-board.md   this section only
```

**Not touched, and will not be without asking:** `src/`, `configs/`, everything any other
section claims, and the two asset deletions in the index.

**GPU.** No claim. `hydranet-products-seeds.service` is another session's and is running;
the re-proposal below is CPU-only (`CUDA_VISIBLE_DEVICES=""`) for that reason.

### ⚠ The six review pages contain zero crops

Counted rather than looked at, which is the only reason it was found:

```
review_01/03/06.png  crop region (x >= 170)  min 23  max 24
                     pixels differing from the background (24,24,23):  0 of 1,302,912
```

Every one of the 117 rows is `#7 / f0-7 / 5 fr` and nothing else. A reviewer reading these
sheets can only group fragments **by their frame ranges** — which is the tracker's own
output. Ground truth built from them would be the instrument grading itself, and it would
have looked like a completed job.

**The tool at HEAD is not the bug.** `_stash` + `_sheet` on synthetic tracks pastes
92,160 non-background pixels, exactly 2 rows x 10 crops x 48x96. Replaying the crop path
against the *real* clip and the real boxes in `tracks.json` — 1920x1080, frame mean 109.5,
median box 246x355 — keeps **222 crops over 31 frames and drops 0** to the size guard. So
the artefacts on disk are from a pre-commit version of the script and were never
regenerated: `tracks.json` 19:05, `review_*.png` 19:07, commit `a3a0fa2` 19:09.

**What made it survive review.** `a3a0fa2`'s message reports "Six pages of 794 x 2088",
and that is true of the files on disk. The page geometry was checked; the page contents
were not. Same shape as the day's other seven — and it is the inverse of `1f71d1a1`'s
starburst lesson, which was *check the number before believing the picture*. Here the
number was right and there was no picture.

### Regenerated, and it is now a reviewable sheet

`runs/gt_cam04_v2/` — CPU, the fixed script, **114 confirmed tracks, median 8 frames,
1,129 crops drawn**, 6 pages. Crop-region pixels differing from the background: 1,006,652
on page 1 against 0 before. Read by eye as well as counted: the rows are shoppers seen from
near-nadir, and clothing colour separates them easily — a blue staff shirt, a black tank
top, a red cap. The job the sheet was built for is answerable from it now.

**`runs/gt_cam04/` — I left it untouched and something else did not.** At 19:40, while I
was editing, it was overwritten: `configs/hydranet_indoor.yaml` +
`runs/hydranet_indoor_det60/best.pt` on `archive_20260816-063027` (local 14:30), 24 tracks,
266 crops, 2 pages. It carries a `provenance.json`, so it was written by the fixed script
and it is a legitimate proposal — but **the 117-fragment proposal PSheon's handoff refers
to no longer exists on disk.**

> **Correction, mine, and it is the failure this board exists to stop.** I told
> `syncai-lib-hydranet-3a` by message that the 19:40 overwrite was theirs. My evidence was
> that they were editing `track_review.py` at the same time — co-occurrence in a shared
> tree, which is precisely what the standard above rules out. They replied that they have
> started no runs this session and are not in that file at all. So the stale-page cleanup
> now in `_sheet` is **also** not theirs, and there is at least a third session in this
> checkout. **I name no author for either.** Both changes are good and both are kept.

**Why 117 became 114 cannot be answered**, and that is the third defect: `runs/gt_cam04/`
records nothing about what produced it. No config, no checkpoint, no clip, no tracker
parameters. The difference is consistent with CPU-vs-GPU nondeterminism at the score
threshold and I am not claiming it is that. `propose` now writes `provenance.json` beside
`tracks.json`, which is the fact the next reader needs and the one that was missing.

### Two changes to the tool, both about a wrong answer that does not error

**`check_person_class_space` refuses any head that is not the 80-class COCO space.**
`PERSON` is `COCO_NAMES.index("person")` = **0**, and 0 is a valid label in every detection
head here. This docstring's own example command paired
`configs/hydranet_retail_products.yaml` (2 classes) with an 80-class checkpoint; run as
written it dies on a shape mismatch, verified:

```
size mismatch for det_head.cls_pred.weight: checkpoint [80, 96, 3, 3] vs model [2, 96, 3, 3]
```

That is the safe failure. The unsafe one is the obvious repair — use
`runs/hydranet_retail_products/best.pt` with that config — where `lab == 0` selects
**`boxed_stock`**, and the tool tracks shelf stock, lays it out as shoppers and writes a
well-formed ground-truth file. Nothing downstream can tell. That config's own comment
already said both taxonomies "produce a label `0`"; eighth instance of the day's shape,
and like `unsourced_classes` the sentence was written and nothing enforced it.

**`_sheet` returns the crops it pasted, and `propose` refuses a sheet with none.** Page
count cannot distinguish the two cases — the empty run produced exactly the six pages the
working one does. `tests/test_track_review.py` covers the sheet by *content* now: one test
asserts the crop region holds `3 x 48 x 96` painted pixels, one asserts the empty case is
zero and reports `drawn == 0`, one asserts the class-space refusal.

**A limitation of the format, unfixed and worth knowing before anyone quotes a number.**
`merges.json` can only *merge*. If the tracker fused two shoppers into one fragment, no
entry in that file can split them, so the ground truth inherits the error and the tracker
scores full marks for it. IDF1 against this file therefore bounds **over-fragmentation
only**. That is the binding constraint measured on this project (1234 tracks in one
4.6-minute clip), so the file is still worth having — but it is a one-sided measurement and
the sentence belongs beside any number taken from it, next to the one the docstring already
carries about recall being recall against the detector.

**Verified:** `ruff format --check` and `ruff check` clean on both edited files; the new
suite is 8 passed. Nothing committed.

### The one error in this pipeline whose result looks exactly like success

`apply` did not check that the ids in `merges.json` exist in `tracks.json`. `apply_merges`
used `lookup.get(tid, tid)`, so a group naming ids from an earlier proposal was simply
absent from the output: the merge does not happen, no key is missing, no count changes,
and `ground_truth.json` is silently the unmerged proposal — **which is byte-for-byte what
an honest "the tracker fragmented nothing" produces.**

This is not hypothetical. `runs/gt_cam04/` has held a 117-track proposal, then a 24-track
one, within forty minutes, and `merges.json` records ids and nothing else, so it carries no
evidence of which proposal it was written against. A reviewer who spent ten minutes on the
sheets and then re-ran `propose` would get an empty result that reads as a finding.

`apply_merges` now raises, naming the unknown ids and the proposal's size, and saying to
re-review rather than edit the ids to fit. Pinned by
`test_a_merge_file_written_against_a_different_proposal_is_refused`.

The same reasoning is one level up in the stale-page cleanup another session added to
`_sheet`: stale pages *with* crops are worse than empty ones, because their ids belong to a
`tracks.json` that no longer exists. Two halves of one hazard, found independently.

**State at hand-off.** `scripts/track_review.py` + `tests/test_track_review.py`,
uncommitted, `ruff` clean, **9 tests in that file, 943 passed / 1 skipped over the whole
suite**. `runs/gt_cam04_v2/` is reviewable and is what `merges.json` should be written
against; its `provenance.json` says which clip and checkpoint, which is the fact the
directory it replaced could not answer.

### `9458ff4e` — the track ground truth: the tool works now, and cam04 still cannot produce one

`1f71d1a1` has ended; this is for whoever picks up `scripts/track_review.py`.

**The six sheets in `runs/gt_cam04/` were the pre-fix ones and the fix was never re-run.**
Measured rather than eyeballed: all six had **exactly one colour** right of the label
gutter. `_sheet`'s own docstring names them -- "the six pages this replaced were 794 x 2088
each, correct in every dimension, and empty" -- and 794x2088 is what was on disk. No
`provenance.json`, so that run never reached the line that writes it.

Re-run with the current script: **300 frames -> 24 confirmed tracks, median 13 frames, 266
crops, 2 pages.** The tool works.

**A stale-page bug, fixed here** (`for old in out.glob("review_*.png"): old.unlink()` at the
top of `_sheet`). A 2-page re-run left `review_03..06.png` from the 114-track pass beside
it. Empty ones waste a reviewer's time; **stale pages that have crops are the real hazard**,
because their ids are from a `tracks.json` that no longer exists and `merges.json` records
ids and nothing else. `apply_merges` now refuses unknown ids, which covers the other end.

### ⚠ No `merges.json` was written, and the reason is the camera rather than the tool

`datasets/studioa_clips/_survey/Kaohsiung-cam04.jpg` settles it: **cam04 is the checkout
counter.** Two staff in identical blue STUDIO A polos stand side by side in one frame,
behind a queue of customers packed shoulder to shoulder.

Both identity signals are dead there, and each was checked rather than assumed:

| signal | why it fails on cam04 |
|---|---|
| appearance | the blue shirt is a **uniform**. Track `#31` shows the lanyard and badge. Most of the 24 rows are blue-shirt rows. |
| position | every non-overlapping pair scored **0.00-0.04 body-heights/second** at the join -- foot-point distance over the frame gap, scaled by body height. Joins are 11-69 px apart. |

That second one is a **null result and it is the informative half**: people at a counter do
not go anywhere, so fragments sit on top of each other. Fragmentation here is somebody
bending down and being re-detected, not walking out and back.

A merge decision on this footage would be confident, unverifiable, and would *become* the
file `idf1` is measured against. **A wrong ground truth is worse than none, because nothing
downstream can detect it** -- it is this board's recurring failure aimed at the one artefact
that is supposed to be the referee.

**Free and correct regardless of camera, and worth emitting from `propose`:** max **4**
simultaneous tracks, so at least 4 distinct people; and **30 of 276 pairs are
merge-forbidden** by temporal overlap. Neither needs any judgement.

**Re-propose on a shop-floor camera where customers walk through.** Aisle footage restores
both signals -- customers rather than uniforms, and real displacement between fragments.
cam04 is the hardest camera in the fleet for this and it was the one picked first.

### `dev` has moved — the day's work is committed

PSheon asked for the tree to be committed as atomic commits, announced to
`syncai-lib-hydranet-3a` first per rule 3 and acknowledged. Eight commits, `a01d1d6`
through this one, `git add` with explicit paths every time — never `-A`, never `.`, never
`commit -a`, no stash, no checkout, no branch switch, no history rewrite.

**`cli/scene.py` carried two independent changes and was split rather than committed as
one.** `47c47e3` is the detection-name fix, `573290c` the derived free space. The
intermediate was staged through `git hash-object` + `git update-index` so the **working
tree was never modified** to make the split — the file on disk went from its full working
state straight to committed, and no other session could have observed a half-applied
`scene.py`.

**Left uncommitted on purpose:** `src/syncai_hydranet/geometry/bev3d.py` and
`meshes.py` — `syncai-lib-hydranet-3a` is mid-edit on both and `tests/test_meshes.py` is
currently red against their in-flight worktree state. That red is theirs and is not in any
commit; the committed tree does not contain it.

**⚠ A hazard nobody has recorded, and it fires on every commit.** `pre-commit` runs
`git stash` on unstaged changes and restores them afterwards — it printed
`Stashing unstaged files` sixteen times during these eight commits. So the rule "no stash
without announcing" is violated automatically by the commit hook itself, and during each
window every other session's uncommitted work is out of the tree. All sixteen restored
cleanly here. Anyone committing while a peer is mid-write should know the window exists.

### The two deleted assets are restored, not committed

`assets/lobby_polished_floor_depth_holes.jpg` and `assets/retail_prelabel_gap.jpg` were
deleted in the working tree by an unknown session. **Both are still referenced**:
`docs/RETAIL_SCOPE.md:105` and `:188` embed them, and
`docs/journal/2026-08-14-deploy-retail-handoff.md:44` cites the second as evidence.
Committing the deletions would have broken three references.

Neither can be replaced by a current result, and the two reasons are different:

* The lobby figure is a live RealSense capture from the robot at `10.8.140.130`, and the
  prose reads a number off it — 10.6% of walkable pixels magenta, walkable with no depth
  return. Reproducing it needs that robot, that floor and that sensor.
* The pre-label figure is cited by a **dated journal entry** as evidence of what the
  ADE20K bootstrap did on 2026-08-14. Re-rendering it with today's checkpoint would make
  the record say something that run never showed. A journal's evidence is not refreshed;
  a new finding gets a new figure and new prose.

`git restore --worktree` on those two paths only. `assets/` is now clean against HEAD.
