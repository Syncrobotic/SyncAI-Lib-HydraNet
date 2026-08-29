# tools/ — data production and dev tooling (upstream of training)

Everything here sits *upstream* of training: it makes the data and checks the inputs a run
depends on. It is not a deployment surface — that is [`../deploy/`](../deploy/), the
downstream end. Filing these under `deploy/` would confuse an input with an output.

**"Training" there means `hydranet-train`**, the wheel's entry point that fits the shipped
network. One tool here does fit a model — `temporal/train_posture.py`, the step-8 posture
model, under 100K parameters — and it is not an exception to the sentence above so much as
a reminder of what that sentence is about: it consumes a `.npz` the other three temporal
tools produce, it ships nothing, and it is here rather than in `src/` for the same reason
everything else is.

## [`commissioning/`](commissioning/) — the per-camera pipeline (PLAN §2.1)

Everything that turns one camera's plates into its `camera.json` and its 3D scene:
`masks_pass.py` (structure vote), `extras_pass.py` (door / product subclasses / the SAM3
floor source), `depth_complete.py` (geometry fills what the teachers miss),
`fp_polygons.py` (derived false-positive zones, human accept/reject),
`scene3d.py` / `scene_mesh.py` (the flat diagnostic panel and the solid-mesh scene with
GLB/OBJ export). Each is idempotent from the caches; re-runs cost no GPU except the two
teacher passes.

## [`pose/`](pose/) — the pose head's teacher

`vitpose_teacher.py` labels the Gold person boxes with ViTPose keypoints — the
distillation source for the bottom-up P3 head (PLAN §2.2).

## [`site30k/`](site30k/) — the campaign toolchain

The 2026-08-20 campaign's orchestration, kept exactly as it ran; `recipe.py`'s
per-camera pre-pass is also the engine `commissioning/masks_pass.py` drives.

## [`temporal/`](temporal/) — NTU RGB+D, and the posture model it feeds (PLAN §6 step 8)

Four tools, in the order they run. The first three exist because **the fall/bend
separation cannot be settled on our own footage**: PLAN §7 records a shopper leaning over
a counter producing a `fall` with a torso at 69° and a box 21% shorter, and a bend passing
every image-space test a fall passes. NTU RGB+D has ground-truth 3D for both actions, so
the question is answerable there and only there.

- **`ntu_survey.py`** — what is in the archive before anything is built from it: which
  action classes, how many sequences, what the class balance is.
- **`ntu_project.py`** — NTU's 3D skeletons rendered through *our* camera geometry. **The
  projection is the domain adaptation**: a model trained on NTU's own frontal view has
  never seen a person from a ceiling corner, and re-projecting the 3D through each
  commissioned camera's pose is what makes the sequences ours without collecting them.
- **`ntu_fall_discriminator.py`** — the measurement that motivated the height feature.
  Against NTU's own ground truth in metres, `A43 falling down` peaks at a 74.5° median
  torso angle and `A06 pick up` at 76.3° — **the angle does not separate them** — while
  head height above the floor does.
- **`train_posture.py`** — the step-8 model itself, fitted on `ntu_project.py`'s `.npz`.
  Under 100K parameters, because it runs per track per frame behind a network that has
  already spent the budget.

## [`annotation/`](annotation/) — the CVAT stack

A version-pinned CVAT deployment (`cvat.sh` + an override over upstream's compose) for
the accept/reject passes. The operator guide lived in METHODOLOGY.md, now in git history
(`git show b7457c2:docs/METHODOLOGY.md`). Site footage is customer premises, which is
why the stack binds to loopback and is reached through a tunnel, never a published port.

**Taxonomies the tooling actually gates.** `hydranet-annotation labels|check` (the
machine-checked half of the setup doc, `src/syncai_hydranet/cli/annotation.py`) validates
against `--scheme` = `indoor` / `retail` / `retail_objects`. Product B's 6-class
retail-security taxonomy is absent from that list **and that is the design, not a gap** —
an earlier version of this paragraph called it a gap on the strength of a claim from the
annotation session that turned out to be wrong, and the correction is worth keeping
because the wrong version is the intuitive one.

Nothing is ever annotated in six classes. Masks are drawn in the seven-class
`retail_objects` taxonomy, gated as `retail_objects`, and read down to six at load time by
the `retail_surfaces_from_objects` label map — which has **seven** entries, because it is a
reader of object masks rather than a taxonomy anything authors. `hydranet_retail_surfaces`
and every config under it point `site_seg` at `datasets/retail_objects_batch02`; there is
no six-class dataset on disk and there should not be.

The direction matters and only works one way. Six is derivable from seven by folding
`product` into `fixture`; seven is not recoverable from six, because the boundary between
merchandise and the fixture holding it is the hardest one in the taxonomy and cannot be
guessed back. So annotating at the finer level and deriving the coarser is strictly better
than the reverse, and a `retail_surfaces` scheme would let someone draw the lossy version
by mistake.
