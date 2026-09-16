# tools/ — offline data, commissioning and bounded training experiments

These tools produce and review data, commission cameras, and run bounded experiments.
Deployment entry points live under [`../deploy/`](../deploy/).

`hydranet-train` is the general training entry point. Experiment workers also include
`commissioning/stage0_train.py`, `annotation/studioa_train.py`,
`commissioning/train_glazing.py` and `temporal/train_posture.py`. A completed experiment
does not automatically replace a deployed model. See [tool status and historical
snapshots](../docs/TOOLING_STATUS.md) for the FTI copy, BEV diagnostics and current routes.

## [`commissioning/`](commissioning/) — the per-camera pipeline (PLAN §2.1)

Everything that turns one camera's plates into its `camera.json`, its zones, and its 3D
scene. Each is idempotent from the caches; re-runs cost no GPU except the two teacher
passes. The groups below distinguish production of evidence from its review and display.

**Build the artefacts.** `masks_pass.py` (structure vote), `extras_pass.py` (door /
product subclasses / the SAM 3 floor source), `depth_complete.py` (geometry fills what the
teachers miss), `fp_polygons.py` (derived false-positive zones, human accept/reject),
`footprints_from_masks.py` (fixture footprints — its own docstring records two attempts and
neither correct yet).

**Zones a store owns.** `service_zones.py` proposes the floor a shopper can stand in from
SAM 3 instances with no human drawing anything; `zones_confirm.py` is the accept/reject
pass that turns those proposals into named zones; `zone_draw.py` is the manual route —
click the floor, get metres; `zones_apply.py` writes human assertions and metre-zone
proposals back into the commissioning artefacts.

**Two 3D panels, and neither replaced the other.** `syncai_bev3d/scene_mesh.py` takes a
camera *name* and draws what commissioning measured for it -- solid geometry, GLB/OBJ
export -- so it needs an existing `camera.json` and its commissioning artefacts.
`syncai_bev3d/bev3d.py` takes arrays from a live forward pass and draws what the network
sees in the frame, needing no commissioning at all, which makes it the only 3D panel
available without those artefacts. Its assumed geometry is not a metric acceptance
result. The README's figures moved to the mesh panel on 2026-08-25 and
the perspective one has looked superseded ever since; the commit that introduced the mesh
scene does not mention it, and nothing said otherwise until 2026-08-30. Retiring either
removes an answer rather than a duplicate. `tests/test_renderer_generations.py` holds the
distinction. The underlying 2D `bev.py` grid remains in use for floor projection;
`hydranet-scene` is an optional assumed-geometry diagnostic. Neither is the current
StudioA mesh reconstruction acceptance route.

**Look at what was built.** `scene3d.py` and `scene_mesh.py` (the flat diagnostic panel
and the solid-mesh scene, with GLB/OBJ export), `scene_overlay.py` (project the
commissioned scene back through its own camera onto its own plate — the check that the
metres agree with the pixels), `masks_diagnose.py` (why `masks_pass` gave a camera the
structure it did, per cluster, with the picture), `cluster_rules.py` (replay a merge rule
against the cached SAM 3 proposals on every camera, no GPU).

`scene_mesh.py CAMERA --out runs/scene_review/NEW_RUN` writes an isolated review
directory with the GLB, OBJ, scene image, final surface overlays, object/support
reports and a source/output manifest. Existing output directories are refused. The
surface report separates pre-trim fitting scores from raw-frame coverage of actual
exported GLB triangles, and marks missing references and unrun stages explicitly.
Without `--out`, the standard commissioned paths are updated as before.

`stage0_baseline.py --out runs/stage0_baseline/NEW_RUN [CAMERA ...]` freezes the source
and inputs, builds a camera at a time and writes `status.json` and `REPORT.zh-TW.md`.
It defaults to all commissioned cameras, preserves the source scenes, and is a finite
worker rather than a daemon. Run it under a persistent user service when it must
survive SSH logout; each camera defaults to a 20-minute timeout. Completed candidates
are under `candidates/CAMERA/`; no failed build has a complete manifest.

`scene_mesh.py CAMERA --opening-controls CONTROL_DIRECTORY --out NEW_REVIEW` uses
source-bound floor contacts and visible frame edges for entrance review candidates.
It records frame residuals separately from original-mask IoU and refuses stale source
identities. See [opening controls](../docs/OPENING_CONTROLS.md) for the schema, cropped
lintel conventions, supported scope and September 12 before/after artifacts.

`rebuild_geometry.py CAMERA_JSON --calib MATCHING_CALIB --out NEW_CACHE.npz` records
fresh depth with adjacent `.depth.npy` and `.depth.json` source bindings. Reuse with
`--depth FILE.npy` validates an adjacent manifest, or one supplied by `--depth-manifest`.
Unrecorded external depth stays explicitly unverified and receives no default teacher
identity. Recorded cache plate hashes are checked when rendering. See the
[depth provenance workflow](../docs/OPENING_CONTROLS.md#depth-reuse-and-scene-provenance).

`stage0_train.py prepare --out runs/NEW_EXPERIMENT --deadline <ISO-time-with-timezone>`
freezes the source, ImageNet weights and selected segmentation data, then writes six
whole-store configurations. `stage0_train.py run --out runs/NEW_EXPERIMENT` runs their
training and held-out evaluation with an absolute deadline, child timeouts and atomic
status/report files. Use the copied script and `PYTHONPATH=<out>/snapshot/src` under a
persistent user service. CUDA is required; data and pretrained files must already be
local. No retail checkpoint initializes a held-out store. The optional final
`staff_store_probe.py` phase uses human-sorted crops and source-only standardisation.
These candidates do not install themselves. See [the protocol and Stage0–4 contracts](
../docs/STUDIOA_STAGE0_4.md) for the teacher-label limits, splits and release status.

**Question the geometry itself.** `geometry_bench.py` scores any depth source against the
floor the commissioned camera already knows is at height zero — no labels, and a flat
control so flatness cannot be read as a verdict; it is what caught MapAnything's rise as a
scale offset rather than a fix (2026-08-30). `map_anything_eval.py` asks an independent
metric model the two questions the fleet cannot check from inside — what the vfov is
(38.26° against the pinned 70.4°, still unresolved) and whether cross-camera registration
holds, with a negative control. Both instruments take two more models since 2026-09-06
(PLAN §7a.32): Depth Anything 3 and VGGT, through `syncai_bev3d.geometry_teachers`, each
by its commit id on the command line — `--backend da3|vggt --revision <sha>` for the
lens, `--source da3|vggt-floorfit --revision <sha>` for the depth. Neither has run yet;
the predictions are written in the item.

**Figures, and the audit that licenses them.** `demo_video.py` renders the three-minute
demo and **blurs every face by two instruments before any panel is drawn**;
`heads_video.py` shows every head of the network in one frame from one forward pass and
blurs the same way; `demo_gif.py` cuts the README figure from a render **and writes the
audit verdict that says it may be published**; `social_card.py` does the same for the
GitHub social preview; `heatmap3d.py` drapes a dwell or traffic heatmap over the
commissioned scene's floor for any window of tracks (`syncai_bev3d.heatmap` holds the
tested maths; `demo_video` accumulates the same field live on its 3D panel). Renders go to `assets/dev/`, which is ignored wholesale — `assets/`
itself holds only results. See CONTRIBUTING.md for the third step a store figure needs.

## [`pose/`](pose/) — the pose head's teacher, and its instruments

`vitpose_teacher.py` labels the Gold person boxes with ViTPose keypoints — the
distillation source for the bottom-up P3 head (PLAN §2.2). The other three measure what
that head learned. `eval_student.py` is gate 3's instrument: per-joint L2 and
PCK@0.2·box_height against the teacher on the test split — agreement with the teacher,
not accuracy, and PLAN carries that caveat wherever its numbers appear. `pose_overlay.py`
answers the question a person actually asks — does it look right on our own cameras —
one forward pass per frame, and gate 3's `reach_to_shelf`/`crouch` check is judged
against it. `dense_vs_box.py` crops every dense-head person the box head missed into a
contact sheet with a matched control, because whether those regions are recall or noise
decides if the dense head can supervise the box head at all.

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

### Structural camera candidates

`commissioning/structural_calibrate.py CAMERA CONTROLS --out NEW_DIR` jointly fits
focal length, pose and division distortion from source-bound vertical and orthogonal
floor lines. It preserves whole-line holdouts, reports raw-curve errors, removes stale
metric zones and writes a diagnostic camera/report without changing commissioning
inputs. Read `line_checks_passed`; CLI completion does not imply acceptance, and
`deployment_ready` remains false. Changing the lens requires a fresh depth rebuild.
See [structural controls](../docs/OPENING_CONTROLS.md#joint-structural-calibration-candidates)
for the evidence contract and the cam07 rejected-candidate comparison.

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

### StudioA scene review preparation

`annotation/studioa_review.py prepare --source DATASET NATIVE_SCHEME --out NEW_DIR`
freezes candidate frames, original masks, conservative entity proposals and empty human
review tasks. Repeat `--source` for multiple datasets. `--per-camera` defaults to three
sessions per source/camera. Merged shell/fixture labels remain ignore rather than being
guessed into new classes. `check --out DIR` validates frozen sources and reviewed
annotations, returning 2 while human review is pending. Prospective store folds do not
make previously used data a blind test set. See the
[StudioA scene contract](../docs/STUDIOA_SCENE_CONTRACT.md) for class boundaries,
annotation format, commands and acceptance limits.

### StudioA AI annotation

`annotation/studioa_autolabel.py prepare --bundle REVIEW_DIR --out NEW_DIR` freezes
images, teacher revision, policy and worker code. `run --out NEW_DIR` runs the frozen
SAM3 worker and resumes completed image checkpoints after source validation.
`refine --source COMPLETED_AI_DIR --out NEW_DIR` retains nonconflicting structural
pixels without repeating inference. Outputs include instance RLEs, unresolved
candidates, positive COCO pseudo-labels, overlays, progress and a source-bound report.
AI completion does not require human review. See the
[AI annotation guide](../docs/STUDIOA_AI_ANNOTATION.md) for commands and supervision
limits; the COCO file alone must not turn uncertain/unlabelled regions into negatives.

`annotation/studioa_supervision.py export --source COMPLETED_AI_DIR --out NEW_DIR`
builds partial 19-class semantic targets, preserving unresolved/overlapping pixels as
ignore. `check --data NEW_DIR` verifies hashes and whole-store folds.
`smoke --data NEW_DIR --held-out Taichung --out NEW_SMOKE_DIR` checks one real train
batch through HydraNet forward/backward/SGD on CPU, without saving a model or claiming
accuracy. See [training-data limits](../docs/STUDIOA_TRAINING_DATA.md); instance detection
supervision and formal training remain separate work.
