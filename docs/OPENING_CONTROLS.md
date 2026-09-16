# Source-bound entrance controls

`scene_mesh.py` can build review candidates from visible floor contacts and frame
edges on the raw commissioned plate:

```bash
PYTHONPATH=src .venv/bin/python tools/commissioning/scene_mesh.py \
  Tao-Hsin-cam03 Tao-Hsin-cam04 Tao-Hsin-cam15 \
  --opening-controls runs/stage0_opening_review_20260912/controls \
  --out runs/NEW_OPENING_REVIEW
```

The output directory must be new. The CLI requires `--out` for controls and refuses
the ragged path. This route uses the existing object-map surface stage; it does not
provide missing object maps, independently measured scale or camera calibration.

Each `CAMERA.json` has schema 1, `camera`, `image_size_px`, `reviewer`, `evidence`,
`source_identity`, two `floor_contact_px`, `replaces_kinds` and `panels`.
`syncai_bev3d.opening_controls.source_identity(root, camera)` records the plate,
camera and relevant mask files, including absent optional material masks. A changed
plate, calibration or mask invalidates controls and fails the build. Candidate mask
overrides cannot be combined with source-bound controls. The export manifest also
hashes the control file and checks that it did not change while building.

Panel records contain a unique `id`, `kind`, `left_edge_px`, `right_edge_px`, optional
`top_edge_px`, and evidence/occlusion notes. All points use raw image coordinates,
including the lens distortion. Two visible ground contacts define an upright plane
under the current calibration; frame-edge rays fix each panel's horizontal interval.
Control scope must bind one connected glazing group. Adjacent fixed panes in that
group reuse the supported plane and retain their original material masks. Door
intervals are then subtracted from fixed panes before meshing.

Missing lintels must explicitly say `top_visibility: cropped` or `occluded`; their
2.4 m top is a drawing convention. This version does not support elevated window
sills. Degenerate, overlapping or unsupported control groups abstain together and
retain the automatic proposal, with the reason recorded. Hinge, swing direction and
the mobility of adjacent panes are not inferred from glass appearance.

The reporting layers remain separate:

- Original material-mask coverage uses `raw-glb-silhouette-v1` after reading back the
  exported GLB. Its references and thresholds are unchanged.
- `boundary_fit_residual_px` measures upright frame edges against fitting observations.
  A maximum above 8 raw pixels triggers calibration/observation review. This is a
  review threshold, not an achieved site accuracy specification.
- `final_glb_boundary_residual_px` checks those observations against the actual exported
  panel silhouette. The overlay draws individual controlled boundaries so central
  frame divisions remain visible even when the union material mask hides them.

None of these is independent validation: the observations were used for fitting.
Improved silhouette IoU cannot overrule an incompatible upright edge, and a better
frame fit does not automatically validate the existing material annotation.

The September 12 review is in `runs/stage0_opening_review_20260912/`: `index.html`
switches source, observations, before/after projection and 3D views; `review.json`
records nine-camera defects and candidate decisions. `final_candidates_v2/` contains
the final review GLBs and `source_snapshot_v2/` freezes their source. Earlier candidate
directories are superseded. No commissioned scenes were installed by this review.

## Material-reference conflicts

`scene_audit` now reports `material_reference_consistency`: glass and glass-door
pixel counts, their intersection, and the fraction of door pixels in that intersection.
Conflicting references remain visible in the original full-frame scores. The audit
does not silently remove overlap or change the scoring version.

`material_review.review_materials(root, camera, review)` returns separate candidate
arrays and an audit record without changing source masks. Its schema-1 review binds
the same source identity, camera, raw image size, reviewer and evidence as opening
controls. `facade_region_px`, `door_regions_px` and `occluded_regions_px` describe
regions observed on the source image. Only existing material-support pixels inside
the facade are allocated to mutually exclusive glass or glass-door masks; occluders
are returned as `material_ignore`, and opaque doors outside the facade are preserved.
The ignore mask is an annotation artifact, not an automatic scoring discount.

Build reviewed masks in an isolated source copy. Existing opening controls must first
validate against the original source; rebinding requires verification of unchanged
plate and camera, retention of the previous source identity, and an explicit record
of the material review. Never silently refresh stale controls. Compare old and new
scenes separately against each fixed reference set: changing references is not a
measured geometry improvement.

## Conditional vertical-pose diagnostics

`vertical_controls.load_vertical_controls(path, root, camera)` validates source-bound
schema-1 observations. In addition to source identity, reviewer and evidence, these
contain `image_size_px`, `pixel_space: raw`, and `lines`. Each line has a unique `id`,
three or more `points_px`, and a boolean `validation` assignment for the whole line.
At least three fitting lines spanning 20% of image width and two held-out lines are
required; each line must span 40 pixels.

`refine_vertical_pose(camera_file, observations, walkable=raw_walkable_mask)` returns
an isolated camera proposal and report. It fits pitch and roll conditional on fixed
intrinsics, lens and height. It checks held-out lines in raw distorted pixel space,
transfers zones through their image vertices, and checks observed floor rays. Failed
zone transfer clears proposal zones and records failure. `deployment_ready` is always
false: vertical lines cannot independently identify focal length or metric scale.
Even a passing orientation check needs entrance consistency, independent focal/scale
constraints, rebuilt depth geometry and whole-scene review before use.

The follow-up evidence is `runs/stage0_pose_review_20260912/REPORT.zh-TW.md`, with
an image switcher, a two-reference comparison matrix, and reproducible focal
sensitivity diagnostics. cam04's material candidate has no label intersection or
exported coplanar-solid overlap; cam03/cam15 pose proposals are held because entrance
contacts do not land in usable ground geometry under the current intrinsics/height.

## Depth reuse and scene provenance

`rebuild_geometry.py` now saves fresh unscaled inference as adjacent `.depth.npy` and
`.depth.json` files. The manifest binds the raw plate bytes/size, calibrated image
size, division-lens preprocessing and code, model revision, and depth bytes. Reusing
`--depth` automatically validates an adjacent JSON manifest; `--depth-manifest PATH`
selects one explicitly. Changed image, lens, preprocessing, depth bytes or dimensions
fail validation. Pitch, roll, focal length and height are not inputs to this depth
teacher and may change, but geometry and its explicit depth scale must be rebuilt.

Legacy external depths without manifests remain usable for diagnostics and are
marked `external_unverified`; the tool no longer assigns the default teacher's model
and revision to arbitrary input arrays. Fresh inference and validated reuse carry
`bound` provenance. This means content association, not independent metric accuracy.
The scene audit exposes `geometry_provenance` separately from material coverage.

The geometry loader checks recorded plate hashes against the actual source file;
the scene builder supplies the plate from its selected `--root`, so isolated review
workspaces cannot accidentally resolve another checkout's image. Historical caches
without plate hashes retain their existing ground-projection consistency check and
remain explicitly unverified in the audit.

`runs/stage0_depth_review_20260912/` contains fresh, isolated cache and scene candidates
for both unsigned caches in the nine-camera baseline: Tao-Hsin-cam04 and Taichung-cam07.
Their same-reference material IoUs remain unchanged to two decimal percentage places.
cam04's previous cache-source gap is addressed in this candidate; visual/scale limits
remain. cam07's floor-height inconsistency and missing walls still require review.

## Visible-floor checks and disappearing walls

The cam07 follow-up separates `walkable` area from **visible** floor. People, stools
and inferred floor may belong to the first but their predicted heights do not measure
the latter. `floor_review.floor_height_consistency` retains both denominators, finite
geometry counts, median heights and absolute p95 values. Missing visibility evidence
returns `visible_floor: null`, never a zero error or a depth-selected clean subset.

An optional `masks/visible_floor.review.json` supplies schema 1, camera, raw image size,
source identity, reviewer, evidence and `regions_px` polygons. They are observed on the
source image, intersected with the original walkable mask, and validated against the
plate/camera/masks. They affect diagnostics only. No selection is driven by depth
height or residual. The scene audit reports these scopes separately; neither is an
independent metric accuracy measurement.

Wall reports now retain the floor-both-sides decision and every aperture cut, including
its surface ID, fitted span, ground-contact span, remaining section count and area.
A fully removed wall is therefore distinguishable from one never fitted or rejected
by floor evidence. The ground-contact span is evidence coverage, not automatically
the full door width when contact is occluded. Geometry thresholds are unchanged.

`runs/stage0_cam07_geometry_20260912/` shows that cam07's wall passed the floor rule,
then its oversized fitted door cut nearly all of it; the remaining 3 cm/5 cm strips
were below the existing 10 cm minimum. The contact-width-only rectangle also failed
the existing fitting gate, so it was not installed. Its walkable-area p95 of 0.622 m
and reviewed-visible-floor p95 of 0.042 m have different scopes and must not be presented
as a model improvement. The underlying door/calibration conflict remains unresolved.

## Joint structural calibration candidates

`structural_controls.py` extends direction evidence to two orthogonal floor-line
families plus vertical edges. With principal point, square pixels and height fixed,
it estimates focal length, pitch, roll, floor azimuth and a centred half-diagonal
negative division lens. Controls use the vertical source identity plus an
`annotation_source` with frame path/hash, original dimensions and uniform
`scale_to_camera`; each line adds `axis: vertical|floor_u|floor_v`. At least three
vertical training lines, two training lines per floor family, and one held-out line
per direction are required. Lines contain at least three in-frame points spanning
40 pixels. `initial_floor_axis_deg` seeds train-only fitting.

The CLI `tools/commissioning/structural_calibrate.py CAMERA CONTROLS --out NEW_DIR`
never overwrites an existing output. `line_checks_passed` is distinct from process
completion and from `deployment_ready`, which remains false. It scores raw distorted
curves, checks conditioning/bounds, removes old metre-space zones, and requires fresh
depth after changing lens preprocessing. Even passing lines cannot validate camera
height, material labels or object depths. Human-reviewed observations and independent
physical controls remain necessary.

The cam07 trial in `runs/stage0_cam07_calibration_20260916/` improves door projection but
fails the 4 px holdout gate and regresses floor/fixture consistency. It is retained as
a rejected diagnostic, with the earlier official scene unchanged. Its annotations
were corrected during diagnosis; do not describe the resulting line scores as blind
validation or its inherited height as measured scale.
