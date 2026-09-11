# Glass-door segmentation and Stage0 geometry

The experimental specialist is trained and runnable, but its StudioA predictions are
**not promoted to commissioned masks**. The first CPU experiment detects too little of
the actual entrance leaves and mistakes some white walls, people and fixtures for glass.
The commissioned world uses the existing reviewed material masks and now draws a
transparent door leaf with an opaque perimeter frame.

## Why the old labels are insufficient

- The indoor/retail `glass` class merges windowpanes and mirrors. The retail-objects
  taxonomy merges glazing and doors into `wall`. Neither directly supervises a glass door.
- The existing SAM3 prompt audit records four glass failure modes across 112 frames,
  including predictions on things behind the glass. These pseudo-labels are not reliable
  glass-door ground truth.
- Three approximate door-leaf masks exist for Tao-Hsin cam03, cam04 and cam15, all from
  the same store. Their boundaries were reviewed image divisions of the older door mask,
  not independent precise annotations or surveyed measurements.
- A single material mask does not identify hinges, handles, opening angle or door state.
  Depth through glass often describes the background, so Stage0 fits calibrated rays to
  floor contact or a supported wall plane.

## First reproducible training experiment

[Trans10K-v2](https://github.com/xieenze/Trans2Seg) has separate transparent-object classes,
including glass doors, walls and windows. This experiment uses the locally cached
[rdyzakya RGB-mask export](https://huggingface.co/datasets/rdyzakya/Trans10K-v2), revision
`3445e93f1e507ea6c2dd36f06badfd5e818b699f`. Decode RGB colours, not numeric class IDs from
another export: grey `(120,120,120)` is the glass-door class in this copy. Unknown colours
and transparent alpha pixels are ignored. Glass shelves, cups, bottles and other small
transparent objects are background for this architectural specialist.

Exact image-content checks found three repeated training images and two validation
images identical to training images. Repeated images are ignored, leaving **4,997 unique
training images and 998 validation images**, with no exact-content overlap. This does not
establish absence of near-duplicate scenes or grouping by physical venue in public data.

The encoder is frozen HydraNet from `runs/hydranet_retail_base/best.pt` using EMA weights.
A separate convolutional head learns background, glass door, glass wall and window from
cached features. Input is letterboxed to 256 × 384; feature maps have stride 8. Training
uses weighted cross-entropy plus foreground Dice for 12 epochs, seed 42. Public validation
foreground mIoU selects the checkpoint. That validation is **not an untouched test set**;
the official test split is not cached and was not evaluated. The shipped person model is
unchanged. The current machine's NVIDIA driver was unavailable, so this run used CPU.

At original image resolution after inverse letterboxing, the selected epoch-12 head
scores **31.81% glass-door IoU**, 48.89% glass-wall IoU and 44.64% window IoU on the 998
unique validation images (foreground mIoU 41.78%). Door precision is 39.73% and recall
61.47%. These use argmax predictions; the deployment preview's 0.7 threshold is a separate
operating point. The old terrain model's generic door channel scores 23.38% IoU against
the same glass-door targets. That is a diagnostic comparison: its door class also covers
opaque doors, and its glass class combines windows and mirrors, so the taxonomies differ.
The public-data gain does not establish StudioA accuracy.

```bash
# --dataset is a directory containing the cached train/validation parquet files.
uv run python tools/commissioning/train_glazing.py cache \
  --dataset <trans10k-rgb-parquet-directory> \
  --config runs/hydranet_retail_base/config.yaml \
  --encoder-checkpoint runs/hydranet_retail_base/best.pt \
  --out runs/glazing/features
uv run python tools/commissioning/train_glazing.py train \
  --cache runs/glazing/features --out runs/glazing/model --epochs 12
uv run python tools/commissioning/glazing_pass.py Tao-Hsin-cam03 \
  --checkpoint runs/glazing/model/best.pt --out runs/glazing/review --scene
uv run python tools/commissioning/evaluate_glazing.py \
  --checkpoint runs/glazing/model/best.pt --cache runs/glazing/features \
  --validation <validation-parquet-file> --out runs/glazing/validation.json
```

Use source-resolution evaluation when comparing different input sizes: feature-resolution
IoU alone changes the scoring grid. The evaluator verifies training cache/source hashes,
checks image identities, excludes duplicate records and refuses to overwrite results.

Output directories must be new. Feature caches record source, encoder and array hashes;
training refuses modified cache arrays. Inference checks encoder identities and records
source/camera/checkpoint hashes, probabilities, masks and connected-component boxes.
Unreviewed CCTV pixels are not silently converted into negative training examples.

## Geometry and promotion

`glazing_pass.py` uses a default softmax threshold of 0.7 to propose masks. It is a proposal
threshold, not a calibrated probability of correctness. New door candidates require a
supported plane, reprojection IoU at least 0.35, estimated span 0.6–2.4 m and height
1.9–2.8 m. These bounds are priors; they do not validate the camera's metric scale.
When no model surface passes, `--scene` retains the existing GLB instead of exposing the
old generic-door fallback as a new opaque storefront. Accepted candidates remain review
previews; the tool never installs its masks.

The current glass-door asset keeps the frame inside the fitted opening. The transparent
leaf and opaque frame have separate materials. Frame depth and rail width are template
priors. There is no invented handle, hinge direction, central divider or swing angle.
Existing window/glass overlap trimming and wall apertures remain in use. Surface reports
identify support evidence and rejection reasons; reported fits precede coplanar trimming.

At threshold 0.7 the first model's mask agreement with the three existing approximate
StudioA door masks was 0.034 for Tao-Hsin cam03 and 0 for cam04/cam15. Cam03's small patch
was inside the door but covered only 3.4% of its coarse reference. Cam04's prediction was
on the floor; cam15 was missed. These are coarse-reference agreement measurements, not
precise held-out segmentation accuracy. No new model glazing surface passed the plane
checks for those three cameras. The three improved commissioned door meshes are based on
reviewed masks, **not on this model's failed predictions**.

## Next data collection

A 42-camera source-bound annotation queue and store contact sheets are available in the
local experiment bundle. Priorities include Kaohsiung cam01/cam03 and Taichung cam09 for
entrance leaves, plus Taichung cam07 and Kaohsiung cam04 for white-wall, opaque-door,
person and fixture hard negatives. Collect daytime, nighttime, glare, occlusion and
open/closed views. Annotate door leaves separately from fixed glazing; keep uncertain
pixels ignored. Group all frames of a camera together and reserve a whole store for
external evaluation. The three existing Tao-Hsin masks cannot alone establish cross-store
performance.

After that, fine-tune with real CCTV labels, evaluate door precision/recall and boundary
error, and measure temporal stability on separate clips. Higher input resolution and
trainable encoder layers are experiments to compare, not guaranteed remedies for missing
site labels. Only then should predictions replace reviewed Stage0 materials. Metric
widths and positions still need independent site measurements.
