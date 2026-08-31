# SyncAI-Lib-HydraNet — security & retail analytics for fixed store CCTV

[![CI](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/actions/workflows/ci.yml)
[![Python 3.11 – 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org/)
[![Licence Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-green)](LICENSE)

One camera, one model, two readings: **loss prevention** (who entered where, what did they
do, did stock leave unpaid) and **retail analytics** (footfall, dwell, paths, queues) from
the fixed CCTV already on the ceiling. No LiDAR, no new hardware.

![Kaohsiung-cam04: detections and tracks on the left, the metric 3D scene on the right](assets/demo_Kaohsiung-cam04.gif)

*Left: person boxes and confirmed tracks, with this camera's false-positive polygons
applied — **staff blue, everything else green**, one verdict per person from nine
torso-colour statistics. Two colours, not three: a track too short to have a verdict is
drawn as a customer, so a member of staff crossing in under about a second is green — the
cost of not putting a third colour on screen that a viewer has to be told how to read.
Right: the
same moment in metres, the store's own furniture reconstructed from one static plate, a
figure at every tracked shopper's floor position. Nothing is drawn by hand and no second
sensor is involved.*

*Three things to read the panels with. **Every face is blurred by `demo_video.py` itself**,
by two instruments, and `demo_gif.py` then re-runs the detector on the source frames at a
far lower threshold and refuses to write the figure unless every person it finds falls
inside a blurred region — it has refused one (PLAN §7.24). **The right panel has fixtures,
not a room**: a fixed camera sees part of one store, so the walls are the runs that were
observed rather than a closed boundary, and wall and column heights are a stated constant,
printed on the panel, because the depth model collapses on white surfaces. And **the window
is the busiest two minutes of a three-minute clip**, chosen automatically by
`demo_gif.py --start auto` — a figure of an empty shop shows nothing, but it is a selection
and this is it being said.*

*The colours are licensed per camera and refused where they are not earned. This camera
scores 1.00 held out on its own 15 labelled crops; the same model is refused on
Tao-Hsin-cam04, which scores 0.417. And it is not a model that paints everyone one colour:
pointed at Taichung-cam01, a repair counter, it calls **98.5% of person-observations
staff**, 2,144 against 34.*

**A second store**, the same code, and a camera whose colours had to be earned twice:

![Taichung-cam10: detections and tracks on the left, the metric 3D scene on the right](assets/demo_Taichung-cam10.gif)

*Taichung-cam10 **is** coloured, and what it took is the point. It had 15 labelled crops,
all 15 staff and 0 customers, so a model held out on it scored a clean 1.000 that measured
only whether it calls staff staff — an accuracy that cannot be wrong about customers cannot
license colouring them, and the gate refused it (PLAN §7.23). It has 127 crops now — 47
staff, 65 customer, 6 unclear — and the model reads **0.874** held out on them. That is
below the derived 0.90 floor, so the exception is stated at the call site with
`--staff-min-accuracy 0.85`, printed on the figure, and recorded in its verdict: a figure
never carries a threshold nobody can see. **Read its metres with the caveat**: this
camera's scale is known to be 1.21x too large (PLAN §7.10, a decision to leave rather than
an oversight), which is why its figures stand 1.98 m rather than 1.70.*

The whole plan — architecture, data strategy, build order, and the measurements behind
every decision — lives in **one document: [docs/PLAN.md](docs/PLAN.md)**. Everything the
project previously documented is in git history (`git show b7457c2:docs/<file>`).

## The design in one paragraph

Two packages with one contract between them. `syncai_bev3d` runs **once per camera**: it
builds a static plate, fits the ground geometry, runs the teachers one time, and emits a
`camera.json` — walkable floor, walls, columns, doors, display tables and shelves,
products down to `iphone / ipad / macbook / boxed_stock`, shelf ROIs, and the derived
false-positive polygons. The same artefacts render a metric 3D scene per camera, exported
as `scene.glb` / `scene.obj`. `syncai_hydranet` runs **every frame**: a shared
RegNetX-800MF + BiFPN trunk carrying **the two heads this product trains** — detection
(`person`, `bag`, `device`, `boxed_stock`) and pose (17 keypoint heatmaps at P3, decoded
inside the detection boxes) — whose boxes become tracks *in metres* through the cached
geometry. The model defines two more families, segmentation and monocular depth, and each
config trains the subset it names; the retail-security configs name neither, which is why
this paragraph says two where `models/hydranet.py` holds four. Everything above that
is rules, a tiny temporal model, and a VLM on trigger.

**Anything constant on a fixed camera is cached, never learned; only what changes
frame-to-frame spends the GPU.** The boundary is enforced rather than remembered:
`tests/test_package_boundaries.py` fails if a serving-path module ever imports
`syncai_bev3d`.

## Two model suites, one product

The project runs **two very different kinds of model**, and confusing them is the
classic failure this architecture exists to prevent:

| | the **teachers** (`syncai_bev3d`) | the **student** (`syncai_hydranet`) |
|---|---|---|
| models | SAM 3, Grounding DINO, Depth-Anything V2, ViTPose — hundreds of millions to billions of parameters | one HydraNet, **~8 M parameters** |
| when | **once per camera** (commissioning) and **once per dataset** (labelling) | **every frame, 96 streams × 5 fps** |
| where the answers go | cached: `camera.json`, masks, keypoint files | inferred: boxes + keypoints per frame |
| allowed to be slow | yes — 40 s per plate is fine | no — the whole budget is 480 frames/s (PLAN §7.4) |

```mermaid
flowchart LR
    subgraph OFFLINE ["once per camera / dataset — syncai_bev3d"]
        PLATE["static plate"] --> TEACH["SAM3 · GDINO · DA-V2<br/>+ depth & floor-boundary completion"]
        TEACH --> CJ[("camera.json<br/>masks · walkable · zones<br/>shelf ROIs · FP polygons")]
        TEACH --> SCENE["3D scene<br/>GLB / OBJ"]
        VIT["ViTPose over Gold boxes"] --> KP[("keypoint labels")]
    end
    subgraph ONLINE ["every frame — syncai_hydranet"]
        NET["HydraNet ~8M"] --> TRK["tracks in metres"] --> EV["rules → events → alerts"]
    end
    CJ --> TRK
    KP -. distillation .-> NET
```

## Install & run

```bash
uv sync                                        # or: pip install -e .
hydranet-train --config configs/<config>.yaml  # train
hydranet-eval --config configs/<config>.yaml   # evaluate
hydranet-infer-video ...                       # overlay inference on a clip
hydranet-export-onnx ...                       # ONNX for the TensorRT path
```

Entry points: `hydranet-train`, `hydranet-eval`, `hydranet-infer-image`,
`hydranet-infer-video`, `hydranet-scene`, `hydranet-export-onnx`, `hydranet-annotation`,
`hydranet-report`, and the two dataset preparers `hydranet-prepare-ade20k` and
`hydranet-prepare-cocostuff` — ten, which is what `[project.scripts]` in `pyproject.toml`
declares.

## Layout

| path | what |
|---|---|
| `src/syncai_bev3d/` | commissioning: calibration fitting, plate pipeline, SAM 3 / Grounding DINO teachers, BEV & scene rendering — runs once per camera |
| `src/syncai_hydranet/` | the per-frame side: models, training engine, data, runtime geometry + the `camera.json` contract, serving, analytics |
| `configs/` | training configs — `config.yaml` inside a run directory is the only authoritative record of what a run trained on |
| `docs/PLAN.md` | the plan; the single source of truth |
| `tools/commissioning/` | the per-camera pipeline: metre-grid verification, structure masks, depth completion, product subclasses, false-positive polygons, and the 3D scene renders (`scene_mesh.py` → GLB/OBJ). `masks_diagnose.py` and `cluster_rules.py` are its instruments: the first keeps the per-cluster verdict the pipeline otherwise prints as a total, the second replays those cached proposals through a different rule with no GPU |
| `tools/pose/` | the ViTPose teacher run that labels the Gold boxes for the pose head |
| `tools/site30k/`, `scripts/` | campaign tooling, static plates, teachers' CLI front ends, and the measurement instruments a claim in this file rests on — `track_review.py` (turn track ground truth into minutes of judgement), `track_idf1.py` (both trackers over one inference pass), `zone_dwell.py` (does a visit survive the tracker) |
| `datasets/`, `runs/`, `exports/`, `weights/` | data and artefacts (largely gitignored) |

## Credits

**Teachers.** The commissioning pass runs four published models once per camera and caches
their answers; none of them is in the serving path. Each is pinned to an exact commit in
`src/syncai_bev3d/teachers/` and `tools/pose/`.

| model | used for |
|---|---|
| [SAM 3](https://huggingface.co/facebook/sam3) (Meta) | promptable segmentation of the static plate — fixtures, floor, products |
| [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) (IDEA Research) | open-vocabulary boxes, and the `person` proposals the student is distilled from |
| [Depth-Anything V2 Metric Indoor](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf) | fixture heights where the depth holds |
| [ViTPose](https://huggingface.co/usyd-community/vitpose-base-simple) (USyd) | the 17-keypoint labels the pose head is trained against |

**Datasets.** Public sets used for pre-training, mixing and evaluation:
[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) ·
[COCO](https://cocodataset.org/) and
[COCO-Stuff](https://github.com/nightrome/cocostuff) ·
[HM3D](https://aihabitat.org/datasets/hm3d/) ·
[NYU Depth v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) ·
[RAP v2](https://www.rapdataset.com/) ·
[PA-100K](https://github.com/xh-liu/HydraPlus-Net) ·
[PETA](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html) ·
[Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) ·
[PoseLift](https://github.com/TeCSAR-UNCC/PoseLift). Each keeps its own licence and terms;
this repository redistributes none of them.

**Store footage.** Every frame of a shop in this repository comes from the deployment
partner's own cameras, is used with their permission, and has every face blurred by
`demo_video.py` before the file exists — see the figure caption above. Raw clips and plates
are gitignored and are not redistributed.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
