# SyncAI-Lib-HydraNet — security & retail analytics for fixed store CCTV

[![CI](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/actions/workflows/ci.yml)
[![Python 3.10 – 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org/)
[![Licence Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-green)](LICENSE)

One camera, one model, two readings: **loss prevention** (who entered where, what did they
do, did stock leave unpaid) and **retail analytics** (footfall, dwell, paths, queues) from
the fixed CCTV already on the ceiling. No LiDAR, no new hardware.

The whole plan — architecture, data strategy, build order, and the measurements behind
every decision — lives in **one document: [docs/PLAN.md](docs/PLAN.md)**. Everything the
project previously documented is in git history (`git show b7457c2:docs/<file>`).

## The design in one paragraph

Two packages with one contract between them. `syncai_bev3d` runs **once per camera**: it
builds a static plate, takes 4 calibration clicks into a pixels↔metres homography, runs
SAM 3 + Grounding DINO one time to cache the structure masks (floor, walls, glass,
fixtures, walkable area), and emits a `camera.json`. `syncai_hydranet` runs **every
frame**: a shared RegNetX-800MF + BiFPN trunk with two heads — detection (`person`, `bag`,
`device`, `boxed_stock`) and pose (17 keypoints) — whose boxes become tracks *in metres*
through the cached homography. Everything above that is rules, a tiny temporal model, and
a VLM on trigger. Anything constant on a fixed camera is cached, never learned; only what
changes frame-to-frame spends the GPU. (`syncai_bev3d` is build-order step 1 in the plan;
its code currently lives in `syncai_hydranet.geometry`, `data/teachers` and
`tools/site30k`.)

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
`hydranet-report` (see `pyproject.toml`).

## Layout

| path | what |
|---|---|
| `src/syncai_hydranet/` | the package: models, training engine, data, teachers, geometry, serving, analytics |
| `configs/` | training configs — `config.yaml` inside a run directory is the only authoritative record of what a run trained on |
| `docs/PLAN.md` | the plan; the single source of truth |
| `tools/`, `scripts/` | commissioning recipe, static plates, campaign tooling |
| `datasets/`, `runs/`, `exports/`, `weights/` | data and artefacts (largely gitignored) |

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
