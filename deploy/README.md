# deploy/ — deployment surfaces

A **deployment surface** is where a trained model becomes a running thing in its target
environment, plus the ops to get it there. It is the *downstream* end of the pipeline.

This repo is **one core, two products**:

| surface | product | environment |
|---|---|---|
| [`robot/`](robot/) | **A — quadruped** | on the Lite3's RK3588 NPU: real-time perception → BEV/costmap → control |
| [`retail-security/`](retail-security/) | **B — retail + security analytics** | server-side, fixed store CCTV: L0 inference → L1 tracks → L2 events |

Both consume the **same core** (`src/syncai_hydranet`: the HydraNet model, `geometry/`,
training, ONNX export). The core never imports a surface; each surface depends on the core.
That one-way arrow is what keeps the two products two *consumers of one codebase* rather
than two copies that drift.

**What is not here:** *data production*. Annotation (CVAT) produces the training labels for
both products and sits *upstream* of training, so it lives in [`../tools/annotation/`](../tools/annotation/),
not under `deploy/`. Putting it here would confuse "how we make labels" with "where the
model runs."

When to split this into separate repos: only on a real forcing function — the two surfaces
gaining separate teams, separate release cadences, or a security/IP boundary (product B
ships to a cloud you run; product A ships to customer hardware). While one team shares the
core, the monorepo's single CI proving a model change against *both* surfaces is worth more
than the isolation.
