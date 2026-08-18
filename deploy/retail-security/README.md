# deploy/retail-security — Product B: retail + security analytics

The server-side deployment surface for **fixed store CCTV**. "Retail" and "security" are
**not two products** — they are two questions the same camera answers, off one model, one
export, one latency budget (see [`docs/RETAIL_SECURITY.md`](../../docs/RETAIL_SECURITY.md)
and [`docs/ARCHITECTURE_DIRECTION.md`](../../docs/ARCHITECTURE_DIRECTION.md)). Retail asks
*what merchandise, where do people go, how long do they stay*; security asks *who entered
where, how many, how long, what did they do*.

Contrast with [`../robot/`](../robot/): same core model, different surface —
server not NPU, tracks-and-events not BEV-and-control, fixed camera not a walking one.

## The pipeline it hosts (three layers, and the line that matters is the first)

| layer | code (already in core) | produces |
|---|---|---|
| **L0** | `HydraNet.forward` → ONNX via `hydranet-export-onnx` → TensorRT engine on the Orin | `terrain` logits + FCOS `det_cls/reg/ctr` over the retail-security vocabulary |
| **L1** | [`analytics/stage.py`](../../src/syncai_hydranet/analytics/stage.py) | `Track`s (+ 17 keypoints/person when the pose model exists) |
| **L2** | [`analytics/events.py`](../../src/syncai_hydranet/analytics/events.py) | one row per event, with the value measured and the threshold crossed |

**Nothing crosses from L2 into L0.** Zones are polygons **in metres on the floor**, not
pixel classes; thresholds ("four minutes is loitering", "six people is too many") are
config values a store manager changes on a Tuesday, not weights. The model contributes
boxes and a terrain map; everything situational is host-side and configurable. The
detection vocabulary that lets one head answer both questions is
[`data/label_maps_retail_security.py`](../../src/syncai_hydranet/data/label_maps_retail_security.py).

## Status — what exists vs what this directory will assemble

**Exists, in `src/`:** the model + export (L0), and the L1/L2 analytics primitives.
Training configs: `configs/hydranet_retail_*` and `configs/hydranet_retail_security*`.

**Not yet assembled here:** the runtime that wires L0 inference (engine) + L1 + L2 into a
running service, its per-site config (the floor-metre zones, the event thresholds), and the
ops to run it (compose/systemd, camera ingest, output sink). That is what belongs in this
directory, symmetric with `deploy/robot/`'s inference service — added when the server-side
deployment is built, not invented before it exists.

The annotation that feeds this product's training is [`../../tools/annotation/`](../../tools/annotation/)
(shared with product A, upstream of training).
