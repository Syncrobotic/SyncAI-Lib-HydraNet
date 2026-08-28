# deploy/retail-security — retail + security analytics

The server-side deployment surface for **fixed store CCTV**. "Retail" and "security" are
**not two products** — they are two questions the same camera answers, off one model, one
export, one latency budget ([`docs/PLAN.md`](../../docs/PLAN.md) §2). Retail asks *what
merchandise, where do people go, how long do they stay*; security asks *who entered where,
how many, how long, what did they do*.

It is the only surface this repository has. There were two until 2026-08-19, when the
quadruped line was removed (`eca3814`) — [`../README.md`](../README.md) keeps the argument
about what that removal proved, because it is about the boundary rather than about the
robot.

## The pipeline it hosts

`docs/PLAN.md` §2.3 numbers five layers; the three below are the ones a serving process
runs on every frame it keeps. **L2 there is the per-track crop heads and L3 is the rules**
— an earlier version of this file called the rules "L2", which collides with PLAN's
numbering on the one term both documents use.

| layer | code (already in core) | produces |
|---|---|---|
| **L0** | `HydraNet.forward` → ONNX via `hydranet-export-onnx` → TensorRT engine | `terrain` logits, FCOS `det_cls/det_reg/det_ctr` over the retail-security vocabulary, and `pose_heatmap_p3` |
| **L1** | [`analytics/stage.py`](../../src/syncai_hydranet/analytics/stage.py) | `Track`s **in metres**, carrying 17 keypoints per person and the detector confidence they were built from |
| **L3** | [`analytics/events/`](../../src/syncai_hydranet/analytics/events/) | one row per event, with the value measured and the threshold crossed |

**The target is one RTX PRO 6000, server-side** — `docs/PLAN.md` §7 decision 4: 96
concurrent streams analysed at 5 fps, with the rest of the card budgeted for a VLM on
trigger. **The Orin is not a target at all** as of 2026-08-28 — the board scripts that
kept a standalone copy of the preprocessing constants went with it, and the last thing in
this repository that read an export's `.classes.json` sidecar went with them.

**Nothing crosses from the rules back into L0.** Zones are polygons **in metres on the
floor**, not pixel classes; thresholds ("four minutes is loitering", "six people is too
many") are config values a store manager changes on a Tuesday, not weights. The model
contributes boxes, keypoints and a terrain map; everything situational is host-side and
configurable. The detection vocabulary that lets one head answer both questions is
[`data/label_maps_retail_security.py`](../../src/syncai_hydranet/data/label_maps_retail_security.py).

## Status — what exists vs what this directory will assemble

**Exists, in `src/`:** the model and its export (L0); the L1/L3 analytics primitives; and
since 2026-08-19 the multi-stream serving pieces themselves —
[`serving/`](../../src/syncai_hydranet/serving/) holds the uint8 graph surgery, the
TensorRT executor with overlapped H2D, the fixed-batch scheduler, per-camera state and the
host-side FCOS decode. `scripts/serve_pilot.py` drives them as a pilot. Training configs:
`configs/hydranet_retail_*` and `configs/hydranet_retail_security*`; per-class working
thresholds: `configs/serving/thresholds_retail_security.json`.

**Not yet assembled here:** a *service*. What is missing is not the inference code any
more — it is the packaging around it: per-site config (the floor-metre zones, the event
thresholds), camera ingest, an output sink, and the ops to run it (compose/systemd). That
is what belongs in this directory, added when it is built rather than invented before it
exists. **Read that sentence rather than the layer table above it:** what ships today is
still an ONNX, a hand-run engine build and a pilot script.

The annotation that feeds this product's training is
[`../../tools/annotation/`](../../tools/annotation/) — upstream of training, which is why
it is not here.
