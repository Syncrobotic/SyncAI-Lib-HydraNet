# Robot deployment — HydraNet on the DEEP Robotics Lite3

This is the **quadruped** deployment surface of the repo (the retail/security surface is
elsewhere). It converts a trained multitask HydraNet to RKNN, runs it live on the Lite3's
Rockchip **RK3588 NPU**, overlays traversability + terrain + detection onto the robot's own
camera, renders the project's `bev3d` metric BEV panel, and serves it all on a dashboard.

Two machines are involved:

| Machine | Role |
|---|---|
| **workstation** (x86, e.g. pro6000) | trains, exports ONNX, converts to RKNN, forwards ports for viewing |
| **Lite3** (RK3588, aarch64, Python 3.8) | runs the NPU inference + dashboard as systemd services |

Nothing here imports from `syncai_hydranet` at runtime on the robot — the robot copy is
standalone (see step 4). The reproducible *conversion* uses the repo's own
`hydranet-export-onnx`.

---

## 1. Export ONNX (workstation, in the repo's `uv` env)

The graph embeds preprocessing (raw RGB 0-255 input `image_rgb_255`, NCHW) and holds only
conv/resize/exp — no NMS, no control flow — so it converts cleanly. Outputs are the two seg
heads (traversability 3-cls, terrain 12-cls) at full res, plus FCOS `det_cls/reg/ctr` for
five FPN levels. `det_reg` is already `exp()`-scaled by stride in-graph.

```bash
# full-res (512x640)
uv run hydranet-export-onnx \
  --config runs/hydranet_joint_coco10/config.yaml \
  --checkpoint runs/hydranet_joint_coco10/best.pt \
  --output hydranet_joint_coco10.onnx --check-parity
```

To export at a smaller input for speed, set **`export.input_size`** in a copy of the config
(the `export` block wins over `data.input_size`; keep both dims divisible by 128 for the p7
FPN level — 384x512 works). NB: a `yaml.safe_dump` round-trip alphabetises `model.heads`,
which **reorders the seg outputs** — the runtime classifies outputs by channel count
(3=traversability, 12=terrain, 80=cls, 4=reg, 1=ctr), so order does not matter, but do not
hard-code it.

## 2. Convert ONNX -> RKNN (workstation)

`rknn-toolkit2 == 1.6.0` — **must match the robot's `librknnrt.so` (1.6.0)**. It is NOT on
PyPI; take the cp310 wheel from `airockchip/rknn-toolkit2` tag `v1.6.0`
(`rknn-toolkit2/packages/`). Install into a throwaway venv, then **pin `numpy<2` and
`setuptools<81`** (its onnxruntime 1.16 breaks on numpy 2; `pkg_resources` was removed in
setuptools 81).

```bash
python convert_rknn_fp16.py hydranet_joint_coco10.onnx hydranet_joint_coco10.rknn
# INT8: grab ~60 calibration frames from the robot's own camera, list them in dataset.txt,
# then quantise. INT8 keeps accuracy here and is ~25-35% faster on the NPU.
python convert_rknn_int8.py hydranet_joint_coco10.onnx hydranet_joint_coco10_int8.rknn dataset.txt
```

Measured on RK3588, pure NPU inference: fp16 512x640 ≈ 4-5 fps; **int8 384x512 ≈ 7.5 fps
single-core**. The dual full-res seg heads (memory-bound upsampling) — not the conv — are
the bottleneck, so quantisation helps less than the usual 3-5x. Lower the input resolution
for real gains.

## 3. Robot runtime deps (Lite3) — installed under `paul`

The robot ships with **no pip, no numpy**, ancient **Pillow 7.0**, and a default route
pointing at eth1's own `192.168.137.120` (a bogus Windows-ICS leftover) so it has no
internet until you repoint it:

```bash
sudo ip route add default via <ap-gateway> dev wlan0 metric 50   # e.g. the workstation AP
python3 get-pip.py                       # get-pip for 3.8: bootstrap.pypa.io/pip/3.8/get-pip.py
python3 -m pip install --user numpy==1.24.4 rknn_toolkit_lite2-1.6.0-cp38-cp38-linux_aarch64.whl
python3 -m pip install --user pillow==10.4.0     # 7.0 lacks rounded_rectangle etc.
sudo cp <path>/librknnrt.so /usr/lib/librknnrt.so    # v1.6.0; e.g. from /home/ysc/track/lib
```

`rknn_toolkit_lite2` wheel: same repo/tag, `rknn-toolkit-lite2/packages/` (cp38 aarch64).
If the robot's network is flaky, download wheels on the workstation and `scp` them over.

## 4. Copy the geometry package to the robot (for the bev3d BEV)

`bev3d` renders the metric BEV. The repo's `geometry/` package is torch-free (numpy+PIL)
but assumes **Python 3.10 / Pillow 9.1+**; the robot is 3.8. Copy the package standalone and
apply these **robot-copy-only** shims (do NOT change the repo — see `geometry/__init__.py`'s
divergence note):

```
playground/syncai_geo/labels.py           -> IGNORE = 255
playground/syncai_geo/geometry/*.py        -> copied from src/syncai_hydranet/geometry/
```

Shims on the robot copy:
- `scene_types.py`  `Scene = X | Y`            -> `typing.Union[...]`   (runtime union, 3.10+)
- `meshes.py`       `Mesh = tuple[...]`         -> `typing.Tuple[...]`   (runtime subscript, 3.9+)
- `bev.py/bev3d.py/shading.py`  `zip(..., strict=True)` -> drop `strict=True`  (3.10+).
  This drops a length guard; `bev.py`'s `zip(pos, ext, labels, scores)` would silently drop
  detections off the map on ragged input — the robot copy re-adds `assert len(pos)==len(labels)`.
- `bev3d.py`  `Image.Transform/Resampling.*`   -> old-style `Image.*`   (Pillow 9.1+ enums)

## 5. Install and start the services

```bash
# on the Lite3
sudo cp systemd/hydra-infer.service systemd/hydra-dash.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now hydra-dash hydra-infer

# on the workstation (for VS Code viewing)
mkdir -p ~/.local/hydra-forward && cp hydra_forward.py ~/.local/hydra-forward/
sudo cp systemd/hydra-forward.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now hydra-forward
```

Copy `hydra_infer.py`, `hydra_dash.py`, and the `.rknn` model to `/home/paul/playground/`
(model under `model/`).

## 6. View it

- **On the robot's network:** `http://<lite3-ip>:8080`.
- **From a laptop via VS Code Remote-SSH into the workstation:** forward port **8080** in the
  PORTS panel and open `http://localhost:8080`. The camera and the inference/BEV panels are
  proxied through 8080 (HLS at `/cam/`, frames at `/infer/frame.jpg` and `/infer/bev.jpg`),
  so a single forwarded port is enough — WebRTC would need UDP and will not cross the tunnel.

## Robot control (dashboard buttons)

Command codes were **captured from the physical controller** on UDP 43893 and confirmed by
the operator (they are undocumented and compiled into the factory `jy_exe`):

| Action | EthCommand `{code, value, type}` sent to `127.0.0.1:43893` |
|---|---|
| Emergency stop | `0x21010C0E, 2, 0` |
| Stand up / crouch (toggle) | `0x21010202, 0, 0` |

The packet is `struct {uint32 code; uint32 value; uint32 type;}` little-endian. The physical
hardware e-stop remains the final safety layer.

## Telemetry source (no extra hooks)

IMU + 12 joints are read **passively** by binding UDP `43897`, where `jy_exe` already streams
its state (380-byte packets, code `0x0906`; layout: 12B header + tick + IMU 9 floats @off20 +
12x JointData @off56). **Never call the MotionSDK `RobotStateInit()`** — it seizes control and
moves the robot. Battery is not in this stream (BMS is proprietary over CAN) and is shown as N/A.

## Environment knobs (hydra_infer.py)

`HYDRA_MODEL` (rknn path), `HYDRA_W`/`HYDRA_H` (model input, match the export),
`HYDRA_VIEW` (`trav`|`terrain`|`both`), `HYDRA_VFOV`/`HYDRA_CAMH`/`HYDRA_PITCH` (BEV camera
assumptions — mono scale is a guess; calibrate `k1`+focal once with `geometry/calibrate.py`
and estimate the plane per-frame from IMU+leg kinematics, or add a depth camera and use
`geometry/depth_scene.py` for true metric geometry), `HYDRA_BEV_EVERY` (render the CPU-heavy
BEV every N-th frame to keep the main overlay fast).
