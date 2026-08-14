# Jetson Orin deployment guide

## Pipeline overview

```
PyTorch training (workstation GPU)
   → export_onnx.py exports ONNX
   → trtexec on the Jetson builds a TensorRT engine (FP16/INT8)
   → C++/Python runtime inference + CPU post-processing (argmax / NMS)
```

## 0. The test rig

| | |
|---|---|
| Board | Jetson AGX Orin Developer Kit, 64 GB |
| L4T | R36.4.0 — JetPack 6.1 |
| Stack | CUDA 12.6, cuDNN 9.3, **TensorRT 10.3** |
| Camera | AVerMedia Live Streamer CAM 313, USB UVC on `/dev/video0`, MJPG/YUYV up to 1920×1080 |

Two things a fresh board needs before any of this works. **TensorRT is not in the base L4T
image** — `sudo apt install nvidia-jetpack` (or just `tensorrt` for a smaller install), with
the NVIDIA apt repo already configured in `/etc/apt/sources.list.d/`. And the user must be
in the `video` group to open the camera at all: `sudo usermod -aG video "$USER"`, then log
back in.

Note the TensorRT major version. TRT 10 removed `--workspace`; use
`--memPoolSize=workspace:N` instead. Opset 17 needs no downgrade for this stack.

**The camera is 16:9 and the model input is 1.25:1.** Letterbox — do not resize — or every
learned notion of shape is squeezed horizontally. `data.letterbox: true` exists for this.

## 1. Export ONNX (on the workstation)

```bash
uv run hydranet-export-onnx \
    --config configs/hydranet_regnet800mf.yaml \
    --checkpoint runs/hydranet_regnet800mf/best.pt \
    --output hydranet.onnx --check-parity
```

By design the forward graph contains only Conv/BN/ReLU/Resize/MaxPool/Exp/Mul, with no NMS,
no dynamic shapes and no custom operators — so TRT converts it on the first attempt.

`--check-parity` runs the exported graph against PyTorch on the same input and fails if they
disagree. It is the acceptance gate, it needs `onnxruntime`, and it compares **relative**
error per output: the outputs span three orders of magnitude, so a single absolute threshold
either fires spuriously on the regression maps or is loosened until it cannot see a real
error in the logits.

### The graph normalises. The robot must not.

The exported graph subtracts the ImageNet mean and divides by the standard deviation
itself, so its input is **raw RGB in 0–255, NCHW**. The robot's job is to letterbox, convert
BGR→RGB, transpose, and hand the pixels over.

This used to be the runtime's job, with the constants copied by hand into
`scripts/bench_camera_orin.py`. Nothing tied that copy to `data/transforms.py`: change one
and no test fails, no error appears, and the model is simply worse in a way that gets
blamed on quantisation. Folded into the graph, the constants ship with the weights and
TensorRT fuses them into the first convolution, so the cost is nil.

**The input binding name carries the contract**, because a TensorRT engine keeps binding
names but not ONNX metadata:

| input name | meaning |
|---|---|
| `image_rgb_255` | the graph normalises — feed raw pixels, subtract nothing |
| `images` | pre-normalised, from an export before this change — the runtime owns mean/std |

`bench_camera_orin.py` and `live_view_orin.py` read that name and switch. A runtime that
ignores it and normalises anyway does so *twice*, which is silent and costs accuracy, not
a crash. `--no-embed-preprocessing` restores the old convention if an existing runtime
needs it.

## 2. Build the TensorRT engine (run on the Jetson)

```bash
# FP16 (the recommended starting point on Orin, roughly 2× FP32 speed)
trtexec --onnx=hydranet.onnx --saveEngine=hydranet_fp16.engine --fp16 \
        --memPoolSize=workspace:4096

# measure latency
trtexec --loadEngine=hydranet_fp16.engine --iterations=200 --avgRuns=100
```

`scripts/bench_orin.sh` does both precisions in one pass. Set the board to its full power
mode first or the numbers understate it:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

**`trtexec` measures GPU inference only, and quoting it as FPS overstates the robot.** For
the number that matters, `scripts/bench_camera_orin.py` runs the whole path from the camera
and times each stage separately, so the bottleneck is visible rather than assumed.

INT8 needs a calibration dataset (a few hundred real images):

```bash
trtexec --onnx=hydranet.onnx --saveEngine=hydranet_int8.engine \
        --int8 --calib=<calibration cache>
```

Recommendation: ship FP16 first. INT8 has a visible effect on segmentation edge quality and
requires re-validating mIoU.

## 3. Output nodes and post-processing

| Output | Shape | Post-processing |
|---|---|---|
| `traversability` | [B, 3, H, W] | channel argmax → 0/1/2 per pixel |
| `terrain` | [B, 12, H, W] | channel argmax |
| `det_cls_p3..p7` | [B, 80, h, w] | sigmoid |
| `det_reg_p3..p7` | [B, 4, h, w] | already pixel distances (l,t,r,b) |
| `det_ctr_p3..p7` | [B, 1, h, w] | sigmoid, multiplied into the cls score |

Detection decoding, for each grid position (x, y) at each level (centre = `(x+0.5)*stride`):

```
score = sigmoid(cls) * sigmoid(ctr)
box   = [cx - l, cy - t, cx + r, cy + b]
```

Then apply a score threshold and NMS. The Python reference implementation is
`syncai_hydranet/models/heads/detection.py::FCOSHead.decode`; a C++ port is around 100 lines.

## 4. Expected performance (RegNetX-800MF + BiFPN96, 512x640, FP16)

| Platform | Estimated latency |
|---|---|
| Orin NX 16GB | ~12–18 ms |
| AGX Orin 64GB | ~5–8 ms |

If that is not fast enough, adjust in this order (cheapest first):

1. Drop the input to 384×512 (smallest loss in segmentation accuracy)
2. `model.neck.num_repeats: 1`
3. Switch the backbone to `regnet_x_400mf`
4. Detection head `num_convs: 2`

## 5. Matching the pre-processing

The engine input is `(pixel/255 - mean) / std` with ImageNet mean/std, in RGB order.
If the camera gives BGR (OpenCV), remember to convert to RGB — otherwise accuracy quietly
drops by 5–10 points.
On the Jetson, prefer a CUDA kernel or VPI for pre-processing to avoid a CPU bottleneck.
