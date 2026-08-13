# Jetson Orin deployment guide

## Pipeline overview

```
PyTorch training (workstation GPU)
   → export_onnx.py exports ONNX
   → trtexec on the Jetson builds a TensorRT engine (FP16/INT8)
   → C++/Python runtime inference + CPU post-processing (argmax / NMS)
```

## 1. Export ONNX (on the workstation)

```bash
uv run hydranet-export-onnx \
    --config configs/hydranet_regnet800mf.yaml \
    --checkpoint runs/hydranet_regnet800mf/best.pt \
    --output hydranet.onnx
```

By design the forward graph contains only Conv/BN/ReLU/Resize/MaxPool/Exp/Mul, with no NMS,
no dynamic shapes and no custom operators — so TRT converts it on the first attempt.

## 2. Build the TensorRT engine (run on the Jetson)

```bash
# FP16 (the recommended starting point on Orin, roughly 2× FP32 speed)
trtexec --onnx=hydranet.onnx --saveEngine=hydranet_fp16.engine --fp16

# measure latency
trtexec --loadEngine=hydranet_fp16.engine --iterations=200 --avgRuns=100
```

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
