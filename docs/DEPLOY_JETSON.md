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

> **INT8 was measured slower than FP16 on at least one board.** On a GB10 with TRT 10.16,
> `--best` came back at 2.48 ms against FP16's 2.31 ms. Narrowing the neck to 64 channels
> was also slower (2.45 ms), and `num_repeats: 1` and `regnet_x_400mf` were both inside
> noise. The reason is the same for all four: `--useCudaGraph` alone takes 2.61 → 1.95 ms,
> so **this graph is launch-bound, not compute-bound** — 416 kernel launches over tensors
> as small as 4×5, where shrinking the arithmetic buys nothing.
>
> That has not been re-measured on the Orin rig this file documents, and it may well differ.
> But INT8 is the first thing anyone reaches for, so it is worth knowing before the
> calibration set is assembled that it is not free money on every board.

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

### Narrowing the detection classes at export

That sigmoid is the single largest item in the frame. On the AGX Orin the measured split
was GPU inference 5.12 ms (14%) and post-processing 16.33 ms (43%), nearly all of it
80 classes × 6,820 positions = **545,600 values per frame**. Most of those classes are
`zebra` and `snowboard`.

`--detection-classes` slices the class channels out of the classification convolution at
export. The checkpoint still trained on all 80 — COCO supervision is free signal for the
shared trunk and RETAIL_SCOPE.md §4 argues at length for keeping it — this only stops the
engine emitting, and the host decoding, what the deployment does not read.

```bash
# a shop robot: what a planner has to react to
hydranet-export-onnx --config configs/hydranet_retail.yaml \
    --checkpoint runs/hydranet_retail/best.pt \
    --output hydranet.onnx --detection-classes robot_8

# retail analytics: merchandise, fixtures, people and customers' belongings
hydranet-export-onnx --config configs/hydranet_retail_objects.yaml \
    --checkpoint runs/hydranet_retail_objects/best.pt \
    --output hydranet.onnx --detection-classes retail_analytics
```

| subset | classes | values/frame at 512×640 | host sigmoid |
|---|---|---|---|
| (none) | 80 | 545,600 | — |
| `retail_analytics` | 32 | 218,240 | **2.5×** less |
| `robot_8` | 8 | 54,560 | **10.0×** less |

### Folding the segmentation argmax into the graph

`--argmax-seg` emits uint8 class maps instead of float logits, so the host stops doing the
argmax and the D2H transfer drops from 9.18 MB to 0.33 MB. It was worth more than either
class-narrowing or any engine lever, and for a reason that had not been looked at: **the
host argmax was the largest single item in the frame, larger than the engine itself.**

GB10, TRT 10.16, 512×640, single-thread, real decode, median of three, milliseconds:

| build | infer | d2h | terrain | detect | **total** | fps |
|---|---|---|---|---|---|---|
| shipped | 2.09 | 0.38 | 2.53 | 1.70 | **6.69** | 144–150 |
| `--argmax-seg` | 2.71 | 0.23 | 0 | 1.71 | **4.00** | 203–250 |
| ` + --detection-classes retail_analytics` | 2.71 | 0.17 | 0 | 0.64 | **3.29** | 284–304 |
| ` + CUDA graph` | 1.49 | 0.15 | 0 | 0.62 | **2.25** | **381–444** |

Nothing above retrains anything or changes a weight. The last row needs no export change
at all — `--useCudaGraph` in trtexec, or `scripts/live_view_orin.py --cuda-graph`, which
saved a measured 0.87 ms/frame (30%) in the runtime.

> **Read these absolutes as ±15%, and the ratios as solid.** That GPU is shared — the same
> baseline engine measured 2.09 and 2.31 ms for `infer` on different runs. The ratios held
> across three repeats and are what the table is for; the two decimal places are the
> measurement's format, not its precision.

The CUDA graph replay was checked for the failure that looks like success: four different
real frames through the *replayed* graph give four different masks, each agreeing with its
own PyTorch reference at 99.81% — the same figure as the eager path, so capture changes
nothing numerically. A capture that replayed baked-in device memory would return a still
image at an excellent frame rate, and neither "it started" nor "it captured" would catch
that.

> **Read the `infer` column before benchmarking this.** `--argmax-seg` makes the *engine*
> slower, 2.09 → 2.71 ms. The work did not vanish; it moved onto the GPU, where it is
> cheap. Measured with `trtexec` alone the flag looks like a 25% regression, and someone
> will revert it on that basis — correctly by their measurement and wrongly by two
> milliseconds a frame. **It is a whole-frame win and it cannot be seen from the engine.**

Correctness on four real store frames, TensorRT FP16 against the FP32 PyTorch reference:
terrain argmax agrees on **99.81%** of pixels, all disagreements on class boundaries. That
is the FP16 build, not the fold — ONNX-vs-PyTorch on the folded graph is exact, zero
disagreeing pixels, which the export's `--check-parity` reports.

**There is a second, similar-looking knob, and reaching for it instead is a real mistake.**
`data.datasets[].classes` also takes a list of COCO names, and it is the one someone
looking through a config will find first:

| | what it narrows | `num_classes` | when |
|---|---|---|---|
| `data.datasets[].classes` | what the head **learns** — it changes the output space | must match it, and `config_schema.py` errors if it does not | training |
| `--detection-classes` | what the engine **emits** | left alone, with the trained weights | export |

Setting the config key against an existing checkpoint renumbers every label under it: the
run completes, the loss falls, and each box is reported as a confident wrong class. It also
throws away COCO supervision the shared trunk gets for free, which is what RETAIL_SCOPE.md
§4 is arguing against. If the goal is a smaller engine from a model you already trained, it
is the export flag, every time.

The two lists are different deployments and neither is a default. `robot_8` deletes
`book`, which the cam08 audit found 1,683 of and which is the strongest merchandise signal
the head produces — narrowing an analytics build with the robot's list would silently
remove the class the audit was about. A comma-separated list of COCO names works too.

**Two things change in the contract, both deliberately loud:**

1. **The `det_cls` bindings are renamed** to carry the count — `det_cls8_p3` rather than
   `det_cls_p3`. Same reasoning as `image_rgb_255` on the input side: a host decoder
   written for 80 classes must fail to find its binding rather than read 8 channels as 80
   and report `zebra` for a customer. `det_reg` and `det_ctr` keep their names, so a
   runtime that only wants boxes is unaffected, and an export without the flag is
   byte-for-byte the contract it always was.
2. **A `<output>.classes.json` sidecar is written**, and it has to ship with the engine.
   A TensorRT engine keeps binding names and nothing else, so the class *identities* do
   not survive the `trtexec` step. `live_view_orin.py --classes hydranet.classes.json`
   reads it; without it, against a narrowed engine, it refuses to draw rather than guess.

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
