# Deployment: from a checkpoint to a board

One file for the whole downstream path, merged 2026-08-19 from `DEPLOY_JETSON.md`,
`DEPLOY.md` and `DEPLOY.md` — three documents that were always read in sequence.

* **Part I — the export contract and the engine.** ONNX, TensorRT, the two flags worth
  knowing before you benchmark, INT8's real cost.
* **Part II — bringing up a board from scratch.** Every step walked on a real AGX Orin;
  every failure listed is one that happened, symptom first.
* **Part III — a workstation that is not a Jetson.** Apple Silicon for local development,
  and what it can and cannot do.

**Two targets, one ONNX.** The retail/security line runs TensorRT on an Orin; the robot
line runs RKNN on the Lite3's RK3588 NPU, toolkit pinned to the board's `librknnrt`. Both
are *builds* of one released model — [RELEASE.md](RELEASE.md) §2 owns that distinction, and
a build is not accepted without its own metrics beside it.

---

# Part I — the export contract and the engine

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
shared trunk and RETAIL.md §4 argues at length for keeping it — this only stops the
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
throws away COCO supervision the shared trunk gets for free, which is what RETAIL.md
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

---

# Part II — bringing up a board from scratch

From a freshly flashed board to a live prediction stream. Every step here was walked on a
real AGX Orin, and the failures listed are ones that actually happened rather than ones
that might — each is written with the symptom first, so a future reader can match what they
are seeing rather than read the whole page.

Budget about an hour, most of it the JetPack download.

For what happens after the board works — the ONNX contract, post-processing, INT8 — see
[DEPLOY.md](DEPLOY.md).

---

## 0. Before you start

You need the board's IP, an account on it with `sudo`, and a workstation that can reach it.

```bash
ping -c1 <orin-ip> && nc -z <orin-ip> 22 && echo reachable
```

**Check the address is actually on your subnet.** `10.8.140.124` and `19.8.140.124` differ
by one character and the second is public routable space. A "connection refused" against a
plausible-looking address is worth ten seconds of `ip route` before anything else.

### Use a key, not the password

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@<orin-ip>
ssh -o BatchMode=yes <user>@<orin-ip> 'echo key auth OK'
```

Dev kits ship with weak shared passwords and this one has `sudo`. One `ssh-copy-id` and the
password stops being the thing standing between the network and the board.

## 1. Identify the board

```bash
tr -d '\0' < /proc/device-tree/model    # e.g. NVIDIA Jetson AGX Orin Developer Kit
head -1 /etc/nv_tegra_release           # e.g. R36 (release), REVISION: 4.0  -> JetPack 6.1
free -g | awk '/Mem:/{print $2" GB"}'
nvpmodel -q                             # power mode
```

L4T R36.4.0 is JetPack 6.1: CUDA 12.6, cuDNN 9.3, TensorRT 10.3. **Write the numbers down** —
an engine is only valid for the exact stack that built it, so this is release metadata, not
trivia.

AGX Orin 64 GB and Orin NX 16 GB differ by roughly 2× in latency. Know which you have before
comparing against anyone's numbers.

## 2. Install the CUDA stack

> **Symptom:** `trtexec: command not found`, no `/usr/src/tensorrt`, `import tensorrt` fails,
> `/usr/local/cuda` does not exist.
>
> **Cause:** TensorRT and the CUDA toolkit are **not in the base L4T image.** A flashed board
> has drivers and multimedia only.

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-jetpack   # ~8-10 GB
```

The NVIDIA apt repo is already configured on a flashed board. Run it detached — an SSH drop
mid-install leaves dpkg half-configured:

```bash
sudo systemd-run --unit=jetpack-install --collect \
  bash -c 'DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-jetpack'
journalctl -u jetpack-install -f
```

Verify:

```bash
export PATH=$PATH:/usr/src/tensorrt/bin
trtexec --version                                  # TensorRT v100300
python3 -c "import tensorrt as t; print(t.__version__)"
```

### Known: the meta-package version does not match the base image

On our board the repo offered `nvidia-jetpack 6.2.1+b38` while the flashed base is
`nvidia-l4t-core 36.4.0` — JetPack **6.2** userspace on a **6.1** base. Everything below
works anyway, but it is the leading suspect for the permission problem in section 4, so
record what you installed:

```bash
dpkg -l | grep -E 'nvidia-jetpack |nvidia-l4t-core '
```

If you want the matching version, pin it rather than taking the candidate.

## 3. Camera access

> **Symptom:** `/dev/video0` exists but opening it raises `PermissionError`.
>
> **Cause:** the login user is not in the `video` group. Flashed images only add the
> account created during first boot.

```bash
sudo usermod -aG video "$USER"     # log out and back in; group changes need a new session
ls -l /dev/video*
```

Check the format and, more importantly, the **aspect ratio**:

```bash
v4l2-ctl --list-formats-ext -d /dev/video0    # or the ioctl probe if v4l2-utils is absent
```

Our camera is 16:9 (1280×720 MJPG) against a model input of 640×512, which is 1.25:1.
**Letterbox, never resize** — a plain resize squeezes the image horizontally and breaks
every learned notion of shape. `data.letterbox: true` exists for exactly this.

## 4. Verify CUDA actually initialises — as the user who will run it

```bash
python3 -c 'import ctypes; print(ctypes.CDLL("libcuda.so.1").cuInit(0))'   # 0 is success
```

> **Symptom:** `801`, and `trtexec` dies with `Cuda failure: operation not supported` right
> after printing `=== Device Information ===`.

This is the one thing on this page we did **not** resolve. On our board:

| Tried | Result |
|---|---|
| `sudo` | `cuInit` → **0** |
| user in `video` | 801 |
| `chgrp video` on `/dev/nvgpu/igpu0/sched` and `/dev/nvhost-sched-gpu`, plus a udev rule | 801 |
| user also in `debug` | 801 |
| reboot after JetPack install | 801 |

Root works and no combination of groups did, which rules out plain device-node permissions.
The JetPack-6.2-on-6.1 mismatch in section 2 is the open suspect.

**Check this early.** Everything after it depends on CUDA working, and discovering it during
a benchmark costs an hour of misattributed debugging. If your board returns 0 as a normal
user, skip the workaround below and run everything unprivileged.

**Workaround, with its cost stated:** run the GPU processes as root. Acceptable for
benchmarking on an internal network; **not** acceptable for a network-facing service on a
robot. Fix it properly before anything ships.

## 5. Copy the model over and build an engine

Export on the workstation with the parity gate, then copy the ONNX — never the engine, which
will not load on a different board or stack:

```bash
# workstation
uv run hydranet-export-onnx --config <run>/config.yaml --checkpoint <run>/best.pt \
    --output hydranet.onnx --check-parity
scp hydranet.onnx scripts/bench_orin.sh scripts/bench_camera_orin.py \
    scripts/live_view_orin.py <user>@<orin-ip>:~/

# orin
sudo nvpmodel -m 0 && sudo jetson_clocks     # or the numbers understate the board
sudo bash ~/bench_orin.sh ~/hydranet.onnx
```

The build takes several minutes per precision — TensorRT profiles many kernel choices.

> **Symptom:** `trtexec` exits immediately and your log has one harmless-looking line.
>
> **Cause:** a filter that is too narrow. `bench_orin.sh` originally grepped for `error`,
> which matched the *setting* `errorOnTimingCacheMiss: Disabled` and hid the real failure.
> When a build fails silently, re-run it with the output unfiltered before theorising.

Note TensorRT 10 removed `--workspace`; use `--memPoolSize=workspace:4096`.

### Expected numbers, AGX Orin 64 GB, 512×640

| | GPU compute | with transfers | throughput |
|---|---|---|---|
| FP16 | 4.79 ms | 5.91 ms | 209 FPS |
| FP32 | 7.62 ms | 8.98 ms | 131 FPS |

FP16 is **1.59×**, not the 2× commonly quoted. An Orin NX should be roughly half this.

## 6. Measure what the robot will actually see

```bash
sudo python3 ~/bench_camera_orin.py ~/hydranet_fp16.engine --frames 200
```

`trtexec` measures GPU inference alone. On our board the honest number was very different:

| stage | ms | share |
|---|---|---|
| capture (MJPG decode) | 5.65 | 15% |
| preprocess | 6.64 | 18% |
| H2D | 0.58 | 2% |
| **inference** | **5.12** | **14%** |
| D2H | 3.37 | 9% |
| **postprocess** | **16.33** | **43%** |
| **total** | **37.77** | **26.5 FPS** |

**The GPU is 14% of the frame.** Post-processing is 43%, nearly all of it the sigmoid over
80 classes at 6,825 positions. Narrowing the detection head at export is the single largest
win available, and it is measured rather than argued.

> **Corrected 2026-08-17: "nearly all of it the sigmoid" was wrong, and this table is why
> nobody could tell.** `postprocess` is one bucket. Split on a GB10 it came apart into a
> detection decode *and* a host argmax over the segmentation logits, and the argmax was the
> larger of the two — larger than the engine itself. Narrowing the detection classes is
> real (a measured 2.7× on the decode) but it was not the single largest win; folding the
> argmax into the graph was.
>
> The numbers on that board do not transfer here — different silicon, TRT 10.16 against
> this rig's 10.3 — so **the table above stands as measured and is not restated**. What
> transfers is the method: re-run `bench_camera_orin.py` on this board with
> `--argmax-seg`, and split `postprocess` before concluding anything from its share again.
> See [DEPLOY.md](DEPLOY.md) §3.

Also measure the camera's own ceiling — ours is 30 FPS, so 26.5 was already near the limit
and the real headroom is in the GPU's 7× spare capacity, not in frame rate.

## 7. Live view

```bash
sudo systemd-run --unit=hydranet-live --collect \
  /usr/bin/python3 /home/<user>/live_view_orin.py /home/<user>/hydranet_fp16.engine \
  --port 8080 --score 0.30
```

Then open `http://<orin-ip>:8080/`.

> **Symptom:** launched with `nohup setsid ... &` under `sudo`, the process is gone the
> moment the SSH session ends, and there is no log to explain it.
>
> **Cause:** the session's cgroup is torn down with it. Use `systemd-run`, which puts the
> process in its own unit and gives you `journalctl -u hydranet-live -f`.

## 8. What to look at first

Point the camera at the floor, then at the ceiling.

On our board a ceiling came back **25% "go"**, with the terrain head calling it
`floor_hard`. That is not a bug: ADE20K is human-height photography and a robot never looks
up, so the viewpoint is absent from training. It took one glance to find and no metric would
have shown it.

That is what this stream is for. Curves tell you the loss is falling; only the picture tells
you the model is calling a ceiling a floor.

## 9. On a robot that is already running: subscribe, do not take the camera

`10.8.140.130` is not a bench rig. It runs nav2, motion, a RealSense node and a speech
stack, and its D435I is held by `realsense2_camera_node`. Opening `/dev/video*` there fails
with `Device or resource busy`, and the fix is not to free the device — taking the camera
stops the robot.

The node already publishes what is needed, including depth registered to the colour frame:

```
/camera/camera/color/image_raw                    bgr8,  1280x720 @ 15 Hz
/camera/camera/aligned_depth_to_color/image_raw   16UC1, millimetres
```

```bash
source /opt/ros/humble/setup.bash
python3 scripts/robot/probe_ros_realsense.py --weights <state_dict>.pt \
    --config configs/hydranet_indoor.yaml --frames 40 --save 6 --range 5.0
```

It subscribes with `qos_profile_sensor_data` (the node publishes best-effort; a default
reliable subscription receives nothing and looks like a dead topic), runs the checkpoint in
PyTorch, and gates walkable on metric depth.

### What it measured, and why the engine still matters

| stage | ms | share |
|---|---|---|
| wait for the next frame | 6.02 | 4% |
| preprocess (1280×720 letterbox, PIL/CPU) | 31.31 | 19% |
| **inference (PyTorch eager, CUDA)** | **122.05** | **74%** |
| postprocess | 6.14 | 4% |
| **total** | **165.51** | **6.0 FPS** |

The same network through TensorRT on `.124` runs in **5.12 ms**. PyTorch eager is ~24×
slower here, and it also logs `cudnnException: CUDNN_STATUS_NOT_SUPPORTED` and falls back to
a slower path. Use this script to *see* what the model does on a real camera; build an engine
before quoting any number to anyone.

### What the ROS camera path actually costs

The camera is not remote — it is a D435I on `/sys/bus/usb/devices/2-2`, USB 3 at 5 Gbps,
plugged into this board. What is remote-shaped is the *access*: `realsense2_camera_node`
owns the device and everything else reads copies over DDS (`rmw_cyclonedds_cpp`,
`ROS_LOCALHOST_ONLY=0`). Measured on the running robot:

| | |
|---|---|
| colour `image_raw` | 2.76 MB/msg, **41.6 MB/s** at 15 Hz |
| aligned depth | 1.84 MB/msg, **27.8 MB/s** |
| subscribers already present | `record`, `ros_interface` — ours was the third |
| colour frame age at the subscriber | mean **145 ms**, p95 227 ms, max 248 ms |
| depth frame age | mean 99 ms, p95 173 ms |

Three things follow, and they matter more for deployment than the model does.

**The wire cost is ~70 MB/s and every subscriber pays for a copy.** Adding an inference
node this way adds serialisation and loopback traffic to a board already running nav2 and
Cartographer. The ROS answer is composition: load the camera driver and the inference node
into one component container and get intra-process zero-copy. That is the difference between
a demo and something that can be left running.

**Nothing sees a fresh frame.** 145 ms of transport is already spent before inference
starts, and the model then adds 90–120 ms in PyTorch. At walking pace, a decision acts on
a world about 25 cm out of date — which is a planner constraint, not a perception one, but
it belongs in whatever latency budget the robot is designed against.

**A slow subscriber makes it worse, silently.** Running inference at ~11 FPS against a 15 Hz
publisher queues callbacks, and `spin_once` returns the *oldest* first: the displayed frame
was 535 ms stale while the loop itself took 90 ms. Draining the queue each iteration and
keeping the newest message brought it back to 150 ms, the transport floor. A lag like that
is invisible in the picture — it looks like a working live view — so it is worth asserting
on `now - header.stamp` rather than trusting the frame rate.

One more, on pairing: take "the latest colour and the latest depth" and the two are up to a
full frame apart (measured: mean 33 ms, max 67 ms). `scripts/robot/live_view_ros.py` keeps a short
depth history and picks the nearest stamp, which brings the skew to 0 ms because the driver
stamps aligned depth with its colour frame's time. Without that, the range mask describes
the world 67 ms beside the image it is drawn on.

### One number worth keeping

The script reports `go where depth is invalid` — pixels the model calls walkable and the
depth sensor returns nothing for. On a first run that was **12.7% of all `go` pixels**.
Glass, mirrors and specular floor reflections are exactly what produces both halves of that
condition at once, so it is a cheap standing check for the failure this deployment fears
most: a surface the camera reads as an open path and the depth sensor cannot see either.

---

## Checklist

```
[ ] address verified, key auth working
[ ] board model and L4T version recorded
[ ] nvidia-jetpack installed, trtexec runs
[ ] jetpack vs l4t-core versions recorded (mismatch is expected)
[ ] user in video group, camera opens, aspect ratio known
[ ] cuInit returns 0 as the user that will run inference  <- check before anything else
[ ] MAXN + jetson_clocks
[ ] ONNX copied (not an engine), engine built on the board
[ ] trtexec numbers recorded
[ ] end-to-end numbers recorded, and the camera's own FPS ceiling
[ ] live view up, checked against floor and ceiling
```

---

# Part III — local development on Apple Silicon

For local development and smoke testing; real training still belongs on a CUDA machine
(AMP is CUDA-only). The numbers below were measured on an M4 Pro / 48 GB unified memory /
PyTorch 2.13.

## 1. Setting up the environment

`uv` downloads a suitable Python itself, so there is no need for pyenv or conda first.

```bash
uv sync --group dev --extra export   # create .venv and install everything (per uv.lock)
```

- `uv sync` restores exactly the dependency versions in `uv.lock` and installs this project
  in editable mode.
- `--group dev` is a PEP 735 dependency group (pytest / ruff / pre-commit) and does not end
  up in the published wheel.

From then on every command has two forms; pick either:

```bash
uv run hydranet-train ...              # no need to activate the venv
# or
source .venv/bin/activate              # then use the command names directly
```

Verifying the environment:

```bash
uv run pytest -q                                              # 1,170 passed, 1+ skipped
uv run python -c "import torch; print(torch.backends.mps.is_available())"   # True
```

## 2. Device selection

`pick_device()` in `src/syncai_hydranet/utils/device.py` picks **CUDA → MPS → CPU** in that
order, and every CLI command shares it.
On a Mac it goes to MPS automatically, and the first line of the training log prints
`device=mps`.

To force a device:

```bash
uv run hydranet-train --config ... --set device=cpu
```

Two things switch themselves off on MPS, both decided in `device.py`, so the config needs no
manual editing:

| Item | State | Reason |
|---|---|---|
| AMP (`train.amp`) | Disabled automatically | `GradScaler` / `autocast` are only fully supported on CUDA |
| DataLoader `pin_memory` | Disabled automatically | MPS does not support pinned memory yet; setting True only produces warnings |

Losing AMP costs little on a Mac: 48 GB of unified memory is enough for batch 16 in FP32.

## 3. Measured performance

RegNetX-800MF + BiFPN×2, all three heads active, including backward and the AdamW step:

| Device | Resolution | batch | ms/step | img/s | Peak memory |
|---|---|---|---|---|---|
| CPU | 384×512 | 4 | 705 | 5.7 | — |
| MPS | 384×512 | 4 | 97 | 41.2 | — |
| MPS | 384×512 | 8 | 186 | 43.0 | 3.4 GB |
| MPS | 512×640 | 8 | 300 | 26.7 | 5.6 GB |
| MPS | 512×640 | 16 | 628 | 25.5 | 10.8 GB |

**MPS is about 7.3× faster than CPU**, so do check that the log prints `device=mps`.
Going from batch 8 to 16 gains almost no throughput (the GPU is already saturated) while
doubling memory, so locally it is worth stopping at 8.

## 4. Suggested local config overrides

```bash
uv run hydranet-train --config configs/hydranet_regnet800mf.yaml --set \
  "data.input_size=[384,512]" \
  "train.batch_size=8" \
  "data.workers=3"
```

### `data.workers` should come down

`MultiTaskLoader` builds **one DataLoader per dataset**, with `persistent_workers` on
whenever `workers > 0`. The default `workers: 8` with three datasets means **24 worker
processes resident at once**, which on a 14-core laptop over-subscribes the machine and ends
up slowing data delivery down. Locally, `3`–`4` is the better range.

### COCO's `sample_ratio` should come down

Steps per epoch is `Σ len(loader_i) × ratio_i`. COCO train2017 holds about 117,000 images,
so at `ratio: 1.0` it alone contributes ~14,600 steps: 80 minutes for one epoch, and 80
hours for 60 epochs.

Drop COCO's `sample_ratio` to `0.1` and each epoch takes a random 10%. Because
`shuffle=True` and the iterator is rebuilt every epoch, each epoch sees a *different* 10%,
so over a full run the whole dataset is still covered.

Note that `set_path` in `config.py` does **not support list indices** —
`--set data.datasets[2].sample_ratio=0.1` raises no error but silently creates a useless key
literally named `datasets[2]`. Two workable approaches:

```bash
# Approach A: copy the config (recommended, the settings persist)
cp configs/hydranet_regnet800mf.yaml configs/local_mac.yaml
# edit local_mac.yaml and change sample_ratio in the coco entry to 0.1

# Approach B: override the whole list at once (--set values go through yaml.safe_load)
--set "data.datasets=[{name: rugd, type: seg_folder, root: datasets/RUGD, \
  split_train: train, split_val: val, supervises: [traversability, terrain], sample_ratio: 1.0}, \
  {name: coco, type: coco, root: datasets/coco, split_train: train2017, \
  split_val: val2017, supervises: [detection], sample_ratio: 0.1}]"
```

After that one epoch takes about 14 minutes and 60 epochs about 14 hours — an overnight run.

### Segmentation-only is a legitimate setup

Trim `data.datasets` down to just RUGD and the detection head is still built, merely left
unsupervised.
RUGD + RELLIS together (about 13,000 images) run at roughly 8 minutes per epoch at 512×640.

## 5. EMA is now safe on short runs (historical issue)

Validation uses the EMA weights, and the EMA starts from the model's **initial random
weights**. The old fixed decay left `ema_decay^n` of that initialisation behind after n
steps, so short runs were dominated by random weights:

| Steps | Random init remaining | Validation score |
|---|---|---|
| 160 steps @ 0.995 (old) | 45% | traversability mIoU 0.16 |
| The same weights, non-EMA | 0% | traversability mIoU **0.95** |

The decay now ramps with the update count, `decay * (1 - exp(-updates / ema_warmup_steps))`
(the standard YOLOv5 / timm approach): the first few steps copy the model almost exactly,
and the smoothing only strengthens as the average accumulates history.
An 18-step smoke run now scores the same with EMA on or off, so **there is no longer any
need to disable it for short runs**.

`ema_warmup_steps` defaults to 2000. If a run really is too short to finish even the ramp,
`Trainer` still prints a warning with the residual fraction; that is the point to consider:

```bash
--set train.ema=false
```

## 6. Checking whether training is data-starved

The log prints img/s every `log_interval` steps. Compare it against the table above:

- Close to the table → the GPU is the bottleneck, which is normal.
- Clearly below the table → data delivery is not keeping up. RELLIS-3D is 1920×1200 JPEG,
  and decoding it is CPU-heavy. Try raising `data.workers` by 1–2 first; if that does not
  help, downsample the dataset offline.

TensorBoard:

```bash
uv run tensorboard --logdir runs/
```

## 7. ONNX export

`hydranet-export-onnx` does not go through device selection — it exports on CPU, so a Mac
can run it directly:

```bash
uv run hydranet-export-onnx --config configs/hydranet_regnet800mf.yaml \
    --checkpoint runs/.../best.pt --output hydranet.onnx
```

Converting to a TensorRT engine with `trtexec` afterwards has to happen on the Jetson; see
[DEPLOY.md](DEPLOY.md).
