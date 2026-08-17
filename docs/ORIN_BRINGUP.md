# Bringing up a new Jetson Orin

From a freshly flashed board to a live prediction stream. Every step here was walked on a
real AGX Orin, and the failures listed are ones that actually happened rather than ones
that might — each is written with the symptom first, so a future reader can match what they
are seeing rather than read the whole page.

Budget about an hour, most of it the JetPack download.

For what happens after the board works — the ONNX contract, post-processing, INT8 — see
[DEPLOY_JETSON.md](DEPLOY_JETSON.md).

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
> See [DEPLOY_JETSON.md](DEPLOY_JETSON.md) §3.

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
python3 scripts/probe_ros_realsense.py --weights <state_dict>.pt \
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
full frame apart (measured: mean 33 ms, max 67 ms). `scripts/live_view_ros.py` keeps a short
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
