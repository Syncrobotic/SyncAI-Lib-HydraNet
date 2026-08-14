# Session handoff

> **Journal entry, 2026-08-13.** A record of one day, not maintained since. Facts here
> about what was running, what was unfinished and who was doing what were true then and
> are the first thing to go stale. The durable material is in `docs/`.

Written 2026-08-13 as one working session's context filled up. This is state and open
threads, not a design document — the durable knowledge is in the other `docs/` files, which
this points to rather than repeats.

## Where the repo is

`origin/dev` is at `ed44b6e`, clean and synced. All work this session is committed and
pushed. `main` and `stage` still sit at the post-rewrite root (`495227b9`); nothing has been
promoted, deliberately — the acceptance gates in [RELEASE.md](../RELEASE.md) are all unmet.

**A second session (`syncai-lib-hydranet-5d`) is running.** Coordinate before touching
`src/`, `configs/`, or `pyproject.toml`. It has been doing the training runs and the ADE20K
data investigation; this session did the deploy/Orin/docs side. We stayed out of each
other's files by announcing scope over SendMessage each time.

## The one true fact about the model

**Nothing is shippable yet, and the blocker is data, not code or infrastructure.** The model
design is sound (measured — see [ARCHITECTURE_REVIEW.md](../ARCHITECTURE_REVIEW.md)); what is
missing is training data at the robot's viewpoint and for three safety classes. A team that
staffs tuning over annotation gets a better-tuned model with the same ceiling.

Concrete evidence gathered this session:
- The live camera view showed a **ceiling coming back 25% "go"** within a minute — ADE20K is
  human-height photography, so the robot's viewpoint is absent from training.
- Three of the four terrain classes that map to `caution` have **zero** training pixels, so
  `caution` is effectively `stairs` wearing another name.
- **`caution` and `stairs` are not measurable yet, at any number.** The peer session showed
  the same `best.pt` scoring `caution` 0.1241 on val and 0.3252 on test, and `stairs` 0.1361
  against 0.3249 — a 2.5x swing from changing which images are scored. Only 24 val images
  contain stairs. Any claim about those two classes, in either direction, is reading noise;
  an earlier draft of this file made exactly that mistake. The trustworthy numbers are the
  common classes (blocked / go / wall / floor) and `detection_mAP`, which is computed over
  5,000 COCO val images. RUN D (a seed-7 rerun of the baseline, due ~07:10) will give the
  first honest scale for how much two identical runs differ.

## Open threads, most time-sensitive first

1. **60-epoch control run (peer session, finished overnight), plus two more behind it.**
   The 60-epoch run is the clean test of whether multi-task *interferes* or just *dilutes*:
   it aligns the segmentation step count (124 × 60 = 7,440, matching the pure-segmentation
   baseline) that the earlier 30-epoch run confounded. RUN C (COCO ratio 1.0, 60 epochs,
   `primary_metric=detection_mAP`) and RUN D (baseline settings, seed 7 — the training-noise
   floor) were queued behind it under `setsid`, due ~06:30 and ~07:10. **The peer session
   owns this thread and the write-up.** Read RUN D before comparing anything about rare
   classes: without it there is no scale to compare against.
2. **ADE20K enrichment: dead end, confirmed by peer.** The full-vocabulary ADE20K does not
   rescue the missing classes — 98% of its `step`/`ramp` pixels are already `stairs`, real
   `ramp` is 5 images, `floor_metal` is 1, `wet_slippery` is 0. All three remain
   annotate-only. A positive side finding: existing `stairs` labels are good quality (98%
   agreement), so its low IoU is a data-*volume* problem — add footage, don't relabel.
3. **CUDA-needs-root on Orin `10.8.140.124`.** `cuInit` returns 801 for non-root there;
   root works. Tried video group, chgrp on scheduler nodes + udev, debug group, reboot —
   none worked. Suspected cause: `nvidia-jetpack 6.2.1` on an `nvidia-l4t 36.4.0` base.
   **The two Unitree Orins do not have this problem** (`cuInit` → 0 as normal user), which
   supports the version-mismatch theory. Documented in [ORIN_BRINGUP.md](../ORIN_BRINGUP.md) §4.
4. **`pre-conventional-commits` tag deleted** at the user's request, local and remote. The
   old history is unreachable but byte-identical to `dev`; recovery is by commit *message*,
   see [CONTRIBUTING.md](../../CONTRIBUTING.md). Do not recreate the tag.

## Machines

| Host | What | Access | Notes |
|---|---|---|---|
| `10.8.140.130` | AGX Orin Dev Kit 64 GB, L4T **R36.4.7**, TRT **10.3.0.30**, CUDA 12.2/12.6 | `unitree`, password — see the team password manager | **The best target we have.** CUDA works unprivileged (`cuInit` → 0), an Intel **RealSense D435I** is attached (`/dev/video2,4,6`), torch 2.3.0 + CUDA, cv2 4.10, MAXN. Also on the Unitree robot LAN (`192.168.123.101`). Caveats: **no `trtexec` binary** (TRT libs only — build engines via the Python API or install it), `libnvonnxparsers8 8.6.2.3` is co-installed beside 10.3, and the disk is **88% full (27 GB free)**. |
| `10.8.140.124` | AGX Orin, JetPack 6.1, TRT 10.3 | `paul`, **SSH key installed** | The bench rig. Has USB camera `/dev/video0`. CUDA needs root here. Streaming service was stopped on request. Has `hydranet.onnx` + both `.engine` + the three scripts in `~`. |
| `10.8.140.116` | AGX Orin `syncai-robot`, TRT **8.6** | `unitree`, password — password manager | Unitree compute unit. CUDA works unprivileged. No local camera. |
| `10.8.140.127` | AGX Orin `ubuntu`, TRT 8.6 | `unitree`, password — password manager | Older sibling of .116. |
| `10.8.100.178` | DDS peer in `.116`'s cyclonedds.xml — the camera source | **offline** | The robot body / camera publisher. Unreachable from every vantage tried. |

> Credentials do not live in this repository. The three Unitree boards ship with the
> vendor's default password and it is still set; that is worth changing on its own merits,
> and [ORIN_BRINGUP.md](../ORIN_BRINGUP.md) §1 says how to move to key auth. Until then the
> password belongs in the team password manager. It was written here in plain text until
> `dev`; the git history still holds it, which is one more reason to rotate rather than to
> rewrite history — the rewrite is forbidden by [RELEASE.md](../RELEASE.md) and would not help.

**A camera does exist after all — on `.130`.** The RealSense D435I there is a local USB
device, so it needs none of the DDS/robot-network path below. It also returns metric depth,
which retires the homography question in [RETAIL_SCOPE.md](../RETAIL_SCOPE.md) §2 for any
robot built on this configuration: the 5 m mask is a depth threshold, not a calibration.

**The Unitree camera is not on `.116`/`.127`.** It is published over DDS by the robot-side
network (behind gateway `10.8.141.254`: `10.8.100.178`, possibly `10.8.180.110`), which is
currently powered off — every address there is unreachable. No `/dev/video`, no CSI (nvargus
says "No cameras available"), no RealSense, and `ros2 topic list` shows only default topics
under both the default domain and Unitree's cyclonedds config. To get a camera: power on the
robot side, then from `.116` source `~/unitree_ros2/.../setup.bash` with the cyclonedds URI
and re-list topics. Engines for these boxes must be rebuilt on TRT 8.6 (ONNX is portable,
engines are not; `--workspace` still exists in 8.6, no `--memPoolSize`).

The `paul@10.8.140.124` key auth uses the ssh-agent at
`/tmp/ssh-XXXXXXg4s2yP/agent.15090` (set `SSH_AUTH_SOCK`); it will not survive a reboot of
this workstation.

## GCP (project `syncrobotic-aisw`)

- `gs://syncai-hydranet` (asia-east1, versioning + lifecycle) — has `runs/` and `assets/`.
  **Datasets deliberately not uploaded** (8 GB, public, and ADE20K is symlinks).
- BigQuery `hydranet_metrics.runs` — 60-epoch run loaded, SQL-queryable.
- Service account `syncai-hydranet@…`, key at `~/.gcp/syncai-hydranet-sa.json` (mode 600,
  gitignored). Bucket-scoped only; verified it cannot touch other buckets.
- CVAT annotation host `hydranet-annotation` (asia-east1-b, no public IP, IAP tunnel).
  **Currently RUNNING and idle** (~$59/month against ~$10 stopped). It still runs the
  hand-made bring-up — CVAT `develop@7064c1b` on the floating `:dev` tag — and holds no
  annotations, so moving it onto the pinned stack in [`deploy/annotation/`](../../deploy/annotation/)
  is free right now and needs a backup later. The admin account still does not exist;
  `./cvat.sh admin <user>` creates it. Stop it when idle:
  `gcloud compute instances stop hydranet-annotation --zone=asia-east1-b`.

## What I'd do next

1. **Start annotating.** It is the critical path, everything else is ahead of it, and the
   environment is now ready rather than nearly ready:
   ```bash
   cd deploy/annotation && ./cvat.sh up && ./cvat.sh admin <you>   # the account that never existed
   hydranet-annotation labels --out cvat_labels.json               # paste into CVAT
   hydranet-annotation check datasets/<site>                       # before the first run on it
   ```
   Priority order is in [METHODOLOGY.md](../METHODOLOGY.md): glass, wet_slippery, floor_metal
   first (LiDAR-blind, hand-label only), threshold_ramp/stairs later. All three
   zero-example classes are confirmed annotate-only — the full-vocabulary ADE20K was
   checked and does not supply them.
2. Read the overnight runs (thread 1), starting with RUN D's noise floor.
3. The tooling is in place — use it, don't rebuild it: `head_disagreement`, `mIoU_classes`,
   the export parity gate, the release bundler, `hydranet-annotation check`, `cvat.sh`.
4. **`void` is a trained class.** `INDOOR_NATIVE_ID` maps 0 → 0 while the losses ignore only
   255, so an annotation export that leaves the background at 0 teaches the model to predict
   `void` over every unlabelled pixel. `hydranet-annotation check` fails on it; the durable
   fix is a one-line change to `label_maps_indoor.py` (0 → 255), which is the peer session's
   file and their call.

## Docs map

`README.md` links all of these. Newest first: [ORIN_BRINGUP.md](../ORIN_BRINGUP.md),
[RELEASE.md](../RELEASE.md), [ARCHITECTURE_REVIEW.md](../ARCHITECTURE_REVIEW.md),
[ANNOTATION_SETUP.md](../ANNOTATION_SETUP.md), [METHODOLOGY.md](../METHODOLOGY.md),
[TRAINING_GUIDE.md](../TRAINING_GUIDE.md), [the CUDA move](2026-08-12-mps-to-cuda.md),
[DEPLOY_JETSON.md](../DEPLOY_JETSON.md), [TRAIN_MACOS.md](../TRAIN_MACOS.md).
