# Session handoff

Written 2026-08-13 as one working session's context filled up. This is state and open
threads, not a design document — the durable knowledge is in the other `docs/` files, which
this points to rather than repeats.

## Where the repo is

`origin/dev` is at `ed44b6e`, clean and synced. All work this session is committed and
pushed. `main` and `stage` still sit at the post-rewrite root (`495227b9`); nothing has been
promoted, deliberately — the acceptance gates in [RELEASE.md](RELEASE.md) are all unmet.

**A second session (`syncai-lib-hydranet-5d`) is running.** Coordinate before touching
`src/`, `configs/`, or `pyproject.toml`. It has been doing the training runs and the ADE20K
data investigation; this session did the deploy/Orin/docs side. We stayed out of each
other's files by announcing scope over SendMessage each time.

## The one true fact about the model

**Nothing is shippable yet, and the blocker is data, not code or infrastructure.** The model
design is sound (measured — see [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md)); what is
missing is training data at the robot's viewpoint and for three safety classes. A team that
staffs tuning over annotation gets a better-tuned model with the same ceiling.

Concrete evidence gathered this session:
- `caution` never learned in the multi-task run — it sat at 0.00–0.02 the whole time; the
  0.09 that `best.pt` caught is a noise spike. Root cause: 3 of its 4 constituent classes
  have zero data, and COCO starves the segmentation heads of optimiser steps.
- The live camera view showed a **ceiling coming back 25% "go"** within a minute — ADE20K is
  human-height photography, so the robot's viewpoint is absent from training.

## Open threads, most time-sensitive first

1. **60-epoch control run (peer session, ~55/60 when last seen).** This is the clean test of
   whether multi-task *interferes* or just *dilutes*: it aligns the segmentation step count
   that the earlier 30-epoch run confounded. When it finishes, update HANDOVER.md section 4 —
   it will be the first baseline row with a real `detection_mAP`. Last seen: best E11, trav
   0.6300, mAP 0.117.
2. **ADE20K enrichment: dead end, confirmed by peer.** The full-vocabulary ADE20K does not
   rescue the missing classes — 98% of its `step`/`ramp` pixels are already `stairs`, real
   `ramp` is 5 images, `floor_metal` is 1, `wet_slippery` is 0. All three remain
   annotate-only. A positive side finding: existing `stairs` labels are good quality (98%
   agreement), so its low IoU is a data-*volume* problem — add footage, don't relabel.
3. **CUDA-needs-root on Orin `10.8.140.124`.** `cuInit` returns 801 for non-root there;
   root works. Tried video group, chgrp on scheduler nodes + udev, debug group, reboot —
   none worked. Suspected cause: `nvidia-jetpack 6.2.1` on an `nvidia-l4t 36.4.0` base.
   **The two Unitree Orins do not have this problem** (`cuInit` → 0 as normal user), which
   supports the version-mismatch theory. Documented in [ORIN_BRINGUP.md](ORIN_BRINGUP.md) §4.
4. **`pre-conventional-commits` tag deleted** at the user's request, local and remote. The
   old history is unreachable but byte-identical to `dev`; recovery is by commit *message*,
   see [CONTRIBUTING.md](../CONTRIBUTING.md). Do not recreate the tag.

## Machines

| Host | What | Access | Notes |
|---|---|---|---|
| `10.8.140.124` | AGX Orin, JetPack 6.1, TRT 10.3 | `paul`, **SSH key installed** | The bench rig. Has USB camera `/dev/video0`. CUDA needs root here. Streaming service was stopped on request. Has `hydranet.onnx` + both `.engine` + the three scripts in `~`. |
| `10.8.140.116` | AGX Orin `syncai-robot`, TRT **8.6** | `unitree`/`123` (password) | Unitree compute unit. CUDA works unprivileged. No local camera. |
| `10.8.140.127` | AGX Orin `ubuntu`, TRT 8.6 | `unitree`/`123` | Older sibling of .116. |
| `10.8.100.178` | DDS peer in `.116`'s cyclonedds.xml — the camera source | **offline** | The robot body / camera publisher. Unreachable from every vantage tried. |

**The Unitree camera is not on the Jetsons.** It is published over DDS by the robot-side
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
  **Admin account not yet created** — see [ANNOTATION_SETUP.md](ANNOTATION_SETUP.md). Stop it
  when idle: `gcloud compute instances stop hydranet-annotation --zone=asia-east1-b`.

## What I'd do next

1. When the 60-epoch run lands, do the multi-task interference write-up (thread 1).
2. Start annotation — it is the critical path and everything else is ahead of it. Priority
   order is in [METHODOLOGY.md](METHODOLOGY.md): glass, wet_slippery, floor_metal first
   (LiDAR-blind, hand-label only), threshold_ramp/stairs later.
3. The measurement tooling added this session (`head_disagreement`, `mIoU_classes`, the
   export parity gate, the release bundler) is in place — use it, don't rebuild it.

## Docs map

`README.md` links all of these. Newest first: [ORIN_BRINGUP.md](ORIN_BRINGUP.md),
[RELEASE.md](RELEASE.md), [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md),
[ANNOTATION_SETUP.md](ANNOTATION_SETUP.md), [METHODOLOGY.md](METHODOLOGY.md),
[TRAINING_GUIDE.md](TRAINING_GUIDE.md), [HANDOVER.md](HANDOVER.md) (MPS→CUDA move),
[DEPLOY_JETSON.md](DEPLOY_JETSON.md), [TRAIN_MACOS.md](TRAIN_MACOS.md).
