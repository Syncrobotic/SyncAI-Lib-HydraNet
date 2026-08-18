# Session handoff: the Lite3 robot line — bring-up, deployment, control, research direction

> **Journal entry, 2026-08-18.** A record of one session, not maintained afterwards. The
> durable material is in the `docs/` files and the memory notes this points to; where they
> disagree, believe those. This session owned **product A, the quadruped** end to end —
> connecting the robot, putting HydraNet on its NPU, building the dashboard and teleop, and
> setting the occupancy research direction. A second session ran repo tidy, annotation
> measurement and the retail line in parallel on the same `dev` checkout, and coordinated
> by cross-session message throughout; its commits are interleaved with these.

## Where the repo is

`dev`, several commits ahead of `origin/dev`, **unpushed on purpose** — the user pushes
themselves ("我自己 push"). This session's commits, newest last:

- `deploy/robot/` — the quadruped deployment surface (conversion, on-robot service,
  dashboard, forwarder, systemd, README).
- annotation move — `deploy/annotation` → `tools/annotation` (data production is not a
  deploy target), plus `deploy/README.md`, `tools/README.md`, `deploy/retail-security/README.md`.
- `0x0901` telemetry wired — battery / ultrasound / odometry, official e-stop, per-frame
  BEV plane.
- teleop control interface — the full documented command set.
- this doc + `docs/RESEARCH_OCCUPANCY.md`.

The peer session's commits (flat-BEV removal, `IGNORE` consolidation, `geometry/__init__`
divergence note, scripts type ratchet) are also on `dev`. Staging was always surgical —
only this session's paths — so the peer's uncommitted WIP (`render_demo.py`, `b03` configs)
was never swept in. Keep doing that.

## The live system, and how to reach it

Nothing here is in the repo's runtime; it runs on two machines as systemd services.

| where | service | what |
|---|---|---|
| **Lite3** `192.168.1.120` | `hydra-dash` (:8080) | dashboard: telemetry, camera, live inference, teleop |
| **Lite3** | `hydra-infer` | HydraNet on the NPU → traversability/detection overlay + bev3d BEV |
| **pro6000** | `hydra-forward` | exposes robot 8080 on pro6000 localhost for VS Code |

Reach it: **`ssh lite3`** (= `user@192.168.1.120`, pw `123456`; sudo `123456`). A sudoer
**`paul` / `Admin123`** was created and owns everything under `/home/paul/playground/`.
From a laptop on VS Code Remote-SSH into pro6000, forward port **8080** → `http://localhost:8080`
(camera + inference are proxied through 8080, so one port is enough).

Full access, network, and gotchas are in memory: [[lite3-wired-access]], [[syncai-wifi-ap]],
[[lite3-playground-dashboard]], [[lite3-rknn-deployment]].

## What got built, and the one document that rewrote half of it

The DEEP Robotics comm spec — `assets/838272320-绝影Lite3运动主机通讯接口-beta-V1-0-7.pdf` —
arrived late and corrected three things this session had reverse-engineered wrong. **Read
it before touching control or telemetry.**

- **Telemetry.** Two streams reach UDP `43897` (bind it to receive — `jy_exe` sends there
  by default): `0x0906` (MotionSDK RobotData, IMU + 12 joints, ~1 kHz) and **`0x0901`
  (RobotStateUpload, 50 Hz)** which carries `battery_level` (was wrongly shown N/A),
  `ultrasound[2]` (the robot is *not* rangeless — two metric returns), and
  `pos_world`/`vel_world`/`vel_body` (legged odometry, off the shelf). Struct offsets are
  in `hydra_dash.py`'s `telem_listener`. Only ONE process should bind `43897` — a second
  `SO_REUSEPORT` reader splits the stream and both miss packets.
- **Control.** Commands are `struct{uint32 code, value, type}` little-endian to `43893`.
  E-stop is **`0x21020C0E`** (the `0x21010C0E` this session first captured off the
  controller is not the spec's soft e-stop). Movement is *axis commands* streamed ≥20 Hz
  (`0x21010130` fwd / `0x21010131` lateral / `0x21010135` yaw), int in `±32767`, and the
  robot **auto-stops after 250 ms** with no command — which the teleop uses as its deadman.
  Autonomous mode `0x21010C03` makes `jy_exe` follow a perception host's velocity, so
  velocity-level autonomy needs no policy replacement. Full code table is in the PDF and
  mirrored in `hydra_dash.py`'s `SIMPLE_CMDS`.
- **RKNN deployment.** `hydranet_joint_coco10` → ONNX → RKNN `1.6.0` (must match the
  robot's `librknnrt`), int8 384×512, ~7.5 fps NPU. Full chain, including the geometry
  package copied to the robot with its Python-3.8/Pillow shims, is in
  [`deploy/robot/README.md`](../../deploy/robot/README.md) and [[lite3-rknn-deployment]].

## Open threads, in priority order

1. **Teleop movement is untested on the real robot.** Plumbing verified without walking it
   (arm/move send nothing while disarmed). The next session, or the user, should arm and
   drive with the robot in a clear area and hardware e-stop in hand, and confirm the axis
   scaling (`FWD_MAX` etc. are deliberately conservative).
2. **The occupancy research direction** — [`docs/RESEARCH_OCCUPANCY.md`](../RESEARCH_OCCUPANCY.md).
   The user has committed to it. First actionable step is **E-prep**: run a monocular-depth
   teacher over robot video and measure its residual against ultrasound/odometry — the
   decision gate for whether a depth sensor is needed. Touches no hardware.
3. **Autonomous patrol** was requested and is teach-and-repeat shaped: the teleop is the
   teach tool; HydraNet traversability + the new ultrasound/odometry give repeat + avoidance;
   a patrolling robot is a mobile product B. Not yet written up.
4. **Two decisions the user has not made:** (Fork 2) whether to keep `jy_exe` or run an own
   RL policy — gates head ⑤; and whether to add a depth sensor — E-prep informs it.

## Gotchas the next session will otherwise hit

- **Shared checkout, live peer.** `git add` only your own paths, never `-A`. Coordinate on
  anything structural (this session cleared every structural move with the peer first).
- **`deploy/robot/hydra_dash.py` and `hydra_infer.py` are committed byte-identical to what
  runs on the robot.** They carry an ops-script exception in `deploy/robot/ruff.toml`
  (embedded HTML + Traditional-Chinese UI + os-based I/O on a 3.8 target). If you edit
  them, redeploy and smoke-test before committing, or the repo copy drifts from the running
  one — the thing that surface exists to prevent.
- **The robot's default route is bogus** (points at eth1's own `192.168.137.120`); it only
  reaches the internet after `ip route ... via 10.42.0.1 dev wlan0`. It has no `pip`/`numpy`
  stock; wheels went over by `scp`. See [[lite3-rknn-deployment]].
- **Reverse-engineered offsets are validated but not authoritative — the PDF is.** Anything
  about codes or telemetry: check the spec, not this session's early captures.
