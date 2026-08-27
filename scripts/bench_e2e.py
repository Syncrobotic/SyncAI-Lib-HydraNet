#!/usr/bin/env python3
"""End to end for gate 3: decode -> engine -> host NMS -> tracker, against 1,440 fps.

`bench_trt.py` measures the engine and says so. This measures the legs around it,
because the gate is end to end and decode, NMS and PCIe were named as the real risk
when 96 streams x 15 fps became a binding requirement rather than headroom.

Four legs, each reported separately and then composed, because a single number that
fails tells you nothing about which leg to fix:

  decode   N concurrent h.264 streams at the target cadence
  engine   the serialized plan (delegated to bench_trt, same method, no second opinion)
  post     `serving.decode.FcosDecoder` -- host-side FCOS decode plus batched NMS
  track    `analytics.tracker.Tracker.update` at a realistic box count per frame

**Refuses to run while anything else is on the GPU.** Both existing bench scripts ask
for an idle card in a comment; a comment did not stop a throughput number being taken
against a training job. `--allow-busy-gpu` exists for deliberate exploratory runs and
stamps every result it produces as contaminated.

Decode backends are resolved and *named* rather than assumed, so `--decode nvdec` fails
with the list of what is missing rather than silently measuring a CPU pipe and calling it
NVDEC. That resolution is done at runtime by `resolve_backends` below, which is the only
statement about this box worth trusting -- when this file was written it said the box had
no NVDEC path at all (no PyNvVideoCodec, no DALI, no PyAV, and the only ffmpeg on PATH
offering `vdpau` alone), and that stopped being true on 2026-08-25: PLAN 7 decision 8
installed PyNvVideoCodec, decode went from 63-69 streams to ~520 and stopped being the
binding leg. It is a declared dependency as of `fc902e5`, so `uv sync --extra bench` has
it; the system ffmpeg still has no cuvid build, which is why the probe reports the two
separately.

  python3 scripts/bench_e2e.py --plan exports/pro6000/xl_b16.fp16.plan \\
      --clip datasets/studioa_clips/Taichung-cam10/archive_*.mp4 --streams 16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TARGET_FPS = 1440.0  # 96 streams x 15 fps
STREAM_FPS = 15.0


# --------------------------------------------------------------------------- guards


def gpu_occupants() -> list[dict]:
    """Every compute process on the card, ours included. Empty list means idle."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            rows.append({"pid": int(parts[0]), "memory": parts[1], "process": parts[2]})
    return rows


def decode_backends() -> dict[str, str | None]:
    """backend -> None if usable, else the reason it is not. Reasons, not booleans."""
    status: dict[str, str | None] = {}

    status["ffmpeg-cpu"] = None if shutil.which("ffmpeg") else "no ffmpeg on PATH"

    ff = shutil.which("ffmpeg")
    if not ff:
        status["ffmpeg-nvdec"] = "no ffmpeg on PATH"
    else:
        dec = subprocess.run([ff, "-hide_banner", "-decoders"], capture_output=True, text=True)
        hw = subprocess.run([ff, "-hide_banner", "-hwaccels"], capture_output=True, text=True)
        if "cuvid" in dec.stdout or "nvdec" in dec.stdout:
            status["ffmpeg-nvdec"] = None
        else:
            have = " ".join(hw.stdout.split()[3:]) or "none"
            status["ffmpeg-nvdec"] = f"{ff} has no cuvid/nvdec decoder (hwaccels: {have})"

    try:
        import PyNvVideoCodec  # noqa: F401

        status["pynvvideocodec"] = None
    except ImportError:
        status["pynvvideocodec"] = "PyNvVideoCodec not installed"
    return status


# ------------------------------------------------------------------------- the legs


def bench_decode(clip: Path, streams: int, seconds: float, backend: str) -> dict:
    """Sustained aggregate decode rate with `streams` clips open at once.

    Each worker decodes as fast as it can rather than being paced at 15 fps: the
    question is the ceiling, and a paced worker measures the pacing.
    """
    counts = [0] * streams
    stop = threading.Event()

    # Every worker REOPENS the clip when it runs out. Without this a fast backend
    # finishes the 2,100-frame clip inside the window and the rate becomes
    # clip_length / seconds -- which is what this measured on its first run: 1, 8 and 24
    # streams returned exactly 1x, 8x and 24x the clip length, a straight line through
    # numbers that describe the clip and not the decoder.
    if backend == "ffmpeg-cpu":
        from syncai_hydranet.data.video import frames as decode_frames

        def worker(i: int) -> None:
            # rgb24 over a pipe: 1920x1080x3 per frame, which is the cost this backend
            # cannot get out of and the reason the NVDEC one exists
            while not stop.is_set():
                for _ in decode_frames(str(clip), 1920, 1080, None):
                    counts[i] += 1
                    if stop.is_set():
                        break

    elif backend == "pynvvideocodec":
        import PyNvVideoCodec

        def worker(i: int) -> None:
            # device memory on purpose: the frame never crosses PCIe, which is the whole
            # argument for NVDEC here. The engine reads it where the decoder left it.
            while not stop.is_set():
                dmx = PyNvVideoCodec.CreateDemuxer(str(clip))
                dec = PyNvVideoCodec.CreateDecoder(
                    gpuid=0, codec=dmx.GetNvCodecId(), usedevicememory=True
                )
                for pkt in dmx:
                    for _ in dec.Decode(pkt):
                        counts[i] += 1
                    if stop.is_set():
                        break

    else:
        raise SystemExit(f"decode backend {backend!r} is resolved but not implemented here")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(streams)]
    t0 = time.time()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=10)
    elapsed = time.time() - t0
    total = sum(counts)
    return {
        "backend": backend,
        "streams": streams,
        "seconds": round(elapsed, 2),
        "frames": total,
        "frames_per_s": round(total / elapsed, 1),
        "per_stream_fps": round(total / elapsed / streams, 2),
        "streams_at_15fps": round(total / elapsed / STREAM_FPS, 1),
    }


def bench_post(
    batch: int, seconds: float, input_size: tuple[int, int], classes: int, candidates: int
) -> dict:
    """Host FCOS decode + batched NMS, per frame.

    `candidates` -- how many locations clear the score floor before NMS -- is the
    parameter that decides this leg's cost, which is why it is an argument and not a
    property of random noise. The pilot measured ~30 ms/frame before `pre_nms_topk`
    existed; that is the number this leg exists to keep honest.
    """
    from syncai_hydranet.serving.decode import FcosDecoder

    strides = (8, 16, 32, 64, 128)
    h, w = input_size
    shapes = [(h // s, w // s) for s in strides]
    rng = np.random.default_rng(0)
    cls_levels, reg_levels, ctr_levels = [], [], []
    total_locs = sum(hh * ww for hh, ww in shapes)
    hot = rng.choice(total_locs, size=min(candidates, total_locs), replace=False)
    hot_set = np.zeros(total_locs, bool)
    hot_set[hot] = True
    off = 0
    for hh, ww in shapes:
        n = hh * ww
        cls = np.full((batch, classes, hh, ww), -6.0, np.float32)
        flat_hot = hot_set[off : off + n].reshape(hh, ww)
        cls[:, 0][:, flat_hot] = 4.0  # comfortably above any sane floor
        ctr = np.full((batch, 1, hh, ww), 3.0, np.float32)
        reg = rng.uniform(4, 40, size=(batch, 4, hh, ww)).astype(np.float32)
        cls_levels.append(cls)
        reg_levels.append(reg)
        ctr_levels.append(ctr)
        off += n

    dec = FcosDecoder(num_classes=classes, strides=strides, pre_nms_topk=1000)
    for _ in range(3):
        dec(cls_levels, reg_levels, ctr_levels, 0.35, img_size=input_size)
    n = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        dec(cls_levels, reg_levels, ctr_levels, 0.35, img_size=input_size)
        n += 1
    elapsed = time.time() - t0
    return {
        "batch": batch,
        "candidates_pre_nms": int(candidates),
        "ms_per_frame": round(elapsed / (n * batch) * 1000, 3),
        "frames_per_s": round(n * batch / elapsed, 1),
    }


def bench_track(seconds: float, boxes_per_frame: int) -> dict:
    """`Tracker.update` per frame, one tracker per stream in serving, so per-frame cost."""
    from syncai_hydranet.analytics.tracker import Tracker

    rng = np.random.default_rng(0)
    tr = Tracker()
    base = rng.uniform(0, 1500, size=(boxes_per_frame, 2))
    n = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        jitter = rng.normal(0, 3, size=(boxes_per_frame, 2))
        xy = base + jitter
        b = np.concatenate([xy, xy + np.array([60.0, 160.0])], axis=1).astype(np.float32)
        tr.update(b, n)
        n += 1
    elapsed = time.time() - t0
    return {
        "boxes_per_frame": boxes_per_frame,
        "ms_per_frame": round(elapsed / n * 1000, 4),
        "frames_per_s": round(n / elapsed, 1),
    }


# ------------------------------------------------------------------------------ cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--plan", type=Path, help="serialized TRT engine for the engine leg")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--clip", type=Path, help="an h.264 clip for the decode leg")
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--decode", default="ffmpeg-cpu")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--input-size", type=int, nargs=2, default=(640, 1120))
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=1200)
    ap.add_argument("--boxes-per-frame", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/bench_e2e")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument(
        "--legs", default="decode,engine,post,track", help="comma-separated subset to run"
    )
    args = ap.parse_args(argv)

    backends = decode_backends()
    print("decode backends:")
    for name, why in backends.items():
        print(f"  {name:16s} {'available' if why is None else 'UNAVAILABLE -- ' + why}")
    if backends.get(args.decode) is not None:
        raise SystemExit(
            f"\n--decode {args.decode}: {backends.get(args.decode)}\n"
            "Gate 3 names NVDEC decode explicitly. Measuring a CPU pipe and reporting it "
            "as the gate's decode leg would answer a question nobody asked."
        )

    legs = [x.strip() for x in args.legs.split(",") if x.strip()]
    busy = gpu_occupants()
    contaminated = False
    gpu_legs = {"engine"} | ({"decode"} if args.decode != "ffmpeg-cpu" else set())
    if busy and gpu_legs & set(legs):
        if not args.allow_busy_gpu:
            lines = "\n".join(f"  pid {b['pid']}  {b['memory']}  {b['process']}" for b in busy)
            raise SystemExit(
                f"GPU is not idle -- {len(busy)} process(es) on the card:\n{lines}\n"
                "A throughput number taken while anything shares the card is not a "
                "measurement. Wait, or pass --allow-busy-gpu to stamp the result as "
                "contaminated."
            )
        contaminated = True

    results: dict[str, object] = {
        "target_frames_per_s": TARGET_FPS,
        "contaminated_by_gpu_neighbours": contaminated,
        "gpu_occupants": busy,
        "decode_backends": {k: (v or "available") for k, v in backends.items()},
    }

    if "decode" in legs:
        if not args.clip:
            raise SystemExit("--clip is required for the decode leg")
        print("\ndecode ...", flush=True)
        results["decode"] = bench_decode(args.clip, args.streams, args.seconds, args.decode)
        print(" ", results["decode"])

    if "engine" in legs:
        if not args.plan:
            raise SystemExit("--plan is required for the engine leg")
        print("\nengine ...", flush=True)
        # `serving.engine.bench_sync` is the same method, already in the package and
        # already the baseline every recovery in `serving/` is measured against.
        # Importing scripts/bench_trt for it was a script-to-script dependency and
        # `tests/test_scripts_are_not_libraries.py` ratchets on exactly that -- it caught
        # this, which is the whole point of a ratchet.
        from syncai_hydranet.serving.engine import bench_sync

        results["engine"] = bench_sync(args.plan, args.batch, args.seconds)
        print(" ", results["engine"])

    if "post" in legs:
        print("\npost (host decode + NMS) ...", flush=True)
        results["post"] = bench_post(
            args.batch, args.seconds, tuple(args.input_size), args.classes, args.candidates
        )
        print(" ", results["post"])

    if "track" in legs:
        print("\ntrack ...", flush=True)
        results["track"] = bench_track(args.seconds, args.boxes_per_frame)
        print(" ", results["track"])

    # Composition. `decode` and `engine` are measured as whole-device ceilings -- the
    # decode leg already runs `--streams` workers, the engine already owns the card.
    # `post` and `track` are per-thread costs of work that serving spreads over cores,
    # so a single-thread figure is NOT a ceiling: reporting one as though it were is how
    # a CPU leg gets declared the blocker when it needs four cores out of ninety-six.
    cores = os.cpu_count() or 1

    # the engine leg reports two figures and the difference between them IS the PCIe
    # story, so it is named rather than averaged away: `compute` is the ceiling once
    # frames arrive in device memory, which is what NVDEC decoding gives.
    def _rate(leg: str) -> float:
        r = results[leg]  # type: ignore[index]
        return float(r["frames_per_s_compute"] if leg == "engine" else r["frames_per_s"])

    rates = {leg: _rate(leg) for leg in ("decode", "engine", "post", "track") if leg in results}
    device_legs = {k: v for k, v in rates.items() if k in ("decode", "engine")}
    thread_legs = {k: v for k, v in rates.items() if k in ("post", "track")}
    if rates:
        print(f"\n{'leg':>8}  {'frames/s':>10}  {'kind':>12}  {'at 1,440 fps':>26}")
        for leg, fps in rates.items():
            if leg in device_legs:
                note = f"{fps / TARGET_FPS:.2f}x target"
            else:
                note = f"needs {TARGET_FPS / fps:.1f} of {cores} cores"
            kind = "device" if leg in device_legs else "per-thread"
            print(f"{leg:>8}  {fps:>10.1f}  {kind:>12}  {note:>26}")

        results["cpu_cores"] = cores
        results["cores_needed_at_target"] = {
            leg: round(TARGET_FPS / fps, 2) for leg, fps in thread_legs.items()
        }
        cores_total = sum(TARGET_FPS / fps for fps in thread_legs.values())
        results["cores_needed_total"] = round(cores_total, 2)

        blockers = [leg for leg, fps in device_legs.items() if fps < TARGET_FPS]
        if cores_total > cores:
            blockers.append(f"cpu ({cores_total:.1f} cores needed, {cores} present)")
        results["blockers"] = blockers
        results["meets_target"] = not blockers and bool(device_legs)

        if device_legs:
            slowest = min(device_legs, key=lambda k: device_legs[k])
            results["ceiling_frames_per_s"] = device_legs[slowest]
            results["binding_device_leg"] = slowest
            print(
                f"\ndevice ceiling {device_legs[slowest]:.1f} fps, bound by {slowest!r}"
                + ("  [CONTAMINATED: GPU was not idle]" if contaminated else "")
            )
        if thread_legs:
            print(
                f"host post-processing needs {cores_total:.1f} of {cores} cores to hold "
                f"{TARGET_FPS:.0f} fps"
            )
        print(
            "verdict: "
            + (
                "MEETS the target"
                if results["meets_target"]
                else f"blocked by {blockers or ['legs not run']}"
            )
        )
        missing = sorted({"decode", "engine", "post", "track"} - set(rates))
        if missing:
            print(f"legs not run: {missing} -- no verdict is complete without them")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
