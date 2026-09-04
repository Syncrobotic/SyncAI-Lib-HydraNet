#!/usr/bin/env python3
"""96-stream serving pipeline, increment 1: H2D recovery bench and a 16-stream pilot.

  # 1. prove the H2D recovery (uint8 binding, pinned + dual-stream overlap)
  .venv/bin/python scripts/serve_pilot.py bench-h2d --out runs/serve_pilot01

  # 2. run 16 simulated streams end to end: decode -> batch -> engine -> per-camera
  .venv/bin/python scripts/serve_pilot.py run --seconds 60 --out runs/serve_pilot01

Context: runs/bench_pro6000 measured the fp16 b16 engine at 3,272 f/s compute but
1,553 f/s with synchronous fp32 H2D -- the copy path, not the model, is the
frontier. bench-h2d measures the three named recoveries; run measures what a real
tick costs once decode, per-camera EMA, tracking and per-class thresholds hang off
the engine.

Decode is ffmpeg CPU from looping site clips. NVDEC is the known next step, not
done here, for a measured reason: the system ffmpeg has no cuvid/nvdec build
(`ffmpeg -decoders | grep nvdec` is empty), so hardware decode needs either a
rebuilt ffmpeg or PyNvVideoCodec -- an increment-2 dependency decision, not a flag.

GPU rule: run ONLY on an idle GPU; a throughput number taken on a shared card is
not a measurement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from syncai_hydranet.data.label_maps_retail_security import get_det_vocab  # noqa: E402
from syncai_hydranet.serving.camera import CameraState, load_thresholds  # noqa: E402
from syncai_hydranet.serving.decode import FcosDecoder  # noqa: E402
from syncai_hydranet.serving.engine import (  # noqa: E402
    TrtExecutor,
    bench_sync,
    build_plan,
)
from syncai_hydranet.serving.scheduler import BatchScheduler  # noqa: E402
from syncai_hydranet.serving.uint8_input import add_uint8_nhwc_input  # noqa: E402

FP32_ONNX = ROOT / "exports/pro6000/xl_b16.fp16.onnx"  # fp16 weights, fp32 IO
CANVAS_H, CANVAS_W = 512, 640
LEVELS = ("p3", "p4", "p5", "p6", "p7")
# Decode floor for score statistics; per-class working thresholds are applied on
# top by CameraState (both score populations stay visible, gdino_person_boxes
# precedent).
SCORE_FLOOR = 0.05
TARGET_FPS = 1440.0


def u8_plan_path() -> Path:
    """Graph-surgery the benchmarked fp16 ONNX to uint8 NHWC and build its engine."""
    u8_onnx = FP32_ONNX.with_name("xl_b16.u8.fp16.onnx")
    if not u8_onnx.is_file():
        add_uint8_nhwc_input(FP32_ONNX, u8_onnx)
        print(f"wrote {u8_onnx}")
    return build_plan(u8_onnx)


# ---------------------------------------------------------------------------
# bench-h2d
# ---------------------------------------------------------------------------


def bench_overlapped(plan: Path, seconds: float, fill: bool, d2h: bool) -> float:
    """Frames/s of the double-buffered executor. ``fill`` adds a per-batch host
    np.copyto into the pinned staging (the scheduler's assemble cost); ``d2h``
    adds the output copies the real pipeline needs."""
    ex = TrtExecutor(plan, enable_d2h=d2h)
    src = np.random.randint(0, 255, size=ex.input_shape).astype(ex.input_dtype)
    for slot in range(ex.SLOTS):
        np.copyto(ex.acquire_input(slot), src)
    # warm-up
    for i in range(10):
        ex.submit(i % ex.SLOTS)
        if d2h:
            ex.outputs(i % ex.SLOTS)
    ex.synchronize()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        slot = n % ex.SLOTS
        buf = ex.acquire_input(slot)
        if fill:
            np.copyto(buf, src)
        ex.submit(slot)
        if d2h:
            # Consume the *other* slot, exactly the pilot's discipline.
            ex.outputs((slot + 1) % ex.SLOTS)
        n += 1
    ex.synchronize()
    elapsed = time.perf_counter() - t0
    fps = n * ex.batch / elapsed
    ex.close()
    return round(fps, 1)


def cmd_bench_h2d(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    fp32_plan = FP32_ONNX.with_suffix(".plan")  # exports/pro6000/xl_b16.fp16.plan
    if not fp32_plan.is_file():
        raise SystemExit(f"{fp32_plan} missing; run scripts/bench_trt.py first")
    u8_plan = u8_plan_path()

    rows = {}
    print("== (a) fp32 sync H2D (baseline method: bench_trt.bench, restated) ==")
    a = bench_sync(fp32_plan, 16, args.seconds)
    rows["fp32_sync"] = {
        "plan": fp32_plan.name,
        "frames_per_s_compute": a["frames_per_s_compute"],
        "frames_per_s_h2d": a["frames_per_s_h2d"],
    }
    print(f"  compute {a['frames_per_s_compute']:.1f}  +H2D {a['frames_per_s_h2d']:.1f}")

    print("== (b) uint8 sync H2D (same method, uint8 binding) ==")
    b = bench_sync(u8_plan, 16, args.seconds)
    rows["uint8_sync"] = {
        "plan": u8_plan.name,
        "frames_per_s_compute": b["frames_per_s_compute"],
        "frames_per_s_h2d": b["frames_per_s_h2d"],
    }
    print(f"  compute {b['frames_per_s_compute']:.1f}  +H2D {b['frames_per_s_h2d']:.1f}")

    print("== (c) uint8 + pinned + dual-stream overlap ==")
    fps_c = bench_overlapped(u8_plan, args.seconds, fill=False, d2h=False)
    rows["uint8_pinned_overlap"] = {"plan": u8_plan.name, "frames_per_s": fps_c}
    print(f"  {fps_c:.1f} f/s")

    print("== (c') same + per-batch host fill of the pinned staging ==")
    fps_cf = bench_overlapped(u8_plan, args.seconds, fill=True, d2h=False)
    rows["uint8_pinned_overlap_fill"] = {"plan": u8_plan.name, "frames_per_s": fps_cf}
    print(f"  {fps_cf:.1f} f/s")

    print("== (c'') same + D2H of all outputs (what serving actually pays) ==")
    fps_cd = bench_overlapped(u8_plan, args.seconds, fill=True, d2h=True)
    rows["uint8_pinned_overlap_fill_d2h"] = {"plan": u8_plan.name, "frames_per_s": fps_cd}
    print(f"  {fps_cd:.1f} f/s")

    report = {
        "measured": "H2D recovery for the fp16 b16 engine, 512x640, RTX PRO 6000",
        "gpu_note": "valid only if the GPU was otherwise idle",
        "target_frames_per_s": TARGET_FPS,
        "baseline_reference": "runs/bench_pro6000/results.json "
        "(compute 3272.1, sync H2D 1553.5)",
        "seconds_per_row": args.seconds,
        "rows": rows,
    }
    out = args.out / "h2d_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# run: the 16-stream pilot
# ---------------------------------------------------------------------------


def probe_wh(clip: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def letterbox_filter(src_w: int, src_h: int) -> str:
    """The same geometry as utils.visualize.letterbox, as an ffmpeg filter chain."""
    s = min(CANVAS_W / src_w, CANVAS_H / src_h)
    nw, nh = max(round(src_w * s), 1), max(round(src_h * s), 1)
    x0, y0 = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
    # 0x72 = 114 = preprocessing.PAD_COLOR
    return f"scale={nw}:{nh},pad={CANVAS_W}:{CANVAS_H}:{x0}:{y0}:color=0x727272"


def reader(
    camera: str,
    clip: Path,
    sched: BatchScheduler,
    stop: threading.Event,
    fps: float,
) -> None:
    """Decode one looping clip with ffmpeg into the scheduler's mailbox.

    ``fps`` > 0 paces delivery to that rate (the reader sleeps; the stdout pipe's
    backpressure then throttles ffmpeg itself, so decode CPU is what a real
    camera at that rate would cost). 0 free-runs -- which on this box saturates
    all cores with 1080p software decode and starves everything downstream; the
    pilot measured 16 free-running decoders dragging the whole pipeline to 1.8
    ticks/s, which is the NVDEC argument in one number.
    """
    src_w, src_h = probe_wh(clip)
    nbytes = CANVAS_H * CANVAS_W * 3
    cmd = [
        "ffmpeg", "-v", "error", "-stream_loop", "-1", "-i", str(clip),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-vf", letterbox_filter(src_w, src_h), "-",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nbytes)
    assert proc.stdout is not None
    seq = 0
    interval = 1.0 / fps if fps > 0 else 0.0
    next_t = time.perf_counter()
    try:
        while not stop.is_set():
            raw = proc.stdout.read(nbytes)
            if raw is None or len(raw) < nbytes:
                break
            if interval:
                next_t += interval
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()  # decode fell behind; do not burst
            seq += 1
            frame = np.frombuffer(raw, np.uint8).reshape(CANVAS_H, CANVAS_W, 3)
            sched.offer(camera, seq, frame)
    finally:
        proc.kill()
        proc.wait()


def discover_streams(clips_root: Path, n: int) -> list[tuple[str, Path]]:
    """First n cameras that have at least one clip; one clip each, spread over the
    camera's available slots so the pilot is not 16 copies of one scene."""
    streams = []
    for cam_dir in sorted(p for p in clips_root.iterdir() if p.is_dir()):
        clips = sorted(cam_dir.glob("*.mp4"))
        if not clips:
            continue
        streams.append((cam_dir.name, clips[len(streams) % len(clips)]))
        if len(streams) == n:
            return streams
    raise SystemExit(f"only {len(streams)} cameras with clips under {clips_root}, need {n}")


def make_tracker_factory(vel_scale: float):
    # The measured tracker, moved into the package as analytics/bytetrack (c6c67a0).
    import numpy as _np
    from scipy.optimize import linear_sum_assignment

    from syncai_hydranet.analytics.bytetrack import OfflineForward, iou
    from syncai_hydranet.serving.camera import BIRTH_REF, KEEP_REF

    class ScipyAssocForward(OfflineForward):
        """OfflineForward with the assignment done by scipy's C solver.

        The parent's pure-Python O(n^3) Hungarian is deliberate where it lives
        (reid_metrics keeps scipy out of the package's dependencies) but it holds
        the GIL for ~9.5 ms per crowded frame, which serialised the pilot's whole
        post pool -- 16 streams x 9.5 ms = 150 ms of bytecode per tick. Same
        matrix, same threshold rule, optimal either way; only the solver changes.
        Increment 2 should decide this inside bytetrack (scipy as an optional
        fast path) instead of here.
        """

        @staticmethod
        def _associate(tracks, boxes, thr):
            if not tracks or len(boxes) == 0:
                return {}, set(range(len(boxes)))
            pred = _np.stack([t.kalman.box for t in tracks])
            m = iou(pred, boxes)
            obs = _np.stack([t.boxes[-1] for t in tracks])
            m = _np.maximum(m, iou(obs, boxes))
            rows, cols = linear_sum_assignment(-m)
            pairs = {
                int(ti): int(di) for ti, di in zip(rows, cols, strict=True) if m[ti, di] >= thr
            }
            return pairs, set(range(len(boxes))) - set(pairs.values())

    # The tracker's global hysteresis; CameraState rescales per-class onto it.
    return lambda: ScipyAssocForward(BIRTH_REF, KEEP_REF, 0.3, 0.4, 5, 2, vel_scale)


def live_tracks(state: CameraState) -> int:
    tracker = state.tracker
    return 0 if tracker is None else len([t for t in tracker.tracks if t.confirmed])


def cmd_run(args) -> int:
    import torch

    # Post runs 16 per-camera torch updates concurrently in a thread pool; the
    # default intra-op thread count (one per core) would oversubscribe 16x24.
    torch.set_num_threads(args.torch_threads)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = u8_plan_path()
    vocab = get_det_vocab("retail_security")
    det_classes = list(vocab.classes)
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    terrain_classes = cfg["data"]["terrain_classes"]

    book = load_thresholds(args.thresholds)
    overridden = sorted(book.cameras)
    streams = discover_streams(ROOT / "datasets/studioa_clips", args.streams)
    factory = make_tracker_factory(vel_scale=25.0 / args.assumed_fps)
    cameras = {}
    for name, _clip in streams:
        cameras[name] = CameraState(
            camera=name,
            num_terrain_classes=len(terrain_classes),
            canvas_hw=(CANVAS_H, CANVAS_W),
            det_classes=det_classes,
            # Per camera, not one dict for the fleet: Kaohsiung-cam04's person score
            # calibration is under investigation and the book holds it at the shipped
            # working point while the rest of the fleet moves.
            thresholds=book.for_camera(name),
            calib_path=ROOT / f"runs/onboard01/{name}.calib.json",
            tracker_factory=factory,
        )
    calibrated = sorted(c for c, s in cameras.items() if s.calib is not None)
    print(f"{len(streams)} streams; {len(calibrated)} with calibration")
    for name in overridden:
        if name in cameras:
            print(f"  {name}: threshold override -- {book.basis_for(name)}")

    ex = TrtExecutor(plan, enable_d2h=True)
    if ex.batch != args.streams:
        raise SystemExit(f"engine batch {ex.batch} != streams {args.streams}")
    # pre_nms_topk: the 0.05 stats floor admits ~6-7k candidates/frame and NMS over
    # them measured ~500 ms/tick; 512 is 5x max_det, so the kept set is unaffected.
    decoder = FcosDecoder(num_classes=len(det_classes), pre_nms_topk=512)
    sched = BatchScheduler([name for name, _ in streams], batch=ex.batch)

    stop = threading.Event()
    threads = [
        threading.Thread(
            target=reader, args=(name, clip, sched, stop, args.stream_fps), daemon=True
        )
        for name, clip in streams
    ]
    for t in threads:
        t.start()

    # Wait until every stream has produced one frame, so t0 measures serving, not
    # ffmpeg start-up.
    while any(s["last_seq"] < 1 for s in sched.stats().values()):
        time.sleep(0.05)

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=min(16, args.post_workers))
    score_hist = np.zeros((len(det_classes), 100), np.int64)  # 0.01-wide bins
    buckets: dict[str, list[float]] = {
        k: [] for k in ("assemble_ms", "submit_ms", "post_ms", "post_decode_ms",
                        "post_update_ms", "idle_ms",
                        "gpu_h2d_ms", "gpu_compute_ms", "gpu_d2h_ms", "fresh")
    }  # fmt: skip
    inflight: list[list | None] = [None, None]
    tick_no = 0
    frames_done = 0
    # Tick on the stream cadence: with paced streams, ticking faster than frames
    # arrive would spend full batch-16 computes on one or two fresh slots. 0 (the
    # free-run worst case) ticks as fast as the pipeline allows.
    tick_interval = 1.0 / args.stream_fps if args.stream_fps > 0 else 0.0
    t0 = time.perf_counter()
    next_tick = t0

    def post_one(item, terrain, det):
        state = cameras[item.camera]
        state.update(item.seq, terrain, det["boxes"], det["scores"], det["labels"])

    while time.perf_counter() - t0 < args.seconds:
        if tick_interval:
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            next_tick = max(next_tick + tick_interval, time.perf_counter())
        t_a = time.perf_counter()
        items = sched.tick()
        if not items:
            time.sleep(0.001)
            buckets["idle_ms"].append((time.perf_counter() - t_a) * 1e3)
            continue
        slot = tick_no % ex.SLOTS
        buf = ex.acquire_input(slot)
        for it in items:
            np.copyto(buf[it.slot], it.frame)  # stale slots keep last content
        t_b = time.perf_counter()
        ex.submit(slot)
        t_c = time.perf_counter()

        prev = (slot + 1) % ex.SLOTS
        prev_items = inflight[prev]
        if prev_items is not None:
            outs = ex.outputs(prev)
            idx = [it.slot for it in prev_items]
            cls = [outs[f"det_cls_{lv}"][idx] for lv in LEVELS]
            reg = [outs[f"det_reg_{lv}"][idx] for lv in LEVELS]
            ctr = [outs[f"det_ctr_{lv}"][idx] for lv in LEVELS]
            t_dec0 = time.perf_counter()
            dets = decoder(cls, reg, ctr, score_thr=SCORE_FLOOR,
                           img_size=(CANVAS_H, CANVAS_W))  # fmt: skip
            buckets["post_decode_ms"].append((time.perf_counter() - t_dec0) * 1e3)
            for det in dets:
                for c in range(len(det_classes)):
                    sel = det["labels"] == c
                    if sel.any():
                        score_hist[c] += np.histogram(
                            det["scores"][sel], bins=100, range=(0.0, 1.0)
                        )[0]
            terrain = outs["terrain_argmax"]
            t_upd0 = time.perf_counter()
            futs = [
                pool.submit(post_one, it, terrain[it.slot], det)
                for it, det in zip(prev_items, dets, strict=True)
            ]
            for f in futs:
                f.result()
            buckets["post_update_ms"].append((time.perf_counter() - t_upd0) * 1e3)
            frames_done += len(prev_items)
            gpu = ex.timings_ms(prev)
            buckets["gpu_h2d_ms"].append(gpu["h2d"])
            buckets["gpu_compute_ms"].append(gpu["compute"])
            buckets["gpu_d2h_ms"].append(gpu["d2h"])
        t_d = time.perf_counter()

        inflight[slot] = items
        buckets["assemble_ms"].append((t_b - t_a) * 1e3)
        buckets["submit_ms"].append((t_c - t_b) * 1e3)
        buckets["post_ms"].append((t_d - t_c) * 1e3)
        buckets["fresh"].append(len(items))
        tick_no += 1

    elapsed = time.perf_counter() - t0
    stop.set()
    ex.synchronize()
    pool.shutdown(wait=True)

    def dist(v):
        if not v:
            return None
        a = np.asarray(v, np.float64)
        return {
            "mean": round(float(a.mean()), 3),
            "p50": round(float(np.percentile(a, 50)), 3),
            "p90": round(float(np.percentile(a, 90)), 3),
        }

    stats = sched.stats()
    per_class_scores = {}
    for c, name in enumerate(det_classes):
        h = score_hist[c]
        total = int(h.sum())
        if total:
            cdf = np.cumsum(h) / total
            centers = np.arange(100) * 0.01 + 0.005
            per_class_scores[name] = {
                "n": total,
                "p50": round(float(centers[np.searchsorted(cdf, 0.5)]), 3),
                "p90": round(float(centers[np.searchsorted(cdf, 0.9)]), 3),
                "p99": round(float(centers[np.searchsorted(cdf, 0.99)]), 3),
                "share_ge_0.30": round(float(h[30:].sum() / total), 4),
            }
        else:
            per_class_scores[name] = {"n": 0}

    result = {
        "measured": "16 simulated streams end to end: ffmpeg decode -> batch-16 ticks "
        "-> uint8 TRT engine (pinned, double-buffered) -> per-camera EMA + tracks",
        "gpu_note": "valid only if the GPU was otherwise idle",
        "seconds": round(elapsed, 1),
        "streams": args.streams,
        "engine_plan": plan.name,
        "ticks": tick_no,
        "ticks_per_s": round(tick_no / elapsed, 1),
        "frames_consumed": frames_done,
        "frames_per_s_end_to_end": round(frames_done / elapsed, 1),
        "per_stream_fps_mean": round(frames_done / elapsed / args.streams, 2),
        "fresh_per_tick": dist(buckets["fresh"]),
        "tick_ms": {
            "assemble": dist(buckets["assemble_ms"]),
            "submit": dist(buckets["submit_ms"]),
            "post": dist(buckets["post_ms"]),
            "post_decode": dist(buckets["post_decode_ms"]),
            "post_update": dist(buckets["post_update_ms"]),
            "idle_spin": dist(buckets["idle_ms"]),
        },
        "gpu_ms_overlapped": {
            "h2d": dist(buckets["gpu_h2d_ms"]),
            "compute": dist(buckets["gpu_compute_ms"]),
            "d2h": dist(buckets["gpu_d2h_ms"]),
        },
        "per_stream": {
            name: {
                "decoded": s["last_seq"],
                "consumed": s["delivered"],
                "dropped_never_consumed": s["dropped"],
                "consumed_fps": round(s["delivered"] / elapsed, 2),
                "decode_fps": round(s["last_seq"] / elapsed, 2),
                "calibrated": cameras[name].calib is not None,
                "live_tracks": live_tracks(cameras[name]),
            }
            for name, s in stats.items()
        },
        "per_class_scores_at_floor_0.05": per_class_scores,
        "notes": [
            "decode is CPU ffmpeg free-running over looping clips; per_stream.decode_fps "
            "is the CPU decode ceiling per stream, consumed_fps what the ticks took",
            "decode is still the CPU pipe: PyNvVideoCodec (PLAN 7 decision 8, ~520 "
            "streams against this pipe's 63-69) is installed and declared, and porting "
            "the serving path to it is authorised work that is not done -- the system "
            "ffmpeg's missing cuvid build is no longer the reason",
            "per-class score table exists to set boxed_stock's working threshold per "
            "checkpoint -- see the threshold book passed as --thresholds",
        ],
    }
    out = args.out / "pilot.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "ticks_per_s", "frames_per_s_end_to_end", "per_stream_fps_mean",
        "tick_ms", "gpu_ms_overlapped")}, indent=2))  # fmt: skip
    print(f"wrote {out}")
    ex.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="serve_pilot", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench-h2d", help="measure the three H2D recoveries")
    b.add_argument("--out", type=Path, default=Path("runs/serve_pilot01"))
    b.add_argument("--seconds", type=float, default=15.0)
    b.set_defaults(fn=cmd_bench_h2d)
    r = sub.add_parser("run", help="16 simulated streams end to end")
    r.add_argument("--out", type=Path, default=Path("runs/serve_pilot01"))
    r.add_argument("--seconds", type=float, default=60.0)
    r.add_argument("--streams", type=int, default=16)
    r.add_argument(
        "--config",
        default="runs/hydranet_retail_security_b03_cw_xl-20260825-162131/config.yaml",
        help="read only for data.terrain_classes",
    )
    r.add_argument("--stream-fps", type=float, default=15.0,
                   help="pace each simulated stream at this rate (the 96x15 target "
                   "rate per stream); 0 free-runs the decoders, which saturates the "
                   "CPU and measures the no-NVDEC worst case")  # fmt: skip
    r.add_argument("--assumed-fps", type=float, default=8.0,
                   help="Kalman noise scale assumes this consumption rate")  # fmt: skip
    r.add_argument("--thresholds", type=Path,
                   default=Path("configs/serving/thresholds_retail_security.json"),
                   help="the threshold book: fleet defaults plus per-camera overrides, "
                   "each of which states the measurement that produced it")  # fmt: skip
    r.add_argument("--post-workers", type=int, default=16)
    r.add_argument("--torch-threads", type=int, default=2,
                   help="intra-op threads per torch op; post overlaps 16 camera "
                   "updates, so small keeps the pool from oversubscribing")  # fmt: skip
    r.set_defaults(fn=cmd_run)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
