#!/usr/bin/env python3
"""Serve live predictions from a robot that is already using its camera.

    source /opt/ros/humble/setup.bash
    python3 live_view_ros.py --weights det60_ema.pt --config configs/hydranet_indoor.yaml \
        --port 8090

Then open http://<robot-ip>:8090/ and point the camera at things. No client to install.

`live_view_orin.py` is the bench-rig version: it opens /dev/video and runs a TensorRT
engine. This one exists because 10.8.140.130 is a working robot -- nav2, motion and a
RealSense node hold the camera, and taking the device would stop the robot. So it
subscribes to the topics that node already publishes, including depth registered to the
colour frame, and runs the checkpoint in PyTorch.

PyTorch eager on this board is ~122 ms a frame against TensorRT's 5 ms. That is fine for
looking and useless for benchmarking; quote numbers from an engine, not from here.

The right panel is the part worth staring at. It splits what the model calls walkable into
three:

    green    walkable, and the depth sensor agrees it is within range
    yellow   walkable, but further away than the range -- not this planner's problem yet
    magenta  walkable, and the depth sensor returns nothing at all

Magenta is the interesting colour. Glass, mirrors and specular reflections off a polished
floor are what make a surface look like open floor to a camera *and* return nothing to a
depth sensor, so magenta is where this deployment's worst failure would appear first.
"""

from __future__ import annotations

import argparse
import io
import json
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# In the repo the package is at ../src; on a robot the deploy copies it to ./src next to
# this script. Accept both rather than making the caller care which layout they are in.
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from bev_page import PAGE as BEV_PAGE  # noqa: E402

from syncai_hydranet.data.coco_subsets import COCO_NAMES  # noqa: E402
from syncai_hydranet.geometry.depth_scene import build_scene  # noqa: E402

from syncai_hydranet.config import load_config  # noqa: E402  # isort: skip
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import (  # noqa: E402
    TRAV_COLORS,
    crop_box,
    overlay,
    preprocess,
    terrain_palette,
)

GO = 2
COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
COLOR_INFO_TOPIC = "/camera/camera/color/camera_info"
DEPTH_INFO_TOPIC = "/camera/camera/aligned_depth_to_color/camera_info"

# reachable / out of range / no depth return
REACH_COLORS = np.array([[0, 0, 0], [40, 220, 90], [250, 200, 40], [230, 60, 230]], np.uint8)

state = {"range": 5.0, "score": 0.30, "view": "both"}
latest = {"jpeg": None, "stats": {}, "scene": None}
lock = threading.Lock()
recorders: list = []  # closed on SIGTERM so the video has a readable index

PAGE = b"""<!doctype html><meta charset=utf-8><title>HydraNet live</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}
 img{max-width:100%;border:1px solid #333}
 a{color:#8cf;text-decoration:none;padding:3px 9px;border:1px solid #345;border-radius:4px}
 a:hover{background:#1b2a3a}
 .row{margin:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 .k{color:#888;margin-right:4px} b{color:#fff} .sw{display:inline-block;width:11px;height:11px;
 border-radius:2px;margin-right:5px;vertical-align:-1px}
</style>
<div class=row><span class=k>range</span>
 <a href="/set?range=2">2 m</a><a href="/set?range=3">3 m</a><a href="/set?range=5">5 m</a>
 <a href="/set?range=8">8 m</a>
 <span class=k style="margin-left:14px">view</span>
 <a href="/set?view=both">trav + reach</a><a href="/set?view=terrain">terrain</a>
 <span class=k style="margin-left:14px">score</span>
 <a href="/set?score=0.15">0.15</a><a href="/set?score=0.30">0.30</a>
 <a href="/set?score=0.50">0.50</a>
</div>
<a href="/3d">3D scene</a>
</div>
<img src="/stream">
<div class=row id=s></div>
<div class=row style="color:#777">
 <span><i class=sw style="background:#28dc5a"></i>walkable, within range</span>
 <span><i class=sw style="background:#fac828"></i>walkable, beyond range</span>
 <span><i class=sw style="background:#e63ce6"></i>walkable, no depth return
 &mdash; glass / mirror / reflection</span>
</div>
<script>
setInterval(async()=>{const r=await fetch('/stats');const d=await r.json();
 document.getElementById('s').innerHTML=Object.entries(d).map(([k,v])=>
 `<span><span class=k>${k}</span><b>${v}</b></span>`).join('');},1000);
</script>
"""


def decode(msg) -> np.ndarray:
    """sensor_msgs/Image -> ndarray, honouring `step` rather than assuming packed rows."""
    if msg.encoding in ("bgr8", "rgb8"):
        buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        arr = buf[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return arr[:, :, ::-1] if msg.encoding == "bgr8" else arr
    if msg.encoding in ("16UC1", "mono16"):
        buf = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)
        return buf[:, : msg.width]
    raise ValueError(f"unhandled encoding {msg.encoding}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, body: bytes, ctype="text/html; charset=utf-8", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(PAGE)
        if url.path == "/3d":
            return self._send(BEV_PAGE)
        if url.path == "/scene.json":
            with lock:
                scene = latest["scene"]
            body = json.dumps(
                scene
                or {
                    "grid": {
                        "nx": 0,
                        "nz": 0,
                        "cells": [],
                        "cell": 0.2,
                        "x_min": 0.0,
                        "blind_fraction": 0.0,
                    },
                    "objects": [],
                    "range_m": state["range"],
                }
            )
            return self._send(body.encode(), "application/json")
        if url.path == "/stats":
            with lock:
                body = json.dumps(latest["stats"]).encode()
            return self._send(body, "application/json")
        if url.path == "/set":
            q = parse_qs(url.query)
            for key, cast in (("range", float), ("score", float), ("view", str)):
                if key in q:
                    state[key] = cast(q[key][0])
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            return None
        if url.path != "/stream":
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with lock:
                    buf = latest["jpeg"]
                if buf is None:
                    time.sleep(0.02)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(buf)}\r\n\r\n".encode())
                self.wfile.write(buf + b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the tab closed


class Recorder:
    """Write a walk-through to disk as three things, not one.

    The overlay video is for review. The raw keyframes are the point: footage at the
    robot's camera height is the single thing this project is short of, and a walk round
    a building produces it as a by-product of testing. They are written in the layout
    `hydranet-annotation check` expects, under a session directory, because a frame
    without a session cannot be split honestly later.

    The stats file is what makes the video searchable -- scrubbing 10 minutes of footage
    for the moment the floor stopped returning depth is worse than sorting a JSONL by it.

    And the calibration: the camera's intrinsic matrix, written once per session. Without
    K, nothing recorded here can ever be projected to metres -- no ground plane, no BEV
    costmap, no distance gate -- and the omission is invisible until someone tries, by
    which time the building has been walked and the robot has moved on. A session without
    its intrinsics is the same class of defect as a frame without its session: usable for
    looking at, useless for the thing it was collected for.
    """

    def __init__(self, root: Path, session: str, keyframe_hz: float):
        import cv2  # pyright: ignore[reportMissingImports]

        self.cv2 = cv2
        self.root = root
        self.img_dir = root / "images" / session
        self.depth_dir = root / "depth" / session
        root.mkdir(parents=True, exist_ok=True)
        if keyframe_hz > 0:
            for d in (self.img_dir, self.depth_dir):
                d.mkdir(parents=True, exist_ok=True)
        # MJPG in AVI: a run that ends with a pulled power cable still plays back. mp4
        # needs its trailer written, and a robot session is exactly when that is missed.
        self.video_path = root / f"{session}_overlay.avi"
        self.writer = None
        self.stats = (root / f"{session}_stats.jsonl").open("a", buffering=1)
        self.calib_path = root / f"{session}_calibration.json"
        self.period = (1.0 / keyframe_hz) if keyframe_hz > 0 else 0.0
        self.last_key = 0.0
        self.n_frames = self.n_keys = 0

    def write_calibration(self, color_info, depth_info, stride: int) -> None:
        """Intrinsics, once. They do not change during a session, and the stride this
        script applies to the stream does change them -- so what is written is the
        calibration of the frames on disk, not of the topic."""
        if self.calib_path.exists():
            return

        def one(msg):
            if msg is None:
                return None
            k = list(msg.k)
            # Subsampling scales fx, fy, cx, cy by exactly the same factor. Writing the
            # topic's K next to decimated frames would be off by that factor, silently.
            if stride > 1:
                k = [v / stride if i in (0, 2, 4, 5) else v for i, v in enumerate(k)]
            return {
                "K": k,
                "distortion_model": msg.distortion_model,
                "D": list(msg.d),
                "width": msg.width // stride,
                "height": msg.height // stride,
                "frame_id": msg.header.frame_id,
            }

        payload = {
            "color": one(color_info),
            "depth_aligned_to_color": one(depth_info),
            "subsample_stride": stride,
            "depth_units": "millimetres, uint16",
            "note": "K is scaled for the frames written here, not the raw topic",
        }
        self.calib_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {self.calib_path}", flush=True)

    def write(self, pair: Image.Image, color: np.ndarray, depth_mm: np.ndarray, stats: dict):
        import numpy as np_

        frame = self.cv2.cvtColor(np_.asarray(pair), self.cv2.COLOR_RGB2BGR)
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = self.cv2.VideoWriter(
                str(self.video_path), self.cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h)
            )
        self.writer.write(frame)
        self.n_frames += 1

        now = time.time()
        if self.period > 0 and now - self.last_key >= self.period:
            self.last_key = now
            stem = f"{self.n_keys:06d}"
            self.cv2.imwrite(
                str(self.img_dir / f"{stem}.jpg"),
                self.cv2.cvtColor(color, self.cv2.COLOR_RGB2BGR),
                [int(self.cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            # 16-bit PNG keeps millimetres exactly; a lossy depth frame is a lie.
            self.cv2.imwrite(str(self.depth_dir / f"{stem}.png"), depth_mm)
            self.n_keys += 1
            stats = {**stats, "keyframe": stem}
        self.stats.write(json.dumps({"t": round(now, 3), **stats}) + "\n")

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.stats.close()


def inference_loop(args):
    import rclpy  # pyright: ignore[reportMissingImports]
    from rclpy.qos import qos_profile_sensor_data  # pyright: ignore[reportMissingImports]
    from sensor_msgs.msg import CameraInfo  # pyright: ignore[reportMissingImports]
    from sensor_msgs.msg import Image as ImageMsg  # pyright: ignore[reportMissingImports]

    cfg = load_config(args.config, ["model.backbone.pretrained=false"])
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    size = cfg["data"]["input_size"]
    terrain_colors = terrain_palette(cfg["data"].get("terrain_classes"))
    print(f"model on {device}, input {size}", flush=True)

    rclpy.init()
    node = rclpy.create_node("hydranet_live")
    msgs = {"color": None}
    # Depth arrives on its own schedule. Taking whichever depth frame happens to be
    # latest pairs images up to a full frame apart -- measured on this robot: mean 33 ms
    # of stamp skew, 67 ms worst case, which is 7 cm of parallax at walking pace and
    # lands the range mask slightly beside the thing it is describing. Keep a short
    # history and pick the nearest stamp instead.
    depth_buf: deque = deque(maxlen=12)
    # Sensor-data QoS: best-effort and shallow. The driver publishes RELIABLE, which a
    # best-effort subscriber may still read; a reliable subscriber on a 41 MB/s topic
    # makes the publisher retransmit for a viewer that nobody is waiting on.
    node.create_subscription(
        ImageMsg, COLOR_TOPIC, lambda m: msgs.__setitem__("color", m), qos_profile_sensor_data
    )
    node.create_subscription(ImageMsg, DEPTH_TOPIC, depth_buf.append, qos_profile_sensor_data)
    # Intrinsics. Latched-ish and cheap, and without them the whole recording is
    # geometrically meaningless later; see Recorder.write_calibration.
    info = {"color": None, "depth": None}
    node.create_subscription(
        CameraInfo,
        COLOR_INFO_TOPIC,
        lambda m: info.__setitem__("color", m),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        CameraInfo,
        DEPTH_INFO_TOPIC,
        lambda m: info.__setitem__("depth", m),
        qos_profile_sensor_data,
    )

    def stamp_of(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def scaled_k(msg, stride: int):
        """The intrinsics of the frames this loop actually works on.

        Subsampling scales fx, fy, cx, cy by the stride. Using the topic's K on a
        decimated image puts every metre off by exactly that factor, and nothing
        complains -- the scene is simply the wrong size.
        """
        k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        k[0, 0] /= stride
        k[1, 1] /= stride
        k[0, 2] /= stride
        k[1, 2] /= stride
        return k

    recorder = None
    if args.record_dir:
        recorder = Recorder(Path(args.record_dir), args.session, args.keyframe_hz)
        recorders.append(recorder)
        print(f"recording to {args.record_dir}/{args.session}*", flush=True)

    fps, last = 0.0, time.perf_counter()
    while True:
        rclpy.spin_once(node, timeout_sec=0.5)
        # Inference runs at ~11 FPS against a 15 Hz publisher, so callbacks queue up and
        # `spin_once` hands back the oldest one first: measured, that put the displayed
        # frame 535 ms behind reality while the loop itself was only 90 ms. Drain what has
        # arrived and keep the newest -- a live view that lags by half a second is worse
        # than one that drops frames, because the lag is invisible in the picture.
        for _ in range(16):  # queue depth is 5 per topic; 16 drains both with room over
            rclpy.spin_once(node, timeout_sec=0.0)
        if msgs["color"] is None or not depth_buf:
            continue
        color_msg = msgs["color"]
        t_color = stamp_of(color_msg)
        depth_msg = min(depth_buf, key=lambda m: abs(stamp_of(m) - t_color))
        skew_ms = 1000 * abs(stamp_of(depth_msg) - t_color)
        age_ms = 1000 * (time.time() - t_color)
        color = decode(color_msg)[:: args.stride, :: args.stride]
        depth_mm = decode(depth_msg)[:: args.stride, :: args.stride]
        depth_m = depth_mm.astype(np.float32) * 0.001

        x, canvas, region = preprocess(Image.fromarray(color), size)
        with torch.no_grad():
            result = model.predict(x.to(device), score_thr=state["score"])
        trav = crop_box(result["traversability"][0].cpu().numpy(), region)
        terr = crop_box(result["terrain"][0].cpu().numpy(), region)
        x0, y0, cw, ch = region
        vis = canvas.crop((x0, y0, x0 + cw, y0 + ch))

        # Walkable is the network's answer; range is the sensor's. They meet here, not
        # inside the weights -- so the threshold is a slider rather than a training run.
        trav_full = np.asarray(
            Image.fromarray(trav.astype(np.uint8)).resize(
                (color.shape[1], color.shape[0]), Image.Resampling.NEAREST
            )
        )
        go = trav_full == GO
        valid = depth_m > 0
        reach = np.zeros_like(trav_full)
        reach[go & valid & (depth_m <= state["range"])] = 1
        reach[go & valid & (depth_m > state["range"])] = 2
        reach[go & ~valid] = 3

        left = overlay(
            vis,
            terr if state["view"] == "terrain" else trav,
            terrain_colors if state["view"] == "terrain" else TRAV_COLORS,
        )
        det = result.get("detection", [{}])[0]
        if det and len(det.get("boxes", [])):
            draw = ImageDraw.Draw(left)
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels, strict=True):
                bx = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
                draw.rectangle(bx, outline=(255, 255, 0), width=2)
                name = COCO_NAMES[int(label)] if int(label) < len(COCO_NAMES) else str(label)
                draw.text((bx[0] + 2, bx[1] + 2), f"{name} {score:.2f}", fill=(255, 255, 0))

        right_ids = np.asarray(
            Image.fromarray(reach.astype(np.uint8)).resize(vis.size, Image.Resampling.NEAREST)
        )
        right = overlay(vis, right_ids, REACH_COLORS, alpha=0.5)
        pair = Image.new("RGB", (left.width * 2 + 8, left.height), (16, 16, 16))
        pair.paste(left, (0, 0))
        pair.paste(right, (left.width + 8, 0))

        buf = io.BytesIO()
        pair.save(buf, format="JPEG", quality=80)
        now = time.perf_counter()
        fps = 0.8 * fps + 0.2 / max(now - last, 1e-6)
        last = now
        frame_stats = {
            "fps": f"{fps:.1f}",
            "range": f"{state['range']:.0f} m",
            "score": f"{state['score']:.2f}",
            "go": f"{100 * go.mean():.1f}%",
            "go within range": f"{100 * (reach == 1).mean():.1f}%",
            "go beyond range": f"{100 * (reach == 2).mean():.1f}%",
            "go with no depth": f"{100 * (reach == 3).sum() / max(go.sum(), 1):.1f}% of go",
            "depth valid": f"{100 * valid.mean():.1f}%",
            "frame age": f"{age_ms:.0f} ms",
            "colour/depth skew": f"{skew_ms:.0f} ms",
            "detections": int(len(det.get("boxes", [])) if det else 0),
        }
        with lock:
            latest["jpeg"] = buf.getvalue()
            latest["stats"] = frame_stats
        # The metric scene for /3d. Boxes come back in canvas coordinates, so they are
        # mapped to the colour frame before any depth is read through them.
        if info["color"] is not None:
            sx = color.shape[1] / max(cw, 1)
            sy = color.shape[0] / max(ch, 1)
            dets = []
            if det and len(det.get("boxes", [])):
                for box, score, label in zip(
                    det["boxes"].cpu().numpy(),
                    det["scores"].cpu().numpy(),
                    det["labels"].cpu().numpy(),
                    strict=True,
                ):
                    dets.append(
                        {
                            "box": (
                                (box[0] - x0) * sx,
                                (box[1] - y0) * sy,
                                (box[2] - x0) * sx,
                                (box[3] - y0) * sy,
                            ),
                            "cls": COCO_NAMES[int(label)]
                            if int(label) < len(COCO_NAMES)
                            else str(label),
                            "score": float(score),
                        }
                    )
            scene = build_scene(trav_full, depth_m, scaled_k(info["color"], args.stride), dets)
            scene["range_m"] = state["range"]
            with lock:
                latest["scene"] = scene

        if recorder is not None:
            if info["color"] is not None:
                recorder.write_calibration(info["color"], info["depth"], args.stride)
            recorder.write(pair, color, depth_mm, frame_stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8090)
    # Matches `live_view_orin.py`, which has had this flag all along while this file
    # hardcoded 0.0.0.0. Same default, so nothing about an existing invocation changes;
    # the point is that a robot on a shop floor can now be told to serve its camera to
    # localhost only, which previously was not expressible here at all.
    ap.add_argument("--bind", default="0.0.0.0", help="interface to serve on")
    ap.add_argument("--range", type=float, default=5.0, help="metres")
    ap.add_argument("--score", type=float, default=0.30)
    ap.add_argument(
        "--stride", type=int, default=2, help="subsample the 1280x720 stream before resizing"
    )
    ap.add_argument(
        "--record-dir",
        help="record the session here: overlay video, raw keyframes, depth, per-frame stats",
    )
    ap.add_argument(
        "--session",
        default="session",
        help="session name. It becomes the frame directory, and session identity is what "
        "makes an honest train/val/test split possible later",
    )
    ap.add_argument(
        "--keyframe-hz",
        type=float,
        default=0.0,
        help="also keep raw colour+depth pairs this often, for annotation later. 0 is off, "
        "which records only what the page shows",
    )
    args = ap.parse_args()
    state["range"], state["score"] = args.range, args.score

    def shutdown(_signum, _frame):
        # An AVI killed mid-write still plays, but only if the writer released its index.
        for rec in recorders:
            rec.close()
            print(f"recorded {rec.n_frames} frames to {rec.video_path}", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    threading.Thread(target=inference_loop, args=(args,), daemon=True).start()
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"serving on http://{args.bind}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
