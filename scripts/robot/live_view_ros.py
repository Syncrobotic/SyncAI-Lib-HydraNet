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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# In the repo the package is at ../src; on a robot the deploy copies it to ./src next to
# this script. Accept both rather than making the caller care which layout they are in.
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from bev_page import PAGE as BEV_PAGE  # noqa: E402

from syncai_hydranet.config import load_config  # noqa: E402  # isort: skip
from syncai_hydranet.live import LiveSettings, Recorder, render_frame  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import terrain_palette  # noqa: E402

COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
COLOR_INFO_TOPIC = "/camera/camera/color/camera_info"
DEPTH_INFO_TOPIC = "/camera/camera/aligned_depth_to_color/camera_info"

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

        now = time.perf_counter()
        fps = 0.8 * fps + 0.2 / max(now - last, 1e-6)
        last = now
        frame = render_frame(
            color,
            depth_m,
            model,
            device,
            size=size,
            terrain_colors=terrain_colors,
            settings=LiveSettings(state["range"], state["score"], state["view"]),
            # Skipped rather than guessed until the intrinsics arrive: a scene built from
            # the wrong K is self-consistent and the wrong size.
            k=None if info["color"] is None else scaled_k(info["color"], args.stride),
            extra_stats={
                "fps": f"{fps:.1f}",
                "frame age": f"{age_ms:.0f} ms",
                "colour/depth skew": f"{skew_ms:.0f} ms",
            },
        )
        with lock:
            latest["jpeg"] = frame.jpeg
            latest["stats"] = frame.stats
            if frame.scene is not None:
                latest["scene"] = frame.scene

        if recorder is not None:
            if info["color"] is not None:
                recorder.write_calibration(info["color"], info["depth"], args.stride)
            recorder.write(frame.panel, color, depth_mm, frame.stats)


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
