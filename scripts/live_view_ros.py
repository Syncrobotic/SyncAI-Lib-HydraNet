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

from live_view_orin import COCO_NAMES  # noqa: E402

from syncai_hydranet.config import load_config  # noqa: E402  # isort: skip
from syncai_hydranet.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.visualize import (  # noqa: E402
    TERRAIN_COLORS,
    TRAV_COLORS,
    crop_box,
    letterbox,
    overlay,
)

GO = 2
COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"

# reachable / out of range / no depth return
REACH_COLORS = np.array([[0, 0, 0], [40, 220, 90], [250, 200, 40], [230, 60, 230]], np.uint8)

state = {"range": 5.0, "score": 0.30, "view": "both"}
latest = {"jpeg": None, "stats": {}}
lock = threading.Lock()

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


def preprocess(img: Image.Image, size):
    img, region = letterbox(img.convert("RGB"), size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))[None], img, region


def inference_loop(args):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as ImageMsg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, ["model.backbone.pretrained=false"])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    size = cfg["data"]["input_size"]
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

    def stamp_of(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

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
        depth_m = decode(depth_msg)[:: args.stride, :: args.stride].astype(np.float32) * 0.001

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
                (color.shape[1], color.shape[0]), Image.NEAREST
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
            TERRAIN_COLORS if state["view"] == "terrain" else TRAV_COLORS,
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
            Image.fromarray(reach.astype(np.uint8)).resize(vis.size, Image.NEAREST)
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
        with lock:
            latest["jpeg"] = buf.getvalue()
            latest["stats"] = {
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--range", type=float, default=5.0, help="metres")
    ap.add_argument("--score", type=float, default=0.30)
    ap.add_argument(
        "--stride", type=int, default=2, help="subsample the 1280x720 stream before resizing"
    )
    args = ap.parse_args()
    state["range"], state["score"] = args.range, args.score

    threading.Thread(target=inference_loop, args=(args,), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving on http://0.0.0.0:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
