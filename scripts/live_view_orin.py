#!/usr/bin/env python3
"""Serve the model's live predictions over the network as MJPEG.

    python3 live_view_orin.py hydranet_fp16.engine --port 8080

Then open http://<orin-ip>:8080/ in a browser. No client to install: MJPEG over
multipart/x-mixed-replace is what every browser already does for webcam streams.

Shows what the model actually sees, which a metric cannot. A falling loss is entirely
compatible with calling a whole floor a wall, and this is where that shows up.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_camera_orin import MEAN, RAW_INPUT, STD, Cudart, letterbox

# Copied rather than imported: this file runs on a Jetson with numpy and TensorRT and
# no syncai_hydranet installed, which is also why MEAN, STD and letterbox live in
# bench_camera_orin.
#
# It must stay equal to utils.visualize's indoor palette, and for a long time the
# comment here claimed it did while the package held the *off-road* palette instead --
# so a live frame and a val_pred grid coloured the same class differently, and the
# claim was the reason nobody checked. tests/test_orin_standalone_copies.py checks it
# now, on the dev box where both are importable.
TRAV_COLORS = np.array([[220, 40, 40], [250, 200, 40], [40, 200, 80]], dtype=np.uint8)
TERRAIN_COLORS = np.array(
    [
        [0, 0, 0], [139, 90, 43], [60, 180, 75], [0, 100, 0], [128, 128, 128],
        [255, 130, 0], [190, 100, 200], [130, 130, 130], [70, 200, 235], [245, 220, 60],
        [230, 80, 80], [255, 100, 180],
    ],
    dtype=np.uint8,
)  # fmt: skip
TRAV_NAMES = ("blocked", "caution", "go")

# COCO's 80 categories in sorted-category-id order, which is the order
# CocoDetDataset assigns contiguous labels in. Index i here is class i out of the head.
COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic-light",
    "fire-hydrant",
    "stop-sign",
    "parking-meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports-ball",
    "kite",
    "baseball-bat",
    "baseball-glove",
    "skateboard",
    "surfboard",
    "tennis-racket",
    "bottle",
    "wine-glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot-dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted-plant",
    "bed",
    "dining-table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell-phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy-bear",
    "hair-drier",
    "toothbrush",
]
STRIDES = (8, 16, 32, 64, 128)


def decode_detections(host, strides=STRIDES, score_thr=0.35, nms_thr=0.5, top_k=50):
    """FCOS decode in numpy: grid centres, distances to corners, then NMS.

    The engine emits raw logits per level because the graph deliberately excludes NMS and
    anything with dynamic shape, which is what lets TensorRT convert it in one piece. That
    means this arithmetic is the host's job, and it is also 43% of the frame time -- almost
    all of it the sigmoid over 80 classes at every one of 6,825 positions.
    """
    boxes, scores, labels = [], [], []
    for i, stride in enumerate(strides):
        cls = host[f"det_cls_p{i + 3}"][0]  # [C, h, w] logits
        reg = host[f"det_reg_p{i + 3}"][0]  # [4, h, w] pixel distances, already exp'd
        ctr = host[f"det_ctr_p{i + 3}"][0]  # [1, h, w] logit
        conf = 1.0 / (1.0 + np.exp(-cls)) * (1.0 / (1.0 + np.exp(-ctr)))
        best = conf.argmax(0)
        peak = conf.max(0)
        ys, xs = np.nonzero(peak > score_thr)
        if not len(ys):
            continue
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        l, t, r, b = (reg[k, ys, xs] for k in range(4))
        boxes.append(np.stack([cx - l, cy - t, cx + r, cy + b], axis=1))
        scores.append(peak[ys, xs])
        labels.append(best[ys, xs])
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    order = scores.argsort()[::-1][:top_k]
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    keep = []
    idx = np.arange(len(boxes))
    while len(idx):
        keep.append(idx[0])
        if len(idx) == 1:
            break
        a, rest = boxes[idx[0]], boxes[idx[1:]]
        xx1 = np.maximum(a[0], rest[:, 0])
        yy1 = np.maximum(a[1], rest[:, 1])
        xx2 = np.minimum(a[2], rest[:, 2])
        yy2 = np.minimum(a[3], rest[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_r = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = inter / np.maximum(area_a + area_r - inter, 1e-6)
        idx = idx[1:][iou < nms_thr]
    keep = np.array(keep, dtype=int)
    return boxes[keep], scores[keep], labels[keep]


latest_jpeg: bytes | None = None
latest_lock = threading.Lock()

PAGE = b"""<!doctype html><meta charset=utf-8><title>HydraNet live</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui;text-align:center}
img{max-width:100%;height:auto;image-rendering:pixelated}
p{opacity:.6;margin:.6rem}</style>
<img src="/stream">
<p>left: traversability + detections &middot; right: terrain</p>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # the default logger writes a line per frame
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with latest_lock:
                    buf = latest_jpeg
                if buf is None:
                    time.sleep(0.02)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(buf)}\r\n\r\n".encode())
                self.wfile.write(buf)
                self.wfile.write(b"\r\n")
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser tab closed


def colourise(ids, palette):
    return palette[np.clip(ids, 0, len(palette) - 1)]


def capture_loop(args):
    global latest_jpeg
    import cv2
    import tensorrt as trt

    cuda = Cudart()
    logger = trt.Logger(trt.Logger.ERROR)
    with Path(args.engine).open("rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    # The engine's input name states whether normalisation is inside the graph. Reading
    # it beats assuming it: an engine built before that change wants the mean subtracted
    # here, one built after wants raw pixels, and doing the wrong one is silent.
    graph_normalises = in_name == RAW_INPUT
    print(f"input '{in_name}': graph normalises = {graph_normalises}", flush=True)
    outs = [n for n in names if engine.get_tensor_mode(n) != trt.TensorIOMode.INPUT]

    host, dev = {}, {}
    for n in names:
        arr = np.empty(
            tuple(ctx.get_tensor_shape(n)), dtype=trt.nptype(engine.get_tensor_dtype(n))
        )
        host[n] = np.ascontiguousarray(arr)
        dev[n] = cuda.malloc(host[n].nbytes)
        ctx.set_tensor_address(n, dev[n].value)
    stream = cuda.stream_create()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.device}")

    fps, last = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        lb = letterbox(frame, args.height, args.width)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32)
        if not graph_normalises:
            rgb = (rgb / 255.0 - MEAN) / STD
        np.copyto(host[in_name], rgb.transpose(2, 0, 1)[None].astype(host[in_name].dtype))

        cuda.memcpy_htod(dev[in_name], host[in_name], stream)
        ctx.execute_async_v3(stream.value)
        for n in outs:
            cuda.memcpy_dtoh(host[n], dev[n], stream)
        cuda.sync(stream)

        trav = colourise(host["traversability"][0].argmax(0), TRAV_COLORS)
        terr = colourise(host["terrain"][0].argmax(0), TERRAIN_COLORS)
        # Blend rather than replace: the overlay is only readable against the frame it
        # describes, and a solid mask hides exactly the detail you are checking.
        left = cv2.addWeighted(lb, 0.45, cv2.cvtColor(trav, cv2.COLOR_RGB2BGR), 0.55, 0)
        right = cv2.addWeighted(lb, 0.45, cv2.cvtColor(terr, cv2.COLOR_RGB2BGR), 0.55, 0)

        # Per-class share of the traversability map: a quick read on whether the model has
        # collapsed to one class, which the picture alone can hide.
        ids = host["traversability"][0].argmax(0)
        shares = [100.0 * float((ids == c).mean()) for c in range(3)]
        label = "  ".join(f"{n} {s:.0f}%" for n, s in zip(TRAV_NAMES, shares, strict=True))

        # Boxes go on the traversability pane, where a person or a chair is the thing a
        # planner has to react to. Drawn after blending so the outline stays readable.
        dets = decode_detections(host, score_thr=args.score)
        for (x1, y1, x2, y2), sc, cl in zip(*dets, strict=True):
            p1 = (int(np.clip(x1, 0, args.width - 1)), int(np.clip(y1, 0, args.height - 1)))
            p2 = (int(np.clip(x2, 0, args.width - 1)), int(np.clip(y2, 0, args.height - 1)))
            name = COCO_NAMES[int(cl)] if int(cl) < len(COCO_NAMES) else str(int(cl))
            cv2.rectangle(left, p1, p2, (255, 255, 255), 2)
            cv2.putText(
                left,
                f"{name} {sc:.2f}",
                (p1[0], max(p1[1] - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                left,
                f"{name} {sc:.2f}",
                (p1[0], max(p1[1] - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        canvas = np.hstack([left, right])
        cv2.putText(
            canvas,
            f"{fps:4.1f} FPS   {label}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{fps:4.1f} FPS   {label}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        ok, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        if ok:
            with latest_lock:
                globals()["latest_jpeg"] = enc.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--score", type=float, default=0.35, help="detection score threshold")
    args = ap.parse_args()

    threading.Thread(target=capture_loop, args=(args,), daemon=True).start()
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"serving on http://{args.bind}:{args.port}/  (ctrl-c to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
