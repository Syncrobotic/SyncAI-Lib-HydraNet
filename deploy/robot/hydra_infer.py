#!/usr/bin/env python3
# HydraNet live inference on the Lite3 NPU.
# Pulls camera frames from the robot's RTSP stream, runs the multi-task RKNN model
# (traversability + terrain + FCOS detection), overlays results, and writes the latest
# annotated JPEG + stats to /dev/shm/hydra for the dashboard to serve.
# Pure numpy + PIL (no torch). FCOS decode ported from models/heads/detection.py.
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw
from rknnlite.api import RKNNLite

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "syncai_geo", ".."))
import contextlib

from syncai_geo.geometry import bev3d
from syncai_geo.geometry.bev import BevGrid, free_space_map, project_mask
from syncai_geo.geometry.bev import scene as bev_scene
from syncai_geo.geometry.ground import Camera, GroundPlane

VFOV = float(os.environ.get("HYDRA_VFOV", "60"))
CAM_HEIGHT = float(os.environ.get("HYDRA_CAMH", "0.35"))
PITCH_DEG = float(os.environ.get("HYDRA_PITCH", "18"))
BEV_PW, BEV_PH = 300, 380
TERRAIN_NAMES = (
    "void",
    "floor_hard",
    "floor_soft",
    "floor_metal",
    "wet_slippery",
    "stairs",
    "threshold_ramp",
    "wall",
    "glass",
    "door",
    "obstacle_furniture",
    "person",
)

MODEL = os.environ.get("HYDRA_MODEL", "/home/paul/playground/model/hydranet_joint_coco10.rknn")
RTSP = "rtsp://127.0.0.1:8554/test"
W, H = (
    int(os.environ.get("HYDRA_W", "640")),
    int(os.environ.get("HYDRA_H", "512")),
)  # model input (WxH)
OUT_DIR = "/dev/shm/hydra"
SCORE_THR = 0.35
NMS_THR = 0.6
MAX_DET = 60
STRIDES = [8, 16, 32, 64, 128]
NC = 80

TRAV_COLORS = np.array(
    [[220, 40, 40], [250, 200, 40], [40, 200, 80]], np.uint8
)  # blocked,caution,go
TERRAIN_COLORS = np.array(
    [
        [0, 0, 0],
        [139, 90, 43],
        [60, 180, 75],
        [0, 100, 0],
        [128, 128, 128],
        [255, 130, 0],
        [190, 100, 200],
        [130, 130, 130],
        [70, 200, 235],
        [245, 220, 60],
        [230, 80, 80],
        [255, 100, 180],
    ],
    np.uint8,
)
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
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
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
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
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
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_points(shapes):
    out = []
    for i, (h, w) in enumerate(shapes):
        s = STRIDES[i]
        xs = (np.arange(w) + 0.5) * s
        ys = (np.arange(h) + 0.5) * s
        xx, yy = np.meshgrid(xs, ys)  # (h,w)
        out.append(np.stack([xx.reshape(-1), yy.reshape(-1)], -1))
    return np.concatenate(out, 0)


def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def decode_det(cls_l, reg_l, ctr_l):
    shapes = [c.shape[-2:] for c in cls_l]
    pts = make_points(shapes)
    cls = np.concatenate([np.transpose(c[0], (1, 2, 0)).reshape(-1, NC) for c in cls_l], 0)
    reg = np.concatenate([np.transpose(r[0], (1, 2, 0)).reshape(-1, 4) for r in reg_l], 0)
    ctr = np.concatenate([c[0].reshape(-1) for c in ctr_l], 0)
    scores = sigmoid(cls) * sigmoid(ctr)[:, None]
    label = scores.argmax(1)
    score = scores[np.arange(len(scores)), label]
    m = score > SCORE_THR
    pts, reg, score, label = pts[m], reg[m], score[m], label[m]
    boxes = np.stack(
        [
            pts[:, 0] - reg[:, 0],
            pts[:, 1] - reg[:, 1],
            pts[:, 0] + reg[:, 2],
            pts[:, 1] + reg[:, 3],
        ],
        1,
    )
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, W)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, H)
    out = []
    for cid in np.unique(label):
        idx = np.where(label == cid)[0]
        k = nms(boxes[idx], score[idx], NMS_THR)
        for j in k:
            out.append((boxes[idx][j], score[idx][j], int(cid)))
    out.sort(key=lambda t: -t[1])
    return out[:MAX_DET]


def blend_mask(rgb, cls_map, palette, alpha, only=None):
    color = palette[cls_map]
    if only is not None:
        m = np.isin(cls_map, only)[..., None]
        return np.where(m, (rgb * (1 - alpha) + color * alpha).astype(np.uint8), rgb)
    return (rgb * (1 - alpha) + color * alpha).astype(np.uint8)


_GRID = BevGrid()


def make_bev(trav, terr, dets):
    """Project-native 3D BEV: geometry/bev.scene builds the metric floor + objects from a
    ground-plane assumption; geometry/bev3d.render draws it as a perspective room panel.
    Mono ground-plane -> scale is an assumption (a depth camera makes it metric-exact,
    see geometry/depth_scene.py). Camera params are env-tunable."""
    cam = Camera.from_vfov(trav.shape[0], trav.shape[1], VFOV)
    # Per-frame ground plane: PITCH_DEG is the fixed camera-mount tilt; the robot's live
    # body attitude (IMU roll/pitch, deg, published by hydra_dash from 0x0901/0x0906) is
    # added on top so the BEV tracks the gait instead of asserting the robot never banks.
    b_pitch, b_roll = 0.0, 0.0
    try:
        with open("/dev/shm/hydra/robot_state.json") as _fh:
            _st = json.load(_fh)
        if time.time() - _st.get("ts", 0) < 1.0:
            b_pitch, b_roll = float(_st.get("pitch", 0.0)), float(_st.get("roll", 0.0))
    except Exception:
        pass
    plane = GroundPlane(
        height=CAM_HEIGHT,
        pitch=math.radians(PITCH_DEG + b_pitch),
        roll=math.radians(b_roll),
    )
    boxes = labels = scores = None
    if dets:
        boxes = np.array([d[0] for d in dets], dtype=np.float32)
        scores = np.array([d[1] for d in dets], dtype=np.float32)
        labels = np.array([d[2] for d in dets], dtype=np.int64)
    payload, bev = bev_scene(
        trav.astype(np.int64),
        cam,
        plane,
        grid=_GRID,
        boxes=boxes,
        labels=labels,
        scores=scores,
        names=dict(enumerate(COCO_NAMES)),
    )
    bev = free_space_map(np.asarray(bev), _GRID)
    terrain_bev = project_mask(terr.astype(np.int64), cam, plane, _GRID)
    return bev3d.render(
        bev,
        terrain_bev,
        _GRID,
        payload["objects"],
        (BEV_PW, BEV_PH),
        trav_colors=TRAV_COLORS,
        terrain_colors=TERRAIN_COLORS,
        class_names=TERRAIN_NAMES,
        bg=(7, 9, 13),
        supersample=1,
    )


FRAME_BYTES = W * H * 3


class FrameReader(threading.Thread):
    """Decode the RTSP H264 in software (mpp HW decoder is busy with the encoder) and
    keep only the LATEST frame, so inference never lags behind a growing pipe buffer."""

    def __init__(self):
        super().__init__(daemon=True)
        self._latest = None
        self._lock = threading.Lock()
        self._run = True

    def _spawn(self):
        cmd = (
            "gst-launch-1.0 -q rtspsrc location=%s latency=50 protocols=tcp ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! "
            "video/x-raw,format=RGB,width=%d,height=%d ! fdsink fd=1" % (RTSP, W, H)
        )
        return subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )

    def run(self):
        while self._run:
            p = self._spawn()
            buf = b""
            while self._run:
                need = FRAME_BYTES - len(buf)
                chunk = p.stdout.read(need)
                if not chunk:
                    break
                buf += chunk
                if len(buf) >= FRAME_BYTES:
                    fr = np.frombuffer(buf[:FRAME_BYTES], np.uint8).reshape(H, W, 3)
                    with self._lock:
                        self._latest = fr
                    buf = buf[FRAME_BYTES:]
            with contextlib.suppress(Exception):
                p.kill()
            time.sleep(0.5)

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r = RKNNLite()
    r.load_rknn(MODEL)
    r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    reader = FrameReader()
    reader.start()
    fps = 0.0
    tprev = time.time()
    frames = 0
    view = os.environ.get("HYDRA_VIEW", "trav")  # trav | terrain | both | none
    BEV_EVERY = int(os.environ.get("HYDRA_BEV_EVERY", "3"))
    while True:
        rgb = reader.latest()
        if rgb is None:
            time.sleep(0.1)
            continue
        rgb = np.ascontiguousarray(rgb)
        t0 = time.time()
        out = r.inference(inputs=[rgb[None]], data_format="nhwc")
        infer_ms = (time.time() - t0) * 1000
        # classify outputs by shape (export order is not stable across configs):
        # full-res seg: 3ch=traversability, 12ch=terrain; det levels: 80=cls,4=reg,1=ctr
        trav = terr = None
        cls_by = []
        reg_by = []
        ctr_by = []
        for o in out:
            c = o.shape[1]
            h = o.shape[2]
            if h == H and c == 3:
                trav = o[0].argmax(0).astype(np.int32)
            elif h == H and c == 12:
                terr = o[0].argmax(0).astype(np.int32)
            elif c == 80:
                cls_by.append(o)
            elif c == 4:
                reg_by.append(o)
            elif c == 1:
                ctr_by.append(o)

        def key(x):
            return -x.shape[2]

        dets = decode_det(
            sorted(cls_by, key=key), sorted(reg_by, key=key), sorted(ctr_by, key=key)
        )
        if trav is None:
            trav = np.zeros((H, W), np.int32)
        if terr is None:
            terr = np.zeros((H, W), np.int32)
        # overlay
        vis = rgb.copy()
        if view in ("trav", "both"):
            vis = blend_mask(vis, trav, TRAV_COLORS, 0.40)
        if view in ("terrain", "both"):
            vis = blend_mask(vis, terr, TERRAIN_COLORS, 0.40, only=list(range(1, 12)))
        im = Image.fromarray(vis)
        dr = ImageDraw.Draw(im)
        counts = {}
        for box, sc, cid in dets:
            nm = COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid)
            counts[nm] = counts.get(nm, 0) + 1
            dr.rectangle(
                [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                outline=(255, 255, 0),
                width=2,
            )
            dr.text((int(box[0]) + 2, int(box[1]) + 1), f"{nm} {sc:.2f}", fill=(255, 255, 0))
        # legend
        dr.rectangle([0, 0, W, 16], fill=(0, 0, 0))
        dr.text((4, 3), "go", fill=(40, 200, 80))
        dr.text((28, 3), "caution", fill=(250, 200, 40))
        dr.text((84, 3), "blocked", fill=(220, 40, 40))
        dr.text((150, 3), "det:%d  %.1f FPS" % (len(dets), fps), fill=(255, 255, 255))
        # write jpeg atomically
        tmp = OUT_DIR + "/.frame.jpg"
        im.save(tmp, "JPEG", quality=80)
        os.replace(tmp, OUT_DIR + "/frame.jpg")
        if frames % BEV_EVERY == 0:
            try:
                bev = make_bev(trav, terr, dets)
                tb = OUT_DIR + "/.bev.jpg"
                bev.save(tb, "JPEG", quality=80)
                os.replace(tb, OUT_DIR + "/bev.jpg")
            except Exception:
                pass
        frames += 1
        now = time.time()
        if now - tprev >= 1.0:
            fps = frames / (now - tprev)
            frames = 0
            tprev = now
        # stats
        trav_px = np.bincount(trav.reshape(-1), minlength=3) / trav.size
        stats = {
            "fps": round(fps, 1),
            "infer_ms": round(infer_ms, 1),
            "n_det": len(dets),
            "det_counts": counts,
            "view": view,
            "trav": {
                "blocked": round(float(trav_px[0]), 3),
                "caution": round(float(trav_px[1]), 3),
                "go": round(float(trav_px[2]), 3),
            },
            "ts": now,
        }
        tmp = OUT_DIR + "/.stats.json"
        open(tmp, "w").write(json.dumps(stats))
        os.replace(tmp, OUT_DIR + "/stats.json")


if __name__ == "__main__":
    main()
