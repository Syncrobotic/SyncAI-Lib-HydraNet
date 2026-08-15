"""Run HydraNet on this robot's live RealSense stream, and gate walkable on metric depth.

    source /opt/ros/humble/setup.bash
    python3 retail_probe_ros.py --weights det60_ema.pt --config configs/hydranet_indoor.yaml \
        --frames 40 --save 6 --range 5.0 --out shots/

Subscribes rather than opening /dev/video: the camera belongs to a running robot stack
(realsense2_camera_node, nav2, motion), and taking the device would stop the robot. The
node already publishes depth registered to the colour frame, which is the harder half of
the job anyway.

Two questions at once. What does the indoor model predict from a robot's viewpoint on a
real camera -- ADE20K is human-height web photography and the last live test had a ceiling
coming back 25% "go". And what does "walkable within 5 m" look like when the range test is
post-processing over metric depth rather than a class the network had to learn.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from PIL import Image, ImageDraw
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg

HERE = Path(__file__).resolve().parent
# In the repo the package is at ../src; on a robot the deploy copies it to ./src next to
# this script. Accept both rather than making the caller care which layout they are in.
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import (  # noqa: E402
    TRAV_COLORS,
    crop_box,
    overlay,
    preprocess,
)

GO = 2
COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
REACH_COLORS = np.array([[0, 0, 0], [40, 220, 90]], dtype=np.uint8)


def decode(msg: ImageMsg) -> np.ndarray:
    """sensor_msgs/Image -> ndarray, honouring `step` rather than assuming packed rows."""
    if msg.encoding in ("bgr8", "rgb8"):
        buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        arr = buf[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return arr[:, :, ::-1] if msg.encoding == "bgr8" else arr
    if msg.encoding in ("16UC1", "mono16"):
        buf = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)
        return buf[:, : msg.width]
    raise ValueError(f"unhandled encoding {msg.encoding}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--save", type=int, default=6)
    ap.add_argument("--range", type=float, default=5.0, help="metres")
    ap.add_argument("--score-thr", type=float, default=0.30)
    ap.add_argument("--out", default="shots")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pretrained=false: the weights come from the checkpoint, and this board should not
    # reach out to download a backbone it is about to overwrite.
    cfg = load_config(args.config, ["model.backbone.pretrained=false"])
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    missing, unexpected = model.load_state_dict(
        torch.load(args.weights, map_location=device), strict=False
    )
    print(f"loaded weights: {len(missing)} missing, {len(unexpected)} unexpected keys")
    size = cfg["data"]["input_size"]

    rclpy.init()
    node = rclpy.create_node("hydranet_probe")
    latest = {"color": None, "depth": None}
    node.create_subscription(
        ImageMsg, COLOR_TOPIC, lambda m: latest.__setitem__("color", m), qos_profile_sensor_data
    )
    node.create_subscription(
        ImageMsg, DEPTH_TOPIC, lambda m: latest.__setitem__("depth", m), qos_profile_sensor_data
    )

    deadline = time.time() + 20
    while (latest["color"] is None or latest["depth"] is None) and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if latest["color"] is None or latest["depth"] is None:
        print("no frames on the camera topics within 20 s")
        return 1
    print(f"streaming {latest['color'].width}x{latest['color'].height} on {device}")

    t = {k: [] for k in ("wait", "preprocess", "inference", "postprocess", "total")}
    saved, seen = 0, 0
    rows = []
    while seen < args.frames:
        f0 = time.perf_counter()
        stamp = latest["color"].header.stamp
        while latest["color"].header.stamp == stamp:
            rclpy.spin_once(node, timeout_sec=0.5)
        color_msg, depth_msg = latest["color"], latest["depth"]
        color, depth_mm = decode(color_msg), decode(depth_msg)
        depth_m = depth_mm.astype(np.float32) * 0.001  # RealSense ROS publishes mm
        t1 = time.perf_counter()

        x, canvas, region = preprocess(Image.fromarray(color), size)
        x = x.to(device)
        t2 = time.perf_counter()

        with torch.no_grad():
            result = model.predict(x, score_thr=args.score_thr)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        trav = crop_box(result["traversability"][0].cpu().numpy(), region)
        x0, y0, cw, ch = region
        vis_img = canvas.crop((x0, y0, x0 + cw, y0 + ch))

        # The whole point: walkable comes from the network, range comes from the sensor,
        # and they meet here rather than inside the weights.
        trav_full = np.asarray(
            Image.fromarray(trav.astype(np.uint8)).resize(
                (color.shape[1], color.shape[0]), Image.NEAREST
            )
        )
        valid = depth_m > 0
        in_range = valid & (depth_m <= args.range)
        go = trav_full == GO
        reachable = go & in_range
        t4 = time.perf_counter()

        for k, v in zip(t, (t1 - f0, t2 - t1, t3 - t2, t4 - t3, t4 - f0), strict=True):
            t[k].append(1000 * v)

        # Where the model says "go" but the sensor sees nothing at all: glass, mirrors,
        # a polished floor reflecting the ceiling. Worth counting separately -- it is the
        # exact failure this deployment is most afraid of.
        go_no_depth = float((go & ~valid).sum()) / max(go.sum(), 1)
        rows.append(
            (100 * go.mean(), 100 * reachable.mean(), 100 * valid.mean(), 100 * go_no_depth)
        )

        if saved < args.save and seen % max(args.frames // max(args.save, 1), 1) == 0:
            left = overlay(vis_img, trav, TRAV_COLORS)
            det = result.get("detection", [{}])[0]
            if det and len(det.get("boxes", [])):
                draw = ImageDraw.Draw(left)
                boxes, scores = det["boxes"].cpu().numpy(), det["scores"].cpu().numpy()
                for box, score in zip(boxes, scores, strict=True):
                    bx = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
                    draw.rectangle(bx, outline=(255, 255, 0), width=2)
                    draw.text((bx[0] + 2, bx[1] + 2), f"{score:.2f}", fill=(255, 255, 0))
            right_mask = np.asarray(
                Image.fromarray(reachable.astype(np.uint8)).resize(vis_img.size, Image.NEAREST)
            )
            right = overlay(vis_img, right_mask, REACH_COLORS)
            pair = Image.new("RGB", (left.width * 2 + 8, left.height), (16, 16, 16))
            pair.paste(left, (0, 0))
            pair.paste(right, (left.width + 8, 0))
            pair.save(out_dir / f"frame_{seen:03d}.jpg", quality=88)
            saved += 1
            print(
                f"frame {seen:3d}  go {rows[-1][0]:5.1f}%  go&<{args.range:.0f}m "
                f"{rows[-1][1]:5.1f}%  depth valid {rows[-1][2]:5.1f}%  "
                f"go-with-no-depth {rows[-1][3]:5.1f}%"
            )
        seen += 1

    node.destroy_node()
    rclpy.shutdown()

    a = np.array(rows)
    print(f"\nover {len(a)} frames")
    print(f"  go                    {a[:, 0].mean():5.1f}%  (of the frame)")
    print(f"  go and within {args.range:.0f} m     {a[:, 1].mean():5.1f}%")
    print(f"  depth valid           {a[:, 2].mean():5.1f}%")
    print(
        f"  go where depth is invalid  {a[:, 3].mean():5.1f}%  <- glass/mirror/reflection risk"
    )
    print("\nstage           mean ms   share")
    total = np.mean(t["total"])
    for k in ("wait", "preprocess", "inference", "postprocess"):
        m = np.mean(t[k])
        print(f"{k:<14} {m:8.2f}  {100 * m / total:5.1f}%")
    print(f"{'total':<14} {total:8.2f}  {1000 / total:5.1f} FPS (camera publishes at 15)")
    print(f"wrote {saved} frames to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
