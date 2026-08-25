"""Pose on real store footage: boxes from the detection head, skeletons from the P3 head.

`eval_student.py` answers "how far from the teacher, in pixels". This answers the other
question, the one a person actually asks of a pose model -- does it look right on our
own cameras -- and it is what gate 3's "`reach_to_shelf` and `crouch` fire correctly on
a watched clip" has to be judged against.

One forward pass per frame: `forward` once, the detection head's `decode_into` for
boxes, `pose.decode_boxes` over the same output's heatmaps for the keypoints. Calling
`predict` and `forward` separately would run the backbone twice for one picture.

Usage:
  uv run python tools/pose/pose_overlay.py <camera> --checkpoint PATH
      [--frames N] [--fps 5] [--stills K] [--video]
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from syncai_hydranet.analytics import Tracker
from syncai_hydranet.analytics.events import pose_posture_events
from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
KP_MIN_CONF = 0.2
SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (0, 1), (0, 2), (1, 3), (2, 4),
)  # fmt: skip
LIMB_COLOR = (70, 230, 255)
JOINT_COLOR = (255, 215, 60)
BOX_COLOR = (120, 220, 120)


def draw_person(d: ImageDraw.ImageDraw, kp: np.ndarray, box_h: float) -> int:
    """Skeleton for one person. Returns how many joints cleared the confidence floor.

    Marker size scales with the person, not with the frame. A fixed radius drew 5 px
    dots on a 140 px shopper and the joints covered the limbs they were supposed to
    locate -- the overlay hid the thing it exists to show.
    """
    ok = kp[:, 2] >= KP_MIN_CONF
    w = max(1, round(box_h / 90))
    r = max(1.2, box_h / 70)
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            d.line([tuple(kp[a, :2]), tuple(kp[b, :2])], fill=LIMB_COLOR, width=w)
    for k in range(17):
        if ok[k]:
            x, y = kp[k, :2]
            d.ellipse([x - r, y - r, x + r, y + r], fill=JOINT_COLOR)
    return int(ok.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/hydranet_retail_pose01.yaml")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--stills", type=int, default=1, help="how many frames to also save as png")
    ap.add_argument(
        "--still-frames",
        default="",
        help="comma-separated frame indices to save instead of the first --stills. "
        "An event names the frames it fired on; this is how you look at them without "
        "rendering the other 880",
    )
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--tag", default="pose")
    ap.add_argument(
        "--events",
        action="store_true",
        help="track the detections, feed the keypoints through, and run "
        "`pose_posture_events`. Until the pose head existed nothing filled "
        "`Track.keypoints` and every pose event refused itself; this is that wire, "
        "end to end on real footage",
    )
    args = ap.parse_args()

    clip = (
        Path(args.clip)
        if args.clip
        else sorted((ROOT / "datasets/studioa_clips" / args.camera).glob("archive_*11*.mp4"))[0]
    )
    cfg = load_config(args.config, validate=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), "ema"))
    size = cfg["data"]["input_size"]
    classes = list(cfg["model"]["heads"]["detection"]["classes"])
    person = classes.index("person")

    out_w, out_h = 1280, 720
    # same rule as demo_video: every render keeps its own file and
    # `<tag>_<camera>.mp4` points at the newest. A fixed name destroys the version
    # someone just approved, and leaves no way to answer "which one is current".
    stamp = time.strftime("%Y%m%d-%H%M%S")
    final = ROOT / f"assets/{args.tag}_{args.camera}_{stamp}.mp4"
    latest = ROOT / f"assets/{args.tag}_{args.camera}.mp4"
    enc = None
    if args.video:
        part = final.with_suffix(".mp4.part")
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{out_w}x{out_h}", "-framerate", str(args.fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
             "-f", "mp4", str(part)],
            stdin=subprocess.PIPE,
        )  # fmt: skip

    saved = n = n_people = n_joints = 0
    tracker = Tracker() if args.events else None
    want = {int(x) for x in args.still_frames.split(",") if x.strip()}
    for frame in decode_frames(str(clip), 1920, 1080, args.fps):
        if n >= args.frames:
            break
        img = Image.fromarray(frame)
        x, _canvas, region = preprocess(img, size)
        with torch.no_grad():
            # predict runs detection then pose over ONE forward pass, and the pose head's
            # adapter decodes against the boxes detection just produced -- the ordering in
            # HydraNet.heads() is the data flow, so this is the production path, not a
            # reconstruction of it
            res = model.predict(x.to(device), score_thr=args.score_thr)
        det = res.get("detection", [{}])[0]
        pose_rows = res.get("pose", [None])[0]
        view = img.resize((out_w, out_h))
        d = ImageDraw.Draw(view)
        x0, y0, cw, _ch = region
        to_view = out_w / cw  # letterboxed input px -> the 1280-wide view
        people = 0
        if det and len(det.get("boxes", [])) and pose_rows is not None:
            labels = det["labels"].cpu().numpy()
            keep = labels == person
            # the pose rows are one per decoded box, in box order, so the person mask
            # indexes both or neither
            b = det["boxes"].cpu().numpy()[keep]
            kps = pose_rows.cpu().numpy()[keep]
            if len(b):
                # source pixels for the tracker and the events: `Track.keypoints` is
                # specified in image pixels, and a view-space copy would make every
                # threshold in `events/pose.py` depend on the panel size
                src_scale = 1920.0 / cw
                boxes_src = (b - np.array([x0, y0, x0, y0])) * src_scale
                kps_src = kps.copy()
                kps_src[:, :, 0] = (kps_src[:, :, 0] - x0) * src_scale
                kps_src[:, :, 1] = (kps_src[:, :, 1] - y0) * src_scale
                if tracker is not None:
                    tracker.update(boxes_src, n, keypoints=kps_src)
                for bi, kp in zip(b, kps, strict=True):
                    bb = (bi - np.array([x0, y0, x0, y0])) * to_view
                    d.rectangle(list(bb), outline=BOX_COLOR, width=2)
                    kp = kp.copy()
                    kp[:, 0] = (kp[:, 0] - x0) * to_view
                    kp[:, 1] = (kp[:, 1] - y0) * to_view
                    n_joints += draw_person(d, kp, float(bb[3] - bb[1]))
                    people += 1
        if tracker is not None and not people:
            tracker.update(np.zeros((0, 4)), n, keypoints=np.zeros((0, 17, 3)))
        n_people += people
        d.rectangle([0, out_h - 18, out_w, out_h], fill=(0, 0, 0))
        d.text(
            (6, out_h - 16),
            f"{args.camera}  pose P3 head + detection head, one forward pass  "
            f"{Path(args.checkpoint).name}  thr={args.score_thr}  frame {n}  people {people}",
            fill=(255, 255, 255),
        )
        if enc:
            enc.stdin.write(np.asarray(view, np.uint8).tobytes())
        if (n in want) if want else (saved < args.stills):
            view.save(ROOT / f"assets/{args.tag}_{args.camera}_{stamp}_{n:04d}.png")
            saved += 1
        if n % 100 == 0:
            print(f"  {n}/{args.frames}", flush=True)
        n += 1

    if enc:
        enc.stdin.close()
        if enc.wait() == 0:
            part.replace(final)
            latest.write_bytes(final.read_bytes())
            print(f"wrote {final}")
            print(f"  newest also at {latest}")
    print(f"{n} frames, {n_people} person detections, {n_joints} joints above {KP_MIN_CONF}")
    if tracker is not None:
        tracks = [t for t in tracker.tracks + tracker.retired if t.confirmed]
        print(f"{len(tracks)} confirmed tracks, all carrying one keypoint set per frame")
        events = pose_posture_events(tracks, args.fps, args.camera)
        if not events:
            print("no fall or crouch cleared its sustained threshold on this clip")
        for e in events:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
