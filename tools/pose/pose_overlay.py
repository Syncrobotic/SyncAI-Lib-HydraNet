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
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from syncai_hydranet.analytics import Tracker
from syncai_hydranet.analytics.events import pose_posture_events, reach_to_shelf_events
from syncai_hydranet.analytics.events.pose import _torso
from syncai_hydranet.analytics.tracker import iou
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.data.video import probe as probe_video
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.serving.camera import BIRTH_REF
from syncai_hydranet.serving.decode import MIN_PERSON_FRACTION, person_pixel_fraction
from syncai_hydranet.shipped import load_model
from syncai_hydranet.utils.visualize import preprocess

# The repo root, derived rather than written out: every one of these 26 tools had it
# as an absolute path, so a second checkout ran against the first one's `runs/` and
# any machine but this one failed at import with a path and no reason. Two levels up
# from `tools/<group>/<tool>.py`, and `tests/test_no_absolute_sys_path.py` keeps it so.
ROOT = Path(__file__).resolve().parents[2]
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
    ap.add_argument("--score-thr", type=float, default=BIRTH_REF)
    ap.add_argument(
        "--dense-confirm",
        action="store_true",
        help="admit low-scoring boxes only where the dense head puts person pixels "
        "under them. The point of pairing it with a low --score-thr: the box head "
        "finds the shoppers behind counters and scores them below 0.35",
    )
    ap.add_argument("--min-person-fraction", type=float, default=MIN_PERSON_FRACTION)
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
        "--events-json",
        default=None,
        help="write the events, and the per-frame box height and keypoint "
        "geometry behind each one, as JSON. A printed repr answers 'did it "
        "fire'; measuring whether it should have fired needs the numbers",
    )
    ap.add_argument(
        "--fall-head-height",
        type=float,
        default=0.80,
        metavar="M",
        help="a fall must bring the head below this many metres above the floor. "
        "Raise it to read what a candidate actually measured -- the event basis "
        "prints the number it reached.",
    )
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
    model, cfg, device = load_model(args.config, args.checkpoint, validate=False)
    size = cfg["data"]["input_size"]
    classes = list(cfg["model"]["heads"]["detection"]["classes"])
    person = classes.index("person")
    terrain_names = list(cfg["data"].get("terrain_classes") or [])
    seg_person = terrain_names.index("person")
    seg_fixture = terrain_names.index("fixture") if "fixture" in terrain_names else None

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

    saved = n = n_people = n_joints = n_unconfirmed = 0
    tracker = Tracker() if args.events else None
    # The dense map per frame, at SOURCE resolution, for `reach_to_shelf_events`.
    # `_require_terrain_in_image_space` refuses a map at the model canvas, and it is
    # right to: the wrist test indexes terrain[y, x] in image pixels, so a 640x1120 map
    # against 1920x1080 keypoints puts almost every wrist off its edge, and the earlier
    # code skipped out-of-bounds wrists -- turning a unit mismatch into "nobody reached".
    # Nearest-neighbour on purpose: these are class ids and interpolating them invents
    # classes that no head ever predicted.
    terrain_frames: list[dict] = []
    # Every frame's person boxes. A posture event in a crowd is the failure
    # `models/heads/pose.py` forecast before any of this ran -- two overlapping boxes
    # stealing each other's peaks inside the intersection -- and the way to tell that
    # from a real fall is whether anybody else's box was on top of this one.
    boxes_by_frame: dict[int, np.ndarray] = {}
    want = {int(x) for x in args.still_frames.split(",") if x.strip()}
    # The clip's own resolution, not 1080p: `data.video.frames` does not scale, so the
    # width and height size the reads on a rawvideo pipe carrying the source. Seven of
    # the fleet's 48 cameras are 704x480 sub-streams, and every 1920 below is a mapping
    # back into *source* pixels -- which `events/pose.py` measures its thresholds in.
    src_w, src_h, _ = probe_video(str(clip))
    for frame in decode_frames(str(clip), src_w, src_h, args.fps):
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
        n_dropped = 0
        if args.dense_confirm and det and len(det.get("boxes", [])):
            # the pose rows are aligned to the detection rows, so the same mask has to
            # index both -- filtering boxes alone would hand every skeleton to the wrong
            # shopper from the first dropped box onwards
            cls_map = res["terrain"][0].cpu().numpy()
            lab_np = det["labels"].cpu().numpy()
            bx_np = det["boxes"].cpu().numpy()
            keep = np.ones(len(lab_np), dtype=bool)
            for i in np.nonzero(lab_np == person)[0]:
                keep[i] = (
                    person_pixel_fraction(bx_np[i], cls_map, seg_person)
                    >= args.min_person_fraction
                )
            n_dropped = int((~keep).sum())
            n_unconfirmed += n_dropped
            kt = torch.from_numpy(keep).to(det["boxes"].device)
            det = {k: v[kt] for k, v in det.items()}
            if pose_rows is not None:
                pose_rows = pose_rows[kt]
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
            scr = det["scores"].cpu().numpy()[keep]
            if len(b):
                # source pixels for the tracker and the events: `Track.keypoints` is
                # specified in image pixels, and a view-space copy would make every
                # threshold in `events/pose.py` depend on the panel size
                src_scale = src_w / cw
                boxes_src = (b - np.array([x0, y0, x0, y0])) * src_scale
                kps_src = kps.copy()
                kps_src[:, :, 0] = (kps_src[:, :, 0] - x0) * src_scale
                kps_src[:, :, 1] = (kps_src[:, :, 1] - y0) * src_scale
                if tracker is not None:
                    # scores as well as keypoints: this is the tool that measured the
                    # threshold sweep, so it is the one whose events have to say what
                    # confidence they were built from (`events.TrackSupport`)
                    tracker.update(boxes_src, n, keypoints=kps_src, scores=scr)
                for bi, kp in zip(b, kps, strict=True):
                    bb = (bi - np.array([x0, y0, x0, y0])) * to_view
                    d.rectangle(list(bb), outline=BOX_COLOR, width=2)
                    kp = kp.copy()
                    kp[:, 0] = (kp[:, 0] - x0) * to_view
                    kp[:, 1] = (kp[:, 1] - y0) * to_view
                    n_joints += draw_person(d, kp, float(bb[3] - bb[1]))
                    people += 1
        if args.events:
            lab_all = (
                det["labels"].cpu().numpy() if det and len(det.get("labels", [])) else None
            )
            if lab_all is not None and (lab_all == person).any():
                b_all = det["boxes"].cpu().numpy()[lab_all == person]
                src_scale = src_w / region[2]
                boxes_by_frame[n] = (
                    b_all - np.array([region[0], region[1], region[0], region[1]])
                ) * src_scale
        if args.events and seg_fixture is not None:
            cls_map = res["terrain"][0].cpu().numpy().astype(np.uint8)
            full = np.asarray(
                Image.fromarray(cls_map).resize((src_w, src_h), Image.NEAREST),
                dtype=np.uint8,
            )
            terrain_frames.append({"frame_index": n, "terrain": full})
        if tracker is not None and not people:
            tracker.update(
                np.zeros((0, 4)), n, keypoints=np.zeros((0, 17, 3)), scores=np.zeros(0)
            )
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
    if args.dense_confirm:
        print(f"  {n_unconfirmed} boxes dropped for want of dense person pixels")
    if tracker is not None:
        tracks = [t for t in tracker.tracks + tracker.retired if t.confirmed]
        print(f"{len(tracks)} confirmed tracks, all carrying one keypoint set per frame")
        # The camera's own commissioned geometry, when it has any: `fall` then requires
        # the head to reach the floor rather than merely the torso to go horizontal. A
        # camera without a camera.json keeps the image-space behaviour and says so in the
        # event's basis -- 40 of this fleet's 48 have no geometry yet.
        cam_json = ROOT / f"runs/commission01/{args.camera}.camera.json"
        cam_file = CameraFile.load(cam_json) if cam_json.exists() else None
        if cam_file is None:
            print(f"no commissioned geometry for {args.camera}: fall stays image-space")
        events = pose_posture_events(
            tracks,
            args.fps,
            args.camera,
            cam_file=cam_file,
            source_size_px=(src_w, src_h),
            fall_head_height_m=args.fall_head_height,
        )
        crowding: dict = {}
        if seg_fixture is not None and terrain_frames:
            # The one output where the retail model and the security model are the same
            # model: the terrain head's `fixture` channel and the pose head's wrist both
            # have to agree before a row exists. Never run on real footage until now --
            # it had tests and no production call site, which is the state gate 3's third
            # condition was actually in.
            reaches = reach_to_shelf_events(
                tracks,
                terrain_frames,
                args.fps,
                args.camera,
                fixture_id=seg_fixture,
                person_id=seg_person,
            )
            print(f"{len(reaches)} reach_to_shelf events")
            events = events + reaches
        for e in events:
            if e.type not in ("fall", "crouch"):
                continue
            t = next((x for x in tracks if x.track_id in e.track_ids), None)
            worst = 0.0
            for fi, box in zip(t.frames, t.boxes, strict=True) if t else []:
                if not (e.frame_start <= fi <= e.frame_end):
                    continue
                others = boxes_by_frame.get(fi)
                if others is None or len(others) < 2:
                    continue
                ov = iou(np.asarray(box)[None], others)[0]
                ov.sort()
                worst = max(worst, float(ov[-2]))  # the best overlap that is not itself
            crowding[(e.type, e.frame_start)] = round(worst, 3)
            # The box's own trajectory, because the other candidate mechanism is
            # occlusion rather than crowding: a shopper walking behind a counter has
            # their box truncated from below, which `_shrank` reads as the collapse a
            # real fall produces. The two are told apart by where the box bottom sits.
            if t:
                spans = [
                    (fi, [round(float(v), 0) for v in b])
                    for fi, b in zip(t.frames, t.boxes, strict=True)
                    if e.frame_start - 6 <= fi <= e.frame_end
                ]
                for fi, b in spans:
                    mark = "*" if e.frame_start <= fi <= e.frame_end else " "
                    print(f"      {mark}f{fi:>4} box {b} h={b[3] - b[1]:.0f}")
            print(
                f"  crowding for {e.type} at {e.frame_start}: "
                f"max IoU with another box {worst:.2f}"
            )
        if not events:
            print("no fall, crouch or reach cleared its sustained threshold on this clip")
        for e in events:
            print(f"  {e}")
        if args.events_json:
            by_id = {t.track_id: t for t in tracks}
            payload = []
            for e in events:
                rows = []
                for tid in e.track_ids:
                    t = by_id.get(tid)
                    if t is None:
                        continue
                    for f, box, kp, sc in zip(
                        t.frames, t.boxes, t.keypoints, t.scores, strict=True
                    ):
                        ang, ext, torso = _torso(kp, 0.3)
                        rows.append(
                            {
                                "frame": int(f),
                                "track_id": int(tid),
                                "score": round(float(sc), 3),
                                "box_h": round(float(box[3] - box[1]), 1),
                                "torso_angle_deg": None
                                if not np.isfinite(ang)
                                else round(ang, 1),
                                "hip_ankle_over_torso": (
                                    None
                                    if not (np.isfinite(ext) and torso > 0)
                                    else round(float(ext / torso), 3)
                                ),
                                "in_event": bool(e.frame_start <= f <= e.frame_end),
                            }
                        )
                payload.append(
                    {
                        "type": e.type,
                        "camera": e.camera,
                        "clip": clip.name,
                        "frame_start": e.frame_start,
                        "frame_end": e.frame_end,
                        "track_ids": list(e.track_ids),
                        "value": e.value,
                        "threshold": e.threshold,
                        "support_score_p50": None if e.support is None else e.support.score_p50,
                        "support_score_min": None if e.support is None else e.support.score_min,
                        "support_observed": None if e.support is None else e.support.observed,
                        "support_span": None if e.support is None else e.support.span,
                        "track_rows": rows,
                    }
                )
            Path(args.events_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.events_json).write_text(
                json.dumps(
                    {"camera": args.camera, "clip": clip.name, "frames": n, "events": payload},
                    indent=2,
                )
                + "\n"
            )
            print(f"wrote {args.events_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
