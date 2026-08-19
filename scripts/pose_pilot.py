"""Pose pilot: the first current through the tier-2 keypoint slot.

`analytics/tracker.py` declares `Track.keypoints` and says plainly "filled by the second
stage's pose model, which does not exist yet". `analytics/events.py` carries
`pose_posture_events` and `reach_to_shelf_events`, both written and both unreachable,
behind `require_keypoints`. This script is the first thing to fill the slot and the first
caller of both functions, so its job is two measurements and a witness statement:

1. **Are top-down ViTPose keypoints believable on these cameras at all?** Overhead
   down-pitched CCTV, person boxes a median 210-553 px tall on the two ground-truthed
   clips -- nothing in COCO keypoints looks like this viewpoint. Skeleton renders plus
   per-track confidence distributions are the instrument; the verdict is visual and goes
   in REPORT.md, not in a number this script invents.
2. **What do the two pose event functions do the first time anyone calls them?** They are
   called exactly as written. If an interface bug fires, it is recorded verbatim in
   events.json rather than worked around, because a bug found by the first caller and
   silently patched in the caller is a bug the second caller finds again.

Boxes come from `runs/offline_tracks01/{cam}/ground_truth.json` -- human-reviewed tracks,
so pose quality is measured on true boxes and not on detector noise. Frame sampling
replicates `scripts/offline_tracks.py` exactly (`data.video.frames` at the provenance's
fps, first `max_frames` frames), because the ground truth's frame indices are indices
into that sampling and nothing else.

The terrain map for `reach_to_shelf_events` comes from the site-adapted
`hydranet_retail_cctv` checkpoint, whose taxonomy names the class `display_fixture`
(id 12); the surfaces scheme's `fixture` is a different, non-comparable class
(memory: surfaces `fixture` is `product` merged in). The map is resized
nearest-neighbour to source resolution because `reach_to_shelf_events` indexes it with
wrist coordinates in image pixels -- that resize is part of this caller's obligations,
not something events.py does for you.

Usage:
    nice -n 10 .venv/bin/python scripts/pose_pilot.py --out runs/pose_pilot01
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from syncai_hydranet.analytics import events as ev
from syncai_hydranet.analytics.stage import TerrainFrame
from syncai_hydranet.analytics.tracker import Track
from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames as video_frames
from syncai_hydranet.data.video import probe
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.device import pick_device
from syncai_hydranet.utils.visualize import crop_box, preprocess

# COCO skeleton over the 17 points of events.KEYPOINT_NAMES, 0-indexed.
BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (0, 5), (0, 6), (5, 6),                  # neck / shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),         # arms
    (5, 11), (6, 12), (11, 12),              # trunk
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
)  # fmt: skip

SCORE_LOW, SCORE_MID = 0.3, 0.5  # 0.3 is events.py's own default score_thr


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/pose_pilot01")
    ap.add_argument("--gt-root", default="runs/offline_tracks01")
    ap.add_argument("--cams", nargs="+", default=["cam11", "cam04"])
    ap.add_argument("--pose-model", default="usyd-community/vitpose-base-simple")
    ap.add_argument("--terrain-config", default="configs/hydranet_retail_cctv.yaml")
    ap.add_argument("--terrain-checkpoint", default="runs/hydranet_retail_cctv/best.pt")
    ap.add_argument("--terrain-weights", default="ema")
    ap.add_argument("--fixture-class", default="display_fixture")
    ap.add_argument("--renders-per-cam", type=int, default=7)
    return ap


# ------------------------------------------------------------------- pose inference


def pose_for_frame(processor, model, device, frame: np.ndarray, boxes_xyxy: np.ndarray):
    """(N,4) xyxy ground-truth boxes on one frame -> (N,17,3) keypoints in image pixels.

    The processor takes COCO xywh and does its own top-down crop to 256x192, so boxes
    are handed over untouched -- including the ones that poke past the frame edge, which
    the ground truth genuinely contains (cam04 track 1 sits above y=0 for most of its
    life). Clamping them here would measure a different question.
    """
    xywh = boxes_xyxy.copy().astype(float)
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    inputs = processor(frame, boxes=[xywh], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    results = processor.post_process_pose_estimation(out, boxes=[xywh])[0]
    kps = np.zeros((len(xywh), 17, 3), dtype=np.float32)
    for i, person in enumerate(results):
        kps[i, :, :2] = person["keypoints"].cpu().numpy()
        kps[i, :, 2] = person["scores"].cpu().numpy()
    return kps


def terrain_for_frame(model, device, frame: np.ndarray, size, src_wh) -> np.ndarray:
    """One frame -> HxW terrain class ids at **source resolution**, uint8.

    `reach_to_shelf_events` indexes the map with wrist keypoints in image pixels, so the
    map must live in the same pixel space as the keypoints. Nearest-neighbour, because
    these are class ids and interpolating them invents classes.
    """
    x, _, region = preprocess(Image.fromarray(frame), size)
    with torch.no_grad():
        res = model.predict(x.to(device))
    terr = crop_box(res["terrain"][0].cpu().numpy(), region).astype(np.uint8)
    w, h = src_wh
    return np.asarray(Image.fromarray(terr).resize((w, h), Image.Resampling.NEAREST))


# ------------------------------------------------------------------------ rendering


def score_color(s: float) -> tuple[int, int, int]:
    if s < SCORE_LOW:
        return (230, 40, 40)
    if s < SCORE_MID:
        return (255, 160, 20)
    return (40, 220, 70)


def draw_skeleton(draw: ImageDraw.ImageDraw, kps: np.ndarray, r: int = 5) -> None:
    """Bones only where both ends clear events.py's 0.3 floor; points always, coloured
    by score, so a render shows exactly what the event functions would get to use."""
    for a, b in BONES:
        if kps[a, 2] >= SCORE_LOW and kps[b, 2] >= SCORE_LOW:
            c = score_color(float(min(kps[a, 2], kps[b, 2])))
            draw.line([tuple(kps[a, :2]), tuple(kps[b, :2])], fill=c, width=3)
    for i in range(17):
        x, y, s = kps[i]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=score_color(float(s)))


def render_crop(
    frame_jpeg: bytes,
    box: np.ndarray,
    kps: np.ndarray,
    caption: str,
    out_path: Path,
) -> None:
    img = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(list(box), outline=(90, 160, 255), width=2)
    draw_skeleton(draw, kps)
    m = 0.25 * max(box[2] - box[0], box[3] - box[1])
    x0 = max(0, int(box[0] - m))
    y0 = max(0, int(box[1] - m))
    x1 = min(img.width, int(box[2] + m))
    y1 = min(img.height, int(box[3] + m))
    crop = img.crop((x0, y0, x1, y1))
    if crop.height < 640:  # upscale so a reviewer can actually see the joints
        k = 640 / crop.height
        crop = crop.resize((int(crop.width * k), 640), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (crop.width, crop.height + 22), (0, 0, 0))
    out.paste(crop, (0, 22))
    ImageDraw.Draw(out).text((4, 5), caption, fill=(255, 255, 255))
    out.save(out_path)


def pick_render_situations(tracks: dict, kps_by_track: dict, n: int) -> list[dict]:
    """Choose (track, frame) pairs that cover the postures the report has to judge:
    tallest box (standing), widest aspect (bent/fallen-looking), lowest and highest mean
    keypoint score, and edge-truncated boxes. Deduped, capped at ``n``."""
    rows = []
    for tid, obs in tracks.items():
        frs = sorted(int(f) for f in obs)
        for i, f in enumerate(frs):
            b = np.asarray(obs[str(f)], dtype=float)
            w, h = b[2] - b[0], b[3] - b[1]
            rows.append(
                {
                    "track": tid,
                    "frame": f,
                    "i": i,
                    "h": h,
                    "aspect": w / max(h, 1e-6),
                    "edge": bool(b[0] < 2 or b[1] < 2 or b[2] > 1918 or b[3] > 1078),
                    "mean_score": float(kps_by_track[tid][i][:, 2].mean()),
                }
            )
    picks: list[tuple[str, dict]] = []

    def take(tag: str, row: dict) -> None:
        if all(r["frame"] != row["frame"] or r["track"] != row["track"] for _, r in picks):
            picks.append((tag, row))

    take("tallest_standing", max(rows, key=lambda r: r["h"]))
    take("widest_aspect", max(rows, key=lambda r: r["aspect"]))
    take("lowest_conf", min(rows, key=lambda r: r["mean_score"]))
    take("highest_conf", max(rows, key=lambda r: r["mean_score"]))
    edge = [r for r in rows if r["edge"]]
    if edge:
        take("edge_truncated", min(edge, key=lambda r: r["mean_score"]))
        take("edge_truncated_best", max(edge, key=lambda r: r["mean_score"]))
    # per-track midpoints for coverage until the budget is spent
    for tid in sorted(tracks, key=int):
        if len(picks) >= n:
            break
        frs = sorted(int(f) for f in tracks[tid])
        mid = frs[len(frs) // 2]
        row = next(r for r in rows if r["track"] == tid and r["frame"] == mid)
        take(f"track{tid}_mid", row)
    return [{"tag": tag, **row} for tag, row in picks[:n]]


# ----------------------------------------------------------------------------- main


def run_cam(cam: str, args, pose_proc, pose_model, terr_model, terr_size, device, out: Path):
    cam_out = out / cam
    cam_out.mkdir(parents=True, exist_ok=True)
    prov = json.loads((Path(args.gt_root) / cam / "provenance.json").read_text())
    gt = json.loads((Path(args.gt_root) / cam / "ground_truth.json").read_text())
    clip, fps, max_frames = prov["clip"], prov["fps"], prov["max_frames"]
    src_w, src_h, real_fps = probe(clip)
    print(f"[{cam}] {clip}  {src_w}x{src_h} @ {real_fps:.2f} real, sampling {fps}")

    # frame index -> [(track_id, box)] so one decode pass serves every track
    by_frame: dict[int, list[tuple[str, np.ndarray]]] = {}
    for tid, obs in gt.items():
        for f, box in obs.items():
            by_frame.setdefault(int(f), []).append((tid, np.asarray(box, dtype=float)))

    kps_by_track: dict[str, list[np.ndarray]] = {tid: [] for tid in gt}
    frame_jpeg: dict[int, bytes] = {}
    # `TerrainFrame`, not `list[dict]`: this is the second stage's real payload
    # producer, and `reach_to_shelf_events` states what it reads. See analytics/stage.py.
    terrain_frames: list[TerrainFrame] = []
    t0 = time.time()
    n_frames = 0
    for idx, frame in enumerate(video_frames(clip, src_w, src_h, fps)):
        if idx >= max_frames:
            break
        n_frames += 1
        entries = by_frame.get(idx, [])
        if entries:
            boxes = np.stack([b for _, b in entries])
            kps = pose_for_frame(pose_proc, pose_model, device, frame, boxes)
            for (tid, _), k in zip(entries, kps, strict=True):
                kps_by_track[tid].append(k)
        terrain_frames.append(
            {
                "frame_index": idx,
                "terrain": terrain_for_frame(
                    terr_model, device, frame, terr_size, (src_w, src_h)
                ),
            }
        )
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=88)
        frame_jpeg[idx] = buf.getvalue()
        if (idx + 1) % 50 == 0:
            print(f"  frame {idx + 1}/{max_frames}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- Tracks, keypoints filled: the first time this field has ever been non-empty.
    tracks: list[Track] = []
    for tid, obs in gt.items():
        frs = sorted(int(f) for f in obs)
        boxes = [np.asarray(obs[str(f)], dtype=float) for f in frs]
        tracks.append(
            Track(
                track_id=int(tid),
                box=boxes[-1].copy(),
                frames=list(frs),
                boxes=boxes,
                confirmed=True,
                keypoints=list(kps_by_track[tid]),
            )
        )
    ev.require_keypoints(tracks)  # the gate that has always raised; it must pass now

    # numpy's stub types `savez_compressed(..., allow_pickle: bool, **kwds)`, so ty matches
    # an unpacked `**dict[str, ndarray]` against `allow_pickle` and rejects it; at runtime
    # the arrays land in `**kwds` as intended.
    np.savez_compressed(
        cam_out / "keypoints.npz",
        **{  # ty: ignore[invalid-argument-type]
            f"track{tid}_frames": np.array(sorted(int(f) for f in gt[tid])) for tid in gt
        },
        **{f"track{tid}_kps": np.stack(kps_by_track[tid]) for tid in gt},
    )

    # ---- confidence measurements
    conf: dict = {"per_track": {}, "per_keypoint": {}}
    all_kps = np.concatenate([np.stack(v) for v in kps_by_track.values()])
    for tid, v in kps_by_track.items():
        arr = np.stack(v)[:, :, 2]
        conf["per_track"][tid] = {
            "frames": int(arr.shape[0]),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "frac_kps_below_0.3": round(float((arr < SCORE_LOW).mean()), 4),
        }
    for i, name in enumerate(ev.KEYPOINT_NAMES):
        s = all_kps[:, i, 2]
        conf["per_keypoint"][name] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(np.median(s)), 4),
            "frac_below_0.3": round(float((s < SCORE_LOW).mean()), 4),
        }
    # what the fall detector would actually see: torso angle resolvability + distribution
    angles = []
    for v in kps_by_track.values():
        for k in v:
            a, _, _ = ev._torso(k, SCORE_LOW)
            angles.append(a)
    angles = np.asarray(angles)
    conf["torso_angle"] = {
        "resolvable_frac": round(float(np.isfinite(angles).mean()), 4),
        "median_deg": round(float(np.nanmedian(angles)), 2)
        if np.isfinite(angles).any()
        else None,
        "p90_deg": round(float(np.nanpercentile(angles, 90)), 2)
        if np.isfinite(angles).any()
        else None,
        "frac_over_55deg": round(float(np.nanmean(angles >= 55.0)), 4)
        if np.isfinite(angles).any()
        else None,
    }
    (cam_out / "confidence.json").write_text(json.dumps(conf, indent=1) + "\n")

    # ---- first call of the two event functions. Failures recorded verbatim, not fixed.
    results: dict = {}
    try:
        rows = ev.pose_posture_events(tracks, fps=fps, camera=cam)
        results["pose_posture_events"] = [e.as_row() for e in rows]
    except Exception:
        results["pose_posture_events"] = {"error": traceback.format_exc()}
    terr_classes = terr_model._pilot_terrain_classes
    fixture_id = terr_classes.index(args.fixture_class)
    try:
        rows = ev.reach_to_shelf_events(
            tracks, terrain_frames, fps=fps, camera=cam, fixture_id=fixture_id
        )
        results["reach_to_shelf_events"] = [e.as_row() for e in rows]
    except Exception:
        results["reach_to_shelf_events"] = {"error": traceback.format_exc()}
    results["fixture_id"] = fixture_id
    results["fixture_class"] = args.fixture_class
    # Bound rather than filtered in the comprehension. `TerrainFrame` types `terrain` as
    # optional and the guard is not pedantry: without it `f["terrain"] == fixture_id` is
    # `False` rather than a mask the moment one frame carries no map, and `.sum()` on a
    # bool raises three lines from where the cause is. A comprehension's `if` does not
    # narrow the subscript that follows it, which is the same gap events.py has.
    fixture_px: dict[int, int] = {}
    for f in terrain_frames:
        terrain_map = f.get("terrain")
        if terrain_map is None:
            continue
        fixture_px[int(f["frame_index"])] = int((terrain_map == fixture_id).sum())
    results["fixture_px"] = {
        "median_per_frame": int(np.median(list(fixture_px.values()))),
        "frames_with_any": int(sum(v > 0 for v in fixture_px.values())),
        "frames_total": len(fixture_px),
    }
    (cam_out / "events.json").write_text(json.dumps(results, indent=1) + "\n")

    # ---- skeleton renders for the visual verdict
    situations = pick_render_situations(gt, kps_by_track, args.renders_per_cam)
    for s in situations:
        tid, f = s["track"], s["frame"]
        i = sorted(int(x) for x in gt[tid]).index(f)
        box = np.asarray(gt[tid][str(f)], dtype=float)
        kps = kps_by_track[tid][i]
        cap = (
            f"{cam} f{f} track{tid} [{s['tag']}] h={s['h']:.0f}px "
            f"w/h={s['aspect']:.2f} mean_conf={s['mean_score']:.2f}"
        )
        render_crop(
            frame_jpeg[f], box, kps, cap, cam_out / f"skeleton_f{f:03d}_t{tid}_{s['tag']}.png"
        )
    (cam_out / "render_index.json").write_text(
        json.dumps([{k: v for k, v in s.items() if k != "i"} for s in situations], indent=1)
        + "\n"
    )
    print(f"[{cam}] done: {n_frames} frames, {len(situations)} renders, "
          f"{time.time() - t0:.0f}s")  # fmt: skip
    return {"frames_read": n_frames, "renders": len(situations), "clip": clip, "fps": fps}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(None)

    from transformers import AutoProcessor, VitPoseForPoseEstimation

    pose_proc = AutoProcessor.from_pretrained(args.pose_model)
    # transformers wraps `PreTrainedModel.to` with functools.wraps, so ty resolves the
    # wrapper's own signature instead of `nn.Module.to` and rejects the device argument.
    pose_model = (
        VitPoseForPoseEstimation.from_pretrained(args.pose_model)
        .to(device)  # ty: ignore[invalid-argument-type]
        .eval()
    )
    # order check, not faith: the processor's id2label must equal events.KEYPOINT_NAMES
    id2label = pose_model.config.id2label
    got = tuple(id2label[i].lower().replace("l_", "left_").replace("r_", "right_")
                for i in range(17))  # fmt: skip
    assert got == ev.KEYPOINT_NAMES, f"keypoint order mismatch: {got}"

    cfg = load_config(args.terrain_config, validate=False)
    terr_model = build_model(cfg).to(device).eval()
    terr_model.load_state_dict(
        select_weights(load_checkpoint(args.terrain_checkpoint), args.terrain_weights)
    )
    # torch annotates `Module.__setattr__` as accepting Tensor | Module only, but at
    # runtime any other value falls through to `object.__setattr__`; run_cam reads
    # this list back for the fixture-id lookup.
    terr_model._pilot_terrain_classes = list(  # ty: ignore[invalid-assignment]
        cfg["data"]["terrain_classes"]
    )
    terr_size = cfg["data"]["input_size"]

    summary = {}
    for cam in args.cams:
        summary[cam] = run_cam(
            cam, args, pose_proc, pose_model, terr_model, terr_size, device, out
        )

    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent
    ).stdout.strip()
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "tool": "scripts/pose_pilot.py",
                "git_commit": git,
                "pose_model": args.pose_model,
                "terrain_config": args.terrain_config,
                "terrain_checkpoint": args.terrain_checkpoint,
                "terrain_weights": args.terrain_weights,
                "fixture_class": args.fixture_class,
                "gt_root": args.gt_root,
                "boxes": "human-reviewed ground_truth.json, not detector output",
                "score_floor_for_events": SCORE_LOW,
                "cams": summary,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
