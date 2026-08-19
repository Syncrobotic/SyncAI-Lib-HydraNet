#!/usr/bin/env python3
"""Offline non-causal tracking: spend future frames to save human merge decisions.

docs/RETAIL_SECURITY.md names "one labelled clip" as the critical path for everything
temporal, and scripts/track_review.py already turns labelling into a merge-review loop:
the human cost is one decision per proposed fragment. The online tracker is a fragment
factory -- greedy IoU, one-step constant velocity, no memory past `max_age` frames -- so
a shopper who steps behind a fixture for three seconds returns as a new id, every time,
and the reviewer pays for each return.

This is the same propose step with three additions the online tracker cannot have,
because they are non-causal or need a noise model it honestly refuses to invent:

1. **ByteTrack-style two-stage association.** High-score detections associate first;
   tracks still unmatched get a second chance against the low-score band, which is where
   a half-occluded person lives. New tracks are born only from high-score detections, so
   the low band can extend a track but never invent one.
2. **Constant-velocity Kalman filter** on (cx, cy, w, h). `analytics/tracker.py` refuses
   a Kalman filter because no measured noise model exists to fit one to; the constants
   here are ByteTrack's published ones (position std h/20, velocity std h/160, tuned on
   MOT17 at 25-30 fps) with the velocity prior rescaled by the *measured* frame-rate
   ratio -- provenance, not invention. The repo's one measured association lesson is
   kept: match against the predicted box OR the last observed box, whichever agrees
   better (tracker.py measured 76 -> 34 fragments from that change alone).
3. **Non-causal stitching.** After the forward pass, fragment endpoints are compared
   across gaps of up to several seconds; time/space-compatible pairs are chained into
   one track and the gap linearly interpolated. Optionally the chain must be confirmed
   by crop-encoder cosine similarity against a threshold measured on this clip:
   fragments that coexist in a frame are guaranteed different people, so their pairwise
   similarity is the clip's own "different person" baseline and the threshold sits on
   it rather than on a guess.

Output is `tracks.json` in scripts/track_review.py's exact schema plus the same review
sheets, so the existing human loop (write merges.json, run `track_review.py apply`)
consumes it unchanged. The deliverable is fewer rows on that sheet.

    nice -n 10 .venv/bin/python scripts/offline_tracks.py \\
        --config configs/hydranet_indoor.yaml \\
        --checkpoint runs/hydranet_indoor_det60/best.pt \\
        --clip datasets/studioa_clips/Taichung-cam11/archive_20260816-062816_*.mp4 \\
        --out runs/offline_tracks01/cam11 \\
        --gt runs/gt_cam11/ground_truth.json \\
        --reproduce runs/gt_cam11/tracks.json

Measurement discipline: detections are computed **once** and cached; the online tracker
(baseline) and the offline tracker consume the same cache. The baseline consumes the
high band only, which is exactly its published operating point (score_thr 0.25 in
runs/gt_cam11/provenance.json); the ablations in metrics.json separate what the low
band buys from what stitching buys. Ground truth here measures association only --
track_review.py's own caveat stands: a person the detector never saw is absent from
both sides of the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import itertools

import track_review

from syncai_hydranet.analytics import Tracker
from syncai_hydranet.analytics.bytetrack import OfflineForward, to_cwh
from syncai_hydranet.analytics.reid_metrics import id_switches, idf1
from syncai_hydranet.config import load_config
from syncai_hydranet.data.transforms import invert_geom
from syncai_hydranet.data.video import frames, probe
from syncai_hydranet.models.crop_encoder import CropEncoder
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.device import pick_device
from syncai_hydranet.utils.visualize import preprocess

PERSON = track_review.PERSON
EMBED_W, EMBED_H = 128, 256  # crop_encoder's training resolution (scripts/train_attributes.py)
EMBED_CROPS_PER_FRAG = 8
EMBED_STRIDE = 3  # every 3rd observation, so 8 crops span 24 frames of life, not the first 8

# `Kalman`, `Fragment` and `OfflineForward` now live in the package, at
# `analytics/bytetrack.py`, because `stable_infer.py` needs them too and was reaching
# them here through a `sys.path` insert. That module's docstring carries the part worth
# reading before using it: it runs a Kalman filter that `analytics/tracker.py` documents
# a refusal to run, and nobody has measured the two against each other on this footage.


def stash_crops(fwd: OfflineForward, frame: np.ndarray) -> None:
    """Review thumbnails (track_review's geometry) + encoder crops, per live fragment.

    A function here rather than a method there. It cuts crops at `track_review.py`'s
    display sizes and the crop encoder's training resolution, which are this script's
    concerns; `Fragment` keeps the two lists as storage for whoever wants to fill them,
    and the tracker itself never looks at them.
    """
    h, w = frame.shape[:2]
    for t in fwd.tracks:
        if t.age != 0 or not t.confirmed:
            continue
        x0, y0, x1, y1 = (int(v) for v in t.boxes[-1])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 4 or y1 - y0 < 8:
            continue
        crop = Image.fromarray(frame[y0:y1, x0:x1])
        if len(t.review_crops) < track_review.MAX_CROPS:
            t.review_crops.append(crop.resize((track_review.CROP_W, track_review.CROP_H)))
        if len(t.embed_crops) < EMBED_CROPS_PER_FRAG and (t.obs_count - 1) % EMBED_STRIDE == 0:
            t.embed_crops.append(
                np.asarray(crop.resize((EMBED_W, EMBED_H), Image.Resampling.BILINEAR))
            )


# ----------------------------------------------------------------- non-causal stitch


def embed_fragments(frags, encoder_path, device) -> dict[int, np.ndarray]:
    """One L2-normalised embedding per fragment: mean of its crop embeddings."""
    # Through the package loader, like every other checkpoint read here. This was a bare
    # `torch.load(..., weights_only=False)`, which is arbitrary code execution on the file
    # -- and it bought nothing: train_attributes.py writes {"model": state_dict,
    # "attributes": [str, ...]}, which the restricted unpickler reads without complaint.
    # The risk was never in today's local file; it is that the next encoder to arrive
    # comes off a release page or a colleague's machine and nothing says so.
    ck = load_checkpoint(encoder_path)
    enc = CropEncoder(len(ck["attributes"])).to(device).eval()
    enc.load_state_dict(ck["model"])
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    out: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for t in frags:
            if not t.embed_crops:
                continue
            x = np.stack([(c.astype(np.float32) / 255.0 - mean) / std for c in t.embed_crops])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
            e = enc.embed_only(x).mean(dim=0).cpu().numpy()
            n = np.linalg.norm(e)
            if n > 1e-9:
                out[t.frag_id] = e / n
    return out


def different_person_cosines(frags, embeds) -> list[float]:
    """Cosine similarity between fragments that coexist in a frame.

    Two tracks alive in the same frame are two different people by construction, so this
    distribution is the clip's own "different person" appearance baseline -- measured,
    not imported from Market-1501, where this encoder is admittedly weak (mAP 0.054).
    """
    vals = []
    for i, a in enumerate(frags):
        fa = set(a.frames)
        for b in frags[i + 1 :]:
            if a.frag_id in embeds and b.frag_id in embeds and fa & set(b.frames):
                vals.append(float(embeds[a.frag_id] @ embeds[b.frag_id]))
    return vals


def stitch(frags, sample_fps, gap_s, dist_base, dist_rate, size_gate,
           app_mode, app_thr, embeds):  # fmt: skip
    """Chain fragments whose endpoints are compatible across a temporal gap.

    Greedy over candidate pairs sorted by normalised distance / allowance; each fragment
    takes at most one successor and one predecessor, so chains are temporally disjoint
    by construction. Returns (chains, stitch log).
    """
    max_gap = max(1, round(gap_s * sample_fps))
    order = sorted(range(len(frags)), key=lambda i: frags[i].frames[0])
    cands = []
    for ii, i in enumerate(order):
        a = frags[i]
        a_end, a_box = a.frames[-1], a.boxes[-1]
        a_c, a_h = to_cwh(a_box)[:2], max(a_box[3] - a_box[1], 1.0)
        a_vel = a.kalman.velocity
        for j in order[ii + 1 :]:
            b = frags[j]
            gap = b.frames[0] - a_end
            if gap < 1:
                continue
            if gap > max_gap:
                # order[] is sorted by *start*; later starts only grow the gap further
                # relative to this a_end only if b.start grows -- so we can break.
                break
            b_box = b.boxes[0]
            b_c, b_h = to_cwh(b_box)[:2], max(b_box[3] - b_box[1], 1.0)
            if min(a_h, b_h) / max(a_h, b_h) < size_gate:
                continue
            dt = gap / sample_fps
            href = (a_h + b_h) / 2
            # Where could a have gone? Either it kept its terminal velocity (walking
            # through the occlusion) or it stayed put (browsing behind the fixture).
            # Score against both and keep the friendlier answer.
            d_still = np.linalg.norm(b_c - a_c) / href
            d_walk = np.linalg.norm(b_c - (a_c + a_vel * gap)) / href
            d = min(d_still, d_walk)
            allow = dist_base + dist_rate * dt
            if d > allow:
                continue
            cos = None
            if a.frag_id in embeds and b.frag_id in embeds:
                cos = float(embeds[a.frag_id] @ embeds[b.frag_id])
            if app_mode == "require" and cos is None:
                continue
            if app_mode in ("confirm", "require") and cos is not None and cos < app_thr:
                continue
            cands.append((d / allow, i, j, gap, d, allow, cos))

    cands.sort(key=lambda c: c[0])
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    log = []
    for _, i, j, gap, d, allow, cos in cands:
        if i in succ or j in pred:
            continue
        succ[i] = j
        pred[j] = i
        log.append(
            {
                "from_frag": frags[i].frag_id,
                "to_frag": frags[j].frag_id,
                "gap_frames": int(gap),
                "gap_seconds": round(gap / sample_fps, 2),
                "dist_box_heights": round(d, 3),
                "allowance": round(allow, 3),
                "embed_cosine": None if cos is None else round(cos, 3),
            }
        )

    chains = []
    for ii in order:
        if ii in pred:
            continue
        chain = [ii]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        chains.append([frags[k] for k in chain])
    return chains, log


def chains_to_tracks(chains) -> tuple[dict, dict[str, list[int]]]:
    """Chains -> track_review's tracks.json schema, gaps linearly interpolated.

    Returns (tracks, interpolated-frames-per-track). Interpolated boxes make the file a
    denser label; they are excluded from the observed-only metric variant because the
    ground truth (built from observed proposals) has no boxes there to agree with.
    """
    tracks: dict[str, dict[str, list[float]]] = {}
    interp: dict[str, list[int]] = {}
    ordered = sorted(chains, key=lambda ch: ch[0].frames[0])
    for tid, chain in enumerate(ordered, start=1):
        byframe: dict[str, list[float]] = {}
        added: list[int] = []
        for frag in chain:
            for f, b in zip(frag.frames, frag.boxes, strict=False):
                byframe[str(int(f))] = [float(v) for v in b]
        for a, b in itertools.pairwise(chain):
            f0, f1 = a.frames[-1], b.frames[0]
            b0, b1 = np.asarray(a.boxes[-1], float), np.asarray(b.boxes[0], float)
            for f in range(f0 + 1, f1):
                w = (f - f0) / (f1 - f0)
                byframe[str(f)] = [float(v) for v in (1 - w) * b0 + w * b1]
                added.append(f)
        tracks[str(tid)] = byframe
        if added:
            interp[str(tid)] = added
    return tracks, interp


# ------------------------------------------------------------------------ metrics


def _int_keys(d: dict) -> dict:
    return {int(t): {int(f): np.asarray(b, float) for f, b in v.items()} for t, v in d.items()}


def score(gt: dict, pred: dict) -> dict:
    g, p = _int_keys(gt), _int_keys(pred)
    return {**idf1(g, p), **id_switches(g, p)}


def track_stats(tracks: dict) -> dict:
    lens = sorted(len(v) for v in tracks.values())
    return {
        "tracks": len(tracks),
        "median_track_frames": lens[len(lens) // 2] if lens else 0,
        "max_track_frames": lens[-1] if lens else 0,
    }


def measured_motion(frags, sample_fps) -> dict:
    """Per-frame centre displacement in box heights, over every observed step.

    This is the number `--stitch-rate` has to sit above: a shopper's measured speed in
    the unit the gate uses. Reported so the default is checkable against this clip.
    """
    v = []
    for t in frags:
        for (f0, b0), (f1, b1) in zip(
            zip(t.frames, t.boxes, strict=False),
            list(zip(t.frames, t.boxes, strict=False))[1:],
            strict=False,
        ):
            if f1 - f0 != 1:
                continue
            h = max(b0[3] - b0[1], 1.0)
            v.append(
                float(
                    np.linalg.norm(to_cwh(np.asarray(b1))[:2] - to_cwh(np.asarray(b0))[:2]) / h
                )
            )
    if not v:
        return {}
    arr = np.array(v) * sample_fps  # heights per second
    return {
        "median_speed_heights_per_s": round(float(np.median(arr)), 3),
        "p90_speed_heights_per_s": round(float(np.percentile(arr, 90)), 3),
        "steps_measured": len(v),
    }


# ---------------------------------------------------------------------------- main


def detect_all(model, device, clip, size, fps, max_frames, floor):
    """One detection pass; both trackers consume this cache. Yields nothing twice."""
    src_w, src_h, _ = probe(clip)
    dets: list[tuple[np.ndarray, np.ndarray]] = []
    raw_frames: list[np.ndarray] = []
    for frame in frames(clip, src_w, src_h, fps):
        x, _, region = preprocess(Image.fromarray(frame), size)
        with torch.no_grad():
            det = model.predict(x.to(device), score_thr=floor)["detection"][0]
        if len(det.get("labels", [])):
            lab = det["labels"].cpu().numpy()
            keep = lab == PERSON
            box = det["boxes"].cpu().numpy()[keep]
            sc = det["scores"].cpu().numpy()[keep]
            x0, y0, cw, ch = region
            box = invert_geom(box.astype(float), (cw / src_w, ch / src_h, x0, y0))
        else:
            box, sc = np.zeros((0, 4)), np.zeros((0,))
        dets.append((box, sc))
        raw_frames.append(frame)
        if len(dets) % 50 == 0:
            print(f"  frame {len(dets)}", flush=True)
        if max_frames and len(dets) >= max_frames:
            break
    return dets, raw_frames


def run_offline(dets, raw_frames, args, sample_fps, vel_scale, encoder_device,
                low_band=True, do_stitch=True, appearance=None):  # fmt: skip
    """Forward pass (+ optional stitch) over the cached detections."""
    low_thr = args.low_thr if low_band else args.high_thr
    fwd = OfflineForward(
        args.high_thr, low_thr, args.iou, args.iou_low, args.max_age, args.min_hits, vel_scale
    )
    for n, (box, sc) in enumerate(dets):
        fwd.update(box, sc, n)
        stash_crops(fwd, raw_frames[n])
    frags = sorted(fwd.finished(), key=lambda t: t.frames[0])

    app_mode = appearance if appearance is not None else args.appearance
    embeds: dict[int, np.ndarray] = {}
    app_thr = args.app_thr
    diff_cos: list[float] = []
    if app_mode != "off":
        embeds = embed_fragments(frags, args.encoder, encoder_device)
        diff_cos = different_person_cosines(frags, embeds)
        if app_thr is None:
            # Measured on this clip: coexisting fragments are different people, and a
            # stitch must look *more* alike than the median different-person pair.
            # Fallback 0.30 only when the clip never shows two people at once.
            app_thr = float(np.median(diff_cos)) if len(diff_cos) >= 5 else 0.30

    if do_stitch:
        chains, stitch_log = stitch(
            frags, sample_fps, args.stitch_gap_s, args.stitch_dist, args.stitch_rate,
            args.size_gate, app_mode, app_thr, embeds,
        )  # fmt: skip
    else:
        chains, stitch_log = [[f] for f in frags], []
    tracks, interp = chains_to_tracks(chains)
    return {
        "frags": frags,
        "chains": chains,
        "tracks": tracks,
        "interpolated": interp,
        "stitch_log": stitch_log,
        "app_thr": app_thr,
        "diff_person_cos": diff_cos,
    }


def observed_only(tracks: dict, interp: dict) -> dict:
    out = {}
    for tid, byframe in tracks.items():
        drop = {str(f) for f in interp.get(tid, [])}
        out[tid] = {f: b for f, b in byframe.items() if f not in drop}
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    # 5.0 / 300: runs/gt_cam11/provenance.json sampled at these, and the ground truth's
    # frame indices only mean anything if this samples identically.
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=300, help="0 means the whole clip")
    # 0.25: the score_thr of the gt_cam11/gt_cam04 proposals (provenance.json), so the
    # high band is byte-identical to the detection set the human reviewed.
    ap.add_argument("--high-thr", type=float, default=0.25)
    # 0.10: ByteTrack's published low-score floor; below it the paper (and a look at
    # these clips' score histograms) finds mostly background.
    ap.add_argument("--low-thr", type=float, default=0.10)
    # 0.3: analytics/tracker.py's default, measured there as not the binding constraint
    # (greedy vs optimal assignment: 76 vs 78 tracks, i.e. nothing).
    ap.add_argument("--iou", type=float, default=0.3)
    # 0.4: ByteTrack's second stage uses 0.5 at 25-30 fps; widened one notch because at
    # 5 fps sampled / 7-8 fps real the same walking speed moves ~5x further per frame.
    ap.add_argument("--iou-low", type=float, default=0.4)
    # 5 / 3: the baseline's published values (runs/gt_cam11/provenance.json). Longer
    # occlusions are the stitcher's job, not coasting's; confirmation stays at 3 because
    # under-count over over-count (tracker.py's stated policy).
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--min-hits", type=int, default=3)
    # None -> 25 / min(sample fps, measured avg_frame_rate): ByteTrack's Kalman noise
    # was tuned at 25-30 fps and this rescales its velocity prior to the real rate.
    ap.add_argument("--vel-scale", type=float, default=None)
    # 4.0 s: the observed failure is a person behind a fixture for 3+ s (RETAIL_SECURITY
    # / task brief); one second of margin on top. At 5 fps that is 20 frames.
    ap.add_argument("--stitch-gap-s", type=float, default=4.0)
    # 0.7 box heights base + 0.5 per second of gap: person boxes are 244-336 px median
    # (track_review.py), so the base is ~170-230 px -- generous against detector jitter,
    # tight against a different shopper one aisle over. The rate term is checkable
    # against metrics.json's measured_motion (median walking speed in heights/s on this
    # clip); it is meant to sit at or above the median, not at the p90 -- a sprinting
    # shopper is rarer than two adjacent ones.
    ap.add_argument("--stitch-dist", type=float, default=0.7)
    ap.add_argument("--stitch-rate", type=float, default=0.5)
    # 0.55: a fixed camera; across a <=4 s occlusion the same person's box height stays
    # well within 2x unless they cross most of the depth of the shop.
    ap.add_argument("--size-gate", type=float, default=0.55)
    # confirm: motion proposes, appearance can veto. The encoder is weak on Market-1501
    # (mAP 0.054) but same-camera same-clip crops are a far easier problem, and the veto
    # threshold is measured on this clip's coexisting (= different-person) pairs.
    ap.add_argument("--appearance", choices=("off", "confirm", "require"), default="confirm")
    ap.add_argument("--app-thr", type=float, default=None,
                    help="None = median cosine of coexisting fragment pairs")  # fmt: skip
    ap.add_argument("--encoder", default="runs/crop_encoder01/last.pt")
    ap.add_argument("--gt", default=None, help="ground_truth.json to score against")
    ap.add_argument(
        "--reproduce",
        default=None,
        help="an existing proposal tracks.json the baseline should reproduce",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config, validate=False)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), args.weights))
    det_head = model.det_head
    assert det_head is not None and det_head.cls_pred.bias is not None, (
        "checkpoint has no detection head"
    )
    track_review.check_person_class_space(int(det_head.cls_pred.bias.shape[0]))
    size = cfg["data"]["input_size"]

    _, _, real_fps = probe(args.clip)
    eff_fps = min(args.fps, real_fps)
    vel_scale = args.vel_scale if args.vel_scale is not None else 25.0 / eff_fps
    print(f"clip: {args.clip}")
    print(f"  avg_frame_rate {real_fps:.2f} fps real, sampling {args.fps} fps, "
          f"kalman vel scale {vel_scale:.1f}")  # fmt: skip

    print("detecting once (both trackers share this cache)...")
    dets, raw_frames = detect_all(
        model, device, args.clip, size, args.fps, args.max_frames, args.low_thr
    )
    n_hi = sum(int((s >= args.high_thr).sum()) for _, s in dets)
    n_lo = sum(int(((s >= args.low_thr) & (s < args.high_thr)).sum()) for _, s in dets)
    print(f"  {len(dets)} frames, {n_hi} high-band dets (>= {args.high_thr}), "
          f"{n_lo} low-band ({args.low_thr}-{args.high_thr})")  # fmt: skip

    # ---- baseline: the current online tracker at its published operating point.
    base = Tracker(iou_threshold=args.iou, max_age=args.max_age, min_hits=args.min_hits)
    for n, (box, sc) in enumerate(dets):
        base.update(box[sc >= args.high_thr], n)
    base_tracks = track_review.tracks_to_json([t for t in base.finished() if t.confirmed])
    (out / "baseline_tracks.json").write_text(json.dumps(base_tracks, indent=1) + "\n")

    reproduced = None
    if args.reproduce:
        ref = json.loads(Path(args.reproduce).read_text())
        same_n = len(ref) == len(base_tracks)
        same_frames = sorted(sorted(int(f) for f in v) for v in ref.values()) == sorted(
            sorted(int(f) for f in v) for v in base_tracks.values()
        )
        reproduced = bool(same_n and same_frames)
        print(f"  baseline reproduces {args.reproduce}: {reproduced} "
              f"({len(ref)} ref tracks vs {len(base_tracks)} here)")  # fmt: skip

    # ---- offline: full method, then ablations on the same cache.
    print("offline tracker (full: low band + kalman + stitch)...")
    full = run_offline(dets, raw_frames, args, args.fps, vel_scale, device)
    print(f"  {len(full['frags'])} fragments -> {len(full['tracks'])} tracks "
          f"({len(full['stitch_log'])} stitches, app_thr {full['app_thr']})")  # fmt: skip
    no_stitch = run_offline(dets, raw_frames, args, args.fps, vel_scale, device,
                            do_stitch=False, appearance="off")  # fmt: skip
    high_only = run_offline(dets, raw_frames, args, args.fps, vel_scale, device,
                            low_band=False)  # fmt: skip

    # ---- outputs in track_review's schema, sheet included, so `apply` just works.
    (out / "tracks.json").write_text(json.dumps(full["tracks"], indent=1) + "\n")
    crops: dict[str, list] = {}
    ordered = sorted(full["chains"], key=lambda ch: ch[0].frames[0])
    for tid, chain in enumerate(ordered, start=1):
        row: list = []
        for frag in chain:
            row.extend(frag.review_crops)
        crops[str(tid)] = row[: track_review.MAX_CROPS]
    pages, drawn = track_review._sheet(full["tracks"], crops, out)
    print(f"  {len(pages)} review pages, {drawn} crops")

    motion = measured_motion(full["frags"], args.fps)

    metrics: dict = {
        "clip": args.clip,
        "frames": len(dets),
        "real_fps_avg_frame_rate": round(real_fps, 3),
        "sample_fps": args.fps,
        "detections": {
            "high_band": n_hi,
            "low_band": n_lo,
            "high_thr": args.high_thr,
            "low_thr": args.low_thr,
        },
        "same_detections_for_both": True,
        "baseline_reproduces_reference": reproduced,
        "measured_motion": motion,
        "appearance": {
            "mode": args.appearance,
            "threshold": full["app_thr"],
            "different_person_cos_median": (
                round(float(np.median(full["diff_person_cos"])), 3)
                if full["diff_person_cos"]
                else None
            ),
            "different_person_pairs": len(full["diff_person_cos"]),
        },
        "baseline": track_stats(base_tracks),
        "offline": {
            **track_stats(full["tracks"]),
            "fragments_before_stitch": len(full["frags"]),
            "stitches": len(full["stitch_log"]),
            "interpolated_frames": sum(len(v) for v in full["interpolated"].values()),
        },
        "ablation_no_stitch": track_stats(no_stitch["tracks"]),
        "ablation_high_band_only": {
            **track_stats(high_only["tracks"]),
            "stitches": len(high_only["stitch_log"]),
        },
    }

    if args.gt:
        gt = json.loads(Path(args.gt).read_text())
        metrics["gt"] = {"path": args.gt, "identities": len(gt)}
        metrics["baseline"].update(score(gt, base_tracks))
        metrics["offline"].update(score(gt, full["tracks"]))
        metrics["offline_observed_only"] = {
            **track_stats(observed_only(full["tracks"], full["interpolated"])),
            **score(gt, observed_only(full["tracks"], full["interpolated"])),
        }
        metrics["ablation_no_stitch"].update(score(gt, no_stitch["tracks"]))
        metrics["ablation_high_band_only"].update(
            score(gt, observed_only(high_only["tracks"], high_only["interpolated"]))
        )

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")

    (out / "provenance.json").write_text(
        json.dumps(
            {
                "tool": "scripts/offline_tracks.py",
                "config": args.config,
                "checkpoint": args.checkpoint,
                "weights": args.weights,
                "clip": args.clip,
                "fps": args.fps,
                "real_fps_avg_frame_rate": round(real_fps, 3),
                "max_frames": args.max_frames,
                "score_thr": args.high_thr,  # the band track_review's flow reviewed at
                "low_thr": args.low_thr,
                "tracker": {
                    "iou": args.iou,
                    "iou_low": args.iou_low,
                    "max_age": args.max_age,
                    "min_hits": args.min_hits,
                    "vel_scale": round(vel_scale, 2),
                    "stitch_gap_s": args.stitch_gap_s,
                    "stitch_dist": args.stitch_dist,
                    "stitch_rate": args.stitch_rate,
                    "size_gate": args.size_gate,
                    "appearance": args.appearance,
                    "app_thr": full["app_thr"],
                },
                "frames_read": len(dets),
                "fragments_before_stitch": len(full["frags"]),
                "confirmed_tracks": len(full["tracks"]),
                "crops_drawn": drawn,
                "stitches": full["stitch_log"],
                "interpolated_frames": full["interpolated"],
            },
            indent=1,
        )
        + "\n"
    )

    print(f"\nbaseline {metrics['baseline']['tracks']} tracks -> "
          f"offline {metrics['offline']['tracks']} tracks "
          f"({len(full['frags'])} fragments, {len(full['stitch_log'])} stitched)")  # fmt: skip
    if args.gt:
        b, o = metrics["baseline"], metrics["offline"]
        print(f"  IDF1 {b['idf1']:.3f} -> {o['idf1']:.3f}   "
              f"switches {b['switches']} -> {o['switches']}   "
              f"tracks/identity {b['tracks_per_identity']:.2f} -> "
              f"{o['tracks_per_identity']:.2f}")  # fmt: skip
    print(f"  wrote {out}/tracks.json, metrics.json, provenance.json, review pages")
    print(
        "  next: a human writes merges.json against the review pages, then "
        "`track_review.py apply --out " + str(out) + "`"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
