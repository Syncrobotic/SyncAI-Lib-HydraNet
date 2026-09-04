#!/usr/bin/env python3
"""Does two-stage association join fragments of one shopper, or two shoppers?

    python3 scripts/track_identity.py --cameras Taichung-cam01 --out runs/identity01

PLAN §7.11 measured that 86% of mid-view track deaths have a box still on the person at a
median 0.338 against the 0.35 threshold, and `runs/endings05/` measured that two-stage
association cuts those deaths 111 -> 38 and doubles median track length. It was **not
adopted**, on a stated objection that is a judgement rather than a number: the Kalman may
coast onto a neighbour, joining two shoppers into one track and inflating dwell.

This turns that objection into a number, using the thing §6 step 5 already names as the
instrument -- "distinguishing 'the same person, a metre on' from 'a different person, a
metre away' is what an appearance model is for". The appearance model is
`analytics/appearance.py`'s nine torso-colour statistics, which read 0.880 balanced
accuracy on `staff/customer` held out by camera (§7.15) and beat every embedding here.

---------------------------------------------------------------------------
THE COMPARISON, AND WHY IT NEEDS NO INVENTED THRESHOLD

Both trackers run over the **same detections**, in one inference pass. The single-stage
tracker sees only the high band, which is what ships; the two-stage tracker sees the low
band as well. So every two-stage track is a sequence of observations, each of which the
single-stage tracker either assigned to one of its own tracks or never saw.

A **join** is where a two-stage track's observations change single-stage identity: it
stitched fragment A to fragment B. The question is whether A and B are the same person,
and it is asked as a distance between their mean torso colours, against two references
built from the same statistic on the same clip:

    within   one single-stage track, first half against second half. Same person by
             construction -- the single-stage tracker never associates across the low
             band, so it has no way to make this mistake. This is what one shopper's
             colour does over time under this camera's lighting.
    between  two single-stage tracks that are **co-present in at least one frame**.
             Different people by construction: one person is not two boxes in a frame.

So "same person" and "different person" both have a measured distribution before any join
is looked at, and a join is placed against them. No threshold is chosen; the overlap of
the two references is itself part of the answer, and if they overlap badly this
instrument cannot settle the question and says so.

**What this cannot see.** Two shoppers in the same colour are the same nine numbers, so
this counts joins it can *prove* wrong and cannot certify the rest -- a difference is
evidence, a match is not. The `between` reference is what prices that: it says how often
two genuinely different people in this shop are indistinguishable by colour.

**Crops come from the distorted frame.** Boxes are undistorted for tracking, which moves
them off the pixels they were found on; the colour is read from the box as detected. Both
are kept per detection so neither reader has to know about the other.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from syncai_hydranet.analytics.appearance import (
    torso_region,
    torso_stats,
)
from syncai_hydranet.analytics.bytetrack import OfflineForward
from syncai_hydranet.analytics.clip_tracks import (
    PERSON,
    to_source_pixels,
    undistort_boxes,
)
from syncai_hydranet.analytics.delivery import report_settings
from syncai_hydranet.analytics.tracker import Tracker, iou
from syncai_hydranet.data.video import frames, probe
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.serving.camera import BIRTH_REF
from syncai_hydranet.shipped import load_model
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path(__file__).resolve().parent.parent


COMMISSIONED = ROOT / "runs/commission01"
_SWEEP = json.loads((ROOT / "configs/sweep_clips.json").read_text())
CLIPS = ROOT / _SWEEP["corpus_root"]
SWEEP_CLIPS: dict[str, str] = _SWEEP["clips"]

# Enough observations to have two halves worth comparing. Below this a "first half" is
# one or two boxes and its mean is a single frame's lighting, which would widen the
# `within` reference with noise and make every join look ordinary.
MIN_OBS_FOR_HALVES = 6

# The pose head's config name. `configs/hydranet_retail_pose01.yaml` states that it must
# be called this, so reading it back by the same name is not a coincidence to protect.
POSE = "pose"


def detect(camera: str, model, cfg, device, args):
    """One inference pass: every person detection at the low band, with its torso colour.

    Returns per frame a list of `(box_undistorted, score, colour)`. The colour is read
    before undistortion, from the pixels the box was actually found on.
    """
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    k1 = cam_file.lens.k1 if cam_file.lens else None
    clip = str(CLIPS / camera / SWEEP_CLIPS[camera])
    src_w, src_h, _ = probe(clip)
    per_frame: list[dict] = []
    thumb = args.render_joins is not None
    by_keypoint = by_band = 0
    for n, frame in enumerate(frames(clip, src_w, src_h, args.fps)):
        if args.frames and n >= args.frames:
            break
        x, _, region = preprocess(Image.fromarray(frame), cfg["data"]["input_size"])
        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=args.low_thr)
        det = res["detection"][0]
        if len(det.get("labels", [])):
            lab = det["labels"].cpu().numpy()
            keep = lab == PERSON
            box = to_source_pixels(det["boxes"].cpu().numpy()[keep], region, src_w, src_h)
            sc = det["scores"].cpu().numpy()[keep]
            # `decode_into` aligns keypoints with *every* decoded box, so the person mask
            # has to be applied to both or a shopper gets a shelf's skeleton.
            has_kp = POSE in res and len(res[POSE][0])
            kp = res[POSE][0].cpu().numpy()[keep] if has_kp else np.zeros((len(box), 17, 3))
            kp = _kps_to_source(kp, region, src_w, src_h)
        else:
            box, sc, kp = np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
        rgb = frame.astype(np.float32) / 255.0
        colour, anchored = _colours(rgb, box, kp)
        by_keypoint += anchored
        by_band += len(box) - anchored
        per_frame.append(
            {
                "boxes": undistort_boxes(box, k1, src_w, src_h) if k1 is not None else box,
                "scores": sc,
                "colour": colour,
                # Kept only when a render is asked for: a clip's worth of thumbnails is
                # ~100 MB, which is fine for one camera and not fine as a default.
                "thumbs": [_thumb(frame, b) for b in box] if thumb else [],
            }
        )
    return per_frame, (src_w, src_h), {"by_keypoint": by_keypoint, "by_band": by_band}


def _kps_to_source(kp: np.ndarray, region, src_w: int, src_h: int) -> np.ndarray:
    """Canvas keypoints -> source pixels, through the box mapping rather than beside it.

    Each point is packed as a degenerate box `(x, y, x, y)` so the letterbox arithmetic is
    `to_source_pixels`'s and not a second copy of it. `clip_tracks` states why that matters:
    getting the padding wrong does not look wrong, it just puts everything a bar's width out.
    """
    if not len(kp):
        return kp
    xy = kp[..., :2].reshape(-1, 2)
    packed = np.concatenate([xy, xy], axis=1)
    out = to_source_pixels(packed, region, src_w, src_h)[:, :2]
    return np.concatenate([out.reshape(*kp.shape[:-1], 2), kp[..., 2:]], axis=-1)


def _colours(rgb: np.ndarray, boxes: np.ndarray, kp: np.ndarray) -> tuple[np.ndarray, int]:
    """Torso colour per detection, anchored to the skeleton where there is one.

    Returns the stack and **how many used the keypoints**. The count is reported rather
    than kept internal because the fallback is a different measurement, not a degraded
    one: `TORSO_BAND` is a fraction of the box and moves off the shirt when the box
    reframes, which is the defect this whole path exists to remove.
    """
    if not len(boxes):
        return np.zeros((0, 9)), 0
    out, anchored = [], 0
    for i, b in enumerate(boxes):
        region = torso_region(kp[i]) if i < len(kp) else None
        if region is None:
            out.append(_crop_stats(rgb, b))
        else:
            out.append(_crop_stats(rgb, np.asarray(region)))
            anchored += 1
    return np.stack(out), anchored


THUMB = (96, 48)  # h, w -- big enough to see a shirt colour and a posture, no bigger


def _thumb(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (round(float(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((*THUMB, 3), dtype=np.uint8)
    im = Image.fromarray(frame[y0:y1, x0:x1]).resize(
        (THUMB[1], THUMB[0]), Image.Resampling.BILINEAR
    )
    return np.asarray(im, dtype=np.uint8)


def render_join(per_frame, pa, pb, path: Path, n: int = 5) -> None:
    """The last crops before a join beside the first crops after it.

    Nine numbers say these two fragments are further apart than most pairs of genuinely
    different shoppers. That is a reason to look, not a verdict: this project's standing
    rule is that a gate decided by eye has to be lookable-at, and a colour distance is
    exactly the kind of evidence that is wrong in a way only a frame shows -- a shopper
    who takes off a coat is the same person and a large distance.
    """
    left = [per_frame[f]["thumbs"][d] for f, d in pa[-n:]]
    right = [per_frame[f]["thumbs"][d] for f, d in pb[:n]]
    gap = np.full((THUMB[0], 6, 3), 220, dtype=np.uint8)
    row = [*left, gap, *right] if left and right else [*left, *right]
    Image.fromarray(np.concatenate(row, axis=1)).save(path)


def _crop_stats(rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (round(float(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(9, dtype=np.float32)
    return torso_stats(rgb[y0:y1, x0:x1])


def run_trackers(per_frame, args):
    """Both trackers over the same detections, each observation tagged with its index.

    The tag is what makes the two runs comparable at all: a track from either tracker
    becomes a list of `(frame, detection index)`, so a two-stage track can be read as a
    sequence of single-stage identities without matching boxes across two coordinate
    conventions.
    """
    single = Tracker(iou_threshold=args.iou, max_age=args.max_age, min_hits=args.min_hits)
    two = OfflineForward(
        high_thr=args.score_thr, low_thr=args.low_thr, iou_thr=args.iou,
        iou_thr_low=args.iou_low, max_age=args.max_age, min_hits=args.min_hits,
        vel_scale=25.0 / args.fps,
    )  # fmt: skip
    for n, f in enumerate(per_frame):
        hi = f["scores"] >= args.score_thr
        single.update(f["boxes"][hi], n, scores=f["scores"][hi])
        two.update(f["boxes"], f["scores"], n)
    return single.finished(), two.finished()


def index_of(per_frame, frame: int, box: np.ndarray) -> int | None:
    """Which detection this observed box was. Exact in principle, by IoU in practice."""
    cand = per_frame[frame]["boxes"]
    if not len(cand):
        return None
    m = iou(box.reshape(1, 4), cand)[0]
    return int(np.argmax(m)) if m.max() >= 0.99 else None


def occurrences(tracks, per_frame) -> dict[int, list[tuple[int, int]]]:
    """Each track as `(frame, detection index)` pairs, dropping anything unmatchable."""
    out = {}
    for t in tracks:
        tid = getattr(t, "track_id", None)
        tid = int(t.frag_id) if tid is None else int(tid)
        pairs = []
        for fr, bx in zip(t.frames, t.boxes, strict=True):
            di = index_of(per_frame, int(fr), np.asarray(bx, dtype=float))
            if di is not None:
                pairs.append((int(fr), di))
        if pairs:
            out[tid] = pairs
    return out


def scale(per_frame) -> tuple[np.ndarray, np.ndarray]:
    """Standardise on this clip's own detections, so a distance is in units of variation.

    Per clip rather than per fleet: the nine statistics are absolute colours and a shop's
    lighting sets their spread, so a fleet-wide scale would make a dim camera's ordinary
    variation look like an identity change.
    """
    all_c = np.concatenate([f["colour"] for f in per_frame if len(f["colour"])])
    return all_c.mean(0), all_c.std(0) + 1e-6


def mean_colour(per_frame, pairs, mu, sd) -> np.ndarray:
    return np.mean([(per_frame[f]["colour"][d] - mu) / sd for f, d in pairs], axis=0)


def references(single_occ, per_frame, mu, sd):
    """The two distributions a join is read against. Same statistic, same clip."""
    within, between = [], []
    for pairs in single_occ.values():
        if len(pairs) < MIN_OBS_FOR_HALVES:
            continue
        h = len(pairs) // 2
        a = mean_colour(per_frame, pairs[:h], mu, sd)
        b = mean_colour(per_frame, pairs[h:], mu, sd)
        within.append(float(np.linalg.norm(a - b)))
    ids = sorted(single_occ)
    seen = {i: {f for f, _ in single_occ[i]} for i in ids}
    means = {i: mean_colour(per_frame, single_occ[i], mu, sd) for i in ids}
    for x, i in enumerate(ids):
        for j in ids[x + 1 :]:
            if seen[i] & seen[j]:
                between.append(float(np.linalg.norm(means[i] - means[j])))
    return within, between


def joins(two_occ, single_occ, per_frame, mu, sd) -> list[dict]:
    """Every place a two-stage track stitched one single-stage fragment to another."""
    owner = {}
    for tid, pairs in single_occ.items():
        for f, d in pairs:
            owner[(f, d)] = tid
    out = []
    for frag, pairs in two_occ.items():
        runs: list[tuple[int, list]] = []
        for f, d in pairs:
            who = owner.get((f, d))
            if who is None:  # a low-band observation the shipped tracker never saw
                continue
            if runs and runs[-1][0] == who:
                runs[-1][1].append((f, d))
            else:
                runs.append((who, [(f, d)]))
        for (ia, pa), (ib, pb) in pairwise(runs):
            a = mean_colour(per_frame, pa, mu, sd)
            b = mean_colour(per_frame, pb, mu, sd)
            out.append(
                {
                    "frag_id": int(frag),
                    "from_track": int(ia),
                    "to_track": int(ib),
                    "frame_a": int(pa[-1][0]),
                    "frame_b": int(pb[0][0]),
                    "obs_a": len(pa),
                    "obs_b": len(pb),
                    "distance": float(np.linalg.norm(a - b)),
                    "_pa": pa,
                    "_pb": pb,
                }
            )
    return out


def _split(js: list[dict], between: list[float]) -> dict:
    """Joins that clear the observation floor on both sides, and the flags among them."""
    if not between:
        return {"reason": "no between-person reference on this camera"}
    m = MIN_OBS_FOR_HALVES
    ok = [j for j in js if j["obs_a"] >= m and j["obs_b"] >= m]
    return {
        "joins": len(js),
        "both_sides_observed": len(ok),
        "flagged_of_those": sum(j["pct_between"] >= 0.5 for j in ok),
        "one_side_a_stub": len(js) - len(ok),
        "flagged_of_stubs": sum(
            j["pct_between"] >= 0.5 for j in js if j["obs_a"] < m or j["obs_b"] < m
        ),
    }


def pct(values: list[float], x: float) -> float:
    """Fraction of `values` below `x`; the position of one join in a reference."""
    return float(np.mean(np.asarray(values) < x)) if values else float("nan")


def summarise(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {
        "n": len(v),
        "p10": round(float(np.percentile(a, 10)), 3),
        "p50": round(float(np.median(a)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
    }


def run_camera(camera: str, model, cfg, device, args) -> dict:
    per_frame, src, anchor = detect(camera, model, cfg, device, args)
    single_tracks, two_tracks = run_trackers(per_frame, args)
    mu, sd = scale(per_frame)
    single_occ = occurrences(single_tracks, per_frame)
    two_occ = occurrences(two_tracks, per_frame)
    within, between = references(single_occ, per_frame, mu, sd)
    js = joins(two_occ, single_occ, per_frame, mu, sd)
    for j in js:
        j["pct_within"] = round(pct(within, j["distance"]), 3)
        j["pct_between"] = round(pct(between, j["distance"]), 3)
        j["distance"] = round(j["distance"], 3)
        pa, pb = j.pop("_pa"), j.pop("_pb")
        if args.render_joins is not None and j["pct_between"] >= args.render_joins:
            name = f"{camera}_frag{j['frag_id']}_{j['from_track']}to{j['to_track']}.png"
            render_join(per_frame, pa, pb, args.out / name)
    return {
        "camera": camera,
        "src": list(src),
        "frames": len(per_frame),
        "detections": int(sum(len(f["boxes"]) for f in per_frame)),
        # How many torso colours came from a skeleton and how many fell back to the
        # box fraction. A run where the fallback dominates has measured the old thing.
        "torso_anchor": anchor,
        "single_tracks": len(single_occ),
        "two_stage_tracks": len(two_occ),
        "within": summarise(within),
        "between": summarise(between),
        # **The floor that makes a join comparable with the references at all.** `within`
        # is built from halves of tracks with at least MIN_OBS_FOR_HALVES observations, so
        # a join whose segment is two boxes is being read against a scale it was never on:
        # a 2-observation mean colour is one moment's lighting, not a person's colour.
        # Reported as a split rather than applied as a filter, because "46% of joins cannot
        # be judged this way" is a result and a silently shorter list is not.
        "comparable": _split(js, between),
        # Every join with its identities, not only the count: the count answers whether
        # two-stage is safe, and the identities answer which clip to go and look at.
        "joins": js,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/identity01"))
    ap.add_argument("--cameras", nargs="*", default=sorted(SWEEP_CLIPS))
    ap.add_argument("--config", default="configs/hydranet_retail_pose01.yaml")
    ap.add_argument(
        "--checkpoint",
        # pose01's weights, not pose02's better ones, on purpose: this run has to be
        # comparable with `runs/endings05/`, which is the measurement it argues about.
        default="runs/hydranet_retail_security_b03_cw_xl-20260825-162131/last.pt",
    )
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--score-thr", type=float, default=BIRTH_REF)
    ap.add_argument("--low-thr", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--iou-low", type=float, default=0.5)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument(
        "--render-joins",
        type=float,
        default=None,
        help="save a crop strip for every join at or past this `pct_between`",
    )
    args = ap.parse_args()

    model, cfg, device = load_model(args.config, args.checkpoint)

    args.out.mkdir(parents=True, exist_ok=True)
    out = []
    for camera in args.cameras:
        r = run_camera(camera, model, cfg, device, args)
        out.append(r)
        s = r["comparable"]
        ok = f"{s.get('flagged_of_those', '?')}/{s.get('both_sides_observed', '?')}"
        stub = f"{s.get('flagged_of_stubs', '?')}/{s.get('one_side_a_stub', '?')}"
        print(
            f"{camera}: {r['single_tracks']} -> {r['two_stage_tracks']} tracks, "
            f"{len(r['joins'])} joins -- {ok} flagged where both sides are observed, "
            f"{stub} where one is a stub "
            f"(within p50 {r['within'].get('p50')}, between p50 {r['between'].get('p50')})"
        )
    (args.out / "fleet.json").write_text(
        json.dumps({"settings": report_settings(args), "cameras": out}, indent=1)
    )
    print(f"-> {args.out / 'fleet.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
