#!/usr/bin/env python3
"""Why do tracks end? One answer holds, and one does not -- both are recorded.

    python3 scripts/track_endings.py --out runs/endings03

`runs/journeys01` measured the consequence: 197 visits, a median of 3.2 s, and eight in 24
minutes lasting 30 s or more. The cause is that tracks are shorter than visits, and
`reid_metrics.idf1`/`id_switches` cannot say why -- both need ground-truth tracks, and
that module's own header records that no labelled site clip exists. Labelling one is the
human cost PLAN section 4 refuses by default. A track's own ending is evidence and it is
free.

---------------------------------------------------------------------------
WHAT THIS MEASURES RELIABLY: WHERE THE TRACK DIED

If the last box sits against a frame edge the person walked out and the track was supposed
to end (**exit**). Otherwise the track died in the middle of the view, and a shopper does
not vanish from a shop floor. That split does not depend on any matching rule, and it did
not move across three runs: **80 of 191 endings are exits (42%), so 58% die mid-view.**

Per camera it is not uniform and that is the finding:

    Tao-Hsin-cam03      35 endings    83% died mid-view
    Kaohsiung-cam04     85 endings    78%
    Taichung-cam01      13 endings    31%
    Taichung-cam11       6 endings    17%
    Taichung-cam10      14 endings    14%
    Taichung-cam04      30 endings    13%

Taichung-cam04 loses thirteen percent of its tracks mid-view; Tao-Hsin-cam03 loses
eighty-three. The tracker is the same tracker. Two cameras carry this, and one of them has
had an open person-score investigation since PLAN step 2.

---------------------------------------------------------------------------
WHAT THIS DOES NOT MEASURE, AND THE THREE RUNS THAT ESTABLISHED THAT

A mid-view death is either the tracker dropping someone the detector still sees
(**lost**) or the detector losing them with nothing to reacquire (**gone**). Separating
those needs a rule for "the same person reappeared", and position alone does not carry one:

    flat 1.0 m radius          exit 80   lost  72   gone 39
    flat 1.0 m radius (rerun)  exit 80   lost  72   gone 39
    0.35 m per frame of gap    exit 80   lost 101   gone 10

`exit` is identical every time. The other two trade places wholesale, because both rules
are wrong in ways that only showed up in the detail. The radius matched the **neighbouring
shopper**: its "reappearances" sat a median 0.71 m away one frame later, which is 3.5 m/s,
and on a camera carrying 87 tracks in three minutes there is always somebody within a
metre. The speed limit fixed that at gap 1 and broke the far end -- 0.35 m per frame is
1.05 m at gap 3 and 3.5 m at gap 10, wider than the radius it replaced, across most of a
small shop.

**The honest conclusion is that this question is not decidable from geometry.** Telling
"the same person, one metre on" from "a different person, one metre away" is what an
appearance model is for; `reid_metrics.cmc_map` is the metric for one and PLAN step 5 is
where it belongs. `lost` and `gone` are still reported, and `lost_detail` carries every
gap, IoU and floor distance, so a later rule can be tried against the same endings -- but
nothing should be concluded from that split until something can tell two shoppers apart.

---------------------------------------------------------------------------
THE WITNESS PASS: A QUESTION THAT NEEDS NO IDENTITY

Added 2026-08-27, and the reason it is a different question rather than a fourth matching
rule: `lost`/`gone` asks *who* reappeared, and that is what moved 72/39 to 101/10. This
asks only **whether anybody was still standing where the track died**, over the three
frames after it -- short on purpose, because a long window finds a different shopper
walking into the spot and calls it the same one.

A second pass runs the model at 0.03 over exactly those frames. Pass one is untouched, so
`exit` still reproduces; the two passes are separate because `decode` thresholds *before*
NMS, and filtering a 0.03 decode down to 0.35 afterwards is not the same set of survivors.

    taken       the box below was observed by another live track on that frame. The
                person next to the dead track has a track; the dead track's person was
                demoted (added 2026-09-10; before it, this read as `available`).
    available   a box at or above the shipped threshold overlapped the dead track by the
                **tracker's own** IoU rule and was still not associated. Judged at
                `--iou`, not at the looser `--witness-iou` the other bins use: a box the
                tracker legitimately refused is not a tracker failure, and scoring this
                bin loosely turns a threshold difference into an accusation.
    demoted     a box is there, below the threshold. The person was seen; the score was
                not enough. The fix is whatever depressed that score -- PLAN section 7.11.
    boxless     the dense head still marks a person and the box head emits nothing at all,
                even at 0.03. No threshold reaches this one.
    vacated     the dense head sees nobody either, so the ending is not a detection
                failure: the shopper stepped behind a fixture or out of view.

Everything is compared in the undistorted source pixels the tracker itself used. Dense
components reach that space through the same `to_source_pixels` -> `undistort_boxes` chain
the boxes take, because a mask resized into distorted pixels and compared against
undistorted boxes lands plausibly and is wrong by the lens.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from syncai_hydranet.analytics.appearance import torso_histograms
from syncai_hydranet.analytics.bytetrack import OfflineForward
from syncai_hydranet.analytics.clip_tracks import (
    PERSON,
    to_source_pixels,
    track_clip,
    undistort_boxes,
)
from syncai_hydranet.analytics.delivery import report_settings
from syncai_hydranet.analytics.tracker import Tracker, iou
from syncai_hydranet.data.video import frames, probe
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points
from syncai_hydranet.serving.camera import BIRTH_REF
from syncai_hydranet.serving.decode import confirm_hook
from syncai_hydranet.shipped import load_model
from syncai_hydranet.utils.device import pick_device
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path(__file__).resolve().parent.parent


COMMISSIONED = ROOT / "runs/commission01"

# The eight clips every fleet measurement of 2026-08-26 ran on. Read from
# `configs/sweep_clips.json` rather than written here, and rather than imported from a
# neighbouring script: what is shared is a corpus *selection*, which is data, and
# `tests/test_scripts_are_not_libraries.py` refuses one script importing another for the
# reason `clip_tracks.py` records -- four copies of one loop that stopped agreeing. A
# dataset path has no business in the wheel either, so the file is the right home for it.
_SWEEP = json.loads((ROOT / "configs/sweep_clips.json").read_text())
CLIPS = ROOT / _SWEEP["corpus_root"]
SWEEP_CLIPS: dict[str, str] = _SWEEP["clips"]


class Recording(Tracker):
    """A `Tracker` that keeps every frame's detections, so endings can be explained.

    Subclassed rather than wrapped because `track_clip` calls `update` and `finished` and
    nothing else; a wrapper would have to forward both and would drift the first time the
    loop learned a third method.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen: dict[int, np.ndarray] = {}
        self.taken: dict[int, list[tuple[int, np.ndarray]]] = {}

    def update(
        self,
        boxes,
        frame_idx,
        keypoints=None,
        scores=None,
        staff_scores=None,
        appearance=None,
    ):
        # Every optional argument the base takes has to appear here, forwarded untouched.
        # An override that drops one is not "ignoring a feature this script does not use":
        # it silently narrows what `track_clip` may pass, and the caller cannot see that a
        # `Tracker` it was handed is a `Tracker` with a smaller signature. `staff_scores`
        # arrived on 2026-08-28 and this override went stale in the same commit -- caught
        # by `scripts/ty_ratchet.sh`, and by nothing else, because a Python call with an
        # unexpected keyword fails only on the run that first passes it. `appearance`
        # arrived on 2026-09-09 and did it again, caught the same way.
        self.seen[int(frame_idx)] = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
        out = super().update(
            boxes,
            frame_idx,
            keypoints=keypoints,
            scores=scores,
            staff_scores=staff_scores,
            appearance=appearance,
        )
        # Which live tracks were OBSERVED on this frame, and where: the witness pass asks
        # whether a box near a dead track was already somebody else's.
        self.taken[int(frame_idx)] = [
            (t.track_id, np.asarray(t.boxes[-1], float).copy())
            for t in self.tracks
            if t.age == 0 and t.boxes
        ]
        return out


@dataclass(frozen=True)
class Ended:
    """The three fields an ending is judged from, for either tracker."""

    track_id: int
    frames: list[int]
    boxes: list[np.ndarray]
    scores: list[float]


class RecordingByteTrack:
    """`bytetrack.OfflineForward` behind the interface `track_clip` calls.

    Wrapped rather than subclassed, the opposite of `Recording` above, because the two
    disagree about argument order -- `OfflineForward.update` takes `(boxes, scores,
    frame_idx)` and `track_clip` calls `update(boxes, frame_idx, scores=...)`. A subclass
    would have to override `update` to swap them, which is a wrapper with extra steps and
    an inherited method that silently means something else.

    **This changes two things at once and the comparison has to say so.** `bytetrack.py`
    brings a high/low band *and* a Kalman filter, and `tracker.py` refuses the Kalman on
    stated grounds (no measured noise model on this footage). So a run with this tracker
    is not "the same tracker with hysteresis"; it is the second tracker, and what it
    measures is the comparison `bytetrack.py`'s own header calls open and unmeasured.

    `seen` records the **high band only**, so the `lost`/`gone` split stays comparable
    with a single-threshold run. Feeding it the low band as well would grow the
    reappearance population and move that split for a reason that is not the tracker.
    """

    def __init__(
        self,
        *,
        high_thr,
        low_thr,
        iou_thr,
        iou_thr_low,
        max_age,
        min_hits,
        vel_scale,
        dense_birth_thr=None,
    ):
        self.inner = OfflineForward(
            high_thr=high_thr, low_thr=low_thr, iou_thr=iou_thr, iou_thr_low=iou_thr_low,
            max_age=max_age, min_hits=min_hits, vel_scale=vel_scale,
            dense_birth_thr=dense_birth_thr,
        )  # fmt: skip
        self.high_thr = high_thr
        self.seen: dict[int, np.ndarray] = {}
        self.taken: dict[int, list[tuple[int, np.ndarray]]] = {}

    def update(self, boxes, frame_idx, keypoints=None, scores=None, confirmed=None):  # noqa: ARG002
        # `keypoints` is part of the interface `track_clip` may call with and this
        # tracker has nothing to do with them; naming it is what keeps the two
        # implementations substitutable.
        b = np.asarray(boxes, dtype=float).reshape(-1, 4)
        s = np.zeros(len(b)) if scores is None else np.asarray(scores, dtype=float).reshape(-1)
        self.seen[int(frame_idx)] = b[s >= self.high_thr].copy()
        self.inner.update(b, s, int(frame_idx), confirmed=confirmed)
        self.taken[int(frame_idx)] = [
            (t.frag_id, np.asarray(t.boxes[-1], float).copy())
            for t in self.inner.tracks
            if t.age == 0 and t.boxes
        ]

    def finished(self) -> list[Ended]:
        # A `Fragment` names its id `frag_id`, and everything downstream reads
        # `track_id`. Rebuilt rather than assigned onto the Fragment: setting an
        # attribute a dataclass does not declare is invisible to the type checker and to
        # the next reader, and only the three fields below are ever used.
        return [Ended(f.frag_id, f.frames, f.boxes, f.scores) for f in self.inner.finished()]


def foot_m(boxes: np.ndarray, cam_file: CameraFile, src) -> np.ndarray:
    """Box bottom-centres to floor metres, through this camera's own calibration."""
    if not len(boxes):
        return np.zeros((0, 2))
    pts = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    w, h = cam_file.image_size_px
    pts = pts * np.array([w / src[0], h / src[1]])
    if cam_file.lens is not None:
        lens = cam_file.lens
        pts = undistort_points(pts, lens.k1, lens.centre_px, lens.radius_px)
    x, z = pixel_to_ground(pts[:, 0], pts[:, 1], cam_file.camera, cam_file.plane)
    return np.stack([x, z], axis=1)


def classify(track, tracker: Recording | RecordingByteTrack, cam_file, src, args) -> dict:
    """The ending, plus what a `lost` one costs to recover.

    A `lost` ending is two different failures wearing one label, and they have different
    fixes. If the person reappears on the **next** frame with a box that overlaps the dead
    track, the tracker had an easy match and did not take it -- an association failure,
    and the association logic is the thing to change. If the reappearance is six or more
    frames later, the track was killed by `max_age`, which at 5 fps is one second of
    coasting: the detector lost the person for longer than the tracker is willing to wait,
    and no association rule reaches that. So `gap` and `iou` come back with the verdict.
    """
    last_frame = track.frames[-1]
    box = np.asarray(track.boxes[-1], dtype=float)
    w, h = src
    edge = (
        box[0] <= args.edge_px
        or box[1] <= args.edge_px
        or box[2] >= w - args.edge_px
        or box[3] >= h - args.edge_px
    )
    if edge:
        return {"why": "exit"}
    here = foot_m(box[None], cam_file, src)[0]
    if not np.isfinite(here).all():
        return {"why": "gone"}
    for f in range(last_frame + 1, last_frame + 1 + args.gap):
        boxes = tracker.seen.get(f)
        if boxes is None or not len(boxes):
            continue
        d = np.linalg.norm(foot_m(boxes, cam_file, src) - here, axis=1)
        # The speed limit, not a radius: how far this person could have walked since
        # their last observation. See the module docstring for what a radius did instead.
        if not np.isfinite(d).any() or float(np.nanmin(d)) > args.speed * (f - last_frame):
            continue
        j = int(np.nanargmin(d))
        return {
            "why": "lost",
            # Frames from the last observation to the reappearance. `max_age` is the
            # tracker's patience; a gap larger than it was never recoverable by matching.
            "gap": f - last_frame,
            "iou": round(float(iou(box[None], boxes[j : j + 1])[0, 0]), 3),
            "metres": round(float(d[j]), 2),
        }
    return {"why": "gone"}


def witness_verdict(
    *,
    best_assoc: float,
    best_score: float,
    dense: bool,
    score_thr: float,
    witness_thr: float,
    taken_by: int | None = None,
) -> str:
    """The four-way rule, as a function so it can be tested without a model or a clip.

    The order is the argument. `available` is checked first and against `best_assoc`,
    which is measured at the **tracker's own** IoU: a box the tracker could have taken and
    did not is a tracker failure, and it outranks every detector verdict because no
    detector change would have saved that track. Everything below it is measured at the
    looser presence IoU, where the question is only whether somebody was there.
    """
    if best_assoc >= score_thr:
        # `taken`: the box the tracker "could have taken" was observed by ANOTHER live
        # track on that frame. Added 2026-09-10 after the first strips of `available`
        # deaths (runs/endings08) showed the same thing thirteen times: a huddle at a
        # counter, the dead track's last box in the low band, and the high-band box next
        # to it already on a neighbour's track. That is the detector demoting a person
        # in a crowd (PLAN 7.11), not the tracker refusing a match, and it was being
        # counted as the tracker's failure.
        return "taken" if taken_by is not None else "available"
    if best_score >= witness_thr:
        return "demoted"
    return "boxless" if dense else "vacated"


def witness(
    clip: str, model, cfg, device, args, targets: dict, k1: float | None, taken: dict
) -> dict:
    """A second pass that asks, at each mid-view death, whether anybody was still there.

    The `lost` / `gone` split above is undecidable from geometry because it needs an
    identity: "is the person a metre away the same person". This asks a **local** question
    instead, which needs no identity at all -- at the frames right after a track died, was
    a person still standing where it died, and did the box head emit anything on them?

    Three verdicts, and they have three different fixes:

      demoted   a box is there, below the shipped threshold. The person was seen and the
                score was not enough, so the fix is whatever is depressing that score.
      boxless   the dense head still marks a person and the box head emits nothing at all,
                even at 0.03. A threshold cannot reach this one.
      vacated   the dense head does not see a person either. The track ending is then not
                a detection failure -- the shopper is behind a fixture, or out of view.

    `available` is the fourth and is a tracker failure rather than a detector one: a box at
    or above the shipped threshold was sitting on the dead track and was not associated.

    Everything is compared in the **undistorted source pixels the tracker itself used**.
    Dense components are put there by the same `to_source_pixels` -> `undistort_boxes`
    chain the boxes take, rather than by a second copy of that arithmetic: a mask resized
    into distorted pixels and compared against undistorted boxes lands plausibly and is
    wrong by the lens.
    """
    src_w, src_h, _ = probe(clip)
    seg_person = list(cfg["data"]["terrain_classes"]).index("person")
    person_label = PERSON
    out: dict[int, dict] = {}
    n = 0
    for frame in frames(clip, src_w, src_h, args.fps):
        if args.frames and n >= args.frames:
            break
        want = targets.get(n)
        n += 1
        if not want:
            continue
        x, _canvas, region = preprocess(Image.fromarray(frame), cfg["data"]["input_size"])
        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=args.witness_thr)
        det = res["detection"][0]
        low = np.zeros((0, 4))
        low_sc = np.zeros(0)
        if len(det.get("labels", [])):
            lab = det["labels"].cpu().numpy()
            keep = lab == person_label
            low = to_source_pixels(det["boxes"].cpu().numpy()[keep], region, src_w, src_h)
            low_sc = det["scores"].cpu().numpy()[keep]
            if k1 is not None and len(low):
                low = undistort_boxes(low, k1, src_w, src_h)
        cmap = res["terrain"][0].cpu().numpy()
        lab_img, ncomp = ndimage.label(cmap == seg_person)
        dense = np.zeros((0, 4))
        if ncomp:
            sizes = ndimage.sum(cmap == seg_person, lab_img, range(1, ncomp + 1))
            boxes_c = []
            for c in np.nonzero(sizes >= args.witness_blob)[0]:
                ys, xs = np.nonzero(lab_img == c + 1)
                boxes_c.append([xs.min(), ys.min(), xs.max(), ys.max()])
            if boxes_c:
                dense = to_source_pixels(np.array(boxes_c, dtype=float), region, src_w, src_h)
                if k1 is not None:
                    dense = undistort_boxes(dense, k1, src_w, src_h)

        for tid, box in want:
            rec = out.setdefault(
                tid, {"best_score": 0.0, "best_assoc": 0.0, "best_iou": 0.0,
                      "dense": False, "frames": 0}
            )  # fmt: skip
            rec["frames"] += 1
            if len(low):
                ov = iou(box[None], low)[0]
                # Two thresholds on purpose. The loose one asks whether anybody is there;
                # the tracker's own asks whether the tracker could have taken this box.
                # Judging the second with the first turns a threshold difference into an
                # accusation, which is what an earlier revision of this pass did.
                hit = ov >= args.witness_iou
                if hit.any():
                    rec["best_score"] = max(rec["best_score"], float(low_sc[hit].max()))
                    rec["best_iou"] = max(rec["best_iou"], float(ov[hit].max()))
                assoc = ov >= args.iou
                if assoc.any():
                    j = int(np.flatnonzero(assoc)[np.argmax(low_sc[assoc])])
                    if float(low_sc[j]) > rec["best_assoc"]:
                        # The box the tracker could have taken, and the frame it was on.
                        rec["best_assoc_box"] = [round(float(x), 1) for x in low[j]]
                        rec["best_assoc_frame"] = n - 1
                        # Was it somebody else's? A live track other than the dead one,
                        # observed on this frame, whose box overlaps it at the tracker's
                        # own IoU.
                        rec["taken_by"] = None
                        for other_id, other_box in taken.get(n - 1, []):
                            if other_id == tid:
                                continue
                            if float(iou(low[j][None], other_box[None])[0, 0]) >= args.iou:
                                rec["taken_by"] = int(other_id)
                                break
                    rec["best_assoc"] = max(rec["best_assoc"], float(low_sc[assoc].max()))
            if len(dense) and (iou(box[None], dense)[0] >= args.witness_iou).any():
                rec["dense"] = True

    for rec in out.values():
        rec["verdict"] = witness_verdict(
            best_assoc=rec["best_assoc"],
            best_score=rec["best_score"],
            dense=rec["dense"],
            score_thr=args.score_thr,
            witness_thr=args.witness_thr,
            taken_by=rec.get("taken_by"),
        )
        for k in ("best_score", "best_assoc", "best_iou"):
            rec[k] = round(rec[k], 3)
    return out


def run_camera(camera: str, model, cfg, device, args) -> dict:
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    seg_person = list(cfg["data"]["terrain_classes"]).index("person")
    if args.two_stage:
        tracker: Recording | RecordingByteTrack = RecordingByteTrack(
            high_thr=args.score_thr,
            low_thr=args.low_thr,
            iou_thr=args.iou,
            iou_thr_low=args.iou_low,
            max_age=args.max_age,
            min_hits=args.min_hits,
            # MOT17's velocity prior is for 25 fps; these clips sample at args.fps.
            vel_scale=25.0 / args.fps,
            dense_birth_thr=args.dense_birth_thr if args.dense_birth else None,
        )
        # The low band has to reach the tracker, so the decode floor drops to it. Births
        # still happen at `--score-thr` inside the tracker, which is the whole point: a
        # looser decode alone would also invent tracks, and that arm was measured on
        # 2026-08-26 and made the event layer worse. `--dense-birth` is the arm where a
        # lower box may be born after all, on the dense head's word (PLAN 7a.41).
        decode_thr = args.dense_birth_thr if args.dense_birth else args.low_thr
    else:
        # `--band` is the survival band alone -- birth at --score-thr, survival to
        # --low-thr, no Kalman, association untouched -- the arm runs/band_probe01
        # measured at 91% of two-stage's gain (PLAN 7.11) and the tracker step 6 and
        # serving run. `--appearance` adds the re-association gate on top, at the
        # camera's own calibrated distance; a camera with none runs ungated and is
        # named in the row, so a fleet total is never read as a gated one.
        gate = None
        if args.appearance:
            gate = cam_file.appearance_thr
        tracker = Recording(
            iou_threshold=args.iou,
            max_age=args.max_age,
            min_hits=args.min_hits,
            birth_thr=args.score_thr if args.band else None,
            appearance_thr=gate,
        )
        decode_thr = args.low_thr if args.band else args.score_thr
    out = track_clip(
        str(CLIPS / camera / SWEEP_CLIPS[camera]),
        model,
        cfg["data"]["input_size"],
        device,
        tracker,
        frames=frames,
        preprocess=preprocess,
        probe=probe,
        fps=args.fps,
        score_thr=decode_thr,
        k1=cam_file.lens.k1 if cam_file.lens else None,
        max_frames=args.frames,
        describe=torso_histograms if args.appearance else None,
        confirm=confirm_hook(seg_person) if args.dense_birth else None,
    )
    src = (out.src_w, out.src_h)
    counts = {"exit": 0, "lost": 0, "gone": 0}
    lost: list[dict] = []
    lengths: dict[str, list[int]] = {k: [] for k in counts}
    # Every mid-view death, and the frames the witness pass has to look at for it.
    targets: dict[int, list] = {}
    mid_view: list[int] = []
    # Where each mid-view death happened, so it can be looked at: the track's last
    # observed frame and box (source pixels, undistorted). Until 2026-09-10 the
    # instrument recorded verdicts and no frames, and the 21 "available" deaths of the
    # band arm -- a box the tracker could have taken and did not -- could not be pulled.
    deaths: dict[str, dict] = {}
    for t in out.tracks:
        if not t.frames:
            continue
        # A track still alive at the last frame has not ended; its ending is unobserved.
        if t.frames[-1] >= out.frames - 1 - args.max_age:
            continue
        v = classify(t, tracker, cam_file, src, args)
        counts[v["why"]] += 1
        lengths[v["why"]].append(len(t.frames))
        if v["why"] == "lost":
            lost.append({"track_id": t.track_id, **{k: v[k] for k in ("gap", "iou", "metres")}})
        if v["why"] != "exit":
            mid_view.append(t.track_id)
            box = np.asarray(t.boxes[-1], dtype=float)
            deaths[str(t.track_id)] = {
                "last_frame": int(t.frames[-1]),
                "last_box": [round(float(x), 1) for x in box],
                "length": len(t.frames),
                "score_last": round(float(t.scores[-1]), 3) if t.scores else None,
            }
            for f in range(t.frames[-1] + 1, t.frames[-1] + 1 + args.witness_frames):
                targets.setdefault(f, []).append((t.track_id, box))

    seen = witness(
        str(CLIPS / camera / SWEEP_CLIPS[camera]), model, cfg, device, args, targets,
        cam_file.lens.k1 if cam_file.lens else None, tracker.taken,
    ) if targets else {}  # fmt: skip
    verdicts: dict[str, int] = {}
    for tid in mid_view:
        # A track whose death is past the end of the clip has no frames to witness; it is
        # counted separately rather than folded into `vacated`, which would read as
        # "nobody was there" when the truth is that nobody looked.
        v = seen.get(tid, {}).get("verdict", "unwitnessed")
        verdicts[v] = verdicts.get(v, 0) + 1
    return {
        "camera": camera,
        "tracks": len(out.tracks),
        "endings_judged": sum(counts.values()),
        "counts": counts,
        "median_length": {k: (int(np.median(v)) if v else None) for k, v in lengths.items()},
        # Every lost ending, so the two failures wearing that one label stay countable:
        # a detection available on the next frame that the tracker did not match, against
        # a detector outage longer than `max_age` that no matching rule could reach.
        "lost_detail": lost,
        # What was still standing where each mid-view death happened. Unlike lost/gone this
        # needs no identity rule, so it does not move when the matching rule moves.
        "witness": verdicts,
        "witness_detail": {str(tid): seen[tid] for tid in seen},
        "deaths": deaths,
        "max_age": args.max_age,
        # Which tracker this row was measured under, so a fleet.json cannot be read as
        # one arm when it was three, and an ungated camera in a gated run is visible.
        "arm": (
            "two_stage"
            if args.two_stage
            else "band+appearance"
            if args.band and args.appearance
            else "band"
            if args.band
            else "single"
        ),
        "appearance_thr": (cam_file.appearance_thr if args.appearance else None),
        "dense_births": getattr(getattr(tracker, "inner", None), "dense_births", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/endings01"))
    ap.add_argument("--cameras", nargs="*", default=sorted(SWEEP_CLIPS))
    ap.add_argument("--config", default="configs/hydranet_retail_pose01.yaml")
    ap.add_argument(
        "--checkpoint",
        default="runs/hydranet_retail_security_b03_cw_xl-20260825-162131/last.pt",
    )
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--score-thr", type=float, default=BIRTH_REF)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--gap", type=int, default=10, help="frames to look for a reappearance")
    ap.add_argument(
        "--speed",
        type=float,
        default=0.35,
        metavar="M_PER_FRAME",
        help="how far a person may have walked between frames; 0.35 at 5 fps is 1.75 m/s",
    )
    ap.add_argument("--edge-px", type=float, default=8.0)
    ap.add_argument(
        "--witness-frames",
        type=int,
        default=3,
        help="frames after a mid-view death to look at; short on purpose, because a "
        "different shopper walking into the spot is what a long window would find",
    )
    ap.add_argument("--witness-thr", type=float, default=0.03, help="the witness pass floor")
    ap.add_argument("--witness-iou", type=float, default=0.2)
    ap.add_argument("--witness-blob", type=int, default=400, help="dense person px, canvas")
    ap.add_argument(
        "--two-stage",
        action="store_true",
        help="track with analytics.bytetrack instead: births at --score-thr, survival "
        "down to --low-thr. Moves the Kalman filter as well as the band -- see "
        "RecordingByteTrack",
    )
    ap.add_argument("--low-thr", type=float, default=0.20, help="two-stage survival band")
    ap.add_argument(
        "--dense-birth",
        action="store_true",
        help="two-stage only: an unmatched box in [--dense-birth-thr, --score-thr) is "
        "born when the dense head puts person pixels under it (PLAN 7a.41)",
    )
    ap.add_argument("--dense-birth-thr", type=float, default=0.15)
    ap.add_argument(
        "--band",
        action="store_true",
        help="the survival band alone on the shipped tracker: births at --score-thr, "
        "survival to --low-thr, no Kalman -- the arm step 6 and serving run",
    )
    ap.add_argument(
        "--appearance",
        action="store_true",
        help="gate re-association on the camera's calibrated appearance_thr "
        "(camera.json); a camera with none runs ungated and says so in its row",
    )
    ap.add_argument("--iou-low", type=float, default=0.5, help="two-stage low-band IoU")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(None)
    model, cfg, _ = load_model(args.config, args.checkpoint, device=device)

    fleet, tot = [], {"exit": 0, "lost": 0, "gone": 0}
    wit: dict[str, int] = {}
    for camera in args.cameras:
        row = run_camera(camera, model, cfg, device, args)
        for k, v in row["counts"].items():
            tot[k] += v
        for k, v in row["witness"].items():
            wit[k] = wit.get(k, 0) + v
        fleet.append(row)
        c = row["counts"]
        print(
            f"{camera:<18} {row['endings_judged']:>3} endings   "
            f"exit {c['exit']:>3}   lost {c['lost']:>3}   gone {c['gone']:>3}   "
            + " ".join(f"{k} {v}" for k, v in sorted(row["witness"].items()))
        )
    n = max(sum(tot.values()), 1)
    print(
        f"\nfleet: {n} endings   "
        + "   ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in tot.items())
    )
    m = max(sum(wit.values()), 1)
    print(
        f"mid-view deaths witnessed: {m}   "
        + "   ".join(f"{k} {v} ({100 * v / m:.0f}%)" for k, v in sorted(wit.items()))
    )
    (args.out / "fleet.json").write_text(
        json.dumps(
            {
                "settings": report_settings(args),
                "cameras": fleet,
                "total": tot,
                "witness_total": wit,
            },
            indent=1,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
