"""Tier 2: events that need 17 keypoints per person, from a second-stage crop model.

The tier boundary is not a matter of taste, and the reasoning is worth keeping at the
top of the file that implements it rather than in a section comment two thirds of the
way down a long module.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..stage import TerrainFrame
from ..tracker import Track
from ._geometry import _runs
from ._types import SecurityEvent

# ------------------------------------------------------- behaviour, tier 2: pose
#
# WHY POSE AND NOT A CLIP-LEVEL ACTION CLASSIFIER, WHICH IS WHAT THE ASK SOUNDS LIKE
#
# Three measurements decide it, all of them already in this repository:
#
#   person box height   244-336 px median      a crop encoder is comfortable
#   head region         31-42 px               anything routed through a face is noise
#   track length        median 9-16 frames     at 5 fps, and this is the one that decides
#
# A clip-level action model wants 16-64 consecutive frames of one person. At 5 fps the
# median track does not contain a whole person for long enough, so such a model would be
# trained on an input this pipeline cannot deliver -- and the fix for that is association
# (ARCHITECTURE.md section 5), not a bigger action model.
#
# **Pose is per-frame, so fragmentation shortens it instead of invalidating it.** Nine
# frames of a track is nine poses, and a fall is a change of torso angle over one or two
# seconds. That is the whole argument, and it is why this tier is buildable now while
# tier 3 is not.
#
# Two further things it gets for free: `person_keypoints_train2017.json` is already on
# disk, so this direction needs no dataset purchase to start; and the pose model is a
# second-stage crop model with its own small ONNX export, which is exactly the boundary
# `stage.py` and ARCHITECTURE.md section 2 describe. It is not a HydraNet head
# and cannot be one -- a keypoint head needs the detection head's boxes, and the `Head`
# protocol's `forward_into(out, feats, size)` has no way to say that.

# COCO's 17, in order. Named rather than indexed at the call site, because 11 and 12
# being the hips is the kind of fact that a reader cannot check and a typo cannot fail.
KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def require_keypoints(tracks: list[Track]) -> None:
    """Refuse a pose event on tracks that carry no pose, naming the missing model.

    An empty `Track.keypoints` is the normal state today -- nothing fills it, because the
    second-stage pose model is not built. Returning no events for that would be
    indistinguishable from "nobody fell", which is the failure this project ranks worst:
    plausible output, no error.
    """
    for track in tracks:
        if len(track.keypoints) != len(track.frames):
            raise NotImplementedError(
                f"track {track.track_id} carries {len(track.keypoints)} keypoint sets for "
                f"{len(track.frames)} observed frames. Pose events need one (17, 3) array "
                "per observed frame, written by the second-stage pose model -- which does "
                "not exist yet. Tier-1 `fall_candidates` is what runs without it."
            )


def _torso(kps: np.ndarray, score_thr: float) -> tuple[float, float, float]:
    """(angle from vertical, hip-to-ankle extent, torso length). Degrees and px, NaN if unseen.

    Image coordinates, y down. The angle is taken from the shoulder-hip vector, which is
    the segment least disturbed by an occluded lower body -- and a lower body behind a
    display table is the normal case here rather than the exception.
    """

    def mid(a: str, b: str) -> np.ndarray:
        pair = kps[[KP[a], KP[b]]]
        if (pair[:, 2] < score_thr).any():
            return np.array([np.nan, np.nan])
        return pair[:, :2].mean(axis=0)

    shoulder = mid("left_shoulder", "right_shoulder")
    hip = mid("left_hip", "right_hip")
    ankle = mid("left_ankle", "right_ankle")
    if not np.isfinite(shoulder).all() or not np.isfinite(hip).all():
        return float("nan"), float("nan"), float("nan")
    dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
    angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
    extent = float(abs(ankle[1] - hip[1])) if np.isfinite(ankle).all() else float("nan")
    return angle, extent, float(np.hypot(dx, dy))


def pose_posture_events(
    tracks: list[Track],
    fps: float,
    camera: str,
    fall_angle_deg: float = 55.0,
    crouch_ratio: float = 0.6,
    sustained_seconds: float = 1.0,
    score_thr: float = 0.3,
) -> list[SecurityEvent]:
    """`fall` and `crouch` from keypoints -- and they are one function on purpose.

    Separating the two is the entire reason this tier exists. `fall_candidates` cannot,
    because from a down-pitched camera a fallen person and a shopper reaching a bottom
    shelf produce the same box. The torso angle does separate them: crouching keeps the
    trunk roughly upright over folded legs, falling puts it near horizontal. So the two
    types are computed from one pass over one quantity, which is also what stops a future
    edit from making them disagree about the same frame.

    `crouch` needs the ankles and `fall` does not, which is deliberate -- ankles are the
    first thing a fixture hides, so the safety-relevant type is the one that survives
    occlusion.

    **Both quantities are measured inside a single frame.** The torso angle needs no
    reference at all, and `crouch` compares the hip-to-ankle extent against the torso
    length of the same person in the same frame rather than against that track's own
    history. Two things follow and the second is why the first is not a detail: it is
    scale-free, so a shopper at 3 m and one at 12 m are judged alike with no calibration;
    and it does not go blind when the posture fills the track. A self-baseline over a
    9-16 frame fragment -- the measured median track length -- compares a crouch against a
    crouch and reports nothing, which is precisely the case that matters.

    Every threshold here is a default and none is measured. They are stated as arguments
    for the reason RETAIL.md gives about the 5 m rule: a number that belongs to a
    store belongs in a config, not in weights, and not in a constant either.
    """
    require_keypoints(tracks)
    events: list[SecurityEvent] = []
    for track in tracks:
        if not track.frames:
            continue
        frames = np.asarray(track.frames)
        angles, ratios = [], []
        for kps in track.keypoints:
            angle, extent, torso = _torso(np.asarray(kps, dtype=float), score_thr)
            angles.append(angle)
            ratios.append(extent / torso if np.isfinite(extent) and torso > 0 else np.nan)
        angle_arr = np.asarray(angles)
        ratio = np.asarray(ratios)
        upright = np.nan_to_num(angle_arr, nan=0.0) < fall_angle_deg
        fallen = np.nan_to_num(angle_arr, nan=0.0) >= fall_angle_deg
        low = upright & (np.nan_to_num(ratio, nan=np.inf) <= crouch_ratio)
        for i0, i1 in _runs(fallen):
            events += _posture_event(
                track,
                frames,
                i0,
                i1,
                fps,
                camera,
                sustained_seconds,
                etype="fall",
                value=float(np.nanmax(angle_arr[i0 : i1 + 1])),
                threshold=fall_angle_deg,
                basis="shoulder-to-hip angle from vertical, in degrees, sustained",
            )
        for i0, i1 in _runs(low):
            events += _posture_event(
                track,
                frames,
                i0,
                i1,
                fps,
                camera,
                sustained_seconds,
                etype="crouch",
                value=float(np.nanmin(ratio[i0 : i1 + 1])),
                threshold=crouch_ratio,
                basis="hip-to-ankle extent over torso length in the same frame, trunk upright",
            )
    return events


def _posture_event(
    track: Track,
    frames: np.ndarray,
    i0: int,
    i1: int,
    fps: float,
    camera: str,
    sustained_seconds: float,
    *,
    etype: str,
    value: float,
    threshold: float,
    basis: str,
) -> list[SecurityEvent]:
    """One posture span as an event, or nothing if it was too short to be one.

    A module-level function rather than a closure inside `pose_posture_events`, which is
    where it started: a closure over the loop variables reads fine and is one refactor
    away from being called after the loop has moved on. Returning a list rather than
    appending keeps the caller's `events` the only mutable thing in that function.
    """
    if (frames[i1] - frames[i0] + 1) / fps < sustained_seconds:
        return []
    return [
        SecurityEvent(
            type=etype,
            camera=camera,
            frame_start=int(frames[i0]),
            frame_end=int(frames[i1]),
            fps=fps,
            track_ids=(track.track_id,),
            value=value,
            threshold=threshold,
            basis=basis,
        )
    ]


def _require_terrain_in_image_space(
    terrain: np.ndarray, image_hw: tuple[int, int], frame_index: int
) -> None:
    """Refuse a terrain map that does not live in the keypoints' pixel space.

    The wrist test indexes `terrain[yi, xi]` with coordinates in **image pixels**, so a
    map at the model's canvas resolution (512x896 against a 1920x1080 source) puts almost
    every wrist off the map's edge -- and the old code skipped out-of-bounds wrists,
    which turned a unit mismatch into zero events that read as "nobody reached". The
    pose pilot (runs/pose_pilot01/REPORT.md section 4-2) hit exactly this and had to
    discover the contract by reading the source.

    **The caller owns the resize**, because only the caller knows which space the
    keypoints were measured in: upsample the terrain map back to the image resolution
    before building the frame payloads -- nearest-neighbour, since these are class ids
    and interpolating them invents classes. `scripts/pose_pilot.py` shows the two lines.
    """
    th, tw = (int(s) for s in terrain.shape[-2:])
    ih, iw = (int(s) for s in image_hw)
    if (th, tw) != (ih, iw):
        raise ValueError(
            f"frame {frame_index}: terrain map is {tw}x{th} but the keypoints are in "
            f"{iw}x{ih} image pixels. Wrists indexed into the wrong pixel space fall "
            "off the map and produce zero events that read as 'nobody reached'. The "
            "caller must upsample the terrain map back to the image resolution "
            "(nearest-neighbour -- class ids, so interpolation invents classes) before "
            "building the frame payloads; events.py cannot do it because it cannot "
            "know which of the two spaces the keypoints were measured in."
        )


def _wrist_contact(
    terrain: np.ndarray,
    x: float,
    y: float,
    radius_px: float,
    fixture_id: int,
    person_id: int | None,
) -> float:
    """Fixture fraction of the non-person terrain within ``radius_px`` of a wrist.

    The wrist pixel itself is not consulted, and that is the entire mechanism: when the
    pose is right and the segmentation is right, that pixel is the person's own hand and
    is labelled person, never fixture. So the question is what the hand is *over* --
    fixture pixels in the disk around the wrist, with the person's own silhouette
    removed from the denominator so a hand that covers most of the disk cannot dilute
    the answer. Returns 0.0 when the disk holds nothing but person (or nothing at all):
    a neighbourhood the segmentation cannot resolve is not evidence of contact.
    """
    h, w = (int(s) for s in terrain.shape[-2:])
    r = float(radius_px)
    x0, x1 = max(int(np.floor(x - r)), 0), min(int(np.ceil(x + r)) + 1, w)
    y0, y1 = max(int(np.floor(y - r)), 0), min(int(np.ceil(y + r)) + 1, h)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    patch = np.asarray(terrain[y0:y1, x0:x1])[disk]
    if person_id is not None:
        patch = patch[patch != person_id]
    if patch.size == 0:
        return 0.0
    return float((patch == fixture_id).mean())


def reach_to_shelf_events(
    tracks: list[Track],
    frames: Sequence[TerrainFrame],
    fps: float,
    camera: str,
    fixture_id: int,
    person_id: int | None = None,
    radius_px: float = 25.0,
    contact_ratio: float = 0.35,
    min_seconds: float = 0.6,
    score_thr: float = 0.3,
    image_size: tuple[int, int] | None = None,
) -> list[SecurityEvent]:
    """A wrist over the terrain head's `fixture` region, sustained.

    **This is the one output where the retail model and the security model are the same
    model**, and it is worth saying because "shared trunk" is usually a claim about
    parameters rather than about answers. Here the segmentation head's `fixture` channel
    and the second stage's wrist keypoint are both required to produce one row: neither
    the terrain map nor the pose model can emit it alone.

    **Why the neighbourhood and not the wrist pixel.** The first version tested
    `terrain[wrist] == fixture_id`, and the pose pilot measured what that means in
    practice (runs/pose_pilot01/REPORT.md section 3): the wrist pixel is covered by the
    person's own hand, so a correct pose over a correct segmentation reads `person`
    there -- structurally, every time. The single-pixel rule could only fire when the
    hand was *occluded by* the fixture or the pose was wrong, which is the opposite of
    the design intent. So the test is now "what is the hand over": the fixture fraction
    of the disk of ``radius_px`` around the wrist, with the person's own pixels removed
    from the denominator (``person_id``; pass None only for a taxonomy that has no
    person class, and know that the hand then dilutes the ratio).

    The two defaults, derived from the pilot's measurements rather than chosen:

    * ``radius_px=25``: person boxes are a median 244-336 px tall on the ground-truthed
      clips, a hand is roughly a tenth of standing height, so the hand's silhouette
      extends ~12-17 px from the wrist point and ViTPose on true boxes adds a few px of
      localisation error. 25 px is one hand-length: the smallest disk that reliably
      reaches *past* the hand's own person-labelled pixels to what it is over. A deep
      bend can still cover ~98% of the disk with the person's own body (measured at the
      pilot's probe frame cam04 t2 f114) -- the sliver beyond the fingertips then
      decides alone, deliberately, because demanding a larger denominator would
      re-create the structural miss this rule replaced.
    * ``contact_ratio=0.35``: on the pilot's clips, frames where the hand is visibly at
      a fixture measured 0.55-1.0 (merchandise stacked on the table is its own terrain
      class and dilutes the disk, hence the low end); non-contact wrists peaked at 0.41
      and only for a frame or two, which ``min_seconds`` removes. 0.35 sits under the
      weakest measured contact; the margin against non-contact is temporal as much as
      spatial.

    `fixture_id` is passed rather than imported because the id is a property of the
    taxonomy a config chose -- 4 under `RETAIL_OBJECTS`, 12 under the retail terrain
    scheme, and a hard-coded number would be wrong the first time a config reorders its
    classes. `data.terrain_classes.index("display_fixture")` is where a caller gets it,
    and `person_id` the same way.

    The terrain map must live in the same pixel space as the keypoints, and that
    contract is now checked rather than implied: a frame payload that carries `image`
    (`stage.StageFrame` requires it) is checked against it, and a slimmer payload can
    state the space with ``image_size`` (height, width). A mismatch raises naming who
    resizes -- see `_require_terrain_in_image_space`.

    What it is not: a purchase, a pick-up, or an interaction with a specific product. It
    is a wrist over a shelf. MERL Shopping's "reach to shelf" is the nearest labelled
    thing and ARCHITECTURE.md's warning about it applies here too -- those are
    closed grocery shelves and these are open display tables, so do not assume the
    behaviour has the same shape before counting it here.
    """
    require_keypoints(tracks)
    terrain_by_frame = {
        int(f["frame_index"]): f["terrain"] for f in frames if f.get("terrain") is not None
    }
    if not terrain_by_frame:
        raise ValueError(
            "no frame carries a `terrain` map, so a wrist cannot be tested against the "
            "fixture region. A detection-only config has no terrain head; this event "
            "needs one."
        )
    for f in frames:
        # Bound once rather than re-read after the guard: `stage.TerrainFrame` types
        # `terrain` as optional, which is the truth, and a second `f["terrain"]` below a
        # `.get(...) is None` check is a narrowing no type checker can follow.
        terrain_map = f.get("terrain")
        if terrain_map is None:
            continue
        hw = image_size if image_size is not None else _image_hw(f)
        if hw is not None:
            _require_terrain_in_image_space(terrain_map, hw, int(f["frame_index"]))
    events: list[SecurityEvent] = []
    for track in tracks:
        idx = np.asarray(track.frames)
        contact = np.zeros(len(idx), dtype=float)
        for i, (frame, kps) in enumerate(zip(track.frames, track.keypoints, strict=True)):
            terrain = terrain_by_frame.get(int(frame))
            if terrain is None:
                continue
            kps = np.asarray(kps, dtype=float)
            for wrist in ("left_wrist", "right_wrist"):
                x, y, score = kps[KP[wrist]]
                if score < score_thr:
                    continue
                contact[i] = max(
                    contact[i],
                    _wrist_contact(terrain, x, y, radius_px, fixture_id, person_id),
                )
        for i0, i1 in _runs(contact >= contact_ratio):
            seconds = (idx[i1] - idx[i0] + 1) / fps
            if seconds < min_seconds:
                continue
            events.append(
                SecurityEvent(
                    type="reach_to_shelf",
                    camera=camera,
                    frame_start=int(idx[i0]),
                    frame_end=int(idx[i1]),
                    fps=fps,
                    track_ids=(track.track_id,),
                    value=float(contact[i0 : i1 + 1].max()),
                    threshold=contact_ratio,
                    basis=(
                        f"fixture fraction of the non-person terrain within "
                        f"{radius_px:g} px of a wrist keypoint, sustained "
                        f"{min_seconds:g}s"
                    ),
                    extra={"radius_px": float(radius_px)},
                )
            )
    return events


def _image_hw(frame: TerrainFrame) -> tuple[int, int] | None:
    """(height, width) of the image a frame payload carries, or None if it carries none.

    `stage.StageFrame` makes `image` required, so the production payload always states
    its own pixel space; a slimmer dict (tests, offline reruns from saved arrays) states
    it via `reach_to_shelf_events`' ``image_size`` instead, or -- carrying neither --
    goes unchecked, which is the caller declining the check rather than the check
    silently passing a mismatch.
    """
    image = frame.get("image")
    if image is None:
        return None
    h, w = image.shape[:2]
    return int(h), int(w)
