"""NTU RGB+D's official `.skeleton` files, in the coordinates the sensor produced.

PLAN §2.3 replaces staged in-store clips with public 3D action data **projected through
our own measured camera parameters**, and that route needs coordinates a projection can
still be applied to. This reads them.

---------------------------------------------------------------------------
WHY THE OFFICIAL FILES AND NOT A REDISTRIBUTION

`tools/temporal/ntu_survey.py` measured a published `NTU60_CS.npz` and its own numbers
convict it: across 16,487 samples the first frame's spine direction has a standard
deviation of 0.10 and its origin sits within 3 cm, which is a canonical rotation rather
than a camera. Every such tensor has had the quantity we mean to re-impose already
removed from it, so the projection route cannot start there.

**That normalisation was not a defect for what the survey asked.** The survey compares
peak torso tilt between action classes, and tilt needs a vertical: in a redistribution it
is the subject's own initial upright, and in these files it is the **Kinect's y axis**,
which is only gravity if that Kinect happened to be level. NTU's setups vary in height
and the three views differ in horizontal angle, so raw camera-frame `y` carries a
per-setup tilt that the redistribution had removed on purpose. Reading the official files
does not make that measurement better; it makes **training** possible, which is a
different job and the one this module exists for.

---------------------------------------------------------------------------
THE FORMAT, WHICH IS PLAIN TEXT AND UNDOCUMENTED IN THE ARCHIVE

    <n_frames>
    per frame:  <n_bodies>
                per body:  <body_id clipped_edges handL_conf handL_state handR_conf
                            handR_state is_restricted lean_x lean_y tracking_state>
                           <n_joints>            -- always 25, Kinect v2's layout
                           per joint:  x y z depth_x depth_y colour_x colour_y
                                       orient_w orient_x orient_y orient_z tracking_state

`x y z` are **metres in the Kinect camera frame** — verified against
`S014C001P025R002A043.skeleton`, whose first joint reads 0.541 0.179 3.426, a person
three and a half metres from a sensor. The seven fields after them are the same point in
the depth and colour images and the joint's orientation quaternion; this module keeps the
three that a projection needs and drops the rest rather than carrying columns nobody
reads.

Members are read **out of the zip**, never extracted: the two archives are 11 GB and hold
56,881 files between them, and a training run wants a filtered handful of classes.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Kinect v2's 25-joint order. Named rather than written as integers at the use site, for
# the reason `ntu_survey.py` gives: a bare `b[:, 8]` inside an angle calculation is
# unreviewable. Kept identical to that file's `J` so the two agree by construction.
JOINTS = {
    "spine_base": 0,
    "spine_mid": 1,
    "neck": 2,
    "head": 3,
    "l_shoulder": 4,
    "l_elbow": 5,
    "l_wrist": 6,
    "l_hand": 7,
    "r_shoulder": 8,
    "r_elbow": 9,
    "r_wrist": 10,
    "r_hand": 11,
    "l_hip": 12,
    "l_knee": 13,
    "l_foot": 14,
    "l_ankle": 15,
    "r_hip": 16,
    "r_knee": 17,
    "r_foot": 18,
    "r_ankle": 19,
    "spine_shoulder": 20,
    "l_hand_tip": 21,
    "l_thumb": 22,
    "r_hand_tip": 23,
    "r_thumb": 24,
}
N_JOINTS = 25

# S001C002P003R002A043 -> setup 1, camera 2, performer 3, replication 2, action 43.
_NAME = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<performer>\d{3})"
    r"R(?P<replication>\d{3})A(?P<action>\d{3})"
)


@dataclass(frozen=True)
class Clip:
    """One `.skeleton` file: what it is, and where its joints are."""

    name: str
    setup: int
    camera: int
    performer: int
    replication: int
    action: int
    #: ``(frames, bodies, 25, 3)`` in metres, Kinect camera frame. Bodies are ragged in
    #: the file and padded here with NaN rather than zeros -- a zero is a point at the
    #: sensor, which is a place a joint can be, and every consumer already reads NaN as
    #: "not measured".
    joints: np.ndarray

    @property
    def n_bodies(self) -> int:
        return int(self.joints.shape[1])


def parse_name(name: str) -> dict[str, int] | None:
    m = _NAME.search(Path(name).stem)
    return {k: int(v) for k, v in m.groupdict().items()} if m else None


def parse_skeleton(text: str, name: str = "") -> Clip:
    """Parse one file's text. Raises rather than guessing on a malformed member.

    A truncated member is the failure this project has been bitten by in video decoding,
    where a short read looked exactly like a finished file. The same rule applies here:
    running out of lines mid-frame raises, so a partial clip cannot enter a training set
    as a short one.
    """
    lines = text.split("\n")
    pos = 0

    def take() -> str:
        nonlocal pos
        while pos < len(lines) and not lines[pos].strip():
            pos += 1
        if pos >= len(lines):
            raise ValueError(f"{name or '<text>'}: ran out of lines mid-clip")
        pos += 1
        return lines[pos - 1].strip()

    n_frames = int(take())
    frames: list[list[np.ndarray]] = []
    max_bodies = 0
    for _ in range(n_frames):
        n_bodies = int(take())
        max_bodies = max(max_bodies, n_bodies)
        bodies: list[np.ndarray] = []
        for _ in range(n_bodies):
            take()  # the ten body-metadata fields; none of them is a coordinate
            n_joints = int(take())
            if n_joints != N_JOINTS:
                raise ValueError(f"{name or '<text>'}: {n_joints} joints, expected {N_JOINTS}")
            xyz = np.empty((N_JOINTS, 3), dtype=np.float32)
            for j in range(N_JOINTS):
                parts = take().split()
                xyz[j] = (float(parts[0]), float(parts[1]), float(parts[2]))
            bodies.append(xyz)
        frames.append(bodies)

    out = np.full((n_frames, max(max_bodies, 1), N_JOINTS, 3), np.nan, dtype=np.float32)
    for t, bodies in enumerate(frames):
        for b, xyz in enumerate(bodies):
            out[t, b] = xyz

    meta = parse_name(name) or {
        "setup": -1,
        "camera": -1,
        "performer": -1,
        "replication": -1,
        "action": -1,
    }
    return Clip(name=name, joints=out, **meta)


def read_clip(zf: zipfile.ZipFile, member: str) -> Clip:
    with zf.open(member) as f:
        return parse_skeleton(f.read().decode("utf-8", "replace"), member)


def members(zf: zipfile.ZipFile, *, actions: set[int] | None = None) -> list[str]:
    """Member names, optionally only for the given action ids (1-based, as in the name).

    Filtering by name rather than by reading: the archives hold 56,881 files and a
    behaviour model wants a handful of classes, so opening every member to find out what
    it is would be the whole 11 GB for a few hundred clips.
    """
    out = []
    for name in zf.namelist():
        if not name.endswith(".skeleton"):
            continue
        meta = parse_name(name)
        if meta is None:
            continue
        if actions is None or meta["action"] in actions:
            out.append(name)
    return sorted(out)


def torso_from_vertical(body: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Per frame: the shoulder-to-hip line's angle from `up`, in degrees.

    `up` defaults to the camera's +y, which is what the sensor calls up and is **only
    gravity if that Kinect was level**. Pass a measured vertical when you have one --
    this signature exists so a caller has to decide, because the same function with a
    silent default is how a per-setup tilt becomes a per-class difference.

    Matches `tools/temporal/ntu_survey.py:torso_from_vertical` in everything but that
    choice, so the two are comparable when the same `up` is used.
    """
    up_vec = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, dtype=float)
    up_vec = up_vec / np.linalg.norm(up_vec)
    sh = (body[:, JOINTS["l_shoulder"]] + body[:, JOINTS["r_shoulder"]]) / 2
    hp = (body[:, JOINTS["l_hip"]] + body[:, JOINTS["r_hip"]]) / 2
    d = sh - hp
    along = d @ up_vec
    perp = np.linalg.norm(d - np.outer(along, up_vec), axis=1)
    return np.degrees(np.arctan2(perp, along))
