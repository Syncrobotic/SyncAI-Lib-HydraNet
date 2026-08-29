"""The synthetic posed shopper two test modules both need, in the one place both can reach.

Extracted from `test_security_events.py` on 2026-08-27, because the second consumer had
reached it the only way a test module can reach another one -- `from tests.test_x import
...` -- and that import does not resolve under the command `CONTRIBUTING.md` documents.
`tests/` has no `__init__.py`, so `tests` is not a package; pytest puts *this* directory on
`sys.path`, not the repository root. `uv run pytest -q` therefore raised
`ModuleNotFoundError: No module named 'tests'` while `python -m pytest -q` passed, on the
same tree in the same second, because the `-m` form adds the working directory. A suite
whose colour depends on which of two spellings you typed is worse than a failing one: it
was reported green all day.

A leading underscore rather than `conftest.py`: these are plain factories, not fixtures,
and importing `conftest` by name is the anti-pattern that would replace the one being
removed. The underscore keeps pytest from collecting the file, and a bare
`from _posture import ...` resolves for every test module because pytest already put this
directory first on `sys.path`.

`tests/test_tests_do_not_import_each_other.py` is what stops the original shape coming
back.
"""

from __future__ import annotations

import math

import numpy as np

from syncai_hydranet.analytics import events as ev
from syncai_hydranet.analytics.tracker import Track


def keypoints(angle_deg: float, hip_ankle_px: float, score: float = 0.9) -> np.ndarray:
    """A synthetic person: torso at ``angle_deg`` from vertical, legs ``hip_ankle_px`` long."""
    kps = np.zeros((17, 3))
    kps[:, 2] = score
    hip = np.array([500.0, 600.0])
    torso = 150.0
    dx = torso * math.sin(math.radians(angle_deg))
    dy = torso * math.cos(math.radians(angle_deg))
    shoulder = hip - np.array([dx, dy])
    for i in (ev.KP["left_hip"], ev.KP["right_hip"]):
        kps[i, :2] = hip
    for i in (ev.KP["left_shoulder"], ev.KP["right_shoulder"]):
        kps[i, :2] = shoulder
    for i in (ev.KP["left_ankle"], ev.KP["right_ankle"]):
        kps[i, :2] = hip + np.array([0.0, hip_ankle_px])
    for i in (ev.KP["left_wrist"], ev.KP["right_wrist"]):
        kps[i, :2] = shoulder + np.array([60.0, 40.0])
    return kps


def posed(track_id: int, poses: list[np.ndarray], heights: list[float] | None = None) -> Track:
    """A track whose box height can follow its posture, because a real one does.

    The boxes used to be one constant for every frame, which asserted a shopper who is
    already lying down when the track opens and stays exactly as tall while doing it.
    `pose_posture_events` now cross-checks the box, so the fixture has to be physically
    possible: stand for a while, then change.
    """
    hs = heights or [400.0] * len(poses)
    boxes = [np.array([400.0, 400.0, 600.0, 400.0 + h]) for h in hs]
    return Track(
        track_id=track_id,
        box=boxes[-1],
        frames=list(range(len(poses))),
        boxes=boxes,
        confirmed=True,
        keypoints=poses,
    )


def _stand_then(pose: np.ndarray, height_after: float, n_stand: int = 5, n_after: int = 10):
    """Five upright frames, then the posture under test -- poses and matching heights."""
    poses = [keypoints(5.0, 200.0)] * n_stand + [pose] * n_after
    hs = [400.0] * n_stand + [height_after] * n_after
    return poses, hs
