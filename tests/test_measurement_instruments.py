"""The instruments that produce PLAN's numbers, held to the properties those numbers need.

Three of the measurements this project now quotes rest on a property that is invisible
in the output. A wrong one does not crash and does not look wrong -- it prints a
plausible number, which is the failure mode every one of these instruments was built to
catch, so it is the one they must not commit themselves.

    rigid_fit         must NOT absorb a scale error. PLAN section 7.13's 7.3 cm is only a
                      measurement because of this: a similarity fit drives Taichung-cam10's
                      measured 1.21x to a residual of 1e-15 m and calls that camera perfect.
    to_our_model      must recover height, pitch and roll from a full pose, or every metre
                      after it is charged to the wrong parameter.
    witness_verdict   must rank `available` above the detector verdicts and judge it at the
                      tracker's own IoU. Scoring it at the looser presence threshold turns
                      a threshold difference into an accusation against the tracker.
    _read             must ask the clip its resolution. Assuming 1080p on the fleet's
                      704x480 sub-streams returns a prefix of the clip, and the only reason
                      that surfaced was `DecodeError` refusing to pretend.

Scripts are not importable as a package -- `test_scripts_are_not_libraries.py` says why --
so they are reached the way five other test modules here reach one: `scripts/` on
`sys.path` for the duration of the import, and off again afterwards.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(SCRIPTS))


wildtrack = _script("wildtrack_ground_eval")
endings = _script("track_endings")
probe_mod = _script("person_score_probe")


# ------------------------------------------------------- the fit that must not rescale


@pytest.mark.parametrize("true_scale", [1.0, 1.21, 0.8824, 1.05])
def test_a_rigid_fit_does_not_absorb_a_scale_error(true_scale):
    """1.21 is Taichung-cam10's measured error and 0.8824 is its correction (PLAN 7.10).

    The residual has to *grow* with the scale error. If it does not, step 5's gate cannot
    see the one calibration fault this fleet is known to have.
    """
    rng = np.random.default_rng(0)
    src = rng.uniform(-5, 5, (400, 2))
    th = math.radians(33.0)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    dst = (true_scale * (rot @ src.T)).T + np.array([7.0, -2.0])

    fitted, scale = wildtrack.rigid_fit(src, dst, allow_scale=False)
    residual = float(np.median(np.hypot(*(fitted - dst).T)))
    assert scale == 1.0, "a rigid fit must report unit scale, whatever it was handed"

    if abs(true_scale - 1.0) < 1e-9:
        assert residual < 1e-9
    else:
        # the error a scale slip commits is proportional to how far a point is from the
        # centroid, so a metre of it survives on data spread over metres
        assert residual > 0.1 * abs(true_scale - 1.0) * 5.0


def test_a_free_fit_hides_exactly_what_the_rigid_one_must_show():
    """The counter-example, kept next to the rule so the reason stays visible."""
    rng = np.random.default_rng(1)
    src = rng.uniform(-5, 5, (400, 2))
    dst = (1.21 * src.T).T + np.array([3.0, 4.0])
    _fitted_rigid, s_rigid = wildtrack.rigid_fit(src, dst, allow_scale=False)
    fitted_free, s_free = wildtrack.rigid_fit(src, dst, allow_scale=True)
    assert s_rigid == 1.0
    assert s_free == pytest.approx(1.21, rel=1e-6)
    assert float(np.median(np.hypot(*(fitted_free - dst).T))) < 1e-9


# --------------------------------------------------------------- the pose conversion


@pytest.mark.parametrize("height", [2.17, 2.5, 2.91])
@pytest.mark.parametrize("pitch_deg", [8.7, 38.8, 52.3])
@pytest.mark.parametrize("roll_deg", [-12.9, 0.0, 8.3])
def test_to_our_model_recovers_the_pose_it_was_given(height, pitch_deg, roll_deg):
    """Fleet poses and WILDTRACK's shallowest, since the gate runs on both.

    Yaw is deliberately non-zero and deliberately lost: `GroundPlane` has no yaw, and
    step 5 recovers it with the rigid fit rather than smuggling it through here.
    """
    from scipy.spatial.transform import Rotation

    from syncai_hydranet.geometry import GroundPlane

    plane = GroundPlane(height, math.radians(pitch_deg), math.radians(roll_deg))
    yaw = math.radians(40.0)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # world (x, y, z up) -> our level frame (x right, y down, z forward)
    r_wl = np.array([[cy, sy, 0.0], [0.0, 0.0, -1.0], [-sy, cy, 0.0]])
    r_wc = plane.rotation @ r_wl
    centre = np.array([1.3, -2.4, height])
    cam = wildtrack.WtCamera(
        name="synthetic",
        k=np.array([[400.0, 0.0, 480.0], [0.0, 400.0, 270.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5),
        rvec=Rotation.from_matrix(r_wc).as_rotvec(),
        tvec=-r_wc @ centre,
    )
    intr, got = wildtrack.to_our_model(cam)
    assert intr.fx == pytest.approx(400.0)
    assert intr.cx == pytest.approx(480.0)
    assert got.height == pytest.approx(height, abs=1e-9)
    assert math.degrees(got.pitch) == pytest.approx(pitch_deg, abs=1e-6)
    assert math.degrees(got.roll) == pytest.approx(roll_deg, abs=1e-6)


def test_the_grid_convention_is_a_choice_the_script_can_still_get_wrong():
    """`check` decides it, so the candidates must actually differ. A set that agreed
    would make that run look like a confirmation when it decided nothing."""
    pid = np.array([456826])
    seen = {
        name: tuple(np.round(wildtrack.grid_to_metres(pid, name)[0], 4))
        for name in wildtrack.GRID_CANDIDATES
    }
    assert len(set(seen.values())) == len(seen), seen


# ----------------------------------------------------------------- the witness verdict


def test_a_box_the_tracker_could_have_taken_outranks_every_detector_verdict():
    assert (
        endings.witness_verdict(
            best_assoc=0.42, best_score=0.42, dense=True, score_thr=0.35, witness_thr=0.03
        )
        == "available"
    )


def test_a_box_below_the_threshold_is_the_detector_s_verdict_not_the_tracker_s():
    """The fleet's own case: a box at 0.338 against a 0.35 threshold (PLAN 7.11)."""
    assert (
        endings.witness_verdict(
            best_assoc=0.338, best_score=0.338, dense=False, score_thr=0.35, witness_thr=0.03
        )
        == "demoted"
    )


def test_a_high_box_the_tracker_could_not_reach_is_not_called_a_tracker_failure():
    """`best_score` clears the shipped threshold but `best_assoc` does not, which is a box
    overlapping too loosely for the tracker's own rule. Judging this `available` is the
    accusation an earlier revision of the witness pass made."""
    assert (
        endings.witness_verdict(
            best_assoc=0.0, best_score=0.60, dense=True, score_thr=0.35, witness_thr=0.03
        )
        == "demoted"
    )


def test_no_box_at_all_splits_on_whether_the_dense_head_still_sees_a_person():
    common = {"best_assoc": 0.0, "best_score": 0.0, "score_thr": 0.35, "witness_thr": 0.03}
    assert endings.witness_verdict(dense=True, **common) == "boxless"
    assert endings.witness_verdict(dense=False, **common) == "vacated"


# --------------------------------------------------------------- reading a clip's size


# `test_video_decode_errors.py` already carries this guard and this test did not, so it
# passed on every machine that has ffmpeg and failed on the one that decides -- the
# GitHub runner ships without it, and the failure there is `FileNotFoundError: 'ffmpeg'`
# rather than anything about resolutions. Found 2026-08-28, red on three matrix rows.
needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg"
)


@needs_ffmpeg
@pytest.mark.parametrize("size", [(1920, 1080), (704, 480)])
def test_a_clip_is_read_at_its_own_resolution(tmp_path, size):
    """704x480 is not hypothetical: seven of the fleet's forty-eight cameras deliver it.

    Assuming 1080p sizes the reads on a rawvideo pipe carrying the real frames, so the
    stream desynchronises and `frames` raises `DecodeError` after a prefix. The whole
    point of `_read` is that a caller cannot make that happen.
    """
    w, h = size
    clip = tmp_path / f"{w}x{h}.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc=size={w}x{h}:rate=5:duration=2", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    got = [f for _i, f in zip(range(4), probe_mod._read(str(clip), 5.0), strict=False)]
    assert got, "no frames decoded"
    for frame in got:
        assert np.asarray(frame).shape == (h, w, 3)
