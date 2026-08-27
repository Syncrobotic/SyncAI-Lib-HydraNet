"""The NTU reader, held to the two things a training set cannot survive getting wrong.

A wrong joint order and a silently short clip both produce arrays of the right shape and
dtype, so neither announces itself downstream: the first trains a model on a skeleton
whose shoulders are its hips, and the second puts a truncated action in a set as a
complete one. That is the same failure `data/video.py` raises `DecodeError` for, and the
reason its docstring gives -- "silent truncation of customer footage is the failure this
project is least able to audit after the fact" -- applies unchanged here.

The fixtures are written as text rather than read from the 11 GB archives, so these run
without the dataset and say what the format is at the same time.
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_hydranet.data.ntu_skeletons import (
    JOINTS,
    N_JOINTS,
    parse_name,
    parse_skeleton,
    torso_from_vertical,
)


def _joint_line(x: float, y: float, z: float) -> str:
    """One joint row: xyz, then the depth/colour pixels and quaternion this module drops."""
    return f"{x} {y} {z} 0 0 0 0 1 0 0 0 2"


def _body(xyz: np.ndarray) -> list[str]:
    meta = "72057594037928947 0 1 1 0 0 0 -0.069 -0.049 2"
    return [meta, str(N_JOINTS)] + [_joint_line(*row) for row in xyz]


def _clip_text(frames: list[list[np.ndarray]]) -> str:
    out = [str(len(frames))]
    for bodies in frames:
        out.append(str(len(bodies)))
        for xyz in bodies:
            out += _body(xyz)
    return "\n".join(out) + "\n"


def _upright(offset: float = 0.0) -> np.ndarray:
    """A skeleton standing up: shoulders above hips along +y."""
    xyz = np.zeros((N_JOINTS, 3), dtype=np.float32)
    xyz[:, 2] = 3.0
    xyz[JOINTS["l_hip"]] = (-0.15, 0.9 + offset, 3.0)
    xyz[JOINTS["r_hip"]] = (0.15, 0.9 + offset, 3.0)
    xyz[JOINTS["l_shoulder"]] = (-0.2, 1.4 + offset, 3.0)
    xyz[JOINTS["r_shoulder"]] = (0.2, 1.4 + offset, 3.0)
    xyz[JOINTS["head"]] = (0.0, 1.65 + offset, 3.0)
    return xyz


def test_the_parsed_array_is_frames_bodies_joints_xyz():
    clip = parse_skeleton(_clip_text([[_upright()], [_upright(0.1)]]), "S001C002P003R004A043")
    assert clip.joints.shape == (2, 1, N_JOINTS, 3)
    assert clip.joints.dtype == np.float32


def test_the_three_kept_columns_are_the_metric_ones():
    """The file carries twelve numbers per joint and only the first three are a position.

    Keeping the wrong three would put a joint at its pixel coordinate, which is a plausible
    number in the wrong unit -- an error of about a thousand, and one that a scale-free
    check would not see.
    """
    xyz = _upright()
    clip = parse_skeleton(_clip_text([[xyz]]), "S001C001P001R001A043")
    assert clip.joints[0, 0, JOINTS["head"]] == pytest.approx([0.0, 1.65, 3.0])


def test_a_clip_that_runs_out_mid_frame_raises_rather_than_returning_a_short_one():
    text = _clip_text([[_upright()], [_upright()]])
    truncated = "\n".join(text.split("\n")[:-6])
    with pytest.raises(ValueError, match="ran out of lines"):
        parse_skeleton(truncated, "S001C001P001R001A043")


def test_a_body_with_the_wrong_joint_count_raises():
    text = _clip_text([[_upright()]]).replace(f"\n{N_JOINTS}\n", "\n24\n", 1)
    with pytest.raises(ValueError, match="expected 25"):
        parse_skeleton(text, "S001C001P001R001A043")


def test_missing_bodies_are_nan_and_not_the_origin():
    """A zero is a joint at the sensor, which is a place a joint can be."""
    clip = parse_skeleton(_clip_text([[_upright(), _upright(0.5)], [_upright()]]), "x")
    assert clip.n_bodies == 2
    assert np.isnan(clip.joints[1, 1]).all()
    assert not np.isnan(clip.joints[0, 1]).any()


def test_the_filename_carries_the_split_the_protocol_needs():
    meta = parse_name("nturgb+d_skeletons/S014C003P025R002A043.skeleton")
    assert meta == {
        "setup": 14,
        "camera": 3,
        "performer": 25,
        "replication": 2,
        "action": 43,
    }
    assert parse_name("not-an-ntu-file.skeleton") is None


def test_an_upright_torso_reads_near_zero_and_a_toppled_one_near_ninety():
    upright = _upright()[None]
    assert torso_from_vertical(upright)[0] == pytest.approx(0.0, abs=1.0)

    lying = _upright().copy()
    lying[JOINTS["l_shoulder"]] = (-0.2, 0.2, 2.5)
    lying[JOINTS["r_shoulder"]] = (0.2, 0.2, 2.5)
    lying[JOINTS["l_hip"]] = (-0.15, 0.2, 3.0)
    lying[JOINTS["r_hip"]] = (0.15, 0.2, 3.0)
    assert torso_from_vertical(lying[None])[0] == pytest.approx(90.0, abs=1.0)


def test_up_is_an_argument_because_the_camera_is_not_gravity():
    """Tilting the reference by 20 degrees must move the answer by 20 degrees.

    The default is the sensor's +y, and NTU's setups vary in height and mounting, so a
    caller with a measured vertical has to be able to say so. A silent default is how a
    per-setup tilt turns into a per-class difference.
    """
    upright = _upright()[None]
    tilted = np.array([0.0, np.cos(np.radians(20)), np.sin(np.radians(20))])
    assert torso_from_vertical(upright, up=tilted)[0] == pytest.approx(20.0, abs=1.0)
