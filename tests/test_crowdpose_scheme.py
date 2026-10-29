"""Re-derive the CrowdPose mapping from the shipped annotation file.

The mapping in `label_maps_crowdpose` is written against CrowdPose's own index order.
That order is data, not documentation: a re-release that reorders `categories[0]
["keypoints"]` would leave every limb mapped to the wrong channel, train without error
and score plausibly. These tests read the order back out of the file rather than
trusting the constant.

Skipped when the dataset is absent, so CI without the 4.7 GB download still runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from syncai_hydranet.data.label_maps_crowdpose import (
    COCO17,
    CROWDPOSE14,
    CROWDPOSE_TO_COCO17,
    UNMAPPED,
    UNSUPPLIED,
    to_coco17,
)

ANN = Path("/data/open_data/CrowdPose/annotations/crowdpose_trainval.json")
needs_data = pytest.mark.skipif(not ANN.is_file(), reason=f"{ANN} not downloaded")


@pytest.fixture(scope="module")
def shipped() -> dict:
    return json.loads(ANN.read_text())


@needs_data
def test_index_order_matches_the_shipped_file(shipped):
    """The constant is a copy of the data; prove it is still an accurate one."""
    assert tuple(shipped["categories"][0]["keypoints"]) == CROWDPOSE14


@needs_data
def test_every_annotation_has_fourteen_triplets(shipped):
    lens = {len(a["keypoints"]) for a in shipped["annotations"][:5000]}
    assert lens == {14 * 3}


@needs_data
def test_occluded_flag_is_the_majority(shipped):
    """v=1 outnumbering v=2 is the reason this dataset was downloaded.

    If a future release drops the occluded flag, CrowdPose stops answering the question
    it is here for and the mixing ratio should be reconsidered rather than inherited.
    """
    v = np.concatenate(
        [np.asarray(a["keypoints"]).reshape(-1, 3)[:, 2] for a in shipped["annotations"][:20000]]
    )
    assert (v == 1).sum() > (v == 2).sum()


def test_twelve_map_and_two_are_dropped():
    assert len(CROWDPOSE_TO_COCO17) == 12
    assert {CROWDPOSE14[i] for i in range(14)} - {CROWDPOSE14[i] for i in CROWDPOSE_TO_COCO17} == set(
        UNMAPPED
    )


def test_no_two_sources_share_a_destination_channel():
    """A silent limb swap would show up here and nowhere else."""
    dsts = list(CROWDPOSE_TO_COCO17.values())
    assert len(set(dsts)) == len(dsts)


def test_mapping_preserves_the_joint_name():
    for src, dst in CROWDPOSE_TO_COCO17.items():
        assert CROWDPOSE14[src] == COCO17[dst]


def test_unsupplied_channels_come_back_not_labelled():
    kp = np.arange(14 * 3, dtype=np.float32).reshape(1, 14, 3)
    kp[..., 2] = 2  # everything visible on the way in
    out = to_coco17(kp)
    assert out.shape == (1, 17, 3)
    for name in UNSUPPLIED:
        assert out[0, COCO17.index(name), 2] == 0, f"{name} must arrive as v=0"


def test_mapped_channels_carry_the_visibility_flag_through():
    kp = np.zeros((1, 14, 3), dtype=np.float32)
    kp[0, 6] = (11.0, 22.0, 1.0)  # left_hip, occluded
    out = to_coco17(kp)
    assert tuple(out[0, COCO17.index("left_hip")]) == (11.0, 22.0, 1.0)
