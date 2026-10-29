"""CrowdPose's 14 keypoints -> the COCO-17 order the pose head is built on.

CrowdPose exists in this project for one reason: `site_pose` is ViTPose run over the
site's own frames, so every occlusion the teacher gets wrong is an occlusion the student
learns wrong, and pose02 measured the shape of that -- PCK@0.2h 0.935 against an L2 p90
of 44.1 px, a tail six times the 7.3 px median. CrowdPose is 20,000 images and 51,501
people selected *for* crowding, with human labels rather than a teacher's guess.

The mapping itself is trivial; the three things that are not are below, and each one
fails silently if it is got wrong.

1. **12 of the 14 map, and the other 2 are dropped, not translated.**
   CrowdPose describes the head with `head` and `neck`; COCO describes it with
   `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`. There is no correspondence
   between the two -- `head` is the top of the skull, not the nose -- so writing `head`
   into the `nose` channel trains that channel on a point that sits systematically
   above where it belongs. Dropped, and the head is not a joint any of the behaviour
   rules read: crouch, fall and reach are decided by shoulders, hips, knees, ankles
   and wrists, all of which CrowdPose supplies.

2. **The five face channels must be written `v = 0`, and the loss must mask the whole
   channel.** COCO's visibility flag is 0 = not labelled, 1 = labelled but occluded,
   2 = labelled and visible. Writing `v = 0` is only half the job: a heatmap loss that
   renders no peak but still scores the channel turns "nobody labelled this" into
   "there is no nose anywhere in this image", which is a false negative on every
   CrowdPose frame. `PoseHeatmapLoss` must skip a channel whose `v` is 0, not merely
   skip its peak.

3. **v = 1 is the majority class here, and that is the point.** Measured over 20,000
   annotations: v=0 94,438, v=1 122,704, v=2 62,858. In a frontal dataset the occluded
   flag is a rounding error; here it outnumbers the visible one, which is the whole
   reason this download is worth its 4.7 GB. A loader that keeps only `v == 2` throws
   away the half of CrowdPose that answers the question it was downloaded for.

`tests/test_crowdpose_scheme.py` re-derives the index order from the shipped
`categories[0].keypoints`, so a re-release that reorders them cannot pass silently.
"""

from __future__ import annotations

import numpy as np

#: COCO-17 order. The pose head's channel i is this list's entry i.
COCO17 = (
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

#: CrowdPose order, as shipped in `categories[0]["keypoints"]`.
CROWDPOSE14 = (
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
    "head",
    "neck",
)

#: CrowdPose names with no COCO-17 channel. See note 1 -- dropped, never translated.
UNMAPPED = ("head", "neck")

#: COCO-17 channels CrowdPose cannot supply. Written v=0; see note 2.
UNSUPPLIED = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")

#: CrowdPose index -> COCO-17 index, built by name so a reorder on either side is a
#: KeyError rather than a silent limb swap.
CROWDPOSE_TO_COCO17 = {
    i: COCO17.index(name) for i, name in enumerate(CROWDPOSE14) if name not in UNMAPPED
}


def to_coco17(keypoints) -> np.ndarray:
    """(N, 14, 3) CrowdPose keypoints -> (N, 17, 3) in COCO order.

    Unsupplied channels come back as ``(0, 0, 0)``: zero coordinates with ``v = 0``,
    which is COCO's "not labelled". The coordinates are meaningless and must never be
    read without checking ``v`` first.
    """
    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 14, 3)
    out = np.zeros((kp.shape[0], 17, 3), dtype=np.float32)
    for src, dst in CROWDPOSE_TO_COCO17.items():
        out[:, dst, :] = kp[:, src, :]
    return out
