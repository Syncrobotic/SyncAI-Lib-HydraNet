"""The two camera calibrations the suite measures against, in the one place they can move.

``HALF_RES_*`` is Taichung-cam01's real shape: intrinsics fitted on 960x540 while the
clips decode at 1920x1080 -- the mismatch that once returned metres instead of an error.
It was copied verbatim into three test files, which was three places to update when the
fleet's numbers move. ``FULL_RES_*`` is the measured store mount -- height 2.38 m, pitch
50.2 deg -- the pose ``hm3d_cctv`` renders at and the one docs/PLAN.md quotes, so a metre
here is the same metre as everywhere else; it was inlined in three more files.

Both dataclasses are frozen, so sharing the instances couples no test to another.

A leading underscore rather than ``conftest.py`` for the same reason as `_posture.py`:
these are plain constants, not fixtures, and pytest puts this directory on ``sys.path``
so a bare ``from _cameras import ...`` resolves under the documented command.
"""

import math

from syncai_hydranet.geometry.ground import Camera, GroundPlane

HALF_RES_SIZE = (960, 540)
HALF_RES_CAM = Camera(fx=382.7, fy=382.7, cx=480.0, cy=270.0)
HALF_RES_PLANE = GroundPlane(height=2.49, pitch=math.radians(49.5))

FULL_RES_SIZE = (1920, 1080)
FULL_RES_CAM = Camera(fx=1490.0, fy=1490.0, cx=960.0, cy=540.0)
FULL_RES_PLANE = GroundPlane(height=2.38, pitch=math.radians(50.2))
