"""What a live view does between reading a camera and writing a picture.

Secondary line: this serves the quadruped, and the robot-only entry points that call
it live under `scripts/robot/`. It deliberately did **not** move there with them --
see the paragraph below on what sitting outside the package costs.

Everything here used to live in `scripts/robot/live_view_ros.py`, reachable only by running a
ROS node on a robot. None of it needs ROS: the reach classification is two numpy arrays
and a threshold, the panel builder takes a colour frame and a depth frame, and the
recorder takes images and writes files. Only the subscriptions are ROS, and those stay in
the script.

The move is not tidying. Outside the package this code sat outside the type ratchet, the
coverage floor and the wheel, so the most involved 560 lines in the project were also the
only ones nothing checked.

`scripts/live_view_orin.py` and `scripts/bench_camera_orin.py` are a
different case and deliberately stay where they are -- they run on a board with no
`syncai_hydranet` installed, which is a real constraint rather than an accident, and
`tests/test_orin_standalone_copies.py` is what keeps those copies honest.
"""

from .reach import GO, REACH_BEYOND, REACH_COLORS, REACH_NO_DEPTH, REACH_WITHIN, classify_reach
from .recorder import Recorder
from .render import LiveFrame, LiveSettings, render_frame

__all__ = [
    "GO",
    "REACH_BEYOND",
    "REACH_COLORS",
    "REACH_NO_DEPTH",
    "REACH_WITHIN",
    "LiveFrame",
    "LiveSettings",
    "Recorder",
    "classify_reach",
    "render_frame",
]
