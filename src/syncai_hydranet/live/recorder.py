"""Write a walk-through to disk as four things, not one.

The overlay video is for review. The raw keyframes are the point: footage at the robot's
camera height is the single thing this project is short of, and a walk round a building
produces it as a by-product of testing. They are written in the layout
`hydranet-annotation check` expects, under a session directory, because a frame without a
session cannot be split honestly later.

The stats file is what makes the video searchable -- scrubbing 10 minutes of footage for
the moment the floor stopped returning depth is worse than sorting a JSONL by it.

And the calibration: the camera's intrinsic matrix, written once per session. Without K,
nothing recorded here can ever be projected to metres -- no ground plane, no BEV costmap,
no distance gate -- and the omission is invisible until someone tries, by which time the
building has been walked and the robot has moved on. A session without its intrinsics is
the same class of defect as a frame without its session: usable for looking at, useless
for the thing it was collected for.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


class Recorder:
    def __init__(self, root: Path, session: str, keyframe_hz: float):
        self.root = root
        self.img_dir = root / "images" / session
        self.depth_dir = root / "depth" / session
        root.mkdir(parents=True, exist_ok=True)
        if keyframe_hz > 0:
            for d in (self.img_dir, self.depth_dir):
                d.mkdir(parents=True, exist_ok=True)
        # MJPG in AVI: a run that ends with a pulled power cable still plays back. mp4
        # needs its trailer written, and a robot session is exactly when that is missed.
        self.video_path = root / f"{session}_overlay.avi"
        self.writer = None
        self.stats = (root / f"{session}_stats.jsonl").open("a", buffering=1)
        self.calib_path = root / f"{session}_calibration.json"
        self.period = (1.0 / keyframe_hz) if keyframe_hz > 0 else 0.0
        self.last_key = 0.0
        self.n_frames = self.n_keys = 0

    @property
    def cv2(self):
        """Imported on first use rather than in __init__.

        Only `write` needs it. Requiring OpenCV to exist before a session can record its
        own intrinsics -- which is JSON -- made `write_calibration` unreachable anywhere
        cv2 is not installed, including every machine that runs the tests.
        """
        import cv2  # pyright: ignore[reportMissingImports]

        return cv2

    def write_calibration(self, color_info, depth_info, stride: int) -> None:
        """Intrinsics, once. They do not change during a session, and the stride this
        script applies to the stream does change them -- so what is written is the
        calibration of the frames on disk, not of the topic."""
        if self.calib_path.exists():
            return

        def one(msg):
            if msg is None:
                return None
            k = list(msg.k)
            # Subsampling scales fx, fy, cx, cy by exactly the same factor. Writing the
            # topic's K next to decimated frames would be off by that factor, silently.
            if stride > 1:
                k = [v / stride if i in (0, 2, 4, 5) else v for i, v in enumerate(k)]
            return {
                "K": k,
                "distortion_model": msg.distortion_model,
                "D": list(msg.d),
                "width": msg.width // stride,
                "height": msg.height // stride,
                "frame_id": msg.header.frame_id,
            }

        payload = {
            "color": one(color_info),
            "depth_aligned_to_color": one(depth_info),
            "subsample_stride": stride,
            "depth_units": "millimetres, uint16",
            "note": "K is scaled for the frames written here, not the raw topic",
        }
        self.calib_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {self.calib_path}", flush=True)

    def write(self, pair: Image.Image, color: np.ndarray, depth_mm: np.ndarray, stats: dict):
        cv2 = self.cv2
        frame = cv2.cvtColor(np.asarray(pair), cv2.COLOR_RGB2BGR)
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h)
            )
        self.writer.write(frame)
        self.n_frames += 1

        now = time.time()
        if self.period > 0 and now - self.last_key >= self.period:
            self.last_key = now
            stem = f"{self.n_keys:06d}"
            cv2.imwrite(
                str(self.img_dir / f"{stem}.jpg"),
                cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            # 16-bit PNG keeps millimetres exactly; a lossy depth frame is a lie.
            cv2.imwrite(str(self.depth_dir / f"{stem}.png"), depth_mm)
            self.n_keys += 1
            stats = {**stats, "keyframe": stem}
        self.stats.write(json.dumps({"t": round(now, 3), **stats}) + "\n")

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.stats.close()

    # A context manager because `close()` is the part that makes a session readable --
    # it releases the video writer and closes the stats handle -- and every caller that
    # reached it through a straight-line `finally`-less path lost both when the loop
    # above it raised. The stats file is opened line-buffered so its content survives
    # regardless; the file *descriptor* does not, and a robot process that restarts its
    # session loop leaks one per restart.
    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
