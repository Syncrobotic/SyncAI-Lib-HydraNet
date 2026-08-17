"""Stabilise a fixed camera's segmentation without touching the model.

On `archive_20260802-125220` the traversability head returns `go` consistently for only
2.2% of the frame and flickers between walkable and blocked on 16.7%, concentrated on the
brighter specular near-field tiles. The floor is not moving. The camera is not moving.
The disagreement is per-frame noise on a scene that is constant, and the cheapest correct
answer to noise on a constant is to look at more than one frame.

**The rule that makes this safe.** A pixel is smoothed only where the *image* is
unchanged. Where the picture differs from the accumulated background -- someone standing
there, a trolley left in the aisle, a box put down -- the live prediction is passed
through untouched. So the filter can settle an argument about empty floor and can never
average a person away, which is the only failure that would matter.

Two consequences worth stating rather than discovering:

* A pixel that has been occupied and becomes free again keeps the occupant in its history
  for up to `window` frames, so it reads `blocked` slightly longer than it should. That is
  the conservative direction, and it costs at most `window / fps` seconds.
* Nothing here makes the model better. A region the model is *consistently* wrong about
  stays wrong, and looks more confident for having been smoothed. The 16.7% this was
  built for is genuine disagreement; the parts of it where the model is steadily wrong
  need labels, not filtering.
"""

from __future__ import annotations

import numpy as np


class FixedCameraStabiliser:
    """Per-pixel temporal majority, applied only where the scene has not changed.

    Args:
        window: frames of history to vote over. At 6 fps, 9 is 1.5 s.
        num_classes: size of the label space being voted on.
        diff_thr: grey-level difference above which a pixel counts as changed.
        plate_alpha: how fast the background plate follows the scene, per frame.
    """

    def __init__(
        self,
        window: int = 9,
        num_classes: int = 3,
        diff_thr: float = 12.0,
        plate_alpha: float = 0.02,
    ):
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = int(window)
        self.num_classes = int(num_classes)
        self.diff_thr = float(diff_thr)
        self.plate_alpha = float(plate_alpha)
        self._ring: list[np.ndarray] = []
        self._plate: np.ndarray | None = None

    @staticmethod
    def _grey(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.float32)
        return frame[..., :3].astype(np.float32).mean(axis=-1)

    def reset(self) -> None:
        self._ring.clear()
        self._plate = None

    def __call__(self, frame: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """`frame` is the RGB image the prediction came from; `pred` is HxW class ids."""
        grey = self._grey(frame)
        if self._plate is None or self._plate.shape != grey.shape:
            self.reset()
            self._plate = grey.copy()

        static = np.abs(grey - self._plate) < self.diff_thr

        self._ring.append(pred.astype(np.uint8, copy=True))
        if len(self._ring) > self.window:
            self._ring.pop(0)

        # The plate follows only what is already static, so a person who walks in never
        # becomes part of the background and never earns the right to be smoothed.
        self._plate = np.where(
            static, self._plate * (1 - self.plate_alpha) + grey * self.plate_alpha, self._plate
        )

        if len(self._ring) < self.window:
            return pred  # not enough history to outvote anything

        stack = np.stack(self._ring)
        counts = np.stack([(stack == c).sum(axis=0) for c in range(self.num_classes)])
        majority = counts.argmax(axis=0).astype(pred.dtype)
        return np.where(static, majority, pred)
