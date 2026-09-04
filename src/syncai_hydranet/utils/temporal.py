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

---------------------------------------------------------------------------
THE SELF-SEALING FAILURE, AND WHY THIS MODULE NOW REFUSES RATHER THAN COASTING

The gate above is also this module's worst failure, and three sessions reproduced it
independently on 2026-08-17. The plate is updated `np.where(static, blended, plate)`, so
it is frozen **exactly where it disagrees with the scene, which is exactly where it is
wrong**. A wrong plate makes a region permanently non-static, and permanent non-static
freezes the plate. It does not recover: 200 empty frames does not help and 20,000 would
not either.

Two ways in, both ordinary rather than exotic:

* **An occupant in the very first frame** is the background from that moment. When they
  leave, the vacated floor permanently disagrees with a plate that still holds them. A
  retail clip that opens with nobody in shot is the exception, not the rule.
* **An illumination change.** `static` is a raw grey-level comparison and cannot tell "the
  scene changed" from "the exposure changed". The cliff is `diff_thr` and it is a cliff:
  an 11-level shift is fully smoothed, 12 is a whole-frame no-op. A ramp survives only
  while `r < diff_thr * plate_alpha` = 0.24 levels/frame, or 1.44 per second at 6 fps.

**What made this expensive is not the bug, it is the shape of the failure.** The module
degrades to a no-op over precisely the floor it was built to settle, while every other
region still smooths -- so the panel looks *better*, not broken. It does not fail quietly;
it actively looks like it is working.

So the gate closing over essentially the whole frame, and staying closed, now raises
`StabiliserSelfSealedError` instead of returning the live prediction forever. A caller that
would rather continue unsmoothed can catch it -- but it has to say so, which is the whole
point. Silence was never the honest option, and a filter that smooths nothing is
indistinguishable in the output from one that works.

The real fix is a **caller-vouched plate**: a background computed once per camera from
footage somebody has confirmed is empty. Per camera *and per lighting condition* -- an IR
night plate on a daytime clip gates 100% of the frame, which is this same failure promoted
to global. That fix is not built; this refusal is what makes its absence visible.
"""

from __future__ import annotations

import numpy as np


class StabiliserSelfSealedError(RuntimeError):
    """The change-gate has been shut over ~the whole frame long enough to be permanent.

    Raised rather than returned because the alternative is the failure this module is most
    criticised for: passing every pixel through untouched, forever, while reporting
    nothing. A caller that genuinely wants to continue unsmoothed should catch this and
    say so in its own output.
    """


class FixedCameraStabiliser:
    """Per-pixel temporal majority, applied only where the scene has not changed.

    Args:
        window: frames of history to vote over. At 6 fps, 9 is 1.5 s.
        num_classes: size of the label space being voted on.
        diff_thr: grey-level difference above which a pixel counts as changed.
        plate_alpha: how fast the background plate follows the scene, per frame.
        min_static_share: below this fraction of static pixels the filter is doing
            essentially nothing. 2% is deliberately generous -- the measured failures sit
            at exactly 0.0%, and a legitimately busy shop floor still leaves most of the
            frame unchanged, so this separates "sealed" from "crowded" with room to spare.
        seal_patience: consecutive frames a PIXEL's gate stays shut before that pixel
            counts as sealed -- the per-pixel `_shut_for` threshold below. Nothing here
            keeps a global counter over `min_static_share`, which is the natural misread
            and would seal the whole frame at once. 60 is 10 s at 6 fps: long enough to
            ride out a delivery trolley crossing the whole view, short enough to catch a
            seal before a five-minute clip has been rendered.

    `diff_thr` and `plate_alpha` had no provenance when this was written -- never
    committed, named in no config, and unreachable from `--stabilise`, which passes only
    `window`. They are still defaults, but the number that has to be justified is their
    **product**, because that is what governs whether the filter runs at all.
    """

    def __init__(
        self,
        window: int = 9,
        num_classes: int = 3,
        diff_thr: float = 12.0,
        plate_alpha: float = 0.02,
        min_static_share: float = 0.02,
        seal_patience: int = 60,
    ):
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = int(window)
        self.num_classes = int(num_classes)
        self.diff_thr = float(diff_thr)
        self.plate_alpha = float(plate_alpha)
        self.min_static_share = float(min_static_share)
        self.seal_patience = int(seal_patience)
        self._ring: list[np.ndarray] = []
        self._plate: np.ndarray | None = None
        # Per pixel, how many consecutive frames the gate has been shut on it. One array
        # answers both readings of the same failure: essentially all of it above
        # `seal_patience` is the global seal, and a stable minority above it is the
        # localised one a seed-time occupant leaves behind.
        self._shut_for: np.ndarray | None = None
        self._static_share = 1.0

    @staticmethod
    def _grey(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.float32)
        return frame[..., :3].astype(np.float32).mean(axis=-1)

    def reset(self) -> None:
        self._ring.clear()
        self._plate = None
        self._shut_for = None
        self._static_share = 1.0

    def sealed_share(self) -> float:
        """Fraction of pixels the gate has been shut on for `seal_patience` frames running.

        These pixels are not merely occupied, they are **stuck**: the plate updates only
        where the scene is already static, so a pixel that has disagreed for this long has
        stopped the very process that would let it agree again. It will not come back on
        its own.

        Deliberately *not* "pixels that were never static". That measure sounds equivalent
        and catches nothing, because the plate is seeded from frame 1 outright -- an
        occupant standing there at construction **is** the background, so their pixels
        read static on the first frame and are marked for the rest of the clip. The
        failure only becomes visible when they leave.
        """
        if self._shut_for is None:
            return 0.0
        return float(np.mean(self._shut_for >= self.seal_patience))

    def diagnostics(self) -> dict[str, float]:
        """What the gate is doing, for a caller that wants to report it rather than guess.

        `sealed_share` below the refusal threshold is the localised failure: a region
        permanently excluded from smoothing while the rest of the frame smooths normally,
        which reads as a *cleaner* panel rather than a broken one. It is reported and not
        refused, because a genuinely busy doorway produces the same signature and is
        legitimate. Only the caller knows which it is looking at -- but it cannot know at
        all unless the number is available.
        """
        return {
            "static_share": self._static_share,
            "sealed_share": self.sealed_share(),
        }

    def __call__(self, frame: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """`frame` is the RGB image the prediction came from; `pred` is HxW class ids.

        Raises:
            StabiliserSelfSealedError: the gate has been shut over more than
                `1 - min_static_share` of the frame for `seal_patience` consecutive
                frames, so this filter is a no-op and will not recover on its own.
        """
        grey = self._grey(frame)
        if self._plate is None or self._plate.shape != grey.shape:
            self.reset()
            self._plate = grey.copy()

        static = np.abs(grey - self._plate) < self.diff_thr
        self._static_share = float(np.mean(static))
        if self._shut_for is None:
            self._shut_for = np.zeros(static.shape, np.int32)
        self._shut_for = np.where(static, 0, self._shut_for + 1)

        # Counted before the vote, so a stabiliser that is already sealed refuses on the
        # frame that proves it rather than after silently returning one more live map.
        sealed = self.sealed_share()
        if sealed >= 1.0 - self.min_static_share:
            raise StabiliserSelfSealedError(
                f"the change-gate has been shut on {sealed:.1%} of the frame for "
                f"{self.seal_patience} consecutive frames. The plate only updates "
                f"where the scene is already static, so this does not recover -- it is a "
                f"no-op from here on, and an unreported no-op looks exactly like a filter "
                f"that ran and found nothing to correct. Usual causes: an occupant present "
                f"in the first frame, or an illumination change past diff_thr="
                f"{self.diff_thr} (a ramp survives only below "
                f"diff_thr*plate_alpha={self.diff_thr * self.plate_alpha:g} levels/frame). "
                f"Prime the plate from footage confirmed empty *under this clip's "
                f"lighting*, or catch this exception and report the clip as unsmoothed."
            )

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
