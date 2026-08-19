"""Per-camera serving state: label EMA, tracker, calibration handle, thresholds.

Each camera owns its own instance; nothing here is shared. The state a stream
accumulates -- smoothed terrain labels, live tracks, its geometry -- is exactly the
state that must never bleed into a neighbouring stream when both ride the same
engine batch, which is what the isolation tests pin down.

Design notes, each tied to a measurement:

* **EMA over one-hot labels, not logits.** The shipped engine folds the terrain
  argmax into the graph (export_onnx --argmax-seg), so the host sees uint8 class
  ids: the logits were deliberately traded for a 17 MB -> 0.7 MB D2H. The EMA that
  measured 1.50% -> 0.67% static flips (runs/stable01) smoothed probabilities; over
  one-hot ids the same recursion becomes majority-vote smoothing with the same flip
  condition -- argmax changes once ``(1 - alpha)^n < 0.5``, 2 frames at alpha 0.35.
  What is lost against the logit version is only the model's own confidence shading
  between frames, which the exported graph does not emit in the first place.
* **Per-class working thresholds, not one constant.** The b03_gdino retrain moved
  ``boxed_stock``'s score distribution wholesale -- 5.5 -> 0.12 boxes/frame at a
  fixed 0.30 render threshold with mAP unchanged (0.1174 -> 0.1179). A score
  calibration is a property of one checkpoint, so thresholds ship as per-class
  config beside the engine, never as a constant in code.
* **Tracker is injected.** The measured tracker is
  ``analytics.bytetrack.OfflineForward`` (the mechanism stable_infer decided into
  the main path). The state object takes a factory rather than importing it, so a
  deployment can swap the tracker without touching per-camera state, and the
  isolation tests run on a stub.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ClassThresholds:
    """Working thresholds for one detection class.

    ``birth`` admits a new track (the 0.35 edge of the measured hysteresis),
    ``keep`` sustains an existing one (the 0.20 edge). Both are per-class because a
    retrain can move one class's score calibration without touching the others.
    """

    birth: float
    keep: float

    def __post_init__(self):
        if not 0.0 < self.keep <= self.birth <= 1.0:
            raise ValueError(f"need 0 < keep <= birth <= 1, got {self}")


# Defaults for the retail_security det vocabulary. person's 0.35/0.20 pair is the
# hysteresis measured in runs/stable01 (one-frame boxes 40% -> 11.7%); bag and
# device inherit it unmeasured. boxed_stock is the class whose calibration the
# b03_gdino retrain shifted -- its working point must be re-derived per checkpoint
# from a score sweep (the pilot records per-class score distributions for exactly
# this); the value here is a placeholder that keeps the class visible rather than a
# measured operating point.
DEFAULT_THRESHOLDS: dict[str, ClassThresholds] = {
    "person": ClassThresholds(birth=0.35, keep=0.20),
    "bag": ClassThresholds(birth=0.35, keep=0.20),
    "boxed_stock": ClassThresholds(birth=0.35, keep=0.20),
    "device": ClassThresholds(birth=0.35, keep=0.20),
}

# The reference edge the injected tracker's single hysteresis is built on. A
# detection's score is rescaled by (BIRTH_REF / its class's birth) before the
# tracker sees it, which turns the tracker's global 0.35/0.20 band into the
# per-class band -- exact whenever a class keeps the default keep/birth ratio.
BIRTH_REF = 0.35


class CameraState:
    """Everything one stream accumulates across ticks. One instance per camera."""

    def __init__(
        self,
        camera: str,
        num_terrain_classes: int,
        canvas_hw: tuple[int, int],
        det_classes: list[str],
        thresholds: dict[str, ClassThresholds] | None = None,
        calib_path: str | Path | None = None,
        tracker_factory: Callable[[], Any] | None = None,
        ema_alpha: float = 0.35,
    ):
        self.camera = camera
        self.num_terrain_classes = int(num_terrain_classes)
        self.canvas_hw = (int(canvas_hw[0]), int(canvas_hw[1]))
        self.det_classes = list(det_classes)
        # Copy so two cameras can never share one dict by aliasing a default.
        base = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
        self.thresholds = {
            c: base.get(c, ClassThresholds(birth=BIRTH_REF, keep=0.20))
            for c in self.det_classes
        }
        self.ema_alpha = float(ema_alpha)
        self._ema: torch.Tensor | None = None  # [C,HW] float32, in _ema_scale units
        self._ema_scale = 1.0
        self._best: torch.Tensor | None = None  # argmax, maintained incrementally
        self._best_val: torch.Tensor | None = None
        self.tracker = tracker_factory() if tracker_factory is not None else None
        self.calib = self._load_calib(calib_path)
        self.frames_seen = 0
        self.last_seq = -1

    @staticmethod
    def _load_calib(path: str | Path | None) -> dict | None:
        """runs/onboard01/<camera>.calib.json, or None -- a camera without geometry
        still serves detection and segmentation; only metric events need this."""
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        calib = json.loads(p.read_text())
        if calib.get("schema") != "hydranet-onboard-calib/v1":
            raise ValueError(f"{p}: unexpected calib schema {calib.get('schema')!r}")
        return calib

    # -- terrain -----------------------------------------------------------------
    def ema_labels(self, labels: np.ndarray) -> np.ndarray:
        """Fold one uint8 class map into the EMA; return the smoothed map.

        Semantically: decay every class by (1 - alpha), add alpha at each pixel's
        observed class, argmax -- the logit-EMA rule with a one-hot observation,
        per the module docstring.

        Implemented incrementally, in torch, because the naive numpy form
        (full-buffer decay + full argmax) measured 10.9 ms/frame -- 175 ms per
        16-stream tick, the pilot's single largest cost -- and numpy's advanced
        indexing holds the GIL, so a thread pool bought almost nothing. Three
        observations make it cheap:

        * Uniform decay preserves the order between classes, so the buffer can
          stay in a scaled space (true value = stored * ``_ema_scale``) and only
          the observed class's row is touched per frame -- one gather+scatter
          instead of a C x HW multiply. The scale is folded back in occasionally
          before float32 runs out of exponent.
        * Only the observed class ever *gains*, so the argmax can only move to
          it: comparing the updated value against the running best replaces the
          full argmax. Strict ``>`` keeps exact ties with the incumbent where
          argmax's lowest-index rule would (a pixel's own class still advances,
          because its updated value strictly exceeds its previous best).
        * torch ops release the GIL where numpy's fancy indexing cannot, so 16
          per-camera updates in a thread pool actually overlap.

        Measured: 10.9 ms/frame naive numpy, 3.6 ms incremental numpy, 0.9 ms
        incremental torch -- and a 16-camera pooled tick 12.7 ms against numpy's
        ~110 ms. Output is bit-identical to the naive recursion (the equivalence
        test runs 120 frames across the renormalisation).
        """
        h, w = self.canvas_hw
        if labels.shape != (h, w):
            raise ValueError(f"{self.camera}: labels {labels.shape}, expected {(h, w)}")
        lab = torch.from_numpy(np.ascontiguousarray(labels.reshape(1, -1))).to(torch.int64)
        if self._ema is None or self._best is None or self._best_val is None:
            self._ema = torch.zeros(self.num_terrain_classes, h * w, dtype=torch.float32)
            self._ema.scatter_(0, lab, 1.0)
            self._ema_scale = 1.0
            self._best = lab[0].to(torch.uint8)
            self._best_val = torch.ones(h * w, dtype=torch.float32)
            return self._best.numpy().reshape(h, w).copy()
        self._ema_scale *= 1.0 - self.ema_alpha
        inc = self.ema_alpha / self._ema_scale
        cur = self._ema.gather(0, lab)[0] + inc
        self._ema.scatter_(0, lab, cur.unsqueeze(0))
        upd = cur > self._best_val
        self._best = torch.where(upd, lab[0].to(torch.uint8), self._best)
        self._best_val = torch.where(upd, cur, self._best_val)
        if self._ema_scale < 1e-20:
            # Fold the pending decay back in before inc overflows float32. History
            # below ~1e-20 of the current frame's weight underflows to zero here,
            # which the true EMA would also treat as gone for every practical
            # comparison.
            self._ema.mul_(self._ema_scale)
            self._best_val.mul_(self._ema_scale)
            self._ema_scale = 1.0
        return self._best.numpy().reshape(h, w).copy()

    # -- detection ---------------------------------------------------------------
    def filter_and_scale(
        self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply per-class keep thresholds; rescale scores onto the tracker's band.

        Returns (boxes, scaled_scores, labels) with only detections at or above
        their class's keep threshold, scores multiplied by BIRTH_REF/birth[class]
        so the tracker's single 0.35/0.20 hysteresis acts per-class.
        """
        if len(boxes) == 0:
            return boxes, scores, labels
        births = np.array(
            [self.thresholds[self.det_classes[int(c)]].birth for c in labels], np.float64
        )
        keeps = np.array(
            [self.thresholds[self.det_classes[int(c)]].keep for c in labels], np.float64
        )
        sel = scores >= keeps
        return boxes[sel], scores[sel] * (BIRTH_REF / births[sel]), labels[sel]

    def update(
        self,
        seq: int,
        terrain_labels: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        """One consumed frame: smooth terrain, feed the tracker, return the stable view."""
        stable = self.ema_labels(terrain_labels)
        boxes, scaled, labels = self.filter_and_scale(boxes, scores, labels)
        tracked = None
        if self.tracker is not None:
            self.tracker.update(boxes, scaled, self.frames_seen)
            tracked = confirmed_track_boxes(self.tracker, labels, boxes, self.canvas_hw)
        self.frames_seen += 1
        self.last_seq = int(seq)
        return {
            "terrain": stable,
            "boxes": boxes,
            "scores": scaled,
            "labels": labels,
            "tracks": tracked,
        }


def confirmed_track_boxes(
    tracker: Any, labels: np.ndarray, boxes: np.ndarray, canvas_hw: tuple[int, int]
) -> list[dict[str, Any]]:
    """Confirmed live tracks -> render/event boxes, stable_infer's consumption rule.

    A track observed this frame contributes its observed box; a coasting track (age
    > 0) contributes its Kalman prediction, which is how a missed detection does not
    blink the box off. Class is the label of the nearest observed box this frame,
    recorded as a vote so a track's class is its majority opinion, not its last one.
    """
    h, w = canvas_hw
    out: list[dict[str, Any]] = []
    for t in tracker.tracks:
        observed = bool(t.frames) and t.age == 0
        if observed and len(boxes):
            j = int(np.abs(boxes - np.asarray(t.boxes[-1])).sum(axis=1).argmin())
            t.__dict__.setdefault("label_votes", []).append(int(labels[j]))
        if not t.confirmed:
            continue
        votes = t.__dict__.get("label_votes")
        if not votes:
            continue
        box = np.asarray(t.boxes[-1] if t.age == 0 else t.kalman.box, np.float64).copy()
        box[0::2] = box[0::2].clip(0, w)
        box[1::2] = box[1::2].clip(0, h)
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue
        counts = np.bincount(np.asarray(votes, np.int64))
        out.append(
            {
                "id": int(t.frag_id),
                "box": box,
                # numpy 2.5 stubs reject a plain argmax(); the code is right (same
                # stub family as stable_infer.py's vote count), so the ignore is
                # scoped.
                "label": int(counts.argmax()),  # ty: ignore[no-matching-overload]
                "coasting": t.age > 0,
            }
        )
    return out
