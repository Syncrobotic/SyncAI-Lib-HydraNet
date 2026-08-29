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
* **And per camera, where a camera has earned an exception.** The 2026-08-26 sweep
  lowered person's birth threshold to 0.15 fleet-wide: seven healthy cameras gained
  +51% detections, Kaohsiung-cam04 went 3,353 -> 13,595, which is its open
  person-score investigation and not recall. `ThresholdBook` carries the fleet
  default plus per-camera overrides, each of which must state the measurement that
  produced it -- see its docstring for why a blank basis is refused at load.
* **Tracker is injected.** The measured tracker is
  ``analytics.bytetrack.OfflineForward`` (the mechanism stable_infer decided into
  the main path). The state object takes a factory rather than importing it, so a
  deployment can swap the tracker without touching per-camera state, and the
  isolation tests run on a stub.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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

THRESHOLD_SCHEMA = "hydranet-thresholds/v1"


@dataclass(frozen=True)
class ThresholdBook:
    """Per-class working thresholds, with per-camera overrides that state their evidence.

    **Why per camera, measured 2026-08-26.** The three-arm threshold sweep lowered the
    person birth threshold to 0.15 across the fleet. On seven healthy cameras that bought
    +51% detections and +141% tracks. On Kaohsiung-cam04 it took detections from 3,353 to
    **13,595** -- four times the fleet's factor -- which is that camera's open
    person-score investigation surfacing, not recall. One number cannot serve both, and
    the module already argued the shape of the answer: a working point is a property of a
    calibration, not a constant in code. A camera is another axis of that calibration.

    **An override must name what measured it.** `basis` is required on every camera entry
    and a blank one is refused at load. A per-camera threshold with no stated reason is
    indistinguishable from a number somebody tuned until the alerts stopped, and this
    repository's standing rule is to check what produced a number before attributing it
    to a mechanism. The fleet default carries the same obligation in
    `DEFAULT_THRESHOLDS`' comment, which is where its 0.35/0.20 hysteresis is sourced.
    """

    default: dict[str, ClassThresholds]
    cameras: dict[str, dict[str, ClassThresholds]] = field(default_factory=dict)
    bases: dict[str, str] = field(default_factory=dict)

    def for_camera(self, camera: str) -> dict[str, ClassThresholds]:
        """This camera's thresholds: the fleet default with its own overrides applied.

        Per class, not per camera wholesale -- a camera whose `person` calibration is
        under investigation has no reason to opt out of the fleet's `bag` working point,
        and copying every class into its entry to change one is how the two drift.
        """
        merged = dict(self.default)
        merged.update(self.cameras.get(camera, {}))
        return merged

    def basis_for(self, camera: str) -> str | None:
        """What measured this camera's override, or None when it has none."""
        return self.bases.get(camera)


def _class_thresholds(raw: Mapping[str, Any], where: str) -> dict[str, ClassThresholds]:
    out: dict[str, ClassThresholds] = {}
    for cls, spec in raw.items():
        try:
            out[cls] = ClassThresholds(birth=float(spec["birth"]), keep=float(spec["keep"]))
        except KeyError as exc:
            raise ValueError(f"{where}: class {cls!r} needs both birth and keep") from exc
    return out


def load_thresholds(path: str | Path) -> ThresholdBook:
    """Read a threshold book from disk. Ships beside the engine, not in code.

    The file names the checkpoint it was derived from, because a score calibration is a
    property of one set of weights -- the b03_gdino retrain moved `boxed_stock` from 5.5
    to 0.12 boxes/frame at a fixed threshold with mAP unchanged. Nothing here enforces
    that the running engine matches; it is recorded so a reader can check, which is the
    same bargain `camera.json` makes with its own provenance fields.
    """
    p = Path(path)
    book = json.loads(p.read_text())
    if book.get("schema") != THRESHOLD_SCHEMA:
        raise ValueError(f"{p}: unexpected thresholds schema {book.get('schema')!r}")
    default = _class_thresholds(book.get("default", {}), f"{p}: default")
    if not default:
        raise ValueError(f"{p}: no default thresholds -- a book of overrides alone is not one")
    cameras: dict[str, dict[str, ClassThresholds]] = {}
    bases: dict[str, str] = {}
    for name, entry in book.get("cameras", {}).items():
        basis = str(entry.get("basis", "")).strip()
        if not basis:
            raise ValueError(
                f"{p}: camera {name!r} overrides a threshold without a `basis`. A "
                "per-camera working point with no stated measurement cannot be told "
                "apart from a number tuned until the alerts stopped."
            )
        classes = _class_thresholds(
            {k: v for k, v in entry.items() if k != "basis"}, f"{p}: camera {name}"
        )
        if not classes:
            raise ValueError(f"{p}: camera {name!r} has a basis and overrides nothing")
        cameras[name] = classes
        bases[name] = basis
    return ThresholdBook(default=default, cameras=cameras, bases=bases)


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
