"""`staff` or `customer` for one person, as a model that can be saved and applied.

PLAN §7.15 measured this and stopped there: `scripts/staff_probe.py` fits a probe, prints
a number and writes `probe.json`. **Nothing was persisted, so nothing downstream could
use it** -- the step-9 deliverable is `staff` on `Track`, and a metric in a JSON file is
not a classifier. This module is the missing half: the same arithmetic the probe measured,
plus the two things an artefact needs that a measurement does not -- the standardisation
it was fitted with, and a record of which camera its accuracy was held out on.

---------------------------------------------------------------------------
WHY TORSO COLOUR ALONE, WHEN §7.15's HEADLINE IS THE COMBINATION

§7.15 reports **0.893 for colour + the ImageNet embedding** against **0.880 for colour on
its own**, and the headline went to the combination. The combination is not adoptable, and
the reason is in `probe.json` rather than in an opinion: the combined arm records a single
pooled `balanced_accuracy` and **no `per_camera` block at all**, because the probe's second
loop keeps only the pooled figure. So for any camera we might deploy on, the combined
model's held-out accuracy *cannot be looked up*. Colour on its own has all sixteen, and
they are what the deployment decision is actually made from:

    Kaohsiung-cam04 1.00 (n=15)   Taichung-cam01 1.00 (n=18)   Taichung-cam11 1.00 (n=15)
    Tao-Hsin-cam04  0.42 (n=12)   Tao-Hsin-cam01 0.40 (n=15)

0.013 of pooled balanced accuracy is not worth trading a per-camera number for, when the
per-camera number is the one that says whether a given store may be coloured at all. The
spread above is also the answer to "is 0.893 good enough": it is an average over sixteen
cameras, and two of them are below a coin toss.

---------------------------------------------------------------------------
THE GATE, AND WHAT IT REFUSES

A model applied to a camera it has no held-out number for looks exactly like a model
applied to a camera it scores 1.00 on -- both draw confident colours on every frame. So
`StaffModel` carries `held_out`, and `require_camera` refuses any camera that is not it.
Colouring a camera with no measurement is then something a caller has to ask for in
writing, rather than something it gets by not thinking about it.

`crop_features` exists for the same reason. The probe's nine statistics are taken *after*
a 256x128 resize, and `torso_stats`'s band is fractional, so a caller that skipped the
resize would get plausible numbers from a different feature and no error. One function
does the whole transform, the model records the size it was fitted at, and a mismatch is
raised rather than absorbed.

---------------------------------------------------------------------------
A VERDICT IS PER TRACK, NOT PER FRAME

`scripts/staff_crops.py` established the unit when it built the labelled batch: on
Kaohsiung-cam04 `analytics/track_attributes.py` labelled **the same staff member `F` and
`M` in adjacent frames**, and a shopper is not a per-frame quantity. The same applies
here -- one person is one decision -- so `staff_verdict` takes a track's observations
together and returns `None` below `MIN_OBSERVATIONS` rather than a coin toss from two
crops. `None` is a state a consumer must render as its own thing, not as `customer`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .appearance import TORSO_STAT_NAMES, torso_stats

# `data/attributes.py`'s crop geometry, which is what §7.15's numbers were measured
# through. Stored on the model as well as named here, so a future change to one is a
# loud mismatch rather than a silent re-definition of the feature.
CROP_SIZE = (256, 128)  # (height, width)

# The accuracy a camera has to reach before anything may be coloured on it, and the
# number is derived rather than felt. A three-minute demo clip carries roughly 4-10
# distinct people (24 minutes of eight cameras produced 202 confirmed tracks, §6 step 5),
# so at 0.90 the expected count of miscoloured *people* in one render is below one, and
# at 0.80 it is one to two -- a render that is expected to contain a visible lie. Two of
# the sixteen labelled cameras are below a coin toss, so this is not a formality.
MIN_DEPLOY_ACCURACY = 0.90

# Below this a track has not been seen enough for one decision about one person. Six is
# a little over a second at the 5 fps the demo renders at, and the median site visit is
# 3.2 s (§6 step 5), so it excludes the two-frame fragments and keeps real visits.
MIN_OBSERVATIONS = 6


def crop_features(rgb: np.ndarray) -> np.ndarray:
    """The nine raw statistics for one person crop, through the probe's exact transform.

    `rgb` is an H x W x 3 array, `uint8` or float in [0, 1]. The resize is part of the
    feature and not a caller's convenience: see the module docstring.
    """
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    resized = img.resize((CROP_SIZE[1], CROP_SIZE[0]), Image.Resampling.BILINEAR)
    return torso_stats(np.asarray(resized, dtype=np.float32) / 255.0)


def fit_logreg(x: np.ndarray, y: np.ndarray, *, iters: int = 400, l2: float = 1e-2):
    """Balanced logistic regression, full-batch, no sklearn dependency for 40 lines.

    Class-balanced because the batch is not: the loss weights each class by its inverse
    frequency, so a probe cannot score well by answering "customer" to everything.

    Moved here from `scripts/staff_probe.py` unchanged on 2026-08-28 -- the probe now
    imports it, and its sixteen per-camera accuracies still reproduce `probe.json` to the
    last digit, which is the check that says this was a move and not a rewrite.
    """
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    w = np.zeros(xb.shape[1], dtype=np.float64)
    n_pos, n_neg = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    weight = np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg) * len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-xb @ w))
        grad = xb.T @ (weight * (p - y)) / len(y) + l2 * w
        hess_diag = xb.T @ (weight * p * (1 - p) * xb.T).T / len(y) + l2 * np.eye(len(w))
        w -= np.linalg.solve(hess_diag + 1e-6 * np.eye(len(w)), grad)
    return w


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    return 1.0 / (1.0 + np.exp(-xb @ w))


@dataclass(frozen=True)
class StaffModel:
    """A fitted staff/customer classifier and the provenance that makes it usable.

    `held_out` is the camera this model's `accuracy` was measured on, and the camera it
    is therefore licensed for. It is `None` for a model fitted on everything, which has
    no held-out number and which `require_camera` refuses for every camera.
    """

    weights: np.ndarray  # (10,) -- nine coefficients then the bias
    mean: np.ndarray  # (9,) standardisation, from the fitted set
    std: np.ndarray  # (9,)
    trained_cameras: tuple[str, ...]
    n_crops: int
    held_out: str | None
    accuracy: float | None  # on `held_out`, or None
    held_out_n: int | None
    crop_size: tuple[int, int] = CROP_SIZE

    def probability(self, stats: np.ndarray) -> np.ndarray:
        """P(staff) for raw nine-vectors -- one crop `(9,)` or a batch `(n, 9)`."""
        x = np.atleast_2d(np.asarray(stats, dtype=np.float64))
        if x.shape[1] != len(TORSO_STAT_NAMES):
            raise ValueError(
                f"expected {len(TORSO_STAT_NAMES)} torso statistics "
                f"({', '.join(TORSO_STAT_NAMES)}), got {x.shape[1]}"
            )
        return predict(self.weights, (x - self.mean) / (self.std + 1e-6))

    def probability_of_crop(self, rgb: np.ndarray) -> float:
        return float(self.probability(crop_features(rgb))[0])

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "feature": "analytics.appearance.torso_stats",
                    "stat_names": list(TORSO_STAT_NAMES),
                    "crop_size": list(self.crop_size),
                    "weights": self.weights.tolist(),
                    "mean": self.mean.tolist(),
                    "std": self.std.tolist(),
                    "trained_cameras": list(self.trained_cameras),
                    "n_crops": self.n_crops,
                    "held_out": self.held_out,
                    "accuracy": self.accuracy,
                    "held_out_n": self.held_out_n,
                },
                indent=1,
            )
            + "\n"
        )
        return path

    @staticmethod
    def load(path: Path | str) -> StaffModel:
        d = json.loads(Path(path).read_text())
        if d.get("stat_names") != list(TORSO_STAT_NAMES):
            raise ValueError(
                f"{path} was fitted on statistics {d.get('stat_names')}, and "
                f"`appearance.torso_stats` now returns {list(TORSO_STAT_NAMES)}. The "
                "model's coefficients are positional; refit it rather than loading it."
            )
        if tuple(d.get("crop_size", [])) != CROP_SIZE:
            raise ValueError(
                f"{path} was fitted at crop size {d.get('crop_size')} and this build "
                f"takes features at {list(CROP_SIZE)}; the band is fractional, so the "
                "numbers would look plausible and mean something else. Refit it."
            )
        return StaffModel(
            weights=np.asarray(d["weights"], dtype=np.float64),
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            trained_cameras=tuple(d["trained_cameras"]),
            n_crops=int(d["n_crops"]),
            held_out=d["held_out"],
            accuracy=d["accuracy"],
            held_out_n=d["held_out_n"],
            crop_size=CROP_SIZE,
        )


def fit_staff_model(
    stats: np.ndarray,
    labels: np.ndarray,
    cameras: list[str],
    *,
    held_out: str | None,
) -> StaffModel:
    """Fit on every camera except `held_out`, and score on it.

    This is deliberately the same protocol §7.15 reports, not a re-derivation of it: the
    accuracy the model carries is the accuracy of *this* model on the camera it will be
    pointed at, rather than a pooled average that includes cameras it was fitted on.
    """
    cam = np.asarray(cameras)
    y = np.asarray(labels, dtype=float)
    x = np.asarray(stats, dtype=np.float64)
    test = cam == held_out if held_out is not None else np.zeros(len(cam), dtype=bool)
    if held_out is not None and not test.any():
        raise ValueError(
            f"{held_out!r} is not in the labelled set ({sorted(set(cameras))}), so "
            "holding it out would fit on everything and report an accuracy for nothing."
        )
    # Standardised on the fitted half only. `staff_probe.py` standardises over the whole
    # set including the held-out camera, which is transductive: harmless for its
    # comparison of sources against each other, wrong for an artefact that will meet
    # crops the fit never saw. Measured 2026-08-28 rather than assumed: doing it honestly
    # changes **none** of the sixteen per-camera accuracies, to four decimal places. So
    # the probe's figures transfer to the artefact exactly, and the reason to fit this
    # way is that the next batch of crops may not be so forgiving.
    mean, std = x[~test].mean(0), x[~test].std(0)
    z = (x - mean) / (std + 1e-6)
    w = fit_logreg(z[~test], y[~test])
    acc, n_held = None, None
    if held_out is not None:
        hit = (predict(w, z[test]) >= 0.5).astype(int) == y[test].astype(int)
        acc, n_held = float(hit.mean()), int(test.sum())
    return StaffModel(
        weights=w,
        mean=mean,
        std=std,
        trained_cameras=tuple(sorted(set(cam[~test].tolist()))),
        n_crops=int((~test).sum()),
        held_out=held_out,
        accuracy=acc,
        held_out_n=n_held,
    )


def require_camera(
    model: StaffModel, camera: str, *, min_accuracy: float = MIN_DEPLOY_ACCURACY
) -> StaffModel:
    """Return `model` if it is licensed for `camera` AND good enough on it, else refuse.

    The refusal is the point. Every failure mode this classifier has is invisible in the
    output -- a wrong colour is as confident as a right one -- so "we have no held-out
    number for this camera" has to stop the caller rather than be discoverable later by
    someone looking at the render and wondering.

    **Two conditions, not one.** Matching the camera is not sufficient and the reason is
    concrete: `runs/staff_model01/model_Tao-Hsin-cam04.json` exists, names its own camera,
    and scores **0.417 on 12 crops**. A gate that only asked "is this the right camera"
    would wave it through and colour half that store's staff as shoppers, which is the
    exact shape of failure this project keeps meeting -- a check that passes because it
    was asked the easier question.
    """
    if model.held_out == camera:
        if model.accuracy is not None and model.accuracy < min_accuracy:
            raise ValueError(
                f"this staff model scores {model.accuracy:.3f} on {camera} "
                f"({model.held_out_n} held-out crops), under the {min_accuracy:.2f} floor. "
                "It is the right camera and the wrong number: colouring here would put a "
                "confident wrong answer on roughly one person in "
                f"{max(round(1 / max(1 - model.accuracy, 1e-9)), 1)}. Render this camera "
                "without staff colours and say why, or pass a lower `min_accuracy` and "
                "state in the caller what makes that acceptable."
            )
        return model
    if model.held_out is None:
        raise ValueError(
            f"this staff model was fitted on every labelled camera, so it has no "
            f"held-out accuracy and nothing is known about how it behaves on {camera}. "
            "Refit it holding that camera out."
        )
    raise ValueError(
        f"this staff model's accuracy ({model.accuracy:.2f} on {model.held_out_n} crops) "
        f"was measured on {model.held_out}, not on {camera}. Applying it to {camera} "
        "would draw the same confident colours with no measurement behind them; refit "
        f"holding {camera} out, or state in the caller why an unmeasured camera is "
        "acceptable here."
    )


def track_staff(track) -> bool | None:
    """One verdict for one `analytics.tracker.Track`, from the scores it carries.

    The reduction lives here rather than as a property on `Track` so that `tracker.py`
    stays free of this module's crop geometry, minimum-observation rule and classifier --
    a track records the evidence, and the thing that knows what the evidence means says
    what it means.
    """
    return staff_verdict(track.staff_scores)


def staff_verdict(probabilities) -> bool | None:
    """One decision for one person: `True` staff, `False` customer, `None` not enough.

    The median rather than the mean, because a track's tail can coast onto a neighbour
    (§7.18's two-stage measurement found exactly that on the longest Taichung-cam01
    tracks) and a handful of another person's crops should not move the answer.
    """
    p = np.asarray(list(probabilities), dtype=float)
    if len(p) < MIN_OBSERVATIONS:
        return None
    return bool(np.median(p) >= 0.5)
