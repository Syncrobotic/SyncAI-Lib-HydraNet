"""Per-pixel frame consensus on a fixed camera: a target-domain measurement with no labels.

**Read this before quoting a number out of it.**

Consensus is a *necessary condition, not a quality metric*. It asks whether the model gives
the same answer twice about a scene that did not change, and nothing else. A model that is
confidently and identically wrong on every frame scores 100%. Everything here can be
excellent while the model is useless, and that is not a flaw in the measurement -- it is
what the measurement is.

What earns it a place anyway is that it is the only instrument in this repository that
works **on the store**, today, with nobody annotating anything. The gap it addresses is the
one RETAIL_DATA.md opens with: `runs/hydranet_retail_objects` scored terrain mIoU
**0.7668** on ADE20K and predicted `column` **0.00%** and `product` **0.00%** on the
footage it is meant to serve. Both numbers are true. Every labelled split this project has
is the first kind, and a labelled split of the second kind needs human-corrected masks (R3)
that do not exist yet. Meanwhile every architecture question -- `num_convs`, a neck
attention block, an open-vocabulary head -- wants to be judged on the shop.

The camera does not move, so a pixel can be asked the same question N times and the spread
of the answers is an error signal that costs no labels. RETAIL.md's audit used it
as the `agree` column; RETAIL.md §5 used it to count a false `caution` on a
structural column through 1,782 of 1,830 frames, where four frames sampled by eye had
suggested the opposite conclusion. Both were one-off scripts.

Three things the design owes to `274a4a08`, who measured the same shape over SAM 3's masks
before this existed, and whose version of it is about label stability where this one is
about model stability:

1. **Excluded classes are not optional.** People move by definition, so without dropping
   every pixel a person ever occupies, a single shopper walking an aisle drags the aisle
   into "inconsistent" and the metric becomes a people-counter.
2. **The denominator has to travel with the number.** Eligible pixels are not all pixels,
   and a figure over one is not comparable to a figure over the other.
3. **The scalar is not the finding; the per-class breakdown of where the instability sits
   is.** Their run put 71.9% of unstable pixels on `fixture` -- a single mean would have
   hidden that, and the mean is the part that gets quoted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# What a report says about itself, carried in the JSON rather than in a docstring nobody
# reads at the point of quoting. The same reason `metrics.jsonl` carries per-class support:
# a number outlives the conversation that produced it.
CAVEAT = (
    "Consensus measures self-consistency on a scene that did not change. It is a "
    "necessary condition, not a quality metric: a model that is confidently wrong on "
    "every frame scores 1.0. It cannot be read as accuracy, and a rise in it is not "
    "evidence of a better model."
)


@dataclass(frozen=True)
class ClassConsensus:
    """One class's share of the eligible pixels, and how settled they are."""

    name: str
    index: int
    pixels: int
    share: float
    mean_agreement: float
    stable_share: float
    unstable_pixels: int


@dataclass(frozen=True)
class ConsensusResult:
    frames: int
    total_pixels: int
    eligible_pixels: int
    excluded_pixels: int
    agree_threshold: float
    # NaN, not 0.0, when no pixel survived exclusion. A camera filmed through a crowd has
    # nothing left to be consistent about, and 0.0 there reads as "completely unstable"
    # -- the opposite end of the scale from what happened. NaN is the value that refuses
    # to be averaged: a fleet mean over six cameras comes back NaN and names the problem,
    # where a 0.0 would quietly drag the mean down and look like a finding.
    stable_share: float
    unstable_share: float
    measurable: bool
    per_class: list[ClassConsensus]
    unstable_composition: dict[str, float]
    excluded_classes: list[str]
    agreement: np.ndarray = field(repr=False)
    modal: np.ndarray = field(repr=False)
    eligible: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        """The JSON form. Arrays are dropped; the caveat and the denominator are not."""
        return {
            "caveat": CAVEAT,
            "measurable": self.measurable,
            "frames": self.frames,
            "agree_threshold": self.agree_threshold,
            "excluded_classes": self.excluded_classes,
            "pixels": {
                "total": self.total_pixels,
                "eligible": self.eligible_pixels,
                "excluded": self.excluded_pixels,
                "eligible_share": self.eligible_pixels / max(self.total_pixels, 1),
            },
            "stable_share": self.stable_share,
            "unstable_share": self.unstable_share,
            "per_class": [
                {
                    "name": c.name,
                    "index": c.index,
                    "pixels": c.pixels,
                    "share_of_eligible": c.share,
                    "mean_agreement": c.mean_agreement,
                    "stable_share": c.stable_share,
                    "unstable_pixels": c.unstable_pixels,
                }
                for c in self.per_class
            ],
            "unstable_composition": self.unstable_composition,
        }


def fleet_summary(results: Mapping[str, ConsensusResult]) -> dict:
    """Aggregate across cameras, carrying the count of what actually contributed.

    **This exists so that nobody writes it themselves, and the reason is specific.**
    A camera with no eligible pixels reports NaN rather than 0.0, which defeats
    ``np.mean``. It does not defeat ``np.nanmean`` -- and ``np.nanmean`` is this
    codebase's established idiom for exactly this shape: ``ConfusionMatrix.mean_iou``
    uses it, and its docstring argues at length that dropping the NaN entries is the
    only defensible choice there. So the next person aggregating six cameras will follow
    the local precedent, reach for ``nanmean``, and report a mean over five as a mean over
    six. **The camera that drops out is the one with a crowd across the frame** -- the
    busiest one, and the one most worth knowing about.

    The trap is that ``nanmean`` looks *more* careful than ``mean``, not less.

    The fix is the one the evaluator already uses one level up: ``terrain_mIoU`` is
    emitted beside ``terrain_mIoU_classes`` so the two cannot be compared without noticing
    the denominator moved. Here that is ``cameras`` beside ``contributing``, and the names
    of the cameras that did not contribute, so a mean of five labelled six is not
    expressible.
    """
    measurable = {k: v for k, v in results.items() if v.measurable}
    dropped = sorted(set(results) - set(measurable))
    stable = [v.stable_share for v in measurable.values()]
    return {
        "caveat": CAVEAT,
        "cameras": len(results),
        "contributing": len(measurable),
        "not_measurable": dropped,
        # None rather than NaN: a JSON reader that sees null asks why, and NaN is not
        # valid JSON in the first place.
        "mean_stable_share": (sum(stable) / len(stable)) if stable else None,
        "eligible_pixels": sum(v.eligible_pixels for v in measurable.values()),
        "frames": sum(v.frames for v in measurable.values()),
        "per_camera": {
            k: {"stable_share": v.stable_share, "eligible_pixels": v.eligible_pixels}
            for k, v in results.items()
        },
    }


class FrameConsensus:
    """Accumulate per-pixel class votes across frames from one fixed camera.

    One instance per camera. Mixing two cameras into one instance is not a bigger sample
    of anything -- it is two scenes averaged, and the result describes neither.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        exclude: Sequence[str] = ("person",),
    ):
        if not class_names:
            raise ValueError("class_names is empty")
        self.class_names = list(class_names)
        unknown = [c for c in exclude if c not in self.class_names]
        if unknown:
            raise ValueError(
                f"cannot exclude {', '.join(unknown)}: not in this taxonomy "
                f"({', '.join(self.class_names)}). An excluded class that silently does "
                f"nothing would leave moving people in the vote, which is the one thing "
                f"exclusion exists to prevent."
            )
        self.excluded_classes = [c for c in exclude if c in self.class_names]
        self._excluded_ids = {self.class_names.index(c) for c in self.excluded_classes}
        self._votes: np.ndarray | None = None
        self._ever_excluded: np.ndarray | None = None
        self.frames = 0

    def update(self, pred: np.ndarray) -> None:
        """Add one frame's class-id map, ``HxW`` integers."""
        if pred.ndim != 2:
            raise ValueError(f"expected an HxW class map, got shape {pred.shape}")
        n = len(self.class_names)
        lo, hi = int(pred.min()), int(pred.max())
        if lo < 0 or hi >= n:
            raise ValueError(
                f"class ids {lo}..{hi} are outside the {n} declared classes. A prediction "
                f"scored against the wrong taxonomy produces a plausible number, not an "
                f"error, which is why this refuses."
            )
        if self._votes is None:
            self._votes = np.zeros((n, *pred.shape), dtype=np.int32)
            self._ever_excluded = np.zeros(pred.shape, dtype=bool)
        elif pred.shape != self._votes.shape[1:]:
            raise ValueError(
                f"frame {self.frames} is {pred.shape}, the first was "
                f"{self._votes.shape[1:]}. Consensus assumes one fixed camera; differing "
                f"sizes mean these frames are not of the same scene."
            )
        assert self._ever_excluded is not None  # narrowing; set together with _votes

        for c in range(n):
            hit = pred == c
            self._votes[c] += hit
            if c in self._excluded_ids:
                self._ever_excluded |= hit
        self.frames += 1

    def result(self, agree_threshold: float = 0.9) -> ConsensusResult:
        """Settle the vote.

        ``agree_threshold`` is the share of frames a pixel's modal class has to hold to
        count as settled. 0.9 over few frames is a weaker claim than 0.9 over many, and
        nothing here can tell them apart -- which is why ``frames`` is in the output.
        """
        if not 0.0 < agree_threshold <= 1.0:
            raise ValueError(f"agree_threshold must be in (0, 1], got {agree_threshold}")
        if self._votes is None or self.frames == 0:
            raise ValueError("no frames: call update() before result()")
        assert self._ever_excluded is not None

        votes = self._votes
        modal = votes.argmax(axis=0).astype(np.int16)
        agreement = votes.max(axis=0) / self.frames
        eligible = ~self._ever_excluded

        total = int(eligible.size)
        n_eligible = int(eligible.sum())
        settled = eligible & (agreement >= agree_threshold)
        unsettled = eligible & ~settled
        n_unstable = int(unsettled.sum())

        per_class: list[ClassConsensus] = []
        for i, name in enumerate(self.class_names):
            here = eligible & (modal == i)
            n_here = int(here.sum())
            if n_here == 0:
                per_class.append(ClassConsensus(name, i, 0, 0.0, 0.0, 0.0, 0))
                continue
            per_class.append(
                ClassConsensus(
                    name=name,
                    index=i,
                    pixels=n_here,
                    share=n_here / n_eligible,
                    mean_agreement=float(agreement[here].mean()),
                    stable_share=float((agreement[here] >= agree_threshold).mean()),
                    unstable_pixels=int((here & unsettled).sum()),
                )
            )

        # Where the instability sits. This is the finding; the scalar above is the
        # headline that hides it.
        composition = {
            c.name: (c.unstable_pixels / n_unstable if n_unstable else 0.0)
            for c in per_class
            if c.unstable_pixels
        }

        return ConsensusResult(
            frames=self.frames,
            total_pixels=total,
            eligible_pixels=n_eligible,
            excluded_pixels=total - n_eligible,
            agree_threshold=agree_threshold,
            measurable=n_eligible > 0,
            stable_share=(int(settled.sum()) / n_eligible if n_eligible else float("nan")),
            unstable_share=(n_unstable / n_eligible if n_eligible else float("nan")),
            per_class=per_class,
            unstable_composition=dict(sorted(composition.items(), key=lambda kv: -kv[1])),
            excluded_classes=list(self.excluded_classes),
            agreement=agreement,
            modal=modal,
            eligible=eligible,
        )
