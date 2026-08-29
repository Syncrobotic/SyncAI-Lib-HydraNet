"""The training and validation engine.

`Trainer` owns a run's lifecycle. The three things beside it are separable ideas with
their own tests -- a weight average, a parameter-group policy and a schedule -- and are
exported here because callers other than `Trainer` legitimately want them.
"""

from .ema import ModelEMA
from .evaluator import ConfusionMatrix, evaluate, select_metric
from .optim import WarmupCosine, build_optimizer
from .trainer import Trainer

__all__ = [
    "ConfusionMatrix",
    "ModelEMA",
    "Trainer",
    "WarmupCosine",
    "build_optimizer",
    "evaluate",
    "select_metric",
]
