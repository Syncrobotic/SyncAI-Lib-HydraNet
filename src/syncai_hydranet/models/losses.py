"""Losses: segmentation (CE + Dice), detection (Focal + GIoU + centerness BCE),
and multi-task balancing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..labels import IGNORE

# ---------------------------- segmentation ----------------------------


class SegLoss(nn.Module):
    """Cross-entropy plus Dice.

    Outdoor and indoor scenes are both heavily imbalanced: sky and grass, or floor and
    wall, cover most pixels while the classes that matter most for safety (stairs,
    water, glass) may be under 1%. Plain CE is dominated by the large classes; Dice
    optimises region overlap directly and is insensitive to class size.

    **Dice alone did not settle it, which is why `class_weights` exists.** Measured on
    the batch02 retail-objects run, as agreement with SAM 3 over six held-out cameras:
    `fixture` reached 92.6% recall at 56.1% precision while every other class sat at
    91-96% precision and 21-69% recall. One class absorbing and the rest under-firing is
    the signature of frequency bias, and it survived `dice_weight` 1.5 against
    `ce_weight` 1.0 -- Dice is `dice.mean()`, so it is already class-balanced, and the
    unweighted CE beside it is what still pulls toward the common classes.

    Off by default. Passing weights changes what a checkpoint means, so it is opt-in per
    config rather than a default that silently reinterprets every previous run.
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = IGNORE,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        # A buffer so it follows `.to(device)` -- CE raises on a device mismatch and it
        # would only fire once training reached the GPU. **`persistent=False`** so it
        # stays out of `state_dict()`: these are a config value, not learned state, and
        # a persistent buffer makes every checkpoint that predates this unloadable into a
        # weighted model -- `Missing key(s): seg_losses.terrain.class_weights`, which is
        # exactly the case of turning weights on and fine-tuning from an existing run.
        # The `config.yaml` written beside every checkpoint already records that a run
        # trained under weights, so nothing is lost by keeping them out.
        if class_weights is None:
            self.class_weights = None
        else:
            if len(class_weights) != num_classes:
                raise ValueError(
                    f"class_weights has {len(class_weights)} entries for {num_classes} "
                    f"classes; a silent broadcast here would reweight the wrong classes"
                )
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float),
                persistent=False,
            )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce_weight * F.cross_entropy(
            logits, target, weight=self.class_weights, ignore_index=self.ignore_index
        )
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self._dice(logits, target)
        return loss

    def _dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        t = target.clone()
        t[~valid] = 0
        onehot = F.one_hot(t, self.num_classes).permute(0, 3, 1, 2).float()
        prob = logits.softmax(dim=1)
        mask = valid[:, None].float()
        prob, onehot = prob * mask, onehot * mask
        inter = (prob * onehot).sum(dim=(0, 2, 3))
        denom = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        dice = (2 * inter + 1.0) / (denom + 1.0)
        return 1.0 - dice.mean()


# ---------------------------- detection ----------------------------


def sigmoid_focal_loss(logits, targets_onehot, alpha=0.25, gamma=2.0, channel_mask=None):
    """``channel_mask`` broadcasts over the class axis and zeroes whole channels.

    Not a class weight. A weight of zero and a mask of zero are the same arithmetic and
    a different claim: a weight says this class matters less, a mask says *this dataset
    cannot answer for this class*, so its images are neither evidence for it nor against
    it. `label_maps_retail_security` has the measurement that makes the distinction worth
    the parameter -- an unlabelled shopper in a site frame, taken as a negative, is what
    held `product` at IoU 0.000 for 22 epochs in the segmentation case.
    """
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none")
    p_t = p * targets_onehot + (1 - p) * (1 - targets_onehot)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = loss * (alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot))
    if channel_mask is not None:
        loss = loss * channel_mask
    return loss.sum()


def giou_loss(pred_ltrb: torch.Tensor, target_ltrb: torch.Tensor) -> torch.Tensor:
    """GIoU on (l, t, r, b) distance form; valid because both share an anchor point."""
    pl, pt, pr, pb = pred_ltrb.unbind(-1)
    tl, tt, tr, tb = target_ltrb.unbind(-1)
    p_area = (pl + pr) * (pt + pb)
    t_area = (tl + tr) * (tt + tb)
    iw = torch.min(pl, tl) + torch.min(pr, tr)
    ih = torch.min(pt, tt) + torch.min(pb, tb)
    inter = iw.clamp(min=0) * ih.clamp(min=0)
    union = p_area + t_area - inter
    iou = inter / union.clamp(min=1e-6)
    cw = torch.max(pl, tl) + torch.max(pr, tr)
    ch = torch.max(pt, tt) + torch.max(pb, tb)
    c_area = (cw * ch).clamp(min=1e-6)
    giou = iou - (c_area - union) / c_area
    return (1.0 - giou).sum()


class FCOSLoss(nn.Module):
    def __init__(self, num_classes: int, cls_weight=1.0, reg_weight=1.0, centerness_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.w = (cls_weight, reg_weight, centerness_weight)

    def forward(
        self, head, cls_out, reg_out, ctr_out, boxes_list, labels_list, class_mask=None
    ):
        """``class_mask`` is [B, C] or [C]: which channels this batch's dataset can label.

        None means "all of them", which is the single-source case and every run before
        the retail+security vocabulary existed.
        """
        device = cls_out[0].device
        shapes = [c.shape[-2:] for c in cls_out]
        _, cls_t, reg_t, ctr_t = head.get_targets(shapes, boxes_list, labels_list, device)
        from .heads.detection import flatten_levels

        flat_cls, flat_reg, flat_ctr = flatten_levels(
            cls_out, reg_out, ctr_out, self.num_classes
        )

        pos = cls_t < self.num_classes
        num_pos = pos.sum().clamp(min=1).float()
        onehot = torch.zeros_like(flat_cls)
        # Match flat_cls's dtype rather than forcing float32: under autocast the
        # logits are bf16/fp16, and index_put_ refuses a mismatched source. This
        # only fires when the detection head is actually supervised on CUDA, which
        # is why it survived every seg-only run.
        onehot[pos] = F.one_hot(cls_t[pos], self.num_classes).to(onehot.dtype)
        if class_mask is not None:
            # [B, C] -> [B, 1, C] against flat_cls's [B, points, C]; a [C] mask
            # broadcasts as it is. Cast rather than assume: under autocast flat_cls is
            # bf16/fp16 and the mask arrives from the collate as float32.
            mask = class_mask.to(flat_cls.dtype)
            class_mask = mask[:, None, :] if mask.dim() == 2 else mask
        cls_loss = sigmoid_focal_loss(flat_cls, onehot, channel_mask=class_mask) / num_pos
        if pos.any():
            reg_loss = giou_loss(flat_reg[pos], reg_t[pos]) / num_pos
            ctr_loss = (
                F.binary_cross_entropy_with_logits(flat_ctr[pos], ctr_t[pos], reduction="sum")
                / num_pos
            )
        else:
            reg_loss = flat_reg.sum() * 0.0
            ctr_loss = flat_ctr.sum() * 0.0
        return (
            self.w[0] * cls_loss + self.w[1] * reg_loss + self.w[2] * ctr_loss,
            {
                "det_cls": cls_loss.detach(),
                "det_reg": reg_loss.detach(),
                "det_ctr": ctr_loss.detach(),
            },
        )


# ---------------------------- multi-task balancing ----------------------------


class UncertaintyWeighting(nn.Module):
    """Kendall et al. 2018: ``L = sum_i exp(-s_i) * L_i + 0.5 * s_i``.

    ``s_i = log(sigma_i^2)`` is learned per task, so a noisy task down-weights itself
    and the ``0.5 * s`` term stops the model from cheating by inflating sigma. Tasks
    without supervision in the current step are simply absent from ``losses`` and so
    their ``s`` is untouched.
    """

    def __init__(self, task_names: list[str]):
        super().__init__()
        self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in task_names})

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        # Collect then stack, rather than seeding an accumulator with None: that loop
        # returned None untouched for an empty `losses`, i.e. it promised a Tensor and
        # handed back None for the caller's `.backward()` to trip over one frame later.
        # An empty step is a caller bug either way; this way it says so where it happens.
        terms = []
        for name, loss in losses.items():
            s = self.log_vars[name]
            terms.append(torch.exp(-s) * loss + 0.5 * s)
        return torch.stack(terms).sum()


class FixedWeighting(nn.Module):
    def __init__(self, weights: dict[str, float]):
        super().__init__()
        self.weights = dict(weights)

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        # Same shape as UncertaintyWeighting.forward above, for the same reason.
        terms = [self.weights.get(name, 1.0) * loss for name, loss in losses.items()]
        return torch.stack(terms).sum()
