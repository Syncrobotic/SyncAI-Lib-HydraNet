"""HydraNet: assembling the backbone, neck and task heads.

Design rules:

1. The shared backbone and neck carry over 80% of the compute; heads stay tiny.
2. Heads are mutually independent: each reads only the neck's feature list, never
   another head's output, so one can be added or removed without touching the others.
3. ``forward`` contains only convolution-graph operators, keeping ONNX/TensorRT export
   clean. Target assignment and NMS live outside it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone
from .heads.detection import SCORE_THR_VIEW, build_det_head
from .heads.segmentation import build_seg_head
from .losses import FCOSLoss, FixedWeighting, SegLoss, UncertaintyWeighting
from .neck import build_neck


class HydraNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        self.backbone = build_backbone(mcfg["backbone"])
        self.neck = build_neck(mcfg["neck"], self.backbone.out_channels)
        ch = self.neck.out_channels

        self.seg_heads = nn.ModuleDict()
        self.seg_losses = nn.ModuleDict()
        self.det_head = None
        self.det_loss = None
        self.det_head_name = None
        head_names = []
        for name, hcfg in mcfg["heads"].items():
            head_names.append(name)
            if hcfg["type"] == "semantic_fpn":
                self.seg_heads[name] = build_seg_head(hcfg, ch)
                lcfg = hcfg.get("loss", {})
                self.seg_losses[name] = SegLoss(
                    hcfg["num_classes"],
                    ce_weight=lcfg.get("ce_weight", 1.0),
                    dice_weight=lcfg.get("dice_weight", 1.0),
                    ignore_index=lcfg.get("ignore_index", 255),
                )
            elif hcfg["type"] == "fcos":
                self.det_head_name = name
                self.det_head = build_det_head(hcfg, ch)
                lcfg = hcfg.get("loss", {})
                self.det_loss = FCOSLoss(
                    hcfg["num_classes"],
                    cls_weight=lcfg.get("cls_weight", 1.0),
                    reg_weight=lcfg.get("reg_weight", 1.0),
                    centerness_weight=lcfg.get("centerness_weight", 1.0),
                )
            else:
                raise ValueError(f"unknown head type: {hcfg['type']}")

        if mcfg.get("loss_balancing", "uncertainty") == "uncertainty":
            self.balancer = UncertaintyWeighting(head_names)
        else:
            self.balancer = FixedWeighting(mcfg.get("fixed_weights", {}))

    def forward(self, images: torch.Tensor) -> dict:
        """Pure convolution graph. This is exactly what gets exported to ONNX."""
        feats = self.neck(self.backbone(images))
        out = {}
        h, w = images.shape[-2:]
        for name, head in self.seg_heads.items():
            out[name] = head(feats, (h, w))  # [B, C, H, W] logits
        if self.det_head is not None:
            cls_o, reg_o, ctr_o = self.det_head(feats)
            out["det_cls"] = cls_o  # list of [B, C, h, w]
            out["det_reg"] = reg_o
            out["det_ctr"] = ctr_o
        return out

    def compute_losses(
        self, outputs: dict, targets: dict, supervised: list[str]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``supervised`` names the heads this batch has labels for.

        Partial supervision across datasets: a RUGD batch only contributes segmentation
        losses, a COCO batch only detection. Unsupervised heads get zero gradient while
        the shared trunk learns from both.

        The returned logs are **detached tensors, not floats**. Converting them here cost
        a CUDA synchronisation per head per step -- three or four every step, on every
        step, whether or not that step was going to be logged. The caller converts them
        at its logging interval instead; see ``Trainer.train_one_epoch``.
        """
        losses: dict[str, torch.Tensor] = {}
        logs: dict[str, torch.Tensor] = {}
        for name in self.seg_heads:
            if name in supervised and name in targets:
                seg_loss = self.seg_losses[name](outputs[name], targets[name])
                losses[name] = seg_loss
                logs[name] = seg_loss.detach()
        if self.det_head is not None and self.det_head_name in supervised:
            det_loss, sub = self.det_loss(
                self.det_head,
                outputs["det_cls"],
                outputs["det_reg"],
                outputs["det_ctr"],
                targets["boxes"],
                targets["labels"],
            )
            losses[self.det_head_name] = det_loss
            logs[self.det_head_name] = det_loss.detach()
            logs.update({k: v.detach() if torch.is_tensor(v) else v for k, v in sub.items()})
        if not losses:
            # The balancer sums an empty dict to None, and the next line would fail as
            # `'NoneType' object has no attribute 'detach'` -- deep in the training loop,
            # naming neither the dataset nor the config key that caused it.
            raise ValueError(
                f"nothing to supervise: this batch declares supervises={supervised!r} and "
                f"carries targets {sorted(targets)!r}, which name no head this model has "
                f"({', '.join([*self.seg_heads, *filter(None, [self.det_head_name])])}). "
                "A dataset that supervises no head costs optimiser steps and contributes "
                "no gradient; fix its `supervises` list."
            )
        total = self.balancer(losses)
        logs["total"] = total.detach()
        return total, logs

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, score_thr: float = SCORE_THR_VIEW, nms_thr: float = 0.6
    ) -> dict:
        """Single-frame inference: per-pixel class maps plus NMS-filtered boxes."""
        out = self.forward(images)
        result = {}
        for name in self.seg_heads:
            result[name] = out[name].argmax(dim=1)  # [B, H, W]
        if self.det_head is not None:
            result[self.det_head_name] = self.det_head.decode(
                out["det_cls"],
                out["det_reg"],
                out["det_ctr"],
                score_thr=score_thr,
                nms_thr=nms_thr,
                img_size=images.shape[-2:],
            )
        return result


def build_model(cfg) -> HydraNet:
    return HydraNet(cfg)
