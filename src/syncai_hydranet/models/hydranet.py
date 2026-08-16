"""HydraNet: assembling the backbone, neck and task heads.

Design rules:

1. The shared backbone and neck carry over 80% of the compute; heads stay tiny.
2. Heads are mutually independent: each reads only the neck's feature list, never
   another head's output, so one can be added or removed without touching the others.
3. ``forward`` contains only convolution-graph operators, keeping ONNX/TensorRT export
   clean. Target assignment and NMS live outside it.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .backbone import build_backbone
from .heads.detection import SCORE_THR_VIEW, FCOSHead, build_det_head
from .heads.segmentation import build_seg_head
from .losses import FCOSLoss, FixedWeighting, SegLoss, UncertaintyWeighting
from .neck import build_neck


class Detection(NamedTuple):
    """The detection head, its loss and the name they answer to, as one value.

    A view over three ``HydraNet`` attributes rather than their home -- see the note in
    ``__init__`` for why the modules have to stay attributes. Handed out by
    ``HydraNet._detection()``, which is what makes "all three or none" true rather than
    merely intended.
    """

    name: str
    head: FCOSHead
    loss: FCOSLoss


class HydraNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        self.backbone = build_backbone(mcfg["backbone"])
        self.neck = build_neck(mcfg["neck"], self.backbone.out_channels)
        ch = self.neck.out_channels

        self.seg_heads = nn.ModuleDict()
        self.seg_losses = nn.ModuleDict()
        # Three attributes rather than one bundled object, because PyTorch registers
        # submodules by attribute assignment: a head parked inside a tuple or dataclass
        # would never reach `.parameters()`, `.to()` or the state_dict. What binds them
        # is `_detection()` below, which is the only thing allowed to read them.
        self.det_head: FCOSHead | None = None
        self.det_loss: FCOSLoss | None = None
        self.det_head_name: str | None = None
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
        det = self._detection()
        if det is not None:
            cls_o, reg_o, ctr_o = det.head(feats)
            out["det_cls"] = cls_o  # list of [B, C, h, w]
            out["det_reg"] = reg_o
            out["det_ctr"] = ctr_o
        return out

    def _detection(self) -> Detection | None:
        """The detection triple, or None if this model has no detection head.

        The three ``det_*`` attributes are written together in ``__init__`` and are
        otherwise all None, but nothing enforced that: an edit that set two of them and
        forgot the third would surface as ``'NoneType' object is not callable`` partway
        into a training run, naming neither the head nor the config that caused it.
        Reading them through here turns that into a refusal at the disagreement.
        """
        present = [self.det_head_name is not None, self.det_head is not None]
        present.append(self.det_loss is not None)
        if not any(present):
            return None
        if not all(present):
            raise RuntimeError(
                "detection head is half-configured -- "
                f"name={self.det_head_name!r}, head={self.det_head is not None}, "
                f"loss={self.det_loss is not None}. All three are set together in "
                "HydraNet.__init__; something set a subset."
            )
        assert self.det_head_name is not None  # narrowing; `all(present)` already checked
        assert self.det_head is not None
        assert self.det_loss is not None
        return Detection(self.det_head_name, self.det_head, self.det_loss)

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
        det = self._detection()
        if det is not None and det.name in supervised:
            det_loss, sub = det.loss(
                det.head,
                outputs["det_cls"],
                outputs["det_reg"],
                outputs["det_ctr"],
                targets["boxes"],
                targets["labels"],
            )
            losses[det.name] = det_loss
            logs[det.name] = det_loss.detach()
            logs.update({k: v.detach() if torch.is_tensor(v) else v for k, v in sub.items()})
        if not losses:
            # The balancer stacks an empty list of terms, which raises out of torch with
            # nothing in the message about datasets or config keys -- deep in the
            # training loop, one frame removed from the cause.
            raise ValueError(
                f"nothing to supervise: this batch declares supervises={supervised!r} and "
                f"carries targets {sorted(targets)!r}, which name no head this model has "
                f"({', '.join([*self.seg_heads, *([det.name] if det else [])])}). "
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
        det = self._detection()
        if det is not None:
            result[det.name] = det.head.decode(
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
