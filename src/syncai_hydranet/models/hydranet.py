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

from ..labels import IGNORE
from .backbone import build_backbone
from .heads.depth import SILogLoss, build_depth_head
from .heads.detection import SCORE_THR_VIEW, FCOSHead, build_det_head
from .heads.pose import build_pose_head, build_pose_loss
from .heads.registry import DepthHead, DetectionHead, Head, PoseHead, SegmentationHead
from .heads.segmentation import build_seg_head
from .heads.text_classifier import TextEmbeddingClassifier, load_matrix_file
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
        # Depth gets its own pair rather than joining `seg_heads`: both are dense maps,
        # but a checkpoint's state_dict keys are the contract with every exporter and
        # deployment script in this repo, and `seg_heads.depth.*` would claim depth is a
        # segmentation head to anything that inspects the keys -- including the RKNN and
        # TensorRT conversions, which decide output handling from exactly that.
        self.depth_heads = nn.ModuleDict()
        self.depth_losses = nn.ModuleDict()
        # Pose follows depth's precedent: its own pair, so `pose_heads.pose.*` names in
        # the state_dict say what the module is.
        self.pose_heads = nn.ModuleDict()
        self.pose_losses = nn.ModuleDict()
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
                    ignore_index=lcfg.get("ignore_index", IGNORE),
                    class_weights=lcfg.get("class_weights"),
                )
            elif hcfg["type"] == "fcos":
                self.det_head_name = name
                self.det_head = build_det_head(hcfg, ch)
                _install_text_embeddings(self.det_head, hcfg, name)
                lcfg = hcfg.get("loss", {})
                self.det_loss = FCOSLoss(
                    hcfg["num_classes"],
                    cls_weight=lcfg.get("cls_weight", 1.0),
                    reg_weight=lcfg.get("reg_weight", 1.0),
                    centerness_weight=lcfg.get("centerness_weight", 1.0),
                )
            elif hcfg["type"] == "depth_fpn":
                self.depth_heads[name] = build_depth_head(hcfg, ch)
                lcfg = hcfg.get("loss", {})
                self.depth_losses[name] = SILogLoss(
                    lam=lcfg.get("lam", 0.85),
                    alpha=lcfg.get("alpha", 10.0),
                    # Read from the head, not the loss: a ceiling that disagreed between
                    # the two would train the model to predict depths the loss then
                    # masks away as invalid, and nothing would report it.
                    max_depth=hcfg.get("max_depth", 10.0),
                )
            elif hcfg["type"] == "pose_p3":
                self.pose_heads[name] = build_pose_head(hcfg, ch)
                self.pose_losses[name] = build_pose_loss(hcfg)
            else:
                raise ValueError(f"unknown head type: {hcfg['type']}")

        if mcfg.get("loss_balancing", "uncertainty") == "uncertainty":
            self.balancer = UncertaintyWeighting(head_names)
        else:
            self.balancer = FixedWeighting(mcfg.get("fixed_weights", {}))

    def forward(self, images: torch.Tensor) -> dict:
        """Pure convolution graph. This is exactly what gets exported to ONNX."""
        feats = self.neck(self.backbone(images))
        out: dict = {}
        h, w = images.shape[-2:]
        for head in self.heads():
            head.forward_into(out, feats, (h, w))
        return out

    def heads(self) -> list[Head]:
        """Every head this model has, in a form the rest of the network can treat alike.

        Rebuilt on each call and deliberately cheap: the adapters hold references to the
        registered modules rather than owning them, which is what keeps `det_head.*` and
        `seg_heads.*` as the state_dict keys every checkpoint written so far uses.

        Segmentation first, then detection. The order reaches the loss balancer, which
        stacks the terms it is given, so it is part of the arithmetic and not cosmetic.
        """
        heads: list[Head] = [
            SegmentationHead(name, self.seg_heads[name], self.seg_losses[name])
            for name in self.seg_heads
        ]
        heads += [
            DepthHead(name, self.depth_heads[name], self.depth_losses[name])
            for name in self.depth_heads
        ]
        det = self._detection()
        if det is not None:
            heads.append(DetectionHead(det.name, det.head, det.loss))
        # after detection: pose decodes inside the boxes detection just produced
        heads += [
            PoseHead(name, self.pose_heads[name], self.pose_losses[name], self.det_head_name)
            for name in self.pose_heads
        ]
        return heads

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
        heads = self.heads()
        for head in heads:
            if head.name not in supervised or not head.supervised_by(targets):
                continue
            loss, extra = head.loss(outputs, targets)
            losses[head.name] = loss
            logs[head.name] = loss.detach()
            logs.update({k: v.detach() if torch.is_tensor(v) else v for k, v in extra.items()})
        if not losses:
            # The balancer stacks an empty list of terms, which raises out of torch with
            # nothing in the message about datasets or config keys -- deep in the
            # training loop, one frame removed from the cause.
            raise ValueError(
                f"nothing to supervise: this batch declares supervises={supervised!r} and "
                f"carries targets {sorted(targets)!r}, which name no head this model has "
                f"({', '.join(h.name for h in heads)}). "
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
        result: dict = {}
        for head in self.heads():
            head.decode_into(
                result,
                out,
                score_thr=score_thr,
                nms_thr=nms_thr,
                img_size=images.shape[-2:],
            )
        return result


def _install_text_embeddings(head: FCOSHead, hcfg: dict, name: str) -> None:
    """Put a real class-name matrix into the head, at construction rather than at export.

    Ordering, because it is the whole point and nothing else enforces it: the head's
    `text_embeddings` buffer is a random *orthogonal placeholder*, and `embed_pred` learns
    to align with whatever occupies that buffer while the gradient flows. Installing here
    means training sees the real text space from step 0; installing at export means the
    visual projection has spent its whole life aiming at random directions, and the result
    is confident, meaningless boxes with no error anywhere.

    A `text_embeddings` path under a `linear` head is refused rather than ignored. Ignoring
    it produces exactly the run the operator thought they were avoiding, and the config
    would still read as if the vocabulary were data.

    The refusal tests the **built classifier** rather than the config string that asked for
    it. Those can only disagree if `build_det_head` and this function read `cls_head`
    differently, which is exactly the kind of drift worth catching at the object rather
    than trusting twice -- and it is also what lets a type checker see that the call below
    is valid, since `cls_pred` is a `Conv2d` on the other branch.
    """
    path = hcfg.get("text_embeddings")
    if path is None:
        return
    classifier = head.cls_pred
    if not isinstance(classifier, TextEmbeddingClassifier):
        raise ValueError(
            f"model.heads.{name}: `text_embeddings` is set but the head built a "
            f"{type(classifier).__name__} (cls_head={hcfg.get('cls_head', 'linear')!r}). A "
            f"linear classifier has no matrix to install, so this file would be silently "
            f"unused and the run would train the frozen-vocabulary head the setting exists "
            f"to avoid."
        )
    matrix = load_matrix_file(path, head.num_classes, hcfg.get("classes"))
    classifier.load_text_embeddings(matrix)


def build_model(cfg) -> HydraNet:
    return HydraNet(cfg)
