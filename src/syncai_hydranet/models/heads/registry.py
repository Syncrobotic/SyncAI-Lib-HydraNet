"""One interface over heads that are not otherwise alike.

`HydraNet` is a multi-head network, and until this file only the *segmentation* heads
were actually plural: they live in a `ModuleDict` and every path iterates them, while
detection was a single optional attribute unrolled by hand in `forward`,
`compute_losses`, `predict`, the evaluator and the ONNX exporter. Counted at the time:
45 mentions of `det_*` against 12 of `seg_heads`, for two things that are peers.

The cost of that is not ugliness, it is arithmetic. A third head family -- depth,
keypoints, whatever comes -- costs another full set of special cases in four files, and
nothing makes the four agree.

**These adapters are deliberately not `nn.Module`s.** PyTorch registers submodules by
attribute assignment, so a head parked inside a module wrapper would appear in the
state_dict one level deeper -- `heads.detection.module.cls_pred.weight` where every
checkpoint written so far says `det_head.cls_pred.weight`. Every existing checkpoint
would stop loading, silently for anything using `strict=False`. So the modules stay
exactly where they are and these are a *view* over them, rebuilt on each call. The
registry is free; the weights never move.

What the two implementations genuinely do not share is why an interface was needed rather
than a base class: a segmentation head returns one dense map and takes `(pred, target)`,
while FCOS returns three pyramid lists and its loss needs the head itself back to
assign targets. The adapters absorb that difference at the boundary instead of spreading
it through every caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..losses import FCOSLoss
from .detection import FCOSHead


class Head(Protocol):
    """What every head can do, from the network's point of view."""

    # A read-only property rather than `name: str`. A plain attribute in a Protocol
    # demands a settable one, and both implementations are frozen dataclasses, so the
    # bare annotation would leave neither of them a Head.
    @property
    def name(self) -> str: ...

    # Positional-only, so an implementation may name an argument it ignores `_size`
    # without falling out of the protocol -- structural matching compares parameter
    # names, and a rename to silence an unused-argument lint would otherwise quietly
    # stop the class being a Head at all.
    def forward_into(self, out: dict, feats: list[torch.Tensor], size, /) -> None:
        """Write this head's raw outputs into the shared dict, under its own keys."""

    def loss(self, out: dict, targets: dict, /) -> tuple[torch.Tensor, dict]:
        """Returns `(loss, extra_logs)`. Only called when this head is supervised."""

    def supervised_by(self, targets: dict, /) -> bool:
        """Whether `targets` carries what this head needs, beyond being named."""

    def decode_into(self, result: dict, out: dict, /, **kw) -> None:
        """Turn raw outputs into the inference-facing answer."""


@dataclass(frozen=True)
class SegmentationHead:
    """A dense per-pixel head: one logits map in, one class map out."""

    name: str
    # Deliberately `Module`: this adapter calls nothing a SemanticFPNHead adds, and
    # `seg_heads` is a ModuleDict, whose subscript is a Module by construction.
    module: torch.nn.Module
    loss_fn: torch.nn.Module

    def forward_into(self, out: dict, feats: list[torch.Tensor], size) -> None:
        out[self.name] = self.module(feats, size)  # [B, C, H, W] logits

    def loss(self, out: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        return self.loss_fn(out[self.name], targets[self.name]), {}

    def supervised_by(self, targets: dict) -> bool:
        return self.name in targets

    def decode_into(self, result: dict, out: dict, **_kw) -> None:
        result[self.name] = out[self.name].argmax(dim=1)  # [B, H, W]


@dataclass(frozen=True)
class DepthHead:
    """A dense regression head: one metric-depth map in, the same map out.

    Structurally the segmentation head's twin -- same trunk, same `(pred, target)` loss
    signature -- and deliberately a separate adapter anyway. What differs is `decode_into`:
    a segmentation head answers with `argmax`, which for a one-channel output is the
    constant 0. Sharing the adapter would produce a head that trains correctly, exports
    correctly, and returns zeros at inference, which is the quietest possible failure.
    """

    name: str
    module: torch.nn.Module
    loss_fn: torch.nn.Module

    def forward_into(self, out: dict, feats: list[torch.Tensor], size) -> None:
        out[self.name] = self.module(feats, size)  # [B, 1, H, W] metres

    def loss(self, out: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        return self.loss_fn(out[self.name], targets[self.name]), {}

    def supervised_by(self, targets: dict) -> bool:
        return self.name in targets

    def decode_into(self, result: dict, out: dict, **_kw) -> None:
        result[self.name] = out[self.name].squeeze(1)  # [B, H, W] metres


@dataclass(frozen=True)
class DetectionHead:
    """FCOS: three pyramid outputs, and a loss that needs the head back.

    The `det_cls`/`det_reg`/`det_ctr` keys are flat rather than nested under `name`
    because the ONNX graph names its outputs from them and every consumer of an exported
    engine -- including the hand-written Jetson scripts -- reads those names.
    """

    name: str
    module: FCOSHead
    loss_fn: FCOSLoss

    def forward_into(self, out: dict, feats: list[torch.Tensor], _size) -> None:
        # `_size`: FCOS predicts at the pyramid's own resolutions and its strides say how
        # those map back, so unlike a segmentation head it never upsamples to the input.
        cls_o, reg_o, ctr_o = self.module(feats)
        out["det_cls"] = cls_o  # each a list of [B, C, h, w], one per FPN level
        out["det_reg"] = reg_o
        out["det_ctr"] = ctr_o

    def loss(self, out: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        # The head goes back in: FCOS assigns targets to points using the strides and
        # regression ranges the head owns, so the loss cannot compute them alone.
        return self.loss_fn(
            self.module,
            out["det_cls"],
            out["det_reg"],
            out["det_ctr"],
            targets["boxes"],
            targets["labels"],
            # Absent unless the dataset was built with a `det_vocab`, and absent means
            # "supervise every channel" -- which is right for a single-source run and
            # wrong the moment two sources share this head. `.get` rather than `[]`
            # deliberately: every checkpoint trained before the vocabulary existed loads
            # and trains unchanged.
            class_mask=targets.get("det_class_mask"),
        )

    def supervised_by(self, _targets: dict) -> bool:
        """True: naming detection in `supervises` is taken at its word.

        Deliberately *not* symmetric with the segmentation head, which also checks its
        targets are present, and deliberately not "improved" to match. A dataset that
        declares `supervises: [terrain, detection]` while its collate produces no boxes
        currently raises KeyError on the first step. Adding the check here would instead
        train terrain only and leave detection with no gradient -- silently, because
        `losses` would be non-empty and the "nothing to supervise" refusal would not
        fire. A loud crash beats half a model.
        """
        return True

    def decode_into(self, result: dict, out: dict, **kw) -> None:
        result[self.name] = self.module.decode(
            out["det_cls"], out["det_reg"], out["det_ctr"], **kw
        )


@dataclass(frozen=True)
class PoseHead:
    """Keypoint heatmaps at P3; decoding borrows the detection head's boxes.

    `det_name` is the config name of the detection head, because the decoded boxes land
    in `result` under that name and grouping-by-box is this head's whole design
    (heads/pose.py). Ordered after detection in `HydraNet.heads()` for the same reason
    -- decode order is data flow here, not cosmetics. With no detection head (or no
    boxes in a frame) it decodes to empty keypoint sets rather than guessing.
    """

    name: str
    module: torch.nn.Module
    loss_fn: torch.nn.Module
    det_name: str | None

    def forward_into(self, out: dict, feats: list[torch.Tensor], _size) -> None:
        out[self.name] = self.module(feats)  # [B, K, H/8, W/8] logits

    def loss(self, out: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        return self.loss_fn(out[self.name], targets[self.name]), {}

    def supervised_by(self, targets: dict) -> bool:
        return self.name in targets

    def decode_into(self, result: dict, out: dict, **_kw) -> None:
        from .pose import decode_boxes

        maps = out[self.name]
        dets = result.get(self.det_name) if self.det_name else None
        decoded = []
        for i in range(maps.shape[0]):
            if dets is not None and len(dets[i].get("boxes", ())):
                decoded.append(decode_boxes(maps[i], dets[i]["boxes"]))
            else:
                decoded.append(torch.zeros((0, maps.shape[1], 3), device=maps.device))
        result[self.name] = decoded
