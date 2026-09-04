"""The second stage's crop encoder: one model, association first, attributes second.

`RETAIL_DATA.md` sets the ordering and gives the reason, which is not a preference
about what is interesting to build:

    A 4.6-minute clip fragments into 1234 tracks. Attributes computed per track on that
    fragmentation produce an age/gender distribution over 1234 "people" who are perhaps
    thirty actual shoppers... The failure is worse than noisy, because it is *biased*: a
    shopper who lingers fragments into more tracks than one who walks through, so
    dwell-weighted traits are over-counted by exactly the amount dwell analytics exist to
    measure.

Measured on this project's own footage today: three `loitering` events on Taichung-cam01
were one shopper at a counter for 110 seconds, cut into three tracks by gaps of a single
frame at 6 and 9 cm. Three alarms, one person.

So this is **one encoder with two outputs**: an embedding that re-links fragments, and
attribute logits. The embedding is not a by-product -- it is the output the ordering says
matters first, and it is scoreable today against the floor `analytics/reid_metrics.py`
already put on the board: an ImageNet resnet18 with no re-identification training scores
mAP 0.0318 / rank-1 0.0962 on Market-1501's standard protocol. Anything built here that
does not clear that number has not earned its place.

**Not a HydraNet head, and it cannot be one.** `hydranet.py`'s design rule 2 says heads
read only the neck's feature list; this reads a *crop cut from the detection head's boxes*,
which the `Head` protocol has no way to express. It gets its own small ONNX export, which
is the boundary `stage.py` and ARCHITECTURE.md section 2 describe.

**Why resnet18 and not the project's regnet trunk.** The floor above was measured on
resnet18, so this is comparable to it by construction. A different backbone would make the
first number this model produces incomparable to the only number it has to beat.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CropEncoder(nn.Module):
    """Person crop -> (L2-normalised embedding, attribute logits).

    The embedding is taken *before* the attribute head rather than after, so association
    does not inherit whatever the attribute loss decides to discard. Two shoppers in the
    same clothes are the case attributes are designed to call identical and association is
    designed to keep apart, and reading the embedding off the attribute logits would make
    that impossible by construction.
    """

    def __init__(self, num_attributes: int, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.stem = nn.Sequential(*list(net.children())[:-1])  # through global pool
        self.embed = nn.Linear(512, embed_dim)
        self.embed_bn = nn.BatchNorm1d(embed_dim)
        self.attr = nn.Linear(512, num_attributes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.stem(x).flatten(1)
        e = F.normalize(self.embed_bn(self.embed(f)), dim=1)
        return e, self.attr(f)

    @torch.no_grad()
    def embed_only(self, x: torch.Tensor) -> torch.Tensor:
        """What `reid_metrics` consumes. Separate so association never pays for the head."""
        return self.forward(x)[0]


def load_crop_encoder(path, device, *, expect: list[str] | tuple[str, ...] | None = None):
    """A `CropEncoder` from a checkpoint, in eval mode, with its own attribute order.

    **Three scripts each rolled this and each got a different piece of it wrong**, which
    is why it is here rather than in any of them (measured 2026-09-04):

    * `scripts/eval_attributes.py` returned `model.to(device)` with **no `.eval()`**.
      `CropEncoder` carries resnet18's BatchNorm plus its own `embed_bn`, so every
      attribute number that script reported was computed while the running statistics
      were still being updated by the batch being scored. Verified: train-mode and
      eval-mode outputs differ on the same input.
    * `scripts/offline_tracks.py` built `CropEncoder(len(ck["attributes"]))`, taking the
      **default** `embed_dim=256` and the **default** `pretrained=True` -- so it
      downloaded ImageNet weights only to overwrite them, and would fail outright on a
      checkpoint whose embedding is not 256 wide.
    * Only `eval_attributes` checked the attribute order at all, so the other two would
      embed happily against a checkpoint whose channels mean something else.

    `expect` is the caller's attribute order. Pass it whenever the names matter --
    scoring, or anything reading a named channel -- and leave it None when only the
    embedding is used.
    """
    from ..utils.checkpoint import load_checkpoint

    ckpt = load_checkpoint(str(path))
    names = list(ckpt["attributes"])
    if expect is not None and names != list(expect):
        raise ValueError(
            f"{path} was trained on a different attribute order than the caller expects; "
            "scoring it against these columns would compare every channel to the wrong "
            f"label. checkpoint={names} caller={list(expect)}"
        )
    model = CropEncoder(
        len(names),
        embed_dim=ckpt["model"]["embed.weight"].shape[0],
        pretrained=False,  # the state dict below replaces every weight
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), names


class AttributeLoss(nn.Module):
    """Multi-label BCE, weighted per attribute by how rare it is.

    ``pos_weight`` is the standard fix and it is worth stating what it does and does not
    buy here. `AgeOver60` is 1.41% of the training split, so an unweighted head reaches a
    low loss by answering "no" to everyone and the metric that notices is per-attribute
    recall, not the mean. Weighting moves the operating point; it does not create evidence.
    A 1,127-crop attribute stays a 1,127-crop attribute, which is why
    `data/attributes.SUPPORT` travels with every number this produces.

    The weight is clamped. `LowerStripe` at 0.51% would otherwise carry 195x the gradient
    of a balanced attribute and dominate a 26-way head on the strength of 404 crops.
    """

    # Annotated at class level as well as registered: `nn.Module.__getattr__` is typed
    # `Tensor | Module`, so without this every read of the buffer is ambiguous to the
    # checker even though `register_buffer` is what puts it there.
    pos_weight: torch.Tensor

    def __init__(self, pos_rate: torch.Tensor, max_weight: float = 20.0):
        super().__init__()
        w = (1.0 - pos_rate) / pos_rate.clamp(min=1e-6)
        self.register_buffer("pos_weight", w.clamp(max=max_weight))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
