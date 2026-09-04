"""One loader for the crop encoder, because three scripts each rolled their own.

Each got a different piece wrong, and none of them raised (measured 2026-09-04):

* `eval_attributes.py` returned the model with **no `.eval()`**. `CropEncoder` carries
  resnet18's BatchNorm plus its own `embed_bn`, so every attribute number that script
  reported was computed while the running statistics were being updated by the batch
  being scored -- train-mode and eval-mode outputs differ on identical input.
* `offline_tracks.py` took the **default** `embed_dim=256` and the default
  `pretrained=True`, downloading ImageNet weights only to overwrite them, and would fail
  outright on a checkpoint whose embedding is a different width.
* Only `eval_attributes` validated the attribute order, so the other two would embed
  against a checkpoint whose channels mean something else and say nothing.

These hold the three properties that differed. A fourth caller inheriting one of those
bugs is what one loader makes impossible.
"""

from __future__ import annotations

import pytest
import torch

from syncai_hydranet.models.crop_encoder import CropEncoder, load_crop_encoder

ATTRS = ["a", "b", "c"]


@pytest.fixture
def ckpt(tmp_path):
    """A checkpoint with a NON-default embedding width, which is the point."""
    model = CropEncoder(len(ATTRS), embed_dim=64, pretrained=False)
    path = tmp_path / "enc.pt"
    torch.save({"model": model.state_dict(), "attributes": ATTRS}, path)
    return path


def test_the_model_comes_back_in_eval_mode(ckpt):
    """The bug that moved numbers: BatchNorm updating from the batch being scored."""
    model, _ = load_crop_encoder(ckpt, "cpu")
    assert not model.training
    assert all(not m.training for m in model.modules())


def test_the_embedding_width_is_read_from_the_checkpoint_not_defaulted(ckpt):
    """`CropEncoder(len(attributes))` takes embed_dim=256; this checkpoint is 64 wide,
    so the version that defaulted would raise here rather than load."""
    model, names = load_crop_encoder(ckpt, "cpu")
    assert names == ATTRS
    with torch.no_grad():
        embedding = model.embed_only(torch.zeros(1, 3, 256, 128))
    assert embedding.shape[1] == 64


def test_a_mismatched_attribute_order_is_refused(ckpt):
    """Silence here means comparing every channel to the wrong label."""
    with pytest.raises(ValueError, match="different attribute order"):
        load_crop_encoder(ckpt, "cpu", expect=["c", "b", "a"])


def test_the_order_is_only_checked_when_the_caller_states_one(ckpt):
    """A caller using the embedding alone does not care about attribute names, and
    should not have to pass a list to say so."""
    model, names = load_crop_encoder(ckpt, "cpu")
    assert names == ATTRS
    assert model is not None
