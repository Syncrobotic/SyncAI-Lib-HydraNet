"""The open-vocabulary classifier: what makes it worth the swap, and what it cannot do.

The problem is measured. `docs/RETAIL.md`'s audit swept the detection head over
Kaohsiung-cam08, an Apple store, and got **1,683 `book` and no `laptop` at any threshold**.
The head is finding the merchandise and naming it with the nearest shape word COCO owns.
The boxes are right; the vocabulary is not, and no amount of COCO training fixes a word
COCO does not have.

So the tests here are about the properties that make the vocabulary swappable rather than
about the arithmetic of a cosine similarity:

* the class list is **data**, so installing a new one does not touch a weight;
* a wrong-shaped matrix is refused, because silently renaming every class is the failure
  `coco_subsets.head_order` exists to prevent one level up;
* narrowing is a row slice, so the visual projection is untouched;
* and the head has to be a drop-in for the linear one, or nothing downstream survives it.

pytest tests/test_text_classifier.py -v
"""

import math

import pytest
import torch

from syncai_hydranet.models.heads.detection import FCOSHead
from syncai_hydranet.models.heads.text_classifier import (
    DEFAULT_LOGIT_SCALE,
    TextEmbeddingClassifier,
)

CH, DIM, K = 32, 64, 12


def _clf(**kw):
    torch.manual_seed(0)
    return TextEmbeddingClassifier(CH, DIM, K, **kw)


# --- drop-in, or nothing downstream survives it ---------------------------


def test_it_produces_the_same_shape_as_the_linear_head_it_replaces():
    x = torch.randn(2, CH, 9, 11)
    assert _clf()(x).shape == torch.nn.Conv2d(CH, K, 3, 1, 1)(x).shape


def test_the_whole_head_builds_and_runs_both_ways():
    feats = [torch.randn(1, 24, 32 // 2**i, 40 // 2**i) for i in range(5)]
    for mode in ("linear", "text_embedding"):
        head = FCOSHead(24, 80, channels=24, num_convs=1, cls_head=mode, embed_dim=32)
        cls, reg, ctr = head(feats)
        assert cls[0].shape[1] == 80
        assert reg[0].shape[1] == 4 and ctr[0].shape[1] == 1


def test_an_unknown_head_kind_is_refused_rather_than_defaulted():
    """Silently falling back to linear would ship a config that asked for open vocabulary
    and got a frozen one, with nothing in the run saying so."""
    with pytest.raises(ValueError, match="cls_head"):
        FCOSHead(24, 8, channels=24, num_convs=1, cls_head="clip")


# --- the class list is data ------------------------------------------------


def test_installing_a_vocabulary_touches_no_weight():
    """The whole point. A new shop's class list is a different matrix, not a retrain."""
    clf = _clf()
    before = {n: p.detach().clone() for n, p in clf.named_parameters()}
    clf.load_text_embeddings(torch.randn(K, DIM))
    assert all(torch.equal(before[n], p) for n, p in clf.named_parameters())
    assert clf.text_is_learned


def test_the_matrix_is_a_buffer_so_it_is_not_trained_by_accident():
    clf = _clf()
    assert "text_embeddings" not in dict(clf.named_parameters())
    assert "text_embeddings" in dict(clf.named_buffers())


def test_a_wrong_shaped_matrix_is_refused():
    """A mismatch here renames every class rather than failing -- the same shape of error
    `head_order` prevents when a config lists COCO names in the wrong order."""
    clf = _clf()
    for bad in ((K + 1, DIM), (K, DIM + 3)):
        with pytest.raises(ValueError, match="expected"):
            clf.load_text_embeddings(torch.randn(*bad))


def test_installed_embeddings_are_normalised_whatever_scale_they_arrive_at():
    """Text encoders differ in output scale and a caller should not have to know which."""
    clf = _clf()
    clf.load_text_embeddings(torch.randn(K, DIM) * 37.0)
    assert torch.allclose(clf.text_embeddings.norm(dim=-1), torch.ones(K), atol=1e-5)


def test_the_placeholder_vocabulary_is_orthogonal_not_gaussian():
    """Before any real embeddings exist the head still has to be trainable. Gaussian rows
    can start nearly parallel, and two classes sharing a direction cannot be separated by
    anything downstream."""
    clf = _clf()
    gram = clf.text_embeddings @ clf.text_embeddings.T
    off = gram - torch.eye(K)
    assert off.abs().max() < 1e-4


# --- narrowing is a row slice ---------------------------------------------


def test_narrowing_keeps_the_visual_projection_untouched():
    """Nothing about the projection was per-class, so a narrowed text head is the same
    model answering a shorter question. The linear head has to be rebuilt instead."""
    clf = _clf()
    before = clf.embed_pred.weight.detach().clone()
    clf.narrow([0, 3, 7])
    assert torch.equal(clf.embed_pred.weight, before)
    assert clf.num_classes == 3


def test_narrowing_keeps_the_kept_classes_scores_exactly():
    """The property the linear head's slice also has to satisfy: the numbers must be the
    same numbers, not merely the right shape."""
    clf = _clf().eval()
    x = torch.randn(1, CH, 7, 9)
    keep = [1, 4, 9]
    with torch.no_grad():
        full = clf(x)
        clf.narrow(keep)
        got = clf(x)
    torch.testing.assert_close(got, full[:, keep])


# --- the scale, which the focal prior depends on --------------------------


def test_the_logit_scale_starts_where_a_cosine_needs_it():
    """A cosine similarity lives in [-1, 1]; the focal loss's prior assumes logits in a
    much wider range, so a scale of 1 would make every class look equally unlikely."""
    assert _clf().logit_scale.exp().item() == pytest.approx(DEFAULT_LOGIT_SCALE, rel=1e-4)


def test_the_bias_carries_the_focal_prior_per_class():
    """One bias per class rather than a scalar: a text embedding gives no reason for every
    class to start at the same rate, and the bias is where a frequency prior would live."""
    clf = _clf(prior_prob=0.01)
    expected = -math.log((1 - 0.01) / 0.01)
    assert clf.bias.shape == (K,)
    assert torch.allclose(clf.bias, torch.full((K,), expected))


def test_scores_are_bounded_by_the_scale_rather_than_unbounded():
    """Both sides are normalised, so a logit cannot run away with feature magnitude --
    which is what stops one loud channel dominating every class at once."""
    clf = _clf().eval()
    with torch.no_grad():
        out = clf(torch.randn(1, CH, 5, 5) * 100.0)
    limit = DEFAULT_LOGIT_SCALE + clf.bias.abs().max().item() + 1e-3
    assert out.abs().max().item() <= limit


@pytest.mark.parametrize(("dim", "n"), [(0, 4), (8, 0), (-1, 4)])
def test_impossible_dimensions_are_refused(dim, n):
    with pytest.raises(ValueError, match="positive"):
        TextEmbeddingClassifier(CH, dim, n)
