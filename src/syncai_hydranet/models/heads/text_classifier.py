"""An open-vocabulary classifier for the detection head: names as data, not as weights.

**The problem this exists for is measured, not hypothetical.** The audit in
`git show b7457c2:docs/RETAIL.md` swept the detection head over Kaohsiung-cam08, an
Apple store: at
score 0.05 it returns **1,683 `book`** and no `laptop` at any threshold. The head is
finding the merchandise and naming it with the nearest shape word COCO owns. **The
localisation already works; the vocabulary does not.** A boxed handset is not a book, and
there is no amount of training on COCO that will make it one, because COCO has no word for
it.

A linear classifier learns one vector per class and the class list is frozen into the
weights. Changing it means retraining. This replaces those vectors with **text embeddings
computed offline**, so the class list becomes a matrix a config points at:

    logits = scale * (normalise(visual) @ normalise(text).T)

What that buys, in the order it matters here:

1. **A word for merchandise.** `boxed phone`, `laptop on a display table`, `accessory
   packet` are things a text encoder can embed and COCO cannot express. The head no longer
   has to round them to `book`.
2. **Per-store vocabularies without retraining.** One shop sells phones and another sells
   clothes; that is a different matrix, not a different model. This is the same argument
   `docs/PLAN.md` §5 makes for every store threshold -- a product parameter belongs in a
   config, not in the weights.
3. **Export narrowing becomes a row slice.** `--detection-classes` already slices
   `cls_pred`'s output channels; here it slices the text matrix instead, and the visual
   projection is untouched.

**Inference costs one convolution and one matmul against a constant.** The text encoder
runs offline and never ships: `scripts/make_text_embeddings.py` writes the matrix, the
matrix is a buffer in the checkpoint, and the exported graph is Conv + MatMul + Mul, all
TensorRT-native. So this is an architecture change with no runtime price, which is why it
is worth doing before either of the other two on the list.

**What it does not do.** It does not make the head detect things it has no visual evidence
for. The cam08 result is what makes this promising -- the boxes are already in the right
places -- and a class with no visual support will still score nothing, more legibly. It
also inherits whatever the text encoder believes:
`git show b7457c2:docs/RETAIL.md` recorded that
web English's `iphone` is a product shot rather than thirty pixels of handset on a shelf,
and an embedding of that word carries the same problem into this matrix.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# CLIP's learned temperature converges near 100; starting there rather than at 1 matters
# because the focal loss's prior assumes logits in a particular range, and a cosine
# similarity lives in [-1, 1] until something scales it.
DEFAULT_LOGIT_SCALE = 100.0


def load_matrix_file(path, num_classes: int, expect_names: list[str] | None = None):
    """Read a `make_text_embeddings.py` sidecar and return its matrix, name-checked.

    The sidecar carries `names` beside `matrix` for one reason:
    `load_text_embeddings` validates the **shape**, and the failure that costs a run is a
    matrix of the right shape whose rows are in a different order. Every channel decodes,
    every box gets a name, and the names are someone else's -- `coco_subsets.head_order`'s
    sentence, one level down.

    `expect_names` comes from the head's own `classes:` config key. It is optional because
    not every config declares names, and checking is refused rather than skipped when it
    can be done: passing names that disagree is an error, omitting them is a choice.
    """
    from ...utils.checkpoint import load_checkpoint

    blob = load_checkpoint(path)
    if not isinstance(blob, dict) or "matrix" not in blob:
        raise ValueError(
            f"{path} is not a text-embedding sidecar: expected a dict with a `matrix` key, "
            f"as written by scripts/make_text_embeddings.py"
        )
    names = blob.get("names")
    if expect_names is not None:
        if names is None:
            raise ValueError(
                f"{path} carries no `names`, so the config's `classes:` cannot be checked "
                f"against it. Regenerate it with scripts/make_text_embeddings.py, which "
                f"always writes them -- an unchecked matrix is the wrong-order failure."
            )
        if list(names) != list(expect_names):
            raise ValueError(
                f"{path} is for classes {list(names)!r} but this head declares "
                f"{list(expect_names)!r}. Order matters as much as membership: row k of "
                f"the matrix is what channel k means, so a permutation renames every "
                f"class while passing every shape check."
            )
    if len(blob["matrix"]) != num_classes:
        raise ValueError(
            f"{path} has {len(blob['matrix'])} rows and this head has {num_classes} "
            f"classes. Names in the file: {names!r}"
        )
    return blob["matrix"]


class TextEmbeddingClassifier(nn.Module):
    """Scores visual features against a frozen matrix of class-name embeddings.

    Args:
        channels: width of the classification tower's output.
        embed_dim: the text encoder's dimension.
        num_classes: how many names. Must match the matrix when one is loaded.
        prior_prob: focal-loss prior, as in ``FCOSHead``. A bias per class rather than a
            scalar, because a text embedding gives no reason for every class to start at
            the same rate and the bias is what carries a class-frequency prior.
    """

    def __init__(
        self,
        channels: int,
        embed_dim: int,
        num_classes: int,
        prior_prob: float = 0.01,
    ):
        super().__init__()
        if embed_dim <= 0 or num_classes <= 0:
            raise ValueError(
                f"embed_dim and num_classes must be positive, got {embed_dim}, {num_classes}"
            )
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        # 3x3 to match the linear head it replaces: the classifier sees the same
        # neighbourhood, only the output space changes.
        self.embed_pred = nn.Conv2d(channels, embed_dim, 3, 1, 1)
        self.bias = nn.Parameter(
            torch.full((num_classes,), -math.log((1 - prior_prob) / prior_prob))
        )
        # log-space so the optimiser sees a well-conditioned parameter, as CLIP does.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(DEFAULT_LOGIT_SCALE)))
        # A buffer, not a parameter: the names are data. It is randomly initialised and
        # **orthogonalised** rather than left Gaussian, so a model trained before any real
        # embeddings exist still has linearly independent class directions -- otherwise two
        # classes can start life nearly parallel and the head cannot separate them at all.
        text = torch.randn(num_classes, embed_dim)
        if num_classes <= embed_dim:
            text = torch.linalg.qr(text.T)[0].T
        self.register_buffer("text_embeddings", F.normalize(text, dim=-1))
        self.text_is_learned = False

    def load_text_embeddings(self, matrix: torch.Tensor) -> None:
        """Install a real text matrix, replacing the orthogonal placeholder.

        Shape-checked rather than broadcast: a matrix with the wrong number of rows would
        silently rename every class after the mismatch, which is the same failure
        ``coco_subsets.head_order`` exists to prevent one level up.
        """
        matrix = torch.as_tensor(matrix, dtype=torch.float32)
        if matrix.shape != (self.num_classes, self.embed_dim):
            raise ValueError(
                f"text embeddings are {tuple(matrix.shape)}, expected "
                f"({self.num_classes}, {self.embed_dim}). A mismatch here renames every "
                f"class rather than failing, so it is refused."
            )
        self.text_embeddings = F.normalize(matrix, dim=-1).to(self.text_embeddings.device)
        self.text_is_learned = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` features to ``[B, num_classes, H, W]`` logits."""
        emb = self.embed_pred(x)
        emb = F.normalize(emb, dim=1)
        # einsum rather than a 1x1 convolution built from the matrix: the matrix is a
        # buffer that can be swapped at export, and a convolution would bake it into a
        # weight that `--detection-classes` would then have to slice in two places.
        logits = torch.einsum("bchw,kc->bkhw", emb, self.text_embeddings)
        return logits * self.logit_scale.exp() + self.bias.view(1, -1, 1, 1)

    def narrow(self, keep: list[int]) -> None:
        """Keep only these classes, in place. The export-time counterpart of a row slice.

        Cheaper and more honest than the linear head's version: the visual projection is
        untouched, because *nothing about it was per-class*. Only the matrix and the bias
        are indexed, so a narrowed head is the same model answering a shorter question.
        """
        idx = torch.as_tensor(keep, dtype=torch.long, device=self.bias.device)
        self.text_embeddings = self.text_embeddings[idx].clone()
        self.bias = nn.Parameter(self.bias.detach()[idx].clone())
        self.num_classes = len(keep)
