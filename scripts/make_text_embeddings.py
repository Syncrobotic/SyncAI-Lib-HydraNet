#!/usr/bin/env python3
"""Write the class-name embedding matrix `TextEmbeddingClassifier` scores against.

    python3 scripts/make_text_embeddings.py \\
        --coco datasets/retail_objects_batch02/annotations/instances_train.json \\
        --out weights/text/retail_products.pt

**This file was cited before it existed.** `models/heads/text_classifier.py`'s docstring
says "`scripts/make_text_embeddings.py` writes the matrix, the matrix is a buffer in the
checkpoint" -- and there was no such script, no config setting `cls_head:
text_embedding`, and no caller of `load_text_embeddings` outside `tests/`. That is the
same shape as `unsourced_classes()` having only test callers and
`label_maps_retail_objects.py` citing a validator that did not exist: the insight was
written down, the mechanism was not built, and nothing surfaced the gap.

---------------------------------------------------------------------------
THE ORDERING CONSTRAINT, WHICH IS THE WHOLE REASON TO READ THIS HEADER

`text_classifier.py` promises "per-store vocabularies without retraining". **That
promise holds only if the model was trained with a real matrix installed.** It is not a
property of the head; it is a property of the training run, and nothing in the code
enforces it.

The head learns `embed_pred`, a convolution into the text encoder's space. What it
learns is *alignment with whatever matrix was in place while the gradient flowed*. The
buffer is initialised to a random orthogonal placeholder, which exists so an untrained
head still has linearly independent class directions -- it is deliberately **not** a
text space. So:

    train with placeholder -> swap in CLIP embeddings   = noise. The visual projection
                                                          is aimed at random directions.
    train with CLIP matrix -> swap in a different one   = works. Both live in the same
                                                          space, which is the point.

So this script runs **before** a run, not after one, and its output belongs in the
config that trains the head. A matrix installed at export time onto a placeholder-trained
head produces confident, meaningless boxes -- the failure this project keeps naming,
where nothing errors and the output is plausible.

---------------------------------------------------------------------------
NAME ORDER IS HEAD ORDER, AND IT IS NOT THE ORDER YOU WROTE

Row `k` of the matrix must be the class the head emits on channel `k`. For a COCO
detection dataset that is `sorted(getCatIds())` -- category-id order, not the order the
annotation file lists them and not the order a config mentions them. `--coco` reads and
sorts, so the caller cannot get it wrong by hand; `--names` is the escape hatch and
trusts you.

`coco_subsets.head_order` makes the same argument one level up and its sentence applies
here unchanged: getting it wrong is the quiet kind of wrong, because every channel still
decodes, every box still gets a name, and the names are someone else's. The output
therefore carries `names` beside the matrix so a loader can check them, since
`load_text_embeddings` checks only the shape -- and a matrix with the right number of
rows in the wrong order passes that check.

---------------------------------------------------------------------------
WHAT IS MEASURED HERE AND WHAT IS NOT

Measured: the pairwise cosine similarities printed at the end, and the refusal that
follows from them. Not measured: the prompt templates. They are written from the same
reasoning as `sam3_prompts_objects.py`'s -- a shelf-level phrase beats a bare noun for
CCTV-scale merchandise -- and that reasoning has never been checked for *text* prompts
in this project. `docs/RETAIL.md` records the related finding for SAM 3, that
web English's `iphone` is a product shot rather than thirty pixels of handset on a
shelf. An embedding of that word carries the same problem into this matrix, and no
amount of ensembling fixes a word that means the wrong picture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# CLIP ViT-B/32's text projection is 512-wide, which is where `FCOSHead`'s
# `embed_dim: int = 512` default comes from. Changing the encoder changes that default
# and the config has to say so, which is why the encoder id travels in the output.
DEFAULT_ENCODER = "openai/clip-vit-base-patch32"

# Prompt ensembling, CLIP's own recipe: encode several phrasings, average the normalised
# vectors, renormalise. It reduces the variance of a single unlucky phrasing. These
# templates are aimed at fixed overhead retail CCTV rather than at web photographs --
# `{}` is the class name.
#
# UNMEASURED. See the header. The honest way to improve them is to score a candidate set
# against pre-labelled site frames, not to add phrasings that sound better.
DEFAULT_TEMPLATES = (
    "a photo of {}",
    "a {} on a shop shelf",
    "a {} in a retail store",
    "a security camera photo of {}",
    "a small {} seen from above",
)

# Words chosen to have nothing to do with any taxonomy this repo has or will have. They
# are never written to the matrix; they exist only to measure what this encoder calls
# "unrelated", because that is not zero and assuming it is makes every collision check
# meaningless.
#
# MEASURED, on `openai/clip-vit-base-patch32` with the templates above: unrelated pairs
# sit at **0.72-0.74**, not near 0. `laptop`/`giraffe` is 0.7248 and `giraffe`/`fire
# truck` is 0.7410. CLIP's text space is a narrow cone and every pair in it scores high.
ANCHOR_WORDS = ("giraffe", "fire truck", "volcano", "violin", "glacier")

# How far up the *remaining* range a pair sits: 0.0 is the encoder's unrelated floor and
# 1.0 is identical. This is the mechanism to pin, not a raw cosine -- a raw threshold
# silently inherits whatever anisotropy the encoder has, so changing encoders would
# change what the check means without changing the number.
#
# The threshold is a **tripwire calibrated on the one collision measured**, in the same
# spirit as `THIN_SUPPORT` in the evaluator, not a statistical criterion. Measured on the
# retail vocabulary: `device`/`person` is cosine 0.9458 -> excess **0.79**, the highest
# pair in the set and a genuine collapse (a shopper and the thing they are holding).
# `boxed_stock`/`device` is 0.8968 -> excess **0.60**, elevated and legitimate: they are
# both merchandise. So the line falls between them.
#
# Note what an absolute threshold would have done here. At `COLLISION_COSINE = 0.95` --
# the first version of this check -- `device`/`person` at 0.9458 passes, and nothing in
# the retail vocabulary ever trips. The check would have shipped, run on every matrix,
# and never fired once.
EXCESS_SIMILARITY = 0.75


def names_from_coco(path: Path) -> list[str]:
    """Category names in head order: sorted by category id, as `CocoDetDataset` does."""
    data = json.loads(path.read_text())
    cats = data.get("categories")
    if not cats:
        raise ValueError(f"{path} has no `categories` array; it is not a COCO detection file")
    return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]


def load_encoder(encoder: str, device: str):
    """The tokeniser and text tower, loaded once and shared by the classes and anchors."""
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tok = AutoTokenizer.from_pretrained(encoder)
    model = CLIPTextModelWithProjection.from_pretrained(encoder).to(device).eval()
    return tok, model


def build_matrix(
    names: list[str],
    templates: tuple[str, ...],
    prompts: dict[str, list[str]],
    tok,
    model,
    device: str,
) -> torch.Tensor:
    """One L2-normalised row per class, ensembled over that class's prompts."""
    rows = []
    for name in names:
        # An explicit prompt list replaces the templates rather than adding to them: a
        # class worth writing prompts for is one where "a photo of {}" is the phrasing
        # that was wrong.
        phrases = prompts.get(name) or [t.format(name.replace("_", " ")) for t in templates]
        batch = tok(phrases, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**batch).text_embeds
        emb = F.normalize(emb, dim=-1).mean(dim=0)
        rows.append(F.normalize(emb, dim=-1))
    return torch.stack(rows).cpu()


def _cosine_pairs(matrix: torch.Tensor) -> list[float]:
    sim = matrix @ matrix.T
    n = len(matrix)
    return [float(sim[i, j]) for i in range(n) for j in range(i + 1, n)]


def encoder_floor(tok, model, device: str) -> float:
    """What this encoder calls "unrelated" -- the median cosine over `ANCHOR_WORDS`.

    Median rather than min: one unlucky anchor pair should not move the floor, and the
    floor is what every later judgement is measured against. Computed per run rather than
    hard-coded, so swapping `--encoder` re-derives it instead of silently reinterpreting
    a constant that was measured on a different model.
    """
    anchors = build_matrix(list(ANCHOR_WORDS), DEFAULT_TEMPLATES, {}, tok, model, device)
    vals = sorted(_cosine_pairs(anchors))
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def report_collisions(
    matrix: torch.Tensor, names: list[str], floor: float
) -> list[tuple[str, str, float, float]]:
    """Every class pair as `(a, b, cosine, excess)`, most similar first.

    Printed in full: the matrix is small, and a reader who sees only the refused pairs
    cannot tell a tight taxonomy from a loose one. `excess` is the cosine rescaled so the
    encoder's own unrelated floor is 0 -- clamped there, because a pair *below* the floor
    is simply unrelated and a negative number invites reading it as opposition.
    """
    span = max(1.0 - floor, 1e-6)
    sim = matrix @ matrix.T
    pairs = [
        (names[i], names[j], float(sim[i, j]), max(0.0, (float(sim[i, j]) - floor) / span))
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    return sorted(pairs, key=lambda p: -p[2])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--coco", type=Path, help="COCO annotations json; names in head order")
    src.add_argument("--names", help="comma-separated class names, in head order")
    ap.add_argument("--out", type=Path, required=True, help="destination .pt")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument(
        "--prompts-json",
        type=Path,
        help='{"class_name": ["phrase", ...]} -- replaces the templates for those classes',
    )
    ap.add_argument("--device", default="cpu", help="the encoder runs once; cpu is fine")
    ap.add_argument(
        "--allow-collisions",
        action="store_true",
        help=f"downgrade the excess>={EXCESS_SIMILARITY} refusal to a warning",
    )
    a = ap.parse_args(argv)

    names = (
        names_from_coco(a.coco)
        if a.coco
        else [s.strip() for s in a.names.split(",") if s.strip()]
    )
    if not names:
        ap.error("no class names")
    if len(set(names)) != len(names):
        ap.error(f"duplicate class names: {names}")

    prompts = json.loads(a.prompts_json.read_text()) if a.prompts_json else {}
    unknown = sorted(set(prompts) - set(names))
    if unknown:
        # Silently ignoring these would leave a class on the default templates while the
        # operator believes their prompts are in use, which is invisible in the output.
        ap.error(f"--prompts-json names classes that are not in the head: {', '.join(unknown)}")

    print(f"encoding {len(names)} classes with {a.encoder}", file=sys.stderr)
    tok, model = load_encoder(a.encoder, a.device)
    matrix = build_matrix(names, DEFAULT_TEMPLATES, prompts, tok, model, a.device)

    floor = encoder_floor(tok, model, a.device)
    print(
        f"\nunrelated floor for this encoder: {floor:.4f} "
        f"(median over {len(ANCHOR_WORDS)} anchor words). Cosines are read against this, "
        f"not against 0.",
        file=sys.stderr,
    )

    pairs = report_collisions(matrix, names, floor)
    print("\n  cosine  excess  pair", file=sys.stderr)
    for x, y, c, e in pairs:
        flag = "  <-- too close" if e >= EXCESS_SIMILARITY else ""
        print(f"  {c:+.4f}  {e:5.2f}   {x} / {y}{flag}", file=sys.stderr)

    bad = [p for p in pairs if p[3] >= EXCESS_SIMILARITY]
    if bad and not a.allow_collisions:
        print(
            f"\nREFUSED: {len(bad)} class pair(s) at excess >= {EXCESS_SIMILARITY}. A linear "
            f"scorer cannot separate directions this close, and the placeholder buffer this "
            f"would replace is orthogonal -- installing this matrix is a regression, not an "
            f"improvement. Rewrite those classes' prompts, or pass --allow-collisions if the "
            f"collision is the honest answer and the classes really are near-synonyms.",
            file=sys.stderr,
        )
        return 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "matrix": matrix,
            # The names are the point of the sidecar: `load_text_embeddings` checks shape
            # only, so this is the only record that can catch a right-sized wrong-ordered
            # matrix. A loader that ignores this field throws away the guard.
            "names": names,
            "encoder": a.encoder,
            # Kept so a matrix on disk can be regenerated or audited without guessing
            # which phrasings produced it -- the templates here are unmeasured, so the
            # next person to measure them needs to know what they are comparing against.
            "prompts": {
                n: prompts.get(n) or [t.format(n.replace("_", " ")) for t in DEFAULT_TEMPLATES]
                for n in names
            },
        },
        a.out,
    )
    print(f"\nwrote {a.out}: {tuple(matrix.shape)}, {len(names)} classes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
