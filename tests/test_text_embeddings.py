"""The class-name matrix, and the two ways it silently renames a taxonomy.

`make_text_embeddings.py` writes the buffer `TextEmbeddingClassifier` scores against.
`load_text_embeddings` checks only the matrix's *shape*, so two failures pass every check
the head has:

* rows in the wrong order -- every channel still decodes and the names are someone
  else's, which is `coco_subsets.head_order`'s argument one level up;
* two classes that CLIP puts in nearly the same direction -- the head is then asked to
  separate directions a linear scorer cannot, and the orthogonal placeholder it replaced
  was strictly better.

The second was nearly missed, and the way it was nearly missed is what these tests are
really for. The first version of the check refused on a raw cosine of 0.95. Measured on
`openai/clip-vit-base-patch32`, **unrelated words score 0.72-0.74, not 0** -- CLIP's text
space is a narrow cone. `device`/`person` measures 0.9458, the worst pair in the retail
vocabulary and a real collapse, and it slides under an 0.95 absolute threshold. The check
would have shipped, run on every matrix, and never once fired.

So what is pinned below is the *relationship* -- excess over the encoder's own measured
floor -- and not the number, for the same reason `test_temporal.py` pins
`diff_thr * plate_alpha` rather than 0.24. Re-tuning the threshold should keep these
green; replacing it with an absolute cosine should not.

pytest tests/test_text_embeddings.py -v
"""

import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def mte():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import make_text_embeddings
    finally:
        sys.path.remove(str(SCRIPTS))
    return make_text_embeddings


# --- head order ----------------------------------------------------------------------


def test_coco_names_come_back_in_category_id_order(mte, tmp_path):
    """Not the order the file lists them. `CocoDetDataset` sorts, so this must too."""
    ann = tmp_path / "instances_train.json"
    ann.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 7, "name": "device"},
                    {"id": 3, "name": "boxed_stock"},
                    {"id": 5, "name": "signage"},
                ]
            }
        )
    )
    assert mte.names_from_coco(ann) == ["boxed_stock", "signage", "device"]


def test_a_file_with_no_categories_is_refused_by_name(mte, tmp_path):
    """A COCO *segmentation* file has no `categories`, and would otherwise yield an empty
    name list -- which builds a zero-row matrix rather than saying what was wrong."""
    ann = tmp_path / "not_detection.json"
    ann.write_text(json.dumps({"images": [], "annotations": []}))
    with pytest.raises(ValueError, match="categories"):
        mte.names_from_coco(ann)


# --- the collision check, which is about the relationship and not the number ----------


def _at(cosine: float) -> torch.Tensor:
    """Two unit vectors at a chosen cosine, so a pair's score is set rather than hoped for."""
    import math

    t = math.acos(max(-1.0, min(1.0, cosine)))
    return torch.tensor([[1.0, 0.0], [math.cos(t), math.sin(t)]])


def test_excess_is_zero_at_the_floor_and_one_at_identical(mte):
    floor = 0.74
    ((_, _, _, at_floor),) = mte.report_collisions(_at(floor), ["a", "b"], floor)
    ((_, _, _, at_one),) = mte.report_collisions(_at(1.0), ["a", "b"], floor)
    assert at_floor == pytest.approx(0.0, abs=1e-6)
    assert at_one == pytest.approx(1.0, abs=1e-6)


def test_a_pair_below_the_floor_is_clamped_rather_than_negative(mte):
    """Unrelated is unrelated. A negative excess reads as opposition, which it is not."""
    ((_, _, cos, excess),) = mte.report_collisions(_at(0.1), ["a", "b"], 0.74)
    assert cos == pytest.approx(0.1, abs=1e-5)
    assert excess == 0.0


def test_the_measured_collision_trips_and_the_measured_legitimate_pair_does_not(mte):
    """The calibration, pinned as the two real measurements it was set between.

    Both are from `openai/clip-vit-base-patch32` with the shipped templates, at the
    measured floor of 0.7369: `device`/`person` 0.9458 is a collapse, and
    `boxed_stock`/`device` 0.8968 is two kinds of merchandise and legitimate.
    """
    floor = 0.7369
    ((_, _, _, collapse),) = mte.report_collisions(_at(0.9458), ["device", "person"], floor)
    ((_, _, _, ok),) = mte.report_collisions(_at(0.8968), ["boxed_stock", "device"], floor)
    assert collapse >= mte.EXCESS_SIMILARITY
    assert ok < mte.EXCESS_SIMILARITY


def test_an_absolute_cosine_threshold_could_not_separate_those_two(mte):
    """Why the check is relative, stated as a test rather than only as a comment.

    This is the regression guard. Both measured pairs sit under 0.95 in raw cosine, so
    any absolute threshold that passes the legitimate pair also passes the collapse.
    Turning this test around means the check has gone back to being an absolute number.
    """
    floor = 0.7369
    collapse, legitimate = 0.9458, 0.8968
    assert collapse < 0.95, "the collapse is under the absolute threshold that was tried"
    # An absolute cut admitting the legitimate pair must sit above it, and the collapse is
    # only 0.049 higher -- inside the spread between phrasings of the same class.
    assert collapse - legitimate < 0.05
    # The relative check does separate them, on the same two numbers.
    ((_, _, _, hi),) = mte.report_collisions(_at(collapse), ["a", "b"], floor)
    ((_, _, _, lo),) = mte.report_collisions(_at(legitimate), ["a", "b"], floor)
    assert lo < mte.EXCESS_SIMILARITY <= hi


def test_pairs_are_reported_most_similar_first(mte):
    """The operator reads the top of this list; an unsorted one buries the collision."""
    m = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.0, 0.0, 1.0]]), dim=-1
    )
    got = [(a, b) for a, b, _, _ in mte.report_collisions(m, ["a", "b", "c"], 0.5)]
    assert got[0] == ("a", "b")


def test_the_floor_is_a_median_so_one_odd_anchor_pair_cannot_move_it(mte, monkeypatch):
    """`encoder_floor` is what every judgement is measured against, so it is deliberately
    not the min. Fed one outlier among ordinary anchors, the floor must ignore it."""
    vals = [0.70, 0.72, 0.74, 0.76, 0.05]  # one anchor pair wildly low
    monkeypatch.setattr(mte, "_cosine_pairs", lambda _m: vals)
    monkeypatch.setattr(mte, "build_matrix", lambda *_a, **_k: torch.zeros(5, 4))
    assert mte.encoder_floor(None, None, "cpu") == pytest.approx(0.72)


# --- argument handling that would otherwise fail silently ----------------------------


def test_prompts_for_a_class_the_head_does_not_have_is_an_error(mte, tmp_path):
    """Ignoring it leaves that class on the default templates while the operator believes
    their prompts are in use -- invisible in the matrix and in the output."""
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps({"devcie": ["a typo"]}))  # transposed on purpose
    with pytest.raises(SystemExit):
        mte.main(
            [
                "--names",
                "boxed_stock,device",
                "--out",
                str(tmp_path / "o.pt"),
                "--prompts-json",
                str(p),
            ]
        )


def test_duplicate_class_names_are_refused(mte, tmp_path):
    """Two rows for one name means one of them is never the argmax, silently."""
    with pytest.raises(SystemExit):
        mte.main(["--names", "device,device", "--out", str(tmp_path / "o.pt")])


# --- the wiring: a matrix on disk has to reach a model, before it trains ---------------

NAMES = ["boxed_stock", "device"]
DIM = 512


def _sidecar(tmp_path, names=NAMES, rows=None):
    """A file shaped like `make_text_embeddings.py`'s output, without the encoder."""
    p = tmp_path / "text.pt"
    torch.save(
        {
            "matrix": torch.nn.functional.normalize(
                torch.randn(rows or len(names), DIM), dim=-1
            ),
            "names": names,
            "encoder": "test",
        },
        p,
    )
    return p


def _cfg(**det):
    head = {"type": "fcos", "num_classes": len(NAMES), "in_levels": [0, 1, 2], "channels": 32}
    head.update(det)
    return {
        "model": {
            "backbone": {"name": "resnet18", "pretrained": False},
            "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
            "heads": {"detection": head},
        },
        "data": {"input_size": [64, 80], "datasets": []},
    }


def test_a_configured_matrix_reaches_the_head_at_construction(tmp_path):
    """The ordering that makes the whole feature work: installed before any gradient flows.

    If this ever regresses to installing at export, `embed_pred` will have spent training
    aligning to the random orthogonal placeholder, and the model will emit confident boxes
    against a text space it has never seen.
    """
    from syncai_hydranet.models.hydranet import build_model

    path = _sidecar(tmp_path)
    model = build_model(
        _cfg(cls_head="text_embedding", text_embeddings=str(path), classes=NAMES)
    )
    clf = model.det_head.cls_pred
    assert clf.text_is_learned, "the placeholder must have been replaced"
    expected = torch.load(path, weights_only=True)["matrix"]
    assert torch.allclose(clf.text_embeddings, expected, atol=1e-6)


def test_the_placeholder_survives_when_no_matrix_is_named():
    """An open-vocabulary head with no matrix yet is legitimate -- it is what you train
    while the vocabulary is still being written -- so this must not raise."""
    from syncai_hydranet.models.hydranet import build_model

    model = build_model(_cfg(cls_head="text_embedding"))
    assert not model.det_head.cls_pred.text_is_learned


def test_a_matrix_under_a_linear_head_is_refused_rather_than_ignored(tmp_path):
    """The quiet misconfiguration: the file is named, nothing reads it, and the config
    still looks as though the vocabulary were data. The run trains the frozen head."""
    from syncai_hydranet.models.hydranet import build_model

    with pytest.raises(ValueError, match="silently unused"):
        build_model(_cfg(text_embeddings=str(_sidecar(tmp_path))))


def test_a_permuted_matrix_is_refused_although_its_shape_is_correct(tmp_path):
    """The failure `load_text_embeddings` cannot see, and the reason names travel with the
    matrix at all. Reversed rows: same shape, every class renamed, no error without this."""
    from syncai_hydranet.models.hydranet import build_model

    path = _sidecar(tmp_path, names=list(reversed(NAMES)))
    with pytest.raises(ValueError, match="Order matters"):
        build_model(_cfg(cls_head="text_embedding", text_embeddings=str(path), classes=NAMES))


def test_a_matrix_with_the_wrong_number_of_classes_names_both_counts(tmp_path):
    from syncai_hydranet.models.hydranet import build_model

    path = _sidecar(tmp_path, names=["a", "b", "c"], rows=3)
    with pytest.raises(ValueError, match="3 rows and this head has 2"):
        build_model(_cfg(cls_head="text_embedding", text_embeddings=str(path)))


def test_a_file_that_is_not_a_sidecar_says_so_rather_than_raising_a_key_error(tmp_path):
    """A bare tensor is the plausible mistake -- someone saves the matrix alone."""
    from syncai_hydranet.models.hydranet import build_model

    path = tmp_path / "bare.pt"
    torch.save(torch.randn(2, DIM), path)
    with pytest.raises(ValueError, match="not a text-embedding sidecar"):
        build_model(_cfg(cls_head="text_embedding", text_embeddings=str(path)))
