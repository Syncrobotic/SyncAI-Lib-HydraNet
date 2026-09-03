"""Pre-processing lives in the graph, so it cannot drift from training.

METHODOLOGY.md assigns "pre-processing parity between training and the robot" to the
deployment stream, and the repository was implementing it twice: `data/transforms.py` for
training, a hand-copied mean/std in `scripts/bench_camera_orin.py` for the Jetson. Nothing
tied the two together. Change one and no test fails, no error appears -- the model on the
robot is just worse, and the blame lands on quantisation. That second copy went with the
Orin on 2026-08-28 (`git show f64520c:scripts/bench_camera_orin.py`), and folding the
constants into the graph is why its removal costs nothing.

Folding the constants into the exported graph turns that from a discipline problem into
an arithmetic one, which is the kind that can be tested. So:

    embedded_graph(raw_rgb_0_255) == plain_graph(training_normalisation(raw_rgb_0_255))

pytest tests/test_export_preprocessing.py -v
"""

import numpy as np
import pytest
import torch
from PIL import Image

from _export_cfg import seg_head, tiny_trunk
from syncai_hydranet.cli.export_onnx import INPUT_NORMALISED, INPUT_RAW, ExportWrapper
from syncai_hydranet.config import load_config
from syncai_hydranet.data.transforms import build_transforms
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.preprocessing import IMAGENET_MEAN, IMAGENET_STD

SIZE = (64, 80)


@pytest.fixture(scope="module")
def model():
    cfg = {
        "model": {
            **tiny_trunk(num_levels=3),
            "heads": {"terrain": seg_head(num_classes=12, channels=16)},
        }
    }
    torch.manual_seed(0)
    return build_model(cfg).eval()


def _raw(seed=0):
    """A frame as a camera delivers it: RGB, 0-255, NCHW float."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(0, 255, (1, 3, *SIZE)).astype(np.float32))


def _normalise_as_training_does(raw: torch.Tensor) -> torch.Tensor:
    arr = raw.numpy()[0].transpose(1, 2, 0) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)[None].astype(np.float32))


def test_embedded_graph_matches_the_training_normalisation(model):
    """The whole point, stated as an equation."""
    raw = _raw()
    embedded = ExportWrapper(model, embed_preprocessing=True)
    plain = ExportWrapper(model, embed_preprocessing=False)
    with torch.no_grad():
        got = embedded(raw)[0]
        want = plain(_normalise_as_training_does(raw))[0]
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_it_agrees_with_the_transform_pipeline_on_a_real_image(model):
    """Not just with a reimplementation of the normalisation -- with the pipeline the
    dataset actually runs, ToTensor included."""
    rng = np.random.default_rng(3)
    img = Image.fromarray(rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8))
    sample = build_transforms(SIZE, train=False)({"image": img, "masks": {}})
    trained_path = sample["image"][None]

    raw = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1)[None])
    with torch.no_grad():
        got = ExportWrapper(model, embed_preprocessing=True)(raw)[0]
        want = ExportWrapper(model, embed_preprocessing=False)(trained_path)[0]
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_the_input_name_states_the_contract(model):
    """The engine keeps binding names but not ONNX metadata, so the name is what tells a
    runtime which convention it is looking at. A runtime written for one convention must
    fail to find its binding in the other rather than feed it the wrong range."""
    assert ExportWrapper(model, embed_preprocessing=True).input_name == INPUT_RAW
    assert ExportWrapper(model, embed_preprocessing=False).input_name == INPUT_NORMALISED
    assert INPUT_RAW != INPUT_NORMALISED


def test_normalisation_constants_come_from_one_place(model):
    """If these are ever re-typed rather than imported, this is the test that notices."""
    w = ExportWrapper(model, embed_preprocessing=True)
    np.testing.assert_allclose(w.pre_mean.numpy().reshape(3), np.array(IMAGENET_MEAN) * 255)
    np.testing.assert_allclose(w.pre_std.numpy().reshape(3), np.array(IMAGENET_STD) * 255)


def test_double_normalisation_is_detectable(model):
    """Guards the test above from passing vacuously: if a runtime normalised *and* the
    graph did, the outputs must differ. A near-miss here would mean the assertion could
    not tell the two apart in the first place."""
    raw = _raw(1)
    embedded = ExportWrapper(model, embed_preprocessing=True)
    with torch.no_grad():
        correct = embedded(raw)[0]
        doubled = embedded(_normalise_as_training_does(raw))[0]
    assert not torch.allclose(correct, doubled, rtol=1e-2, atol=1e-2)


def test_shipped_configs_export_with_preprocessing_by_default():
    """The default is what ends up on robots; an opt-in fix is a fix nobody applies."""
    cfg = load_config("configs/hydranet_indoor.yaml")
    m = build_model(cfg).eval()
    assert ExportWrapper(m).embed_preprocessing is True
