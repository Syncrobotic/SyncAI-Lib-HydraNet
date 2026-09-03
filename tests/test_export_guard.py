"""An untrained head must not reach an engine.

A head that no dataset supervises still gets built, still gets exported and still emits
output at inference -- initial random weights producing boxes that nothing downstream can
distinguish from real ones. Training only warns, because assembling datasets incrementally
is normal; export is where it becomes a deployed defect.

pytest tests/test_export_guard.py -v
"""

from pathlib import Path

import pytest

from syncai_hydranet.cli.export_onnx import check_heads_are_trained
from syncai_hydranet.config import load_config
from syncai_hydranet.config_schema import unsupervised_heads

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _cfg():
    return load_config(CONFIG_DIR / "hydranet_indoor.yaml")


def _drop_dataset(cfg, name):
    cfg["data"]["datasets"] = [d for d in cfg["data"]["datasets"] if d["name"] != name]
    return cfg


# ------------------------------------------------------- unsupervised_heads


def test_shipped_configs_supervise_every_head():
    """If this fails, a shipped config would export a head trained on nothing."""
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert paths, "no configs found; the loop below would silently pass"
    for path in paths:
        cfg = load_config(path)
        assert unsupervised_heads(cfg) == set(), f"{path.name} strands a head"


def test_dropping_coco_strands_the_detection_head():
    assert unsupervised_heads(_drop_dataset(_cfg(), "coco")) == {"detection"}


def test_dropping_the_segmentation_dataset_strands_both_seg_heads():
    assert unsupervised_heads(_drop_dataset(_cfg(), "ade20k")) == {
        "traversability",
        "terrain",
    }


def test_no_datasets_at_all_strands_everything():
    cfg = _cfg()
    cfg["data"]["datasets"] = []
    assert unsupervised_heads(cfg) == {"traversability", "terrain", "detection"}


def test_missing_sections_do_not_raise():
    """Called before the model is built, so it must tolerate a malformed config."""
    assert unsupervised_heads({}) == set()
    assert unsupervised_heads({"model": {}}) == set()
    assert unsupervised_heads({"model": {"heads": {"a": {}}}, "data": None}) == {"a"}


# ------------------------------------------------------- the guard itself


def test_guard_passes_when_every_head_is_supervised():
    check_heads_are_trained(_cfg(), allow=False)  # must not raise


def test_guard_refuses_an_unsupervised_head():
    with pytest.raises(SystemExit) as e:
        check_heads_are_trained(_drop_dataset(_cfg(), "coco"), allow=False)
    assert "detection" in str(e.value)


def test_refusal_names_the_remedies():
    """A refusal that does not say what to do next just gets worked around."""
    with pytest.raises(SystemExit) as e:
        check_heads_are_trained(_drop_dataset(_cfg(), "coco"), allow=False)
    msg = str(e.value)
    assert "--allow-untrained-heads" in msg
    assert "model.heads" in msg


def test_the_escape_hatch_is_explicit(capsys):
    """Deliberate shape-only exports stay possible, but say so on stdout."""
    check_heads_are_trained(_drop_dataset(_cfg(), "coco"), allow=True)
    out = capsys.readouterr().out
    assert "WARNING" in out and "detection" in out


def test_escape_hatch_is_silent_when_nothing_is_stranded(capsys):
    check_heads_are_trained(_cfg(), allow=True)
    assert capsys.readouterr().out == ""
