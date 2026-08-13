"""The config loader: dot-path overrides and scalar parsing.

pytest tests/test_config.py -v
"""

import pytest

from syncai_hydranet.config import Config, load_config
from syncai_hydranet.config_schema import ConfigError

CONFIG = "configs/hydranet_indoor.yaml"


# ------------------------------------------------------------ scalar parsing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1e-4", 1e-4),  # YAML 1.1 alone would return the string "1e-4"
        ("2.0e-4", 2.0e-4),
        ("0.05", 0.05),
        ("8", 8),
        ("-1", -1),
        ("true", True),
        ("false", False),
        ("[384, 512]", [384, 512]),
        ("cosine", "cosine"),
        ("traversability_mIoU", "traversability_mIoU"),
    ],
)
def test_override_values_parse_to_their_obvious_type(text, expected):
    cfg = Config({})
    cfg.set_path("k", text)
    assert cfg["k"] == expected
    assert isinstance(cfg["k"], type(expected))


def test_numeric_override_is_a_number_not_a_string():
    """A string LR happens to survive float() in the optimizer, but reaches the
    checkpoint and meta.json as text, so runs cannot be compared numerically."""
    cfg = load_config(CONFIG, ["train.lr=1e-4"])
    assert cfg["train"]["lr"] == pytest.approx(1e-4)
    assert not isinstance(cfg["train"]["lr"], str)


# ---------------------------------------------------------------- dot paths


def test_nested_override():
    cfg = load_config(CONFIG, ["model.neck.name=fpn"])
    assert cfg["model"]["neck"]["name"] == "fpn"


def test_attribute_access_reaches_nested_values():
    cfg = load_config(CONFIG)
    assert cfg.model.neck.name == "bifpn"
    assert cfg.get_path("train.optimizer") == "adamw"
    assert cfg.get_path("train.nope", "fallback") == "fallback"


def test_list_index_syntax_is_now_caught_rather_than_ignored():
    """set_path cannot index lists; it used to create a key literally named
    "datasets[0]" and the real setting stayed untouched. Validation now rejects it."""
    with pytest.raises(ConfigError, match=r"datasets\[0\]"):
        load_config(CONFIG, ["data.datasets[0].root=/tmp/x"])


def test_override_without_equals_is_rejected():
    with pytest.raises(ValueError, match="key=value"):
        load_config(CONFIG, ["train.lr"])


def test_clone_is_a_deep_copy():
    cfg = load_config(CONFIG)
    twin = cfg.clone()
    twin["train"]["lr"] = 999
    assert cfg["train"]["lr"] != 999
