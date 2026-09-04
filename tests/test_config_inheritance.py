"""`_base_` inheritance, and the property that made it safe to introduce.

The three shipped configs were 390 lines of which the optimiser, schedule and export
blocks were repeated verbatim. The cost of that is not the duplication -- it is that
changing `warmup_iters` in two files out of three produces an experiment whose
difference nobody intended, and which no test would catch.

The bar for the refactor is that every shipped config still merges to *exactly* what its
file used to contain, because runs already in flight have to stay comparable to runs
started afterwards.

pytest tests/test_config_inheritance.py -v
"""

from pathlib import Path

import pytest
import yaml

from syncai_hydranet.config import load_config, merge_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return path


# ------------------------------------------------------------------------ merge rules


def test_dicts_merge_key_by_key():
    base = {"train": {"lr": 1e-4, "epochs": 60}}
    over = {"train": {"lr": 5e-4}}
    assert merge_config(base, over) == {"train": {"lr": 5e-4, "epochs": 60}}


def test_lists_replace_rather_than_concatenate():
    """`data.datasets` is the case that matters: a config swapping ADE20K for store
    footage must get one list, not both."""
    merged = merge_config(
        {"data": {"datasets": [{"name": "a"}]}}, {"data": {"datasets": [{"name": "b"}]}}
    )
    assert merged["data"]["datasets"] == [{"name": "b"}]


def test_merge_does_not_mutate_the_base():
    base = {"train": {"lr": 1e-4}}
    merge_config(base, {"train": {"lr": 9.0}})
    assert base == {"train": {"lr": 1e-4}}


# --------------------------------------------------------------------------- loading


def test_child_wins_over_base(tmp_path):
    write(tmp_path, "base.yaml", {"experiment": "b", "train": {"lr": 1e-4, "epochs": 60}})
    child = write(
        tmp_path,
        "child.yaml",
        {"_base_": "base.yaml", "experiment": "c", "train": {"lr": 3e-4}},
    )
    cfg = load_config(child, validate=False)
    assert cfg["experiment"] == "c"
    assert cfg["train"] == {"lr": 3e-4, "epochs": 60}


def test_base_paths_are_relative_to_the_file_that_declares_them(tmp_path):
    write(tmp_path, "_base/core.yaml", {"train": {"epochs": 60}})
    child = write(tmp_path, "child.yaml", {"_base_": "_base/core.yaml", "experiment": "c"})
    assert load_config(child, validate=False)["train"]["epochs"] == 60


def test_several_bases_apply_left_to_right(tmp_path):
    write(tmp_path, "a.yaml", {"train": {"lr": 1.0, "epochs": 1}})
    write(tmp_path, "b.yaml", {"train": {"lr": 2.0}})
    child = write(tmp_path, "c.yaml", {"_base_": ["a.yaml", "b.yaml"]})
    cfg = load_config(child, validate=False)
    assert cfg["train"] == {"lr": 2.0, "epochs": 1}


def test_the_base_key_does_not_survive_into_the_config(tmp_path):
    """It would otherwise reach the schema, meta.json and the checkpoint as a real key."""
    write(tmp_path, "base.yaml", {"train": {"epochs": 1}})
    child = write(tmp_path, "child.yaml", {"_base_": "base.yaml"})
    assert "_base_" not in load_config(child, validate=False)


def test_a_circular_chain_is_refused(tmp_path):
    write(tmp_path, "a.yaml", {"_base_": "b.yaml"})
    write(tmp_path, "b.yaml", {"_base_": "a.yaml"})
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml", validate=False)


def test_a_missing_base_is_an_error_not_a_silent_default(tmp_path):
    child = write(tmp_path, "child.yaml", {"_base_": "nope.yaml"})
    with pytest.raises(FileNotFoundError):
        load_config(child, validate=False)


def test_overrides_still_win_over_everything(tmp_path):
    write(tmp_path, "base.yaml", {"train": {"lr": 1e-4}})
    child = write(tmp_path, "child.yaml", {"_base_": "base.yaml", "train": {"lr": 2e-4}})
    cfg = load_config(child, ["train.lr=9e-4"], validate=False)
    assert cfg["train"]["lr"] == 9e-4


# ------------------------------------------------------------------- shipped configs


@pytest.mark.parametrize("name", ["hydranet_indoor", "hydranet_retail"])
def test_shipped_configs_still_validate(name):
    load_config(CONFIG_DIR / f"{name}.yaml")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("hydranet_indoor", 12),
        ("hydranet_retail", 13),
    ],
)
def test_terrain_head_width_matches_the_class_list(name, expected):
    """The head width and the class list are declared in different files now. If they
    ever disagree, the last class is silently unlearnable."""
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    assert cfg["model"]["heads"]["terrain"]["num_classes"] == expected
    assert len(cfg["data"]["terrain_classes"]) == expected


def test_the_base_is_not_itself_a_shippable_config():
    """It omits experiment, output_dir and datasets on purpose: a config that forgets
    one should fail validation rather than inherit somebody else's."""
    base = yaml.safe_load((CONFIG_DIR / "_base" / "hydranet.yaml").read_text())
    assert not {"experiment", "output_dir"} & set(base)
    assert "datasets" not in base["data"]


def test_configs_dir_glob_does_not_pick_up_the_base():
    """CI exports every `configs/*.yaml`; the base is not exportable and must stay out
    of that glob."""
    assert not any(p.name == "hydranet.yaml" for p in CONFIG_DIR.glob("*.yaml"))
