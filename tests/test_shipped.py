"""`shipped.py`: which run the tools ship, and which checkpoint answers which question.

Untested until 2026-09-04, on the module that exists *because* this project shipped the
wrong checkpoint: `best.pt` is selected on one head's metric, and on the run before the
current one it was a 40% worse person detector than `last.pt` while every tool pointed
at it. The module's answer is two functions rather than one path, and six call sites
were consolidated into it so that promoting a run is one edit.

What is worth holding here is what a caller depends on and cannot see: that both
questions resolve to a file that exists inside the run this module names, that the
config sits beside the checkpoint rather than in `configs/` (a config in `configs/` is
what the NEXT run will use), and that the paths are absolute so a tool's working
directory cannot change which weights it loads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from syncai_hydranet import shipped


def test_the_run_is_named_absolutely_so_a_working_directory_cannot_change_it():
    """Every consumer is a script or a tool, run from wherever the operator happens to
    be. A relative run path would make the shipped weights a property of the shell."""
    assert shipped.SHIPPED_RUN.is_absolute()
    assert shipped.SHIPPED_CONFIG.is_absolute()


def test_the_config_comes_from_the_run_and_not_from_configs():
    """Stated in the module and load-bearing: `configs/<name>.yaml` is what the next run
    will use and may already have moved, so a tool reading it would preprocess with one
    config and load weights trained under another."""
    assert shipped.SHIPPED_CONFIG.parent == shipped.SHIPPED_RUN
    assert shipped.SHIPPED_CONFIG.name == "config.yaml"
    assert Path("configs") not in shipped.SHIPPED_CONFIG.parents


@pytest.mark.parametrize("resolve", [shipped.for_terrain, shipped.for_detection])
def test_both_questions_resolve_inside_the_run_they_name(resolve):
    """The two-answers doctrine: a caller names the head it is judged on and gets the
    checkpoint that run won on. Whatever they resolve to must live in the shipped run --
    a checkpoint from a different directory is the six-copies-of-a-string bug returning.
    """
    ckpt = resolve()
    assert ckpt.is_absolute()
    assert ckpt.parent == shipped.SHIPPED_RUN
    assert ckpt.suffix == ".pt"


@pytest.mark.skipif(
    not shipped.SHIPPED_RUN.is_dir(),
    reason="runs/ is gitignored; this asserts against the real run when it is present",
)
def test_the_run_on_disk_actually_holds_what_this_module_promises():
    """The half a path check cannot make. Skipped on a clean checkout by design -- but
    on the box that trains and renders, a promoted run whose files are missing is
    exactly the failure this module was written after."""
    assert shipped.SHIPPED_CONFIG.is_file(), f"{shipped.SHIPPED_CONFIG} is missing"
    for resolve in (shipped.for_terrain, shipped.for_detection):
        assert resolve().is_file(), f"{resolve.__name__}() -> {resolve()} is missing"


@pytest.mark.skipif(
    not (shipped.SHIPPED_RUN / "selection.json").is_file(),
    reason="runs/ is gitignored; the quoted numbers are checked where the run is",
)
def test_the_numbers_the_docstring_quotes_are_this_run_s_numbers():
    """The module docstring carried an epoch-15-vs-60 table for a run it no longer ships.

    It read as this run's evidence and belonged to `..._b03_cw_xl`, whose best.pt really
    was a 40% worse person detector -- while person01 selected epoch 118 of 120 on a flat
    curve, which is why both accessors return `last.pt`. A quoted measurement with nothing
    checking it is this project's most familiar defect; this is the check.
    """
    import json

    sel = json.loads((shipped.SHIPPED_RUN / "selection.json").read_text())
    assert sel["primary_metric"] == "detection_mAP/site_person"
    person = sel["heads"]["detection_mAP/site_person"]
    assert person["selected_epoch"] == 118
    assert round(person["at_selected"], 4) == 0.7387
    # The claim that matters: the two checkpoints are the same model to three decimals,
    # which is what lets both accessors return last.pt.
    terrain = sel["heads"]["terrain_mIoU/site_seg03"]
    assert abs(terrain["at_selected"] - 0.6716) < 5e-4


def test_load_model_returns_an_eval_model_with_its_config_and_device(tmp_path):
    """The four lines twenty-six call sites each wrote.

    Held on a tiny config rather than the shipped run, so it runs on a clean checkout:
    what matters is the contract -- eval mode, the config back for `input_size`, and the
    device back for moving tensors -- not which weights arrive.
    """
    import torch

    from syncai_hydranet.config import load_config
    from syncai_hydranet.models.hydranet import build_model

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "hydranet_indoor.yaml"
    cfg = load_config(str(cfg_path), ["model.backbone.pretrained=false"])
    ckpt = tmp_path / "w.pt"
    torch.save({"model": build_model(cfg).state_dict()}, ckpt)

    model, out_cfg, device = shipped.load_model(cfg_path, ckpt, device="cpu", weights="model")
    assert not model.training, "a loaded model must be in eval mode"
    assert out_cfg["data"]["input_size"] == cfg["data"]["input_size"]
    assert str(device) == "cpu"


def test_load_model_takes_the_weights_choice_as_an_argument():
    """`select_weights` records why: EMA weights on a run too short to have earned them
    scored 0.16 mIoU against the raw weights' 0.95, and a tool that hardcodes the choice
    cannot say which one it rendered. Twenty-six call sites hardcoded `"ema"`."""
    import inspect

    sig = inspect.signature(shipped.load_model)
    assert sig.parameters["weights"].default == "ema"
    assert sig.parameters["weights"].kind is inspect.Parameter.KEYWORD_ONLY


def test_load_model_says_so_when_it_could_not_give_the_weights_asked_for():
    """The fallback inside `select_weights` is the one that is worth a word.

    Asking for EMA on a checkpoint that has none returns the raw tensors, which on a short
    run is a different model -- 0.16 mIoU against 0.95 in the incident that put the
    docstring there. Every caller before this defaulted to `"ema"` without checking, so a
    run trained without EMA rendered from raw weights and said nothing.
    """
    import warnings

    from syncai_hydranet.utils.checkpoint import chosen_weights

    with_ema = {"model": {"a": 1}, "ema": {"a": 2}}
    without = {"model": {"a": 1}}
    assert chosen_weights(with_ema, "ema") == ({"a": 2}, "ema")
    assert chosen_weights(without, "ema") == ({"a": 1}, "model"), "the fallback is named"
    assert chosen_weights(with_ema, "model") == ({"a": 1}, "model")
    # An empty EMA dict is not EMA weights, and falls back like a missing one.
    assert chosen_weights({"model": {"a": 1}, "ema": {}}, "ema")[1] == "model"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chosen_weights(without, "ema")
    assert not caught, "the primitive is silent; the warning belongs to load_model"


def test_load_model_warns_on_the_real_checkpoint_it_cannot_satisfy(tmp_path):
    """The branch above, on the path a caller actually takes.

    `weights="ema"` is the default, so the run that trained without EMA is the one that
    trips this, and it is exactly the run -- short -- on which the two sets of tensors
    differ most.
    """
    import warnings

    import torch

    from syncai_hydranet.config import load_config
    from syncai_hydranet.models.hydranet import build_model

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "hydranet_indoor.yaml"
    cfg = load_config(str(cfg_path), ["model.backbone.pretrained=false"])
    ckpt = tmp_path / "no_ema.pt"
    torch.save({"model": build_model(cfg).state_dict()}, ckpt)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shipped.load_model(cfg_path, ckpt, device="cpu")  # weights="ema" by default
    said = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert said and "there are none" in said[0], f"loaded raw weights in silence: {caught}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shipped.load_model(cfg_path, ckpt, device="cpu", weights="model")
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], (
        "asking for what the checkpoint holds is not worth a warning"
    )
