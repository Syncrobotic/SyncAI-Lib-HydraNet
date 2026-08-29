"""An export canvas may be stated by a config; it may never be inherited by one.

`cli/export_onnx.py` resolves the export size as::

    ecfg = cfg.get("export") or {}
    h, w = ecfg.get("input_size") or cfg["data"]["input_size"]

and its comment says why the fallback is the right default: "a TensorRT engine built for
the wrong input size is a much later and much more confusing failure than a missing config
key". The fallback is correct and, until `cbffd6a`, it could not fire for any config that
inherited `_base/hydranet.yaml` — the base supplied `export.input_size: [512, 640]`,
`export` is a dict, dicts merge key by key, and `or` then takes the inherited size over the
trained one.

Measured 2026-08-27, resolved through `load_config` rather than by reading the merge rules:
**34 configs reached that key, and 6 of them exported at a size they had never trained at**
— `hydranet_retail_cctv`, `hydranet_retail_cocostuff` and the three
`hydranet_retail_security_b03_cw_hires` variants at 512x896, plus `hydranet_hm3d_cctv` at
480x640, all emitting a 512x640 engine.

**THE OTHER 28 WERE RIGHT BY COINCIDENCE, WHICH IS WHY THIS TEST ASSERTS THE MECHANISM.**
They happen to train at 512x640 — the size the removed default named — so the inherited
value matched by accident. Any one of them changing its training resolution would have
joined the six with nothing to announce it. That makes the defect the *mechanism* rather
than the six outputs that happened to show it, and a test that only compared the two
resolved numbers would go green again the day someone re-added an inherited default that
matched everything. So the property checked here is not "the sizes agree". It is:

    a resolved `export.input_size` must be stated by the config file itself.

**Stating one is legal and stays legal.** A run that genuinely wants a different export
canvas than its training canvas says so in its own file, and
`tests/test_export_cli.py::test_export_input_size_overrides_the_training_size` pins that
the override still takes effect. No shipped config states one today; the rule is written so
that the day one does, it passes.

**Nothing else could have caught this.** CI's export-parity job exports every config and
compares PyTorch against ONNX *at the same size*, so a wrong-but-consistent canvas is
green. The engine builds and runs, because the trunk is fully convolutional — the model
simply sees a canvas it was never trained on, which is exactly the late, confusing failure
the exporter's comment predicts.

`validate=False`, deliberately: this measures how `_base_` merges, and a config that fails
schema validation for an unrelated reason should red that test rather than this one.

pytest tests/test_export_size_is_not_inherited.py -v
"""

from __future__ import annotations

from pathlib import Path

import yaml

from syncai_hydranet.config import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _states_its_own(path: Path) -> bool:
    """Whether this file names `export.input_size`, as opposed to inheriting one."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return False
    export = raw.get("export")
    return isinstance(export, dict) and "input_size" in export


def _resolved_export_size(path: Path):
    cfg = load_config(str(path), validate=False)
    export = cfg.get("export") or {}
    return export.get("input_size")


def _shipped() -> list[Path]:
    """The runnable configs. `_base/` is excluded because a base is not a run.

    It needs no special case even so: a size re-added there reaches its children, and
    every one of them fails the rule below.
    """
    return sorted(CONFIGS.glob("*.yaml"))


def test_no_config_inherits_an_export_input_size():
    offenders = [
        p.name
        for p in _shipped()
        if _resolved_export_size(p) is not None and not _states_its_own(p)
    ]
    assert not offenders, (
        f"{len(offenders)} config(s) resolve an `export.input_size` they do not state:\n"
        + "\n".join(f"  {name}" for name in offenders)
        + "\n\nThe value is reaching them through `_base_`, which is how six configs came "
        "to export a TensorRT engine at a resolution they had never trained at. Remove it "
        "from the base and let `cli/export_onnx.py` fall back to `data.input_size`; a "
        "config that genuinely wants a different export canvas states it in its own file."
    )


def test_every_config_therefore_exports_at_the_size_it_trains():
    """The consequence of the rule above, resolved the way the exporter resolves it.

    Kept as a separate test rather than folded in: this one names the number a reader
    cares about, and it is the assertion that would have caught the original defect from
    the outside without knowing anything about `_base_`.
    """
    wrong = []
    for path in _shipped():
        cfg = load_config(str(path), validate=False)
        trained = (cfg.get("data") or {}).get("input_size")
        if trained is None or _states_its_own(path):
            continue
        exported = (cfg.get("export") or {}).get("input_size") or trained
        if list(exported) != list(trained):
            wrong.append(f"  {path.name}: trains {trained}, exports {exported}")
    assert not wrong, "config(s) exporting at a size they never trained at:\n" + "\n".join(
        wrong
    )


def test_a_config_may_still_state_an_export_size_of_its_own(tmp_path):
    """The rule bans inheriting one, not having one.

    A guard that banned the key outright would be satisfied by deleting a legitimate
    feature, and `test_export_cli.py` pins that the override takes effect once stated. So
    the discrimination is exercised on real files through a real `_base_` chain: the child
    that writes the key is accepted with a size that deliberately disagrees with its
    training size, and the child that only inherits one is rejected -- which is the shape
    `cbffd6a` removed from `_base/hydranet.yaml`.
    """
    base = tmp_path / "base.yaml"
    base.write_text("export:\n  opset: 17\n  input_size: [512, 640]\n", encoding="utf-8")

    inheritor = tmp_path / "inheritor.yaml"
    inheritor.write_text(f"_base_: {base}\ndata:\n  input_size: [512, 896]\n", encoding="utf-8")
    assert not _states_its_own(inheritor)
    assert _resolved_export_size(inheritor) == [512, 640], "the base's value did reach it"

    stater = tmp_path / "stater.yaml"
    stater.write_text(
        f"_base_: {base}\ndata:\n  input_size: [512, 896]\nexport:\n  input_size: [192, 256]\n",
        encoding="utf-8",
    )
    assert _states_its_own(stater)
    assert _resolved_export_size(stater) == [192, 256]


def test_the_scan_reads_the_configs_it_thinks_it_does():
    """A guard whose search returns nothing passes forever.

    Pinned on the training sizes rather than a count, because the set of configs grows.
    The two that matter are the sizes the defect was measured at: 512x896 and 640x1120
    both have to be present for the mismatch this test exists to prevent to be possible
    at all -- if every config trained at one resolution, an inherited default could never
    disagree with any of them.
    """
    sizes = set()
    for path in _shipped():
        cfg = load_config(str(path), validate=False)
        trained = (cfg.get("data") or {}).get("input_size")
        if trained:
            sizes.add(tuple(trained))
    assert len(_shipped()) > 20, "configs/ looks empty -- has it moved?"
    assert (512, 896) in sizes and (640, 1120) in sizes, sorted(sizes)
