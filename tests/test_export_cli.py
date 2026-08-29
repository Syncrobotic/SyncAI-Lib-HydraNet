"""`export_onnx.main()` end to end — the function nothing called.

Seven test files exercise this module's helpers: preprocessing, the wrapper, the
unsupervised-head guard, class narrowing, the folded argmax, weight selection. Between
them they call `main(` **zero** times, and `main` is 145 lines carrying every refusal and
every piece of flag wiring. `test_export_guard.py` has ten tests on `unsupervised_heads`,
the predicate; the refusal that consumes it was never executed.

That is the same shape as the other defects found on 2026-08-17: every component
verified, the composition not. What ships to the robot is the composition.

`torch.onnx.export` and `onnx` are stubbed rather than run. The point here is the
decisions `main` makes — what it refuses, in what order, and what metadata it attaches —
not whether torch can serialise a graph, which `test_smoke.py::test_onnx_export` already
covers with a real export.

pytest tests/test_export_cli.py -v
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from syncai_hydranet.cli import export_onnx

BASE = {
    "experiment": "export_cli_test",
    "output_dir": "runs/export_cli_test",
    "model": {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
        "heads": {
            "traversability": {
                "type": "semantic_fpn",
                "num_classes": 3,
                "in_levels": [0, 1, 2],
                "channels": 32,
            },
            "detection": {"type": "fcos", "num_classes": 80, "channels": 32, "num_convs": 1},
        },
    },
    # Small on purpose -- this exercises control flow and every test builds a model -- but
    # not smaller than 128x160. At 64x80 the deepest FPN level is 1x1 and `F.group_norm`
    # in the FCOS tower refuses a 1-element map outright. That is a real constraint on any
    # export, not a test artefact: the smallest resolution this project ships is 384x512.
    "data": {
        "input_size": [128, 160],
        "datasets": [
            {
                "name": "coco",
                "type": "coco",
                "root": "datasets/coco",
                "split_train": "train2017",
                "supervises": ["detection"],
            },
            {
                "name": "ade",
                "type": "seg_folder",
                "root": "datasets/ADE20K",
                "split_train": "train",
                "supervises": ["traversability"],
            },
        ],
    },
    "train": {"epochs": 1, "batch_size": 1, "lr": 1e-4},
}


def _write_cfg(tmp_path, **overrides) -> str:
    cfg = json.loads(json.dumps(BASE))  # deep copy of plain data
    for dotted, value in overrides.items():
        node = cfg
        *path, leaf = dotted.split(".")
        for key in path:
            node = node[key]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def _main(*args: str) -> None:
    """`main` with the shape-only flag every test here needs, and the reason for it.

    Nothing in this file has a checkpoint -- `torch.onnx.export` is stubbed and the point
    is the decisions `main` makes, not the weights it makes them about. Since exporting
    with no `--checkpoint` is refused (it would emit the initialisation, which
    `--check-parity` cannot distinguish from a trained model), each call says so once,
    here, rather than thirteen times below.
    """
    export_onnx.main([*args, "--allow-random-weights"])


@pytest.fixture
def exported(monkeypatch):
    """Records what `main` decided, without serialising anything.

    `torch.onnx.export` writes a marker so `Path.exists()` on the output means "the
    export was reached", which is what the ordering tests below assert on. The `onnx`
    stub captures `metadata_props`, so the properties `main` attaches -- the contract a
    TensorRT engine does *not* carry -- are checked rather than assumed.
    """
    calls = SimpleNamespace(export=None, props={}, saved=0)

    def fake_export(model, dummy, path, **kw):
        calls.export = {"path": str(path), "model": model, "dummy": dummy, **kw}
        Path(path).write_text("not a real onnx graph")

    class _Props(list):
        def add(self):
            entry = SimpleNamespace(key=None, value=None)
            self.append(entry)
            return entry

    def fake_load(_path):
        return SimpleNamespace(metadata_props=_Props())

    def fake_save(model, _path):
        calls.saved += 1
        calls.props = {e.key: e.value for e in model.metadata_props}

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(
            load=fake_load, save=fake_save, checker=SimpleNamespace(check_model=lambda _m: None)
        ),
    )
    # onnxsim is optional in `main` and its absence is an accepted path; force it so the
    # test does not depend on whether the dev extra happens to be installed.
    monkeypatch.setitem(sys.modules, "onnxsim", None)
    return calls


# ----------------------------------------------------- the refusals, through main


@pytest.mark.usefixtures("exported")
def test_narrowing_a_model_with_no_detection_head_is_refused(tmp_path):
    # The coco dataset goes with it: a dataset supervising an undeclared head is a
    # config error in its own right, and this test is about the narrowing refusal.
    cfg = _write_cfg(
        tmp_path,
        **{
            "model.heads": {"traversability": BASE["model"]["heads"]["traversability"]},
            "data.datasets": [BASE["data"]["datasets"][1]],
        },
    )
    out = tmp_path / "m.onnx"
    with pytest.raises(SystemExit) as exc:
        _main("--config", cfg, "--output", str(out), "--detection-classes", "robot_8")
    assert "nothing to narrow" in str(exc.value)


@pytest.mark.usefixtures("exported")
def test_an_unknown_detection_subset_is_refused_naming_the_flag(tmp_path):
    cfg = _write_cfg(tmp_path)
    out = tmp_path / "m.onnx"
    with pytest.raises(SystemExit) as exc:
        _main("--config", cfg, "--output", str(out), "--detection-classes", "not-a-subset")
    assert str(exc.value).startswith("--detection-classes:"), (
        "the message must name the flag; a bare ValueError text leaves the operator "
        "guessing which argument was wrong"
    )


def test_an_unsupervised_head_is_refused_before_anything_is_written(tmp_path, exported):
    """`test_export_guard.py` proves `unsupervised_heads` finds it. This proves `main` acts.

    A head no dataset supervises is exported at its initial random weights and emits
    confident-looking numbers on the robot. The predicate having ten tests is not the
    same as the refusal happening.
    """
    cfg = _write_cfg(tmp_path, **{"data.datasets": [BASE["data"]["datasets"][0]]})
    out = tmp_path / "m.onnx"
    with pytest.raises(SystemExit):
        _main("--config", cfg, "--output", str(out))
    assert exported.export is None, "torch.onnx.export ran despite the refusal"
    assert not out.exists()


@pytest.mark.usefixtures("exported")
def test_a_refused_narrowing_leaves_no_output_file(tmp_path):
    """Ordering, the property that mattered in `sam3_prelabel` for the same reason.

    A refusal that happens after the write leaves a `.onnx` on disk that nobody meant to
    produce, and an `.onnx` on disk is indistinguishable from a successful export -- the
    next step is `trtexec`, which will happily build an engine from it.
    """
    cfg = _write_cfg(tmp_path)
    out = tmp_path / "m.onnx"
    with pytest.raises(SystemExit):
        _main("--config", cfg, "--output", str(out), "--detection-classes", "nope")
    assert not out.exists()
    assert not export_onnx.sidecar_path(out).exists()


# --------------------------------------------------------- the wiring, through main


def test_the_default_export_names_every_head_and_embeds_preprocessing(tmp_path, exported):
    cfg = _write_cfg(tmp_path)
    out = tmp_path / "m.onnx"
    _main("--config", cfg, "--output", str(out))

    names = exported.export["output_names"]
    assert any(n.startswith("det_cls_p") for n in names), names
    assert "traversability" in names
    assert exported.export["opset_version"] == 17
    assert exported.export["dynamo"] is False, (
        "the exporter must stay TorchScript-based here; the wrapper is not dynamo-safe"
    )
    # embedded preprocessing means the graph is fed 0-255, so the dummy must be too --
    # standard-normal noise would make a parity check pass on inputs no camera produces.
    assert exported.export["dummy"].max() > 1.5
    assert exported.props["input_range"] == "0-255"
    assert exported.props["preprocessing"] == "embedded"
    assert exported.props["segmentation_output"] == "logits"


def test_no_embed_preprocessing_changes_both_the_dummy_and_the_metadata(tmp_path, exported):
    cfg = _write_cfg(tmp_path)
    out = tmp_path / "m.onnx"
    _main("--config", cfg, "--output", str(out), "--no-embed-preprocessing")
    assert exported.export["dummy"].max() < 10.0
    assert exported.props["preprocessing"] == "external"
    assert exported.props["input_range"] == "imagenet-normalised"


def test_argmax_seg_reaches_the_wrapper_and_the_metadata(tmp_path, exported):
    """The flag is wired inside `main`, so nothing was checking that setting it works."""
    cfg = _write_cfg(tmp_path)
    out = tmp_path / "m.onnx"
    _main("--config", cfg, "--output", str(out), "--argmax-seg")
    names = exported.export["output_names"]
    assert any(n.endswith("_argmax") for n in names), names
    assert exported.props["segmentation_output"] == "class_ids_uint8"


def test_export_input_size_overrides_the_training_size(tmp_path, exported):
    """A TensorRT engine built for the wrong input size fails much later and far more
    confusingly, so the override existing is not enough -- it has to take effect."""
    cfg = _write_cfg(tmp_path, export={"input_size": [192, 256]})
    out = tmp_path / "m.onnx"
    _main("--config", cfg, "--output", str(out))
    assert tuple(exported.export["dummy"].shape[-2:]) == (192, 256)


# ------------------------------------------------------------------- the sidecar


def test_narrowing_writes_a_sidecar_naming_every_kept_class(tmp_path, exported):
    """A TensorRT engine keeps binding names and nothing else, so the ONNX metadata never
    reaches the board. The sidecar is the only record of what each channel means."""
    cfg = _write_cfg(tmp_path)
    out = tmp_path / "m.onnx"
    _main("--config", cfg, "--output", str(out), "--detection-classes", "robot_8")

    side = json.loads(export_onnx.sidecar_path(out).read_text())
    assert side["input_size"] == [128, 160]
    assert len(side["detection_classes"]) == len(side["source_indices"]) == 8
    assert len(side["cls_outputs"]) == 5, "one class-logit binding per FPN level"
    assert all(n.startswith("det_cls8_p") for n in side["cls_outputs"]), side["cls_outputs"]
    assert exported.props["detection_classes"] == ",".join(side["detection_classes"])


@pytest.mark.usefixtures("exported")
def test_the_sidecar_indices_point_back_into_the_trained_class_list(tmp_path):
    """`source_indices` is what lets someone map a narrowed channel back to the checkpoint
    it came from. Wrong indices are unrecoverable later and look like nothing at export."""
    from syncai_hydranet.data.coco_subsets import COCO_NAMES

    cfg = _write_cfg(tmp_path)
    out = tmp_path / "m.onnx"
    _main("--config", cfg, "--output", str(out), "--detection-classes", "robot_8")
    side = json.loads(export_onnx.sidecar_path(out).read_text())
    assert [COCO_NAMES[i] for i in side["source_indices"]] == side["detection_classes"]


def test_no_sidecar_when_nothing_was_narrowed(tmp_path, exported):
    cfg = _write_cfg(tmp_path)
    out = tmp_path / "m.onnx"
    _main("--config", cfg, "--output", str(out))
    assert not export_onnx.sidecar_path(out).exists()
    assert "detection_classes" not in exported.props


@pytest.mark.usefixtures("exported")
def test_every_narrowing_refusal_precedes_the_model_mutation(tmp_path, monkeypatch):
    """Stronger than "no file was written", and raised by `26251130` reviewing this file.

    `narrow_detection_head` mutates the model in place. Every `--detection-classes`
    refusal currently happens before it, so a refused run leaves neither a file nor a
    half-narrowed model -- but nothing enforces that order, and it is exactly the kind of
    invariant that decays when a later check is added at the bottom of the block rather
    than the top. Pinned per refusal rather than once, so adding a fourth refusal in the
    wrong place fails here.
    """
    mutated: list[bool] = []
    real = export_onnx.narrow_detection_head
    monkeypatch.setattr(
        export_onnx,
        "narrow_detection_head",
        lambda *a, **k: (mutated.append(True), real(*a, **k))[1],
    )
    out = tmp_path / "m.onnx"

    for cfg_kwargs, flag in (
        (
            {
                "model.heads": {"traversability": BASE["model"]["heads"]["traversability"]},
                "data.datasets": [BASE["data"]["datasets"][1]],
            },
            "robot_8",
        ),  # no detection head
        ({}, "not-a-subset"),  # unknown subset
    ):
        with pytest.raises(SystemExit):
            _main(
                "--config",
                _write_cfg(tmp_path, **cfg_kwargs),
                "--output",
                str(out),
                "--detection-classes",
                flag,
            )
        assert mutated == [], "the model was narrowed before the run was refused"
        assert not out.exists()

    # ...and the mutation does happen on the path that is not refused, so the assertion
    # above is not passing merely because the patch never took effect.
    _main(
        "--config",
        _write_cfg(tmp_path),
        "--output",
        str(out),
        "--detection-classes",
        "robot_8",
    )
    assert mutated == [True]


# ------------------------------------------------- the gate that could not fail


def test_parity_refuses_when_onnxruntime_is_absent_rather_than_reporting_a_pass():
    """A skipped acceptance gate must not exit 0 the way a passed one does.

    `check_parity` used to print a note and `return True` when the import failed, so
    `main`'s `if args.check_parity and not check_parity(...)` saw a pass, the command
    exited 0, and any wrapper reading `$?` was told the exported graph had been compared
    against PyTorch when nothing had run. `--check-parity` describes itself as "the
    deployment acceptance gate" in its own help.

    CI could never catch it: `ci.yml`'s export-parity job installs `--extra export`. The
    machine that would not have onnxruntime is the one an engine gets built on.

    `sys.modules[name] = None` is the documented way to make a later `import name` raise
    `ImportError`, so this exercises the real branch rather than a stubbed function.
    """
    with (
        pytest.MonkeyPatch.context() as mp,
        pytest.raises(SystemExit) as excinfo,
    ):
        mp.setitem(sys.modules, "onnxruntime", None)
        export_onnx.check_parity(object(), object(), "unused.onnx", ["terrain"])

    message = str(excinfo.value)
    assert "onnxruntime" in message
    assert "has NOT been compared" in message, "the refusal has to say what did not happen"
    assert "--extra export" in message, "and how to make it happen"


# --------------------------------------------- the export that had no weights in it


@pytest.mark.usefixtures("exported")
def test_exporting_without_a_checkpoint_is_refused(tmp_path):
    """The initialisation is not a model, and no artefact said which one it was.

    `main` loaded weights under `if args.checkpoint:` and did nothing otherwise, so
    omitting the flag exported the ImageNet-initialised network with random heads.
    Three things then failed to notice, in the order an operator would meet them:

    * `--check-parity` PASSES -- PyTorch and ONNX agree perfectly on random weights, so
      the acceptance gate is not merely silent, it is affirmative;
    * the ONNX `metadata_props` record preprocessing, input range, layout, channel order
      and class names, and nothing about which checkpoint produced the weights;
    * the sidecar records the class mapping and the input size, and is only written when
      `--detection-classes` narrowed something.

    So a random-weight engine and a trained one are indistinguishable from their
    artefacts, and the engine emits boxes that look like every other box.
    """
    out = tmp_path / "m.onnx"
    with pytest.raises(SystemExit) as exc:
        export_onnx.main(["--config", _write_cfg(tmp_path), "--output", str(out)])

    message = str(exc.value)
    assert "--checkpoint" in message, "the refusal has to name the flag that fixes it"
    assert "--allow-random-weights" in message, "and the escape hatch, for benchmarking"
    assert not out.exists(), "refused before anything was written"


@pytest.mark.usefixtures("exported")
def test_the_shape_only_escape_hatch_exports(tmp_path):
    """The refusal must not remove latency benchmarking, which legitimately has no run.

    Same shape as `--allow-untrained-heads`: the unsafe thing stays reachable and has to
    be asked for by name. `ci.yml`'s export-parity job is the standing consumer -- it
    exports every shipped config on random weights on purpose, because the graph is what
    it is checking.
    """
    out = tmp_path / "m.onnx"
    _main("--config", _write_cfg(tmp_path), "--output", str(out))
    assert out.exists()


# ------------------------------------------- which run these weights came from


def test_a_shape_only_export_says_so_in_the_artefact(tmp_path, exported):
    """`weights=random` is the marker that makes the escape hatch auditable.

    Refusing the unflagged case stops the accident; this stops the deliberate one being
    mistaken for a trained export later. A benchmarking ONNX is a legitimate artefact and
    it has to be one a reader can identify.
    """
    _main("--config", _write_cfg(tmp_path), "--output", str(tmp_path / "m.onnx"))
    assert exported.props["weights"] == "random"
    assert exported.props["checkpoint"] == "none"
    assert exported.props["checkpoint_sha256"] == "none"


def test_the_artefact_carries_the_checkpoint_it_was_exported_from(tmp_path, exported):
    """The digest is of the file, so `sha256sum` against it is the whole verification.

    Not a hash of the tensors: a reader holding an engine and a `runs/` directory can run
    one command and learn whether they match. Verified against the real CLI on
    `runs/hydranet_retail_pose02/best.pt`, where this string equals `sha256sum`'s.
    """
    import hashlib

    ckpt = tmp_path / "best.pt"
    torch.save({"model": {}, "ema": {}}, ckpt)
    expected = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    monkey = pytest.MonkeyPatch()
    monkey.setattr(export_onnx, "load_checkpoint", lambda _p: {"model": {}, "ema": {}})
    monkey.setattr(export_onnx, "select_weights", lambda ck, _prefer: ck["ema"])
    monkey.setattr(torch.nn.Module, "load_state_dict", lambda _self, _sd, **_kw: None)
    try:
        export_onnx.main(
            [
                "--config",
                _write_cfg(tmp_path),
                "--output",
                str(tmp_path / "m.onnx"),
                "--checkpoint",
                str(ckpt),
            ]
        )
    finally:
        monkey.undo()

    assert exported.props["checkpoint"] == str(ckpt)
    assert exported.props["checkpoint_sha256"] == expected
    assert exported.props["weights"] == "ema"


def test_weights_records_what_was_taken_not_what_was_asked(tmp_path, exported):
    """`--weights ema` against a checkpoint with no EMA silently gets the raw tensors.

    `select_weights` falls back on purpose -- and `utils/checkpoint.py` records why it
    matters: a run scoring 0.16 mIoU on EMA weights and 0.95 on the raw ones from the
    same training, because the average starts at the initialisation. On a short run the
    two are different models, so an artefact claiming the flag rather than the outcome
    would name the wrong one.
    """
    ckpt = tmp_path / "last.pt"
    torch.save({"model": {}}, ckpt)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(export_onnx, "load_checkpoint", lambda _p: {"model": {}})
    monkey.setattr(torch.nn.Module, "load_state_dict", lambda _self, _sd, **_kw: None)
    try:
        export_onnx.main(
            [
                "--config",
                _write_cfg(tmp_path),
                "--output",
                str(tmp_path / "m.onnx"),
                "--checkpoint",
                str(ckpt),
                "--weights",
                "ema",
            ]
        )
    finally:
        monkey.undo()

    assert exported.props["weights"] == "model", (
        "asked for ema, got the raw tensors because there is no ema in this checkpoint; "
        "the artefact must say which it actually holds"
    )
