"""The four commands a user actually types, end to end.

Every other test file exercises the internals; these had **no coverage at all**, which
is the wrong way round. A mistyped argument name, a `--set` path that stopped resolving,
a checkpoint key that got renamed -- none of it would fail a test, and all of it would
fail on the first command someone ran.

Each test runs the real `main(argv)` against a tiny on-disk dataset and a config built
in `tmp_path`, so nothing is downloaded and nothing touches the repo. They are slow by
unit-test standards (a few seconds each: a real model, real optimiser steps) and worth it
-- this is the layer where a break is embarrassing rather than subtle.

pytest tests/test_cli_smoke.py -v
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from syncai_hydranet.cli import evaluate as eval_cli
from syncai_hydranet.cli import infer_image, infer_video, scene
from syncai_hydranet.cli import train as train_cli

ADE_FLOOR, ADE_WALL = 4, 1  # ids from objectInfo150.txt
FRAME = (64, 80)  # H, W -- small enough that a real training step is quick


@pytest.fixture
def dataset(tmp_path) -> Path:
    """A four-frame segmentation dataset in the seg_folder layout."""
    root = tmp_path / "tiny"
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        for i in range(4):
            rgb = np.zeros((*FRAME, 3), np.uint8)
            rgb[FRAME[0] // 2 :] = 200
            Image.fromarray(rgb).save(img_dir / f"{i}.jpg")
            ann = np.full(FRAME, ADE_WALL, np.uint8)
            ann[FRAME[0] // 2 :] = ADE_FLOOR
            Image.fromarray(ann).save(ann_dir / f"{i}.png")
    return root


@pytest.fixture
def config(tmp_path, dataset) -> Path:
    """A two-head config with no pretrained download and no detection dataset.

    `pretrained: false` matters beyond speed: a test that reaches the network is a test
    that fails when the network does.
    """
    cfg = {
        "experiment": "smoke",
        "output_dir": str(tmp_path / "run"),
        "seed": 0,
        "device": "cpu",
        "model": {
            "backbone": {"name": "resnet18", "pretrained": False},
            "neck": {"name": "fpn", "out_channels": 32, "num_levels": 3},
            "heads": {
                "traversability": {
                    "type": "semantic_fpn",
                    "num_classes": 3,
                    "in_levels": [0, 1, 2],
                    "channels": 16,
                },
                "terrain": {
                    "type": "semantic_fpn",
                    "num_classes": 12,
                    "in_levels": [0, 1, 2],
                    "channels": 16,
                },
            },
        },
        "data": {
            "input_size": list(FRAME),
            "terrain_classes": ["void", "floor_hard"] + [f"c{i}" for i in range(2, 12)],
            "datasets": [
                {
                    "name": "tiny",
                    "type": "seg_folder",
                    "root": str(dataset),
                    "split_train": "train",
                    "split_val": "val",
                    "split_test": "test",
                    "label_map": "ade20k_indoor",
                    "supervises": ["traversability", "terrain"],
                    "sample_ratio": 1.0,
                }
            ],
            "workers": 0,
        },
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "warmup_iters": 1,
            "ema": False,
            "amp": False,
            "log_interval": 1,
        },
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def trained(config) -> Path:
    """Run training once; the other CLIs need a checkpoint to load.

    `--allow-dirty` because the suite runs against a working checkout, which is dirty
    whenever anyone is mid-change -- and on a shared checkout that is most of the time.
    The gate it bypasses is about releasability, not correctness, and a smoke run is
    never released. A test that failed on the state of someone else's editor would be
    measuring the wrong thing.
    """
    train_cli.main(["--config", str(config), "--allow-dirty"])
    return Path(yaml.safe_load(config.read_text())["output_dir"])


# ------------------------------------------------------------------ argument parsing


@pytest.mark.parametrize(
    "module", [train_cli, eval_cli, infer_image, infer_video, scene], ids=lambda m: m.__name__
)
def test_help_builds(module, capsys):
    """--help exercises every argument definition, and costs milliseconds."""
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    assert module.build_parser().prog in capsys.readouterr().out


@pytest.mark.parametrize("module", [train_cli, eval_cli], ids=lambda m: m.__name__)
def test_missing_required_argument_exits(module):
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code == 2


# -------------------------------------------------------------------------- training


def test_train_writes_everything_a_run_needs(trained):
    """The four artefacts every downstream tool reads. A run missing any of them is
    unreportable, unresumable, or unreproducible."""
    for name in ("best.pt", "last.pt", "meta.json", "metrics.jsonl", "train.log"):
        assert (trained / name).is_file(), name

    meta = json.loads((trained / "meta.json").read_text())
    assert meta["config"]["experiment"] == "smoke"
    assert meta["datasets"][0]["train_size"] == 4

    rows = [json.loads(x) for x in (trained / "metrics.jsonl").read_text().splitlines()]
    assert rows and rows[-1]["epoch"] == 1
    assert "traversability_mIoU" in rows[-1]


def test_set_overrides_reach_the_run(config, tmp_path):
    """`--set` is the whole interface for experiments. If a dot-path silently fails to
    resolve, the run records the value it was asked for and trains on another."""
    out = tmp_path / "override"
    train_cli.main(
        [
            "--allow-dirty",
            "--config",
            str(config),
            "--set",
            f"output_dir={out}",
            "train.batch_size=1",
            "train.lr=0.002",
        ]
    )
    meta = json.loads((out / "meta.json").read_text())
    assert meta["config"]["train"]["batch_size"] == 1
    assert meta["config"]["train"]["lr"] == 0.002


def test_resume_continues_rather_than_restarting(config, trained):
    """Resuming must advance the epoch counter, not replay epoch 1 at full LR."""
    train_cli.main(
        [
            "--allow-dirty",
            "--config",
            str(config),
            "--set",
            "train.epochs=2",
            "--resume",
            str(trained / "last.pt"),
        ]
    )
    rows = [json.loads(x) for x in (trained / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2]


# ------------------------------------------------------------------------ evaluation


def test_eval_reports_and_writes_json(config, trained, tmp_path):
    out = tmp_path / "metrics.json"
    eval_cli.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(trained / "best.pt"),
            "--weights",
            "model",
            "--json",
            str(out),
        ]
    )
    record = json.loads(out.read_text())
    assert record["split"] == "val"
    assert "traversability_mIoU" in record
    assert "git" in record, "a metric without its commit is not comparable to anything"
    # The same argument applied to what was measured rather than to which code measured
    # it. `--set` can redefine the datasets entirely -- a detection mAP over 25
    # categories and one over 80 serialise to the same key -- so the record has to carry
    # the definition, not just the number.
    assert record["config"] == str(config)
    assert record["set"] == []
    assert [d["name"] for d in record["datasets"]]


def test_eval_can_read_the_test_split(config, trained, tmp_path):
    """The split that decides released numbers. If `--split test` breaks, the honest
    number is the one nobody can produce."""
    out = tmp_path / "test.json"
    eval_cli.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(trained / "best.pt"),
            "--weights",
            "model",
            "--split",
            "test",
            "--json",
            str(out),
        ]
    )
    assert json.loads(out.read_text())["split"] == "test"


# ------------------------------------------------------------------------- inference


def test_infer_image_writes_an_overlay(config, trained, dataset, tmp_path):
    out = tmp_path / "pred"
    infer_image.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(trained / "best.pt"),
            "--input",
            str(dataset / "images" / "val" / "0.jpg"),
            "--output",
            str(out),
        ]
    )
    written = list(out.glob("*_pred.jpg"))
    assert len(written) == 1
    with Image.open(written[0]) as im:
        assert im.size[0] >= FRAME[1], "the overlay should be at least the frame's width"


def test_infer_image_accepts_a_directory(config, trained, dataset, tmp_path):
    out = tmp_path / "batch"
    infer_image.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(trained / "best.pt"),
            "--input",
            str(dataset / "images" / "val"),
            "--output",
            str(out),
        ]
    )
    assert len(list(out.glob("*_pred.jpg"))) == 4


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg"
)
def test_infer_video_writes_a_video(config, trained, tmp_path):
    src = tmp_path / "clip.mp4"
    import subprocess

    lavfi = f"testsrc=size={FRAME[1]}x{FRAME[0]}:rate=5:duration=1"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", lavfi, str(src)], check=True
    )
    out = tmp_path / "pred.mp4"
    infer_video.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(trained / "best.pt"),
            "--input",
            str(src),
            "--output",
            str(out),
            "--max-frames",
            "3",
        ]
    )
    assert out.is_file() and out.stat().st_size > 0


# ----------------------------------------------------------------------------- scene


def test_scene_writes_a_panel_and_a_payload(config, trained, dataset, tmp_path):
    """The geometry module's entry point: a frame in, a picture and metres out."""
    png = tmp_path / "scene.png"
    doc = tmp_path / "scene.json"
    scene.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(trained / "best.pt"),
            "--weights",
            "model",
            "--input",
            str(dataset / "images" / "val" / "0.jpg"),
            "--output",
            str(png),
            "--json",
            str(doc),
        ]
    )
    assert png.is_file() and png.stat().st_size > 0
    payload = json.loads(doc.read_text())
    assert payload["grid"]["z_max"] == 9.0
    assert payload["plane"]["height_m"] == 1.5
    # The panel is only honest while it says the pose was assumed rather than measured.
    assert payload["pose_is_assumed"] is True
    assert 0.0 <= payload["known_fraction"] <= 1.0


def test_scene_refuses_to_do_nothing(config, trained, dataset):
    """Neither --output nor --json means the run would compute a scene and drop it."""
    with pytest.raises(SystemExit) as exc:
        scene.main(
            [
                "--config",
                str(config),
                "--checkpoint",
                str(trained / "best.pt"),
                "--input",
                str(dataset / "images" / "val" / "0.jpg"),
            ]
        )
    assert exc.value.code != 0


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg"
)
def test_scene_over_a_clip_writes_one_payload_per_frame(config, trained, tmp_path):
    """JSON lines, not an array: a reader that wants frame 0 should not load the clip."""
    import subprocess

    src = tmp_path / "clip.mp4"
    lavfi = f"testsrc=size={FRAME[1]}x{FRAME[0]}:rate=5:duration=2"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", lavfi, str(src)], check=True
    )
    out = tmp_path / "bev.mp4"
    doc = tmp_path / "scenes.jsonl"
    scene.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(trained / "best.pt"),
            "--weights",
            "model",
            "--input",
            str(src),
            "--output",
            str(out),
            "--json",
            str(doc),
            "--max-frames",
            "3",
        ]
    )
    assert out.is_file() and out.stat().st_size > 0
    rows = [json.loads(line) for line in doc.read_text().splitlines()]
    assert len(rows) == 3
    assert [r["frame"] for r in rows] == [0, 1, 2]
