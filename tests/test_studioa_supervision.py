import copy
import json

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.data.studioa_autolabel import Candidate, annotate, encode
from syncai_hydranet.data.studioa_review import digest, write_json
from syncai_hydranet.data.studioa_supervision import (
    CLASSES,
    IGNORE,
    StudioAPartialDataset,
    check_supervision,
    export_supervision,
    semantic_target,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)


def row(entity, box):
    mask = np.zeros((32, 40), dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return {"entity": entity, "segmentation": encode(mask)}


def test_unknown_conflicting_and_legacy_rack_pixels_are_not_negatives():
    data = {
        "image_size_px": [40, 32],
        "entities": [row("floor", (0, 0, 30, 25)), row("person", (10, 10, 20, 20))],
        "unresolved_candidates": [row("other_shelf", (0, 0, 5, 32))],
    }
    target, stats = semantic_target(data)
    assert target[5, 8] == CLASSES["floor"]
    assert target[15, 15] == IGNORE  # positive overlap, not arbitrary draw order
    assert target[5, 2] == IGNORE  # unresolved rack cannot become a floor negative
    assert target[31, 39] == IGNORE  # unlabelled is not background
    assert stats["pixels_by_class"]["display_cabinet"] == 0
    assert stats["positive_conflict_pixels"] == 100
    reverse = copy.deepcopy(data)
    reverse["entities"].reverse()
    np.testing.assert_array_equal(target, semantic_target(reverse)[0])


def test_same_class_overlap_survives_but_three_way_conflicts_stay_ignored():
    data = {
        "image_size_px": [40, 32],
        "unresolved_candidates": [],
        "entities": [row("floor", (0, 0, 20, 20)), row("floor", (5, 5, 25, 25))],
    }
    assert semantic_target(data)[0][8, 8] == CLASSES["floor"]
    data["entities"] += [row("person", (5, 5, 15, 15)), row("floor", (0, 0, 20, 20))]
    assert semantic_target(data)[0][8, 8] == IGNORE


def source_package(root):
    (root / "frames").mkdir(parents=True)
    (root / "images").mkdir()
    frames = []
    for store in ("Kaohsiung", "Taichung", "Tao-Hsin"):
        for index, split in enumerate(("train", "val", "test"), 1):
            identity = f"{len(frames):04d}-{store}-cam{index:02d}"
            name = f"images/{identity}.png"
            Image.new("RGB", (40, 32), color=(len(frames) * 20, 50, 80)).save(root / name)
            frames.append(
                {
                    "id": identity,
                    "image": name,
                    "image_sha256": digest(root / name),
                    "store": store,
                    "camera": f"{store}-cam{index:02d}",
                    "original_split": split,
                    "image_size_px": [40, 32],
                }
            )
    write_json(root / "job.json", {"frames": frames})
    outputs = {}
    for frame in frames:
        mask = np.zeros((32, 40), dtype=bool)
        mask[:24, :30] = True
        data = annotate([Candidate("floor", mask, 0.9)], mask.shape)
        data.update(
            frame_id=frame["id"],
            job_sha256=digest(root / "job.json"),
            image_sha256=frame["image_sha256"],
        )
        name = f"frames/{frame['id']}.json"
        write_json(root / name, data)
        outputs[name] = digest(root / name)
    write_json(
        root / "report.json",
        {"status": "completed", "outputs": outputs, "job_sha256": digest(root / "job.json")},
    )
    return frames


def test_real_export_reader_preserves_folds_source_labels_and_ignore(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    frames = source_package(source)
    original = (source / "frames" / (frames[0]["id"] + ".json")).read_bytes()
    export_supervision(source, out)
    manifest = check_supervision(out)
    assert manifest["folds"]["Taichung"]["counts"] == {"train": 2, "val": 2, "test": 3}
    ds = StudioAPartialDataset(out, "Taichung", "train", input_size=(64, 64))
    assert len(ds) == 2
    assert all(f["store"] != "Taichung" and f["original_split"] == "train" for f in ds.frames)
    assert set(ds[0]["targets"]["scene"].unique().tolist()) == {CLASSES["floor"], IGNORE}
    assert ds[0]["image"].shape == (3, 64, 64)
    assert (out / "companions" / (frames[0]["id"] + ".json")).read_bytes() == original
    assert (source / "frames" / (frames[0]["id"] + ".json")).read_bytes() == original
    with pytest.raises(FileExistsError):
        export_supervision(source, out)
    (out / ds.frames[0]["mask"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="output changed"):
        check_supervision(out)


def test_export_refuses_decoded_content_leakage_even_with_different_encoded_files(tmp_path):
    source = tmp_path / "source"
    frames = source_package(source)
    a, b = frames[0], frames[1]
    # New encoding metadata would also change the file hash; pixel identity is the gate.
    from PIL.PngImagePlugin import PngInfo

    metadata = PngInfo()
    metadata.add_text("different", "encoding")
    with Image.open(source / a["image"]) as image:
        image.save(source / b["image"], pnginfo=metadata)
    b["image_sha256"] = digest(source / b["image"])
    assert b["image_sha256"] != a["image_sha256"]
    write_json(source / "job.json", {"frames": frames})
    report = json.loads((source / "report.json").read_text())
    report["job_sha256"] = digest(source / "job.json")
    for frame in frames:
        name = f"frames/{frame['id']}.json"
        data = json.loads((source / name).read_text())
        data.update(image_sha256=frame["image_sha256"], job_sha256=report["job_sha256"])
        write_json(source / name, data)
        report["outputs"][name] = digest(source / name)
    write_json(source / "report.json", report)
    with pytest.raises(ValueError, match="identical image leakage"):
        export_supervision(source, tmp_path / "out")
    assert json.loads((tmp_path / "out/status.json").read_text())["status"] == "failed"


def test_factory_keeps_partial_ids_folds_and_cross_dataset_leak_checks(tmp_path):
    from syncai_hydranet.data.datasets import build_dataset, split_leaks
    from syncai_hydranet.data.fingerprint import fingerprint_dataset

    source, out = tmp_path / "source", tmp_path / "out"
    source_package(source)
    export_supervision(source, out)
    cfg = {
        "name": "partial",
        "type": "studioa_partial",
        "root": str(out),
        "held_out": "Tao-Hsin",
        "split_train": "train",
        "split_val": "val",
        "split_test": "test",
        "supervises": ["scene"],
    }
    dataset = build_dataset(cfg, (64, 64), "val", letterbox=True)
    assert dataset.supervises == ["scene"]
    assert set(dataset[0]["targets"]["scene"].unique().tolist()) == {0, 255}
    assert not split_leaks([cfg])
    assert split_leaks([cfg, {**cfg, "name": "other-fold", "held_out": "Taichung"}])
    fp = fingerprint_dataset(cfg)
    assert fp["manifest_sha256"] == digest(out / "manifest.json")
    assert len(fp["splits"]["train"]["frames"]) == 2
    with pytest.raises(ValueError, match="unchanged splits"):
        build_dataset({**cfg, "split_train": "test"}, (64, 64), letterbox=True)
    with pytest.raises(ValueError, match="evaluation partitions"):
        StudioAPartialDataset(out, "Tao-Hsin", "test", train=True)


def test_partial_data_trains_validates_checkpoints_and_resumes_existing_trainer(
    tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path

    import torch

    from syncai_hydranet.engine import trainer as trainer_module
    from syncai_hydranet.utils.checkpoint import load_checkpoint

    tool = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_train.py"
    spec = importlib.util.spec_from_file_location("studioa_pilot_test", tool)
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    source, out = tmp_path / "source", tmp_path / "out"
    source_package(source)
    export_supervision(source, out)
    cfg = worker.pilot_config(out, tmp_path / "pilot", check_supervision(out), "Tao-Hsin")
    cfg.update(device="cpu")
    cfg["model"]["backbone"].update(pretrained=False)
    cfg["data"].update(input_size=[32, 64], workers=0)
    cfg["train"].update(epochs=2, batch_size=2, amp=False, warmup_iters=0)
    monkeypatch.setattr(trainer_module, "_open_tensorboard", lambda _path: None)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        trainer = trainer_module.Trainer(cfg)
        trainer.train_one_epoch(1)
        trainer.record_epoch(1, trainer.validate(1))
        checkpoint = tmp_path / "pilot/model/last.pt"
        saved = load_checkpoint(checkpoint)
        assert saved["epoch"] == 1 and saved["global_step"] == 1
        resumed = trainer_module.Trainer(cfg, resuming=True)
        resumed.load(str(checkpoint), resume=True)
        resumed.train()
        saved = load_checkpoint(checkpoint)
        assert saved["epoch"] == 2 and saved["global_step"] == 2
        assert (tmp_path / "pilot/model/best.pt").exists()
        assert all(f["store"] != "Tao-Hsin" for f in resumed.val_sets[0][1].frames)
    finally:
        torch.set_num_threads(threads)
