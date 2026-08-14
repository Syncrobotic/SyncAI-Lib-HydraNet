"""The annotation gate has to fail on the mistakes that are otherwise invisible.

Every case below is a dataset that trains without complaint and produces a plausible
number: a background exported as 0, a mask saved as RGB, a session that appears in two
splits. If `hydranet-annotation check` ever stops catching one of these, the failure
mode returns to being silent.

pytest tests/test_annotation.py -v
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.cli import annotation
from syncai_hydranet.cli.annotation import label_spec, main
from syncai_hydranet.data.label_maps_indoor import INDOOR_TERRAIN


def write_frame(
    root: Path,
    split: str,
    session: str,
    stem: str,
    mask: np.ndarray,
    *,
    mask_mode: str = "L",
    image_size: tuple[int, int] | None = None,
    write_mask: bool = True,
    write_image: bool = True,
) -> None:
    """Write one image/annotation pair into the seg_folder layout."""
    h, w = mask.shape
    img_dir = root / "images" / split / session
    ann_dir = root / "annotations" / split / session
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    if write_image:
        size = image_size or (w, h)
        Image.new("RGB", size, (128, 128, 128)).save(img_dir / f"{stem}.jpg")
    if write_mask:
        if mask_mode == "RGB":
            Image.fromarray(np.stack([mask] * 3, axis=-1).astype(np.uint8), "RGB").save(
                ann_dir / f"{stem}.png"
            )
        else:
            Image.fromarray(mask.astype(np.uint8), mask_mode).save(ann_dir / f"{stem}.png")


def good_mask(h: int = 8, w: int = 8) -> np.ndarray:
    """Half floor_hard, half wall, with the unlabelled remainder set to ignore."""
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[: h // 2] = INDOOR_TERRAIN["floor_hard"]
    mask[h // 2 : h - 1] = INDOOR_TERRAIN["wall"]
    return mask


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "site"
    write_frame(root, "train", "lobby-a", "000001", good_mask())
    write_frame(root, "train", "lobby-a", "000002", good_mask())
    write_frame(root, "val", "lobby-b", "000001", good_mask())
    write_frame(root, "val", "corridor-c", "000002", good_mask())
    return root


def check(root: Path, *extra: str) -> int:
    return main(["check", str(root), *extra])


# ------------------------------------------------------------------ label spec


def test_label_spec_covers_every_class_except_void():
    spec = label_spec()
    assert {e["name"] for e in spec} == set(INDOOR_TERRAIN) - {"void"}
    assert all(e["type"] == "mask" for e in spec)


def test_label_colours_are_unique():
    """Two classes sharing a colour is unreviewable, and unrecoverable after a mask
    export that carries the class only as a colour."""
    colours = [e["color"] for e in label_spec()]
    assert len(set(colours)) == len(colours)


def test_label_ids_match_the_training_scheme():
    for entry in label_spec():
        assert entry["terrain_id"] == INDOOR_TERRAIN[entry["name"]]


def test_labels_command_writes_importable_json(tmp_path: Path):
    import json

    out = tmp_path / "labels.json"
    assert main(["labels", "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == label_spec()


# ----------------------------------------------------------------- the happy path


def test_a_clean_dataset_passes(dataset, capsys):
    assert check(dataset) == 0
    assert "Contract checks passed" in capsys.readouterr().out


def test_coverage_reports_the_classes_that_are_absent(dataset, capsys):
    check(dataset)
    out = capsys.readouterr().out
    assert "glass" in out and "priority" in out


# -------------------------------------------------------------- the silent errors


def test_background_left_as_zero_fails_when_zero_is_a_trained_class(
    dataset, capsys, monkeypatch
):
    """The check tracks the label map rather than restating it.

    `indoor_native` sends 0 to ignore, so a background of zeros is harmless today. Point
    it back at an identity map -- what any new scheme would be written as, and what this
    one was -- and the same dataset has to be rejected, because then every unlabelled
    pixel teaches the model to predict `void` there.
    """
    monkeypatch.setattr(annotation, "INDOOR_NATIVE_ID", {v: v for v in INDOOR_TERRAIN.values()})
    mask = good_mask()
    mask[mask == 255] = 0
    write_frame(dataset, "train", "lobby-a", "000003", mask)
    assert check(dataset) == 1
    assert "void" in capsys.readouterr().out


def test_background_as_zero_can_be_accepted_explicitly(dataset, monkeypatch):
    monkeypatch.setattr(annotation, "INDOOR_NATIVE_ID", {v: v for v in INDOOR_TERRAIN.values()})
    mask = good_mask()
    mask[mask == 255] = 0
    write_frame(dataset, "train", "lobby-a", "000003", mask)
    assert check(dataset, "--allow-void") == 0


def test_zero_and_255_are_both_counted_as_unlabelled(dataset, capsys):
    """Under the shipped map both mean "nobody labelled this", and the coverage table
    reports them together -- a batch that comes back mostly ignore is a batch that
    mostly did not happen, and the per-class shares alone will not show it."""
    mask = good_mask()
    mask[mask == 255] = 0
    write_frame(dataset, "train", "lobby-a", "000003", mask)
    assert check(dataset) == 0
    assert "unlabelled (ignored by the loss" in capsys.readouterr().out


def test_a_mostly_unlabelled_batch_warns(tmp_path: Path, capsys):
    """Forgetting the background and leaving it blank on purpose are the same pixels, so
    this cannot be an error -- but it is worth saying, because the per-class shares are
    shares of what was labelled and stay healthy-looking either way."""
    root = tmp_path / "sparse"
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[0] = INDOOR_TERRAIN["glass"]
    for split, session in (("train", "lobby-a"), ("val", "lobby-b")):
        write_frame(root, split, session, "0001", mask)
    assert check(root) == 0
    assert "% of pixels are ignore" in capsys.readouterr().out


def test_rgb_mask_fails(dataset, capsys):
    write_frame(dataset, "train", "lobby-a", "000003", good_mask(), mask_mode="RGB")
    assert check(dataset) == 1
    assert "single channel" in capsys.readouterr().out


def test_unknown_class_id_fails(dataset, capsys):
    mask = good_mask()
    mask[0, 0] = 42
    write_frame(dataset, "train", "lobby-a", "000003", mask)
    assert check(dataset) == 1
    assert "42" in capsys.readouterr().out


def test_mask_image_size_mismatch_fails(dataset, capsys):
    write_frame(dataset, "train", "lobby-a", "000003", good_mask(), image_size=(16, 16))
    assert check(dataset) == 1
    assert "geometrically wrong" in capsys.readouterr().out


def test_unpaired_image_fails(dataset, capsys):
    """The loader drops it silently, so a third of an annotation run can go untrained."""
    write_frame(dataset, "train", "lobby-a", "000003", good_mask(), write_mask=False)
    assert check(dataset) == 1
    assert "no counterpart" in capsys.readouterr().out


def test_unpaired_annotation_fails(dataset, capsys):
    write_frame(dataset, "train", "lobby-a", "000003", good_mask(), write_image=False)
    assert check(dataset) == 1
    assert "no counterpart" in capsys.readouterr().out


# ------------------------------------------------------------------ split hygiene


def test_a_session_in_two_splits_fails(dataset, capsys):
    """The leak that looks like success: near-identical frames on both sides."""
    write_frame(dataset, "val", "lobby-a", "000009", good_mask())
    assert check(dataset) == 1
    assert "split by session" in capsys.readouterr().out


def test_flat_layout_warns_that_sessions_cannot_be_checked(tmp_path: Path, capsys):
    root = tmp_path / "flat"
    for split in ("train", "val"):
        write_frame(root, split, ".", "0001", good_mask())
        write_frame(root, split, ".", "0002", good_mask())
    assert check(root) == 0
    assert "session discipline cannot be checked" in capsys.readouterr().out


def test_single_session_val_warns(tmp_path: Path, capsys):
    root = tmp_path / "one"
    write_frame(root, "train", "lobby-a", "0001", good_mask())
    write_frame(root, "val", "lobby-b", "0001", good_mask())
    assert check(root) == 0
    assert "single session" in capsys.readouterr().out


def test_empty_split_fails(tmp_path: Path):
    root = tmp_path / "empty"
    write_frame(root, "train", "lobby-a", "0001", good_mask())
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations" / "val").mkdir(parents=True)
    assert check(root) == 1
