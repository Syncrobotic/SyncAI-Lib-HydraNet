"""Dataset construction and split handling.

Builds a tiny on-disk segmentation dataset, so these run in CI without downloads.

pytest tests/test_datasets.py -v
"""

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.data.datasets import SPLITS, build_dataset, resolve_split
from syncai_hydranet.data.transforms import LetterboxResize, LetterboxScaleCrop

ADE_FLOOR, ADE_WALL = 4, 1  # ids from objectInfo150.txt
SIZE = (128, 160)


@pytest.fixture
def dataset_root(tmp_path):
    """images/<split>/*.jpg plus annotations/<split>/*.png, as SegFolderDataset expects."""
    for split in ("train", "val", "test"):
        img_dir = tmp_path / "images" / split
        ann_dir = tmp_path / "annotations" / split
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        for i in range(2):
            rgb = np.zeros((90, 160, 3), np.uint8)
            rgb[45:] = 200  # a floor-ish lower half
            Image.fromarray(rgb).save(img_dir / f"{split}_{i}.jpg")
            ann = np.full((90, 160), ADE_WALL, np.uint8)
            ann[45:] = ADE_FLOOR
            Image.fromarray(ann).save(ann_dir / f"{split}_{i}.png")
    return tmp_path


def _cfg(root, **extra):
    return {
        "name": "tiny",
        "type": "seg_folder",
        "root": str(root),
        "split_train": "train",
        "split_val": "val",
        "supervises": ["traversability", "terrain"],
        "label_map": "ade20k_indoor",
        **extra,
    }


# ----------------------------------------------------------------- splits


def test_split_names_are_fixed():
    assert SPLITS == ("train", "val", "test")


def test_resolve_split_maps_to_the_configured_folder(dataset_root):
    cfg = _cfg(dataset_root, split_test="holdout")
    assert resolve_split(cfg, "train") == "train"
    assert resolve_split(cfg, "val") == "val"
    assert resolve_split(cfg, "test") == "holdout"


def test_missing_test_split_says_what_to_do(dataset_root):
    """Not defaulting to val is the point: a test split that silently aliases the
    selection split reports the same optimistic number under a more trustworthy name."""
    with pytest.raises(KeyError, match="split_test"):
        resolve_split(_cfg(dataset_root), "test")


def test_unknown_split_is_rejected(dataset_root):
    with pytest.raises(ValueError, match="unknown split"):
        resolve_split(_cfg(dataset_root), "validation")


def test_test_split_loads_when_configured(dataset_root):
    ds = build_dataset(_cfg(dataset_root, split_test="test"), SIZE, "test")
    assert len(ds) == 2


# ---------------------------------------------------------------- datasets


def test_only_the_train_split_is_augmented(dataset_root):
    """Augmenting evaluation data would make the score depend on the dice roll."""
    cfg = _cfg(dataset_root, split_test="test")
    train = build_dataset(cfg, SIZE, "train", letterbox=True)
    assert isinstance(train.transform.ts[0], LetterboxScaleCrop)
    for split in ("val", "test"):
        ds = build_dataset(cfg, SIZE, split, letterbox=True)
        assert isinstance(ds.transform.ts[0], LetterboxResize)


def test_sample_carries_both_heads_at_input_resolution(dataset_root):
    ds = build_dataset(_cfg(dataset_root), SIZE, "val", letterbox=True)
    sample = ds[0]
    assert sample["image"].shape == (3, *SIZE)
    assert sample["targets"]["terrain"].shape == SIZE
    assert sample["targets"]["traversability"].shape == SIZE
    assert sample["supervises"] == ["traversability", "terrain"]


def test_one_annotation_produces_both_terrain_and_traversability(dataset_root):
    """The traversability labels are derived from the terrain ones via the scheme, so a
    single annotation pass supervises two heads."""
    ds = build_dataset(_cfg(dataset_root), SIZE, "val")
    sample = ds[0]
    terrain = sample["targets"]["terrain"].numpy()
    trav = sample["targets"]["traversability"].numpy()
    assert set(np.unique(terrain)) <= set(range(12)) | {255}
    assert set(np.unique(trav)) <= {0, 1, 2, 255}
    # Wall is never walkable, floor always is; both appear in the fixture.
    assert (trav == 0).any() and (trav == 2).any()


def test_letterbox_padding_is_ignore_labelled(dataset_root):
    """The fixture is 16:9 into a 4:5 canvas, so there must be padding, and it must not
    be counted as any real class."""
    ds = build_dataset(_cfg(dataset_root), SIZE, "val", letterbox=True)
    assert (ds[0]["targets"]["terrain"].numpy() == 255).any()


def test_missing_directory_names_the_expected_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        build_dataset(_cfg(tmp_path), SIZE, "train")


def test_sessions_with_the_same_filenames_are_all_kept(tmp_path):
    """Every per-session numbering scheme produces `0000.png` more than once.

    Keyed on the bare stem, the second session's frames overwrite the first's in the
    lookup and the loader trains on a third of what was annotated -- with no error, and
    with the run's meta.json reporting the smaller dataset size as though it were right.
    `hydranet-annotation check` reported 60 masks with no counterpart on a three-session
    batch that had one for every frame, which is how this was found.
    """
    from syncai_hydranet.data.datasets import _index_pairs

    img_dir, ann_dir = tmp_path / "images", tmp_path / "annotations"
    for session in ("cam_a", "cam_b", "cam_c"):
        (img_dir / session).mkdir(parents=True)
        (ann_dir / session).mkdir(parents=True)
        for i in range(4):
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(
                img_dir / session / f"{i:04d}.jpg"
            )
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(ann_dir / session / f"{i:04d}.png")

    pairs = _index_pairs(img_dir, ann_dir)
    assert len(pairs) == 12, f"expected every frame, got {len(pairs)}"
    for img, ann in pairs:  # and each is paired with its own session's mask
        assert img.parent.name == ann.parent.name


def test_a_flat_dataset_still_pairs_by_stem(tmp_path):
    """ADE20K and RUGD have no session directories; the relative path is the stem there."""
    from syncai_hydranet.data.datasets import _index_pairs

    img_dir, ann_dir = tmp_path / "images", tmp_path / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(img_dir / "x.jpg")
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(ann_dir / "x.png")
    assert len(_index_pairs(img_dir, ann_dir)) == 1


def test_differently_nested_trees_fall_back_to_stem_matching(tmp_path):
    """Images in a subdirectory, masks flat: a real layout, and it must keep working."""
    from syncai_hydranet.data.datasets import _index_pairs

    img_dir, ann_dir = tmp_path / "images", tmp_path / "annotations"
    (img_dir / "nested").mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(img_dir / "nested" / "y.jpg")
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(ann_dir / "y.png")
    assert len(_index_pairs(img_dir, ann_dir)) == 1
