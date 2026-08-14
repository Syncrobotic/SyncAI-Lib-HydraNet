"""The script that decides which frames every downstream number is computed from.

It had 21% coverage: `_is_test` was tested thoroughly (tests/test_test_split.py) and
everything around it -- the filter that picks the frames, the split writer, the
re-run behaviour -- was not. Since its output *is* the dataset, a change here moves
every mIoU in the project without touching a line of model code.

pytest tests/test_prepare_ade20k.py -v
"""

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.cli.prepare_ade20k import (
    FLOOR_IDS,
    SKY_ID,
    VEGETATION_IDS,
    build_parser,
    load_scene_categories,
    main,
    score,
)

H, W = 16, 16


def write_ann(path, spec):
    """One annotation image. `spec` maps an ADE20K class id to a pixel count."""
    flat = np.zeros(H * W, dtype=np.uint8)
    i = 0
    for class_id, count in spec.items():
        flat[i : i + count] = class_id
        i += count
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(flat.reshape(H, W)).save(path)


# ------------------------------------------------------------------- the filter


def test_score_returns_the_three_fractions(tmp_path):
    p = tmp_path / "a.png"
    write_ann(p, {FLOOR_IDS[0]: 64, SKY_ID: 32, VEGETATION_IDS[0]: 16})
    floor, sky, veg = score(p)
    assert floor == pytest.approx(0.25)
    assert sky == pytest.approx(0.125)
    assert veg == pytest.approx(0.0625)


def test_both_floor_ids_count_as_floor(tmp_path):
    """floor and rug. A rug is walkable, and an indoor frame can be mostly rug."""
    p = tmp_path / "b.png"
    write_ann(p, {FLOOR_IDS[0]: 64, FLOOR_IDS[1]: 64})
    assert score(p)[0] == pytest.approx(0.5)


def test_every_vegetation_id_counts(tmp_path):
    p = tmp_path / "c.png"
    write_ann(p, dict.fromkeys(VEGETATION_IDS, 16))
    assert score(p)[2] == pytest.approx(16 * len(VEGETATION_IDS) / (H * W))


def test_an_rgb_annotation_reads_its_first_channel(tmp_path):
    """PNGs saved as RGB do occur. Reading all three channels would divide every
    fraction by three and silently reject almost everything."""
    p = tmp_path / "rgb.png"
    flat = np.zeros((H, W), dtype=np.uint8)
    flat.reshape(-1)[:64] = FLOOR_IDS[0]
    Image.fromarray(np.stack([flat] * 3, axis=-1)).save(p)
    assert score(p)[0] == pytest.approx(0.25)


def test_an_empty_annotation_does_not_divide_by_zero(tmp_path):
    p = tmp_path / "void.png"
    Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(p)
    assert score(p) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- scene lookup


def test_missing_scene_file_is_not_an_error(tmp_path):
    """It is optional metadata used only for a printed summary; the dataset must still
    build without it."""
    assert load_scene_categories(tmp_path) == {}


def test_scene_categories_are_parsed(tmp_path):
    (tmp_path / "sceneCategories.txt").write_text(
        "ADE_train_00000001 airport_terminal\nADE_train_00000002 bathroom\n"
    )
    assert load_scene_categories(tmp_path) == {
        "ADE_train_00000001": "airport_terminal",
        "ADE_train_00000002": "bathroom",
    }


def test_short_rows_are_skipped(tmp_path):
    (tmp_path / "sceneCategories.txt").write_text("only_a_stem\nADE_x kitchen\n\n")
    assert load_scene_categories(tmp_path) == {"ADE_x": "kitchen"}


# ------------------------------------------------------------------ the parser


def test_parser_defaults_are_the_documented_ones():
    args = build_parser().parse_args(["--src", "s", "--dst", "d"])
    assert (args.min_floor, args.max_sky, args.max_vegetation) == (0.08, 0.02, 0.05)
    assert args.test_fraction == 0.0
    assert args.workers == 0


# -------------------------------------------------------------- the whole run


@pytest.fixture
def source(tmp_path):
    """A source tree with one clearly-indoor and one clearly-outdoor frame per split,
    plus an annotation whose image is missing."""
    src = tmp_path / "ADEChallengeData2016"
    for ade_split in ("training", "validation"):
        for stem, spec in {
            f"{ade_split}_indoor": {FLOOR_IDS[0]: 128},
            f"{ade_split}_outdoor": {SKY_ID: 128, FLOOR_IDS[0]: 64},
            f"{ade_split}_garden": {FLOOR_IDS[0]: 128, VEGETATION_IDS[0]: 64},
            f"{ade_split}_orphan": {FLOOR_IDS[0]: 128},
        }.items():
            write_ann(src / "annotations" / ade_split / f"{stem}.png", spec)
            if not stem.endswith("orphan"):
                d = src / "images" / ade_split
                d.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (W, H)).save(d / f"{stem}.jpg")
    return src


def names(root, kind, split):
    d = root / kind / split
    return {p.name for p in d.iterdir()} if d.is_dir() else set()


def test_it_keeps_indoor_and_drops_sky_and_vegetation(source, tmp_path):
    dst = tmp_path / "out"
    main(["--src", str(source), "--dst", str(dst), "--workers", "1"])
    assert names(dst, "images", "train") == {"training_indoor.jpg"}
    assert names(dst, "images", "val") == {"validation_indoor.jpg"}


def test_an_annotation_without_its_image_is_skipped(source, tmp_path):
    """A dangling symlink would fail at training time, thousands of steps later."""
    dst = tmp_path / "out"
    main(["--src", str(source), "--dst", str(dst), "--workers", "1"])
    assert not [n for n in names(dst, "images", "train") if "orphan" in n]


def test_outputs_are_symlinks_that_resolve(source, tmp_path):
    dst = tmp_path / "out"
    main(["--src", str(source), "--dst", str(dst), "--workers", "1"])
    for kind in ("images", "annotations"):
        for p in (dst / kind / "train").iterdir():
            assert p.is_symlink() and p.resolve().is_file()


def test_thresholds_are_honoured(source, tmp_path):
    """Raising min_floor above every frame's floor fraction must keep nothing --
    the filter, not a fixed frame list, is what decides."""
    dst = tmp_path / "out"
    main(["--src", str(source), "--dst", str(dst), "--min-floor", "0.99", "--workers", "1"])
    assert names(dst, "images", "train") == set()


def test_only_validation_is_ever_divided(source, tmp_path):
    dst = tmp_path / "out"
    main(["--src", str(source), "--dst", str(dst), "--test-fraction", "1.0", "--workers", "1"])
    assert names(dst, "images", "train") == {"training_indoor.jpg"}
    assert names(dst, "images", "val") == set()
    assert names(dst, "images", "test") == {"validation_indoor.jpg"}


def test_rerunning_replaces_rather_than_accumulates(source, tmp_path):
    dst = tmp_path / "out"
    main(["--src", str(source), "--dst", str(dst), "--workers", "1"])
    main(["--src", str(source), "--dst", str(dst), "--workers", "1"])
    assert names(dst, "images", "val") == {"validation_indoor.jpg"}


def test_rerunning_without_test_fraction_does_not_put_test_back_into_val(source, tmp_path):
    """The regression this file was written for.

    `--test-fraction` defaults to 0, so omitting it on a re-run is the easy mistake.
    The old code cleared only the directories it was about to write, so the previous
    test split survived while val was rebuilt over every kept frame -- and the held-out
    images ended up in both. Nothing warns you; the test split simply stops being
    held out, and every number computed on it is quietly contaminated by selection.
    """
    dst = tmp_path / "out"
    main(["--src", str(source), "--dst", str(dst), "--test-fraction", "1.0", "--workers", "1"])
    assert names(dst, "images", "test") == {"validation_indoor.jpg"}

    main(["--src", str(source), "--dst", str(dst), "--workers", "1"])

    val, test = names(dst, "images", "val"), names(dst, "images", "test")
    assert val == {"validation_indoor.jpg"}
    assert not (val & test), "a held-out image is back in validation"
    assert test == set(), "the stale test split should be gone, not merely unused"


def test_a_split_with_no_annotations_is_skipped(tmp_path, capsys):
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "annotations" / "training").mkdir(parents=True)
    main(["--src", str(src), "--dst", str(tmp_path / "out"), "--workers", "1"])
    assert "no annotations" in capsys.readouterr().out


def test_a_wrong_src_says_so_instead_of_writing_an_empty_dataset(tmp_path):
    with pytest.raises(SystemExit, match="images not found"):
        main(["--src", str(tmp_path), "--dst", str(tmp_path / "out"), "--workers", "1"])
