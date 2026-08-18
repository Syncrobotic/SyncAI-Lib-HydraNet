"""Dataset wrappers.

SegFolderDataset: a generic "images folder + annotations folder" segmentation dataset,
    covering RUGD (RGB palette), RELLIS-3D and ADE20K (integer ids).
CocoDetDataset: COCO-format object detection.

Each dataset declares ``supervises``, naming the heads it provides labels for. The
multi-task loader uses that to compute only the relevant losses per step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import label_maps
from .label_maps_retail_security import get_det_vocab
from .transforms import GEOM_IDENTITY, Sample, build_transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _index_pairs(img_dir: Path, ann_dir: Path) -> list[tuple[Path, Path]]:
    """Pair images with annotations by their path under the split, ignoring extensions.

    By *path*, not by bare filename. Session directories are what carry an honest
    train/val split, and every per-session numbering scheme produces ``0000.png`` more
    than once. Keyed on the stem alone, the second session's frames overwrite the first's
    in the lookup and the loader silently trains on a third of what was annotated --
    caught by `hydranet-annotation check`, which reported 60 masks with no counterpart on
    a three-session batch that had one for every frame.

    Flat datasets are unaffected: with no subdirectories the relative path *is* the stem.
    Layouts that nest images and annotations differently fall back to stem matching, so
    ADE20K and RUGD keep working.
    """
    anns_by_rel, anns_by_stem = {}, {}
    for p in ann_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            anns_by_rel[p.relative_to(ann_dir).with_suffix("")] = p
            anns_by_stem.setdefault(p.stem, p)

    images = [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    pairs = [
        (p, anns_by_rel[rel])
        for p in images
        if (rel := p.relative_to(img_dir).with_suffix("")) in anns_by_rel
    ]
    if pairs:
        return pairs
    # Nothing matched by path: the two trees are shaped differently, so fall back.
    return [(p, anns_by_stem[p.stem]) for p in images if p.stem in anns_by_stem]


class SegFolderDataset(Dataset):
    """Layout::

        root/images/<split>/**/*.png
        root/annotations/<split>/**/*.png

    Produces masks for both the terrain and traversability heads from one annotation.
    """

    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        train: bool,
        label_format: str = "auto",
        supervises=("traversability", "terrain"),
        label_map: str | None = None,
        letterbox: bool = False,
        augment: dict | None = None,
    ):
        self.root = Path(root)
        img_dir = self.root / "images" / split
        ann_dir = self.root / "annotations" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"{img_dir} does not exist; see README for the expected layout."
            )
        self.pairs = _index_pairs(img_dir, ann_dir)
        if not self.pairs:
            raise RuntimeError(
                f"no image/annotation pairs found under {self.root} split={split}"
            )
        # label_map names a complete scheme (preferred). Without it we fall back to the
        # legacy auto-detection between RGB palette and integer ids.
        self.scheme = label_maps.get_scheme(label_map) if label_map else None
        self.label_format = label_format
        self.transform = build_transforms(
            input_size, train, letterbox=letterbox, augment=augment
        )
        self.supervises = list(supervises)
        self._color_lut = None
        self._id_lut = None

    def __len__(self):
        return len(self.pairs)

    def _decode_ann(self, ann_path: Path) -> np.ndarray:
        ann = Image.open(ann_path)
        if self.scheme is not None:
            fmt = "color" if self.scheme.fmt == "color" else "id"
            mapping = self.scheme.mapping
        else:
            fmt = self.label_format
            if fmt == "auto":
                is_rgb = ann.mode in ("RGB", "P") and np.asarray(ann.convert("RGB")).ndim == 3
                fmt = "color" if is_rgb else "id"
            elif fmt == "rugd_color":
                fmt = "color"
            mapping = (
                label_maps.RUGD_COLOR_TO_TERRAIN
                if fmt == "color"
                else label_maps.RELLIS_ID_TO_TERRAIN
            )

        if fmt == "color":
            rgb = np.asarray(ann.convert("RGB"))
            if self._color_lut is None:  # 24-bit RGB -> terrain id lookup table
                lut = np.full(1 << 24, 255, dtype=np.uint8)
                for (r, g, b), t in mapping.items():
                    lut[(r << 16) | (g << 8) | b] = t
                self._color_lut = lut
            key = (
                (rgb[..., 0].astype(np.int32) << 16)
                | (rgb[..., 1].astype(np.int32) << 8)
                | rgb[..., 2].astype(np.int32)
            )
            return self._color_lut[key]

        # Single-channel integer ids (RELLIS-3D, ADE20K). A lookup table beats one
        # boolean mask per class: ADE20K has 150 of them.
        raw = np.asarray(ann)
        if raw.ndim == 3:
            raw = raw[..., 0]
        if self._id_lut is None:
            n = max(int(max(mapping)) + 1, 256)
            lut = np.full(n, 255, dtype=np.uint8)
            for src_id, t in mapping.items():
                lut[int(src_id)] = t
            self._id_lut = lut
        return self._id_lut[np.clip(raw, 0, len(self._id_lut) - 1)]

    def __getitem__(self, idx: int):
        img_path, ann_path = self.pairs[idx]
        terrain = self._decode_ann(ann_path)
        masks = {"terrain": terrain}
        # Only build the traversability target if some head is going to read it. It is a
        # per-pixel lookup through the policy table plus a second mask carried through
        # every augmentation and collated into every batch, and a config without that
        # head (configs/hydranet_retail_objects.yaml) pays all of it for a tensor
        # `compute_losses` then ignores. `supervises` is the declaration that decides,
        # which keeps this consistent with what the trainer actually asks for.
        if "traversability" in self.supervises:
            trav_map = self.scheme.trav if self.scheme else None
            masks["traversability"] = label_maps.terrain_to_traversability(terrain, trav_map)
        s = Sample(image=Image.open(img_path).convert("RGB"), masks=masks)
        s = self.transform(s)
        return {"image": s["image"], "targets": dict(s["masks"]), "supervises": self.supervises}


class CocoDetDataset(Dataset):
    """COCO detection: ``root/{split}/`` images, ``root/annotations/instances_{split}.json``."""

    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        train: bool,
        supervises=("detection",),
        letterbox: bool = False,
        augment: dict | None = None,
        classes: list[str] | None = None,
        score_classes: list[str] | None = None,
        det_vocab: str | None = None,
    ):
        from pycocotools.coco import COCO

        self.root = Path(root)
        ann_file = self.root / "annotations" / f"instances_{split}.json"
        if not ann_file.is_file():
            raise FileNotFoundError(f"{ann_file} does not exist.")
        self.coco = COCO(str(ann_file))
        self.img_dir = self.root / split

        # A subset spends the head's capacity on the classes the robot meets. COCO's 80
        # include zebra and snowboard; indoors, nine tenths of the output channels are
        # learning to say "no". Names are checked against the annotation file rather
        # than trusted, because a typo would otherwise silently drop a whole class.
        if classes:
            cat_ids = sorted(self.coco.getCatIds(catNms=list(classes)))
            found = {c["name"] for c in self.coco.loadCats(cat_ids)}
            missing = [n for n in classes if n not in found]
            if missing:
                raise ValueError(f"not COCO category names: {', '.join(sorted(missing))}")
        else:
            cat_ids = sorted(self.coco.getCatIds())
        # Two numbering regimes, and which one is in force is the difference between a
        # head that can hold both questions and one that cannot.
        #
        # Without `det_vocab`: contiguous and 0-based *per dataset*, which is correct for
        # a single-source run and is exactly what stops two of them sharing a head --
        # COCO's `person` and the site's `boxed_stock` both become label 0.
        #
        # With it: the vocabulary assigns the id, by category *name*, so both files agree
        # before either is opened. See `label_maps_retail_security` for the ids and for
        # the half of the problem this does not solve.
        self.vocab = get_det_vocab(det_vocab) if det_vocab else None
        if self.vocab is None:
            self.cat_ids = cat_ids
            self.cat_to_label = {c: i for i, c in enumerate(cat_ids)}  # contiguous, 0-based
            self.label_to_cat = {i: c for c, i in self.cat_to_label.items()}
        else:
            cat_ids, mapped = self._map_to_vocab(cat_ids)
            self.cat_ids = cat_ids
            self.cat_to_label = mapped

        # Scoring can be narrowed without touching the head. This is how an 80-class
        # checkpoint is compared against a narrower one: both are scored over the same
        # categories, so the comparison is not just a smaller denominator.
        if score_classes:
            score_ids = sorted(self.coco.getCatIds(catNms=list(score_classes)))
            found = {c["name"] for c in self.coco.loadCats(score_ids)}
            missing = [n for n in score_classes if n not in found]
            if missing:
                raise ValueError(f"not COCO category names: {', '.join(sorted(missing))}")
            outside = sorted(set(score_ids) - set(cat_ids))
            if outside:
                names = ", ".join(c["name"] for c in self.coco.loadCats(outside))
                raise ValueError(f"score_classes not trained by this dataset: {names}")
            self.score_cat_ids = score_ids
        else:
            self.score_cat_ids = cat_ids

        # Images whose only annotations were dropped carry no signal for this subset, and
        # would train the head that everything in them is background.
        keep = set(self.cat_to_label)
        self.ids = [
            i
            for i in sorted(self.coco.imgs.keys())
            if any(
                a["category_id"] in keep
                for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=i, iscrowd=False))
            )
        ]
        # After the image filter, because the filter reads the annotation file's own ids
        # and this rewrites them. Merging is what lets `backpack`, `handbag` and
        # `suitcase` be one `bag` channel and still be scored as one category.
        if self.vocab is not None:
            self._merge_categories()

        # The classes this dataset can actually put a box on, which is not the same as
        # the classes the head has. `FCOSLoss` reads it as a channel mask so that a
        # shopper nobody drew a box around is *no gradient* for `person` rather than a
        # labelled negative for it. See label_maps_retail_security's second section.
        self.supplied_labels = tuple(sorted(set(self.cat_to_label.values())))
        self.class_mask = (
            self.vocab.class_mask(self.supplied_labels) if self.vocab is not None else None
        )

        self.transform = build_transforms(
            input_size, train, letterbox=letterbox, augment=augment
        )
        self.supervises = list(supervises)

    def _map_to_vocab(self, cat_ids: list[int]) -> tuple[list[int], dict[int, int]]:
        """Source category ids -> vocabulary labels, dropping what the vocabulary omits.

        A dropped category is not an error: COCO carries 80 and this vocabulary wants
        four of them. A dataset where *nothing* maps is an error, and a loud one -- it
        would otherwise train on an empty image list, which reads as a dataset that is
        simply small.
        """
        assert self.vocab is not None
        cats = {c["id"]: c["name"] for c in self.coco.loadCats(cat_ids)}
        mapped = {}
        for cid in cat_ids:
            label = self.vocab.label_of(cats[cid])
            if label is not None:
                mapped[cid] = label
        if not mapped:
            raise ValueError(
                f"det_vocab {self.vocab.name!r} maps none of this dataset's categories "
                f"({', '.join(sorted(cats.values()))}). Its vocabulary is "
                f"{', '.join(self.vocab.classes)}; add an alias in "
                f"label_maps_retail_security.DET_ALIASES, or point this dataset at a "
                f"vocabulary that covers it."
            )
        return sorted(mapped), mapped

    def _merge_categories(self) -> None:
        """Rewrite this dataset's in-memory COCO ground truth into vocabulary categories.

        **In memory only** -- nothing on disk changes, and a second dataset built from
        the same annotation file gets its own untouched handle.

        Why rewrite the ground truth rather than only the labels: the evaluator scores
        each detection dataset against `ds.coco` and maps a prediction back through
        `ds.label_to_cat`. With three source categories behind one channel there is no
        honest single id to map `bag` to -- pick `backpack` and every `handbag` box in
        the ground truth becomes an unmatchable category, and mAP drops for a reason that
        has nothing to do with the model. Merging the ground truth the same way the
        labels were merged is what makes the two agree.
        """
        assert self.vocab is not None
        rep: dict[int, int] = {}
        for cid, label in sorted(self.cat_to_label.items()):
            rep.setdefault(label, cid)
        to_rep = {cid: rep[label] for cid, label in self.cat_to_label.items()}
        data = self.coco.dataset
        for ann in data["annotations"]:
            if ann["category_id"] in to_rep:
                ann["category_id"] = to_rep[ann["category_id"]]
        data["categories"] = [
            {"id": cid, "name": self.vocab.classes[label], "supercategory": self.vocab.name}
            for label, cid in sorted(rep.items())
        ]
        self.coco.createIndex()
        self.cat_ids = sorted(rep.values())
        self.cat_to_label = {cid: label for label, cid in rep.items()}
        self.label_to_cat = dict(rep)
        self.score_cat_ids = sorted({to_rep[c] for c in self.score_cat_ids if c in to_rep})

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        boxes, labels = [], []
        for a in anns:
            # An image kept for one subset class can still carry boxes from classes the
            # subset drops; those are not background, they are simply not this head's
            # problem, so they are left out rather than mapped to some other channel.
            label = self.cat_to_label.get(a["category_id"])
            x, y, w, h = a["bbox"]
            if label is not None and w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(label)
        s = Sample(
            image=img,
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            labels=np.asarray(labels, dtype=np.int64),
        )
        s = self.transform(s)
        targets = {"boxes": s["boxes"], "labels": s["labels"]}
        if self.class_mask is not None:
            # Per sample rather than per dataset, because that is the only channel the
            # collate has: every sample in a batch comes from one dataset, so the stacked
            # [B, C] is C repeated B times and broadcasts over the loss unchanged.
            targets["det_class_mask"] = torch.from_numpy(self.class_mask)
        return {
            "image": s["image"],
            "targets": targets,
            "supervises": self.supervises,
            "image_id": img_id,
            # How this image was scaled and padded, so COCOeval gets boxes in original
            # image coordinates instead of an after-the-fact guess from the frame size.
            "geom": s.get("geom", GEOM_IDENTITY),
        }


SPLITS = ("train", "val", "test")


def resolve_split(dcfg: dict, split: str) -> str:
    """Map a logical split to the folder name this dataset uses for it.

    ``test`` is optional and deliberately unconfigured by default: a held-out split is
    only meaningful if nothing in training ever reads it, so it has to be created on
    purpose rather than defaulted into existence.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {', '.join(SPLITS)}")
    key = f"split_{split}"
    if key not in dcfg:
        raise KeyError(
            f"dataset {dcfg.get('name')!r} has no {key}. Add it to the config and "
            f"populate the matching folder; see README on splitting by sequence."
        )
    return dcfg[key]


def build_dataset(
    dcfg, input_size, split: str = "train", letterbox: bool = False, augment: dict | None = None
):
    """Build one dataset. Augmentation is applied to the train split only."""
    folder = resolve_split(dcfg, split)
    train = split == "train"
    sup = dcfg["supervises"]
    if dcfg["type"] == "seg_folder":
        return SegFolderDataset(
            dcfg["root"],
            folder,
            input_size,
            train,
            label_format=dcfg.get("label_format", "auto"),
            supervises=sup,
            label_map=dcfg.get("label_map"),
            letterbox=letterbox,
            augment=augment,
        )
    if dcfg["type"] == "coco":
        return CocoDetDataset(
            dcfg["root"],
            folder,
            input_size,
            train,
            supervises=sup,
            letterbox=letterbox,
            augment=augment,
            classes=dcfg.get("classes"),
            score_classes=dcfg.get("score_classes"),
            det_vocab=dcfg.get("det_vocab"),
        )
    raise ValueError(f"unknown dataset type: {dcfg['type']}")
