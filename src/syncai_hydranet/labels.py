"""The one value that means "nobody said", and why it is not a `utils` constant.

``IGNORE`` is the **mask-file contract**: the pixel value an annotation PNG carries where
the class is unknown, and the value the segmentation loss must be told to skip. Those are
two different systems agreeing on one number, which is exactly the kind of agreement that
rots quietly -- nothing crashes when they disagree, the loss simply starts treating
unlabelled pixels as a trainable class and every metric stays plausible.

It was defined four times before this module existed -- `geometry/bev.py`,
`cli/annotation.py`, `scripts/sam3_prelabel.py`, `scripts/annotation_batch.py` -- written
twice more as a literal default in `models/losses.py` and `models/hydranet.py`, and
repeated in eight configs as `ignore_index: 255`. Fourteen places, no check.

**A root-level module rather than a package, deliberately.** Every layer needs it: the
loss (models), the annotation gate (cli), the floor rasteriser (geometry), the label maps
(data) and the pre-labellers (scripts). `geometry` and `data` are siblings that import
nothing from each other, and putting a shared constant in either would couple them for
one integer. Root level is the only place that costs no edge -- the same reason
`config_schema.py` and `preprocessing.py` live here.

**Why 255 and not -1 or 0.** It has to survive a round trip through a single-channel
8-bit PNG, which is what annotators export and what `SegFolderDataset` reads, so the
value must be in 0-255 and must not be a class id. 255 is the top of that range and the
convention ADE20K, Cityscapes and COCO-Stuff all use for the same purpose. **0 is the
trap**: `label_maps_cocostuff.py` documents that COCO-Stuff PNGs put `person` at value 0,
so a pipeline that treats 0 as void deletes every person in the dataset and reports a
perfectly ordinary-looking run.

`config_schema._check_ignore_index` refuses a config whose `ignore_index` is not this
value, which is what turns the agreement from a convention into a checked invariant.
"""

from __future__ import annotations

IGNORE = 255
