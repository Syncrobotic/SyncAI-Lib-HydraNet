"""PA-100K person attributes: the second stage's first supervised signal.

`RETAIL_DATA.md` ranked RAP v2 first on distribution and PA-100K second on volume.
RAP v2 was not obtained, and measuring the alternatives changed the ranking rather than
merely losing the first choice:

    crop height, median px      site person boxes are 244-336 px
      GRID (metro CCTV)   262      indoor, 1,275 crops -- a test set
      PA-100K (street)    197-207  outdoor, 100,000 crops -- the training source
      TownCentre          184
      Market-1501         128 fixed
      CAVIAR4REID          70      a *shopping centre*, and far too small to be one

CAVIAR4REID is the trap: the right scene at the wrong scale. PA-100K's mismatch is
viewpoint and lighting, not instrument -- its crops are the size these cameras produce,
which is the axis that decides whether an encoder can see anything at all.

---------------------------------------------------------------------------
WHAT THE LABEL SET DECIDES ABOUT THE PRODUCT ASK

The ask is "age, gender, action". Counted over the 80,000 training crops:

    Female       36,492   45.62%
    Age18-60     74,721   93.40%
    AgeLess18     4,152    5.19%
    AgeOver60     1,127    1.41%
    Front/Side/Back  36% / 28% / 36%

**Age is three bands and one of them holds 93% of everybody.** So the honest output is two
binary questions -- is this person under 18, is this person over 60 -- and the second rests
on 1,127 crops. That is the same conclusion `ARCHITECTURE.md` reached from the
other direction (a head region of 31-42 px), arrived at independently from the data.

`SUPPORT` below carries the count for every attribute, because a report that quotes
`LowerStripe` (0.51%) beside `Female` (45.62%) formats them identically.

Viewpoint is labelled here and was one of the two things RAP was wanted for. Occlusion,
the other and the one that fragments tracks, is in neither PA-100K nor PETA.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ..preprocessing import IMAGENET_MEAN, IMAGENET_STD

# Column order in the parquet, which is the channel order of the head trained on it.
ATTRIBUTES = (
    "Female",
    "AgeOver60",
    "Age18-60",
    "AgeLess18",
    "Front",
    "Side",
    "Back",
    "Hat",
    "Glasses",
    "HandBag",
    "ShoulderBag",
    "Backpack",
    "HoldObjectsInFront",
    "ShortSleeve",
    "LongSleeve",
    "UpperStride",
    "UpperLogo",
    "UpperPlaid",
    "UpperSplice",
    "LowerStripe",
    "LowerPattern",
    "LongCoat",
    "Trousers",
    "Shorts",
    "Skirt&Dress",
    "boots",
)

# Positive count in the 80,000-crop train split, measured 2026-08-18. Any number this head
# produces is quoted with its support or it is not quoted: `AgeOver60` and `Female` are the
# same kind of output and not the same kind of evidence.
SUPPORT = {
    "Female": 36492,
    "AgeOver60": 1127,
    "Age18-60": 74721,
    "AgeLess18": 4152,
    "Front": 28718,
    "Side": 22706,
    "Back": 28576,
    "HandBag": 14680,
    "ShoulderBag": 15290,
    "Backpack": 11807,
    "HoldObjectsInFront": 771,
    "LowerStripe": 404,
    "boots": 495,
    "LowerPattern": 1421,
}

# The three the product actually asked for, and the two that are honest.
AGE_BANDS = ("AgeLess18", "Age18-60", "AgeOver60")


class PA100K(Dataset[dict[str, Any]]):
    """One split of PA-100K, read from the HuggingFace parquet already on disk.

    The images live in the parquet as JPEG bytes, so nothing is unpacked to a directory:
    100,000 small files is a worse object than one 421 MB file, and the decode is the same
    either way. The table is memory-mapped by pyarrow rather than materialised, so the
    resident cost is the batch and not the split.
    """

    def __init__(self, root: str, split: str, size=(256, 128), train: bool = False):
        import pyarrow.parquet as pq

        self.table = pq.read_table(f"{root}/{split}-00000-of-00001.parquet")
        names = set(self.table.schema.names)
        missing = [a for a in ATTRIBUTES if a not in names]
        if missing:
            raise ValueError(
                f"{split} parquet is missing {', '.join(missing)}. ATTRIBUTES is the head's "
                "channel order and a silently shorter one trains every attribute against "
                "the wrong channel."
            )
        self.images = self.table.column("image")
        self.labels = np.stack(
            [np.asarray(self.table.column(a).to_numpy(), dtype=np.float32) for a in ATTRIBUTES],
            axis=1,
        )
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, index: int):
        import torch

        rec = self.images[index].as_py()
        img = Image.open(io.BytesIO(rec["bytes"] if isinstance(rec, dict) else rec))
        img = img.convert("RGB").resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        if self.train and np.random.rand() < 0.5:
            # Horizontal flip only. Not vertical, and not rotation: `Front`/`Side`/`Back`
            # is a labelled attribute, and a flip preserves it while a rotation would make
            # the label wrong -- which is the kind of augmentation that trains a head to
            # contradict its own targets.
            x = x[:, ::-1].copy()
        # Imported, not restated: `preprocessing`'s docstring is an argument that these
        # must exist once, because a value that drifts between two of them does not raise
        # -- it feeds the network inputs it was never trained on and the only symptom is
        # predictions that are slightly worse. This was the copy inside `src/`, and it was
        # rebuilding both arrays on every sample.
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        x = ((x - mean) / std).transpose(2, 0, 1)
        return {"image": torch.from_numpy(x), "labels": torch.from_numpy(self.labels[index])}
