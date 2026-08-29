#!/usr/bin/env python3
"""Prepare RAP v2 for `scripts/train_attributes.py`, without touching that script.

    .venv/bin/python scripts/prep_rapv2.py
    .venv/bin/python scripts/train_attributes.py --root datasets/RAP-v2/prepared \\
        --out runs/rapv2_crop01

Writes ``datasets/RAP-v2/prepared/{train,val,test}-00000-of-00001.parquet`` in exactly the
convention `data/attributes.PA100K` already reads: an ``image`` column of encoded bytes
plus one 0/1 column per name in ``ATTRIBUTES``, same file naming, same column-order
contract. The trainer is not modified; the data is shaped to it.

Three sources, and why each is needed:

* ``datasets/_incoming/attr_bundle/rap_zs.pkl`` — the **only RAP v2 label table on disk**
  (84,928 rows x 119 attributes) and the zero-shot identity-disjoint partition
  (train 17,062 / val 4,648 / test 4,928 = 26,638, the subset that carries person
  identity). The split is mandatory: RETAIL_DATA.md's val-selects discipline is exactly
  the identity-overlap mistake this split exists to avoid.
* ``RAP_annotation.mat`` — **v1's** annotation (41,585 rows, 92 attributes). Both
  annotation zips in ``_incoming/attr_bundle`` contain the identical v1 file (same md5);
  the official v2 .mat never landed. It is still load-bearing here because the 119-attr
  pkl **dropped viewpoint**, and `ATTRIBUTES` requires Front/Side/Back. Every one of the
  26,638 zs rows exists in v1 (filenames differ only by ``-`` vs ``_``), viewpoint there
  is exactly one-hot over face{Front,Back,Left,Right}, and on shared attributes v1 and
  the pkl agree 100% — measured before this was trusted.
* ``datasets/RAP-v2/RAP_dataset/`` — the 84,928 crop PNGs (the 84,929th file is
  ``.gitsave``).

Label semantics: RAP labels are {0,1,2} with 2 = uncertain; 2 is mapped to 0 before any
OR (58 cells in the 26,638 rows, measured). OR-merges (e.g. three RAP age bands into
``Age18-60``) are the max of the mapped columns.

Seven PA-100K attributes have **no RAP counterpart** and are written as all-zero columns
so the trainer's schema check passes: UpperStride, UpperLogo, UpperPlaid, UpperSplice,
LowerStripe, LowerPattern, LongCoat. Those head channels train against a constant
negative — frozen, not learned — and must not be quoted from a RAP-trained checkpoint.
``LongSleeve`` is the derived complement of ``ub-ShortSleeve`` (RAP does not label it),
which is a CCTV-reasonable proxy and is flagged as derived, not annotated.

All 119 native RAP columns ride along under a ``rap_`` prefix (plus ``rap_face*`` from
v1), so the RAP-unique labels the product cares about — ``rap_Employee`` (staff),
the 13 ``rap_action-*``, the 8 occlusion columns that no other dataset labels — are in
the same parquet when a wider head wants them. `PA100K` ignores columns it was not asked
for, so their presence costs the current trainer nothing.
"""

from __future__ import annotations

import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PKL = REPO / "datasets/_incoming/attr_bundle/rap_zs.pkl"
ANNOT_ZIP = REPO / "datasets/_incoming/attr_bundle/RAPV2/RAP_annotation.zip"
# From RAP_unzip_code.txt — the two lines there are swapped: the password labelled
# "RAP1.0" is the one that opens both annotation zips (measured; the "RAP2.0" one fails).
ANNOT_PWD = b"casia_cripac_isee_5610"
CROPS = REPO / "datasets/RAP-v2/RAP_dataset"
OUT = REPO / "datasets/RAP-v2/prepared"

# The head's channel order. Imported, not copied: a drift here would train every channel
# against the wrong column and the schema check would not notice.
sys.path.insert(0, str(REPO / "src"))
from syncai_hydranet.data.attributes import ATTRIBUTES  # noqa: E402

LOW_SUPPORT_RATE = 0.01  # positive rate below this is flagged for pos_weight review

# PA-100K name -> RAP v2 pkl columns OR'd together (labels 2->0 first).
# "~" prefix = complement of the OR (only LongSleeve uses it).
PKL_MAP: dict[str, list[str]] = {
    "Female": ["Femal"],
    "AgeOver60": ["AgeBiger60"],
    "Age18-60": ["Age17-30", "Age31-45", "Age46-60"],
    "AgeLess18": ["AgeLess16"],  # RAP's boundary is 16, not 18 — nearest available
    "Hat": ["hs-Hat"],
    "Glasses": ["hs-Glasses", "hs-Sunglasses"],
    "HandBag": ["attachment-HandBag"],
    "ShoulderBag": ["attachment-ShoulderBag"],
    "Backpack": ["attachment-Backpack"],
    "HoldObjectsInFront": ["action-Holding"],  # approximate: RAP labels the action
    "ShortSleeve": ["ub-ShortSleeve"],
    "LongSleeve": ["~ub-ShortSleeve"],  # derived complement, not annotated
    "Trousers": ["lb-LongTrousers", "lb-Jeans", "lb-TightTrousers"],
    "Shorts": ["lb-Shorts"],
    "Skirt&Dress": ["lb-Skirt", "lb-ShortSkirt", "lb-LongSkirt", "lb-Dress"],
    "boots": ["shoes-Boots"],
}
# PA-100K name -> v1 .mat viewpoint columns OR'd together.
V1_MAP: dict[str, list[str]] = {
    "Front": ["faceFront"],
    "Side": ["faceLeft", "faceRight"],
    "Back": ["faceBack"],
}
NO_COUNTERPART = [a for a in ATTRIBUTES if a not in PKL_MAP and a not in V1_MAP]


def load_pkl():
    """rap_zs.pkl pickles `easydict.EasyDict`s; a dict-subclass stub avoids installing it."""
    if "easydict" not in sys.modules:
        mod = types.ModuleType("easydict")

        class EasyDict(dict):
            def __getattr__(self, k):
                try:
                    return self[k]
                except KeyError as e:  # pragma: no cover
                    raise AttributeError(k) from e

            def __setattr__(self, k, v):
                self[k] = v

        setattr(mod, "EasyDict", EasyDict)  # noqa: B010  (shim module built at runtime)
        sys.modules["easydict"] = mod
    import pickle

    with PKL.open("rb") as f:
        return pickle.load(f)


def load_v1_mat():
    """v1's .mat, decrypted straight out of the zip — nothing is written outside OUT."""
    import scipy.io as sio

    with zipfile.ZipFile(ANNOT_ZIP) as z:
        raw = z.read("RAP_annotation/RAP_annotation.mat", pwd=ANNOT_PWD)
    ra = sio.loadmat(io.BytesIO(raw))["RAP_annotation"][0, 0]
    names = [str(a[0][0]).replace("_", "-") for a in ra["imagesname"]]
    attrs = [str(a[0][0]) for a in ra["attribute_eng"]]
    return names, attrs, ra["label"]


def binarise(col: np.ndarray) -> np.ndarray:
    """{0,1,2} -> {0,1}: uncertain is negative, same as the pkl's own convention."""
    return (col == 1).astype(np.int8)


def track_key(name: str) -> str:
    """Camera + video segment + tracker id: the finest same-person unit a filename proves."""
    m = re.match(r"(.+?-tarid\d+)-", name)
    assert m is not None, f"filename does not carry a tarid segment: {name}"
    return m.group(1)


def main() -> int:
    d = load_pkl()
    imgs = [str(s) for s in d["image_name"]]
    attr_idx = {str(a): i for i, a in enumerate(d["attr_name"])}
    labels = d["label"]
    part = {s: np.asarray(d["partition"][s]) for s in ("train", "val", "test")}

    v1_names, v1_attrs, v1_labels = load_v1_mat()
    v1_row = {n: i for i, n in enumerate(v1_names)}
    v1_col = {a: i for i, a in enumerate(v1_attrs)}

    # ---------------------------------------------------------------- validation first
    all_rows = np.concatenate(list(part.values()))
    sizes = {s: len(v) for s, v in part.items()}
    print(f"splits: {sizes}  (sum {len(all_rows)})")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = set(part[a]) & set(part[b])
        if inter:
            raise SystemExit(f"split rows overlap: {a} ∩ {b} = {len(inter)}")
    missing_v1 = [imgs[i] for i in all_rows if imgs[i] not in v1_row]
    if missing_v1:
        raise SystemExit(f"{len(missing_v1)} zs rows missing from v1 .mat (viewpoint source)")
    missing_png = [imgs[i] for i in all_rows if not (CROPS / imgs[i]).is_file()]
    if missing_png:
        raise SystemExit(f"{len(missing_png)} crops missing on disk, first: {missing_png[:3]}")

    # Identity disjointness, to the depth a filename can prove it: the official person ids
    # live in the v2 .mat which is not on disk, so the check is per (video segment, tarid).
    keys = {s: {track_key(imgs[i]) for i in part[s]} for s in part}
    leaks = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = keys[a] & keys[b]
        if shared:
            leaks[f"{a}∩{b}"] = sorted(shared)
    n_leak_imgs = sum(
        1 for i in all_rows if any(track_key(imgs[i]) in v for v in leaks.values())
    )
    if leaks:
        print(
            f"WARNING: {sum(len(v) for v in leaks.values())} track keys cross splits "
            f"({n_leak_imgs} crops, {100 * n_leak_imgs / len(all_rows):.2f}%) — "
            "either tracker-id switches the official person ids corrected, or a real leak; "
            "the v2 .mat that could adjudicate is not on disk. Recorded, not fixed: "
            "the zs split is used exactly as published."
        )

    uncertain = int((labels[all_rows] == 2).sum())
    print(f"uncertain (==2) label cells in zs rows: {uncertain} -> mapped to 0")

    # ---------------------------------------------------------------- build the columns
    def pa_column(name: str, rows: np.ndarray, v1_rows: np.ndarray) -> np.ndarray:
        if name in V1_MAP:
            cols = [binarise(v1_labels[v1_rows, v1_col[c]]) for c in V1_MAP[name]]
            # numpy 2.5 stubs reject ufunc.reduce on a list of arrays; the code is
            # right (same stub family as stable_infer's argmax).
            return np.maximum.reduce(cols)  # ty: ignore[no-matching-overload]
        if name in PKL_MAP:
            out = np.zeros(len(rows), dtype=np.int8)
            for c in PKL_MAP[name]:
                if c.startswith("~"):
                    out = np.maximum(out, 1 - binarise(labels[rows, attr_idx[c[1:]]]))
                else:
                    out = np.maximum(out, binarise(labels[rows, attr_idx[c]]))
            return out
        return np.zeros(len(rows), dtype=np.int8)  # no counterpart: frozen negative

    import pyarrow as pa
    import pyarrow.parquet as pq

    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict = {"splits": sizes, "uncertain_cells": uncertain, "track_leaks": leaks}
    for split, rows in part.items():
        v1_rows = np.array([v1_row[imgs[i]] for i in rows])
        data: dict[str, object] = {}
        data["image"] = [(CROPS / imgs[i]).read_bytes() for i in rows]
        data["image_name"] = [imgs[i] for i in rows]
        for name in ATTRIBUTES:
            data[name] = pa_column(name, rows, v1_rows)
        for a, j in attr_idx.items():
            data[f"rap_{a}"] = binarise(labels[rows, j])
        for a in ("faceFront", "faceBack", "faceLeft", "faceRight"):
            data[f"rap_{a}"] = binarise(v1_labels[v1_rows, v1_col[a]])
        table = pa.table(data)
        dest = OUT / f"{split}-00000-of-00001.parquet"
        pq.write_table(table, dest)
        print(f"wrote {dest}  ({table.num_rows} rows, {dest.stat().st_size / 1e6:.1f} MB)")

    # ---------------------------------------------------------------- positive rates
    train_rows = part["train"]
    train_v1 = np.array([v1_row[imgs[i]] for i in train_rows])
    print(f"\nper-attribute positive rate on train ({len(train_rows)} crops):")
    rates = {}
    for name in ATTRIBUTES:
        col = pa_column(name, train_rows, train_v1)
        rate = float(col.mean())
        rates[name] = {"positives": int(col.sum()), "rate": rate}
        flag = "  LOW_SUPPORT" if 0 < rate < LOW_SUPPORT_RATE else ""
        if name in NO_COUNTERPART:
            flag = "  NO_RAP_COUNTERPART (frozen negative)"
        print(f"  {name:20s} {int(col.sum()):6d}  {100 * rate:6.2f}%{flag}")
    print("\nRAP-unique columns the product asked about (train positive rate):")
    kept = ["Employee", "Customer"] + [
        a for a in attr_idx if a.startswith(("action-", "Occlusion", "occlustion-"))
    ]
    for a in kept:
        col = binarise(labels[train_rows, attr_idx[a]])
        rate = float(col.mean())
        rates[f"rap_{a}"] = {"positives": int(col.sum()), "rate": rate}
        flag = "  LOW_SUPPORT" if 0 < rate < LOW_SUPPORT_RATE else ""
        print(f"  rap_{a:28s} {int(col.sum()):6d}  {100 * rate:6.2f}%{flag}")
    stats["train_rates"] = rates
    (OUT / "prep_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT / 'prep_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
