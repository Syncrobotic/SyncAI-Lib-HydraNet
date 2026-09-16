#!/usr/bin/env python3
"""Fit staff/customer probes with a whole store held out and one row per track.

Uses human-sorted crops and the existing extraction-defect exclusions. Standardisation
is fitted on source stores only. Outputs are research artefacts, not StaffModel objects:
the deployed camera gate and the unknown verdict remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from syncai_hydranet.analytics.staff import fit_logreg, predict
from syncai_hydranet.data.store_split import STORES, camera_store


def rates(y, pred):
    recalls = [
        float((pred[y == cls] == cls).mean()) if np.any(y == cls) else None for cls in (0, 1)
    ]
    observed = [value for value in recalls if value is not None]
    return {
        "n": len(y),
        "customer_recall": recalls[0],
        "staff_recall": recalls[1],
        "balanced_accuracy": float(np.mean(observed)) if len(observed) == 2 else None,
        "accuracy": float((pred == y).mean()),
        "confusion": [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # Reuse the established crop geometry and known bad-track exclusion.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from staff_probe import encode, encoder, load_batch, raw_torso_colour

    torch.manual_seed(42)
    files, labels, cameras, people = load_batch(args.batch)
    if not files:
        raise ValueError("no reviewed staff/customer crops")
    device = torch.device("cuda")
    embeddings = encode(files, encoder(None, device), device, stem_only=True)
    colour = raw_torso_colour(files)
    groups = sorted(set(people))
    people_arr = np.array(people)
    x_colour, x_combined, y, cams = [], [], [], []
    for person in groups:
        indices = np.flatnonzero(people_arr == person)
        if len(set(labels[indices])) != 1:
            raise ValueError(f"mixed class within track: {person}")
        x_colour.append(colour[indices].mean(0))
        x_combined.append(
            np.concatenate((colour[indices].mean(0), embeddings[indices].mean(0)))
        )
        y.append(labels[indices[0]])
        cams.append(cameras[indices[0]])
    y = np.asarray(y)
    stores = np.array([camera_store(cam)[1] for cam in cams])
    report = {
        "protocol": "Whole-store holdout; one averaged feature row per track; "
        "source-only standardisation; ImageNet frozen stem, no retail pretraining; "
        "human-sorted crop labels. No automatic deployment or new-camera licence.",
        "crops": len(files),
        "tracks": len(groups),
        "results": [],
    }
    for arm, features in (("torso_colour", x_colour), ("colour_imagenet", x_combined)):
        x = np.asarray(features)
        for store in STORES:
            train, test = stores != store, stores == store
            if len(set(y[train])) != 2 or not test.any():
                raise ValueError(f"insufficient labelled stores for {store}")
            mean, std = x[train].mean(0), x[train].std(0) + 1e-6
            z = (x - mean) / std
            weights = fit_logreg(z[train], y[train], iters=40)
            probability = predict(weights, z[test])
            pred = (probability >= 0.5).astype(int)
            record = dict(
                arm=arm,
                held_out_store=store,
                training_tracks=int(train.sum()),
                **rates(y[test], pred),
            )
            test_indices = np.flatnonzero(test)
            record["tracks"] = [
                {
                    "track": groups[i],
                    "camera": cams[i],
                    "label": int(y[i]),
                    "staff_probability": float(probability[j]),
                }
                for j, i in enumerate(test_indices)
            ]
            record["per_camera"] = {}
            test_cameras = np.asarray(cams)[test]
            for cam in sorted(set(test_cameras)):
                selected = test_cameras == cam
                record["per_camera"][cam] = rates(y[test][selected], pred[selected])
            report["results"].append(record)
            np.savez(
                args.out / f"candidate_{arm}_{store}.npz",
                weights=weights,
                mean=mean,
                std=std,
                source_stores=np.array([s for s in STORES if s != store]),
                held_out_store=store,
                feature=arm,
                deployable=False,
            )
            (args.out / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {k: v for k, v in record.items() if k not in ("tracks", "per_camera")}
                )
            )


if __name__ == "__main__":
    main()
