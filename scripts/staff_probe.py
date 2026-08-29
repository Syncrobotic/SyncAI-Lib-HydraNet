#!/usr/bin/env python3
"""staff vs customer, as a linear probe on frozen features, held out by camera.

    python3 scripts/staff_probe.py --batch datasets/staff_customer_batch01

`staff/customer` is the missing signal under every retail output -- without it
`reach_to_shelf` is 11.7 alerts a minute of staff working, measured. This trains the
first one, on the crops a human sorted, and reports what it can and cannot claim.

---------------------------------------------------------------------------
WHY A PROBE AND NOT A FINE-TUNE

Because the tree already measured what a fine-tune costs here. `runs/rapv2_eval01`:
17k crops and 16 epochs of attribute training took the crop encoder's Market-1501
association from the untrained ImageNet floor of **mAP 0.0318 down to 0.0113** -- 2.8x
*worse* than not training at all, the general features ground away into RAP-specific
ones. That encoder's embedding is the output PLAN's ordering puts first, so a
staff/customer head that damaged it again would be a net loss whatever its own accuracy
said. A linear probe on frozen features cannot do that, and if it is good enough the
question of risking the backbone never arises.

Which frozen features is measured rather than chosen: the ImageNet floor, the PA-100K
encoder and the RAP one are all probed, so "the fine-tune hurt the embedding" and "the
fine-tune helps this task" are separate questions with separate answers.

---------------------------------------------------------------------------
THE SPLIT, AND WHAT THIS FLEET'S LABELS DO TO IT

**Leave one camera out**, because the deployment question is "does it work on a camera
we did not train on", and a random split cannot answer it: crops from one track share a
person, a background, a lens and a white balance, so a model can score well by
recognising the room.

That is not a hypothetical here. Nine of the 32 labelled cameras carry **exactly one
class** -- Taichung-cam02, cam10 and Tao-Hsin-cam14 are all staff, Kaohsiung-cam01,
cam02, cam08, Taichung-cam09, cam12 and Tao-Hsin-cam03 are all customer. On those,
"which camera" *is* "which class". Held out, a single-class camera can only measure
recall for the class it has, so its accuracy is not comparable with a mixed camera's and
is reported separately rather than averaged in.

Crops are grouped by `camera__clip__track` throughout, so a person is never split across
the fold. One person in the batch is dropped by name: Kaohsiung-cam04 t0002 is a tracker
identity switch -- a customer at 14:30:27 and a member of staff at 14:32:25 under one id
-- which the labeller correctly sorted into two folders and which is an extraction defect,
not a labelling one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from syncai_hydranet.analytics.staff import (  # noqa: E402
    crop_features,
    fit_logreg,
    fit_staff_model,
    predict,
)
from syncai_hydranet.data.attributes import ATTRIBUTES  # noqa: E402
from syncai_hydranet.models.crop_encoder import CropEncoder  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402

SIZE = (256, 128)  # data/attributes.py's crop geometry, so the encoder sees what it trained on
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Known extraction defect rather than a labelling one -- see the module docstring.
DROP_PEOPLE = {"Kaohsiung-cam04__20260816-063027_20260816-063527__t0002"}


def load_batch(batch: Path) -> tuple[list[Path], np.ndarray, list[str], list[str]]:
    files, labels, cams, people = [], [], [], []
    for cls, y in (("customer", 0), ("staff", 1)):
        for f in sorted((batch / cls).glob("*.jpg")):
            cam, clip, track, _frame = f.stem.split("__")
            person = f"{cam}__{clip}__{track}"
            if person in DROP_PEOPLE:
                continue
            files.append(f)
            labels.append(y)
            cams.append(cam)
            people.append(person)
    return files, np.array(labels), cams, people


def encode(
    files: list[Path], model, device, batch_size: int = 64, *, stem_only: bool = False
) -> np.ndarray:
    """Embeddings for a list of crops.

    `stem_only` takes the backbone's pooled 512-d features instead of `embed_only`, and
    it exists because a bug in the first version of this file made the ImageNet floor
    unreproducible: `CropEncoder.embed` is an untrained `nn.Linear(512, 256)` on a
    checkpoint-free encoder, so "the floor" was a **random projection** of ImageNet
    features -- a different one each run. Two runs of identical code on identical data
    gave 0.757 and 0.839. It was also an unfair comparison, since the two trained
    encoders have a fitted `embed` and the floor did not.
    """
    out = []
    for i in range(0, len(files), batch_size):
        chunk = files[i : i + batch_size]
        arr = np.stack(
            [
                (
                    (
                        np.asarray(
                            Image.open(f)
                            .convert("RGB")
                            .resize((SIZE[1], SIZE[0]), Image.Resampling.BILINEAR),
                            dtype=np.float32,
                        )
                        / 255.0
                        - MEAN
                    )
                    / STD
                ).transpose(2, 0, 1)
                for f in chunk
            ]
        )
        with torch.no_grad():
            t = torch.from_numpy(arr).to(device)
            f = model.stem(t).flatten(1) if stem_only else model.embed_only(t)
            out.append(f.cpu().numpy())
    return np.concatenate(out)


def torso_colour(files: list[Path]) -> np.ndarray:
    """The obvious feature, as a control: what colour is the torso.

    A uniform is a colour before it is anything else, and if a handful of colour
    statistics match a 256-dimension embedding then the embedding is not earning its
    place -- a per-store colour reference is cheaper, interpretable, and re-fittable from
    three photographs when the shop changes its shirts. Running it through the *same*
    leave-one-camera-out protocol is what makes the comparison mean anything.

    The nine statistics moved to `analytics/appearance.py` on 2026-08-27 when a second
    caller needed them, and the resize in front of them moved to `analytics.staff` on
    2026-08-28 when a *deployed* model needed the identical transform -- if the probe and
    the artefact resize differently they are measuring and shipping different features,
    and neither would say so. The numbers are unchanged: all sixteen per-camera
    accuracies still reproduce the `probe.json` that predates both moves.

    **The standardisation stays here**, because it is a property of the set being fitted
    rather than of a crop, and a probe is what needs its features on one scale. Note that
    it is taken over the *whole* set, held-out camera included. That is transductive, and
    for comparing four sources against each other under one protocol it is harmless; it
    is not acceptable in an artefact, so `analytics.staff.fit_staff_model` standardises
    on the fitted half only and its accuracy is reported separately below.
    """
    return standardise(raw_torso_colour(files))


def raw_torso_colour(files: list[Path]) -> np.ndarray:
    """The nine statistics in their own units, one row per crop."""
    return np.stack([crop_features(np.asarray(Image.open(f).convert("RGB"))) for f in files])


def standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def encoder(path: str | None, device):
    """A frozen `CropEncoder`. `None` is the ImageNet floor -- untrained, and the number
    every other option in this project has had to clear."""
    if path is None:
        return CropEncoder(len(ATTRIBUTES), embed_dim=256, pretrained=True).to(device).eval()
    ckpt = load_checkpoint(path)
    model = CropEncoder(
        len(ckpt["attributes"]), embed_dim=ckpt["model"]["embed.weight"].shape[0],
        pretrained=False,
    )  # fmt: skip
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", default="datasets/staff_customer_batch01")
    ap.add_argument("--out", default="runs/staff_probe01")
    ap.add_argument(
        "--fit-for",
        action="append",
        default=[],
        metavar="CAMERA",
        help="also SAVE a deployable torso-colour model licensed for CAMERA: fitted on "
        "every other labelled camera and scored on this one, written to "
        "<out>/model_<CAMERA>.json. Repeatable. Without this the script only measures.",
    )
    a = ap.parse_args()

    device = pick_device(None)
    files, y, cams, people = load_batch(Path(a.batch))
    by_cam = Counter(zip(cams, y, strict=True))
    mixed = sorted({c for c in set(cams) if by_cam[(c, 0)] and by_cam[(c, 1)]})
    single = sorted(set(cams) - set(mixed))
    n_people = len(set(people))
    print(
        f"{len(files)} crops / {n_people} people / {len(set(cams))} cameras   "
        f"staff {int(y.sum())} crops, customer {int((1 - y).sum())}"
    )
    print(f"mixed-class cameras: {len(mixed)}   single-class: {len(single)} -> {single}")

    sources = {
        "imagenet floor (backbone, 512-d)": None,
        "crop_encoder01 (PA-100K)": "runs/crop_encoder01/last.pt",
        "rapv2_crop01 (RAP v2)": "runs/rapv2_crop01/last.pt",
    }
    report: dict = {"cameras": {"mixed": mixed, "single_class": single}, "sources": {}}
    embeddings: dict[str, np.ndarray] = {}
    for label, path in [*sources.items(), ("torso colour (control)", "__colour__")]:
        if path == "__colour__":
            feats = torso_colour(files)
        else:
            if path is not None and not (ROOT / path).exists():
                print(f"\n{label}: checkpoint missing, skipped")
                continue
            feats = encode(files, encoder(path, device), device, stem_only=path is None)
            embeddings[label] = feats
        cam_arr = np.array(cams)

        per_cam: dict[str, dict] = {}
        held_true, held_pred = [], []
        for cam in mixed:
            test = cam_arr == cam
            w = fit_logreg(feats[~test], y[~test].astype(float))
            p = predict(w, feats[test])
            hit = (p >= 0.5).astype(int) == y[test]
            per_cam[cam] = {
                "n": int(test.sum()),
                "staff": int(y[test].sum()),
                "accuracy": float(hit.mean()),
            }
            held_true.append(y[test])
            held_pred.append(p)
        yt = np.concatenate(held_true)
        pp = np.concatenate(held_pred)
        pred = (pp >= 0.5).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        bal = 0.5 * (rec + int(((pred == 0) & (yt == 0)).sum()) / max(int((yt == 0).sum()), 1))
        print(
            f"\n{label}\n  leave-one-camera-out over {len(mixed)} mixed cameras, "
            f"{len(yt)} held-out crops\n"
            f"  balanced accuracy {bal:.3f}   staff precision {prec:.3f} recall {rec:.3f}"
        )
        worst = sorted(per_cam.items(), key=lambda kv: kv[1]["accuracy"])[:3]
        print(
            "  worst cameras: "
            + ", ".join(f"{c} {v['accuracy']:.2f} (n={v['n']})" for c, v in worst)
        )
        report["sources"][label] = {
            "balanced_accuracy": bal,
            "staff_precision": prec,
            "staff_recall": rec,
            "per_camera": per_cam,
        }

    # colour + the best embedding, same protocol: does the embedding add anything the
    # colour does not already have?
    best = max(
        (k for k in report["sources"] if k != "torso colour (control)"),
        key=lambda k: report["sources"][k]["balanced_accuracy"],
        default=None,
    )
    if best and best in embeddings:
        feats = np.concatenate([torso_colour(files), embeddings[best]], axis=1)
        cam_arr = np.array(cams)
        ht, hp = [], []
        for cam in mixed:
            test = cam_arr == cam
            w = fit_logreg(feats[~test], y[~test].astype(float))
            ht.append(y[test])
            hp.append(predict(w, feats[test]))
        yt, pp = np.concatenate(ht), np.concatenate(hp)
        pred = (pp >= 0.5).astype(int)
        rec = int(((pred == 1) & (yt == 1)).sum()) / max(int((yt == 1).sum()), 1)
        spec = int(((pred == 0) & (yt == 0)).sum()) / max(int((yt == 0).sum()), 1)
        print(f"\ncolour + {best}\n  balanced accuracy {0.5 * (rec + spec):.3f}")
        report["sources"][f"colour + {best}"] = {"balanced_accuracy": 0.5 * (rec + spec)}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n-> {out}/probe.json")

    # A measurement nothing can load is not a classifier. `--fit-for` is what turns the
    # torso-colour arm above into an artefact `analytics.staff` can apply, one per
    # camera, each carrying the accuracy it was scored at on that camera and refusing
    # every other one. Colour alone rather than the combined arm on purpose: the
    # combination's 0.893 has no per-camera breakdown, so it cannot answer "may this
    # store be coloured" for any store. See `analytics/staff.py`.
    if a.fit_for:
        raw = raw_torso_colour(files)
        print()
        for cam in a.fit_for:
            model = fit_staff_model(raw, y, cams, held_out=cam)
            path = model.save(out / f"model_{cam}.json")
            print(
                f"{cam}: held-out accuracy {model.accuracy:.3f} on {model.held_out_n} "
                f"crops, fitted on {model.n_crops} crops from "
                f"{len(model.trained_cameras)} cameras -> {path}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
