#!/usr/bin/env python3
"""Step 8's temporal model: posture from a keypoint sequence, under 100K parameters.

    uv run python tools/temporal/train_posture.py --npz runs/ntu_project01/seq5.npz

PLAN §2.3 specifies a "<100K-parameter temporal model over pose sequences, CPU", trained
on public 3D action data projected through our measured camera pose. `ntu_project.py`
produces that data; this trains the model and reports the one number that decides whether
it earned its parameters.

---------------------------------------------------------------------------
THE BAR IS A MEASUREMENT, NOT A DEMONSTRATION

Two things already do this job, and both have numbers:

* the **shipped geometric rule**, whose tuned form reads 96% fall recall at a 29%
  `pick_up` false rate on NTU's ground truth (§7.14) -- about 0.835 balanced accuracy;
* a **linear model on time-pooled features**, which reads **0.944** on `fall` against
  `pick_up` held out by performer (§7.16, at 5 fps).

The linear floor is the bar. A temporal model that does not clear it is not a temporal
model worth shipping -- it is 33,000 parameters restating a mean.

---------------------------------------------------------------------------
WHAT THE PROTOCOL REFUSES TO DO

**Held out by performer.** NTU repeats every action per subject, so a clip-level split
puts the same body on both sides and reports a number the deployment cannot reproduce.
The several placements of one clip move together for the same reason.

**No early stopping on the held-out fold.** Choosing the epoch by the number being
reported is how a fold stops being held out. The epoch count is fixed in advance and the
last one is scored, which costs a little accuracy and buys the number's meaning.

**Frame rate is not a hyperparameter.** The features carry a per-frame velocity, so the
model is trained at whatever rate the npz was built at and the npz records it. Training at
30 and serving at 5 is a sixfold change in the dynamics with nothing to announce it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from syncai_hydranet.utils.device import pick_device


class PostureNet(nn.Module):
    """Two 1-D convolutions over time, masked pooling, a linear head.

    Convolutional rather than recurrent: the receptive field is what matters here -- a
    fall is a half-second event at 5 fps -- and a convolution states its window in the
    kernel size where a recurrence hides it in the training. It is also the shape that
    exports and runs on a CPU beside 96 streams without a second thought.
    """

    def __init__(self, n_features: int, n_classes: int, width: int = 48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, width, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(width * 2, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x (B, T, F) -> (B, F, T); mask (B, T) with 1 on real frames
        h = self.conv(x.transpose(1, 2))
        m = mask.unsqueeze(1)
        mean = (h * m).sum(-1) / m.sum(-1).clamp(min=1)
        # padded frames must not win a max; -inf on them rather than a large constant,
        # which would become the answer for a sequence shorter than its own padding
        mx = h.masked_fill(m == 0, float("-inf")).max(-1).values
        return self.head(torch.cat([mean, mx], dim=1))


def _standardise(v, m, mu, sd, device) -> torch.Tensor:
    """Zero-mean, unit-variance on the fold's own training statistics, padding left at 0."""
    return torch.from_numpy(np.nan_to_num((v - mu) / sd) * m[..., None]).to(device)


def fit_fold(
    xtr, mtr, ytr, xte, mte, *, n_classes: int, epochs: int, device: str, seed: int
) -> np.ndarray:
    torch.manual_seed(seed)
    model = PostureNet(xtr.shape[2], n_classes).to(device)
    counts = torch.bincount(ytr, minlength=n_classes).float()
    weight = (counts.sum() / counts.clamp(min=1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx], mtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(xte, mte).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--npz", default="runs/ntu_project01/seq5.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/posture01")
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=False)
    x_all, length, label, performer = d["x"], d["length"], d["label"], d["performer"]
    fps = float(d["fps"][0])
    classes = sorted(set(label.tolist()))
    y_all = np.array([classes.index(v) for v in label])
    mask = (np.arange(x_all.shape[1])[None, :] < length[:, None]).astype(np.float32)

    # standardise on the training half of each fold only -- fitting the scaler on
    # everything leaks the held-out performer's distribution into the model
    device = str(pick_device())
    n_params = sum(p.numel() for p in PostureNet(x_all.shape[2], len(classes)).parameters())
    print(
        f"{len(x_all)} sequences / {len(set(performer.tolist()))} performers / "
        f"{len(classes)} classes at {fps:g} fps   model {n_params:,} parameters"
    )
    assert n_params < 100_000, f"PLAN 2.3 specifies under 100K, this is {n_params:,}"

    logits = np.zeros((len(x_all), len(classes)), dtype=np.float32)
    for person in sorted(set(performer.tolist())):
        te = performer == person
        tr = ~te
        if not te.any() or len(set(y_all[tr].tolist())) < len(classes):
            continue
        flat = x_all[tr].reshape(-1, x_all.shape[2])[mask[tr].reshape(-1) > 0]
        mu, sd = flat.mean(0), flat.std(0) + 1e-6

        xtr, xte = (
            _standardise(x_all[tr], mask[tr], mu, sd, device),
            _standardise(x_all[te], mask[te], mu, sd, device),
        )
        logits[te] = fit_fold(
            xtr, torch.from_numpy(mask[tr]).to(device), torch.from_numpy(y_all[tr]).to(device),
            xte, torch.from_numpy(mask[te]).to(device),
            n_classes=len(classes), epochs=a.epochs, device=device, seed=a.seed,
        )  # fmt: skip

    scored = logits.any(axis=1)
    pred = logits.argmax(1)
    per_class = {
        c: float((pred[scored & (y_all == i)] == i).mean())
        for i, c in enumerate(classes)
        if (scored & (y_all == i)).any()
    }
    print(f"\nheld out by performer, {int(scored.sum())} sequences scored")
    print("multi-class recall: " + "  ".join(f"{c} {v:.3f}" for c, v in per_class.items()))

    print(f"\n{'pair':30s} {'balanced acc':>13s} {'floor §7.16':>12s}")
    floors = {
        "fall vs pick_up": 0.944, "fall vs sit_down": 0.835, "fall vs stand_still": 0.932,
        "fall vs stagger": 0.923, "pick_up vs sit_down": 0.932,
    }  # fmt: skip
    report: dict[str, float] = {}
    for key, floor in floors.items():
        pos, neg = key.split(" vs ")
        i, j = classes.index(pos), classes.index(neg)
        m = scored & ((y_all == i) | (y_all == j))
        if not m.any():
            continue
        # the two-way decision the pair asks for: which of the two logits is larger
        said_pos = logits[m][:, i] > logits[m][:, j]
        truth = y_all[m] == i
        rec = float(said_pos[truth].mean()) if truth.any() else float("nan")
        spec = float((~said_pos[~truth]).mean()) if (~truth).any() else float("nan")
        acc = 0.5 * (rec + spec)
        report[key] = acc
        mark = "" if acc >= floor else "   <- below the linear floor"
        print(f"{key:30s} {acc:13.3f} {floor:12.3f}{mark}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "posture.json").write_text(
        json.dumps(
            {"fps": fps, "parameters": n_params, "pairs": report, "recall": per_class}, indent=1
        )
        + "\n"
    )
    print(f"\n-> {out}/posture.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
