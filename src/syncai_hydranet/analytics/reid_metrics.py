"""Metrics for identity: does a track follow one person, and does an embedding know them.

**Why this exists before any re-identification model does.** `tracker.py` produces tracks
and `dwell.py` turns them into dwell, and until now nothing measured whether the tracks
were right. The consequence is already on record: one 4.6-minute site clip fragments into
**1234 tracks** with a median length of **9-16 frames** at 5 fps. Every analytic
downstream counts one shopper many times over -- and biased, not merely noisy, because a
shopper who lingers fragments into more tracks than one who walks through, so the
over-count scales with exactly the quantity `dwell.py` exists to measure.

That number was a symptom nobody could act on, because improving a tracker is
unfalsifiable without a metric. These are the two families that make it falsifiable:

**Tracking identity** -- `idf1`, `id_switches`, `fragmentation`. They compare predicted
tracks against ground-truth tracks on a video and answer "does one track follow one
person". A labelled site clip does not exist yet; these work the moment one does, and
they are tested here against hand-built sequences whose right answer is countable by eye.

**Retrieval** -- `cmc_map`. Given an embedding per crop, does the same person rank first
across cameras. This one runs today against `datasets/Market-1501`, which is why the
association encoder can be measured before it is wired into anything.

**The floor, measured 2026-08-17 so that the first real encoder has something to beat.**
An ImageNet-pretrained `resnet18` backbone with no re-identification training at all,
global-average-pooled to 512-d, over the standard 3,368-query / 19,732-gallery protocol:

    mAP 0.0318    rank1 0.0962    rank5 0.1986    rank10 0.2699

All 3,368 queries scored, which is the published query count and the cheapest check that
the protocol ran rather than something adjacent to it. Any encoder built for association
has to clear this or it is contributing nothing an off-the-shelf backbone did not.

Pure numpy, and no `motmetrics` or `scipy` dependency: this package installs on robots.
The Hungarian assignment IDF1 needs is written out below at the size these problems are
(tens of tracks), where an O(n^3) implementation is not the bottleneck.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

# --------------------------------------------------------------- tracking identity


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ar_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = ar_a[:, None] + ar_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment. Written out because scipy is not a dependency here.

    Jonker-Volgenant is not worth the lines at this scale; this is the O(n^3) Hungarian
    algorithm on a padded square matrix, which for tens of identities is instant and,
    unlike a greedy pass, is actually optimal. Greedy matters here: IDF1 is *defined* as
    the best possible identity assignment, so a greedy approximation would report a
    tracker as worse than it is and the error would grow with the number of people.
    """
    n, m = cost.shape
    size = max(n, m)
    big = float(cost.max()) + 1.0 if cost.size else 1.0
    c = np.full((size, size), big, dtype=float)
    if cost.size:
        c[:n, :m] = cost

    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, size + 1) if p[j] - 1 < n and j - 1 < m]


def idf1(
    gt: dict[int, dict[int, np.ndarray]],
    pred: dict[int, dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Identity F1: the global best one-to-one mapping between predicted and true tracks.

    Both arguments are ``{track id: {frame index: xyxy box}}``.

    IDF1 rather than MOTA as the headline, and the difference is the whole point here. A
    tracker that detects every person perfectly but swaps their identities every few
    frames scores well on MOTA -- which counts detection errors frame by frame and charges
    only a small constant for each switch -- and badly on IDF1, which asks how much of
    each person's trajectory carries one consistent id. **Fragmentation is precisely the
    failure MOTA under-reports and IDF1 does not**, and fragmentation is this project's
    measured problem.

    Returns ``idf1``, ``idp`` (precision), ``idr`` (recall), and the raw counts.
    """
    gt_ids, pred_ids = sorted(gt), sorted(pred)
    if not gt_ids:
        return {"idf1": float("nan"), "idp": float("nan"), "idr": float("nan"),
                "idtp": 0, "idfp": 0, "idfn": 0}  # fmt: skip

    # cost[i, j] = frames where gt i and pred j do NOT agree, if paired
    cost = np.zeros((len(gt_ids), len(pred_ids)))
    tp = np.zeros((len(gt_ids), len(pred_ids)))
    for i, g in enumerate(gt_ids):
        gframes = gt[g]
        for j, p in enumerate(pred_ids):
            pframes = pred[p]
            shared = gframes.keys() & pframes.keys()
            hits = 0
            for f in shared:
                a = np.asarray(gframes[f], float)[None]
                b = np.asarray(pframes[f], float)[None]
                if _iou_matrix(a, b)[0, 0] >= iou_threshold:
                    hits += 1
            tp[i, j] = hits
            cost[i, j] = (len(gframes) - hits) + (len(pframes) - hits)

    idtp = 0
    if len(pred_ids):
        for i, j in _hungarian(cost):
            idtp += int(tp[i, j])
    n_gt = sum(len(v) for v in gt.values())
    n_pred = sum(len(v) for v in pred.values())
    idfn, idfp = n_gt - idtp, n_pred - idtp
    idp = idtp / n_pred if n_pred else float("nan")
    idr = idtp / n_gt if n_gt else float("nan")
    denom = 2 * idtp + idfp + idfn
    return {
        "idf1": (2 * idtp / denom) if denom else float("nan"),
        "idp": idp,
        "idr": idr,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
    }


def id_switches(
    gt: dict[int, dict[int, np.ndarray]],
    pred: dict[int, dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """How often a true identity changes which predicted track carries it, and how badly.

    ``switches`` is the count. ``fragmentation`` is the number of predicted tracks a true
    identity is spread across, averaged over identities and reported as
    ``tracks_per_identity`` -- **the direct measure of this project's problem.** One is
    perfect; the site clip's 1234 tracks imply a large number, and this is the quantity a
    re-identification embedding exists to push back toward one.

    ``mostly_tracked`` counts identities where one predicted track covers at least 80% of
    the frames, the usual MOT definition, because an average hides whether the failure is
    everywhere or concentrated.
    """
    switches = 0
    per_id_tracks, mostly = [], 0
    for gframes in gt.values():
        assigned: dict[int, int | None] = {}
        for f in sorted(gframes):
            box = np.asarray(gframes[f], float)[None]
            best, best_iou = None, iou_threshold
            for p, pframes in pred.items():
                if f not in pframes:
                    continue
                v = _iou_matrix(box, np.asarray(pframes[f], float)[None])[0, 0]
                if v >= best_iou:
                    best, best_iou = p, v
            assigned[f] = best
        seen = [v for v in assigned.values() if v is not None]
        distinct = set(seen)
        per_id_tracks.append(len(distinct) if distinct else 0)
        for a, b in pairwise(seen):
            if a != b:
                switches += 1
        if distinct:
            top = max(seen.count(x) for x in distinct)
            if top >= 0.8 * len(gframes):
                mostly += 1
    n = len(gt) or 1
    return {
        "switches": switches,
        "tracks_per_identity": float(np.mean(per_id_tracks)) if per_id_tracks else float("nan"),
        "mostly_tracked": mostly,
        "mostly_tracked_frac": mostly / n,
    }


# ------------------------------------------------------------------- retrieval


def cmc_map(
    query_feat: np.ndarray,
    query_pid: np.ndarray,
    query_cam: np.ndarray,
    gallery_feat: np.ndarray,
    gallery_pid: np.ndarray,
    gallery_cam: np.ndarray,
    ranks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Market-1501 protocol: mean average precision and cumulative match characteristic.

    **The same-camera exclusion is the protocol, not a detail.** A gallery crop of the
    same person from the same camera is a near-duplicate frame, and counting it turns the
    benchmark into "can you match an image to itself". Every published Market-1501 number
    excludes it, so a number computed without the exclusion is not comparable to any of
    them -- and it is a specific, large over-estimate rather than a small bias.

    Distractors (`pid == -1` in Market-1501's filenames) stay in the gallery: they are
    what makes the ranking hard, and dropping them inflates every number.

    Features are L2-normalised here rather than assumed normalised, so a caller passing
    raw backbone output gets cosine similarity and not whatever its magnitudes encode.
    """
    q = query_feat / np.maximum(np.linalg.norm(query_feat, axis=1, keepdims=True), 1e-12)
    g = gallery_feat / np.maximum(np.linalg.norm(gallery_feat, axis=1, keepdims=True), 1e-12)
    sim = q @ g.T
    order = np.argsort(-sim, axis=1)

    aps, hits = [], np.zeros(max(ranks))
    counted = 0
    for i in range(len(q)):
        keep = ~(
            (gallery_pid[order[i]] == query_pid[i]) & (gallery_cam[order[i]] == query_cam[i])
        )
        keep &= gallery_pid[order[i]] != -1
        match = (gallery_pid[order[i]] == query_pid[i])[keep]
        if not match.any():
            continue  # this identity appears on no other camera; it cannot be scored
        counted += 1
        first = int(np.argmax(match))
        for r, k in enumerate(ranks):
            if first < k:
                hits[r] += 1
        cum = np.cumsum(match)
        prec = cum / (np.arange(len(match)) + 1)
        aps.append(float((prec * match).sum() / match.sum()))

    out = {"mAP": float(np.mean(aps)) if aps else float("nan"), "queries_scored": counted}
    for r, k in enumerate(ranks):
        out[f"rank{k}"] = float(hits[r] / counted) if counted else float("nan")
    return out
