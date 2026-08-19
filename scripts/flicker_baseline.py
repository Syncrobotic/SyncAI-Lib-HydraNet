#!/usr/bin/env python3
# 本檔的說明、註解與輸出訊息是繁體中文，全形標點是正確排版而非誤植；改成半形會毀掉
# 排版，出現在輸出字串裡的更屬行為變更。僅豁免 unicode 混淆檢查，其餘規則照常。
# ruff: noqa: RUF001, RUF002, RUF003
"""輸出穩定度基線量測：優化前先立量尺。

  nice -n 10 .venv/bin/python scripts/flicker_baseline.py \
    --config configs/hydranet_retail_security_b03_cw_xl.yaml \
    --checkpoint runs/hydranet_retail_security_b03_cw_xl/best.pt \
    --input datasets/studioa_clips/Kaohsiung-cam04/archive_20260816-112757_20260816-113301.mp4 \
    --static-mask datasets/studioa_static/Kaohsiung-cam04/static_20260816-112757.png \
    --out runs/flicker_baseline01

demo 影片看到的兩個症狀——seg 破碎、label 彈跳——在這裡各自變成數字，
之後每一項優化（靜態合成、軌跡渲染、logits EMA）都對著同一組數字比。

量的是四件事，每一件都分「靜態遮罩內 / 遮罩外」兩欄，因為靜態區域的翻面
全是噪音，動態區域的翻面可能是真變化：

1. 逐幀 label 翻面率，以及翻面像素中在 --bounce-window 幀內又翻回原類的
   比例（真彈跳 vs 真變化）。
2. 分割連通塊數量與小塊（面積 < --small-area-frac 幀面）佔比的逐幀分佈。
3. 偵測穩定度：逐幀框數方差；IoU >= --track-iou 貪婪跨幀配對後的框存活
   長度分佈，單幀即逝框的比例。
4. 逐類別（6 類）翻面率；wall/column/fixture 以眾數圖邊界膨脹出 ±r 邊界帶，
   帶內 vs 類內部的翻面率對比。

跨幀一致性沿用 engine/consensus.py 的 FrameConsensus 定義（同一個
agree_threshold、同一個 person 排除、同一份 caveat），不另發明平行指標；
本腳本增量補的是它不量的「相鄰幀」行為：翻面、彈跳、破碎、框存活。

彈跳（bounce）的定義：在第 t 幀翻面（label[t-1] != label[t]）的像素，若在
之後 --bounce-window 幀內任一幀讀回 label[t-1]，記為彈跳。片尾視窗未走完
的翻面屬「不可判定」，從分母中扣除而不是當成非彈跳——否則片尾必然拉低
彈跳率。

GPU 上有共用的訓練鏈時：batch 恆為 1，整個行程請用 `nice -n 10` 啟動。
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames, probe
from syncai_hydranet.engine.consensus import FrameConsensus
from syncai_hydranet.models.heads.detection import SCORE_THR_VIEW
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.device import pick_device
from syncai_hydranet.utils.visualize import crop_box, preprocess


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="flicker_baseline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="固定相機影片")
    ap.add_argument(
        "--static-mask",
        required=True,
        help="static_plates.py 產出的靜態遮罩（255=整段 clip 從未變過的像素），"
        "必須是與 clip 同一個 time slot 的那張——跨 slot 的照度差會吃掉遮罩的意義",
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="runs/flicker_baselineNN 之類的輸出目錄"
    )
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--fps", type=float, default=5.0, help="取樣 fps（與 demo 影片一致）")
    ap.add_argument("--max-frames", type=int, default=900, help="0 表示整段")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR_VIEW)
    ap.add_argument("--nms-thr", type=float, default=0.6)
    ap.add_argument(
        "--track-iou",
        type=float,
        default=0.5,
        help="跨幀貪婪配對的 IoU 門檻（同類別、僅相鄰幀）",
    )
    ap.add_argument(
        "--small-area-frac",
        type=float,
        default=0.001,
        help="連通塊面積小於（此值 × 內容區面積）記為小塊，即『破碎度』的碎片定義",
    )
    ap.add_argument(
        "--boundary-radius",
        type=int,
        default=5,
        help="邊界帶半徑（像素，對眾數圖類別區域做膨脹/侵蝕各 r 得 ±r 帶）",
    )
    ap.add_argument(
        "--boundary-classes",
        nargs="*",
        default=["wall", "column", "fixture"],
        help="要做邊界帶分析的 terrain 類別",
    )
    ap.add_argument(
        "--bounce-window",
        type=int,
        default=25,
        help="翻面後多少幀內翻回原類才算彈跳（5fps 下預設 25 幀 = 5 秒）",
    )
    ap.add_argument(
        "--box-static-thr",
        type=float,
        default=0.5,
        help="框內靜態像素占比超過此值，該框歸入『靜態區』欄",
    )
    ap.add_argument(
        "--agree-threshold",
        type=float,
        default=0.9,
        help="直接傳給 FrameConsensus.result 的 agree_threshold，沿用其定義",
    )
    ap.add_argument("--device", default=None)
    return ap


# ---------------------------------------------------------------------------
# 偵測跨幀配對
# ---------------------------------------------------------------------------


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """成對 IoU，a: [N,4] b: [M,4]（xyxy）。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def greedy_tracks(per_frame: list[dict], iou_thr: float) -> list[dict]:
    """相鄰幀、同類別、IoU 由高至低的貪婪配對，串成 track。

    刻意只配相鄰幀、不設 max-age：這裡量的是「未經任何平滑的原始輸出有多不穩」，
    允許跨幀補洞的 tracker 會把本要量的斷裂縫回去。
    """
    tracks: list[dict] = []
    active: list[dict] = []  # 每項: {label, boxes, static_fracs, last_box}
    for det in per_frame:
        boxes, labels, sfr = det["boxes"], det["labels"], det["static_frac"]
        matched_cur = np.zeros(len(boxes), dtype=bool)
        matched_act = np.zeros(len(active), dtype=bool)
        if len(active) and len(boxes):
            last = np.stack([t["last_box"] for t in active])
            m = iou_matrix(last, boxes)
            act_labels = np.array([t["label"] for t in active])
            m[act_labels[:, None] != labels[None, :]] = 0.0
            order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
            for ai, bi in order:
                if m[ai, bi] < iou_thr:
                    break
                if matched_act[ai] or matched_cur[bi]:
                    continue
                matched_act[ai] = matched_cur[bi] = True
                active[ai]["length"] += 1
                active[ai]["last_box"] = boxes[bi]
                active[ai]["static_fracs"].append(float(sfr[bi]))
        survivors = []
        for ai, t in enumerate(active):
            if matched_act[ai]:
                survivors.append(t)
            else:
                tracks.append(t)
        active = survivors
        for bi in range(len(boxes)):
            if not matched_cur[bi]:
                active.append(
                    {
                        "label": int(labels[bi]),
                        "length": 1,
                        "last_box": boxes[bi],
                        "static_fracs": [float(sfr[bi])],
                    }
                )
    tracks.extend(active)
    return tracks


# ---------------------------------------------------------------------------
# 統計小工具
# ---------------------------------------------------------------------------


def dist(values) -> dict:
    """一個逐幀量的分佈摘要。空序列回 None 欄——寧可 JSON 裡是 null 也不要編一個 0。"""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def rate(num: float, den: float) -> float | None:
    return (num / den) if den > 0 else None


def save_heatmap(arr: np.ndarray, path: Path, title: str, vmax: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.2, 6.7))
    im = ax.imshow(arr, cmap="inferno", vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config, [])
    device = pick_device(args.device or cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(select_weights(ckpt, args.weights))

    size = cfg["data"]["input_size"]  # (H, W)
    class_names: list[str] = list(cfg["data"]["terrain_classes"])
    n_cls = len(class_names)
    for c in args.boundary_classes:
        if c not in class_names:
            raise SystemExit(f"--boundary-classes {c!r} 不在 terrain_classes {class_names}")
    det_classes: list[str] = []
    det_cfg = (cfg.get("model", {}).get("heads", {}) or {}).get("detection") or {}
    det_classes = list(det_cfg.get("classes", []))

    src_w, src_h, src_fps = probe(args.input)
    sample_fps = args.fps if args.fps and args.fps < src_fps else None
    print(
        f"input {src_w}x{src_h} @ {src_fps:.2f}fps, 取樣 {args.fps}fps, "
        f"最多 {args.max_frames or '全部'} 幀, device={device}"
    )

    static_img = Image.open(args.static_mask).convert("L")

    # 沿用 consensus.py 的定義：同一個 person 排除、同一個 agree_threshold。
    consensus = FrameConsensus(class_names)

    # 逐像素累積器（在 letterbox 內容區座標；第一幀建立）。
    flip_map = None  # 每像素翻面次數
    bounce_map = None  # 每像素彈跳次數
    censored_map = None  # 片尾視窗未走完、不可判定的翻面次數
    occupancy = None  # [C,H,W] prev==c 的幀數（翻面率分母）
    flip_from = None  # [C,H,W] prev==c 且翻面的次數
    static = None  # 靜態遮罩（內容區解析度）
    pending: deque = deque()  # 彈跳判定中的翻面事件

    prev = None
    per_frame_flip = []  # (整體, 靜態內, 靜態外) 逐幀翻面率
    comp_rows = []  # 逐幀連通塊統計
    det_frames = []  # 逐幀偵測
    content_area = None
    n_frames = 0

    for frame in frames(args.input, src_w, src_h, sample_fps):
        x, _lb, region = preprocess(Image.fromarray(frame), size)
        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=args.score_thr, nms_thr=args.nms_thr)
        cur = crop_box(res["terrain"][0].cpu().numpy(), region).astype(np.int16)
        h, w = cur.shape
        x0, y0 = region[0], region[1]

        if flip_map is None:
            content_area = h * w
            flip_map = np.zeros((h, w), dtype=np.int32)
            bounce_map = np.zeros((h, w), dtype=np.int32)
            censored_map = np.zeros((h, w), dtype=np.int32)
            occupancy = np.zeros((n_cls, h, w), dtype=np.int32)
            flip_from = np.zeros((n_cls, h, w), dtype=np.int32)
            static = np.asarray(static_img.resize((w, h), Image.Resampling.NEAREST)) > 127
            n_static = int(static.sum())
            print(f"內容區 {w}x{h}, 靜態像素 {n_static} ({n_static / content_area:.1%})")

        # 首幀初始化後這些累積器恆非 None；ty 無法跨迭代收窄，以 assert 收窄。
        assert flip_map is not None and bounce_map is not None and censored_map is not None
        assert occupancy is not None and flip_from is not None and static is not None
        assert content_area is not None

        consensus.update(cur.astype(np.int64))

        # -- 1. 翻面 + 彈跳 --------------------------------------------------
        if prev is not None:
            changed = prev != cur
            flip_map += changed
            for c in range(n_cls):
                was_c = prev == c
                occupancy[c] += was_c
                flip_from[c] += was_c & changed
            per_frame_flip.append(
                (
                    float(changed.mean()),
                    float(changed[static].mean()) if static.any() else float("nan"),
                    float(changed[~static].mean()) if (~static).any() else float("nan"),
                )
            )
            # 先結算既有事件（今天的幀就是它們的「之後」）
            for ev in pending:
                back = ev["mask"] & (cur == ev["orig"])
                bounce_map += back
                ev["mask"] &= ~back
                ev["ttl"] -= 1
            while pending and pending[0]["ttl"] <= 0:
                pending.popleft()
            if changed.any():
                pending.append(
                    {"mask": changed.copy(), "orig": prev.copy(), "ttl": args.bounce_window}
                )

        # -- 2. 連通塊 -------------------------------------------------------
        small_cut = args.small_area_frac * content_area
        row = {
            "total": 0,
            "small": 0,
            "static": 0,
            "dynamic": 0,
            "small_static": 0,
            "small_dynamic": 0,
        }
        for c in range(n_cls):
            mask_c = cur == c
            if not mask_c.any():
                continue
            lab, n = ndimage.label(mask_c)
            if n == 0:
                continue
            idx = np.arange(1, n + 1)
            areas = ndimage.sum_labels(np.ones_like(lab, dtype=np.int32), lab, idx)
            static_px = ndimage.sum_labels(static.astype(np.int32), lab, idx)
            is_small = areas < small_cut
            is_static = (static_px / areas) >= 0.5
            row["total"] += int(n)
            row["small"] += int(is_small.sum())
            row["static"] += int(is_static.sum())
            row["dynamic"] += int((~is_static).sum())
            row["small_static"] += int((is_small & is_static).sum())
            row["small_dynamic"] += int((is_small & ~is_static).sum())
        comp_rows.append(row)

        # -- 3. 偵測 ---------------------------------------------------------
        det = res.get("detection", [{}])[0]
        boxes = det.get("boxes")
        if boxes is not None and len(boxes):
            b = boxes.cpu().numpy().astype(np.float64)
            b[:, 0::2] -= x0
            b[:, 1::2] -= y0
            b[:, 0::2] = b[:, 0::2].clip(0, w)
            b[:, 1::2] = b[:, 1::2].clip(0, h)
            labels = det["labels"].cpu().numpy().astype(np.int64)
            keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
            b, labels = b[keep], labels[keep]
            sfr = np.array(
                [
                    float(static[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))].mean())
                    if y2 > y1 and x2 > x1
                    else 0.0
                    for x1, y1, x2, y2 in b
                ]
            )
        else:
            b = np.zeros((0, 4))
            labels = np.zeros((0,), dtype=np.int64)
            sfr = np.zeros((0,))
        det_frames.append({"boxes": b, "labels": labels, "static_frac": sfr})

        prev = cur
        n_frames += 1
        if n_frames % 100 == 0:
            print(f"  {n_frames} 幀")
        if args.max_frames and n_frames >= args.max_frames:
            break

    if n_frames < 2:
        raise SystemExit(f"只解出 {n_frames} 幀，量不了相鄰幀行為")
    # 走到這裡至少處理了兩幀，累積器必已在首幀建立；同樣是為 ty 收窄。
    assert flip_map is not None and bounce_map is not None and censored_map is not None
    assert occupancy is not None and flip_from is not None and static is not None

    # 片尾視窗未走完的翻面：從彈跳分母扣掉，而不是當成「沒彈」。
    for ev in pending:
        censored_map += ev["mask"]

    n_trans = n_frames - 1
    result = consensus.result(args.agree_threshold)
    modal = result.modal

    # -- 匯總 1：翻面 / 彈跳 -------------------------------------------------
    def zone_stats(zone: np.ndarray) -> dict:
        zpx = int(zone.sum())
        flips = int(flip_map[zone].sum())
        evaluable = flips - int(censored_map[zone].sum())
        bounced = int(bounce_map[zone].sum())
        return {
            "pixels": zpx,
            "flip_rate": rate(flips, zpx * n_trans),
            "flips": flips,
            "bounce_share_of_flips": rate(bounced, evaluable),
            "bounced": bounced,
            "flips_evaluable": evaluable,
        }

    flip_stats = {
        "overall": zone_stats(np.ones_like(static)),
        "in_static": zone_stats(static),
        "out_static": zone_stats(~static),
    }

    per_class_flip = {}
    for c, name in enumerate(class_names):
        entry = {}
        for zname, zone in (("in_static", static), ("out_static", ~static)):
            den = int(occupancy[c][zone].sum())
            num = int(flip_from[c][zone].sum())
            entry[zname] = {"flip_rate": rate(num, den), "pixel_frames": den, "flips": num}
        entry["share_of_all_flips"] = rate(int(flip_from[c].sum()), int(flip_map.sum()))
        per_class_flip[name] = entry

    # -- 匯總 4：邊界帶（眾數圖上的類別邊界 ±r） -----------------------------
    boundary = {}
    r = args.boundary_radius
    for name in args.boundary_classes:
        c = class_names.index(name)
        region_c = modal == c
        if not region_c.any():
            boundary[name] = None
            continue
        grown = ndimage.binary_dilation(region_c, iterations=r)
        shrunk = ndimage.binary_erosion(region_c, iterations=r)
        band = grown & ~shrunk
        interior = shrunk
        boundary[name] = {
            "band_pixels": int(band.sum()),
            "interior_pixels": int(interior.sum()),
            "band_flip_rate": rate(int(flip_map[band].sum()), int(band.sum()) * n_trans),
            "interior_flip_rate": rate(
                int(flip_map[interior].sum()), int(interior.sum()) * n_trans
            ),
            "band_flip_rate_in_static": rate(
                int(flip_map[band & static].sum()), int((band & static).sum()) * n_trans
            ),
            "band_flip_rate_out_static": rate(
                int(flip_map[band & ~static].sum()), int((band & ~static).sum()) * n_trans
            ),
        }

    # -- 匯總 2：破碎度 ------------------------------------------------------
    comp_stats = {
        "total_components_per_frame": dist([r_["total"] for r_ in comp_rows]),
        "small_share_per_frame": dist(
            [r_["small"] / r_["total"] for r_ in comp_rows if r_["total"]]
        ),
        "in_static_components_per_frame": dist([r_["static"] for r_ in comp_rows]),
        "out_static_components_per_frame": dist([r_["dynamic"] for r_ in comp_rows]),
        "in_static_small_share": dist(
            [r_["small_static"] / r_["static"] for r_ in comp_rows if r_["static"]]
        ),
        "out_static_small_share": dist(
            [r_["small_dynamic"] / r_["dynamic"] for r_ in comp_rows if r_["dynamic"]]
        ),
    }

    # -- 匯總 3：偵測穩定度 --------------------------------------------------
    counts = np.array([len(d["boxes"]) for d in det_frames], dtype=np.float64)
    in_counts = np.array(
        [int((d["static_frac"] >= args.box_static_thr).sum()) for d in det_frames],
        dtype=np.float64,
    )
    out_counts = counts - in_counts
    tracks = greedy_tracks(det_frames, args.track_iou)

    def track_stats(sel: list[dict]) -> dict:
        lengths = np.array([t["length"] for t in sel], dtype=np.float64)
        if lengths.size == 0:
            return {"tracks": 0, "length": dist([]), "one_frame_share": None, "ge5_share": None}
        return {
            "tracks": int(lengths.size),
            "length": dist(lengths),
            "one_frame_share": float((lengths == 1).mean()),
            "ge5_share": float((lengths >= 5).mean()),
        }

    def t_static(t: dict) -> float:
        return float(np.mean(t["static_fracs"]))

    det_stats = {
        "frames": len(det_frames),
        "boxes_per_frame": {
            "overall": {"mean": float(counts.mean()), "var": float(counts.var())},
            "in_static": {"mean": float(in_counts.mean()), "var": float(in_counts.var())},
            "out_static": {"mean": float(out_counts.mean()), "var": float(out_counts.var())},
        },
        "tracks": {
            "overall": track_stats(tracks),
            "in_static": track_stats([t for t in tracks if t_static(t) >= args.box_static_thr]),
            "out_static": track_stats([t for t in tracks if t_static(t) < args.box_static_thr]),
        },
        "per_class": {},
    }
    for ci, cname in enumerate(det_classes):
        c_counts = np.array(
            [int((d["labels"] == ci).sum()) for d in det_frames], dtype=np.float64
        )
        det_stats["per_class"][cname] = {
            "boxes_per_frame": {"mean": float(c_counts.mean()), "var": float(c_counts.var())},
            "tracks": track_stats([t for t in tracks if t["label"] == ci]),
        }

    metrics = {
        "measured": "flicker baseline: 相鄰幀翻面/彈跳、破碎度、偵測存活、邊界帶",
        "input": str(args.input),
        "static_mask": str(args.static_mask),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "frames": n_frames,
        "transitions": n_trans,
        "sample_fps": args.fps,
        "params": {
            "score_thr": args.score_thr,
            "nms_thr": args.nms_thr,
            "track_iou": args.track_iou,
            "small_area_frac": args.small_area_frac,
            "boundary_radius": args.boundary_radius,
            "boundary_classes": args.boundary_classes,
            "bounce_window": args.bounce_window,
            "box_static_thr": args.box_static_thr,
            "agree_threshold": args.agree_threshold,
        },
        "static_pixel_share": float(static.mean()),
        "flip": flip_stats,
        "flip_per_frame": {
            "overall": dist([v[0] for v in per_frame_flip]),
            "in_static": dist([v[1] for v in per_frame_flip]),
            "out_static": dist([v[2] for v in per_frame_flip]),
        },
        "per_class_flip": per_class_flip,
        "boundary_bands": boundary,
        "fragmentation": comp_stats,
        "detection": det_stats,
        # 沿用的既有量測，原封不動放進來（含它自己的 caveat）。
        "consensus": result.to_dict(),
    }

    out_json = args.out / "metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"寫出 {out_json}")

    # -- 熱圖 ----------------------------------------------------------------
    # 標題用 ASCII：跑量測的機器上 matplotlib 多半沒有 CJK 字型，中文標題只會變豆腐字。
    flip_rate_map = flip_map / n_trans
    save_heatmap(
        flip_rate_map,
        args.out / "flip_rate_heatmap.png",
        f"per-pixel flip rate over {n_trans} frame pairs",
        vmax=float(np.percentile(flip_rate_map, 99.5)) or None,
    )
    evaluable_map = np.clip(flip_map - censored_map, 1, None)
    save_heatmap(
        np.where(flip_map > 0, bounce_map / evaluable_map, 0.0),
        args.out / "bounce_share_heatmap.png",
        "bounce share of flips (returned to original class / evaluable flips)",
        vmax=1.0,
    )
    save_heatmap(
        flip_rate_map * static,
        args.out / "flip_rate_in_static_heatmap.png",
        "flip rate inside static mask (any nonzero here is noise)",
        vmax=float(np.percentile(flip_rate_map, 99.5)) or None,
    )
    print(f"熱圖寫入 {args.out}/")
    return metrics


if __name__ == "__main__":
    main()
