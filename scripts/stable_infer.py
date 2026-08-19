#!/usr/bin/env python3
# 本檔的說明、註解與輸出訊息是繁體中文，全形標點是正確排版而非誤植；改成半形會毀掉
# 排版，出現在輸出字串裡的更屬行為變更。僅豁免 unicode 混淆檢查，其餘規則照常。
# ruff: noqa: RUF001, RUF002, RUF003
"""推論端輸出穩定化三件套，用 flicker_baseline 的同一把尺量收益。

  nice -n 10 .venv/bin/python scripts/stable_infer.py --suite \
    --config configs/hydranet_retail_security_b03_cw_xl.yaml \
    --checkpoint runs/hydranet_retail_security_b03_cw_xl/best.pt \
    --input datasets/studioa_clips/Kaohsiung-cam04/archive_20260816-112757_20260816-113301.mp4 \
    --static-mask datasets/studioa_static/Kaohsiung-cam04/static_20260816-112757.png \
    --out runs/stable01

三個機制，各自獨立開關（消融就是開關的排列組合）：

1. **靜態合成**（--static-composite）：離線先對該相機該時段的 plate 跑一次模型得
   「背景標籤板」；運行時逐幀算與 plate 的差異圖，低於門檻的像素直接貼背景板標籤。
   門檻沿 static_plates.py 的噪聲底思路：index.json 裡該 slot 的 noise_floor_used ×
   --static-mult（預設 8，即 static_plates.DYNAMIC_MULT）。在 cam04 上實測過這兩個
   母體分得很開——貼著 plate 的像素 diff 中位數 3–4 灰階，被人佔據的 50–150，
   門檻 12–32 之間曲線平坦，25 落在間隙正中。
2. **動態區 logits EMA**（--ema）：機率圖逐像素指數平滑後 argmax，--ema-alpha 是
   新幀權重。靜態合成接管的像素最終蓋上 plate 標籤，所以 EMA 實際只在未接管處生效
  （EMA 狀態全圖更新，避免像素進出接管區時狀態陳舊）。
3. **軌跡化框渲染**（--tracks）：偵測先過 offline_tracks.OfflineForward 的線上部分
  （import，不複製迴圈）——出生 0.35／維持 0.20 的遲滯二段關聯，框類別取軌跡多數決，
   漏檢幀由 Kalman 運動模型補到 --track-max-age。

量測不重寫：monkeypatch flicker_baseline.build_model 讓 flicker_baseline.main()
原封不動地跑在穩定化後的輸出上——同一條迴圈、同一份翻面／彈跳／破碎／存活算術、
同一組預設閾值。第二份算術就是第二次寫錯的機會，所以這裡一行都沒有。

對 temporal.py 自封閉失效的免疫，設計上而非測試上：plate 與背景標籤板是**離線受信任
的工件，線上絕不更新**。閘門判斷錯的最壞情況是該像素退回即時預測——即基線行為——
而不是把錯誤凍進背景裡。temporal.py 的失效鏈（線上更新 plate → 錯的地方永遠非靜態 →
永遠不修）在這裡沒有第一環。

代價，先說清楚而不是等人發現：
* 背景標籤板錯的地方會**穩定地錯**。板上仍讀成 person 的區域是 plate 的髒區
  （plate 只有它的 clip 那麼空：counter 前站了幾分鐘的人在中位數裡——cam04 這張
  實測 8.6%），預設把它標為**不受信任、永不接管**，退回即時預測＝基線行為；
  硬把「強制改成非 person 的標籤」貼上去，就是把 fixture 貼在真人身上（smoke 已重現）。
  這也意味著靜態遮罩內的翻面不會歸零——髒區像素走的仍是即時（或 EMA）路徑。
* EMA 對真變化的延遲：argmax 翻過去需要 (1-α)^n < 0.5，α=0.35 是 2 幀（0.4 秒）。
* 軌跡渲染：min-hits 2 讓新目標晚 1 幀出現；目標消失後框最多滯留 max-age 幀。

GPU 上有共用的訓練鏈時：batch 恆為 1，整個行程請用 `nice -n 10` 啟動。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flicker_baseline as fb
from offline_tracks import OfflineForward

from syncai_hydranet.data.label_maps_retail_security import get_det_vocab
from syncai_hydranet.data.video import finish_encoder, probe
from syncai_hydranet.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from syncai_hydranet.utils.visualize import (
    letterbox,
    overlay,
    preprocess,
    terrain_palette,
)


def _save_heatmap_pil(arr, path, title, vmax=None):  # noqa: ARG001  title 進不了無字型的 PIL 圖
    """fb.save_heatmap 的 PIL 替補：venv 目前沒有 matplotlib（未列在 pyproject），
    量測算術不受影響——這只是把同一個陣列畫成近似 inferno 的 PNG。"""
    a = np.asarray(arr, np.float64)
    hi = float(vmax) if vmax else float(a.max() or 1.0)
    t = np.clip(a / max(hi, 1e-9), 0.0, 1.0)
    anchors = np.array(
        [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], np.float64
    )
    pos = t * (len(anchors) - 1)
    lo = np.floor(pos).astype(np.int64)
    hi_i = np.minimum(lo + 1, len(anchors) - 1)
    frac = (pos - lo)[..., None]
    Image.fromarray((anchors[lo] * (1 - frac) + anchors[hi_i] * frac).astype(np.uint8)).save(
        path
    )


try:
    import matplotlib  # noqa: F401
except ImportError:
    fb.save_heatmap = _save_heatmap_pil


# static_plates.DYNAMIC_MULT 的同一個 8：門檻 = 噪聲底 × 8。不直接 import 是因為那個
# 常數量的是 0.5fps 時間軸上的逐像素偏差，這裡量的是單幀對 plate 的空間差異——
# 兩者共用「幾倍噪聲底」的思路，不共用同一個實測依據，寫死反而誠實。
STATIC_MULT_DEFAULT = 8.0
# 差異圖前的盒狀模糊（單邊）。plate 是 960x540 升採樣、幀是 1920x1080 降採樣，
# 重採樣銳度差集中在高頻邊緣；5px 均值濾波把它壓回噪聲底以下（實測 p99 差 <5%）。
DIFF_BLUR = 5


def _area_filter(mask: np.ndarray, min_area: float) -> np.ndarray:
    """小於 min_area 的 True 連通塊清成 False。碎不碎用面積說話，不用核大小猜。"""
    from scipy import ndimage

    cc, n = ndimage.label(mask)
    if not n:
        return mask
    areas = ndimage.sum_labels(np.ones_like(cc, dtype=np.int64), cc, np.arange(1, n + 1))
    bad = np.flatnonzero(areas < min_area) + 1
    if not len(bad):
        return mask
    return mask & ~np.isin(cc, bad)


def _fill_small_components(lab: np.ndarray, min_frac: float) -> np.ndarray:
    """把小於 min_frac × 面積的連通塊，填成最近的大塊標籤。離線、只對背景板做一次。"""
    from scipy import ndimage

    cut = min_frac * lab.size
    keep = np.ones_like(lab, dtype=bool)
    for c in np.unique(lab):
        mask = lab == c
        cc, n = ndimage.label(mask)
        if not n:
            continue
        areas = ndimage.sum_labels(np.ones_like(cc, dtype=np.int64), cc, np.arange(1, n + 1))
        small_ids = np.flatnonzero(areas < cut) + 1
        if len(small_ids):
            keep &= ~np.isin(cc, small_ids)
    if keep.all() or not keep.any():
        return lab
    _, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
    return lab[iy, ix]


# ---------------------------------------------------------------------------
# 穩定化包裝：對 flicker_baseline.main 而言它就是一個 model
# ---------------------------------------------------------------------------


class StabilisedModel:
    """包住真模型，predict() 回穩定化後的輸出，介面與 HydraNet.predict 對齊。

    flicker_baseline.main 只碰四件事：.to(device)、.eval()、.load_state_dict()、
    .predict(x, score_thr=, nms_thr=)。前三個委派，最後一個是三件套的家。
    有狀態（EMA、追蹤器、幀號），一個實例只能餵一段影片、按時間順序。
    """

    def __init__(self, inner, opts, cfg, renderer=None):
        self.inner = inner
        self.opts = opts
        self.size = tuple(cfg["data"]["input_size"])  # (H, W)
        self.class_names = list(cfg["data"]["terrain_classes"])
        self.person_idx = (
            self.class_names.index("person") if "person" in self.class_names else None
        )
        self.renderer = renderer
        self.device = torch.device("cpu")
        self.frame_idx = 0
        self._ema: torch.Tensor | None = None
        self._plate_labels: torch.Tensor | None = None
        self._plate_onehot: torch.Tensor | None = None
        self._plate_trusted_np: np.ndarray | None = None
        self._plate_blur: torch.Tensor | None = None
        self._calm_streak: np.ndarray | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self.static_thr = opts.static_thr  # None 則首幀時從 index.json 推
        self.stats: dict[str, list] = {"takeover_share": [], "ghost_person_px": []}
        self.fwd = (
            OfflineForward(
                opts.track_birth,
                opts.track_keep,
                opts.track_iou,
                opts.track_iou_low,
                opts.track_max_age,
                opts.track_min_hits,
                opts.vel_scale,
            )
            if opts.tracks
            else None
        )

    # -- flicker_baseline.main 需要的三個委派 --------------------------------
    def to(self, device):
        self.inner.to(device)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return self

    def eval(self):
        self.inner.eval()
        return self

    def load_state_dict(self, sd):
        return self.inner.load_state_dict(sd)

    # -- 工具 ----------------------------------------------------------------
    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        """正規化張量 -> [1,3,H,W] float，0..255 灰階（畫布座標）。"""
        assert self._mean is not None and self._std is not None
        return (x * self._std + self._mean).clamp(0, 1) * 255.0

    @staticmethod
    def _blur(img: torch.Tensor) -> torch.Tensor:
        k = DIFF_BLUR
        return torch.nn.functional.avg_pool2d(img, k, stride=1, padding=k // 2)

    def _resolve_static_thr(self) -> float:
        """噪聲底 × --static-mult。噪聲底出自 static_plates 的 index.json——離線工件。"""
        mask = Path(self.opts.static_mask)
        slot = mask.stem.split("_", 1)[1]
        cam = mask.parent.name
        index = mask.parent.parent / "index.json"
        if not index.exists():
            raise SystemExit(f"{index} 不存在，噪聲底無從查起；請明給 --static-thr（灰階）")
        slot_info = json.loads(index.read_text())["cameras"][cam]["slots"][slot]
        floor = slot_info.get("noise_floor_used", slot_info.get("noise_floor_used_median"))
        if floor is None:
            raise SystemExit(f"index.json 的 {cam}/{slot} 沒有 noise_floor_used 欄")
        return float(floor) * self.opts.static_mult

    def _ensure_plate(self) -> None:
        """背景標籤板與比對用 plate 畫布，一次算好、線上絕不更新（見檔頭）。

        兩道離線整理，都在「板是受信任工件」的前提下做一次、不再碰：

        * 小連通塊填平：板上小於 --plate-min-frac 的碎塊是模型在單張 plate 上的噪聲，
          貼進每一幀就成了恆定的碎片，改填最近鄰的大塊標籤。
        * 板上讀成 person 的區域標為**不受信任**，接管永不觸及（實測 cam04 這張
          plate 有 8.6% 被讀成 person——counter 前站了幾分鐘的人留在中位數裡）。
          在那裡貼「被強制改成非 person 的標籤」就是把 fixture 貼在真人身上；
          退回即時預測（＝基線行為）才是對 plate 髒區誠實的處理。
        """
        if self._plate_labels is not None:
            return
        if self.static_thr is None:
            self.static_thr = self._resolve_static_thr()
        plate_img = Image.open(self.opts.plate).convert("RGB")
        px, _lb, _region = preprocess(plate_img, self.size)
        px = px.to(self.device)
        with torch.no_grad():
            logits = self.inner.forward(px)["terrain"][0]  # [C,H,W]
        lab = logits.argmax(0).cpu().numpy().astype(np.int64)
        lab = _fill_small_components(lab, self.opts.plate_min_frac)
        trusted = np.ones_like(lab, dtype=bool)
        if self.opts.plate_no_person and self.person_idx is not None:
            trusted = lab != self.person_idx
        self._plate_labels = torch.from_numpy(lab).to(self.device)
        self._plate_onehot = (
            torch.nn.functional.one_hot(self._plate_labels, len(self.class_names))
            .permute(2, 0, 1)
            .float()
        )  # [C,H,W]，EMA 同開時的「觀測」
        self._plate_trusted_np = trusted
        self._plate_blur = self._blur(self._denorm(px))  # [1,3,H,W]
        print(
            f"背景標籤板就緒：接管門檻 {self.static_thr:.1f} 灰階，"
            f"板上不受信任（person）區 {1 - trusted.mean():.1%}"
        )

    # -- 三件套 --------------------------------------------------------------
    def predict(self, x: torch.Tensor, score_thr: float, nms_thr: float = 0.6) -> dict:
        x = x.to(self.device)
        with torch.no_grad():
            out = self.inner.forward(x)
        probs = out["terrain"].softmax(dim=1)  # [1,C,H,W]
        live_labels = probs.argmax(dim=1)  # [1,H,W]

        # 1. 靜態合成的接管遮罩（真正蓋標籤在後面，怎麼蓋取決於 EMA 是否同開）
        calm_t: torch.Tensor | None = None
        if self.opts.static:
            self._ensure_plate()
            assert self._plate_blur is not None and self._plate_labels is not None
            assert self._plate_trusted_np is not None
            diff = (self._blur(self._denorm(x)) - self._plate_blur).abs().mean(dim=1)
            dyn = (diff[0] >= self.static_thr).cpu().numpy()  # [H,W]
            area = float(dyn.size)
            if self.opts.dynamic_min_frac > 0:
                # 比人小得多的孤立 diff 島是壓縮噪聲，不是進場的東西；留著它們，
                # 接管區就是拼布，連通塊數會炸（smoke 上 37 -> 129/91 的教訓，
                # 面積制實測把 stab 塊數壓回 live +10 左右，形態學開閉都做不到）
                dyn = _area_filter(dyn, self.opts.dynamic_min_frac * area)
            if self.opts.static_dilate > 0:
                # 動態區向外長 r 像素：人影邊緣一圈交給即時預測，防拖影貼背景
                from scipy import ndimage

                dyn = ndimage.binary_dilation(dyn, iterations=self.opts.static_dilate)
            calm = ~dyn & self._plate_trusted_np
            if self.opts.calm_min_frac > 0:
                # 動態區裡的小 calm 島同理：貼上去只會多一塊碎片，不貼
                calm = _area_filter(calm, self.opts.calm_min_frac * area)
            if self.opts.static_entry > 1:
                # 進場遲滯：連續 calm 滿 k 幀才接管，退出即時。900 幀全跑實測過
                # 沒有這道門的代價——calm 閘門在門檻附近逐幀抖，板標籤/即時標籤
                # 來回切，靜態區翻面率被前緣抖動吃回原點（1.53% vs 基線 1.50%）。
                # 方向是安全的那一邊：晚接管只是多看幾幀即時預測，早放行則即刻。
                if self._calm_streak is None:
                    self._calm_streak = np.zeros(calm.shape, dtype=np.int32)
                self._calm_streak = (self._calm_streak + 1) * calm
                calm = self._calm_streak >= self.opts.static_entry
            calm_t = torch.from_numpy(calm).to(self.device)
            self.stats["takeover_share"].append(float(calm.mean()))

        # 2. 動態區 logits EMA，與兩機制的組合語義
        if self.opts.ema:
            obs = probs
            if calm_t is not None:
                # 兩者同開：接管像素把背景板 one-hot 當「觀測」餵進 EMA，而不是
                # 硬貼。硬貼的 900 幀教訓：人群邊緣的動態區逐幀掃動，板標籤與
                # 即時標籤對切，靜態區翻面率反而高於 EMA 單開（0.93% vs 0.67%）；
                # 讓 EMA 的遲滯吸收掃動，前緣要翻頁得先贏過累積的機率。
                assert self._plate_onehot is not None
                obs = torch.where(calm_t[None, None], self._plate_onehot[None], probs)
            self._ema = (
                obs
                if self._ema is None
                else self.opts.ema_alpha * obs + (1.0 - self.opts.ema_alpha) * self._ema
            )
            labels = self._ema.argmax(dim=1)
        else:
            labels = live_labels.clone()
            if calm_t is not None:
                # 靜態合成單開：規格的硬貼——差異低於門檻的像素直接貼背景板標籤
                assert self._plate_labels is not None
                labels = torch.where(calm_t[None], self._plate_labels[None], labels)

        if self.person_idx is not None:
            ghost = int(
                ((labels == self.person_idx) & (live_labels != self.person_idx)).sum().item()
            )
            self.stats["ghost_person_px"].append(ghost)

        # 3. 軌跡化框渲染
        det_head = self.inner.det_head
        assert det_head is not None, "checkpoint 沒有偵測頭"
        meta: list[dict] = []
        if self.fwd is not None:
            raw = det_head.decode(
                out["det_cls"],
                out["det_reg"],
                out["det_ctr"],
                score_thr=self.opts.track_keep,  # 低到維持閾值；出生閾值在關聯器裡
                nms_thr=nms_thr,
                img_size=self.size,
            )[0]
            boxes = raw["boxes"].cpu().numpy().astype(np.float64)
            scores = raw["scores"].cpu().numpy().astype(np.float64)
            labs = raw["labels"].cpu().numpy().astype(np.int64)
            self.fwd.update(boxes, scores, self.frame_idx)
            self._vote(boxes, scores, labs)
            det = self._tracked_boxes(meta)
        else:
            det = det_head.decode(
                out["det_cls"],
                out["det_reg"],
                out["det_ctr"],
                score_thr=score_thr,
                nms_thr=nms_thr,
                img_size=self.size,
            )[0]

        result = {"terrain": labels, "detection": [det]}
        if self.renderer is not None:
            self.renderer(self, x, labels[0], live_labels[0], det, meta)
        self.frame_idx += 1
        return result

    def _vote(self, boxes: np.ndarray, scores: np.ndarray, labs: np.ndarray) -> None:
        """把本幀觀測的類別記回吃到它的 fragment（值相等配對；框是 update 原樣 copy）。"""
        if not len(boxes):
            return
        assert self.fwd is not None, "_vote is only reached when --tracks built the forward"
        for t in self.fwd.tracks:
            if not t.frames or t.frames[-1] != self.frame_idx:
                continue
            j = int(np.abs(boxes - np.asarray(t.boxes[-1])).sum(axis=1).argmin())
            t.__dict__.setdefault("label_votes", []).append(int(labs[j]))
            t.__dict__["last_score"] = float(scores[j])

    def _tracked_boxes(self, meta: list[dict]) -> dict:
        """確認過的活軌跡 -> 框。age==0 用觀測框，否則用 Kalman 預測補到 max-age。"""
        h, w = self.size
        bs, ls, ss = [], [], []
        assert self.fwd is not None, (
            "_tracked_boxes is only reached when --tracks built the forward"
        )
        for t in self.fwd.tracks:
            if not t.confirmed:
                continue
            votes = t.__dict__.get("label_votes")
            if not votes:
                continue
            box = np.asarray(t.boxes[-1] if t.age == 0 else t.kalman.box, np.float64)
            box[0::2] = box[0::2].clip(0, w)
            box[1::2] = box[1::2].clip(0, h)
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            bs.append(box)
            ls.append(int(np.bincount(votes).argmax()))  # 類別＝軌跡多數決
            ss.append(float(t.__dict__.get("last_score", 0.0)))
            meta.append({"id": t.frag_id, "coasting": t.age > 0})
        if bs:
            return {
                "boxes": torch.from_numpy(np.stack(bs)),
                "labels": torch.from_numpy(np.asarray(ls, np.int64)),
                "scores": torch.from_numpy(np.asarray(ss, np.float64)),
            }
        return {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,)),
        }


# ---------------------------------------------------------------------------
# 渲染（只在全開那一趟掛上：影片與量測共用同一次前向，量到什麼就看見什麼）
# ---------------------------------------------------------------------------


class Renderer:
    GHOST_DUMP_PX = 1500  # 穩定化多出的 person 像素超過此數就抓拍（找拖影用）
    GHOST_DUMP_CAP = 24

    def __init__(self, out_mp4, frames_dir, region, palette, names, fps, dump_every):
        self.out_mp4 = str(out_mp4)
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.region = region  # (x0, y0, w, h)
        self.palette = palette
        self.names = names
        self.fps = fps
        self.dump_every = dump_every
        self.writer = None
        self.ghost_dumps = 0

    def __call__(self, sm, x, labels, live_labels, det, meta):
        x0, y0, cw, ch = self.region
        canvas = sm._denorm(x)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        content = Image.fromarray(canvas[y0 : y0 + ch, x0 : x0 + cw])
        lab = labels.cpu().numpy()[y0 : y0 + ch, x0 : x0 + cw]
        vis = overlay(content, lab, self.palette)
        self._draw_boxes(vis, det, meta)

        if self.writer is None:
            ow, oh = vis.width - vis.width % 2, vis.height - vis.height % 2
            self.enc_size = (ow, oh)
            self.writer = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{ow}x{oh}", "-r", f"{self.fps}", "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", self.out_mp4],
                stdin=subprocess.PIPE,
            )  # fmt: skip
        img = vis if vis.size == self.enc_size else vis.resize(self.enc_size)
        assert self.writer.stdin is not None
        self.writer.stdin.write(np.asarray(img, np.uint8).tobytes())

        # 抓拍：定期一張＋拖影候選（穩定化多出 person 的幀），三聯圖便於逐幀對照
        ghost = sm.stats["ghost_person_px"][-1] if sm.stats["ghost_person_px"] else 0
        periodic = self.dump_every and sm.frame_idx % self.dump_every == 0
        ghosty = ghost >= self.GHOST_DUMP_PX and self.ghost_dumps < self.GHOST_DUMP_CAP
        if periodic or ghosty:
            live = overlay(
                content, live_labels.cpu().numpy()[y0 : y0 + ch, x0 : x0 + cw], self.palette
            )
            trip = Image.new("RGB", (cw * 3, ch))
            for i, p in enumerate((content, live, vis)):
                trip.paste(p, (cw * i, 0))
            tag = "ghost" if ghosty and not periodic else "every"
            trip.resize((cw * 3 // 2, ch // 2)).save(
                self.frames_dir / f"{tag}_{sm.frame_idx:04d}.png"
            )
            if ghosty:
                self.ghost_dumps += 1

    def _draw_boxes(self, vis, det, meta):
        x0, y0, cw, ch = self.region
        boxes = det.get("boxes")
        if boxes is None or not len(boxes):
            return
        draw = ImageDraw.Draw(vis)
        metas = meta if len(meta) == len(boxes) else [{}] * len(boxes)
        for box, score, label, m in zip(
            boxes.cpu().numpy(), det["scores"].cpu().numpy(),
            det["labels"].cpu().numpy(), metas, strict=True,
        ):  # fmt: skip
            b = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
            if b[2] <= 0 or b[3] <= 0 or b[0] >= cw or b[1] >= ch:
                continue
            coast = m.get("coasting", False)
            color = (180, 180, 180) if coast else (255, 255, 255)
            draw.rectangle(b, outline=color, width=2)
            tid = m.get("id")
            text = f"{self.names(int(label))}"
            if tid is not None:
                text += f" #{tid}"
            text += " ~" if coast else f" {score:.2f}"
            draw.text((b[0] + 2, b[1] + 2), text, fill=color)

    def close(self):
        code = finish_encoder(self.writer)
        if code not in (None, 0):
            raise RuntimeError(f"ffmpeg exited {code} while encoding {self.out_mp4}")


# ---------------------------------------------------------------------------
# 變體執行：monkeypatch flicker_baseline.build_model，儀器一行不改
# ---------------------------------------------------------------------------


def run_variant(args, name: str, *, static: bool, ema: bool, tracks: bool, renderer=None):
    out_dir = args.out / name
    opts = argparse.Namespace(
        static=static,
        ema=ema,
        tracks=tracks,
        plate=args.plate,
        static_mask=args.static_mask,
        static_thr=args.static_thr,
        static_mult=args.static_mult,
        static_dilate=args.static_dilate,
        dynamic_min_frac=args.dynamic_min_frac,
        calm_min_frac=args.calm_min_frac,
        static_entry=args.static_entry,
        plate_min_frac=args.plate_min_frac,
        plate_no_person=not args.plate_person,
        ema_alpha=args.ema_alpha,
        track_birth=args.track_birth,
        track_keep=args.track_keep,
        track_iou=args.track_iou,
        track_iou_low=args.track_iou_low,
        track_max_age=args.track_max_age,
        track_min_hits=args.track_min_hits,
        vel_scale=args.vel_scale,
    )
    holder: dict = {}
    orig = fb.build_model

    def patched(cfg):
        holder["sm"] = StabilisedModel(orig(cfg), opts, cfg, renderer)
        return holder["sm"]

    fb.build_model = patched
    try:
        print(f"\n=== 變體 {name}：static={static} ema={ema} tracks={tracks} ===")
        metrics = fb.main(
            [
                "--config",
                args.config,
                "--checkpoint",
                args.checkpoint,
                "--input",
                args.input,
                "--static-mask",
                args.static_mask,
                "--out",
                str(out_dir),
                "--fps",
                str(args.fps),
                "--max-frames",
                str(args.max_frames),
                "--score-thr",
                str(args.score_thr),
                "--nms-thr",
                str(args.nms_thr),
            ]
        )
    finally:
        fb.build_model = orig
    sm = holder["sm"]
    stats = {
        "static_thr": sm.static_thr,
        "takeover_share_mean": (
            float(np.mean(sm.stats["takeover_share"])) if sm.stats["takeover_share"] else None
        ),
        "ghost_person_px": fb.dist(sm.stats["ghost_person_px"]),
        "mechanisms": {"static": static, "ema": ema, "tracks": tracks},
        "params": {
            k: getattr(opts, k)
            for k in (
                "static_mult",
                "static_dilate",
                "dynamic_min_frac",
                "calm_min_frac",
                "static_entry",
                "plate_min_frac",
                "plate_no_person",
                "ema_alpha",
                "track_birth",
                "track_keep",
                "track_max_age",
                "track_min_hits",
            )
        },
    }
    (out_dir / "stabiliser_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n"
    )
    return metrics, stats


def summarise(m: dict) -> dict:
    """metrics.json -> 對照表的一列。只挑欄位，不再做任何新算術。"""
    det = m["detection"]
    frag = m["fragmentation"]
    fx = m["boundary_bands"].get("fixture")
    person = m["per_class_flip"].get("person", {})
    return {
        "flip_rate_in_static": m["flip"]["in_static"]["flip_rate"],
        "flip_rate_out_static": m["flip"]["out_static"]["flip_rate"],
        "bounce_share_in_static": m["flip"]["in_static"]["bounce_share_of_flips"],
        "bounce_share_overall": m["flip"]["overall"]["bounce_share_of_flips"],
        "person_flip_in_static": (person.get("in_static") or {}).get("flip_rate"),
        "person_pixel_frames_in_static": (person.get("in_static") or {}).get("pixel_frames"),
        "fixture_band_flip_rate": fx["band_flip_rate"] if fx else None,
        "fixture_band_over_interior": (
            fx["band_flip_rate"] / fx["interior_flip_rate"]
            if fx and fx["interior_flip_rate"]
            else None
        ),
        "components_mean": frag["total_components_per_frame"]["mean"],
        "components_p90": frag["total_components_per_frame"]["p90"],
        "small_share_mean": frag["small_share_per_frame"]["mean"],
        "boxes_per_frame_mean": det["boxes_per_frame"]["overall"]["mean"],
        "boxes_per_frame_var": det["boxes_per_frame"]["overall"]["var"],
        "tracks": det["tracks"]["overall"]["tracks"],
        "track_len_p50": det["tracks"]["overall"]["length"]["p50"],
        "track_len_mean": det["tracks"]["overall"]["length"]["mean"],
        "one_frame_share": det["tracks"]["overall"]["one_frame_share"],
        "ge5_share": det["tracks"]["overall"]["ge5_share"],
    }


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="stable_infer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="固定相機影片（與基線同段）")
    ap.add_argument("--static-mask", required=True, help="static_plates.py 的同 slot 遮罩")
    ap.add_argument(
        "--plate",
        default=None,
        help="同 slot 的 plate（預設由 static-mask 檔名推：static_ -> plate_）",
    )
    ap.add_argument("--out", type=Path, required=True, help="runs/stable01")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--score-thr", type=float, default=0.30, help="非軌跡路徑的偵測閾值")
    ap.add_argument("--nms-thr", type=float, default=0.6)
    # 機制開關（單跑模式用；--suite 自己排消融）
    ap.add_argument("--static-composite", action="store_true", dest="static_on")
    ap.add_argument("--ema", action="store_true", dest="ema_on")
    ap.add_argument("--tracks", action="store_true", dest="tracks_on")
    ap.add_argument("--suite", action="store_true", help="跑 基線引用＋三個單開＋全開 全套")
    ap.add_argument(
        "--baseline-metrics",
        default="runs/flicker_baseline01/metrics.json",
        help="既有基線的 metrics.json（同儀器同段；引用而不重跑）",
    )
    # 機制 1
    ap.add_argument("--static-thr", type=float, default=None, help="灰階；預設噪聲底×mult")
    ap.add_argument("--static-mult", type=float, default=STATIC_MULT_DEFAULT)
    ap.add_argument("--static-dilate", type=int, default=4, help="動態區外擴像素，防貼背景拖影")
    ap.add_argument(
        "--dynamic-min-frac",
        type=float,
        default=0.001,
        help="小於此面積比的動態島視為噪聲、改判 calm（0 停用）",
    )
    ap.add_argument(
        "--calm-min-frac",
        type=float,
        default=0.003,
        help="小於此面積比的 calm 島不接管（0 停用）",
    )
    ap.add_argument(
        "--static-entry",
        type=int,
        default=5,
        help="連續 calm 滿此幀數才接管（進場遲滯，防前緣抖動；退出恆即時；<=1 停用）",
    )
    ap.add_argument(
        "--plate-min-frac", type=float, default=0.001, help="背景板小於此面積比的碎塊填平"
    )
    ap.add_argument(
        "--plate-person",
        action="store_true",
        help="信任背景板的 person 區並照貼（預設：板上 person 區＝plate 髒區，永不接管）",
    )
    # 機制 2
    ap.add_argument("--ema-alpha", type=float, default=0.35, help="新幀權重；愈小愈平滑愈延遲")
    # 機制 3
    ap.add_argument("--track-birth", type=float, default=0.35, help="出生閾值（遲滯上緣）")
    ap.add_argument("--track-keep", type=float, default=0.20, help="維持閾值（遲滯下緣）")
    ap.add_argument("--track-iou", type=float, default=0.3)
    ap.add_argument("--track-iou-low", type=float, default=0.4)
    ap.add_argument("--track-max-age", type=int, default=5, help="漏檢幀 Kalman 補到此為止")
    ap.add_argument("--track-min-hits", type=int, default=2, help="連中幾幀才確認（殺單幀框）")
    ap.add_argument("--vel-scale", type=float, default=None, help="預設 25/取樣fps")
    ap.add_argument("--dump-every", type=int, default=100, help="每 N 幀存一張三聯對照圖")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plate is None:
        m = Path(args.static_mask)
        args.plate = str(m.with_name(m.name.replace("static_", "plate_", 1)))
    if not Path(args.plate).exists():
        raise SystemExit(f"plate 不存在：{args.plate}")
    if args.vel_scale is None:
        args.vel_scale = 25.0 / args.fps

    cfg = fb.load_config(args.config, [])
    size = cfg["data"]["input_size"]
    palette = terrain_palette(cfg["data"].get("terrain_classes"))
    vocabs = {d["det_vocab"] for d in cfg["data"].get("datasets", []) if d.get("det_vocab")}
    if len(vocabs) == 1:
        det_names = list(get_det_vocab(next(iter(vocabs))).classes)
    else:
        det_names = list(
            ((cfg.get("model", {}).get("heads", {}) or {}).get("detection") or {}).get(
                "classes", []
            )
        )
    name_of = lambda i: det_names[i] if i < len(det_names) else str(i)  # noqa: E731

    src_w, src_h, _ = probe(args.input)
    _, region = letterbox(Image.new("RGB", (src_w, src_h)), size)

    if not args.suite:
        renderer = None
        if args.static_on or args.ema_on or args.tracks_on:
            renderer = Renderer(
                args.out / "stable_cam04.mp4", args.out / "frames", region, palette,
                name_of, args.fps, args.dump_every,
            )  # fmt: skip
        try:
            metrics, stats = run_variant(
                args, "single",
                static=args.static_on, ema=args.ema_on, tracks=args.tracks_on,
                renderer=renderer,
            )  # fmt: skip
        finally:
            if renderer is not None:
                renderer.close()
        print(json.dumps(summarise(metrics), ensure_ascii=False, indent=1))
        return 0

    # ---- 全套：基線（引用）＋三個單開＋全開（全開那趟同時渲染影片）------------
    baseline = json.loads(Path(args.baseline_metrics).read_text())
    rows: dict[str, dict] = {"baseline": summarise(baseline)}
    stab: dict[str, dict] = {}

    for name, mech in (
        ("static_only", {"static": True, "ema": False, "tracks": False}),
        ("ema_only", {"static": False, "ema": True, "tracks": False}),
        ("tracks_only", {"static": False, "ema": False, "tracks": True}),
    ):
        metrics, stats = run_variant(args, name, **mech)
        rows[name] = summarise(metrics)
        stab[name] = stats

    renderer = Renderer(
        args.out / "stable_cam04.mp4", args.out / "full" / "frames", region, palette,
        name_of, args.fps, args.dump_every,
    )  # fmt: skip
    try:
        metrics, stats = run_variant(
            args, "full", static=True, ema=True, tracks=True, renderer=renderer
        )
    finally:
        renderer.close()
    rows["full"] = summarise(metrics)
    stab["full"] = stats

    table = {
        "measured": "輸出穩定化對照：基線 vs 各機制單開 vs 全開（同儀器同段同閾值）",
        "instrument": "scripts/flicker_baseline.py（monkeypatch build_model 重用其 main）",
        "baseline_source": str(args.baseline_metrics),
        "input": args.input,
        "static_mask": args.static_mask,
        "plate": args.plate,
        "frames": args.max_frames,
        "sample_fps": args.fps,
        "rows": rows,
        "stabiliser": stab,
    }
    out_json = args.out / "metrics.json"
    out_json.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    print(f"\n對照表寫出 {out_json}")

    cols = list(rows)
    keys = list(next(iter(rows.values())))
    print(f"\n{'指標':32s}" + "".join(f"{c:>14s}" for c in cols))
    for k in keys:
        line = f"{k:32s}"
        for c in cols:
            v = rows[c].get(k)
            line += f"{v:14.4f}" if isinstance(v, float) else f"{v!s:>14s}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
