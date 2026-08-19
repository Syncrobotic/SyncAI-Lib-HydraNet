#!/usr/bin/env python3
# 本檔的說明、註解與輸出訊息是繁體中文，全形標點是正確排版而非誤植；改成半形會毀掉
# 排版，出現在輸出字串裡的更屬行為變更。僅豁免 unicode 混淆檢查，其餘規則照常。
# ruff: noqa: RUF001, RUF002, RUF003
"""機隊開通標定掃描：把 calib01／calib02 驗證過的每相機空間標定串成可重複的流程。

  nice -n 10 .venv/bin/python scripts/onboard_camera.py --out runs/onboard01
  nice -n 10 .venv/bin/python scripts/onboard_camera.py --camera Taichung-cam01
  .venv/bin/python scripts/onboard_camera.py --report-only --out runs/onboard01

對每台 selling_floor 相機（`datasets/studioa_clips/cameras.json`）：

1. **方位**：挑白天 plate，重用 `calibrate_from_plate.py` 的管線（import，不複製幾何
   算術——第二份算術是第二次寫錯的機會）：undistort（除法模型 k1 = −0.225，機隊硬體
   假設，僅 Taichung-cam01 由磚縫網格實測）→ DA-V2 一次 → RANSAC「最低可信水平面」
   → pitch／roll／plane 高。calib01 已證明這條路俯角對錨點 ±0.7°（vfov 釘對的前提下）。
2. **尺度（可自動子集）**：DA-V2 的公尺不可信（本機隊高估 1.45–1.6×，calib01），
   自動化的是人高統計——`datasets/retail_person_gdino01` 的框過同一套閘（score ≥ 0.5、
   邊緣、長寬比）推過擬合平面，中位數對 1.70 m 先驗回推尺度修正。**≥ 10 個可用高度
   才出數**，不足標 `unmeasured`。門高／地磚目視（calib02 的 ±5–8% 法）無法自動，
   欄位留 null、標 `needs_visual_reference`。
3. **量測紀律**：每台 vfov 55／70.4／85 三點敏感度全跑全存；主值取 70.4°（cam01 磚縫
   釘定值，其餘機隊假設）。單一數字不帶帶寬不出表——帶寬就是 vfov 掃描的散佈。
4. **plate 髒區**：HydraNet（stable01 的同一組 config／checkpoint）在 plate 上讀成
   person 的比例——stable_infer 的經驗值：Kaohsiung-cam04 的 112757 板是 8.6%。
   髒板的意義：中位數裡站著人，靜態合成在那裡永不接管，plate 上的幾何量測也別取那區。

輸出 `runs/onboard01/<camera>.calib.json`（欄位名含單位，供 hydranet-scene／events
消費）與全機隊 `runs/onboard01/REPORT.md`。

GPU 上有共用的訓練鏈時：DA-V2 與 HydraNet 都是單張、batch 恆為 1，整個行程請用
`nice -n 10` 啟動。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import calibrate_from_plate as cfp  # noqa: E402  幾何與檢核全部重用

SCHEMA = "hydranet-onboard-calib/v1"
CAMERAS_JSON = Path("datasets/studioa_clips/cameras.json")
PERSON_ANNS = Path("datasets/retail_person_gdino01/annotations/instances_all.json")
PLATE_MODEL_CONFIG = Path("configs/hydranet_retail_security_b03_cw_xl.yaml")
PLATE_MODEL_CKPT = Path("runs/hydranet_retail_security_b03_cw_xl/best.pt")
K1_FLEET = -0.225  # Taichung-cam01 磚縫網格實測；其餘相機是同機隊硬體假設
VFOV_PRIMARY = 70.4  # 同上：cam01 釘定，其餘機隊假設
MIN_HEIGHTS = 10  # 人高統計出數的最低樣本數（任務規格）
DIRTY_PLATE_FRAC = 0.05  # person 佔比超過此值標髒板（cam04 經驗值 8.6% 在此之上）
PERSON_PRIOR_SYS_FRAC = 0.11  # calib01：人高尺度 vs 磚縫錨點的縫隙（姿態偏差，系統項）
VFOV_UNPINNED_SYS_FRAC = 0.05  # calib02：vfov 未釘定相機的 ±5% 系統項


# ---------------------------------------------------------------------------
# DA-V2 pipeline 快取：cfp.run_depth 每次呼叫都重建 pipeline，掃 22 台會重載 22 次
# Large 權重。這裡把 transformers.pipeline 換成鍵值快取版——run_depth 的前後處理
# 一行不動，只是同一組 (task, model, device) 回傳同一個實例。


def _install_pipeline_cache() -> None:
    import transformers

    real = transformers.pipeline
    cache: dict = {}

    def cached(task, model=None, device=None, **kw):
        key = (task, model, device)
        if key not in cache:
            cache[key] = real(task, model=model, device=device, **kw)
        return cache[key]

    transformers.pipeline = cached


# ---------------------------------------------------------------------------
# plate 髒區：stable_infer 的同一種量法（同 config／checkpoint／preprocess／argmax），
# 只是不建背景板，只回報 person 佔比。


class PlatePersonMeter:
    def __init__(self, config: Path, checkpoint: Path):
        import torch

        from syncai_hydranet.config import load_config
        from syncai_hydranet.models.hydranet import build_model
        from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
        from syncai_hydranet.utils.device import pick_device

        self._torch = torch
        cfg = load_config(str(config), [])
        self.device = pick_device(cfg.get("device"))
        self.model = build_model(cfg).to(self.device).eval()
        self.model.load_state_dict(select_weights(load_checkpoint(checkpoint), "ema"))
        self.size = tuple(cfg["data"]["input_size"])  # (H, W)
        self.classes = list(cfg["data"]["terrain_classes"])
        self.person_idx = self.classes.index("person")

    def person_frac(self, plate: Image.Image) -> dict:
        from syncai_hydranet.utils.visualize import crop_box, preprocess

        px, _canvas, region = preprocess(plate, self.size)
        with self._torch.no_grad():
            lab = self.model.forward(px.to(self.device))["terrain"][0].argmax(0).cpu().numpy()
        content = crop_box(lab, region)
        return {
            "person_frac_plate": round(float((content == self.person_idx).mean()), 4),
            # stable_infer 印的是整張畫布的比例（含 letterbox 邊），留著對表
            "person_frac_canvas": round(float((lab == self.person_idx).mean()), 4),
        }


# ---------------------------------------------------------------------------
# 單台相機


def selling_floor_cameras() -> list[str]:
    d = json.loads(CAMERAS_JSON.read_text())
    return sorted(c for c, v in d["cameras"].items() if v.get("role") == "selling_floor")


def _band(vals: list[float | None]) -> list[float] | None:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return [round(min(xs), 2), round(max(xs), 2)]


def onboard_one(
    camera: str,
    vfovs: list[float],
    k1: float,
    meter: PlatePersonMeter | None,
    plates_root: Path,
) -> dict:
    now = _dt.date.today().isoformat()
    is_pinned = camera == "Taichung-cam01"  # 磚縫網格：k1 與 vfov 皆實測
    flags: list[str] = []
    if not is_pinned:
        flags += ["vfov_fleet_assumed", "k1_fleet_assumed"]

    result: dict = {
        "schema": SCHEMA,
        "camera": camera,
        "generated": now,
        "vfov_assumed_deg": VFOV_PRIMARY,
        "vfov_source": "tile_grid_pinned" if is_pinned else "fleet_hardware_assumed",
        "k1_division_model": k1,
        "k1_source": "tile_grid_measured" if is_pinned else "fleet_hardware_assumed",
        # calib02 的目視先驗（門高／地磚）不可自動化：欄位保留、值留空，等 SOP 的
        # 人工步驟回填。這是「未量測」，不是「量出 null」。
        "visual_reference": {
            "door_height_implied_height_m": None,
            "tile_pitch_implied_height_m": None,
            "status": "needs_visual_reference",
        },
        "provenance": {
            "orientation_pipeline": "scripts/calibrate_from_plate.py (imported)",
            "depth_model": cfp.MODEL,
            "person_boxes": str(PERSON_ANNS),
            "person_height_prior_m": cfp.ADULT_M,
            "plate_person_model": f"{PLATE_MODEL_CONFIG} + {PLATE_MODEL_CKPT} (ema)",
        },
    }

    cam_dir = plates_root / camera
    if not cam_dir.is_dir():
        result.update(
            {
                "plate_used": None,
                "pitch_deg": None,
                "roll_deg": None,
                "height_m": None,
                "height_source": "unmeasured",
                "scale": None,
                "scale_source": "unmeasured",
                "uncertainty": None,
                "flags": [*flags, "no_plate", "scale_unmeasured", "needs_visual_reference"],
            }
        )
        return result

    slot = cfp.pick_daytime_slot(cam_dir)
    plate_path = cam_dir / f"plate_{slot}.png"
    rgb = np.asarray(Image.open(plate_path).convert("RGB"))
    h, w = rgb.shape[:2]
    result.update(
        {"plate_used": str(plate_path), "plate_slot_utc": slot, "frame_hw_px": [h, w]}
    )

    # plate 髒區（原始 plate，非 undistort 後——stable_infer 量的就是原板）
    if meter is not None:
        result.update(meter.person_frac(Image.fromarray(rgb)))
        if result["person_frac_plate"] > DIRTY_PLATE_FRAC:
            flags.append("dirty_plate_person_frac_gt_0p05")

    # 方位：undistort → DA-V2 一次 → 每個 vfov 各自 RANSAC 選地板
    rgb_u = cfp.undistort_image(rgb, k1)
    depth = cfp.run_depth(rgb_u)

    boxes = cfp.load_person_boxes(PERSON_ANNS, camera, w, h, k1)
    result["person_boxes_after_gates"] = len(boxes)

    by_vfov: list[dict] = []
    for vfov in vfovs:
        cam = cfp.Camera.from_vfov(h, w, vfov)
        cands = cfp.floor_candidates(depth, cam, inlier_m=0.05)
        plane, residual, cand_rows = cfp.choose_floor(cands, 0.05)
        row: dict = {"vfov_deg": vfov, "candidates": cand_rows}
        if plane is None:
            row["failed"] = "no plausible floor plane among RANSAC candidates"
            flags.append(f"no_floor_at_vfov_{vfov:g}")
            by_vfov.append(row)
            continue
        import math

        pitch_deg = math.degrees(plane.pitch)
        hrow = cfp.horizon_row(pitch_deg, cam.fy, h)
        finite = np.isfinite(residual)
        row.update(
            {
                "height_dav2_raw_m": round(plane.height, 3),
                "pitch_deg": round(pitch_deg, 2),
                "roll_deg": round(math.degrees(plane.roll), 2),
                "inlier_frac_frame": round(float((np.abs(residual[finite]) < 0.05).mean()), 4),
                "horizon_row": round(hrow, 1),
                "horizon_inside_frame": bool(0 <= hrow < h),
                "column": cfp.column_health(cam, plane, h, w),
                "floor_scale_dav2_raw": cfp.floor_scale(cam, plane, h, w),
            }
        )
        if len(boxes):
            row["person"] = cfp.person_checks(boxes, cam, plane, (h, w), vfov)
        by_vfov.append(row)
    result["by_vfov"] = by_vfov

    ok_rows = [r for r in by_vfov if "pitch_deg" in r]
    primary = next((r for r in by_vfov if r["vfov_deg"] == VFOV_PRIMARY), None)
    if primary is None or "pitch_deg" not in primary:
        primary = ok_rows[0] if ok_rows else None
        if primary is not None:
            flags.append("primary_vfov_failed_using_fallback")

    if primary is None:
        result.update(
            {
                "pitch_deg": None,
                "roll_deg": None,
                "height_m": None,
                "height_source": "unmeasured",
                "scale": None,
                "scale_source": "unmeasured",
                "uncertainty": None,
                "flags": [
                    *flags,
                    "no_floor_any_vfov",
                    "scale_unmeasured",
                    "needs_visual_reference",
                ],
            }
        )
        return result

    if primary.get("horizon_inside_frame"):
        flags.append("horizon_inside_frame_pose_rejected")
    if abs(primary["roll_deg"]) > 5.0:
        flags.append("roll_abs_gt_5deg")

    # 尺度：人高統計（可自動子集）。scale 是「乘在 DA-V2 平面量上的公尺修正」。
    person = primary.get("person", {})
    n_heights = person.get("n_heights", 0)
    scale = person.get("scale_person")
    scale_rows = [(r["vfov_deg"], r.get("person", {}).get("scale_person")) for r in ok_rows]
    if scale is not None and n_heights >= MIN_HEIGHTS:
        med = person["implied_person_height_med_m"]
        mad = person["implied_person_height_mad_m"]
        height_m = round(primary["height_dav2_raw_m"] * scale, 2)
        height_src = "dav2_plane_height_x_person_height_scale"
        scale_src = f"person_height_median_vs_{cfp.ADULT_M}m_prior_n{n_heights}"
        stat_frac = round(mad / med, 3)
        # 標定高的 vfov 帶：每個 vfov 用它自己的 plane 高 × 它自己的人高尺度——
        # 兩者同向隨 vfov 滑動，乘積比裸高穩定，帶寬會量出這件事而不是假設它
        heights_scaled = []
        for r in ok_rows:
            s = r.get("person", {}).get("scale_person")
            if s:
                heights_scaled.append(round(r["height_dav2_raw_m"] * s, 2))
        scale_band = _band([s for _, s in scale_rows])
    else:
        height_m, height_src = None, "unmeasured"
        scale, scale_src = None, "unmeasured"
        stat_frac, heights_scaled, scale_band = None, [], None
        flags += ["scale_unmeasured", "needs_visual_reference"]
        if n_heights:
            flags.append(f"person_heights_n{n_heights}_below_min{MIN_HEIGHTS}")

    unc: dict = {
        "vfov_scan_deg": [r["vfov_deg"] for r in ok_rows],
        "pitch_deg_band_vfov_scan": _band([r["pitch_deg"] for r in ok_rows]),
        "pitch_deg_vs_anchor_at_pinned_vfov": 0.7,  # calib01 磚縫錨點交叉檢核
        "roll_deg_band_vfov_scan": _band([r["roll_deg"] for r in ok_rows]),
        "height_dav2_raw_m_band_vfov_scan": _band([r["height_dav2_raw_m"] for r in ok_rows]),
        "height_m_band_vfov_scan": _band(heights_scaled) if heights_scaled else None,
        "scale_band_vfov_scan": scale_band,
        "scale_frac_stat_person_mad": stat_frac,
        "scale_frac_sys_person_prior": PERSON_PRIOR_SYS_FRAC if scale else None,
        "scale_frac_sys_vfov_unpinned": (None if is_pinned else VFOV_UNPINNED_SYS_FRAC),
        "dominant_unknown": "vfov",  # calib01／calib02 的一致結論
    }

    result.update(
        {
            "pitch_deg": primary["pitch_deg"],
            "roll_deg": primary["roll_deg"],
            "height_m": height_m,
            "height_source": height_src,
            "height_dav2_raw_m": primary["height_dav2_raw_m"],
            "scale": scale,
            "scale_source": scale_src,
            "floor_m_per_px_u_at_ref": (
                round(primary["floor_scale_dav2_raw"]["m_per_px_u"] * scale, 4)
                if scale and primary["floor_scale_dav2_raw"].get("on_floor")
                else None
            ),
            "floor_m_per_px_v_at_ref": (
                round(primary["floor_scale_dav2_raw"]["m_per_px_v"] * scale, 4)
                if scale and primary["floor_scale_dav2_raw"].get("on_floor")
                else None
            ),
            "floor_ref_px": primary["floor_scale_dav2_raw"].get("ref_px"),
            "uncertainty": unc,
            "flags": flags,
        }
    )
    return result


# ---------------------------------------------------------------------------
# 彙總報告


def _fmt_band(band: list[float] | None, unit: str = "") -> str:
    if not band:
        return "—"
    return f"[{band[0]:g}, {band[1]:g}]{unit}"


def write_report(out_dir: Path, cameras: list[str]) -> None:
    rows = []
    for cam in cameras:
        p = out_dir / f"{cam}.calib.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    done = [r for r in rows if r.get("pitch_deg") is not None]
    scaled = [r for r in rows if r.get("scale") is not None]
    unscaled = [r for r in rows if r.get("scale") is None]
    rolled = [r for r in done if abs(r["roll_deg"]) > 5.0]
    dirty = sorted(
        (r for r in rows if r.get("person_frac_plate") is not None),
        key=lambda r: -r["person_frac_plate"],
    )

    lines: list[str] = []
    lines.append("# onboard01 — 機隊開通標定掃描（23 台 selling_floor）\n")
    lines.append(
        f"產生：{_dt.date.today().isoformat()}・`scripts/onboard_camera.py`・"
        f"方位管線 import 自 `calibrate_from_plate.py`（calib01 驗證：俯角 ±0.7°，"
        "vfov 釘對的前提下）・尺度自動子集＝人高統計（calib02 驗證的目視先驗法"
        "±5–8% 需人工，欄位留 null 標 `needs_visual_reference`）。\n"
    )
    lines.append(
        f"**完成率：{len(done)}/{len(cameras)} 台出方位；"
        f"{len(scaled)} 台尺度可自動定出、{len(unscaled)} 台需目視參照物。**\n"
    )
    lines.append(
        "帶寬紀律：主值取 vfov 70.4°（僅 Taichung-cam01 為磚縫釘定，其餘為機隊硬體"
        "假設），括號帶寬＝vfov 55–85° 掃描的散佈。**帶寬不是誤差條裝飾——vfov 是"
        "主導未知數，calib01 量到跨 vfov 高度滑 ~0.9 m、尺度滑 0.96→0.56。**\n"
    )
    lines.append("## 全機隊表\n")
    lines.append(
        "| 相機 | pitch°@70.4（55–85 帶） | roll°（帶） | H_DA-V2 raw m（帶） | "
        "scale（來源） | H m 標定（帶） | plate person 佔比 | 旗標 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        u = r.get("uncertainty") or {}
        cam = r["camera"]
        if r.get("pitch_deg") is None:
            lines.append(f"| {cam} | — | — | — | — | — | — | {', '.join(r['flags'])} |")
            continue
        n_boxes = ""
        if r.get("scale"):
            n_boxes = r["scale_source"].rsplit("_n", 1)[-1]
        scale_cell = f"{r['scale']:.3f}（人高 n={n_boxes}）" if r.get("scale") else "unmeasured"
        height_cell = (
            f"**{r['height_m']:.2f}** {_fmt_band(u.get('height_m_band_vfov_scan'))}"
            if r.get("height_m")
            else "—（需目視參照）"
        )
        pf = r.get("person_frac_plate")
        pf_cell = f"{pf:.1%}" if pf is not None else "—"
        interesting = [
            f for f in r["flags"]
            if f not in ("vfov_fleet_assumed", "k1_fleet_assumed")
        ]  # fmt: skip
        lines.append(
            f"| {cam} | {r['pitch_deg']:.1f} "
            f"{_fmt_band(u.get('pitch_deg_band_vfov_scan'))} | "
            f"{r['roll_deg']:+.1f} {_fmt_band(u.get('roll_deg_band_vfov_scan'))} | "
            f"{r['height_dav2_raw_m']:.2f} "
            f"{_fmt_band(u.get('height_dav2_raw_m_band_vfov_scan'))} | "
            f"{scale_cell} | {height_cell} | {pf_cell} | "
            f"{', '.join(interesting) or '—'} |"
        )
    lines.append("")

    lines.append("## 側裝相機（|roll| > 5°）\n")
    if rolled:
        lines.append(
            "VLM 試點說至少三台側裝；下面是全機隊第一次逐台量出來的 roll——"
            "plane fit 白送的量，管線裡沒有別的東西在建模 roll：\n"
        )
        for r in sorted(rolled, key=lambda r: -abs(r["roll_deg"])):
            u = r.get("uncertainty") or {}
            lines.append(
                f"- **{r['camera']}：roll {r['roll_deg']:+.1f}°**"
                f"（vfov 帶 {_fmt_band(u.get('roll_deg_band_vfov_scan'))}）"
            )
    else:
        lines.append("（無）")
    lines.append("")

    lines.append("## plate 髒區（模型 person 佔比，stable_infer 同一種量法）\n")
    lines.append(
        f"參考點：stable01 量過 Kaohsiung-cam04 的 112757 板是 8.6%。"
        f"超過 {DIRTY_PLATE_FRAC:.0%} 標 `dirty_plate_person_frac_gt_0p05`——"
        "髒區在靜態合成裡永不接管，plate 上的幾何量測（磚縫、門緣）也該避開那區。\n"
    )
    for r in dirty[:8]:
        mark = "  ← 髒板" if r["person_frac_plate"] > DIRTY_PLATE_FRAC else ""
        lines.append(
            f"- {r['camera']}（{r.get('plate_slot_utc', '?')}）："
            f"{r['person_frac_plate']:.1%}{mark}"
        )
    lines.append("")

    if scaled:
        svals = [r["scale"] for r in scaled]
        lines.append("## 尺度：自動（人高）vs 需目視\n")
        lines.append(
            f"- 人高尺度可出數（≥{MIN_HEIGHTS} 高度樣本）：**{len(scaled)} 台**，"
            f"scale 中位 {float(np.median(svals)):.3f}、"
            f"範圍 [{min(svals):.3f}, {max(svals):.3f}]"
            "（calib01 三台是 0.69–0.76；散佈若同量級，機隊常數修正仍是"
            "「可信但未證」）。"
        )
        lines.append(
            f"- 需目視參照物（門高／地磚，calib02 法）：**{len(unscaled)} 台**。"
            "另注意：人高尺度自帶 ±11% 系統項（1.70 m 先驗含姿態偏差，calib01），"
            "**所有相機的目視欄位都留 null**——要收斂到 ±5–8% 還是得走一次 calib02 "
            "的門／磚步驟，人高只是把「完全沒有公尺」變成「有個 ±11% 的公尺」。"
        )
        lines.append("")

    lines.append("## 量測紀律備忘\n")
    lines.append(
        "- 每台三點 vfov 敏感度全存在 `<camera>.calib.json` 的 `by_vfov`；"
        "表中每個主值旁的括號就是它的帶。\n"
        "- k1 與 vfov 只有 Taichung-cam01 是實測（磚縫網格），其餘 22 台掛"
        "`vfov_fleet_assumed`／`k1_fleet_assumed`，各背 ±5% 系統項（calib02）。\n"
        "- DA-V2 永不定公尺：`height_dav2_raw_m` 是形狀量，"
        "乘上 `scale` 才是公尺；`scale_source: unmeasured` 的相機沒有公尺。\n"
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"REPORT.md written: {len(done)}/{len(cameras)} cameras calibrated")


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", action="append", help="只跑這些相機（可重複）")
    ap.add_argument("--out", type=Path, default=Path("runs/onboard01"))
    ap.add_argument("--plates-root", type=Path, default=cfp.PLATES)
    ap.add_argument("--k1", type=float, default=K1_FLEET)
    ap.add_argument("--vfovs", default="55,70.4,85")
    ap.add_argument("--skip-person-frac", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)

    fleet = selling_floor_cameras()
    cameras = args.camera or fleet
    args.out.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        write_report(args.out, fleet)
        return 0

    _install_pipeline_cache()
    meter = None
    if not args.skip_person_frac:
        meter = PlatePersonMeter(PLATE_MODEL_CONFIG, PLATE_MODEL_CKPT)
    vfovs = [float(x) for x in args.vfovs.split(",")]

    for i, cam in enumerate(cameras, 1):
        print(f"[{i}/{len(cameras)}] {cam}")
        try:
            result = onboard_one(cam, vfovs, args.k1, meter, args.plates_root)
        except Exception:
            traceback.print_exc()
            result = {
                "schema": SCHEMA,
                "camera": cam,
                "generated": _dt.date.today().isoformat(),
                "pitch_deg": None,
                "roll_deg": None,
                "height_m": None,
                "height_source": "unmeasured",
                "scale": None,
                "scale_source": "unmeasured",
                "uncertainty": None,
                "flags": ["error_see_log"],
            }
        (args.out / f"{cam}.calib.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        p, r0 = result.get("pitch_deg"), result.get("roll_deg")
        s = result.get("scale")
        print(
            f"  pitch {p if p is not None else '—'}  roll {r0 if r0 is not None else '—'}  "
            f"scale {s if s is not None else 'unmeasured'}  "
            f"flags {result['flags']}"
        )

    write_report(args.out, fleet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
