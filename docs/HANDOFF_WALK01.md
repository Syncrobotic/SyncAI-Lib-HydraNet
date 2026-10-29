# HydraNet walk01 交接 — pose02 三頭 + 二元 walkable 頭

寫給 gb300 上的 AI 看。你沒有這個專案的任何上下文，這份文件要讓你在十分鐘內接手。
寫於 2026-10-29，來源機器 hgpn118（8×H100，repo 在 /data/SyncAI-Lib-HydraNet）。
決策者是 Finn；遇到「待 Finn 決定」的項目先問他，不要自己拍板。

---

## 0. 目標（一句話）

做一顆新的 HydraNet：**pose02 的三個頭（terrain / detection / pose）原封不動，加第四個頭
`traversability`（二元 blocked / go，即舊四頭模型的 walkable 頭）**，從 pose02 權重暖啟動
訓到收斂，匯出 ONNX 與 TensorRT engine，交給建場景 worker 與 syncairules 使用。

## 1. 現況快照（2026-10-29 離開 hgpn118 時）

已完成，全部在這包裡：

| 項目 | 狀態 |
|---|---|
| 程式碼 | 8 個檔改好（見 §4），未 commit，疊在 main@9bbba99 之上 |
| config | `configs/hydranet_retail_walk01.yaml`（正式）、`configs/hydranet_retail_walk01_smoke.yaml`（本機資料 smoke） |
| 單元測試 | `tests/test_walkable_head.py` 14 個全過；相關子集 353 過 / 1 失敗（失敗的是舊的 `hydranet_detpose01.yaml`，與本案無關） |
| smoke | 1 epoch、40 步、從 pose02 暖啟動：四個頭都有 loss，驗證全跑完，best.pt 有存 |
| 匯出 | smoke 的 best.pt 匯出 ONNX：18 個輸出，`traversability` 1×2×640×1120，parity PASS |
| 0.5.0 上重驗 | 同一套改動移植到上游 4810b24：smoke 再跑一次（trav mIoU 0.890、四頭皆有 loss）、ONNX 18 輸出 parity PASS；整套測試「1 failed, 3286 passed, 18 skipped in 80.13s (0:01:20)」（剩下的 `test_version_has_a_single_source` 是 hgpn118 venv 裝的是 0.2.0 metadata，gb300 `uv sync` 後即一致） |

**版本：GitHub main 已經比 hgpn118 新。** hgpn118 停在 9bbba99（0.2.0，9/1），上游 main 在
4810b24（0.5.0，9/8），中間 151 個 commit（off-road taxonomy 退役、config+checkpoint loader 合一、
sys.path bootstrap 拿掉等）。這包裡兩種 patch 都有：`walk01_tracked_on_0.5.0.patch` 對 4810b24，
`walk01_tracked.patch` 對 9bbba99，**二選一，不可混用**。0.5.0 版已在 hgpn118 驗過：config 驗證通過、
pose02 best.pt 嚴格載入、smoke 與匯出都重跑過（移植只碰到 2 個衝突 hunk：visualize.py 保留新加的
WALK_COLORS 區塊、pose_overlay.py 保留上游的 ROOT 註解）。**建議直接用 0.5.0。**
0.5.0 多了幾條規則，這包已經配合改好（細節在 §7）：train/val 不得同相機、lr 必須隨 batch 走、
source 只能是英文、score 門檻要用具名常數。

**卡住正式訓練的唯一一件事：資料。** pose02 配方裡的三個站內資料集
`retail_objects_batch02`、`retail_objects_batch03`、`site30k_v1` **hgpn118 上沒有**，
pose02 是在另一台機器（hostname `syncai-Pro-ET700I-W7`）訓的。問 Finn 它們在哪。

## 2. 模型規格

trunk：regnet_x_800mf + BiFPN（96 ch、2 repeats、5 levels），輸入 `image_rgb_255`
1×3×640×1120，RGB 0–255，正規化在圖裡，letterbox 補 114，**主機端不可再減平均**。

| 頭 | 類型 | 輸出 | 來源 |
|---|---|---|---|
| traversability（新） | semantic_fpn，2 類 blocked / go，P3–P5，ch 64，CE + 1.5 dice | `traversability` 1×2×H×W | r6 四頭模型的 walkable 頭配方 |
| terrain | semantic_fpn，6 類 void/floor/wall/column/fixture/person | `terrain` 1×6×H×W | pose02 |
| detection | FCOS，4 類 person/bag/boxed_stock/device，P3–P7 | `det_cls/det_reg/det_ctr _p3..p7` | pose02 |
| pose | pose_p3，17 點 heatmap @P3 | `pose_heatmap_p3` 1×17×80×140 | pose02 |

traversability 的標籤**從不人工標**：每個分割資料集用 `_walk` 結尾的 label scheme，從
terrain 遮罩推導 — floor → go，其餘（含 stairs、person、fixture）→ blocked，void → ignore。
這是 2026-08-18 的決定（取消 caution 等級，距離交給深度）。

loss_balancing = uncertainty（Kendall），所以 `fixed_weights` 那張表不會被讀。

## 3. 血統與決定

- **pose02** = `runs/hydranet_retail_pose02/best.pt`：120 epoch，dev 分支 commit cf1ddfbe，乾淨樹，
  2026-08-26 訓完。val：pose PCK@0.2h 0.9345、terrain mIoU 0.637、det mAP50 site_boxes03 0.346。
  best.pt（E119）與 last.pt（E120）分數到小數第四位才有差，用 best.pt。
- **2026-09-02 決定**：pose02 三頭是主線；本機分支 local-work-20260901 的 r6 四頭
  （traversability / terrain 11 類+深度 / detection COCO 80 / person_attr）退為參考，架構不相容，分數不可互比。
- **2026-10-08 決定**：HydraNet 確定上線；建場景 worker 走 Rust（`syncai-trt-vision`），engine 放 `/models/hydranet/<name>/`。
- **2026-10-12**：pose02 匯出 ONNX + H100 fp16 engine（TensorRT 10.9，dev image 內建），對拍 shimen-cam07 一致率 0.995。
- 10/05 的 `three_heads_1ep` / `three_heads_depth_1ep` / `terrain_v3_1ep` 是 1 epoch 實驗（terrain 改 11 類、
  depth 併進 terrain 頭），不是主線；它們留下的程式碼（seg_depth_fpn 等）也在這包的 patch 裡，別刪。

## 4. 這包裡有什麼（/mnt/handoff_walk01/）

```
HANDOFF_WALK01.md            這份文件
walk01_tracked_on_0.5.0.patch  git diff 對上游 main 4810b24（0.5.0），22 個已追蹤檔 — 建議用這個
walk01_tracked.patch           同一批改動對 9bbba99（0.2.0，hgpn118 現況）— 只在要和 hgpn118 完全一致時用
walk01_untracked.tgz         未追蹤但必要的檔：新 configs、新 label_maps_*.py、backbone_dinov3.py、tests、tools
                             （不含 hydranet_detpose01.yaml：它在兩個版本的 schema 都過不了、沒有任何 run 用過、
                             已被 detpose_coco01 取代；hgpn118 上仍留著原檔）
pose02_run/                  pose02 的 best.pt(133M)、config.yaml、selection.json、metrics.jsonl、meta.json、train.log
models_hydranet_pose02/      pose02.onnx、h100.fp16.engine（只能在 H100 + TRT 10.9 載）、.json、trtexec log
walk01_smoke_run/            9bbba99 上 smoke 的 config/metrics/selection/train.log（沒有 .pt，數字只代表「沒壞」）
walk01_smoke_run_0.5.0/      同一個 smoke 在 0.5.0 程式碼上的結果
```

這次（10/29）改的 8 個程式檔，都在 patch 裡：

| 檔 | 改了什麼 |
|---|---|
| `data/label_maps.py` | `LabelScheme` 加 `trav_classes`；`walkable_from_trav()`；註冊四個 `_walk` scheme：`ade20k_retail_surfaces_walk`、`retail_surfaces_native_walk`、`retail_surfaces_from_objects_walk`、`retail_v3_native_walk` |
| `config_schema.py` | `_check_trav_width`：trav 頭的 num_classes 必須等於每個供應它的 scheme 的寬度，否則拒跑 |
| `engine/evaluator.py` | `_trav_names(cfg)`：2 類頭時類別名是 blocked/go，不再寫死 3 類 |
| `engine/trainer.py` | `--load` 走 `_load_weights`：容忍 checkpoint 沒有的 tensor（新頭）並在 log 列出；多餘 key 或 shape 不符則拒絕 |
| `utils/visualize.py` | `WALK_COLORS`、`trav_colors(n)`、`trav_colors_for(model)` |
| `cli/infer_image.py` / `infer_video.py` / `scene.py` | 調色盤依頭寬度選 |

## 5. 在 gb300 上的步驟

```bash
# 1. 程式碼。先看 GitHub 有沒有 feat/walk01-fourhead 分支（整包改動在 0.5.0 上的一個 commit）：
git clone git@github.com:Syncrobotic/SyncAI-Lib-HydraNet.git && cd SyncAI-Lib-HydraNet
git checkout feat/walk01-fourhead && uv sync              # 有分支就到此為止，跳到第 2 步
#    沒有分支（還沒 push）就用這包的 patch：
git checkout 4810b24                                      # 0.5.0；備案 9bbba99 + walk01_tracked.patch
git apply /mnt/handoff_walk01/walk01_tracked_on_0.5.0.patch
tar xzf /mnt/handoff_walk01/walk01_untracked.tgz          # 解在 repo 根目錄
uv sync                                                   # Python 3.11，uv.lock 鎖版本
git status --short | wc -l                                # 預期 24 個 M + 23 個 ??

# 2. pose02 權重與 engine
mkdir -p runs/hydranet_retail_pose02 && cp /mnt/handoff_walk01/pose02_run/* runs/hydranet_retail_pose02/
sudo mkdir -p /models/hydranet && sudo cp -r /mnt/handoff_walk01/models_hydranet_pose02 /models/hydranet/pose02

# 3. 資料：先把 hgpn118 有的拉過來。symlink 情況各不同，順序與旗標照抄：
#    coco 21G（真檔）；ADE20K 287M 是指向 ADEChallengeData2016 的「絕對路徑」symlink，要 -L 展開；
#    coco_pose 是指向 ../../coco 的「相對路徑」symlink，不加 -L、放在 coco 旁邊就有效；retail_v3_512 884M 真檔
rsync -a  <user>@<hgpn118>:/data/SyncAI-Lib-HydraNet/datasets/coco          datasets/
rsync -aL <user>@<hgpn118>:/data/SyncAI-Lib-HydraNet/datasets/ADE20K        datasets/
rsync -a  <user>@<hgpn118>:/data/SyncAI-Lib-HydraNet/datasets/coco_pose     datasets/
rsync -a  <user>@<hgpn118>:/data/SyncAI-Lib-HydraNet/datasets/retail_v3_512 datasets/
#    缺的三個（batch02 / batch03 / site30k_v1）問 Finn

# 4. 驗證環境
.venv/bin/python -c "from syncai_hydranet.config import load_config; load_config('configs/hydranet_retail_walk01.yaml', [])"
.venv/bin/pytest -q tests/test_walkable_head.py tests/test_config_schema.py tests/test_trainer.py
#    預期：全過（整套 pytest 也應全過；hgpn118 上唯一剩的 version 測試是 venv 版本殘留）

# 5. smoke（本機資料、1 epoch、幾分鐘）— 先確認四頭在這台 GPU 上跑得通
CUDA_VISIBLE_DEVICES=0 .venv/bin/hydranet-train --config configs/hydranet_retail_walk01_smoke.yaml \
    --load runs/hydranet_retail_pose02/best.pt --allow-dirty
#    log 裡要看到：「[model] 39 tensor(s) are not in the checkpoint ... seg_heads.traversability (38 tensors)」

# 6. 正式（三個站內資料集到位後）
.venv/bin/hydranet-train --config configs/hydranet_retail_walk01.yaml \
    --load runs/hydranet_retail_pose02/best.pt --allow-dirty

# 7. 匯出
.venv/bin/hydranet-export-onnx --config configs/hydranet_retail_walk01.yaml \
    --checkpoint runs/hydranet_retail_walk01/best.pt --output /models/hydranet/walk01/walk01.onnx \
    --batch 1 --check-parity

# 8. engine：一定在 dev image 裡建（host 的 TensorRT 版本載不了），engine 綁 GPU 型號
docker run --rm --gpus all -v /models/hydranet:/models/hydranet \
    ghcr.io/syncrobotic/syncai-lib-gstreamerpipeline:amd64 \
    trtexec --onnx=/models/hydranet/walk01/walk01.onnx \
            --saveEngine=/models/hydranet/walk01/<gpu>.fp16.engine --fp16 --skipInference
#    然後照 /models/hydranet/pose02/h100.fp16.engine.json 的格式寫一份 .json（sha256、輸入輸出、來源 checkpoint）
```

`--allow-dirty` 是因為樹裡有未 commit 的改動；要讓這顆能走 `scripts/release_bundle.sh`
（它拒絕 dirty 樹訓出來的權重），就先把 patch 內容 commit 到一個分支再訓。要不要 commit、
commit 到哪個分支，問 Finn。

## 6. 驗收標準

正式 run 以 `traversability_mIoU` 選 best.pt。但三個舊頭不能退步 — 每個 epoch 的 metrics.jsonl 都有它們：

| 指標 | pose02 基準（val，EMA） | 要求 |
|---|---|---|
| pose_PCK@0.2h | 0.9345 | 不低於 0.93 |
| terrain_mIoU | 0.637 | 不低於 0.63 |
| detection_mAP50/site_boxes03 | 0.346 | 不明顯掉（±0.02） |
| traversability_mIoU（site_seg03 / retail_v3） | — | 目標 ≥ 0.85；smoke 在 retail_v3 已 0.84 |
| head_disagreement | — | trav 頭與 terrain 頭互相矛盾的像素比例，應 < 3% |

真正的驗收在站內影片：`tools/pose/pose_overlay.py` 與 `hydranet-scene` 疊圖肉眼看，
walkable 不能把貨架、櫃台、人畫成 go。

## 7. 已知陷阱

- **trav 標籤寬度**：2 類頭只能配 `_walk` scheme；用原本 3 類 scheme 會被 config 檢查擋下（故意的）。
- **retail_v3_512 只供應 traversability，而且只訓不驗**：它的遮罩是 11 類 retail_v3 taxonomy，6 類 terrain 頭
  讀不了，所以 `supervises: [traversability]`；它的 val 與 train 共用 19 支相機（按影格切的），0.5.0 的 trainer
  視為 leak 直接拒跑，所以 config 拿掉了它的 `split_val`。想要站內 walkable 的 val，就用
  `datasets/retail_v3/_camera_split.json` 重建一份按相機切的 512 版。
- **lr 必須 = 1.25e-5 × batch_size**（0.5.0 的 `tests/test_config_defaults.py`）：walk01 是暖啟動、刻意用一半，
  已登記在該檔的 `DECLARED_DEVIATIONS`；10/05 那些 1-epoch 實驗 config 也登記了。改 lr 或 batch 要同步改那裡。
- **source 必須是英文**（`tests/test_source_is_english.py`，只掃 git 追蹤的檔）：這包裡的中文註解都已翻成英文。
  例外是 `label_maps_retail22.py` 的 22 個類名本身就是 CVAT 標籤的中文（`地板`、`玻璃門`…），它目前未追蹤所以
  沒被掃到；一旦 commit 就會紅。要改名還是豁免，問 Finn。
- **score 門檻要用具名常數**（`SCORE_THR_VIEW` 0.30 / `SCORE_THR_RETAIL` 0.20 / `SCORE_THR_EVAL` 0.05）：
  `--score-thr` 的 default 寫裸數字會被 `tests/test_shared_constants.py` 擋。
- **trainer 拒絕 dirty 樹**：src/ 或該 config 有未 commit 改動就要 `--allow-dirty`；它會把 diff 存成 `runs/<exp>/uncommitted.patch`。
- **選模指標**：`primary_metric` 一定要是每個 epoch 都會產生的指標；`detection_val_interval: 5` 時不能選 detection_mAP。
- **EMA 短跑陷阱**：很短的 run EMA 權重是雜訊，`select_weights` 預設取 EMA；smoke 看數字時注意。
- **engine 綁定**：pose02 的 h100.fp16.engine 只能在 H100 + TensorRT 10.9（dev image）載；gb300 要重建。
- **coco_person 只教 person/bag**：boxed_stock 與 device 在 COCO 上是 masked 不是 negative，log 會說明。
- **datasets/ 有 symlink**：ADE20K 是絕對路徑 symlink（要 `-L`），coco_pose 是相對路徑 symlink 指向 coco（不要 `-L`，否則 coco 複製兩次）；§5 第 3 步的指令已經排好。
- **pose02 的 val 是站內資料**：smoke 的 pose PCK 0.73 是在 COCO val 上、只 501 人，不能跟 0.9345 比。

## 8. 待 Finn 決定

1. 三個站內資料集從哪裡拿（或改用 hgpn118 有的資料訓：terrain 只剩 ADE20K、det/pose 只剩 COCO，三個舊頭會往 web 資料漂）。
2. 要不要先 commit 這批改動（分支名？），讓正式 run 可 release。
3. epochs（config 先寫 40，lr 2e-4 = pose02 一半）。
4. 正式 run 的 `primary_metric` 維持 traversability_mIoU，還是改回 pose_PCK@0.2h。
5. retail22 taxonomy 的中文類名（見 §7）：改成英文 id 還是在英文檢查裡豁免。
6. walkable 的站內 val：要不要從 `_camera_split.json` 重建按相機切的 retail_v3_512。

## 9. 延伸閱讀

- `README.md`、`docs/PLAN.md`（2072 行，整個專案的單一真相來源，但只更新到 2026-08-31；之後的決定在本文件 §3）
- GStreamerPipeline repo 的 `scene-worker-plan.md`、`scene-build-handoff.md`：engine 怎麼被建場景 worker 吃
- `scripts/release_bundle.sh`：正式 bundle 的格式與兩道 gate
