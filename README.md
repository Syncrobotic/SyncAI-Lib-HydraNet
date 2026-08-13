# SyncAI-Lib-HydraNet

四足機器人多頭感知網路：**一次前向，同楨輸出可走路面、地形類別、物體偵測**。
架構理念對齊 Tesla HydraNet —— 共享 backbone/neck 承擔絕大部分算力，任務頭極輕量、彼此零耦合。

```text
                                ┌─ Traversability Head ─ [B, 3, H, W]   可走 / 謹慎 / 不可走
 image ─ RegNetX ─ BiFPN(P3–P7) ┼─ Terrain Head ──────── [B, 12, H, W]  地形類別
 [B,3,H,W]  (backbone)  (neck)  └─ FCOS Det Head ─────── boxes + labels 人 / 車 / 障礙物
```

參數分佈驗證了這個設計：**共享主幹 84.4%、三個頭合計 15.6%**（總計 8.32M）。
第四個頭的邊際成本約 3–9% 參數，卻能重用已經付過錢的那 84%。

- **Backbone**：torchvision RegNetX，可一鍵換 ResNet18/34/50
- **Neck**：BiFPN（fast-normalized fusion），備選標準 FPN
- **頭 1 可走路面**：Semantic-FPN 式分割，3 類
- **頭 2 地形分類**：同款分割頭，12 類
- **頭 3 物體偵測**：FCOS anchor-free（focal + GIoU + centerness）
- **多任務平衡**：Kendall 可學習不確定性加權（或固定權重）
- **多資料集部分監督**：每 step 從單一資料集抽 batch，只回傳該資料集監督的頭的損失
- **一份標註、兩個頭**：地形標註經策略表自動產生可走性標註
- **TensorRT 友善**：forward 圖僅含 Conv/BN/ReLU/Resize/MaxPool/Exp，NMS 在後處理

## 安裝

專案使用 [uv](https://docs.astral.sh/uv/) 管理環境與相依。

```bash
uv sync --group dev --extra export   # 建立 .venv 並安裝全部相依
uv run pytest                        # 冒煙測試，不需要任何資料集
```

Apple Silicon Mac（MPS）另見 [docs/TRAIN_MACOS.md](docs/TRAIN_MACOS.md)。
搬到 CUDA 工作站見 [docs/HANDOVER.md](docs/HANDOVER.md)。
訓練／評估／推論會自動選擇 CUDA → MPS → CPU。

## 指令

安裝後提供七個 console script：

| 指令 | 用途 |
|---|---|
| `hydranet-train` | 訓練 |
| `hydranet-eval` | 對 checkpoint 跑驗證 |
| `hydranet-infer-image` | 單張／資料夾推論疊圖 |
| `hydranet-infer-video` | 影片推論（走系統 ffmpeg，不需 opencv） |
| `hydranet-export-onnx` | 匯出 ONNX 供 TensorRT |
| `hydranet-prepare-ade20k` | 把 ADE20K 濾成室內子集並整理成 `seg_folder` 結構 |
| `hydranet-report` | 摘要單一 run，或跨 run 比較與 diff 設定 |

全部以 `uv run <指令>` 執行，或先 `source .venv/bin/activate`。

## 兩套場域設定

| Config | 場域 | 地形類別 | 分割資料集 |
|---|---|---|---|
| `hydranet_regnet800mf.yaml` | 越野 | 12 類戶外（草／碎石／樹叢…） | RUGD、RELLIS-3D |
| `hydranet_indoor.yaml` | 室內（大廳／走廊／廠房） | 12 類室內（地板／玻璃／樓梯…） | ADE20K + 自有標註 |

模型結構、損失、訓練機制兩者完全相同 —— 差異只在 `data.terrain_classes`、
`label_map` 與資料來源。偵測頭都用 COCO，不需更動。
標籤方案定義在 [`label_maps.py`](src/syncai_hydranet/data/label_maps.py) 的 `SCHEMES`，
室內映射見 [`label_maps_indoor.py`](src/syncai_hydranet/data/label_maps_indoor.py)。

自有相機的長寬比若與 `input_size` 不同，務必開 `data.letterbox: true`
（直式手機影片直接壓縮會橫向擠扁 2 倍以上）。

## 資料集準備

| 資料集 | 用途 | 下載 |
|---|---|---|
| ADE20K | 室內地形／可走性 | <https://groups.csail.mit.edu/vision/datasets/ADE20K/> |
| RUGD | 越野地形／可走性 | <http://rugd.vision/> |
| RELLIS-3D | 越野地形／可走性 | <https://github.com/unmannedlab/RELLIS-3D> |
| COCO 2017 | 物體偵測 | <https://cocodataset.org/#download> |

放置結構：

```text
datasets/
├── ADE20K/                                # 由 hydranet-prepare-ade20k 產生
│   ├── images/{train,val}/**.jpg
│   └── annotations/{train,val}/**.png     # 單通道整數 0..150
├── RUGD/
│   ├── images/{train,val}/**.png
│   └── annotations/{train,val}/**.png     # RGB 調色盤
└── coco/
    ├── train2017/  val2017/
    └── annotations/instances_{train,val}2017.json
```

RUGD／RELLIS 官方未提供 train/val 切分，請按 sequence 切以避免同序列洩漏。
自錄影片同理：務必按**錄影 session** 切，相鄰楨極度相似，隨機切分會嚴重高估效能。

只想先跑通？把 config 的 `data.datasets` 刪到剩你有的那些即可 —— 頭會照常建立，只是沒被監督
（啟動時會警告哪個頭沒人監督）。

### 三個 split

`best.pt` 是用 **val** 挑出來的，所以 val 上的分數對它本身而言偏樂觀。要報告可信數字，
請另外準備一份訓練流程完全不會讀到的 **test**：

```yaml
- name: ade20k
  split_train: train
  split_val: val
  split_test: test        # 選填，且刻意不預設為 val
```

```bash
uv run hydranet-eval --config ... --checkpoint ... --split test
```

沒設定 `split_test` 就指定 `--split test` 會直接報錯並告訴你要建什麼，不會默默退回 val。

### 資料版本

資料集不進 git，也沒有用 DVC。取而代之：每次訓練會把各 split 的**指紋**
（檔案數、總位元組、路徑與大小清單的 digest）寫進 `runs/<experiment>/meta.json`，
所以任何一個 checkpoint 都能回答「我是吃哪一份資料訓出來的」。
重新匯出標註、加了幾百張現場照片、換了過濾門檻，兩次訓練的指紋就會不同。

### ADE20K 室內子集

ADE20K 的 2 萬張橫跨 1055 種場景，大半是戶外。以下指令依**標註內容**過濾
（地板佔比夠高、天空與植被夠低），選出地面機器人視角的室內楨：

```bash
uv run hydranet-prepare-ade20k \
    --src datasets/ADEChallengeData2016 --dst datasets/ADE20K
# training -> train: kept 5998/20210 (29.7%)
```

輸出是 symlink，不佔額外磁碟、可重複執行。

## 訓練

```bash
uv run hydranet-train --config configs/hydranet_indoor.yaml

# 覆寫任意設定（dot-path）
uv run hydranet-train --config configs/hydranet_indoor.yaml \
    --set train.batch_size=8 model.neck.name=fpn 'data.input_size=[384,512]'

# 續訓：接著排程往下跑，不是重播
uv run hydranet-train --config ... --resume runs/hydranet_indoor/last.pt
```

訓練特性：AMP 混合精度（`train.amp_dtype` 可選 `bfloat16`）、cosine + warmup、
EMA（decay 會爬升，短跑也安全）、backbone 低學習率、梯度累積、best/last checkpoint。

`--set` 的錯字不會被吞掉：設定會在覆寫套用後整份驗證，未知的鍵、對不上的型別、
`supervises` 指到不存在的頭、`terrain_classes` 數量與頭對不起來，都會一次列出並中止。

### 顯存不夠就累積梯度

```bash
--set train.batch_size=4 train.grad_accum_steps=4     # 等效 batch 16
```

排程、EMA 與 `global_step` 都按 optimizer step 前進，所以換機器時只要維持
`batch_size × grad_accum_steps` 不變，學習率就不用重調。

### 模型是用哪個指標挑的

`train.primary_metric` 明確指定唯一決定 `best.pt` 的數字，預設 `traversability_mIoU`。
驗證輸出的任何 key 都能用，包含逐類別的：

```yaml
primary_metric: IoU/traversability/00_blocked   # 室內真正會讓機器人卡住的類別
```

指標名稱打錯會中止並列出所有可用的 key —— 而不是整趟訓練用了別的標準挑模型。

### 每次訓練留下什麼

```text
runs/<experiment>/
├── meta.json          git commit、是否 dirty、環境、資料集指紋、完整設定
├── config.yaml        套用 --set 之後的設定快照，可直接重跑
├── uncommitted.patch  工作區有未提交修改時才有（commit hash 不足以還原程式碼）
├── metrics.jsonl      每次驗證一行，可直接程式化比較
├── train.log
└── tb/
```

同名目錄已經有訓練結果時，新的一次會寫到帶時間戳的旁邊目錄，不會蓋掉既有的
`best.pt`，也不會把兩次的 TensorBoard 事件混在一起。要接續請用 `--resume`
（續訓時若設定與 checkpoint 內存的不同，會逐項列出差異）。

這些檔案就是給 `hydranet-report` 讀的：

```bash
uv run hydranet-report runs/hydranet_indoor        # 單一 run 的細節與曲線
uv run hydranet-report runs/* --diff               # 跨 run 排名 + 設定差異
```

```text
run                          commit    epochs  best      @epoch  metric
-----------------------------------------------------------------------
indoor-b                     eaba6b8e  40      0.6412    37      traversability_mIoU
indoor-a                     3c12224f* 40      0.5980    31      traversability_mIoU

indoor-a -> indoor-b
  train.lr: 0.0002 -> 0.0004
```

commit 後面的 `*` 代表那次訓練的工作區是 dirty 的。

### 監看訓練

```bash
uv run tensorboard --logdir runs/
```

TensorBoard 除了損失曲線，還會寫入：

- **`val_pred/*` 對照圖** —— 每次驗證輸出「輸入 | 預測 | 標註」三欄圖。
  灰色是 letterbox 補邊、黑色是 ignore 區。曲線只告訴你損失在降，這張圖才看得出
  模型是不是把整片地板判成牆。
- **`IoU/<head>/<class>`** —— 逐類別 IoU。室內的致命類別（glass、stairs）像素佔比極低，
  只看 mIoU 會被大類蓋掉。
- **`task_weight/*`** —— 不確定性加權學到的 `exp(-s)`，看得出哪個任務在主導主幹梯度。
  某個頭崩到接近 0 代表它實際上停止學習了。

## 評估與推論

```bash
uv run hydranet-eval --config ... --checkpoint runs/.../best.pt

# 在保留的 test split 上報告最終數字，並輸出成 JSON 方便跨 run 比較
uv run hydranet-eval --config ... --checkpoint runs/.../best.pt \
    --split test --json reports/best_test.json

uv run hydranet-infer-image --config ... --checkpoint runs/.../best.pt \
    --input photo.jpg --output out/

uv run hydranet-infer-video --config ... --checkpoint runs/.../best.pt \
    --input clip.mp4 --output clip_pred.mp4 --fps 10
```

## 部署到 Jetson Orin

```bash
uv run hydranet-export-onnx --config ... --checkpoint runs/.../best.pt --output hydranet.onnx
# Jetson 上：
trtexec --onnx=hydranet.onnx --saveEngine=hydranet_fp16.engine --fp16
```

輸出節點定義、C++ 後處理、INT8 量化、延遲預估：見 [docs/DEPLOY_JETSON.md](docs/DEPLOY_JETSON.md)。

## 新增任務頭（例：單目深度）

1. 在 `src/syncai_hydranet/models/heads/` 新增頭模組（輸入 FPN 特徵列表）
2. 在 `hydranet.py::HydraNet.__init__` 註冊 type 分支、`compute_losses` 加損失
3. 在 config 的 `model.heads` 加一段設定、對應資料集 `supervises` 列出新頭名

頭之間零耦合，不影響既有頭的訓練與部署。

## 開發

```bash
uv run ruff check --fix .    # lint + import 排序
uv run ruff format .         # 格式化
uv run pytest --cov          # 測試 + 覆蓋率
uv run pre-commit install    # 啟用 commit 前檢查
```

CI 在 GitHub Actions 上跑 lint 與 Python 3.10／3.12 的測試矩陣，覆蓋率低於 55% 會失敗。
測試不需要任何資料集：模型測試跑隨機張量，資料集測試在 `tmp_path` 現搭一份，
`test_overfit.py` 用合成的一個 batch 驗證訓練迴路真的會收斂（未訓練 34% → 訓練後 >95%）。

## 專案結構

```text
src/syncai_hydranet/
├── config.py                 # YAML 設定 + dot-path 覆寫
├── config_schema.py          # 設定驗證：未知鍵、型別、跨欄位一致性
├── cli/                      # console script 進入點（train/eval/infer/export/report）
├── models/
│   ├── backbone.py           # RegNet / ResNet 多尺度特徵
│   ├── neck.py               # BiFPN / FPN
│   ├── heads/segmentation.py # Semantic-FPN 分割頭（兩個頭共用）
│   ├── heads/detection.py    # FCOS 頭 + target assignment + decode/NMS
│   ├── losses.py             # CE+Dice / Focal+GIoU / 不確定性加權
│   └── hydranet.py           # 組裝 + compute_losses + predict
├── data/
│   ├── label_maps.py         # 越野映射 + 方案登錄表 SCHEMES
│   ├── label_maps_indoor.py  # 室內 12 類 + ADE20K 映射
│   ├── datasets.py           # SegFolderDataset / CocoDetDataset / split 解析
│   ├── transforms.py         # 影像+mask+框 聯合增強、letterbox、幾何反算
│   ├── fingerprint.py        # 資料集指紋，寫進 meta.json
│   └── multitask.py          # 多資料集 round-robin loader
├── engine/
│   ├── trainer.py            # AMP / EMA / cosine / 梯度累積 / checkpoint / TB
│   └── evaluator.py          # mIoU + COCO mAP + primary metric 選擇
└── utils/
    ├── device.py             # CUDA → MPS → CPU
    ├── checkpoint.py         # 安全載入（weights_only）+ 格式版本
    ├── runmeta.py            # git / 環境 / 設定快照 / metrics.jsonl
    ├── seeding.py            # 全域種子、worker 種子、後端旗標
    └── visualize.py          # 調色盤、疊圖、letterbox、對照圖
```

## 授權

Apache-2.0（ADE20K／RUGD／RELLIS-3D／COCO 各有其資料授權，商用前請確認）。
