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
訓練／評估／推論會自動選擇 CUDA → MPS → CPU。

## 指令

安裝後提供五個 console script：

| 指令 | 用途 |
|---|---|
| `hydranet-train` | 訓練 |
| `hydranet-eval` | 對 checkpoint 跑驗證 |
| `hydranet-infer-image` | 單張／資料夾推論疊圖 |
| `hydranet-infer-video` | 影片推論（走系統 ffmpeg，不需 opencv） |
| `hydranet-export-onnx` | 匯出 ONNX 供 TensorRT |
| `hydranet-prepare-ade20k` | 把 ADE20K 濾成室內子集並整理成 `seg_folder` 結構 |

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

只想先跑通？把 config 的 `data.datasets` 刪到剩你有的那些即可 —— 頭會照常建立，只是沒被監督。

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

# 續訓
uv run hydranet-train --config ... --resume runs/hydranet_indoor/last.pt
```

訓練特性：AMP 混合精度、cosine + warmup、EMA、backbone 低學習率、best/last checkpoint。

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

CI 在 GitHub Actions 上跑 lint 與 Python 3.10／3.12 的測試矩陣。

## 專案結構

```text
src/syncai_hydranet/
├── config.py                 # YAML 設定 + dot-path 覆寫
├── cli/                      # console script 進入點
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
│   ├── datasets.py           # SegFolderDataset / CocoDetDataset
│   ├── transforms.py         # 影像+mask+框 聯合增強、letterbox
│   └── multitask.py          # 多資料集 round-robin loader
├── engine/
│   ├── trainer.py            # AMP / EMA / cosine / checkpoint / TensorBoard
│   └── evaluator.py          # mIoU + COCO mAP
└── utils/
    ├── device.py             # CUDA → MPS → CPU
    └── visualize.py          # 調色盤、疊圖、letterbox、對照圖
```

## 授權

Apache-2.0（ADE20K／RUGD／RELLIS-3D／COCO 各有其資料授權，商用前請確認）。
