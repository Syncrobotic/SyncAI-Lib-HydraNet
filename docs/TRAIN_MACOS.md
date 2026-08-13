# 在 Apple Silicon Mac 上用 uv 訓練

本機開發與冒煙驗證用；正式訓練仍建議在 CUDA 機器上跑（AMP 只有 CUDA 支援）。
以下數字為 M4 Pro / 48 GB 統一記憶體 / PyTorch 2.13 實測。

## 一、建立環境

`uv` 會自己下載對應版本的 Python，不需要先裝 pyenv 或 conda。

```bash
uv sync --group dev --extra export   # 建立 .venv 並安裝全部相依（含 uv.lock）
```

- `uv sync` 會依 `uv.lock` 還原完全一致的相依版本，並以可編輯模式安裝本專案。
- `--group dev` 是 PEP 735 相依群組（pytest / ruff / pre-commit），不會進入發佈的 wheel。

之後所有指令有兩種寫法，擇一即可：

```bash
uv run hydranet-train ...              # 不必啟用 venv
# 或
source .venv/bin/activate              # 啟用後直接用指令名
```

驗證環境：

```bash
uv run pytest -q                                              # 20 passed
uv run python -c "import torch; print(torch.backends.mps.is_available())"   # True
```

## 二、裝置選擇

`src/syncai_hydranet/utils/device.py` 的 `pick_device()` 依序挑
**CUDA → MPS → CPU**，所有 CLI 指令共用。
Mac 上會自動走 MPS，訓練 log 第一行會印出 `device=mps`。

要強制指定：

```bash
uv run hydranet-train --config ... --set device=cpu
```

MPS 上有兩項自動關閉，都由 `device.py` 判斷，不需手動改 config：

| 項目 | 狀態 | 原因 |
|---|---|---|
| AMP (`train.amp`) | 自動關閉 | `GradScaler` / `autocast` 目前只有 CUDA 完整支援 |
| DataLoader `pin_memory` | 自動關閉 | MPS 尚未支援 pinned memory，設 True 只會噴警告 |

關掉 AMP 在 Mac 上不痛：48 GB 統一記憶體足以在 FP32 下跑 batch 16。

## 三、實測效能

RegNetX-800MF + BiFPN×2、三個頭全開、含 backward + AdamW step：

| 裝置 | 解析度 | batch | ms/step | img/s | 峰值記憶體 |
|---|---|---|---|---|---|
| CPU | 384×512 | 4 | 705 | 5.7 | — |
| MPS | 384×512 | 4 | 97 | 41.2 | — |
| MPS | 384×512 | 8 | 186 | 43.0 | 3.4 GB |
| MPS | 512×640 | 8 | 300 | 26.7 | 5.6 GB |
| MPS | 512×640 | 16 | 628 | 25.5 | 10.8 GB |

**MPS 比 CPU 快約 7.3 倍**，所以務必確認 log 印的是 `device=mps`。
batch 8 → 16 幾乎沒有吞吐增益（GPU 已飽和），記憶體卻翻倍，本機建議停在 8。

## 四、建議的本機 config 覆寫

```bash
uv run hydranet-train --config configs/hydranet_regnet800mf.yaml --set \
  "data.input_size=[384,512]" \
  "train.batch_size=8" \
  "data.workers=3"
```

### `data.workers` 要調小

`MultiTaskLoader` 是**每個資料集各建一個 DataLoader**，且 `persistent_workers=True`。
預設 `workers: 8` 搭配三個資料集，代表同時常駐 **24 個 worker process**，
在 14 核的筆電上會過度訂閱、反而拖慢資料供給。本機建議 `3`～`4`。

### COCO 的 `sample_ratio` 要調小

每個 epoch 的 step 數是 `Σ len(loader_i) × ratio_i`。COCO train2017 有約 11.7 萬張，
`ratio: 1.0` 時它單獨就貢獻 ~14,600 steps，一個 epoch 要 80 分鐘，60 epochs 是 80 小時。

把 COCO 的 `sample_ratio` 降到 `0.1`：每個 epoch 隨機取 10%，而因為 `shuffle=True`
且每個 epoch 會重建 iterator，各 epoch 看到的是不同的 10%，長期仍會掃過整個資料集。

注意 `config.py` 的 `set_path` **不支援 list 索引** —— `--set data.datasets[2].sample_ratio=0.1`
不會報錯，但會默默建出一個名為 `datasets[2]` 的無用鍵。兩個可行做法：

```bash
# 做法 A：複製一份 config（推薦，設定留得住）
cp configs/hydranet_regnet800mf.yaml configs/local_mac.yaml
# 編輯 local_mac.yaml，把 coco 那段的 sample_ratio 改成 0.1

# 做法 B：整個 list 一次覆寫（--set 的值會被 yaml.safe_load 解析）
--set "data.datasets=[{name: rugd, type: seg_folder, root: datasets/RUGD, \
  split_train: train, split_val: val, supervises: [traversability, terrain], sample_ratio: 1.0}, \
  {name: coco, type: coco, root: datasets/coco, split_train: train2017, \
  split_val: val2017, supervises: [detection], sample_ratio: 0.1}]"
```

調整後單一 epoch 約 14 分鐘、60 epochs 約 14 小時 —— 可以跑一個晚上。

### 只跑分割也是合法的

把 `data.datasets` 刪到剩 RUGD，偵測頭照樣會建立、只是不被監督。
RUGD + RELLIS 兩個資料集（約 1.3 萬張）在 512×640 下約 8 分鐘/epoch。

## 五、短跑實驗一定要關掉 EMA

驗證用的是 EMA 權重，而 EMA 從**模型的初始隨機權重**起步 —— n 步之後仍殘留
`ema_decay^n` 的初始化。預設 `ema_decay: 0.9998` 需要約 2 萬步才洗得掉：

| 步數 | 殘留的隨機初始化 | 驗證分數 |
|---|---|---|
| 160 步 @ 0.995 | 45% | 可走性 mIoU 0.16 |
| 同一份權重（非 EMA） | 0% | 可走性 mIoU **0.95** |

分數會靜默偏低，看起來像模型完全沒學到東西。本機做幾百步的冒煙實驗時：

```bash
--set train.ema=false
```

`Trainer` 會在總步數不足時主動印出警告，但仍請養成短跑關 EMA 的習慣。

## 六、觀察訓練是否被資料供給卡住

log 每 `log_interval` 步會印 img/s。拿它跟上表對照：

- 接近表中數字 → GPU 是瓶頸，正常。
- 明顯低於表中數字 → 資料供給跟不上。RELLIS-3D 是 1920×1200 的 JPEG，解碼很吃 CPU。
  先試著把 `data.workers` 往上加 1～2，仍不行就先把資料集離線降取樣。

TensorBoard：

```bash
uv run tensorboard --logdir runs/
```

## 七、ONNX 匯出

`hydranet-export-onnx` 不涉及裝置選擇，在 CPU 上匯出，Mac 可直接執行：

```bash
uv run hydranet-export-onnx --config configs/hydranet_regnet800mf.yaml \
    --checkpoint runs/.../best.pt --output hydranet.onnx
```

後續 `trtexec` 轉 engine 必須在 Jetson 上做，見 [DEPLOY_JETSON.md](DEPLOY_JETSON.md)。
