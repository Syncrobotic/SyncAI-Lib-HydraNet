# 交接：從 Apple Silicon 移到 CUDA 機器

本機（M4 Pro / MPS）只適合開發與冒煙驗證。這份文件是把訓練搬到 CUDA 工作站
（如 RTX PRO 6000 Blackwell）時的 runbook，以及交接當下的已知狀態。

macOS 端的設定見 [TRAIN_MACOS.md](TRAIN_MACOS.md)；Jetson 部署見 [DEPLOY_JETSON.md](DEPLOY_JETSON.md)。

## 一、程式碼不需要為了換卡而改

`utils/device.py` 的 `pick_device()` 依序挑 **CUDA → MPS → CPU**，
AMP 與 DataLoader 的 `pin_memory` 也都由同一處判斷後自動啟用。
在 CUDA 機器上就是同一套指令，不必加任何 flag。

## 二、四樣東西不會跟著 git 走

`.gitignore` 把資料、產出與大型媒體全部排除，所以 clone 之後這些都不存在：

| 路徑 | 大小（交接當下） | 對策 |
|---|---|---|
| `datasets/` | 1.0 GB | 在 box 上重抓，不要從筆電上傳 |
| `runs/<experiment>/*.pt` | 122 MB / checkpoint | 見下方說明，多半不值得傳 |
| `assets/*.mp4` | 19 MB | **必須傳**，無法重新產生 |
| COCO | 未下載 | 需要偵測頭時在 box 上抓 |

`assets/VID_20260813_145154.mp4` 是自錄的室內場域影片，是唯一無法用指令重建的資產。
要做推論對照就得先把它複製過去。

**checkpoint 通常不必傳。** 在 M4 Pro 上 20 epochs 要 55 分鐘，同樣的訓練在
Pro 6000 配 bf16 與較大 batch 大約 5–10 分鐘。為了省這點時間去搬 122 MB、
還要處理舊格式相容，不划算 —— 從 ImageNet 初始重訓更乾淨，而且新產出的 run
才會帶 `meta.json`，`hydranet-report` 才看得到它。

## 三、在 CUDA 機器上啟動

```bash
git clone -b dev https://github.com/Syncrobotic/SyncAI-Lib-HydraNet.git
cd SyncAI-Lib-HydraNet
uv sync --group dev --extra export

# 第一道關卡
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`uv.lock` 鎖的是 **CUDA 13** 版 torch（相依裡有 `nvidia-cudnn-cu13`、`nvidia-nccl-cu13`）。
驅動太舊的話 **`uv sync` 會成功，要到上面那行 import 才失敗** —— 先跑這行，
不要等訓練啟動才發現。Blackwell（sm_120）本身在 CUDA 12.8 之後就支援，版本沒有問題。

接著跑測試確認環境完整：

```bash
uv run pytest -q
```

資料（repo 裡沒有）：

```bash
curl -LO https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip -q ADEChallengeData2016.zip -d datasets/
uv run hydranet-prepare-ade20k --src datasets/ADEChallengeData2016 --dst datasets/ADE20K
```

`hydranet-prepare-ade20k` 依**標註內容**過濾（地板佔比夠、天空與植被夠低），
而不是靠場景類別白名單，選出地面機器人視角的室內楨：20,210 → 5,998 張。

## 四、交接當下的基準

Apple M4 Pro / MPS、384×512、batch 8、20 epochs、約 55 分鐘，僅 ADE20K 室內子集：

| 指標 | 值 |
|---|---|
| traversability mIoU | 0.665 |
| ├ blocked | 0.954 |
| ├ caution | 0.200 |
| └ go | 0.843 |
| terrain mIoU | 0.569 |
| └ glass | 0.505 |

CUDA 上重訓後應該要**至少**打平這組數字。明顯低於它就代表搬遷過程有東西壞掉，
先回頭查而不是繼續調參。

## 五、已知問題，依優先序

### 1. `caution` 卡在 0.200，加 epoch 沒用

這是資料缺口不是訓練不足。可走性的「謹慎」在室內對應四個地形類別
（見 `data/label_maps_indoor.py`），但 ADE20K 只涵蓋其中的 `stairs`，
佔全體像素 **0.3%**；`floor_metal`（金屬格柵）、`wet_slippery`（濕滑／積水）、
`threshold_ramp`（門檻高低差）它**一張都沒有**。

這三類直接決定機器人會不會滑倒或卡住，只能靠自有標註補。
標註格式用 `label_map: indoor_native`（標註 id 直接就是室內 12 類）。

### 2. 偵測頭完全沒被監督

目前只餵了分割資料集，所以偵測頭停在初始權重、推論不會有任何框。
訓練啟動時 config 檢查會明確警告：

```
config: head 'detection' is not supervised by any dataset:
        it will be built, exported and left at its initial weights
```

要有框就得下載 COCO 並把 `sample_ratio` 開回 `1.0`
（indoor config 目前是 `0.2`，那是為了讓筆電跑得動）。

### 3. `channels_last` 尚未實作

`train.tf32` 與 `train.amp_dtype=bfloat16` 都已經有了，但卷積網路在 tensor core 上
改用 NHWC 記憶體格式通常還有一截增益。這是 `src/` 裡剩下唯一的 CUDA 專屬優化。

## 六、建議的起手設定

96 GB VRAM：

```bash
uv run hydranet-train --config configs/hydranet_indoor.yaml --set \
    train.batch_size=48 \
    train.lr=6.0e-4 \
    train.epochs=60 \
    train.warmup_iters=2000 \
    train.amp_dtype=bfloat16 \
    train.tf32=true \
    train.cudnn_benchmark=true \
    data.workers=16 \
    'data.input_size=[512,640]'
```

各項理由：

- **batch 48** —— M4 Pro 上 fp32、512×640、batch 8 的峰值是 5.6 GB。
  CUDA 上 AMP 會把活化記憶體大致砍半，48 約落在 20 GB，96 GB 很寬裕。
  再往上吞吐收益遞減，而大 batch 對分割任務不一定更好。
- **lr 6e-4** —— batch 8→48 若用線性縮放會到 1.2e-3，對 AdamW 太衝。
  這裡用 sqrt 縮放（`2e-4 × √6 ≈ 4.9e-4`）再抓寬一點，warmup 同步拉長到 2000 步。
- **bfloat16 而非 float16** —— bf16 保有 fp32 的指數範圍，不需要 loss scaler，
  Blackwell 原生支援。fp16 在多任務損失（Dice 與 Focal 量級差很多）下較容易溢位。
- **workers 16** —— M4 Pro 上 38 img/s 有相當比例卡在 JPEG 解碼。
  server 端核多，這個瓶頸會消失；仍請對照 log 印出的 img/s 確認。
- **512×640** —— 筆電為了速度降到 384×512，CUDA 上沒有理由不用 config 預設值。

## 七、換卡後第一件該做的事

**先用相同設定重現第四節的基準，再開始改東西。** 一次跑通之後：

```bash
uv run hydranet-report runs/*          # 跨 run 比較，讀 meta.json 與 metrics.jsonl
uv run tensorboard --logdir runs/
```

TensorBoard 裡優先看 `val_pred/*` 三欄對照圖與 `IoU/<head>/<class>` 逐類別曲線 ——
`glass` 和 `stairs` 佔比極低，只看 mIoU 會被 floor 與 wall 完全蓋掉。
