# webcam_lidar

Webcam → MoGe-2 (ViT-S) metric depth → Lidar 風格點雲，WebSocket 即時推到瀏覽器。單幀輸出，不累積。

獨立的 uv 專案（Python 3.12），不屬於 `syncai-hydranet` wheel；Ruff 沿用 repo 根目錄設定。

```bash
cd demos/webcam_lidar
uv sync                         # 首次會從 GitHub 裝 moge、從 HF 拉 Ruicheng/moge-2-vits-normal
uv run server.py                # voxel 模式（4 cm）
uv run server.py --scanlines    # 模擬 32 線 Lidar
uv run server.py --synthetic    # 不用 webcam/模型，解析場景，測 viewer 用
```

直接用瀏覽器開 `viewer.html`（`file://` 即可）。Server 不在本機時：`viewer.html?ws=ws://<host>:8765`。
按 `C` 或左上按鈕切換顏色：高度 turbo ⇄ 距離灰階。

## Pipeline

| 段 | 做什麼 | 在哪跑 |
|---|---|---|
| cap | 擷取執行緒只保留最新一幀；1280×720 → 518 寬（INTER_AREA） | CPU |
| infer | MoGe-2 `infer()`，fp16；取 depth、mask、normalized intrinsics | GPU |
| post | depth EMA(α=0.3) → Sobel 邊緣 → 反投影 → 去邊緣點 → 4 cm voxel 質心（或 scanlines） | GPU |
| send | float16 平面 xyz 打包，`broadcast()` | CPU |

Server 每秒印一次各段滾動平均。下面是 `--synthetic` 在**無 GPU 的 CPU 容器**上的實測（infer 是解析場景，不是 MoGe）：

```
cap 0.1ms | infer 13.4ms | post 55.5ms | send 0.6ms |  9.7 FPS |  19580 pts  115 KiB  1 client(s)
```

`send` 是打包 + 進 socket 佇列的時間；實際線上時間看 viewer 右上角的 latency（同一台機器時鐘才準）。
`--fps` 是上限（預設 10）；推論端能跑多快，看 `infer + post` 即可估。

## 幾個與原始規格不同、或補充的決定

- **邊緣「正規化」= 除以深度本身，門檻預設 0.05 而非 0.15**：`|∇d| / (8·d)` 是每像素相對深度變化。
  不用逐幀 min-max：一個大斷層會把整張圖的門檻拉動，幀間不穩定；相對梯度與尺度無關。
  在這個單位下 0.15 太鬆：網路把遮擋邊緣抹成數個像素，2:1 深度斷層抹 5 px 後每像素只有 0.10–0.14，
  會整段漏成飛點（`test_threshold_015_lets_smeared_edges_through` 重現）。真實表面即使 85° 斜看也只有約 0.02，
  0.05 夾在中間、兩側都有餘裕。另外膨脹 1 px（`--edge-dilate`）。要回到 0.15：`--edge-thresh 0.15`。
- **內參也做 EMA**：MoGe 每幀重估焦距，焦距抖動會讓整個點雲「呼吸」。
- **EMA 不保留舊值**：本幀無效的像素直接無效，避免移走的物體留下殘影。
- **EMA 跳變重置（`--ema-reset 0.05`）**：新舊深度相差超過 5% 的像素直接用新值，不混合。
  純 EMA 在物體移動時會把舊表面在幾幀內「拖」過半空，形成空間上平滑、Sobel 抓不到的飛點
  （合成場景：移動方塊時最多 800+ 個；重置後 ≤ 2 個）。10% 不夠：被抹開的邊緣尾端從 4.5 m 跳到 5 m 正好 10%。
  Server 每秒印出 `ema-reset` 比例；靜止場景下這個值偏高，代表 MoGe 幀間整體尺度在跳、EMA 形同沒作用。
- **voxel 取質心**而非任取一點：質心本身就把同格內牆面雜訊平均掉。
- **`--max-depth 10`**：遠處 MoGe 深度雜訊大，預設丟掉 10 m 外的點。
- **Viewer 不每幀 new BufferAttribute**：r128 不會釋放被替換掉的 GL buffer（會洩漏）。改為
  預配置容量、每幀整批覆寫 + `setDrawRange(0, N)`，語意上仍是「整批替換、不累積」。
- **OrbitControls 從 jsDelivr 載入**（`three@0.128.0`，同版本）：cdnjs 的 three.js 只放 core build。
- **scanlines**：仰角 -25°…+15° 均分 32 條、方位每 0.25°，射線投影回影像、雙線性取 z-depth
  再沿射線縮放。邊緣/無效遮罩用同樣的雙線性取樣，碰到任何邊緣像素的射線直接丟掉
  （否則雙線性跨斷層會生出飛點）。一般 webcam 垂直 FOV 約 ±18°，所以上下幾圈落在畫面外、本來就不會有點。

座標：感測器在原點，右手系，Y 上，看向 -Z（Three.js 慣例）。沒有 IMU，「高度」是相機座標的 Y，鏡頭仰俯會讓地面變成斜的。

## 測試

```bash
uv run pytest test_pointcloud.py
```

用 `SyntheticDepth`（盒形房間 + 方塊，1% 深度雜訊、5×5 模糊模擬網路在邊緣的拖影）驗：
不過濾時確實有飛點、門檻 0.15 仍有飛點、0.05 時 0 個；方塊移動時純 EMA 有 >100 個拖尾飛點、加跳變重置後 ≤ 5；牆面 EMA+voxel 後標準差 < 2 cm；scanlines 每點都在 32 條仰角錐面上。
這些只驗後處理；模型輸出品質與 GPU FPS 必須在有 webcam + GPU 的機器上量。
