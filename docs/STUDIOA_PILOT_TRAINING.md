# StudioA 首輪 19 類語意試訓練

這一輪檢查保留的 AI 標籤能否訓練出有用的場景語意模型。未知區域保持 ignore 255，
地板為 0；不把缺漏的商品標註當成背景。這不是完整物件偵測、3D 還原或顧客分析。

已完成 **40 epochs、440 次更新**，來源店 val 選出 **epoch 32**。系統服務正常退出。
[圖表與預測對照](../runs/studioa_scene_pilot_20260916_v2/index.html)、
[可追溯結果](reviews/studioa_scene_pilot_20260916.json)。

| 項目 | 對保留 AI 標籤的 mIoU |
| --- | ---: |
| 訓練前 val（未訓練 scene head） | 0.29% |
| 最佳 epoch 32 val | 25.06% |
| Tao-Hsin 整店 test（選模後一次評估） | 17.89% |

以上平均均計入 19 類。test 地板 IoU 73.33%、牆壁 60.48%、展示桌 57.03%、
展示櫃／架 38.82%、boxed_stock 16.97%、人物 62.45%。門、手機、平板、紙箱、
椅子有正向 test 像素但 IoU 為零；消防沒有 test 正向像素，其零值源於錯誤預測，
不能用來量測召回率。小類別多數不足有效標籤像素的 1%，不宜只看整體均值。

預覽保留 test 的前兩張（同一支 Tao-Hsin-cam01），並非全店視覺驗收。可見桌面、
地板與人物的部分區域，但入口橫樑被誤塗為展示櫃，架上商品與架體仍混淆。
19 類輸出已能訓練，但不是 19 類皆已學會，尚不能正式部署。

## 固定方案

- 資料：`runs/studioa_partial_supervision_20260916_v2/`，19 類部分監督。
- 整店留出：Tao-Hsin；train 46、val 33、test 26 張。來源店原 test 鏡頭排除。
- train 的 19 類皆有有效像素；驗證和 test 不參與類別權重計算。
- ResNet18 ImageNet 預訓練骨幹、64 通道 FPN、19 類 scene head。
- 512×896 等比縮放與 padding、batch 4、AdamW、lr 0.0002、骨幹 lr 為其 0.1 倍。
- train 才使用縮放、翻轉、亮度／對比／飽和度增強；seed 42、bf16 AMP。
- 加權 CE + 0.5 Dice；權重為 train 像素中位數與逐類像素比值的平方根，限制 0.25–4。
- 最多 40 epochs，連續 10 次驗證未改善即停止。來源店 val 的 `scene_mIoU` 選 best。
- 選模完成後才以 best 評估整店 test，不依 test 調整本輪設定。

此處 mIoU 是對保留 AI 像素的類別平均交集比；不是獨立標註準確率。沒有 test
有效像素且也沒有預測像素的類別不計入該平均；無標籤但有預測的類別仍計入為零。
這些鏡頭曾用於先前實驗，因此也不宣稱全新盲測。

## 執行與追溯

```bash
.venv/bin/python tools/annotation/studioa_train.py prepare \
  --source runs/studioa_partial_supervision_20260916_v2 \
  --out runs/studioa_scene_pilot_20260916_v2 --held-out Tao-Hsin

.venv/bin/python tools/annotation/studioa_train.py run \
  --out runs/studioa_scene_pilot_20260916_v2
```

prepare 拒絕覆寫，複製影像、目標、程式、設定與既有 ImageNet 權重，逐檔保存 SHA-256。
run 重新執行凍結版本，使用檔案鎖防止雙 worker，訓練前後驗證輸入 hash。
GPU 工作由 `studioa-scene-pilot-20260916-v2.service` 持續執行，最長兩小時；
SSH／編輯器斷線不會停止訓練。不需要重新下載權重。
本輪實際從初始化至評估／報告完成約 115 秒，`Result=success`、退出碼 0。

結果目錄 `runs/studioa_scene_pilot_20260916_v2/`：

- `job.json`、`config.json`：commit、來源 manifest、逐檔 hash、分割與固定超參數。
- `status.json`、`worker.log`：進度、epoch、選模指標或完整失敗原因。
- `model/last.pt`、`model/best.pt`：每個完整 epoch 保存、綁定 job hash。
- `model/metrics.jsonl`、`baseline_val.json`：學習曲線與未訓練 scene head 的 val 基線。
- `test.json`、`test_scene_preview.jpg`、`report.json`：完成選模後的整店評估及產物 hash。

失敗時以相同 run 命令從 last 完整 epoch 恢復 optimizer、scheduler 等狀態。
未保存全部隨機數狀態，因此不是逐 bit 重播；完成訓練後的重新執行不重做訓練。
共享 `.venv` 並非容器快照；程式與 lockfile 已凍結，執行環境另由 Trainer meta 留存。
程式 commit 為 `68376601aa5a998972c038096ce919a38ed358c7`；執行時 dirty 僅來自
文件與無關的 `:memory:.ses`，未修改訓練程式。完成後另核對全部 570 個凍結檔案、
產物 hash、best／last 的 job 綁定、epoch 與模型張量有限值，以及 40 輪指標連續性。

216 項相關測試通過，包含 19 類／ignore 保留、跨資料集分割洩漏、真實 Trainer
訓練／驗證／checkpoint 與續跑；修改模組 lint、格式、型別及提交檢查通過。

## 範圍與下一步

本輪模型未包含實例偵測，也未預測實例 ID、3D 幾何或顧客行為。後續已完成
[首批實例部分監督接線](STUDIOA_PARTIAL_DETECTION.md)，尚未完成正式偵測訓練。
其他整店留出方案仍有 train 缺類，雲端補資料受登入阻塞。先分析本輪類別失敗與
來源店 val 預測，再依資料缺口進行 AI 補標；若依本輪 test 發現修改方向，後續
需新的封存評估集，不能將同一 test 反覆調整後的分數當作獨立泛化證據。

第一個 v1 工作完成 epoch 1 訓練與驗證，但輸出 TensorBoard 圖時因缺少 19 類
配色而失敗，尚未保存 checkpoint 或執行 test。失敗狀態與 log 原封保留；新增正式
配色、prepare 前置檢查及含 floor／ignore 的預覽回歸測試後，以新凍結 v2 從頭訓練。
配色及資料整合測試另通過 26 項；v1 的失敗未用於調整模型或超參數。
