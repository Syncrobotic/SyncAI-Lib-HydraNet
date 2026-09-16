# StudioA 夜間訓練與五視角 3D 交付

使用者要求在 **2026-09-17 06:00 Asia/Taipei** 前看到新模型完成，以及目前
StudioA 場域 3–5 個 CCTV 視角的 3D 世界效果圖。

## 固定方案

- 先完成一個新訓練的 bootstrap 模型（弱類放大取樣、seed 42、495 次更新），
  並輸出五視角，建立先行交付物；後續批次出問題時保留這份已完成產物。
- 主批次：一般取樣／弱類放大取樣 × seeds 42、43、44，共六組；各 4,950 次更新
  （330 epochs），每 495 次更新驗證。共用新版 62 張 train／33 張 val，舊版
  46 張 train 計算的固定類別權重；同 seed 初始化逐張量核對。
- 弱類放大：50% 機率從該 train 影格已有的柱子、門、玻璃及小物件正例選類、
  選像素，裁出保留上下文的 2–3 倍放大區域。另一半保留完整場景；ignore 保持
  ignore，不將未知區域補成負例。驗證資料不裁切、不增強。
- 依共用 AI source-val 的 19 類 mIoU 選已完成的新 checkpoint；保存全部逐類
  結果與各 seed 分數，不宣稱獨立真值準確率，也不自動替換部署模型。
- 不執行 test 評分。固定出圖視角是部署預覽，不參與選模。

## 五個固定視角

Kaohsiung-cam04、Taichung-cam01、Taichung-cam10、Taichung-cam11、Tao-Hsin-cam03。

訓練開始前凍結原 CCTV plate、camera.json 與對應的深度幾何快取。
五個快取都含 plate hash 與幾何 signature，並通過目前校正的地面投影一致性檢查。

每視角交付：

1. CCTV 原圖與新模型 19 類分類疊圖。
2. 由新模型分類和既有校正／深度產生的紋理 3D、語意 3D 效果圖。
3. 語意與紋理 GLB，以及模型／影像／校正／深度的雜湊與來源紀錄。

3D 為每鏡頭的可見表面。沒有補造遮擋背面，沒有把各店或各相機當成已註冊的
共用世界座標；絕對公尺尺度仍未經獨立量測驗收。分類來自新模型，沒有借用
舊場景物件或教師遮罩來冒充新模型輸出。

## 無人值守與截止時間

工作根目錄：`runs/studioa_overnight_20260917_v2/`。

- `index.html`：完成後可直接開啟的模型下載與五視角圖集，隨最後交付更新。
- `delivery.json`：目前已完成的模型、視角與時間，綁定同一 checkpoint。
- `status.json`／`summary.json`：批次進度、逐類結果及失敗項目。
- `jobs/*/model/{best,last}.pt`：每組完整 epoch checkpoint。
- `deliveries/*/`：每個已發布候選的模型、PNG、GLB 與來源證據，互不覆寫。

systemd user service 在登出後持續執行，機器已啟用 linger。每組訓練最多 30 分鐘，
失敗即記錄並接續其他組；05:00 停止新增訓練，保留一小時完成模型選擇與出圖。
每張圖先 GPU，失敗可 CPU 重試；至少三個視角驗證成功才更新交付入口。
既有成功交付保留；部分工作失敗時狀態會明示，不能標為全部成功。

## 啟動前驗證

弱類裁切、來源／資料切分、實際 Trainer、幾何投影、出圖完整性、局部相機失敗、
子程序逾時與選模等相關測試已通過。Taichung-cam10 的實際 GPU 推論與
PNG／兩種 GLB 已完成 smoke test；這張 smoke 圖使用前一批模型，只驗證流程，
不作為本夜新模型的正式交付。

## 已啟動的正式版本

凍結程式 `1c1f9e4`，服務 `studioa-overnight-20260917-v2.service` 已啟動。
完整 CPU 回歸 **4,211 通過、12 項因 CUDA 條件跳過**；型別與提交檢查通過。
三組配對設定核對完成，只有 focus crop 與凍結路徑不同。主批次每組 330 epochs、
每 33 epochs 驗證一次，實際更新預算 4,950、驗證間隔 495。

第一版完整回歸發現出圖推論不應放入 commissioning 套件，已移到 offline CLI。
原有套件邊界檢查保留；正式版本重新凍結。第一版已完成的新 bootstrap 模型與
五視角保留在 `runs/studioa_overnight_20260917/deliveries/bootstrap/`，不刪除或改標
為 v2。已啟動的 v2 會另立自己的交付物。

系統確認 linger 已開啟、插電閒置休眠 timeout 為 0、logind IdleAction 為 ignore。
系統未允許取得額外 sleep inhibitor；本輪沒有修改電源設定。

[正式圖集入口](../runs/studioa_overnight_20260917_v2/index.html) ·
[即時狀態](../runs/studioa_overnight_20260917_v2/status.json) ·
[機器可讀交接](reviews/studioa_overnight_20260917.json)。
圖集由背景工作在完整出圖後發布，文件本身記錄啟動快照；最新進度以狀態檔為準。

正式 v2 的 bootstrap 已完成 495 更新，AI source-val mIoU 26.3404%。五張 PNG、
五組 GLB 與 checkpoint／來源雜湊已全部核對，194 個模型 state 張量皆有限。
六組較長訓練已接續啟動；背景流程最後會更新同一圖集入口。
