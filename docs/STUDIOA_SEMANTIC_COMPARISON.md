# StudioA 新版語意資料受控比較

2026-09-16。這輪比較資料修訂整體是否改善 Stage1 對保留 AI 像素的學習；
不將訓練完成或標籤一致率當成整店 3D／顧客分析的驗收。

## 訓練前固定的方案

| 項目 | 對照 | 候選 |
| --- | --- | --- |
| train 來源 | `studioa_partial_supervision_20260916_v2`，46 張 | `studioa_semantic_ready_20260916_v1`，62 張 |
| val 來源 | 共用 `studioa_semantic_ready_20260916_v1` 的 33 張 | 相同 |
| 更新預算 | 495 次，45 epochs | 495 次，33 epochs |
| 選模時點 | 初始化及第 165／330／495 次更新 | 相同 |
| 選模依據 | 全部 19 類 source-val scene mIoU | 相同 |

兩組使用 ResNet18 ImageNet 骨幹、64 通道 FPN、19 類 scene head、512×896
輸入、batch 4、seed 42、AdamW、lr 0.0002、相同增強與初始化。
共用舊版 46 張 **train** 計算的類別權重，驗證像素不參與權重計算。
固定 deterministic 模式、停用 TF32 與 early stopping；完成後逐張量核對
`initial_state.pt`，並核對兩組實際更新数、驗證時點與評估輸入雜湊。

候選同時增加影格與修訂標籤，因此結論只針對整份資料修訂，不能單獨歸因於
多出的 16 張或某一類補標。單一 seed 是先導比較，不提供統計穩定性結論。

## 隔離與保存

`studioa_train.py prepare --scene-validation-source` 將驗證來源另行凍結，
只允許相同 source-val 影格、相機與影像內容；訓練資料只由各自的 train 來源提供。
現有 Trainer 的 validation-only 設定與跨資料集鏡頭洩漏檢查繼續生效。
兩组不建立 test loader，不執行 test 推論，不依 test 結果調參。

工作根目錄：`runs/studioa_semantic_comparison_20260916_v1/`。
每組保存程式／設定／資料／權重雜湊、持久服務 log、status、初始／best／last
權重及 baseline／validation 報告。工作由 systemd user service 依序執行，
完成後才檢查結果；SSH／編輯器斷線不會停止。

## 結果判讀規則

共同新版 val 的 19 類 mIoU、每類 IoU 與差值全部呈現，另外明列柱子、門、
固定玻璃、手機、消防設備；同時指出其他類是否退步。玻璃 val 僅一個相機，
重複影格不是獨立場景，已揭露 AI 標籤不是獨立真值。

這輪不自動替換既有模型。若整體上升但弱類未改善，下一步仍須針對失敗類型
檢查標註、解析度與採樣；不能只凭整體均值宣稱資料已足夠。

## 執行狀態

方案已固定；工具的獨立驗證來源、真實 CPU 訓練路徑與相關資料／完整性檢查
共 58 項測試通過，Ruff 與修改模組 ty 通過。GPU 結果完成後補入本頁。
