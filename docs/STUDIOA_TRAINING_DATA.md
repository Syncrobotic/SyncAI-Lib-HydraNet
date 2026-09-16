# StudioA 部分監督訓練資料

這份資料將 AI 標註轉成可讀取的語意目標，並接入既有 Trainer 進行有限試訓練。
訓練狀態與結果見 [首輪試訓練](STUDIOA_PILOT_TRAINING.md)，不表示分類正確率已通過驗收。
不需要使用者人工標註；缺漏與衝突仍由後續 AI 重判、補標處理。

## 類別與監督規則

`studioa.partial-semantic.v1` 提供獨立的 19 類輔助語意目標，包含場景結構、
展示設備、商品及人物。小商品仍需要實例偵測；這份輔助目標不取代偵測分支。
原標註 entity ID 為 1–19；輸出 mask 明確使用 **0–18**，對應順序記在
`manifest.json` 的 `policy.classes`，**255 是忽略**。0 是地板，不是背景。
舊模型的 6 類／7 類標籤不能直接讀取這些 mask。

- 圓桌、長桌屬於 `display_table`。
- `display_cabinet` 的目標範圍包含直立展示架。舊 `other_shelf` 未決候選只記錄
  這個對應，不會自動升格為正向標籤。
- 保留標籤只有在該像素沒有其他類別的保留遮罩、也沒有任何未決遮罩時才提供監督。
- 同類重疊取聯集；不同類重疊、未決候選覆蓋區，以及沒有標註的像素全部設為 255。
  不以分數、畫圖順序或外框大小決定像素類別。
- companion JSON 保留完整原始遮罩、屬性、候選與來源。玻璃材質、結帳用途
  不會因為語意匯出而變成獨立物件。

這個保守規則可以阻止未知區域被當成背景，但不能修正「看似確定、實際錯誤」
的教師標籤；先前抽查發現的地板反射誤判等問題仍需 AI 更正。
正向 COCO 仍不能直接用於一般 FCOS loss，`detection_training_ready=false`。

## 匯出與驗證

```bash
.venv/bin/python tools/annotation/studioa_supervision.py export \
  --source runs/studioa_ai_labels_20260916_v3 \
  --out runs/studioa_partial_supervision_20260916_v1

.venv/bin/python tools/annotation/studioa_supervision.py check \
  --data runs/studioa_partial_supervision_20260916_v1
```

輸出包含凍結影像、PNG 目標、原始 companion、產生程式、manifest、report 與
status。拒絕覆寫既有目錄；來源 job、逐張標註與影像 hash 必須一致。
每張影像記錄有效／忽略像素、正向類別重疊像素及逐類有效像素數。

三個門市各自有一份整店留出分割：目標店全數為 test；來源店原 test 鏡頭排除。
同鏡頭或相同解碼像素出現在不同分割時，匯出失敗。每個分割另記逐類有效像素，
並明列訓練分割中沒有正向像素的類別。這些影像曾用於舊實驗，不能宣稱盲測。

## 真實影像接線檢查

```bash
.venv/bin/python tools/annotation/studioa_supervision.py smoke \
  --data runs/studioa_partial_supervision_20260916_v1 \
  --held-out Taichung \
  --out runs/studioa_partial_smoke_20260916_v1
```

`StudioAPartialDataset` 明確指定留出店與分割，使用既有等比縮放／padding。
此 reader 可由 PyTorch DataLoader 與現有 `collate` 消費；目前也已加入通用 Trainer
的資料集工廠，設定 type 為 `studioa_partial`，只監督 19 類 `scene`。
工廠禁止 label_map、分割角色重映射與驗證／測試增強，並檢查跨資料集鏡頭洩漏。

smoke 僅取兩張 train 影像，使用隨機初始化的 ResNet18 + FPN + 19 類 scene
語意分支，CPU 做一次前向、反向與 SGD 更新。確認有限損失／梯度、忽略像素對
logit 的梯度為零、分類器有更新，留下來源與程式 hash。無下載、無 checkpoint，
不做 test 集選模，也不把這一步稱為訓練成果或精度提升。

分割損失已處理全忽略批次：交叉熵依有效權重正規化，無有效像素時回傳可微分的
零，而非 NaN；正常批次保留原本的加權 CE 與 Dice 意義。

後續先依逐類有效像素檢查 AI 標註缺口，再做修正與擴充。正式方案仍需實例偵測的
部分監督、頭部與權重遷移設定、較高解析度小商品檢查及持久化試訓練。

## 第一版匯出結果（v1）

121 張、24 支鏡頭，共 128,112,723 個可監督像素、115,343,277 個忽略像素。
可監督比例為 52.62%，是篩選後的資料量，不是準確率。
三個整店留出分割均通過鏡頭與解碼像素隔離檢查。

| 留出門市 | train / val / test 張數 | train 展示櫃像素 | train 展示桌像素 | train 櫃檯像素 |
| --- | --- | ---: | ---: | ---: |
| Kaohsiung | 38 / 29 / 35 | 1,129 | 65,422 | 87 |
| Taichung | 22 / 20 / 60 | 0 | 5,933 | 0 |
| Tao-Hsin | 46 / 33 / 26 | 1,129 | 70,585 | 87 |

來源店原 test 張數未計入上表。台中留出方案缺展示櫃與櫃檯正向監督，其餘方案
雖非零也很稀少，因此本批資料不宜直接當完整 19 類正式訓練集。

台中留出方案的 train 兩張影像完成 CPU 單步接線檢查：26,410 個有效像素、
22,742 個忽略像素，loss 4.674478，分類器權重更新範數 0.003657；忽略位置
logit 梯度全為零。此 loss 是隨機初始化單步數值，不是學習進度或模型表現。
[可追溯結果](reviews/studioa_partial_supervision_20260916.json)保存來源與結果 hash。

## 補標後匯出結果（v2）

最新來源是 `runs/studioa_ai_relabel_20260916_v4/`，訓練資料為
`runs/studioa_partial_supervision_20260916_v2/`。仍為 121 張、24 支鏡頭；
177,318,087 個可監督像素、66,137,913 個忽略像素，覆蓋率 72.83%（不是準確率）。
整店分割不變，三個方案的展示櫃／架、展示桌與櫃檯均已有有效監督。

| 留出門市 | train / val / test | train 展示櫃／架像素 | train 展示桌像素 | train 櫃檯像素 | train 缺類 |
| --- | --- | ---: | ---: | ---: | --- |
| Kaohsiung | 38 / 29 / 35 | 7,356,581 | 13,331,806 | 517,988 | column, door, cardboard_box |
| Taichung | 22 / 20 / 60 | 2,808,824 | 5,540,034 | 996,083 | fire_equipment |
| Tao-Hsin | 46 / 33 / 26 | 9,071,675 | 10,988,248 | 1,462,965 | 無 |

高雄留出仍缺柱子、門、紙箱，台中留出仍缺消防；桃新留出 train 的 19 類都有
非零監督，但非零不等於數量足夠或正確率已驗收。正式全類別跨店訓練仍需補資料。
完整 hash／鏡頭與影像隔離檢查、50 項相關測試及新資料的 CPU 單步接線檢查通過；
該次 smoke 未儲存 checkpoint；後續獨立 GPU pilot 見上述試訓練文件。
見 [AI 補查方法與雲端登入阻塞](STUDIOA_AI_COMPLETION.md)及
[結果記錄](reviews/studioa_ai_completion_20260916.json)。

## 小商品修正與部分偵測接線（v3）

`runs/studioa_partial_supervision_20260916_v3/` 修正一張來源 train 影像：新增兩個
手機遮罩、將一個遙控器錯標改列 unknown。手機 train 像素 40,021 → 44,982；
其餘 120 張語意 mask、所有分割及 val／test 保持不變。

原始正向 COCO 仍不是完整實例真值。另有明確 AI 覆核的 `studioa_instances`
資料集與 FCOS 部分監督損失，首批 6 張、15 個實例及 6 塊空白區已完成 GPU
三步接線檢查，未知位置梯度為零。尚未完成全類別偵測資料或正式偵測訓練。
見 [小商品修正、資料契約與驗證](STUDIOA_PARTIAL_DETECTION.md)。
