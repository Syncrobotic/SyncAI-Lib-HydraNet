# StudioA AI 補查與重判

本流程回應「把資料補齊」：對既有 121 張影像補查容易漏掉的展示設備，再以
另一個本機視覺模型逐區域判斷類別。結果仍為 AI 標註，不宣稱每個物件均無漏標，
也不把模型回覆當成獨立正確率。使用者不需要提交人工標註。

## 方法與來源

- 分割模型：本機 SAM3，固定 revision `3c879f39826c281e95690f02c7821c4de09afae7`。
- 重判模型：本機 Cosmos-Reason1-7B，固定 revision
  `3210bec0495fdc7a8d3dbb8d58da5711eab4b423`。
- 增加八個文字查詢，涵蓋直立架、壁面商品架、圓桌、長桌、結帳／服務櫃檯、
  紙箱及滅火器。新遮罩保留提示文字與分割分數。
- 舊正向標籤、未決候選及新遮罩全部進入重判；只把 IoU 超過 0.75 的近似
  遮罩分為同一組，不用包含關係合併桌子和桌上的商品。
- 小商品提供隔離後的目標近照，遮罩外以灰色取代；家具提供保留桌上設備的局部原圖，
  協助區分展示用途與服務用途。模型只回覆類別，沒有逐案文字理由；原始回覆全部保留。
  不支持的物件、混合遮罩、不完整回覆都保留為 unknown。
- 類別定義明確區分展示桌、直立架、服務櫃檯、建築門和櫃體面板，避免將櫃門
  算成房門、商品包裝圖案算成真實裝置。
- 原本已接受的純地板／牆壁／天花板沿用 SAM 語意來源，只扣除前景遮擋，
  明記 source_semantic_preservation，不能宣稱通過第二模型驗證。商品不能任意
  改判成其他商品或櫃檯，人物不能改判成桌子；不支持的改類列未決。
- 接受的前景人物、商品、展示設備與門窗遮罩，依可見遮擋層次從後方遮罩扣除。
  同層分類衝突仍列未決。完整原遮罩保存在來源與 raw 檔，並記錄扣除前後面積。

這是另一個 AI 儀器的重新判斷，不是人工覆核。SAM 分數不等於重判分類可信度。
背景反射、透明材質及遮罩形狀仍可能錯誤，需要抽查和後續更正；對尚未看清的
區域保持忽略，不能為了消除缺類統計而硬補標籤。

## 凍結、執行與續跑

```bash
.venv/bin/python tools/annotation/studioa_autolabel.py relabel-prepare \
  --source runs/studioa_ai_labels_20260916_v3 \
  --out runs/studioa_ai_relabel_20260916_v3 \
  --decisions docs/reviews/studioa_fixture_decisions_20260916.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
TOKENIZERS_PARALLELISM=false \
.venv/bin/python tools/annotation/studioa_autolabel.py relabel-run \
  --out runs/studioa_ai_relabel_20260916_v3 --device cuda
```

prepare 凍結影像、來源標註、完整 src、worker、模型 revision、提示與規則。
run 自動切換到凍結版本。來源／程式變動、另一工作包的 checkpoint，或完成影格
對應 raw 決定檔被改動時拒絕續跑。每十六組寫入一次決定，每張寫入一次標註；
`status.json` 包含影格與組數，`worker.log` 記錄結果與錯誤。

正式批次由 `studioa-ai-relabel-20260916-v3.service` 執行，兩小時上限。
GPU 工作不依賴前景終端存活；沒有向外部服務傳送影像，也不需下載模型。

輸出包括 `sources/` 原標註、`raw/` 全部候選組與模型原始回覆、`frames/` 新標註
及疊圖、`instances_ai.json`、`index.html`、報告與 hash。
後續以 [部分監督匯出](STUDIOA_TRAINING_DATA.md)重算每個整店留出分割的有效資料，
檢查是否仍有缺類；原有 test 影像不會因補標改列 train。

第一版 `studioa_ai_relabel_20260916_v1` 的左右並排原場景／局部輪廓，讓模型
受到場景中的大型櫃檯干擾，曾將指定的人物、商品或地板誤判為櫃檯。助理在
前幾張原圖／遮罩抽查時發現，主動停止該 service；第一版不作為可用訓練資料。
第二版全面灰底隔離，改善大物件干擾，但圓桌失去桌上展示品後被誤判為櫃，
因此也停止，不作訓練來源。第三版使用上述分物件呈現方式，回覆改成單一類名。
助理另看過 22 張原圖與指定遮罩：18 個服務櫃檯、4 個耳機展示桌；判讀綁定
原圖 hash、來源標註 hash、候選 ID，並保留 VLM 原始回覆。

## 同物件的重複遮罩與衝突

IoU 分組不會把桌面和桌體等不同範圍的遮罩合為一組。因此即使已確認一個
櫃檯，另一個範圍較大的遮罩仍可能被判成展示桌，使整區失去監督。
`relabel-review` 可用助理逐圖判讀的 group 編號更正類別，再從原遮罩重算
可見部分、衝突與 coverage。每個更正都綁定原圖和 raw 檔 hash；原始 VLM
決定不修改，另存 AI 決定檔、程式、父版本 hash。這不等於全部遮罩都已驗證。

## 跨類別候選的第二次隔離查核

若同一遮罩同時收到家具與筆電、手機、海報或門窗等候選，家具近照提示可能
使分類偏向展示櫃。`studioa_isolated_recheck.py` 只對這些群組再做隔離目標判讀，
不重新執行 SAM，也不修改原始回覆；輸出綁定原圖、raw、程式及模型版本。
可接續讀取第一批工作已完成的影格，背景服務保留逐張 checkpoint。

實際抽查發現，隔離後模型仍常回答展示櫃。因此「模型重複回答相同類別」不能
用來消除競爭類別：只要來源跨物件類型、模型仍答家具，又沒有助理的指定遮罩
視覺確認，便以 `constrain_isolated_recheck` 保留為 unknown。這個限制套用於
第二次回覆；不能把查核流程完成稱為所有標註已正確。

第三版原始補查與隔離重判保留為來源；最終合併版還包含助理逐圖檢查的消防、
紙箱、商品包裝和重複櫃檯遮罩。一般模型回覆與助理的指定圖像判讀分別記錄。

既有來源中，助理已對前景印表機及三個不成立的玻璃面板候選做過指定遮罩
延後判定。補查不能僅憑 VLM 回答把它們重新升為正向標籤；合併時保留這四個
來源綁定的未決判讀。它們與同圖中已另行確認的左上服務櫃檯是不同候選。

## 本機批次結果與剩餘缺口

最終版 `runs/studioa_ai_relabel_20260916_v4/` 完成 **121 張、24 支鏡頭、19 類**：
2,825 個正向 AI 標籤、5,088 個未決候選。前兩個試跑已停止；第三版完整跑完
後，另對 587 個跨類別群組做隔離重判，最終合併 695 個群組更正／保留未決決定。
助理的指定圖像判讀包含 22 個家具候選、100 個稀少類別候選和 11 個重複櫃檯遮罩。
模型回覆重複不代表正確；不支持的類別、印表機誤判、反射與邊緣不清的區域仍忽略。

`runs/studioa_partial_supervision_20260916_v2/` 完成匯出與完整 hash／分割檢查：
177,318,087 個可監督像素、66,137,913 個忽略像素。監督覆蓋率由 52.62% 變成
72.83%，**是資料量，不是準確率**。未決遮罩依然可以排除看似有顏色的正向區域；
資料集 PNG 和 manifest 的像素統計才代表實際監督量。不同範圍的家具部件遮罩
仍可能衝突，不能宣稱每個櫃檯或每件商品都已完整標好。

| 留出門市 | train / val / test | train 展示櫃／架像素 | train 展示桌像素 | train 櫃檯像素 | train 缺類 |
| --- | --- | ---: | ---: | ---: | --- |
| Kaohsiung | 38 / 29 / 35 | 7,356,581 | 13,331,806 | 517,988 | column, door, cardboard_box |
| Taichung | 22 / 20 / 60 | 2,808,824 | 5,540,034 | 996,083 | fire_equipment |
| Tao-Hsin | 46 / 33 / 26 | 9,071,675 | 10,988,248 | 1,462,965 | 無 |

展示櫃／直立架、圓桌／長桌、櫃檯的核心缺類已補上；但 **19 類全資料有標籤**
不等於 **每個留出方案的 train 都有 19 類**。目前仍有上述四個分割／類別缺口。
不能搬入 test 影像來消除缺類，也不能降低未決規則來製造覆蓋率。

50 項相關測試通過。兩張 train 影像再做一次 CPU 前向／反向／SGD 接線檢查：
loss 4.528347，30,627 個有效像素、18,525 個忽略像素；忽略位置的 logit 梯度全零。
未寫入 checkpoint，未開始正式訓練，未宣稱偵測 loss 或 3D 幾何已完成。

已嘗試讀取使用者提供的 `gs://studioa` 補找新素材，但目前主要帳號需要重新驗證；
其他已存帳號分別沒有物件讀取權限或登入失效。本輪沒有下載新雲端影片，也沒有
變更預設帳號。需要在本機執行 `gcloud auth login`，登入可讀取該 bucket 的帳號，
才能接續補抓資料。**本機補標批次完成，整體資料完整性尚未完成。**

- 全類別標註瀏覽：`runs/studioa_ai_relabel_20260916_v4/index.html`。
- 地板／展示設備／盒裝商品合併預覽：`runs/studioa_ai_relabel_combined_20260916_v4/index.html`。
- 最終訓練資料：`runs/studioa_partial_supervision_20260916_v2/`。
- [可追溯結果與 hash](reviews/studioa_ai_completion_20260916.json)。

重現最終合併與匯出：

```bash
.venv/bin/python tools/annotation/studioa_autolabel.py relabel-review \
  --source runs/studioa_ai_relabel_20260916_v3 \
  --decisions docs/reviews/studioa_group_decisions_20260916.json \
  --out /tmp/studioa-ai-reviewed
.venv/bin/python tools/annotation/studioa_supervision.py export \
  --source /tmp/studioa-ai-reviewed --out /tmp/studioa-partial-targets
```

輸出目錄必須不存在。原始教師輸出、逐組回覆及修正程式均以 hash 留存。
