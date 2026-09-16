# StudioA 場景分類與標註契約 v1

2026-09-16，版本 `studioa.scene.v1`。這份契約落實
[能力審查](STUDIOA_CAPABILITY_AUDIT.md) 的下一步，供標註包與驗證器共同使用。
主要模型的輸出頭、既有標籤 ID 和部署模型尚未切換；不能拿新 ID 直接餵舊模型。

**標註方式更新：依使用者要求，由 AI 完成標註，不要求人工標註。**
[AI 自動標註流程](STUDIOA_AI_ANNOTATION.md) 直接產出逐物件遮罩與 AI 完成狀態。
以下人工工作單保留為可選流程；其 pending 狀態不阻擋 AI 標註的交付。
AI 標籤可用於後續弱監督訓練，但不作為獨立人工精度真值。

機器可讀定義在 [studioa_contract.py](../src/syncai_hydranet/data/studioa_contract.py)，
標註包會輸出其 `contract.json` 與內容 SHA-256。改動分類或遷移規則會使舊包檢查失敗，
必須明確遷移，不能默默沿用。

## Stage 分工與輸出邊界

以下是產品 Stage 名稱；既有 `analytics/stage.py` 的單幀／時序技術介面保留。
本次可執行的新增部分是 Stage1 標註契約、資料準備與人工審查檢查器；
下表其餘輸出為後續接線要求，不宣稱已有完整 runtime producer。

| 階段 | 輸出要求 | 不可省略的未知與證據 |
|---|---|---|
| Stage0 校正 | camera ID、校正版本、原圖尺寸、pixel space、coordinate frame、尺度來源 | 未量測尺度不能標成實測；每鏡頭地面座標不能直接當全店座標 |
| Stage1 場景理解 | source frame/time、契約版本、實體 ID、類別、可見輪廓、材質／用途、3D 位置／支承候選、場景版本 | 未知屬性為 unknown/null；3D 需附校正版本、座標系、估計來源與誤差依據 |
| Stage2 追蹤與屬性 | track ID、camera/time、人物位置與姿態、可觀測衣著／攜物、員工／顧客／unknown、觀測品質 | 單幀被遮擋不能變成否定標籤；跨鏡頭身分與外觀屬性分別驗收 |
| Stage3 時序事件 | start/end time、人物 track ID、場景物件 ID、事件類型、證據影格與狀態變化 | 伸手候選、商品位移、拿取、放回分開；購買需交易對照 |
| Stage4 統計回看 | window、zone、去重範圍、分母、有效觀測時間、缺失時間、事件引用 | 時間積分不能用影格數冒充；單鏡頭人次不能當跨鏡頭獨立顧客數 |

Stage1 分為低頻更新的結構地圖，以及高頻更新的人物、商品位置與開門狀態。
不必每幀重新估計牆面，但物件移動必須反映到場景版本與 Stage3 的關聯。

## 實體、材質、用途及對外分類

19 個基礎實體使用穩定的 annotation ID 1–19；0 保留，255 表示 ignore。
它們不是一個必須共同訓練的 19 通道 dense head。
20 個對外查詢項目涵蓋使用者的 18 個物理分類，加上地面與人物。
`void` 另外作為標註狀態，不是第 21 個物件。

| ID | 基礎實體 | 標註界線 |
|---|---|---|
| 1 | floor／地面 | 可見地面；桌面、櫃底遮擋區不補畫為可見地面 |
| 2 | wall／牆壁 | 可確認的牆面，不含天花板、柱子、門及玻璃面 |
| 3 | ceiling／天花板 | 可見天花板；鏡頭未拍到不等於店內不存在 |
| 4 | column／柱子 | 結構柱；不可因直立形狀就把櫃體標成柱 |
| 5 | door／門 | 同一扇門一個實例，`glazed` 記錄玻璃面是否可確認 |
| 6 | glass_panel／固定玻璃 | 可確認邊界的固定玻璃面；門上的玻璃屬於門實例 |
| 7 | display_cabinet／展示櫃 | 有櫃體的展示設施；開放式層架不直接等同展示櫃 |
| 8 | display_table／展示桌 | 可確認的展示桌體；商品另標實例 |
| 9 | counter／櫃檯 | 櫃檯實體；結帳用途需另外確認 |
| 10 | laptop／筆電 | 開蓋或闔蓋仍須有辨認依據，不能只靠長方形輪廓 |
| 11 | phone／手機 | 與平板無法區分時保留未決區域，不依模糊尺寸硬分 |
| 12 | tablet／平板 | 同上；螢幕展示照片不是實體平板 |
| 13 | boxed_stock／盒裝物品 | 商品包裝，需有包裝／商品用途證據 |
| 14 | cardboard_box／紙箱 | 可辨識的運輸／收納紙箱；用途不明的盒子保留未決 |
| 15 | speaker／喇叭 | 實體喇叭；包裝上的喇叭圖案屬於盒裝物品 |
| 16 | poster／海報 | 實體宣傳印刷面；畫面中的人／商品不另外算實體 |
| 17 | fire_equipment／消防設備 | 暫採滅火器、消防栓／箱等實體設備；細類未確認，火煙另屬事件 |
| 18 | chair／椅子 | 包括凳子；每張可辨認座椅有自己的實例 |
| 19 | person／人物 | 真實人物；海報、反射影像不重複計為人 |

對外 `glass` 對應固定玻璃實體；`glass_door` 是 `door + glazed=true`，且材質包含 glass。
`door` 包含玻璃門，因此兩個查詢結果**不可相加當不同物件數**，必須按 instance ID 去重。
玻璃展示櫃仍是一個 cabinet，可同時帶 glass/metal 等材質，不另外增加一個玻璃門實例。
門框和玻璃面需要細部幾何時，另行建立部件，不從材質清單猜測各部件邊界。

`checkout_counter` 是 `counter + role=checkout`；role_evidence 必須描述店面設定、
人工確認或可見用途證據。只看外型不能保證用途。尚不能確認時使用 `role=unknown`。
未知材質使用空清單；未知 `glazed` 使用 null。這些值不能自動變成負樣本。

## 可見範圍與標註格式

標註座標是**凍結原圖**的像素，`pixel_space=source_image`，不套用 camera.json 的
縮圖尺寸或模型輸入尺寸。多邊形座標有限且在 `[0, width-1] × [0, height-1]` 內。
同一實例可有多個 `polygons_px`，表示被遮擋後分離的可見部分；不補畫背面或不可見尺寸。
透明平面與後方人物可以有重疊輪廓，不能直接壓成單一互斥分割圖再假設沒有資訊損失。

每個 `review.json` 是一張影像的人工工作單。實例 ID 在該影格內唯一，尚非跨影格 track ID。
例如玻璃門實例（僅格式示例，不是 StudioA 實際標註）：

```json
{
  "id": "door-1",
  "entity": "door",
  "polygons_px": [[[10, 10], [30, 10], [30, 50], [10, 50]]],
  "visibility": "visible",
  "glazed": true,
  "materials": ["glass", "metal"],
  "role": "unknown"
}
```

每個對外分類都要明確填寫 `coverage`：

- `present`：至少有一個對應的可見實例及輪廓。
- `absent`：已檢查此影像可見範圍，沒有觀察到該類；不是對整間店的不存在宣告。
- `unobservable`：畫面、解析度或遮擋不足以判斷。
- `unresolved`：有候選，但類別、材質或用途無法確認。
- `unreviewed`：初始值，不能通過完成檢查。

盒裝商品／紙箱不明時填 `unresolved_regions`，保留 polygon、reason 及
`possible_entities=["boxed_stock", "cardboard_box"]`；不能同時把這兩類都宣告 absent。
看不清門的玻璃材質或櫃檯用途時，也不能宣告 glass_door／checkout_counter absent。
`ignore_regions` 保留不監督區域與原因。`present` 的數量仍可能受未決區域限制，
不是已證明完整召回；後續訓練 exporter 必須保留 ignore／未決區域及逐類監督遮罩。

完成時必須填 `reviewer`、含時區的 `reviewed_at`，並將 status 改成 reviewed。
驗證器檢查來源、格式、輪廓與分類自洽，**不會證明填寫者的身分或人工判斷正確**。
即使全數 review_complete，也還要通過獨立資料切分、逐類覆蓋與幾何／事件驗收。

## 舊資料的保守遷移

| 原始格式 | 保留為建議標籤 | 改為 ignore 並記錄待重標 |
|---|---|---|
| SITE30K native | floor、column、display_table、person、laptop、tablet、phone、boxed_stock | wall、shelf |
| RETAIL_OBJECTS native | floor、column、person | wall、fixture、product |
| RETAIL_SURFACES native | floor、column、person | wall、fixture |

「保留」只保留來源已做過的語意區分，仍是教師候選，不是人工真值。
原始 void 與 255 都忽略；未知 label ID、RGB 彩色 mask、尺寸不符會拒絕處理。
這裡只接受原始格式名稱，不接受已合併過的 `site30k_to_surfaces` 冒充 native。
遷移不修改來源資料，也不註冊新模型的訓練 LabelScheme。

## 產生及檢查工作包

```bash
.venv/bin/python tools/annotation/studioa_review.py prepare \
  --source datasets/site30k_v1 site30k_native \
  --source datasets/retail_objects_batch02 retail_objects_native \
  --source datasets/retail_objects_batch03 retail_objects_native \
  --per-camera 3 --out runs/studioa_contract_review_20260916_v1

.venv/bin/python tools/annotation/studioa_review.py check \
  --out runs/studioa_contract_review_20260916_v1
```

prepare 只讀取原資料，建立新輸出目錄，拒絕覆寫已有工作包。
選樣按每資料源／鏡頭分散 session，取各 session 的中間影格，沒有依模型分數挑好看的例子。
`per-camera` 是 session 數上限，不保證每種物件都已出現。索引數是配對影像數；
類別像素統計僅涵蓋選中的樣本，不能冒充全資料集統計。

工作包包含原圖、原始 mask、保守遷移 mask、空白人工標註單、瀏覽頁、來源 manifest、
契約與審核狀態。SHA-256 綁定凍結檔案；PNG/JPEG 解碼後像素 hash 用來檢查所選樣本的
跨分組重複。check 對待審核回傳 2，來源或標註錯誤則失敗；review_complete 才回傳 0。

三個店別輪流留出規劃使用既有 store_split：目標店全為 test，來源店舊 test 不使用。
空分組、相機跨分組及相同像素跨分組會標為 blocked。它只是**所選樣本**的未來切分規劃，
沒有對全資料集做內容去重，也不代表現有模型沒看過它們。
現有資料可能已參與訓練或調參；真正盲測需另外保留未使用的來源與訓練排除清單。
本工具不輸出訓練集、不啟動 GPU 工作，也不宣告任何新類別的模型精度。

## 後續驗收順序

先補齊本包中缺少／未決的各類標註與困難負例，確認消防細類與開放層架的處理。
之後再設計結構分割、物件實例、材質／用途的各自訓練輸出與 head-specific exporter。
小型手機／平板需驗證實際解析度，不能只增加 dense head 通道便假定可分辨。

此包是單張場景標註，不供 Stage3 事件真值。第一條互動流程要另外保存連續片段及時間戳：
人物到展示桌 → 停留 → 伸手候選 → 商品位移 → 拿取／放回。
先核對人／物實例與觀察品質，再量測事件誤報、漏報、延遲；不能從三張離散影像推定事件。
