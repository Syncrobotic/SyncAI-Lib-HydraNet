# StudioA AI 自動標註

依使用者要求，標註由 AI 完成，不需要另外提交人工標註工作單。
AI 與人工來源分開記錄；AI 完成狀態是 `ai_labeled`，不是人工工作單的 `reviewed`。
使用同一份 [scene.v1 分類契約](STUDIOA_SCENE_CONTRACT.md)，保留完整需求分類與未知屬性。

## 流程與產物

`tools/annotation/studioa_autolabel.py` 使用本機快取的 SAM 3，固定模型與 revision，
每張影像編碼一次，共查詢 38 個文字提示。除了需求類別，另查詢螢幕與開放層架，
避免把這些範圍外物件硬塞進筆電或展示櫃。

- 同類別重疊實例去重。玻璃門保留同一 door ID，加上 AI 推測的 glazed/materials。
- 結帳櫃檯由用途提示產生 AI role 候選，附 prompt 來源，不宣稱交易功能已被實測確認。
- 地面、牆面、天花板合併為 semantic region；它們不是已識別的獨立 3D 平面。
- 不同類別對同一物件的強重疊保留為 unresolved，不用最高分強行決定類別。
- 結構類只將衝突像素標為未決，保留其餘可用區域。物件實例的類別衝突則保留完整候選。
- 海報內的裝置／人物與櫃體／門內的玻璃面另列疑似圖案或部件，不重複算成獨立物件。
- 沒有輸出是 `not_detected`；屬性不明或衝突是 `uncertain`，都不等於確定不存在。

每張輸出包含：

- 原始解析度的 COCO RLE 遮罩，保留洞與不連通區域；bbox 與 area 必須和遮罩相符。
- entity、AI score、prompt、材質／用途、來源影像 hash、模型版本、job hash。
- 未決候選遮罩與原因；`human_reviewed=false`。
- 原圖和疊圖，以及 `index.html` 瀏覽頁。

`instances_ai.json` 彙整正向實例／語意區域標籤。它是 AI 偽標籤的交換格式，
不是可以忽略缺漏直接拿來算完整偵測 loss 的 COCO 真值。
後續訓練需一起消費 companion JSON 的未決區域、逐類監督狀態與模型 head 對應，
避免把沒偵測到的商品當背景。`training_ready=false` 表示尚未完成這個 exporter，
不是要求使用者改做人工標註。

後續已新增[部分監督語意匯出與接線檢查](STUDIOA_TRAINING_DATA.md)，會保留未知與
衝突的 ignore 像素；這不會讓原始正向 COCO 自動具備完整偵測監督。
針對展示設備缺口，另有[本機 AI 補查與逐區域重判](STUDIOA_AI_COMPLETION.md)，
保留舊來源與模型回覆，不以提高分數或放寬衝突門檻代替分類更正。

## 執行及續跑

```bash
.venv/bin/python tools/annotation/studioa_autolabel.py prepare \
  --bundle runs/studioa_contract_review_20260916_v1 \
  --out runs/studioa_ai_labels_20260916_v1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
.venv/bin/python tools/annotation/studioa_autolabel.py run \
  --out runs/studioa_ai_labels_20260916_v1
```

prepare 凍結來源影像、模型 revision、分類規則和程式 snapshot；不修改舊標註包。
run 自動切換到凍結的 worker/source，透過檔案鎖避免重複 worker。
每張完成後原子寫入 JSON，更新 status.json；再次執行會驗證已完成影格後續跑。
錯誤與終止訊號會留下失敗原因。長批次使用 systemd user service，設定時間上限，
並將 stdout/stderr 寫入工作目錄的 worker.log。

`refine` 可沿用完整分割結果，單獨調整衝突處理，不重跑模型、不覆寫來源結果：

```bash
.venv/bin/python tools/annotation/studioa_autolabel.py refine \
  --source runs/studioa_ai_labels_20260916_v1 \
  --out runs/studioa_ai_labels_20260916_v2
```

refine 檢查來源完整報告與逐張標註 hash，保留 parent annotation/report 身分及新的規則。
語意區域與未決區域的所有輸出仍保留原始 pixel space。完成報告綁定每張 JSON 與 COCO 檔。

## AI 視覺檢查更正

`visual-review --source COMPLETED_AI_DIR --decisions DECISIONS_JSON --out NEW_DIR`
套用助理查看原圖後的 AI 檢查記錄。每筆決定綁定影格 ID、影像 hash、原標註 hash
及候選 ID，只允許將正向候選改列未決，不會憑空新增類別或修改遮罩。
記錄仍是 AI 來源，不是人工審核。新版本保留基礎 job 身分，另附 AI 決定檔、
執行程式、規則 hash 與來源報告 hash；原版本保持不變。

## 品質界線

本流程完成的是 AI 標註，不是主要 HydraNet 模型新增類別的訓練。
score 是教師分數，不是實際正確率。相同模型的多個提示也不構成獨立驗證。
遮擋、小物件、鏡面反射、透明玻璃、用途判斷仍可能錯誤；未決區域會保留供後續
AI 重判或多影格約束處理，不要求使用者逐張人工補標。

影像來自既有資料集，因此不能冒充未見過的盲測集；標註遮罩也不提供實測尺寸、
跨鏡頭幾何或拿取／放回的時序真值。這些能力會在後續階段分開處理。

## 本批結果

最終結果在 `runs/studioa_ai_labels_20260916_v3/`：121 張、24 支鏡頭，
4,687 筆正向語意區域／實例候選，另有 4,817 筆未決候選。
原始推論 v1、語意衝突整理 v2 及 AI 視覺更正 v3 均保留。
助理抽查代表影像，對一張影像記錄四筆更正，並非宣稱逐張完成獨立精度驗證。
詳見 [可追溯報告](reviews/studioa_ai_labels_20260916.json)。

## 地板、展示桌與展示櫃顯示檢查

原疊圖的地板為低透明度灰藍色，與原地板接近，且不顯示未決候選。
v3 的 121 張都有地板保留遮罩，但不代表邊界完整或正確；展示櫃只有 12 筆保留
候選（11 張），另有 359 筆因分類衝突列為未決。展示櫃不足是標註問題，不能用
單純改色或把未決候選全部升格來解決。

使用者所指的展示設備包含圓桌、長桌；分類契約將它們列為展示桌
（`display_table`），不是 `display_cabinet`。v3 展示桌有 101 筆保留候選、
651 筆未決候選，不能拿展示櫃的數量代表所有展示設備。
`focus_v1` 漏了展示桌預覽；`focus_v2` 已加入。可檢查這三類，無需重新推論：

```bash
.venv/bin/python tools/annotation/studioa_autolabel.py focus-preview \
  --source runs/studioa_ai_labels_20260916_v3 \
  --out runs/studioa_ai_labels_20260916_focus_v2
```

開啟輸出的 `index.html`。綠色為地板保留遮罩，藍色為展示桌保留遮罩，
桃紅色為展示櫃保留遮罩，橘色斜線為未決候選；可切換隱藏未決候選。
三類分開顯示，避免其他物件遮罩
蓋住著色。原圖與每張候選數一起提供；未上色不表示物件不存在。
此命令驗證來源 hash、保留原標註包，另存預覽程式與輸出 hash。
這是顯示改善，展示櫃的 AI 重判與缺漏補標仍待處理，尚未開始新訓練。

使用者進一步指定「展示櫃」包括擺放 boxed-stock 的直立展示架。
整合預覽把 `display_cabinet` 與原先範圍外的 `other_shelf` 候選一起顯示為櫃架，
與地板、圓桌／長桌、盒裝商品疊在同一張圖上：

```bash
.venv/bin/python tools/annotation/studioa_autolabel.py focus-preview \
  --source runs/studioa_ai_labels_20260916_v3 \
  --out runs/studioa_ai_labels_20260916_combined_v1 --combined
```

綠色為地板、藍色為展示桌、桃紅色為櫃架、黃色為 boxed-stock。
實色為保留的 AI 標籤，同色斜線為該類未決候選，可在網頁切換隱藏。
圖例直接附在每張圖上。重疊按地板、桌、櫃架、商品順序顯示，
這只是視覺順序，不代表跨類別衝突已解決；既有 `other_shelf` 仍維持未決來源。
本次依使用者用語新增顯示分組，沒有把舊標籤直接升格成新的訓練真值。
