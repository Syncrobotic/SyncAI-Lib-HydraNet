# STUDIO A 目標分類與 Stage 1–4 能力核對

2026-09-16。依本次需求整理；這是目標與實作差距審查，不代表新增類別已訓練或部署。
結論：**以場景理解支援顧客屬性、時序行為和統計是合理設計；目前專案部分具備，尚未完整符合。**

## 階段定義先對齊

目前 README 的操作流程是 Stage0 建置、Stage1 連續分析、Stage2 報表；
`STUDIOA_STAGE0_4.md` 的舊目標則把 Stage1 定義為人員追蹤。`analytics/stage.py`
又使用第一階段單幀感知／第二階段追蹤分析的技術邊界。這些不同編號不能混作驗收。
本次以使用者所說的 **Stage1 看懂場景** 作為新的產品目標，建議如下；程式介面未改名。

| 階段 | 責任與輸出 | 目前狀態 |
|---|---|---|
| Stage0 相機與世界座標基礎 | 鏡頭、內外參、尺度、來源與校正版本；多鏡頭映射至店面座標 | 有單鏡頭 camera.json、地面座標、cache、審核工具；獨立尺度與完整跨鏡頭店面座標仍不足 |
| Stage1 場景／物件理解 | 結構分割、物件實例、人物偵測、材質／角色、3D 位置及支承關係；異動更新 | 有粗分類 student、離線教師、部分實例與場景建模；未涵蓋所列全部分類，也未完成整店 3D 驗收 |
| Stage2 人員追蹤與屬性 | track、位置、姿態、員工／顧客／unknown、可用外觀屬性及觀測品質 | 追蹤、crop encoder、屬性池化與員工分類存在；新店準確率、遮擋穩定度、跨鏡頭關聯仍需驗收 |
| Stage3 時序行為與互動 | 進出、停留、區域占用、排隊；人—商品的拿取／試用／放回等事件 | 區域／軌跡規則與部分姿態候選已實作，pilot 能記錄事件；特定商品拿取／放回尚非已驗證能力 |
| Stage4 統計與回看 | 分區人數、停留人秒、動線、屬性分群、事件回看、服務健康狀態 | 有離線報表、熱圖與事件記錄；完整常駐、多鏡頭、具驗證時延的介面仍未完成 |

Stage4 以已驗證觀測彙整統計；不應另外猜測 Stage2/3 沒看清楚的屬性或行為。
到結帳區只能表示到訪／停留；若要宣稱購買或交易轉換，仍需交易資料對照。

## 逐類核對

「有標籤／提示詞／mesh 模板」不等於「主要 student 能辨識並已在新店通過驗收」。
目前主要零售 student 偵測頭為 `person, bag, boxed_stock, device`，分割頭為
`void, floor, wall, column, fixture, person`；`void` 轉為 ignore。SITE30K 另有較細資料
格式，但六類訓練會把桌架及四種商品重新合併成 `fixture`。

| 本次需求 | 目前對應證據 | 是否完整符合 |
|---|---|---|
| 筆電 | SITE30K `laptop`；SAM3 個別實例與 laptop 模板；主 student 偵測歸 `device` | 部分，非主要模型獨立完整類別 |
| 手機 | SITE30K `phone`；教師實例與模板；主偵測歸 `device` | 部分 |
| 平板 | SITE30K `tablet`；教師實例與模板；主偵測歸 `device` | 部分 |
| 盒裝物品 | `boxed_stock` 偵測、SITE30K 與離線商品遮罩；部分輸出為示意排列 | 部分，實例／支承與泛化仍需驗收 |
| 紙箱 | 沒有獨立於盒裝商品的正式類別與驗收鏈 | 缺少；不能把商品包裝自動當運輸紙箱 |
| 喇叭 | 教師 product 提示詞包含 `speaker`，但合併為一般商品 | 缺少獨立 student 類別與場景資產鏈 |
| 海報 | ADE20K 海報映射為 `wall`；部分假陽性分析提到海報 | 缺少獨立辨識／建模類別 |
| void | 已有未標註／不監督語意，mask ignore 為 255 | 應作忽略區域，不能當可重建實體；也不同於實例的 unknown |
| 消防 | 未找到獨立消防設備或火煙訓練／輸出類別 | 缺少；暫按消防設備理解，具體子類待確認 |
| 椅子 | SAM3 `chair`、`stool` 實例與模板 | 部分，未納入主要四類偵測頭 |
| 牆壁 | 粗分割與牆平面／開口切割／回投影 | 部分，既有 wall 同時混入天花板、門、玻璃等 |
| 天花板 | 舊分割將 ceiling 合併為 wall | 缺少獨立 ceiling 平面與驗收 |
| 門 | 舊分割併入 wall；離線提示、接地平面與開口控制存在 | 部分 |
| 柱子 | `column` 分割、實例／幾何與場景輸出存在 | 部分，仍需新店逐類與幾何驗收 |
| 玻璃 | 舊粗分割合併為 wall；另有玻璃專用候選與人工／控制點流程 | 部分；專用候選不能冒充已驗收主模型 |
| 玻璃門 | 專用 glazing 候選、門框控制與材質 mesh | 部分；尚未達到新店可靠驗收 |
| 展示櫃 | SITE30K `shelf`、場景 `display_shelf`；主要分割仍為 fixture | 部分；層架、封閉櫃、玻璃展示櫃的細分不完整 |
| 展示桌 | SITE30K `display_table`、桌面支承與場景擬合 | 部分；主要分割仍為 fixture |
| 結帳櫃檯 | 有 counter 模板／till 區域概念；主要類別仍混入 fixture/table | 缺少獨立已驗證的結帳角色辨識；需店面設定或人工確認 |

現有資料或教師通道只是可沿用的起點，不能由上表推算整個需求「完成百分比」。

## 分類契約需要的調整

1. **補 floor、person。** 地面提供支承與位置基準，人物偵測提供後續追蹤；兩者不能從
   Stage1 的輸出契約省略。看不到的背面、被遮擋區域與未量測尺度需明示未知／推測。
2. **結構、實例、材質與用途分開表達。** 牆、地面、天花板適合密集分割；筆電、手機、
   椅子、櫃桌需要 instance ID 與輪廓。玻璃門可以對使用者輸出一類，內部同時記錄
   `door + material=glass`；玻璃櫃也應保留櫃體及玻璃面，避免彼此搶同一個互斥類別。
3. **功能性角色不能僅憑外型保證。** 結帳櫃檯可記為 `counter + role=checkout`，
   由可見證據及店面設定確認。外觀無法區分的紙箱／盒裝商品可暫存 box + unknown role，
   但要有一致的人工標註準則，不能用不可靠的細分當 ground truth。
4. **void 不生成物件。** 未標註像素繼續由 ignore 控制；已偵測但無法可靠細分的物件則
   保留 unknown／父類，避免把未知區域偷偷訓練成某種場景實體。
5. **靜態地圖低頻更新、人物與互動逐幀處理。** 不需每幀重建牆與天花板，但商品、椅子、
   開門狀態會變，不能永久停留在 commissioning 底圖。場景異動需要更新版本及下游通知。

單張分類圖只能告訴系統畫面上哪裡像什麼。還原可量測 3D 仍需校正、尺度、深度／
多視角、可見表面與支承關係；把辨識到的 laptop 放入既定模型，不代表已恢復該實物形狀。
本專案現階段較接近「受幾何約束的語意場景代理」，尚非完整精密數位分身。

## 屬性與行為的現況界線

`data/attributes.py` 與 `models/crop_encoder.py` 有 PA-100K 外觀屬性，包含粗年齡段、
性別外觀標籤、視角、衣著、包等；`track_attributes.py` 做軌跡聚合。這些能力不能因
逐幀結果穩定，就被視為在 StudioA 已有人工真值精度。`staff.py` 明確要求相機驗證。

`WorldFrame` 目前描述單相機的 `camera_floor(camera_id)`，不是已完成對齊的全店座標。
已有世界座標轉換、追蹤、區域規則、`CameraAlerts → ZoneMonitor → record_alert` pilot。
`reach_to_shelf` 是手腕接近設施的候選，程式明確說明它不是購買、拿起或與特定商品互動。
真正的拿取／放回需要穩定的人與商品實例、時間窗口、遮擋處理及狀態變化證據。

## 建議接續順序（本次不啟動訓練）

後續已落實 [v1 分類與標註契約](STUDIOA_SCENE_CONTRACT.md)，以及本機標註包準備／
檢查工具；主要模型分類與本頁所列能力差距尚未因此改變。

先凍結完整分類／屬性／用途契約、資料遷移與 Stage 命名；其次為每一類建立人工核對的
店別留出集，查清「模型有輸出」與「場景有實例」之間缺哪些環節，再決定分割／偵測頭、
教師補標與訓練配置。舊粗標籤無法直接還原出已合併的天花板、玻璃門與結帳角色。

Stage1 要按類別驗收 instance precision/recall、輪廓及支承／投影與實測幾何誤差；
Stage2 驗收追蹤斷裂與屬性的人工留出結果；Stage3 驗收時序事件 precision/recall、誤報與
漏報；Stage4 驗收去重範圍、時間積分、端到端延遲與長時間運行。不同證據不得互相替代。
先打通「人到展示桌 → 停留 → 伸手 → 商品位置改變 → 放回」的一條可核對流程，
比在尚未明確的標籤上繼續擴大訓練更直接。

## 主要程式依據

- [主產品模型配置](../configs/hydranet_retail_person02.yaml)、[偵測詞彙](../src/syncai_hydranet/data/label_maps_retail_security.py)
- [粗分割映射](../src/syncai_hydranet/data/label_maps_retail_objects.py)、[SITE30K 細分類及合併](../src/syncai_hydranet/data/label_maps_site30k.py)
- [離線物件實例提示](../src/syncai_bev3d/object_instances.py)、[商品補充分割](../tools/commissioning/extras_pass.py)、[物件模板](../src/syncai_bev3d/object_assets.py)
- [人物屬性](../src/syncai_hydranet/data/attributes.py)、[軌跡屬性](../src/syncai_hydranet/analytics/track_attributes.py)、[員工分類限制](../src/syncai_hydranet/analytics/staff.py)
- [單幀／分析契約](../src/syncai_hydranet/analytics/stage.py)、[WorldFrame](../src/syncai_hydranet/analytics/world.py)、[事件類型](../src/syncai_hydranet/analytics/events/_types.py)、[姿態事件界線](../src/syncai_hydranet/analytics/events/pose.py)
- [服務端事件接線](../src/syncai_hydranet/serving/alerts.py)、[過去 Stage0–4 分工](STUDIOA_STAGE0_4.md)
