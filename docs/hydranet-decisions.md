> **已被取代（2026-08-22）。** 現行文件是 [VISION.md](VISION.md)（願景與成功定義）與
> [PLAN.md](PLAN.md)（模型設計、資料策略、建置順序）。本檔保留作為稽核軌跡——§8 記錄了
> 08-19/20 的偏離，完整量測在
> [journal/2026-08-22-drift-audit-and-course-correction.md](journal/2026-08-22-drift-audit-and-course-correction.md)。
> **不要拿本檔當作現行計畫。**

# HydraNet(LocalMind Retail)決議摘要 — Session Handoff

日期:2026-08-22
狀態:架構與資料管線已定案,待執行

---

## 1. 系統形態:三個獨立模型(已定案)

**不是單一網路。** 屬性 / ReID / 行為刻意不掛在主網路上(crop 解析度物理限制 + 80 路算力預算)。

### 模型 1:HydraNet-Dense(GPU,每幀)
- 架構:RegNetX-800MF → BiFPN×2(P3–P7, 96ch)
- 輸入:`[B, 3, 544, 960]` FP16(推理解析度 960×544)
- 輸出 1 — Seg(掛 P3,**低頻排程執行**,非每幀):`[B, 10, 68, 120]`
- 輸出 2 — FCOS 偵測(P3–P7):6 類 = person / iphone / ipad / macbook / boxed_stock / stack
- 參數 ~8–9M,FP16 ~17–18MB
- 主網路輸出保持「笨」:只出框 + mask,不出屬性/ID/行為

### 模型 2:PersonNet(GPU,crop 二階段,每 track 每 2–3 秒)
- 輸入:person crop `[B, 3, 192, 96]`
- 小 backbone(RegNetX-200MF 或 OSNet)→ 四頭:
  - reid_embed `[B, 128]`(L2-normalized)
  - age `[B, 4]`:child / teen / adult / senior
  - gender `[B, 3]`:male / female / unknown
  - staff `[B, 3]`:staff / customer / unknown ← **已確認要加**
- 屬性採 track 級投票,非每幀

### 模型 3:BehaviorNet(CPU,track 時序)
- 輸入:`[B, 30, 12]`(3 秒軌跡:地面座標、速度、加速度、bbox 幾何、zone)
- 2 層 1D-CNN,<100K 參數
- 輸出 4 類:sit / crouch / fall / none
- **walk / run / stand / loiter 不進模型**,由規則引擎用 m/s 速度 + dwell time 判定

---

## 2. Seg Head:保留,6 類(已定案,2026-08-22 修訂)

**機器狗線已移除。** 本專案只服務安防零售 CCTV 俯視單一視域。原本的 10 類裡有 4 類
(`floor_carpet` / `ramp` / `stairs` / `movable_obstacle`)只為機器狗的步態與避障存在,
一併刪除。機器狗程式碼本身已於 2026-08-19 在 `cc80fc3` 刪除(24 檔,−4,118 行)。

保留 seg head 的理由改為:zone 遮罩、展示架防盜、出入口事件。三者都是固定相機上的靜態
結構,所以 §1 的「低頻排程執行,非每幀」是這個 head 的正確用法。

| ID | Class | 主要用途 |
|----|-------|---------|
| 0 | background | — |
| 1 | floor | zone + 可通行 |
| 2 | wall | 邊界(含柱) |
| 3 | display_table | retail zone(只標桌面投影) |
| 4 | display_shelf | zone + 防盜 |
| 5 | glass_door | 出入口事件(門框歸 wall) |

**✅ 雙權重決策(原 §7 #1)隨機器狗線一併關閉。** 只有一個視域,只有一套權重,
teacher 不需要 Depth Anything 3 / Metric3Dv2 的幾何 traversability 路線。

### 這個 taxonomy 與出貨契約的三處差異(需在啟動 Pipeline A 前處理)

出貨線 `configs/hydranet_retail_security_b03_cw_xl.yaml` 現行是
`[void, floor, wall, column, fixture, person]`,與上表差三處:

1. **`person` 不在新的 6 類裡。** `RETAIL.md` §1 與 `ARCHITECTURE.md` §3 有量測:把
   `person` 從分割拿掉,會在可通行地面上留下一個人形空洞。§4 Pipeline A 的「空景幀自動
   選取」在**訓練**上解決了這件事;**推論**要用同一道閘門——seg 只在無人幀重跑,否則站在
   那裡的人會污染快取的 zone 遮罩。
2. **`column` 消失。** 其歷史紀錄不佳(val 0.86–0.88,未見過的相機 0.00–0.51),拿掉可以
   接受,但 `RETAIL_DATA.md` 的 R4/R5/R7 是以它為例寫的,規則要一併重述。
3. **`fixture` 一拆為二**(display_table / display_shelf)。所有既有的 `fixture` 數字
   與新數字**不可比較**,舊 checkpoint 的分數不能拿來當基準。

---

## 3. 偵測類別與屬性(已定案)

- 6 類:person, iphone, ipad, macbook, boxed_stock, stack(堆疊整堆一框 + 可辨識單盒另框)
- iphone/ipad:標籤分開收,訓練先合併為 handheld_device 看混淆矩陣再決定拆分
- screen_on / in_hand:幾何後處理自動產(亮度統計 / 框中心包含判定),不進類別
- 假人:person + is_mannequin=true 當 hard negative
- 螢幕/海報/玻璃反射中的人與商品:ignore region

---

## 4. 資料管線:零人工標註(已定案,本次最重要決議)

**約束:絕對不做人工標註,可接受人工抽檢(accept/reject)。**

Teacher-Student 架構:大模型離線產標籤,HydraNet 三件套當 student。

| Pipeline | Teacher | 關鍵機制 |
|----------|---------|---------|
| A. Seg | Grounding DINO(prompt ensemble)+ SAM3 | 空景幀自動選取(反選無人時段);玻璃門用 IR 夜間幀補;stairs 用 Depth Anything 3 深度突變交叉驗證 |
| B. 偵測 | GDINO ensemble + CoTracker3 時序過濾 | **時序一致性過濾是核心**:特徵點追蹤連貫=真物體,漂散=幻覺;背景差分補 FN;student self-training 迴圈 1–2 輪解俯視角 recall |
| C. 屬性 | 本地 VLM 夜間批次 | structured output + track 級投票(一致率<60% → unknown);staff 判定餵 3 張制服參考照(唯一一次性人工輸入);投票標籤回貼全 track crop,樣本放大 10 倍 |
| D. ReID | tracklet 自監督 | 零標註;軌跡交會 ±1 秒的 crop 排除出 positive pair |
| E. 行為 | 速度規則預篩 + VLM 確認 | sit/crouch 自動撈 clip;fall 由同仁演出 50–100 段(資料生成,非標註);規則誤報當 hard negative 收錄 |

**信心分層(Pipeline B 核心):**
- Gold(ensemble 一致 + 時序通過 + score>0.6)→ 直接訓練
- Silver(0.35–0.6)→ loss 降權 0.5
- Gray(時序不一致/低分)→ **ignore region,絕不當負樣本**

**人工投入總帳:一人約 3 天,全部是抽檢 accept/reject:**
- Seg 48 張 overlay 全檢(0.5 天)
- 偵測 Gold/Silver 各抽 300 幀(1 天;Gold precision<95% 或 Silver<85% 才回調閾值)
- 屬性抽 200 track、行為抽 100 clip(各 0.5 天)
- fall 演出 1 小時 + 制服照 10 分鐘

**已聲明風險:**
1. Teacher 盲區會遺傳(畫面遠端 <20px 小人),靠 self-training 第二輪收斂,第一版接受遠端 recall 偏低
2. 屬性精度天花板 = VLM 判斷力(俯視 age ~80%),屬性只當規則引擎輔助訊號,不當客戶 KPI

---

## 5. 架構 Review 結論(需求對照)

| # | 項目 | 判定 | 行動 |
|---|------|------|------|
| 1 | 80 路單卡即時 | ✅ | seg head 必須低頻排程,否則爆 |
| 2 | Overhead 域偏移 | ⚠️ | 屬性/行為資料 100% 用自家 CCTV,不混公開資料集 |
| 3 | 軌跡碎裂 | ✅ | **先 overhead 重訓偵測,再評估 ReID 增量**,不同時上兩個變因;ByteTrack 參數按相機高度分組調 |
| 4 | Seg 雙視角 | ✅ | **已關閉**(2026-08-22):機器狗線移除,只剩 CCTV 俯視一個域,一套權重 |
| 5 | 速度規則 | 🔴 | **每相機做地板 homography 標定**(標 4 個地面點,~10 分鐘/相機),px/s → m/s,否則行為規則全不可維護 |
| 6 | VLM 佇列 | ⚠️ | 去重:同 track 同事件 cooldown 5 分鐘;boxed_stock 30 秒滑窗;每相機每小時上限 |
| 7 | 多工訓練干擾 | ⚠️ | 先 detection 收斂 → 凍 backbone 前三 stage → 再訓 seg head(或 uncertainty weighting) |
| 8 | P3 解析度 | ✅ | seg 用 stride 8 夠,不加 P2 |
| 9 | VLM 事件流 | ✅ | **事件 schema 現在定死**(zone/track_id/屬性快照/行為序列/關鍵幀 refs),experience library 檢索 key 依賴它 |

---

## 6. 執行優先序

1. Homography 標定(所有速度規則的地基)
2. Pipeline B 偵測自動標註啟動(量最大、lead time 最長)
3. Overhead person 重訓 → 評估 ReID 增量
4. ~~Seg 雙權重決策拍板~~(已關閉)→ 對齊 §2 的 6 類 taxonomy → 啟動 Pipeline A
5. VLM 抑制規則(一天工作量)
6. Pipeline C/E(依賴偵測穩定後的 crop 品質)

---

## 7. 待下個 session 確認的開放問題

1. ~~**Seg 雙權重(選項 A)是否拍板?**~~ → **已答(2026-08-22):機器狗線移除,問題關閉。**
   後續動作見 §2 的三處契約差異。
2. 資料範圍:目前手上是 48 路(先前規劃文件寫 80 路為部署目標)——teacher 管線先跑 48 路現有資料,確認即可
3. 下一步交付物三選一:
   - (a) Pipeline B 完整腳本規格(GDINO+SAM3+CoTracker3 串接、信心分層、COCO 輸出)
   - (b) Pipeline C VLM prompt 全文 + schema + 投票邏輯
   - (c) 夜間 GPU 窗口排程設計(teacher 推理 vs LoRA 訓練切分)

---

## 8. 執行現況與偏離(2026-08-22 稽核)

本節在 2026-08-22 加入。上面 §1–§7 是**計畫**;這一節是**實際做到哪裡**,以及 08-19/20
的工作偏離計畫多少。完整量測與證據在
[`journal/2026-08-22-drift-audit-and-course-correction.md`](journal/2026-08-22-drift-audit-and-course-correction.md)。

### 優先序執行狀態

| §6 優先序 | 狀態 | 說明 |
|---|---|---|
| 1. Homography 標定 | ❌ **未做** | 改走了 DA-V2 深度 → RANSAC 地面 → 人高尺度,23 支相機中 **14 支卡住**。計畫要的 4 點標定不受此限 |
| 2. Pipeline B 偵測自動標註 | ⚠️ **半做,核心機制缺席** | `tools/site30k/box_pass.py` 只有單一 GDINO,**無 CoTracker3 時序過濾、無 Gold/Silver/Gray 分層** |
| 3. Overhead person 重訓 | ❌ 未開始 | |
| 4. Seg 雙權重拍板 → Pipeline A | 🔴 **順序顛倒** | 決策(§7 #1)仍未拍板,但 Pipeline A 已跑完 **29,211 幀 / 10.4 GPU 小時** |
| 5. VLM 抑制規則 | ❌ 未開始 | |
| 6. Pipeline C/E | ❌ 未開始 | |

### 三個必須修正的偏離

1. **§4「絕對不做人工標註」被違反。** 08-20 產出 `tools/site30k/zones.json`——四個**手繪**
   多邊形,加上四輪目視審查。**已停止,且未套用**(`masks_prestamp/` 不存在,一個像素都沒改)。
2. **§1 taxonomy 不符。** 計畫的 seg 是 10 類;site30k_v1 是 11 個不同的 id,且
   `floor_carpet` / `ramp` / `stairs` / `movable_obstacle` / `glass_door` **五類在 29,211
   張 mask 裡一個像素都沒有**。偵測同樣不符:計畫 6 類,出貨詞彙 4 類,box_pass 寫 3 類。
3. **§7 #2 資料範圍。** 計畫是 48 路現有資料;campaign 跑了 9 支,而且**這 9 支全部早已在
   `retail_objects_batch02`/`batch03` 裡**——10.4 GPU 小時換到 0 支新相機。
   METHODOLOGY.md §0 對此已有結論:「More *cameras* fixes this; more frames from the same
   ones does not」。

### 計畫本身需要修正的三處

1. **§4 Pipeline A 的「玻璃門用 IR 夜間幀補」已被實測否證。** 08-20 建了本專案第一批夜間
   plate:Tao-Hsin-cam03 在夜間**兩面展示牆都拉下鐵捲門**,白天讀成 `wall` 的白櫃在夜間根本
   不可見;而店內夜間清空,日夜差分整間店都在變,無法隔離玻璃。這條路關閉。
2. **§1 / §5 的「80 路」是舊數字。** 記錄在案的生產上限是 `cc80fc3`(2026-08-19)設定的
   **96 路 @ 15 fps(1,440 frames/s)**。ReID 128 vs 256 的取捨建立在錯的數字上。
3. ~~**§2 保留 seg head 的理由是「機器狗需求」,但機器狗線已刪除。**~~
   → **已解決(2026-08-22):使用者裁定移除機器狗任務,只剩安防零售。** §2 已改寫為 6 類,
   雙權重問題關閉。新 taxonomy 與出貨契約仍有三處差異(`person` / `column` / `fixture`
   拆分),列在 §2 末,**必須在啟動 Pipeline A 前處理**。

### 可回收的部分

- **`instances_all_<split>.json` 保留了每個框到 score 0.10、且未做 NMS。**
  Gold/Silver/Gray 分層可以直接從現有檔案建,**不需要任何 GPU**。
- site30k_v1 的 floor / wall / display_table / shelf 可對應計畫 10 類中的 4 類。
- 夜間玻璃的否證結論、以及 Tao-Hsin 兩支相機「櫃檯逐日翻轉」(cam03 10/29 天、cam04 6/29 天
  整排櫃檯倒向 `wall`)都是 Pipeline A 的驗收條件,無論最後選哪種 taxonomy。

### 修正後的順序

**0.** ~~先答 §7 #1~~ → **已完成(2026-08-22):機器狗線移除,seg 定為 6 類。**
剩下的前置是 §2 末的三處契約差異(`person` 是否留在 seg、`column` 移除後 R4/R5/R7 重述、
`fixture` 拆分後舊分數不可比)。**Pipeline A 在這三項處理完之前不重啟。**
**1.** Homography 標定(§6 #1),48 路,每支 4 點約 10 分鐘。← **目前的下一步**
**2.** 用現有檔案建 Pipeline B 分層並抽檢 300 幀,再決定 CoTracker3 要不要跑。
**3.** Overhead person 重訓,不與 ReID 同時上。
**4.** 最後才回 Pipeline A,且每支相機取 5–10 張靜態幀(計畫值),不是 3,246 張。
