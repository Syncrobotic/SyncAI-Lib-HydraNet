# StudioA 小商品修正與部分實例監督

已完成來源店資料診斷、一次 AI 小商品修正、首批實例與負樣本區域標註，以及
FCOS 部分監督接線。GPU 三步檢查通過；尚不是新一輪完整偵測訓練。
[六張標註預覽](../runs/studioa_small_objects_20260916_v1/index.html)及
[來源、逐類資料量與檢查紀錄](reviews/studioa_small_objects_20260916.json)。

## 來源店的實際缺口

固定沿用 Tao-Hsin 整店留出。原 train 的手機有效監督只來自 6 張影像／3 支鏡頭，
紙箱只有 3 張／2 支鏡頭，門只有 4 張／1 支鏡頭。類別有非零像素不代表能涵蓋
外觀、尺寸、視角與遮擋變化。稀少商品還受到未知遮罩排除，不能直接將全圖當作
完整實例標註。

助理檢查來源原圖與隔離遮罩後，在 train 的 `0031-Taichung-cam08` 做以下修正：

- raw group 5、8：可見手機機身／螢幕，從未決改為 phone，新增 2 個手機遮罩。
- raw group 14：可見實體數字鍵與導覽環，是遙控器而非手機；詞彙沒有遙控器，
  改回 unknown，不創造不適用的類別。
- 保留既有 group 10、11 的未決判讀，避免重算時復活舊錯標。

新版 AI 標註為 `runs/studioa_ai_relabel_20260916_v5/`，語意匯出為
`runs/studioa_partial_supervision_20260916_v3/`。手機 train 有效像素由 40,021
變為 44,982；全批有效像素 177,323,048，忽略像素 66,132,952。
只有上述一張的語意 PNG 改變，其餘 120 張逐檔一致，包括 33 張 val、26 張桃新
test 及 16 張排除影像；所有分割不變。沒有新增影片、沒有重新測試模型精度。

這是 AI 目視判讀，不是獨立真值。原始 raw 回覆、父版本及修正理由均保留並綁定 hash。
[語意修正決定](reviews/studioa_small_object_decisions_20260916.json)。

## 部分實例資料的規則

「已標出這件商品」不能推出「其餘位置沒有商品」。一般 FCOS／COCO 訓練將未匹配
位置當成背景，直接使用部分 AI 標註會懲罰漏標商品的正確預測。

新 `studioa.partial-instances.v1` 明確區分：

1. 經 AI 查看原圖與遮罩確認的單一實例：由可見遮罩取 xyxy 外框，FCOS 只監督
   被分配的正類別通道，其他類別通道保持未知；回歸與 centerness 只訓練匹配點。
2. 經 AI 查看原圖確認沒有這十類物件的局部空白區：允許負樣本分類損失。
3. 其餘影像、漏標區、padding：不提供背景損失。即使正框超出某 FPN 層的尺度
   範圍，也不能在那一層把框內位置變成負樣本。

`det_negative_mask` 用 1 表示明確負樣本、0 表示未知、255 表示 padding；它與影像／
外框一起縮放、翻轉、裁切。缺少此欄位的既有完整標註資料沿用原 FCOS 損失。
資料集工廠 type 為 `studioa_instances`，只監督 `detection`，禁止分割角色及類別重映射。
head 必須使用精確的十類順序：laptop、phone、tablet、boxed_stock、cardboard_box、
speaker、poster、fire_equipment、chair、person。這與語意 head 的 19 類 ID 是不同契約。

首批 `runs/studioa_partial_instances_20260916_v1/` 是 **6 張 train、5 支鏡頭、
15 個實例、6 塊空白區（198,600 原圖像素）**，由助理逐圖確認，沒有要求人工標註。
原圖、companion、review、實例外框與負樣本 PNG 均凍結。

| 類別 | 本批確認實例 |
| --- | ---: |
| 筆電 | 2 |
| 手機 | 4 |
| 平板 | 3 |
| 紙箱 | 3 |
| 喇叭 | 1 |
| 消防設備 | 2 |
| 盒裝商品、海報、椅子、人物 | 各 0 |

這是用來驗證資料契約的首批子集，不是全店實例數。原語意批次仍有其他類別標籤，
但未經單一實例檢查的遮罩不會自動升格為偵測真值。
[實例／空白區的 AI 決定](reviews/studioa_instance_decisions_20260916.json)。

## GPU 與測試結果

`runs/studioa_partial_detection_smoke_20260916_v1/` 凍結 188 個資料／程式檔案，
commit `cab2cb147b2bd4d9c3d1bbaff4d466ec4e93276c`。以持久化 systemd user service
執行，五分鐘上限；實際正常退出，`Result=success`、exit 0。

ResNet18 + FPN + 10 類 FCOS 隨機初始化，512×896、batch 2、bf16 AMP、seed 42，
6 張 train 共做 3 次 SGD 更新。無下載、未讀取 val／test、未儲存 checkpoint。

- 三批 loss 分別 2.788436、2.714539、2.780726；不同影像批次，不能解讀為學習曲線。
- 所有參數梯度有限，分類器更新範數 0.00370684。
- 檢查 551,580 個未知分類輸出，非零梯度數為 **0**。
- 訓練前後凍結檔案 hash 一致，資料分割與 review 身分核對通過。

260 項相關測試通過；補上精確類別順序防護後，187 項受影響測試再次通過。
涵蓋未知位置／其他類別通道／padding 零梯度、明確負樣本梯度、空批次、混合精度、
跨 FPN 層的正框保護、幾何增強、來源 hash、分割與舊完整標註相容性。lint、格式、
型別及提交檢查通過。

不完整標註不能產生可信的標準 COCO mAP。設定檢查與 evaluator 均會拒絕將這種資料
當作完整偵測驗證集，避免未標物件被記成假陽性或缺少標註造成虛高分數。

## 重現

```bash
.venv/bin/python tools/annotation/studioa_autolabel.py relabel-review \
  --source runs/studioa_ai_relabel_20260916_v4 \
  --decisions docs/reviews/studioa_small_object_decisions_20260916.json \
  --out /tmp/studioa-ai-small-objects
.venv/bin/python tools/annotation/studioa_supervision.py export \
  --source /tmp/studioa-ai-small-objects --out /tmp/studioa-semantic-small-objects
```

review 文件綁定正式語意來源 manifest hash；manifest 包含來源路徑，因此上面的
另一位置匯出不能直接搭配既有 instance review。正式實例匯出使用已驗證的 v3 來源：

```bash
.venv/bin/python tools/annotation/studioa_instances.py export \
  --source runs/studioa_partial_supervision_20260916_v3 \
  --reviews docs/reviews/studioa_instance_decisions_20260916.json \
  --out /tmp/studioa-instances
.venv/bin/python tools/annotation/studioa_instances.py prepare \
  --data /tmp/studioa-instances --out /tmp/studioa-instance-smoke
.venv/bin/python tools/annotation/studioa_instances.py run \
  --out /tmp/studioa-instance-smoke
```

所有輸出目錄須尚不存在；run 切換到凍結程式並要求 CUDA。長期 GPU 工作應依本輪
方式使用 systemd，而非依賴 SSH 前景程序。

## 下一個必要步驟

先擴充來源店 train 的實例與鏡頭，補足上述四個尚無正向實例的類別，再建立來源店
val 的實例覆核與明確評估規則。資料量與評估契約成立後才啟動 scene／detection
聯合訓練。桃新 test 保持封存；既有語意 pilot checkpoint 沒有被改寫。
