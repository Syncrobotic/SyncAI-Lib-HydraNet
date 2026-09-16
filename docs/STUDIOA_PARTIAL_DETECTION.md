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

## 第一批完成時的下一步

先擴充來源店 train 的實例與鏡頭，補足上述四個尚無正向實例的類別，再建立來源店
val 的實例覆核與明確評估規則。資料量與評估契約成立後才啟動 scene／detection
聯合訓練。桃新 test 保持封存；既有語意 pilot checkpoint 沒有被改寫。

## 2026-09-16：聯合 pilot 的資料與評估契約

第二批 `runs/studioa_partial_instances_20260916_v2/` 綁定相同語意 v3，
包含 11 張 train／31 個實例、6 張 val／31 個實例；兩邊均涵蓋十類。
新增決定見 [AI 實例覆核](reviews/studioa_joint_instance_decisions_20260916.json)。
排除價牌誤認手機、耳機誤認喇叭、電視廣告中的手機、合併多盒商品及污染遮罩。
只推進可確認的實例；語意 v3 和既有 test 標籤不變。

`partial_eval: reviewed_regions_v1` 顯式啟用部分標註驗證：score > 0.20、
同類 IoU ≥ 0.50、依信心排序一對一配對；報告確認實例的召回及各類支持數。
只有預測框向外取整的像素面積至少 95% 位於覆核空白區，才計為空白區誤報。
padding 不算空白；其他未配對預測列為未知，不算 precision 或 COCO mAP。
NMS 0.6、每圖上限 100 固定於訓練前；誤報另依覆核空白區百萬像素正規化，
不可與不同區域覆蓋的批次直接比較。沒有顯式啟用時，原本 COCO 評估防護仍有效。

`studioa_train.py prepare --instances … --initial-checkpoint …` 可凍結聯合 pilot。
從前次 scene best 初始化共享 backbone／neck 和 scene head，FCOS 隨機初始化；
30 epochs、batch 2、lr 1e-4、backbone ×0.1、scene loss ×1／detection ×0.5。
模型仍由來源 val scene mIoU 選擇，包含 warm-start 第 0 輪；偵測指標只作診斷。
聯合設定不宣告 test split，結束後僅回看來源 val。資料、程式、初始權重均綁 hash，
既有 checkpoint 不覆寫。此資料量只足以執行有限 pilot，不能證明部署可靠度。

## 聯合 pilot 結果：保留原模型，未通過升級

`runs/studioa_joint_pilot_20260916_v1/` 由 commit `97a246b` 凍結 645 個輸入檔案，
systemd 正常退出。原訂最多 30 輪，因連續 10 輪沒有超過語意基準，在第 10 輪
停止，共 280 次 optimizer 更新，其中 50 次來自 instance 批次。

| 檢查項目 | 結果 |
| --- | --- |
| 初始來源 val scene mIoU | 25.0633% |
| 第 10 輪來源 val scene mIoU | 24.5417% |
| 選定 epoch | 0：共享特徵與語意權重和初始模型逐 tensor 相同 |
| 第 10 輪部分偵測召回 | 3 / 31 = 9.68% |
| 命中類別 | person 3 / 9；其他九類皆 0 |
| 第 10 輪未配對、真偽未定的預測 | 349 |
| 覆核空白區誤報 | 0；只有 51,761 個網路輸入像素，不能推出整圖 precision |
| Tao-Hsin test | 本輪未評估 |

這些是 AI 覆核子集的指標，不是獨立真實準確率。第 0 輪的偵測 head 仍是隨機
初始化；`best.pt` 不能被宣稱為成功訓練的聯合模型。第 10 輪保留於 `last.pt`，
僅供診斷，沒有替換原模型。

逐圖檢查顯示多個重疊預測框與類別混淆；手機四個覆核實例全部漏檢。train 每類僅
2–6 個實例，且負樣本只有空地板，不能有效約束不同物品之間的混淆。共享特徵或
normalization 漂移可能影響語意表現，但本輪尚未隔離因果，不把它當成已證實原因。

[逐圖對照](../runs/studioa_joint_review_20260916_v1/joint_last_val_review.jpg)：左為 AI
覆核子集，右為第 10 輪每圖前 20 個預測；評估使用每圖最多 100 個，未因圖面省略。
[完整診斷](../runs/studioa_joint_review_20260916_v1/joint_last_diagnostics.json)、
[可追溯結果](reviews/studioa_joint_pilot_20260916.json)。

下一步先固定共享特徵及 normalization，單獨暖身偵測 head，並驗證場景輸出保持
一致；AI 補充 source train 的物件實例與明確的類別混淆負樣本。之後預先設定同時
要求偵測改善、語意不退步的接受條件，不能只靠 scene mIoU 選擇偵測模型。
目前不適合加長相同設定的訓練或推進部署。

驗證：212 項相關測試通過，型別相容性修正後 5 項針對測試通過；lint、format、
型別 ratchet 與提交 hook 通過。645 個凍結輸入及 6 個輸出 hash 核對，checkpoint
job 身分／有限 tensor／初始權重保留皆通過。另有低解析度 CPU 全流程檢查，
只驗證程式連接，沒有把其分數當作模型效果。

## 2026-09-16：固定場景的偵測頭暖身契約

GCS 擴充後 instances v3 有 27 張 train／129 個物件觀測、6 張 val／31 個觀測。
來源為 `studioa_gcs_training_extension_20260916_v1/semantic`；舊 val 與 test 分配
不變。同一物品可能跨日期重複出現，129 不代表獨立物品數。

`studioa_train.py prepare --detector-warmup --instances … --initial-checkpoint …`
從 scene pilot v2 的 best 初始化，只讓 `det_head.*` 參數可訓練。所有其他模組
固定 eval，包含 BatchNorm running statistics 與 dropout，並在每次儲存前驗證
凍結 tensor 完全相同。模型 train/eval 切換及 checkpoint 重載均保留此契約。
scene dataset 設為 `validation_only`，不建立它的 train loader；metadata 明記
train_size=0。全部 33 張來源 val 的 float32 scene logits 在訓練前和選定模型
載入後計算 SHA256，必須完全相同。續跑也重新對照原始 scene checkpoint。

預先固定：60 epochs 上限、batch 2、lr 2e-4、26 steps warmup、20 輪無改善停止、
bf16 training、deterministic、關閉 TF32／cuDNN benchmark、EMA 關閉。
best 選擇最大 reviewed-positive recall（沿用 score >0.20／IoU ≥0.50），同分
保留較早 epoch。第 0 輪仍參與選擇。完成後只有 recall 嚴格改善、覆核空白區
誤報不增加且場景完全一致，才記為通過此次暖身；這不是部署驗收。
未知區預測持續獨立報告，不視為正確或錯誤；本輪不讀取 test 進行推論。

紙箱仍只有 3 個 train 觀測；負樣本仍僅已覆核空地板，尚無物品間類別混淆的
負向監督。暖身結果只能回答「固定既有場景特徵，這批部分監督能否學出偵測」，
不能回答真實整店精度、3D 尺寸或顧客行為是否正確。

## 暖身結果：場景保留成功，偵測仍不足以使用

`runs/studioa_detector_warmup_20260916_v1/` 完成，程式版本 `562532d`；
第 40 輪因連續 20 輪無嚴格改善停止，共 520 次偵測 optimizer 更新。
選定第 20 輪（260 次更新），systemd 正常退出，test 未推論。

| 檢查項目 | 結果 |
| --- | --- |
| 已覆核正實例召回 | 初始 0/31 → 選定 7/31（22.58%） |
| 命中類別 | person 5/9、phone 1/4、poster 1/2；其他七類 0 |
| 覆核空白區誤報 | 0 → 0；僅覆蓋 51,761 個輸入像素 |
| 未配對、真偽未定預測 | 593；六張影像每張均達 100 框上限 |
| 凍結參數及 buffers | 194 個 tensors；每次儲存皆核對，best/last 與原模型完全相同 |
| 33 張來源 val 場景 logits | 訓練前後 SHA256 完全相同 |
| 場景 AI 標籤 mIoU | 40 輪均為 0.2506537344807518 |

通過的是「固定場景後，偵測頭得到有限學習」這項暖身契約。重疊框與類別混淆
仍明顯，593 個未知預測不能當作真陽性，也不能據此計算 precision。相比前次
聯合 pilot，本輪同時改了資料量與訓練隔離，不能把改善全歸因於其中一項。
本輪關閉 TF32 並固定 deterministic 設定，mIoU 與較早 pilot 的微小差異也不能
當成場景改善；直接證據是相同設定下完整 logits 和原場景 tensors 不變。

[六張逐圖對照](../runs/studioa_detector_warmup_20260916_v1/review/selected_val_review.jpg)
左側是 AI 覆核子集，右側只顯示前 20 框；評估仍用固定上限 100 框。
[逐框 JSON](../runs/studioa_detector_warmup_20260916_v1/review/selected_diagnostics.json)
可完整重算並精確重現所有偵測驗證指標。
[可追溯結果](reviews/studioa_detector_warmup_20260916.json) 記錄設定、檢查和 hashes。

757 個凍結輸入、8 個正式輸出 hash 核對通過；best/last 無非有限 tensor，
job 身分正確，原始 scene checkpoint 沒有改寫。主要測試批次 171 項通過，
資料與契約追加批次 20 項通過（兩批有重疊）；lint、型別與提交 hook 通過。
metadata 的 dirty 警示來自既有無關未追蹤檔，訓練程式使用已提交且逐檔凍結
的 snapshot；語意資料記為 train_size=0、val_size=33。

下一步優先補逐類混淆負樣本的資料契約：例如確認價牌不是手機，只否定 phone
通道，不能將實際物品整片畫成十類共同背景。紙箱三個 train 觀測全部來自同一
張影像與同一鏡頭，應由其他允許的 source-train 鏡頭補充。所有新增覆核由 AI
執行，val/test 分配及 score／NMS 門檻保持固定。尚不推進 Stage 2–4 的效果宣稱。
