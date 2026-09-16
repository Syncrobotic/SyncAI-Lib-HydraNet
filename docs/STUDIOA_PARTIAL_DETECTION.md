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

## 2026-09-16：逐類負樣本與第二支紙箱鏡頭

instances v4 保留 v3 的 33 張影像、正框、空白區 masks、companions 與 split
逐檔一致；新增 `0058-Kaohsiung-cam08` 中可分離的左側紙箱。相鄰兩箱合併的
proposal 被拒絕，只接受單箱遮罩。train 現為 28 張／130 個觀測；紙箱 4 個、
2 支鏡頭，仍十分稀少。val 保持 6 張／31 個正例，語意資料完全不改。

AI 重新查看 62 張來源 train 總覽及候選原圖，在 5 張既有影像選定 19 個保守
區域、47 項逐類負向決定：桌墊與遙控器不是手機、包裝印刷的平板不是實體平板、
桌上型螢幕不是筆電、海報上的 HomePod 不是實體喇叭、木椅不是紙箱。
這批資料重用已下載且分割合法的影像，不需要新增 GCS 下載。

`class_negative_rects` 必須由綁定影像／companion hash 的 AI 覆核提供，且只准
用於 train。匯出為逐類 0/1 masks；沒有明確否定的類別仍未知。dataset 對每個
通道共同做縮放、裁切、翻轉與 padding，再組成 `[C,H,W]`，padding 255 忽略。
FCOS 在對應 grid point 只加入被否定通道的 focal loss，不改 regression 或
centerness。相同類別的正框在所有金字塔尺度都有優先權；匯出時也拒絕矛盾覆核。
既有全類空白 mask 與部分驗證指標契約不變。無此欄位的舊資料維持原行為。

[覆核決定](reviews/studioa_class_negative_decisions_20260916.json) ·
[逐類負樣本總覽](../runs/studioa_confusers_20260916_v1/class_negative_contact.jpg) ·
[新增單箱遮罩](../runs/studioa_confusers_20260916_v1/vlm-unknown-0016.jpg)。

重訓沿用固定場景 warmup 設定與原始 scene 初始化，score／NMS／max_det 不調整。
本輪完整資料變更與前版 7/31 比較；不把資料增補與 loss 修改當成單因素因果實驗。
未知預測數減少不等於 precision 改善；僅通過隨機初始化基準也不等於超過前版。

### 第二輪結果：不升級模型

`runs/studioa_detector_warmup_20260916_v2/` 使用 commit `c274cb2`，第 25 輪
觸發 20 輪無改善停止（350 次更新），選定第 5 輪（70 次更新）。

| 同一來源 val、同一評估契約 | 前版 v1 | 本輪 v2 |
| --- | ---: | ---: |
| 覆核正例召回 | 7/31（22.58%） | 6/31（19.35%） |
| person / phone / poster 命中 | 5 / 1 / 1 | 5 / 0 / 1 |
| 覆核空白區誤報 | 0 | 0 |
| 未配對、真偽未定預測 | 593 | 594 |
| 場景輸出 | 與原模型一致 | 與原模型一致 |

因此 **保留 v1 作為較佳實驗基準，v2 不升級**。worker `warmup.accepted=true`
只代表超過第 0 輪隨機偵測頭，不代表超過 v1；明確決定另見
[promotion_decision.json](../runs/studioa_detector_warmup_20260916_v2/promotion_decision.json)。
兩版都未達部署條件；新程式與 AI 資料保留，沒有改寫任何舊 checkpoint。

五張已訓練負樣本影像中的 3,543 個 grid/class 對，score >0.20 的數量由
1,776 降至 15，說明選定模型在這些訓練區域的反應較低。這是訓練區診斷，
且兩版選定 epoch 不同，不能當作驗證 precision 改善或單因素因果證明。
val 六張影像仍全數達 100 框上限，尚未改善其他鏡頭的混淆。

固定門檻下的解碼前手機診斷進一步區分：v2 四個 val 手機中，兩個完全沒有
IoU ≥0.50 的原始候選框；另外兩個雖有定位候選，phone score 最高只有約
0.024–0.025，低於原定 0.20。這四個漏檢不能單靠提高 max_det 解決。
下一步先量化每類正負 loss 比例、限制逐類負項影響，再進行預先設定的比較，
並補充小物件的定位正例；不利用這組 val 調門檻掩蓋問題。

79 項測試、lint、型別、提交 hook 通過。GPU smoke 對 28 張 train 做 14 次
更新，2,457,821 個未知分類輸出梯度均為 0。319 個 smoke 輸入、779 個訓練
輸入及 8 個正式輸出 hashes 核對；best/last 有限、job 身分正確，194 個共享
tensors 與原模型相同。33 張 scene logits SHA256 完全一致，所有 test 未推論。
兩個 systemd 工作正常退出。儲存的逐框結果能完整重算官方偵測指標。

[本輪逐圖對照](../runs/studioa_detector_warmup_20260916_v2/review/selected_val_review.jpg) ·
[解碼前手機診斷](../runs/studioa_detector_warmup_20260916_v2/review/phone_candidate_probe.json) ·
[完整可追溯結果](reviews/studioa_class_negative_warmup_20260916.json)。

## 2026-09-16：逐類負項正規化的固定比較

先對 28 張 source train、不增強、float32、逐張影像做分類 loss 與 logit 梯度
分解。v1 模型的 phone 正向梯度 L1 合計約 0.0271，新增逐類負項約 9.631；
其中一張沒有 phone 正例的桌面影像，610 個 phone 負向點貢獻約 7.168。
這說明新增負項會受到覆核區面積與每張正例數影響，但不是原先 batch 2 增強
訓練的梯度重播，也不是模型參數梯度；不能單憑比例就判定退步原因。
130 個 train 正框均有可分配的特徵點；「無原始框達 IoU 0.50」和「無訓練
指派點」是不同問題，仍需另外衡量小物件框回歸品質。

只比較一個變因 `class_negative_normalization`：`sum` 保留原行為；
`positive_budget` 將每個 batch/class 新增負項權重總和限制在
`max(該類正向指派點數, 1)`，每個負點權重最多 1。已有正項、全類空白負項、
regression、centerness 與未知通道均不改；缺少該類正例的 batch 仍可提供
最多一個等效負向點。這限制的是監督權重總量，不是梯度範數。

兩組從同一 scene checkpoint、同 seed、同 instances v4 開始；所有訓練與
驗證條件沿用 v2。固定 score >0.20／IoU ≥0.50／NMS 0.6／每圖 100 框，
不得看結果後調門檻。只有候選超過 v1 的 7/31、人物至少 5、手機至少 1、
海報至少 1、覆核空白誤報不增加且 scene logits 完全一致，才更新實驗基準。
本次只有這兩組，不依此 val 分數追加參數搜尋。

[執行前比較計畫與診斷](reviews/studioa_negative_balance_plan_20260916.json)。

### 比較結果：修正新增負項權重，但仍不升級

程式 commit `633cfbd`。`studioa_detector_balance_20260916_control` 與
`studioa_detector_balance_20260916_budget` 都在第 25 輪停止，350 次更新，
選定第 5 輪／70 次更新。除設定檔中的正規化方式與必要輸出路徑外，兩組
778 個凍結輸入完全相同。對照組 best、last 的每個模型 tensor 及驗證指標
均精確重現之前的 v2。

| 同一來源 val | 原實驗基準 v1 | sum 對照 | positive_budget |
| --- | ---: | ---: | ---: |
| 命中／31 | 7 | 6 | 7 |
| 人物 | 5 | 5 | 5 |
| 手機 | 1 | 0 | 0 |
| 海報 | 1 | 1 | 1 |
| 紙箱 | 0 | 0 | 1 |
| 其他六類 | 0 | 0 | 0 |
| 覆核空白誤報 | 0 | 0 | 0 |
| 未配對、真偽未定預測 | 593 | 594 | 593 |

限制負項權重在本 seed/fold 多命中一個紙箱，但沒有超過基準的總命中數，
手機也未恢復，**未通過事先固定的升級條件**。保留 v1 作為實驗基準。
所有結果僅是 AI 覆核子集上的選模診斷，六張影像仍均達 100 框上限。

手機的兩個定位候選最高 score 約 0.065／0.076，相較 sum 的約 0.024／0.025
回升，但仍低於固定 0.20。另兩個手機仍無 IoU ≥0.50 的原始候選框。
不繼續對這份 val 搜尋負項係數或調整門檻；下一步應補跨鏡頭小物件正例及
定位訓練，尤其手機／平板，而不是只增加負向監督。

173 項測試、lint、型別、提交 hook 通過。兩組各 779 個輸入與 8 個正式輸出
hashes 核對通過，best/last 有限且綁定正確 job；所有非偵測 tensors 與初始
模型相同。兩組 33 張 scene float32 logits SHA256 都與原基準完全一致。
逐框紀錄可重算候選模型的所有正式偵測指標。診斷與依序比較的 systemd 工作
均正常退出；本輪未推論 test，也未部署或改寫舊模型。

[逐圖對照](../runs/studioa_detector_balance_20260916_budget/review/selected_val_review.jpg) ·
[升級判定](../runs/studioa_negative_balance_20260916_v1/promotion_decision.json) ·
[完整比較結果](reviews/studioa_negative_balance_results_20260916.json)。

## 2026-09-16：補跨鏡頭手機／平板正例

從已下載且允許訓練的影像中，AI 逐張檢查原圖、局部放大與候選遮罩，覆核
8 張影像、70 個候選，接受 32 個裝置框（手機／平板各 16）。印刷裝置、
工作墊、類別不清、遮罩混入其他物件或重複候選仍為未知；沒有自動轉成負例。
框沿用被接受遮罩的可見邊界，手持遮擋物件不宣稱完整物件範圍。

instances v5 訓練影像由 28 增至 35，正例觀測由 130 增至 162：手機
17 → 33，鏡頭 4 → 5；平板 15 → 31，鏡頭 5 → 7。其餘八類數量不變。
同一實體可能跨日期重複出現，不能解讀成 162 件不同物品。所有既有正例、
空白負區與逐類負區保留，六張 val 的影像、標註與空白遮罩逐位元相同。
162 個框在固定 512×896、不增強、各框獨立指派時均有 FCOS 特徵點；
這不保證隨機縮放裁切後每次都保留指派點。

只執行一組補資料候選，沿用 `positive_budget`、原 scene checkpoint、seed42
及既定訓練與解碼設定；使用已完成的同設定 v4 作對照，執行前核對凍結
程式、初始化權重、正規化後設定與 val。每輪更新次數因資料增加由 14 變成
17（既有 drop_last=True），因此比較的是補資料後的完整訓練流程，不是等更新次數的純標註因果實驗。
升級仍須超過原 v1 的 7/31、人物至少5、手機至少1、海報至少1、覆核空白
誤報不增加、33張 scene logits 完全一致。只有一個候選，不依結果追加搜尋。

[接受框總覽](../runs/studioa_positive_expansion_20260916_v1/accepted_positive_review.jpg) ·
[AI 決策](reviews/studioa_positive_decisions_20260916.json) ·
[執行前計畫與資料核對](reviews/studioa_positive_expansion_plan_20260916.json)。

### 補正例比較結果：資料擴充完成，模型仍不升級

資料與執行前計畫 commit `a8bcdd1`。候選
`studioa_detector_positive_20260916_v1` 完成 25 輪／425 次更新，選定第5輪／
85次更新；相同設定的 v4 對照為25輪／350次、選定第5輪／70次。
事先計畫估算每輪18次更新時漏掉既有 `drop_last=True`，實際為17次；
這是計數勘誤，沒有修改已凍結設定或補跑候選。每輪洗牌後最後一張不組成
完整 batch 而略過；比較不具等運算量的因果解讀。

| 同一來源 val | 保留基準 v1 | v4 positive_budget | 補正例 v5 |
| --- | ---: | ---: | ---: |
| 命中／31 | 7 | 7 | 7 |
| 人物／9 | 5 | 5 | 6 |
| 手機／4 | 1 | 0 | 0 |
| 平板／1 | 0 | 0 | 0 |
| 海報／2 | 1 | 1 | 1 |
| 紙箱／2 | 0 | 1 | 0 |
| 其他五類 | 0 | 0 | 0 |
| 覆核空白誤報 | 0 | 0 | 0 |
| 未配對、真偽未定預測 | 593 | 593 | 593 |

**總命中未增加且手機未恢復，未通過升級條件，繼續保留原 v1。**
所有六張影像仍達100框上限；大量未配對預測真偽未知，不能把零空白區誤報
解讀成高 precision。這些是同一小型 AI 子集上的選模結果，不是獨立準確率。

固定解碼條件下，四個 val 手機中兩個仍沒有 IoU ≥0.50 的原始候選框；
一個雖有合格候選、最高 phone score 約0.279，最後被 NMS／100框上限移除；
另一個有定位候選，但最高 phone score 約0.167，仍未達既定0.20。
補資料後部分候選分數回升，最終召回仍未改善，不據此調整門檻。
下一步應在 train-only 診斷上比較保留物件邊界的局部裁切訓練，量化手機／
平板被裁切及 FCOS 指派狀況，並使用相同更新次數作對照；新做法仍需保護
既有場景分支，不能宣稱補標已解決小物件定位。

原基準、v4對照與新候選的凍結輸入／正式輸出 hashes、best/last checkpoint
綁定與有限值重新核對通過；新候選為807個輸入、8個正式輸出。
194個非偵測 tensors 與原 scene 模型相同，33張 float32 scene logits SHA256
也相同。逐框紀錄完整重算所有正式偵測指標；20項相關測試通過。
訓練與診斷 systemd 工作均正常退出，沒有 test 推論、部署、3D幾何或Stage2–4修改。

[候選逐圖對照](../runs/studioa_detector_positive_20260916_v1/review/selected_val_review.jpg) ·
[手機漏檢診斷](../runs/studioa_detector_positive_20260916_v1/review/phone_candidate_probe.json) ·
[完整資料、驗證與升級判定](reviews/studioa_positive_expansion_results_20260916.json)。

## 2026-09-16：固定更新次數的局部裁切比較

在 instances v5 保留全部資料，以 `data.datasets[].small_object_crop` 啟用僅供
StudioA 部分偵測訓練的增強：有手機／平板正框的影像，以50%機率均勻選一框，
使用原 letterbox 比例的2倍（乘原0.9–1.1縮放），限定裁切位置使選中框保持
完整且離邊至少2px；放不下則回到原整圖增強。其他物件可能被裁掉或截斷。
影像、所有正框、空白負區及逐類負區共用幾何；padding保持255、未知保持0。

35張 train、每張10次固定種子的純幾何診斷：手機保留觀測321→263，但可分配
FCOS點3595→4919；平板310→289、點8830→9942。這是放大物件與減少周邊觀測
的取捨，不是準確率提升。手機仍有2次保留框無可指派點，平板為0。
六張 val 的輸入影像及所有 target tensors，在開關設定下逐位元一致。

兩組均重新從原scene checkpoint開始，seed42、positive_budget、25輪、
每輪17次更新，共425次；關閉early stopping，25輪中按既定val指標選best，
平手保留較早epoch，另保存last。除裁切開關外設定完全相同。原v5跑的是60輪
排程加early stopping，不能作為這輪等預算對照。推論仍是512×896整圖、
score>0.20、IoU≥0.50、NMS0.6、max_det100，不用val搜尋縮放倍率或門檻。
候選除原升級條件外，還須嚴格超過本輪對照組，才更新實驗基準。

201項相關測試、lint及型別檢查通過。新增測試涵蓋四角與中央物件完整保留、
裁切後框與遮罩對齊、負區幾何、padding為未知、無合適類別／過大物件的回退，
以及拒絕把此設定套用到其他資料集。

[執行前計畫](reviews/studioa_focused_crop_plan_20260916.json) ·
[訓練裁切圖例](../runs/studioa_focused_crop_20260916_v1/training_views.jpg)。

### 局部裁切結果：拒絕升級

程式與計畫 commit `0cafb5f`。兩組各完成25輪／425次更新，服務正常退出。
選定對照第11輪／187次更新，裁切候選第10輪／170次更新；這是同樣25次
epoch驗證中的best選擇，不是相同更新位置的checkpoint，故也記錄第25輪結果。

| 固定 source val | 保留基準 v1 | 本輪整圖對照 | 本輪裁切候選 |
| --- | ---: | ---: | ---: |
| best命中／31 | 7 | 7 | 5 |
| 人物／9 | 5 | 6 | 4 |
| 手機／4 | 1 | 0 | 0 |
| 平板／1 | 0 | 0 | 0 |
| 海報／2 | 1 | 1 | 1 |
| 其他六類 | 0 | 0 | 0 |
| best覆核空白誤報 | 0 | 0 | 0 |
| best未配對、真偽未定預測 | 593 | 593 | 595 |
| 第25輪命中／31 | 不適用 | 6 | 4 |

**候選低於同預算對照，且沒有恢復手機，不採用裁切設定、不替換原v1模型。**
設定預設仍關閉，沒有自動部署或改寫舊checkpoint。此結果限於本seed、資料與
訓練方式，不能推論所有裁切都無效。訓練幾何診斷的指派點是逐框獨立估算，
不是整批實際loss梯度或偵測準確率。

候選四個val手機：兩個仍無IoU≥0.50的原始框；另外兩個有定位框，最高手機
score約0.275與0.227，超過0.20，但對應點的其他類別勝出，因而不進入手機NMS。
這是解碼前的分類競爭，不能以降低score門檻或增加max_det解決。
下一步先在train正例區量化正確類別與競爭類別的分數／梯度，再決定是否在
已覆核且類別互斥的正例上補足分類約束；另查兩個缺少定位框的案例。
不得把整張未覆核區域直接改成負例，或依這六張val搜尋更多裁切參數。

兩組各807個凍結輸入、8個正式輸出hashes全部核對；除了設定檔，806個輸入
逐位元相同，設定也只差裁切開關。best/last有限、job綁定正確，194個共享
模型tensors與原scene checkpoint相同；33張scene logits SHA256維持原值。
兩組儲存的逐框預測均能完整重算官方指標。程式相關201項測試及工具邊界
11項測試、lint、型別、提交hook通過；所有test鏡頭未推論。

[裁切候選逐圖對照](../runs/studioa_detector_crop_20260916_focused/review/selected_val_review.jpg) ·
[等預算比較與完整核對](reviews/studioa_focused_crop_results_20260916.json)。

## 2026-09-16：已指派正例的分類競爭

先對35張source train做整圖、不增強、float32診斷；沒有利用val選新參數。
上輪整圖對照的手機344個FCOS指派點中，正確類別只在7點勝出；其餘337點
的勝出錯誤類別梯度全部為0。平板0／885、筆電0／442正確勝出，其勝出錯誤
類別梯度亦全部為0。錯誤主要是海報及人物。這是逐圖logit梯度，不是歷史
batch梯度、參數梯度或真實準確率；但與程式契約一致：原loss只拉高指派類別，
其餘通道未知，解碼卻只讓最高分類分數的類別取得該候選框。

新增可選 `loss.positive_classification=assigned_object`，預設仍為
`positive_only`。僅在FCOS已指派給覆核正物件的點，將其他類別納入one-hot
focal分類監督，使分類與該點回歸的物件身分一致；這不是宣稱其他物件在畫面中
不存在。若另一個已覆核類別的框也涵蓋此點，則保護該類別通道，跨所有金字塔
層級，不把重疊物件互判為負例。未指派位置、未知區、padding保持原監督；
原本逐類負項budget先計算，再合併新mask，不改回歸與centerness loss公式。
部分標註仍可能漏掉重疊物件，因此此假設與影響必須由實驗檢查，不能當作
新增的整圖absence標註。

只跑一組整圖對照及一組候選；兩組同instances v5、同初始化、seed42、
positive_budget、25輪／425次更新、無early stopping、關閉局部裁切。
固定原score>0.20／IoU≥0.50／NMS0.6／max_det100，選best且另報last。
候選須通過原升級條件且嚴格超過等預算對照。另要求對照精確重現上輪整圖
對照的best/last模型tensors與驗證指標，驗證預設行為未改。

178項測試、lint、型別檢查通過。新增測試驗證只抑制正指派點的競爭通道、
正確類別及box/centerness梯度不變、未知／padding／其他層級不增梯度、
重疊覆核物件與資料集未知類別受到保護，以及bfloat16／完整標註相容性。

[執行前計畫及訓練診斷](reviews/studioa_positive_competition_plan_20260916.json)。
