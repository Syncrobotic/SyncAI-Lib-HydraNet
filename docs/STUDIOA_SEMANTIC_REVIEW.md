# StudioA AI語意標籤修正

`studioa.semantic-review.v1`產生獨立的train修正版，保留原資料及歷史評估。
每張review綁定原圖、原mask SHA-256及判定理由。先把指定舊類別撤回ignore，
再將AI逐張確認的polygon內部標成正例；其他類別正例不被覆蓋。
polygon只代表已確認的部分語意區域，不代表物件全貌、instance框或3D平面。

```sh
.venv/bin/python tools/annotation/studioa_semantic_review.py \
  --source runs/studioa_gcs_training_extension_20260916_v1/semantic \
  --review docs/reviews/studioa_semantic_glass_review_20260916.json \
  --out runs/studioa_semantic_glass_review_20260916_v1
```

輸出目錄必須不存在。來源保留；輸出封存review、parent manifest/report、修改前
mask與producer。`check_supervision()`重播每筆修正，檢查像素統計及未修改影格；
改mask後僅更新檔案雜湊也無法通過。v1不能修改val/test，其他held-out
fold不能使用此修正版，instance exporter亦拒絕將語意內部區域當完整實例。
原companion僅為歷史證據，修正版target必須由parent mask加review重播。

此版本不支援連鎖修正。需要新修正時，從明確的原始來源建立新的完整review；
評估標籤改版必須使用下述明示的v2契約，不能用v1繞過train-only限制。
AI標註仍不是獨立真值。19類有train正例只表示沒有缺類，不表示資料多樣性足夠。

`studioa_train.py prepare --scene-comparison-updates N
--scene-validation-updates K --scene-class-weights-source PATH`另提供固定更新次數、
驗證次數及共享train類別權重的語意對照模式；保留初始權重並只推論source val。
N及K必須對齊完整epoch，且N可被K整除。此模式已通過設定測試，尚未執行新的
GPU對照。標籤補齊後須讓對照與候選共用同一修正版val，再凍結訓練預算。

## v2：明示的source train／val改版

`studioa.semantic-review.v2`允許同一fold的source train與val修訂；每筆item必須
標明實際`split`，不可修改test或excluded，亦不可移動既有分割。review必須包含：

- `revision_kind: source_train_val_correction`
- `evaluation_exposure: previously_inspected_not_blind`
- 非空的`evaluation_revision_reason`，說明改版目的。

除撤回整類舊標籤外，v2的`quarantine_regions`可指定polygon、原類別清單與理由，
只把該區內指定的誤標撤回ignore。例如把柱子上的wall誤標撤回，再加入保守的柱面
正例；人物與商品等未指定類別仍受保護。未確認邊緣維持ignore，不能宣稱完整遮罩。

從原始來源合併完整review，保留先前train修正與新val修正。來源、舊benchmark與
test逐檔保留；兩個比較模型都必須在相同新val上重新評估。新舊標籤的分數不可直接
比較，新val也不是未揭露盲測。後續只允許使用該review綁定的held-out fold。

```sh
.venv/bin/python tools/annotation/studioa_semantic_review.py \
  --source runs/studioa_gcs_training_extension_20260916_v1/semantic \
  --review docs/reviews/studioa_readiness_review_20260916.json \
  --out runs/studioa_semantic_ready_20260916_v1
```

本輪資料成果見[訓練前補齊紀錄](STUDIOA_TRAINING_READINESS.md)。
