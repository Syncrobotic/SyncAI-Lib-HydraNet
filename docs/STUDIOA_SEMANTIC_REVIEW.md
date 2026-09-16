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
改mask後僅更新檔案雜湊也無法通過。val/test不能被這個工具修改，其他held-out
fold不能使用此修正版，instance exporter亦拒絕將語意內部區域當完整實例。
原companion僅為歷史證據，修正版target必須由parent mask加review重播。

此版本不支援連鎖修正。需要新修正時，從明確的原始來源建立新的完整review；
評估標籤改版則需另外凍結用途與隔離規則，不能使用這個train-only工具繞過。
AI標註仍不是獨立真值。19類有train正例只表示沒有缺類，不表示資料多樣性足夠。

`studioa_train.py prepare --scene-comparison-updates N
--scene-validation-updates K --scene-class-weights-source PATH`另提供固定更新次數、
驗證次數及共享train類別權重的語意對照模式；保留初始權重並只推論source val。
N及K必須對齊完整epoch，且N可被K整除。此模式已通過設定測試，尚未執行新的
GPU對照；本輪因val玻璃分類品質問題暫不啟動。
