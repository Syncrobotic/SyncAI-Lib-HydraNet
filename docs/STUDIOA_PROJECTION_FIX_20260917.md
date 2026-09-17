# Kaohsiung-cam04 中間長櫃回投影修正

> 後續[相機投影基準核對](STUDIOA_PROJECTION_BASIS_20260917.md)發現校正與結構線不一致。
> 下列結果是固定舊校正下的局部擬合改善，不代表相機或 3D 幾何已正確。

2026-09-17。修正新模型輸入轉換器遺漏桌面幾何證據的問題，沿用既有物件建模器，
重新匯出 GLB 並投回來源 CCTV。高雄中間長櫃的朝向及檯面範圍改善；
**右端及櫃體輪廓仍有偏差，整個場景尚未驗收通過。**

[修正前後圖集](../runs/studioa_projection_fix_20260917_v2/index.html) ·
[高雄可旋轉 3D 候選](../runs/studioa_projection_fix_20260917_v2/Kaohsiung-cam04/scene.html) ·
[量化紀錄](reviews/studioa_projection_fix_20260917.json)。

## 修正內容與證據界線

- 保留並驗證來源影像 SHA 綁定的 `support_tops.npz`，以像素重疊重新關聯新實例。
  不沿用 `instances.npz` 的舊物件 ID。
- 將既有 commissioning 實例輪廓保存為無語意類別的 `support_bodies.npz`，
  避免 student 信心門檻造成的輪廓缺洞被誤當成櫃體邊界。來源身體遮罩與新實例
  IoU、桌面覆蓋率均須至少 0.5；沒有符合者仍使用候選實例。
- 恢復既有桌面／櫃體聯合細化，局部調整櫃子的方向、位置和長寬。
  不將 commissioning 舊房間方向視為真值，不修改相機校正、深度或模型權重。
  原本的碰撞、裝置支承及接受門檻未降低。
- 最終 GLB 加入設施線框回投影及逐桌面比較；評分使用匯出後頂點和節點變換，
  不採用擬合器自行回報的分數。輸出記錄於 `scene.surfaces.json.fixture_geometry`。
  此為擬合核對，`fixtures.gate_d_status` 仍為 `not_evaluated`。

## 結果

比較基準為 `runs/studioa_object_world_20260917_v2`，本次產物為
`runs/studioa_projection_fix_20260917_v2`。下列 IoU 均以同一份來源 AI 桌面輪廓凸包，
核對最終 GLB 在完整解析度的回投影；**不是獨立真值準確率或實測尺寸驗收。**

| 鏡頭／物件 | 修正前檯面 IoU | 修正後檯面 IoU | 結果 |
| --- | ---: | ---: | --- |
| Kaohsiung-cam04／table_4 | 43.14% | 86.27% | 局部細化接受，右端與櫃體仍有殘差 |
| Taichung-cam01／table_2 | 34.98% | 78.85% | 局部細化接受 |
| Taichung-cam01／table_10 | 26.25% | 77.73% | 局部細化接受 |
| Taichung-cam01／table_1、6、9 | 53.00%、30.24%、75.10% | 不變 | 聯合改善不足，保留原幾何 |
| Taichung-cam10／table_2 | 81.33% | 81.33% | 已有擬合保留 |
| Taichung-cam10／table_8 | 40.66% | 40.66% | 長櫃仍未修正 |

高雄長櫃 heading 由 −39.03° 調整為 −50.06°；長×寬×高約由
1.794×0.842×0.766 m 變為 2.358×0.690×0.766 m。公尺尺度沿用既有校正，
沒有獨立實測背書；櫃底受畫面截斷，高度依既有規則保留，不能宣稱接地位置已驗證。
低解析度擬合時的櫃體 IoU 由 0.6140 降為 0.5299，屬既有聯合門檻容許的
檯面改善／櫃體折衷，並非所有邊界同時改善。完整可見桌面遮罩含商品及缺洞，
修正後最終 GLB 與該遮罩 IoU 為 0.4277，不能與凸包 IoU 混用。

已逐張查看本輪三個鏡頭的最終 GLB 回投影及 3D 靜態總覽，另查看高雄長櫃
前後對照。高雄架體偏移及假柱、台中 cam01 中央櫃跨入走道及牆上假櫃、
台中 cam10 中後段長櫃與多餘小桌仍存在。Taichung-cam11、Tao-Hsin-cam03
缺少來源桌面觀測，本輪沒有重建；前一批五鏡頭拒收判斷仍有效。
尚未逐格驗收旋轉動畫，亦未完成全店座標整合。

## 驗證與重現

- 完整測試：**4,214 passed、12 skipped**（CUDA 不可用），121.64 秒。
- Ruff、三個變更模組的 ty 檢查通過。
- 新增回歸：錯誤來源影像拒絕、舊／新物件 ID 改變仍按像素關聯、
  最終頂點移動時必須反映投影退步，不能相信擬合器的成功宣告。
- 三份輸出 manifest 的產物 SHA 全數相符。相機／深度快取一致性最大偏差
  約 0.00000623 m，小於既有 0.002 m 門檻；這不代表現場絕對尺度準確。
- 程式與來源快照在產物 `inputs/`；完整測試 log、JUnit XML 隨產物保存。

```sh
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 .venv/bin/python \
  tools/commissioning/scene_mesh.py \
  Kaohsiung-cam04 Taichung-cam01 Taichung-cam10 \
  --model-run runs/studioa_overnight_20260917_v2/jobs/focus_seed43 \
  --out runs/studioa_projection_fix_20260917_v2 --device cpu --gif
```

重跑時使用新的輸出路徑，保留本輪凍結產物。後續先處理高雄右端／櫃體殘差，
再補其他錯誤物件的來源幾何觀測；不得因檯面分數改善而通過整個場景。
