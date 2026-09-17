# StudioA 物件式 3D 場景交付更正

> 2026-09-17 視覺覆核更正：五個鏡頭均發現幾何錯誤，本批拒收、尚未修正。
> 詳見 [逐鏡頭缺陷與原因](STUDIOA_OBJECT_WORLD_VISUAL_REVIEW_20260917.md)。


2026-09-17。使用者要求有桌、櫃、架、商品等物件模型的 3D 世界。
昨晚新增的可見表面三角網格流程沒有滿足此要求；先前「3D 世界完成」的說法撤回。
夜間七組訓練（bootstrap 加六組完整訓練）完成的事實與 checkpoint 保留。

## 本次修正

移除 `src/syncai_hydranet/cli/studioa_preview.py`、
`tools/commissioning/studioa_model_preview.py` 及只服務該實作的測試。
`studioa_overnight.py` 改為只交付模型，拒絕重啟舊表面預覽計畫；不再將訓練成功當成世界建置成功。
凍結實驗的歷史程式快照保持不變，原圖集入口已改為指向本次更正。

在既有 `tools/commissioning/scene_mesh.py` 增加 `--model-run`，沿用
`syncai_bev3d.scene_mesh.build_scene_regular` 的物件擬合、支撐、GLB/OBJ 匯出及回投影稽核。
`syncai_hydranet.cli.scene_evidence` 只轉換模型輸出為該建模器的輸入，沒有另寫渲染器。
離線互動頁面使用已安裝 trimesh 內附的檢視器，包含模型及 JavaScript，不需 CDN。

```bash
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 .venv/bin/python \
  tools/commissioning/scene_mesh.py \
  Kaohsiung-cam04 Taichung-cam01 Taichung-cam10 Taichung-cam11 Tao-Hsin-cam03 \
  --model-run runs/studioa_overnight_20260917_v2/jobs/focus_seed43 \
  --out runs/studioa_object_world_REVIEW --device cpu --gif
```

`--out` 必須是新目錄。checkpoint、訓練工作及模型設定須相符；類別順序必須符合
StudioA 19 類。原始 commissioning 資料不覆寫。

## 證據與使用界線

- 分類來自 `focus_seed43` 新模型 forward pass，沒有把教師分類當成 student 輸出。
- 既有 instance ID 僅分開相接的預測物件，不提供它們的類別；沒有預測支援的地方不添入物件。
- 房間地板範圍沿用既有校正的 walkable footprint，避免被家具遮住的地板變成孔洞；
  新模型的 floor mask 另行保留。相機、深度與實例分界的來源雜湊均有記錄。
- 門／玻璃以新模型遮罩進入原有平面擬合門檻；沒有預測或無法定位時不虛構門窗。
- 形狀沿用參數化資產，包括圓桌、展示桌、層架、櫃檯及部分商品；這不是逐件掃描的精細外觀模型。
- 天花板、紙箱、喇叭、海報、消防設備及人物尚未接入本次靜態資產放置。
  支援的類別也可能漏辨識；例如這批未放置成功的盒裝商品不能寫成已還原。
- 五個鏡頭分屬不同分店，保持各自座標，不拼成一間店。整店註冊、獨立尺寸與
  每件設施 Gate D 驗收尚未完成；模型同源遮罩的回投影分數不是獨立準確率。

## 本次產物與核對

[五視角場景入口](../runs/studioa_object_world_20260917_v2/index.html)。
每視角含 `scene.html`、`scene.glb`、`scene.obj`、`scene.png`、28 格 `scene.orbit.gif`、
原圖／新模型疊圖，以及 `scene.manifest.json`、`model_evidence.json`、幾何及商品定位報告。

五個 GLB 分別包含 12、16、15、12、11 個 mesh node（包括地板與相機標記）。
已核對 checkpoint 綁定、產物雜湊、PNG/GIF 可解碼、GLB 有限頂點及非空面。
地板修正版保存在 v2；v1 及 smoke 為過程紀錄，不作為最新入口。

完整 CPU 回歸：**4,209 通過、12 項 CUDA 條件跳過**。
後續補上的模型設定竄改拒絕及離線檢視器模型內容核對，連同相關建模／套件邊界測試
共 **43 通過**；Ruff 與修改模組的 ty 檢查通過。

v2 的幾何產物由 `inputs/` 中的程式及來源快照重建；稍後加入的 HTML 匯出及
模型設定綁定檢查不改變已生成幾何。互動頁面的匯出程式、GLB 與 HTML 雜湊另存
`viewer.manifest.json`。訓練品質及全店世界驗收仍需各自推進。

瀏覽器核對：伺服器的無頭 Firefox 無法建立 WebGL context，故不能宣稱已在該環境
驗證互動旋轉；已實際截圖確認會顯示場景 PNG 與 GLB 下載備援，而不留下空白頁。
離線 HTML 內嵌模型與原 GLB 的幾何內容由測試核對。
