# 工具用途與歷史副本查核（2026-09-16）

查核基準為主專案 `600fbae`，範圍包含程式引用、CLI 入口、封裝、測試，以及
本機 `/home/paul/hydranet_fti_stage0`。結論：**確有舊副本與過時文件，但不能
把所有 BEV 工具當成已廢棄的 3D 前身。**

| 項目 | 實際用途／狀態 | 處理 |
| --- | --- | --- |
| `/home/paul/hydranet_fti_stage0` | 2026-09-10 FTI 工廠 Stage0 隔離試驗副本，沒有 `.git`；並非目前 StudioA 訓練入口 | 保留試驗證據，新工作以主專案版本及獨立輸出目錄執行 |
| 該副本的 `datasets/studioa_static` | 符號連結到該副本的 `datasets/fti_static`，是舊工具硬編碼路徑的相容措施 | 不得因目錄名稱就當作 StudioA 資料使用 |
| `syncai_bev3d/bev.py` | 地面投影、格網與可通行區計算，仍被 CLI、3D 診斷與測試引用 | 保留為幾何運算；2D 地面表示不等於完整 3D 重建 |
| `hydranet-scene` / `cli/scene.py` | 給缺少 commissioning 資料的影像使用假定視角、高度與俯角，輸出透視診斷及 JSON | 可用的選用診斷入口；其絕對尺度是條件假設，不是 StudioA 驗收輸出 |
| `syncai_bev3d/bev3d.py` | 將格網與逐幀偵測結果畫成透視示意圖，不需 `camera.json` | 保留診斷用途，不等同可匯出的場景 mesh |
| `tools/commissioning/scene3d.py` | 用幾何快取與既有五類結構畫診斷圖 | 舊分類範圍的輔助工具，不能用來驗收目前19類語義／10類偵測模型 |
| `tools/commissioning/scene_mesh.py` | 用相機設定及 commissioning 證據產生 GLB/OBJ 和獨立覆核產物 | 目前場景幾何覆核入口；形狀與公尺尺度仍需分別驗證 |
| `tools/annotation/studioa_train.py` | 凍結資料與程式，執行目前 StudioA 語義／部分偵測實驗 | 目前模型實驗入口；結果以 run manifest/report 和升級判定為準 |

## 舊副本的具體差異

`runs/REVIEW.md` 自述它是2026-09-10的工作樹快照，含當時尚未提交的幾何工具。
以主專案 `600fbae` 逐檔 SHA256 比較：

| 子目錄 | 相同 | 不同 | 僅舊副本有 |
| --- | ---: | ---: | ---: |
| src | 117 | 15 | 0 |
| tools | 39 | 11 | 0 |
| scripts | 48 | 0 | 0 |
| configs | 38 | 0 | 0 |

例如舊 `geometry_review.py` 缺少目前的 plate SHA256 檢查；舊 `scene_mesh.py`
CLI 缺少目前獨立 `--out` 覆核目錄與來源／輸出稽核能力。因此不能把該副本當成
目前程式版本重新訓練或驗收。但它保存5個 FTI 相機的試驗產物，不應直接刪除。
兩個 checkpoint 目錄也連回主專案的歷史 runs，並非完全獨立的環境。

主專案的 `src/configs/tools/scripts/deploy/pyproject.toml` 沒有引用該外部路徑。
本次 `.venv` 實際匯入兩個 package 的路徑均為主專案 `src/`；可讀的程序 cwd、
路徑參數與 user systemd 設定未發現使用該副本的入口。這不是所有使用者或所有
外部排程的完整盤點，不能只憑沒有找到程序就判定可刪資料。

需要重做 FTI 場景時，可用主專案 `scene_mesh.py --root <FTI資料根目錄>
--out <新的覆核目錄>` 指定輸入，先檢查既有幾何快取與來源綁定。這是遷移方向，
本次未執行重建、改寫 FTI 產物或刪除舊目錄。

## 文件修正與可驗證邊界

工具索引原有「只有一個工具會訓練模型」、「21個工具」及「8/48個相機」等
固定敘述已與現況不符。本次改為依功能與實際輸入契約描述，移除這些過時數字。
README 同時區分現有零售模型設定與目前 StudioA 的19類 scene／10類 detection
實驗，避免把舊展示圖當成最新模型輸出。

`test_renderer_generations.py` 驗證無 commissioning 輸入仍可畫透視診斷，而
mesh 需要相機產物；`test_package_boundaries.py` 驗證正式 serving 模組不匯入
離線 commissioning package。保留功能的理由是它們的輸入與責任不同，而非
只因檔案存在或測試通過。

[逐檔查核證據](reviews/studioa_tool_audit_20260916.json) ·
[目前 StudioA 訓練結果](STUDIOA_PARTIAL_DETECTION.md) ·
[Stage0–4 契約](STUDIOA_STAGE0_4.md)

## 2026-09-17 場景出圖更正

物件式世界沿用 `scene_mesh.py`，新增 `--model-run` 接入已完成的新模型。
昨晚新增的 `studioa_model_preview.py` 與 `cli/studioa_preview.py` 已移除；
夜間訓練工具只負責 checkpoint，不再宣稱產出 3D 世界。
詳見 [本次更正](STUDIOA_OBJECT_WORLD_20260917.md)。
