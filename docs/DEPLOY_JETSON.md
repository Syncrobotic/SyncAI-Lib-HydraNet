# Jetson Orin 部署指南

## 流程總覽

```
PyTorch 訓練 (工作站 GPU)
   → export_onnx.py 匯出 ONNX
   → Jetson 上 trtexec 轉 TensorRT engine (FP16/INT8)
   → C++/Python runtime 推論 + CPU 後處理 (argmax / NMS)
```

## 1. 匯出 ONNX（工作站）

```bash
uv run hydranet-export-onnx \
    --config configs/hydranet_regnet800mf.yaml \
    --checkpoint runs/hydranet_regnet800mf/best.pt \
    --output hydranet.onnx
```

設計上 forward 圖只含 Conv/BN/ReLU/Resize/MaxPool/Exp/Mul，
無 NMS、無動態 shape、無自訂算子 —— TRT 一次轉換成功。

## 2. 轉 TensorRT engine（Jetson 上執行）

```bash
# FP16（Orin 建議起點，約 2 倍於 FP32 速度）
trtexec --onnx=hydranet.onnx --saveEngine=hydranet_fp16.engine --fp16

# 量測延遲
trtexec --loadEngine=hydranet_fp16.engine --iterations=200 --avgRuns=100
```

INT8 需要校正資料集（幾百張真實影像）：

```bash
trtexec --onnx=hydranet.onnx --saveEngine=hydranet_int8.engine \
        --int8 --calib=<校正快取>
```

建議：先 FP16 上線。INT8 對分割邊緣品質有可見影響，需重新驗證 mIoU。

## 3. 輸出節點與後處理

| 輸出 | 形狀 | 後處理 |
|---|---|---|
| `traversability` | [B, 3, H, W] | channel argmax → 每像素 0/1/2 |
| `terrain` | [B, 12, H, W] | channel argmax |
| `det_cls_p3..p7` | [B, 80, h, w] | sigmoid |
| `det_reg_p3..p7` | [B, 4, h, w] | 已是像素距離 (l,t,r,b) |
| `det_ctr_p3..p7` | [B, 1, h, w] | sigmoid，乘進 cls 分數 |

偵測解碼：對每個 level 的每個 grid 位置 (x, y)（中心 = (x+0.5)*stride）：

```
score = sigmoid(cls) * sigmoid(ctr)
box   = [cx - l, cy - t, cx + r, cy + b]
```

再做 score threshold + NMS。Python 參考實作見
`syncai_hydranet/models/heads/detection.py::FCOSHead.decode`，
C++ 移植約 100 行。

## 4. 效能預期（RegNetX-800MF + BiFPN96, 512x640, FP16）

| 平台 | 估計延遲 |
|---|---|
| Orin NX 16GB | ~12–18 ms |
| AGX Orin 64GB | ~5–8 ms |

不夠快時的調整順序（成本由低到高）：
1. 輸入降到 384×512（分割精度損失最小）
2. `model.neck.num_repeats: 1`
3. backbone 換 `regnet_x_400mf`
4. 偵測頭 `num_convs: 2`

## 5. 前處理對齊

Engine 輸入 = `(pixel/255 - mean) / std`，ImageNet mean/std，RGB 順序。
相機 BGR（OpenCV）記得轉 RGB，否則精度悄悄掉 5-10 個點。
在 Jetson 上建議用 CUDA kernel 或 VPI 做前處理，避免 CPU 瓶頸。
