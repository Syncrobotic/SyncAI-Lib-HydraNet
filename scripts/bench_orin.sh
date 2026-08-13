#!/usr/bin/env bash
# Build a TensorRT engine from hydranet.onnx and benchmark it on a Jetson Orin.
#
#   scp hydranet.onnx orin:~/         # from the workstation
#   ./bench_orin.sh hydranet.onnx     # on the Orin
#
# Reports pure GPU inference latency. That is NOT end-to-end FPS: it excludes image
# decode, pre-processing, and the host-side argmax and NMS. See the note at the end.
set -euo pipefail

ONNX="${1:-hydranet.onnx}"
[ -f "$ONNX" ] || { echo "no such file: $ONNX" >&2; exit 1; }
STEM="$(basename "${ONNX%.onnx}")"

command -v trtexec >/dev/null || export PATH=$PATH:/usr/src/tensorrt/bin
command -v trtexec >/dev/null || { echo "trtexec not found; try /usr/src/tensorrt/bin" >&2; exit 1; }

echo "=== platform ==="
head -1 /etc/nv_tegra_release 2>/dev/null || true
tr '\0' '\n' < /proc/device-tree/model 2>/dev/null | head -1 || true
dpkg -l 2>/dev/null | awk '/nvinfer-bin|libnvinfer[0-9]/ {print "  " $2, $3}' | head -3
echo
echo "power mode (benchmark on MAXN, or the numbers understate the board):"
nvpmodel -q 2>/dev/null | head -2 || echo "  (nvpmodel unavailable)"
echo

# Clock behaviour dominates short benchmarks. Left to the operator on purpose: locking
# clocks is a system-wide change and this script should not make one silently.
echo "For a stable measurement, first run:  sudo nvpmodel -m 0 && sudo jetson_clocks"
echo

for PREC in fp16 fp32; do
  ENGINE="${STEM}_${PREC}.engine"
  echo "=== building $PREC engine ==="
  FLAG=""; [ "$PREC" = "fp16" ] && FLAG="--fp16"
  # memPoolSize replaces the removed --workspace flag in TensorRT 10.
  trtexec --onnx="$ONNX" --saveEngine="$ENGINE" $FLAG \
          --memPoolSize=workspace:4096 --skipInference 2>&1 \
    | grep -E "Engine built|error|Error|failed" | head -5 || true

  echo "=== benchmarking $PREC ==="
  trtexec --loadEngine="$ENGINE" --iterations=200 --avgRuns=100 --warmUp=500 2>&1 \
    | grep -E "Throughput|Latency: min|mean =|median|GPU Compute Time" | head -6 || true
  echo
done

echo "=== engine sizes ==="
ls -lh "${STEM}"_*.engine 2>/dev/null | awk '{print "  " $5, $9}'

cat <<'NOTE'

What these numbers are, and are not
-----------------------------------
trtexec measures GPU inference only. End-to-end frame rate on the robot will be lower,
because it also pays for:

  * camera capture and colour conversion (do it with a CUDA kernel or VPI, not on the CPU)
  * pre-processing: (pixel/255 - mean) / std, ImageNet statistics, RGB order
  * argmax over the two segmentation maps
  * decoding and NMS for detection -- at 512x640 the class logits alone are
    80 x 6825 = 546,000 values per frame to move and reduce

That last item is why narrowing the detection head at export is worth doing: ~8 classes
instead of 80 cuts that traffic tenfold. Measure end-to-end separately before quoting an
FPS figure to anyone.
NOTE
