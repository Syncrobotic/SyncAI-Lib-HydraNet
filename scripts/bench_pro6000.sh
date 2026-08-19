#!/usr/bin/env bash
# Throughput sweep for the 96-stream x 15 fps target (= 1,440 frames/s) on the RTX PRO
# 6000 Blackwell Max-Q. Run ONLY on an idle GPU -- numbers taken while training shares
# the card are not measurements (rule 2: thresholds relative to a measured baseline).
#
#   scripts/bench_pro6000.sh exports/pro6000 runs/bench_pro6000
#
# Engines are fixed-batch on purpose: one engine per batch size composes with CUDA
# graphs, and the serving plan is "fill a batch from N streams every tick" rather than
# dynamic shapes. Throughput line to read: trtexec's "Throughput: X qps" -> X * batch
# = frames/s; the target line is 1,440.
set -euo pipefail
SRC=${1:-exports/pro6000}
OUT=${2:-runs/bench_pro6000}
TRTEXEC=${TRTEXEC:-$(command -v trtexec || echo "$PWD/.venv/bin/trtexec")}
mkdir -p "$OUT"
echo -e "onnx\tprecision\tbatch\tqps\tframes_per_s" > "$OUT/results.tsv"
for onnx in "$SRC"/xl_b*.onnx; do
    b=$(basename "$onnx" .onnx); b=${b#xl_b}
    for prec in fp16 best; do   # "best" lets TRT pick fp8/int8 kernels where calibrated-free
        flag="--fp16"; [ "$prec" = best ] && flag="--best"
        log="$OUT/$(basename "$onnx" .onnx)_$prec.log"
        "$TRTEXEC" --onnx="$onnx" $flag --useCudaGraph --noTF32 \
            --iterations=200 --warmUp=1000 --duration=20 > "$log" 2>&1 || {
            echo "FAILED $onnx $prec (see $log)"; continue; }
        qps=$(grep -oP 'Throughput: \K[0-9.]+' "$log" | tail -1)
        fps=$(python3 -c "print(round(${qps:-0} * $b, 1))")
        echo -e "$(basename "$onnx")\t$prec\t$b\t${qps:-?}\t$fps" >> "$OUT/results.tsv"
        echo "$(basename "$onnx") $prec batch=$b  ${qps:-?} qps  -> $fps frames/s"
    done
done
echo; echo "target: 1440 frames/s (96 streams x 15 fps)"; column -t "$OUT/results.tsv"
