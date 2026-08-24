# The 96-stream number, finally measured — and the config bug it uncovered

[PLAN.md](../PLAN.md) §7 gate 2. The 1,440 frames/s target (96 streams × 15 fps) was set in
`cc80fc3` on 2026-08-19 with the note that "the numbers wait for an idle GPU on purpose".
They waited five days. This is them.

## Verdict: it passes, with 8% headroom

| | frames/s | vs 1,440 |
|---|---:|---:|
| eager PyTorch FP16, batch 16 | 941 | 0.65× |
| **TensorRT FP16, batch 16** | **1,552** | **1.08×** |
| TensorRT + CUDA graph | 1,555 | 1.08× |
| TensorRT FP16, batch 8 | 1,542 | 1.07× |

TensorRT gives **1.65× over eager**. Per-frame budget: 0.644 ms achieved against 0.694 ms
required.

**CUDA graphs buy nothing here** — 1,555 against 1,552, inside noise. That is the opposite
of the Orin, where `DEPLOY.md` measured `--useCudaGraph` alone taking 2.61 → 1.95 ms because
that graph is launch-bound over 416 kernel launches. On a Blackwell card at batch 16 the
launches are amortised and the graph is compute-bound. Worth knowing before anyone spends
time on graph capture in the server runtime.

Batch 8 and batch 16 are within 1% of each other, consistent with the eager measurement that
batch saturates at 8–16. `exports/pro6000/xl_b64.onnx` remains pointless.

## What this does NOT say, stated plainly

1. **It is the wrong second head.** The graph is `terrain + detection`, because that is what
   the trained checkpoint has. [PLAN.md](../PLAN.md) §3.2 puts **pose** in that slot. Eager
   measured terrain at 3% of throughput; if pose costs meaningfully more, the 8% headroom
   goes. **The architecture's margin is currently unmeasured on the architecture's actual
   heads.**
2. **It is engine-only.** No video decode, no host-side NMS, no tracking, no PCIe. `cc80fc3`
   said this in its own commit message — "the arithmetic says FLOPs are not the risk; decode,
   PCIe and host overhead are" — and this measurement does not touch any of them.
3. **8% is thin.** There is no room for a third dense head, and no room for the resolution to
   rise.

So the honest reading: **the per-frame network fits, and everything around it is unproven.**

## The config bug this uncovered

`exports/pro6000/` — the ONNX the benchmark script was written for — is at **512 × 640**.
The cause is in `configs/hydranet_retail_security_b03_cw_xl.yaml` itself:

```yaml
data:
  input_size: [640, 1120]     # what it trains at
export:
  input_size: [512, 640]      # what it exports at
```

The `export` block was copied from an ancestor config in `ee1bdc1` — the very commit whose
message is "push the retail canvas to 640x1120, the lever that measured largest" — and never
updated. So the file spends twenty lines of header arguing that 512 × 640 is unusable
(8.0–12.5% of boxes under FCOS's stride-8 cell, "not hard to learn, unlearnable") and then
exports at exactly that.

**Every engine and every deployment number derived from this config is at 0.57× the scale
the weights learned**, including whatever fed `exports/pro6000/`. Benchmarking those files
would also have flattered the result: 512 × 640 is 0.46× the pixels, so it would have
reported roughly 2× the true throughput and the target would have looked comfortably clear.

Fixed by removing the stale override so the export inherits `data.input_size`.

## Method

`trtexec` is not installed on this machine — only TensorRT's Python bindings — so
`scripts/bench_pro6000.sh` cannot run as written. The engine was built and timed through the
Python API instead: same builder, same fixed-batch engines, warm-up then steady state.

Two API notes for whoever automates this:

* **TensorRT 11 removed `BuilderFlag.FP16`.** Networks are strongly typed and precision comes
  from the ONNX, so the graph is converted with
  `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)` first. The
  `keep_io_types` matters: the exported graph takes RGB 0–255 and does its own
  normalisation, and that input must stay float32.
* Engine build takes ~28 s per batch size on this card. Cached next to the ONNX.

## What gate 2 unblocks

640 × 1120 stands, and so does the two-head shared trunk — provisionally. The next number
that matters is the same one taken with a **pose** head in the second slot, which is gate 5.
Until then the 8% is a margin on a proxy.
