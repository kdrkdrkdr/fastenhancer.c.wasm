# fastenhancer.c.wasm

Real-time speech enhancement in the browser using pure C + WASM SIMD 128-bit.

A from-scratch C inference engine for the [FastEnhancer](https://github.com/aask1357/fastenhancer) model, targeting WebAssembly SIMD for AudioWorklet deployment.

| | |
|---|---|
| **Target** | WASM SIMD 128-bit (f32x4), AudioWorklet |
| **Model** | FastEnhancer-Tiny (22K params, 95 KB weights) |
| **Sample Rate** | 16 kHz |
| **Frame Size** | 256 samples (16 ms) |
| **Latency** | 1 frame (16 ms algorithmic) |
| **Output** | 183 KB single-file JS (WASM + weights inline) |
| **Accuracy** | max\_abs\_diff < 1e-7 vs PyTorch (per-frame) |

---

## Architecture

```
Audio In (256 samples)
    |
    v
+-----------------------------------------------+
| Streaming STFT (hop=256, n_fft=512, Hann)     |
|   overlap-save cache, in-place radix-2 FFT    |
+-----------------------------------------------+
    |  [256, 2] complex spectrum
    v
+-----------------------------------------------+
| Spectral Compression (mag^0.3 * phase)        |
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
| Encoder                                       |
|   StridedConv(2->24, k8s4) -> [24, 64]        |
|   2x Conv1d(24->24, k3) + SiLU                |
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
| RNNFormer (x2 blocks)                         |
|   Linear [64->16] + Conv1d [24->20, k1]       |
|   GRU(20) + Linear(20) + residual             |
|   MHSA(20, 4-head) + Linear(20) + residual    |
|   Conv1d [20->24, k1] + Linear [16->64]       |
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
| Decoder                                       |
|   2x skip-concat -> Conv(48->24) + Conv(24)   |
|   skip-concat -> Conv(48->24) + ConvT(24->2)  |
+-----------------------------------------------+
    |  [256, 2] complex mask
    v
+-----------------------------------------------+
| Complex Multiply + Decompression + iSTFT      |
+-----------------------------------------------+
    |
    v
Audio Out (256 samples)
```

---

## Differences from the Original PyTorch Repository

The [original FastEnhancer](https://github.com/aask1357/fastenhancer) is a PyTorch model for offline batch inference. This repository is a **complete C rewrite** optimized for real-time browser deployment:

| | Original (PyTorch) | This Engine (C/WASM) |
|---|---|---|
| Runtime | Python / CUDA / CPU | WASM SIMD 128-bit (AudioWorklet) |
| Processing | Batch, variable-length | Single-frame streaming (256 samples) |
| Memory | Dynamic allocation + GC | Pre-allocated static buffers, zero malloc at inference |
| Weights | `state_dict` with metadata | Flat binary blob, zero-copy pointer references within engine |
| BatchNorm | Runtime normalization | Pre-fused into Conv/Linear weights |
| FFT | `torch.fft.rfft` | Custom radix-2 DIT, half-complex trick, SoA twiddle |
| Q/K/V | 3 separate `nn.Linear` | Single combined `[3*C2, C2]` matmul |
| GRU | Per-timestep, separate gates | Batch matmul + fused single-pass gates |
| Conv padding | `F.pad()` + conv | Boundary-separated loops, no pad allocation |
| Binary size | ~100 MB+ (typical PyTorch runtime + model) | 183 KB (single JS file) |

38 optimizations applied across SIMD vectorization, memory layout, algorithmic shortcuts, and WASM-specific tuning. See [OPTIMIZATION.md](OPTIMIZATION.md) for the full breakdown.

---

## Weights

Pre-exported weights (`weights/fe_tiny.bin`, 95 KB) are included in this repository. No additional download is needed — just clone and build.

### Re-exporting from PyTorch (optional)

If you want to export weights yourself from the original PyTorch checkpoint:

1. Clone the [FastEnhancer](https://github.com/aask1357/fastenhancer) repository as a sibling directory:
   ```
   parent/
   ├── fastenhancer/       # original PyTorch repo (git clone)
   └── fastenhancer.c.wasm/ # this repo
   ```
2. Download the FastEnhancer-Tiny checkpoint into the cloned repo (see their README for download links)
3. Install Python 3 + PyTorch
4. Run the export script:

```bash
python export_weights.py \
    --config path/to/config.yaml \
    --ckpt-dir path/to/checkpoint_dir \
    --output weights/fe_tiny.bin
```

The exporter imports model definitions from the original repo (`../fastenhancer/`), fuses BatchNorm, removes weight normalization, and writes a flat float32 binary.

---

## Build

### Prerequisites
- [Emscripten](https://emscripten.org/) (emsdk)

```bash
bash build.sh
# Output: fastenhancer-worklet.js (183 KB)
```

The output is a self-contained AudioWorklet processor with WASM and weights inlined as base64. Load it in a browser:

```js
await audioContext.audioWorklet.addModule('fastenhancer-worklet.js');
const node = new AudioWorkletNode(audioContext, 'FastEnhancerProcessor');
source.connect(node);
```

---

## File Structure

```
fastenhancer.c.wasm/
├── src/
│   ├── fastenhancer.h      # Model constants, structs, API declarations
│   ├── fastenhancer.c      # Main inference pipeline (fe_process)
│   ├── api.c               # WASM export API (fe_init, fe_run, fe_free)
│   ├── simd.c              # SIMD vector ops, matmul, gemv
│   ├── fft.c               # Radix-2 DIT FFT (SoA twiddle tables)
│   ├── stft.c              # Streaming STFT / iSTFT
│   ├── conv.c              # Conv1d, StridedConv, ConvTranspose1d
│   ├── gru.c               # GRU (batch matmul + fused gates)
│   ├── attention.c         # Multi-Head Self-Attention (100% SIMD)
│   └── activations.c       # SiLU, sigmoid, softmax
├── weights/
│   └── fe_tiny.bin         # 95,288 bytes flat binary weights (included)
├── build.sh                # WASM SIMD build script
├── export_weights.py       # PyTorch → binary weight exporter
└── OPTIMIZATION.md         # Detailed optimization log (38 optimizations)
```

---

## Performance

All measurements are single-core, single-thread (AudioWorklet runs on one thread).

| Device | Frame Time | RTF |
|---|---|---|
| Snapdragon 8 Gen 2 (Chrome) | ~4–5 ms | 0.28 |
| Exynos 9825 (Chrome) | ~8 ms | 0.50 |

| Metric | Value |
|---|---|
| Binary size | 183 KB |
| Accuracy vs PyTorch | max\_abs\_diff < 1e-7 |

See [OPTIMIZATION.md](OPTIMIZATION.md) for the full optimization log.
