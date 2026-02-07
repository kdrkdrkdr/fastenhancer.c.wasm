# Optimization Log

Every optimization applied during the PyTorch → C → WASM SIMD porting process.

## How This Differs from the Original PyTorch Implementation

The [original FastEnhancer](https://github.com/aask1357/fastenhancer) is a PyTorch model designed for offline batch processing. This C engine is a complete rewrite targeting real-time streaming inference in the browser:

| | Original (PyTorch) | This Engine (C/WASM) |
|---|---|---|
| **Runtime** | Python / CUDA / CPU | WASM SIMD 128-bit (browser) |
| **Processing** | Batch, variable-length | Single-frame streaming (256 samples) |
| **Memory** | Dynamic tensor allocation + GC | Pre-allocated static buffers, zero malloc at inference |
| **Weights** | `state_dict` with metadata | Flat binary blob, zero-copy pointer references within WASM memory |
| **BatchNorm** | Runtime normalization | Pre-fused into Conv/Linear weights at export |
| **Weight Norm** | Runtime reparameterization | Removed before export (`remove_weight_reparameterizations()`) |
| **FFT** | `torch.fft.rfft` (full library) | Custom radix-2 DIT with half-complex trick, SoA twiddle tables |
| **STFT** | Full-length spectrogram | Overlap-save streaming cache (256 samples) |
| **Q/K/V Projection** | 3 separate `nn.Linear` | Single combined `[3*C2, C2]` matmul |
| **GRU** | Per-timestep, separate gate arrays | Batch matmul over all freq bins + fused single-pass gates |
| **Conv Boundary** | `F.pad()` + conv | Boundary-separated loops (no pad tensor, no bounds check in main loop) |
| **Layout** | `[B, C, T]` with batch dim | `[C, F]` channels-first, no batch dimension |
| **Activation** | `torch.nn.SiLU` module | Inline SIMD SiLU with `sigmoid(x) * x` fused |
| **Binary Size** | ~100 MB+ (typical PyTorch runtime + model) | 183 KB (WASM + weights + worklet, single file) |

---

## 1. Memory & Allocation

### 1.1 Compile-Time Constants
All tensor dimensions fixed via `#define` — enables stack/static allocation, constant folding, and loop unrolling by the compiler. Zero runtime shape inference.

### 1.2 Pre-Allocated FeState
Every intermediate buffer (encoder, RNNFormer, attention, FFT, spec I/O) lives in a single `FeState` struct. One `calloc()` at init, zero `malloc`/`free` during inference. Total ~15 KB, fits in L1/L2 cache.

### 1.3 Zero-Copy Weight Loading
`fe_load_weights()` walks through the binary blob with pointer arithmetic — `const float *ptr = (data + offset)`. No `memcpy`, no deserialization within the engine. Weight data stays in-place after the initial load into WASM memory.

### 1.4 Ping-Pong Buffer Reuse
`buf_a` and `buf_b` alternate as input/output across encoder/decoder blocks via pointer swap (`float *tmp = cur; cur = nxt; nxt = tmp`), eliminating per-block buffer copies.

### 1.5 Skip Connection Array
Encoder skip connections stored in a fixed 2D array `enc_skip[ENC_BLOCKS+1][C1*F1]` — direct indexing, no pointer chasing or dynamic allocation.

### 1.6 Channels-First [C, F] Layout
Fixed channels-first layout throughout. Frequency dimension is contiguous for SIMD operations. No runtime transpose between layers.

### 1.7 Interleaved Complex Spectrum
Spec stored as `[re, im, re, im, ...]` for cache-friendly complex operations. Converted to channels-first once at encoder entry via SIMD shuffle.

---

## 2. SIMD Vectorization

### 2.1 f32x4 Matmul / GEMV
`fe_matmul`, `fe_matmul_bias`, `fe_gemv` — all inner product loops use `wasm_f32x4_mul` + `wasm_f32x4_add` accumulation with `hsum_f32x4()` (2-stage shuffle+add reduction, no memory round-trip).

### 2.2 Matmul Scalar Tail Elimination
`FE_C2 = 20` is not a multiple of 4. Instead of scalar fallback for the last 4 elements, a pre-computed tail mask (`wasm_f32x4_make(1,1,1,0)` etc.) zeros invalid lanes. Applied to `fe_matmul`, `fe_matmul_bias`, `fe_gemv` — **100% SIMD inner loops, zero scalar code**.

GRU+Attention FC: `fe_matmul_bias(freq=16, N=60, K=20)` = 960 inner products × 4 scalar ops = ~3840 unnecessary scalar ops per frame → eliminated.

### 2.3 Conv1d SIMD
Weight splatted to all 4 lanes (`wasm_f32x4_splat`), then 4 frequency bins processed per iteration. For k=1 projections, achieves near-optimal memory bandwidth utilization.

### 2.4 Conv Boundary Separation
k=3 convolution: boundary positions (f=0, f=freq-1) handled explicitly. Main loop (f=1..freq-2) runs without any bounds check, fully SIMD.

### 2.5 ConvTranspose1d Safe Range
Pre-compute `[f_safe_start, f_safe_end]` where all K taps are valid. For Ci=24, F1=64: boundary = 2 positions, safe = 60 positions — eliminates 480 branch checks per channel.

### 2.6 Strided Conv Reshape
k=8, s=4 strided convolution reshaped to k=2 at input level. Bit-shift operations for stride=4: `>>2` for division, `&3` for modulo.

### 2.7 4x4 SIMD Transpose
Layout conversion [C,F] ↔ [F,C] via 4x4 tile transposition using 8 shuffle operations. 16 elements per tile: 4 loads, 8 shuffles, 4 stores vs 16 scattered loads/stores.

### 2.8 SIMD Deinterleave/Interleave
Complex data [r0,i0,r1,i1,...] → separate [r0,r1,...] / [i0,i1,...] via 2-vector shuffle. Used in STFT, iSTFT, complex mask. Halves memory bandwidth.

### 2.9 Fused Complex Multiply + Layout Convert
Single pass: deinterleave input spec → complex multiply with mask → interleave output. Eliminates 2 intermediate buffers and 3x loop overhead.

### 2.10 SIMD FFT Butterfly
Stages with half ≥ 4 (6 out of 8 stages for 512-point FFT) run fully vectorized. Smallest stages (len=2, len=4) use scalar where SIMD shuffle overhead would dominate.

### 2.11 SIMD SiLU
Inline `sigmoid(x) * x` with SIMD sigmoid approximation path, 4 elements per iteration.

### 2.12 SIMD Softmax
SIMD max-reduction (shuffle pattern), per-lane scalar `expf` with SIMD store/accumulate, SIMD sum-reduction, vectorized normalization. F2=16 is always multiple of 4.

### 2.13 Attention SIMD Q·K and Score×V
head_dim=5: inner product via f32x4 + compile-time tail mask (`fe_attn_tail_mask()`). Score×V accumulation uses same tail mask pattern. Zero scalar fallback.

### 2.14 SIMD Vector Primitives
`fe_vec_add`, `fe_vec_mul`, `fe_vec_scale`, `fe_vec_copy`, `fe_vec_zero` — all SIMD paths with scalar tail handling.

---

## 3. FFT / STFT

### 3.1 Half-Complex FFT
N-point real FFT via N/2-point complex FFT: pack real input as complex pairs, run N/2 FFT, post-process to get N/2+1 bins. Halves FFT computation (256-point instead of 512-point).

### 3.2 Structure-of-Arrays Twiddle Tables
Twiddle factors stored as separate `cos[]` and `sin[]` arrays instead of interleaved `(cos,sin)` pairs. Enables contiguous SIMD loads in butterfly.

### 3.3 Per-Stage Flattened Twiddle Tables
Pre-flatten twiddle factors for each SIMD stage into sequential arrays. Eliminates gather/scatter patterns, enabling sequential SIMD loads at full memory bandwidth.

### 3.4 Overlap-Save Streaming STFT
256-sample cache for frame-to-frame continuity. No full-length spectrogram — processes exactly 1 frame per call with fixed-size buffers.

---

## 4. GRU

### 4.1 Batch Matmul
Instead of 16 per-frequency gemv calls × 2 (W_ih, W_hh) = 32 function calls, batch all freq bins into 2 matmul calls:
```
fe_matmul_bias(x, W_ih, b_ih, ih_batch, freq=16, D3=60, D=20)
fe_matmul_bias(h, W_hh, b_hh, hh_batch, freq=16, D3=60, D=20)
```
32× → 2× function call overhead. Larger matrices = better SIMD throughput.

### 4.2 Fused Gate Computation
Single-pass inline computation of all 3 gates + hidden state update. No intermediate gate arrays (r[], z[], n[]).
```c
float r = 1.0f / (1.0f + expf(-(ih[i] + hh[i])));
float z = 1.0f / (1.0f + expf(-(ih_z[i] + hh_z[i])));
float nv = tanhf(ih_n[i] + r * hh_n[i]);
hf[i] = nv + z * (hf[i] - nv);  // algebraically equivalent to (1-z)*n + z*h, saves 1 multiply
```

### 4.3 Gate Offset Hoisting
Pointers `ih_z`, `hh_z`, `ih_n`, `hh_n` pre-computed outside the frequency loop. Eliminates `D` and `2*D` offset arithmetic per inner iteration.

---

## 5. Attention

### 5.1 Combined QKV Projection
Three separate Q, K, V linear layers merged into single `[3*C2, C2]` weight matrix. One matmul instead of three.

### 5.2 Compile-Time Tail Mask
`fe_attn_tail_mask()` generates mask based on `FE_HEAD_DIM % 4` at compile time. Invalid lanes zeroed — completely eliminates scalar fallback in Q·K dot product and Score×V accumulation.

### 5.3 Loop Invariant Hoisting
`hd4`, `has_tail`, `tmask`, `freq_sq`, `HD3` computed once outside the per-head loop (head-independent values).

### 5.4 Scale Factor Pre-computation
`1.0f / sqrtf(HD)` computed once before all heads, not per query-key pair.

---

## 6. Weight Export

### 6.1 BN Fusion
All BatchNorm layers folded into preceding Conv/Linear at export time:
```
W_new = W * gamma / sqrt(var + eps)
b_new = (b - mean) * gamma / sqrt(var + eps) + beta
```
Eliminates per-element normalization at runtime.

### 6.2 Weight Norm Removal
`remove_weight_reparameterizations()` applied before export. Runtime weight_v / weight_g computation eliminated.

### 6.3 Row-Major Flat Binary
Single `.bin` file, row-major float32. 95,288 bytes for Tiny. No serialization format overhead (no pickle, no safetensors metadata).

---

## 7. Micro-Optimizations

### 7.1 Power Compression via log2/exp2
`powf(mag, -0.7)` → `exp2f(-0.35f * log2f(mag_sq))`. Avoids `powf` (slow on most architectures) and `sqrtf` (uses mag² directly).

### 7.2 Redundant memset Removal
`calloc()` already zero-initializes — removed explicit `memset(gru_h[i], 0, ...)` calls.

### 7.3 RNNFormer Residual Direct Addition
`fe_vec_add(rf_c, rf_b)` + `memcpy(rf_fc, rf_c)` (2 passes) → single SIMD pass writing directly to output buffer.

### 7.4 Dead Code Removal
`fast_powf()` in stft.c: defined but never called — deleted.

### 7.5 Conditional SIMD Compilation
`#ifdef __wasm_simd128__` ensures scalar fallback is dead-code eliminated when compiled with `-msimd128`. Zero runtime feature detection overhead.

### 7.6 Unused Parameter Suppression
`(void)n;` for compile-time-known parameters. API consistency preserved without compiler warnings.

### 7.7 Bit Shift for Stride Operations
Stride=4 operations use `>>2` (divide) and `&3` (modulo) instead of integer division. Single-cycle vs multi-cycle.

---

## 8. WASM-Specific

### 8.1 Single-File Embedding
`-sSINGLE_FILE=1` inlines WASM binary as base64 in the JS loader. No separate `.wasm` fetch — single `<script>` or `addModule()` call.

### 8.2 Fixed Memory
`-sALLOW_MEMORY_GROWTH=0 -sINITIAL_MEMORY=16MB`. No `memory.grow` calls or associated overhead at runtime.

### 8.3 Minimal Allocator
`-sMALLOC=emmalloc`. Smaller than dlmalloc, sufficient for 1 `calloc` + 3 `malloc` calls total.

### 8.4 Stripped Runtime
`-sFILESYSTEM=0 -sSUPPORT_LONGJMP=0 -sDISABLE_EXCEPTION_CATCHING=1`. No filesystem, no setjmp/longjmp, no exception handling — removes ~30 KB of unused runtime.

### 8.5 Global Singleton API
WASM exports only 3 functions: `fe_init`, `fe_run`, `fe_free`. Global state avoids pointer passing across JS↔WASM boundary.

### 8.6 AudioWorklet Integration
Weights embedded as base64 in the worklet JS file. Decoded once at init. Zero-GC Int16 buffer pool with 4 pre-allocated ArrayBuffers for PCM output.
