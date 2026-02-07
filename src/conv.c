/*
 * fe_conv.c — Conv1d, StridedConv1d, ConvTranspose1d (optimized)
 *
 * All convolutions operate on single time-step (T=1), so they reduce
 * to matrix operations over the frequency dimension.
 *
 * Data layout: [C, F] (channels-first, frequency as spatial dim)
 * All BN already fused into weights (bias always present).
 *
 * Optimizations:
 *   - k=1: SIMD inner loop for freq-axis accumulation
 *   - k=3: SIMD inner loop with boundary handling
 *   - StridedConv: bit-shift for stride=4 reshape
 *   - ConvTranspose1d: pre-computed valid range, no inner branches
 */
#include "fastenhancer.h"
#include <string.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#define SIMD 1
#else
#define SIMD 0
#endif

/* ------------------------------------------------------------------ */
/*  Conv1d: handles k=1 and k=3 (with padding=(k-1)/2)               */
/*  Data: [C, F], weight: [Co, Ci, K]                                */
/* ------------------------------------------------------------------ */

void fe_conv1d(const FeConv1d *c, const float *in, float *out, int freq) {
    const int Ci = c->in_ch;
    const int Co = c->out_ch;
    const int K  = c->kernel;

    if (K == 1) {
        /* Y[Co, F] = W[Co, Ci] @ X[Ci, F] + b */

        /* Init output with bias (SIMD) */
        for (int o = 0; o < Co; o++) {
            float *out_row = out + o * freq;
#if SIMD
            v128_t vb = wasm_f32x4_splat(c->bias[o]);
            int f = 0;
            for (; f + 3 < freq; f += 4)
                wasm_v128_store(out_row + f, vb);
            for (; f < freq; f++)
                out_row[f] = c->bias[o];
#else
            float b = c->bias[o];
            for (int f = 0; f < freq; f++) out_row[f] = b;
#endif
        }

        /* Accumulate: out[o,f] += w[o,i] * in[i,f] */
        for (int o = 0; o < Co; o++) {
            float *out_row = out + o * freq;
            const float *w_row = c->weight + o * Ci;
            for (int i = 0; i < Ci; i++) {
                float w = w_row[i];
                const float *in_row = in + i * freq;
#if SIMD
                v128_t vw = wasm_f32x4_splat(w);
                int f = 0;
                for (; f + 3 < freq; f += 4) {
                    v128_t va = wasm_v128_load(out_row + f);
                    v128_t vx = wasm_v128_load(in_row + f);
                    wasm_v128_store(out_row + f, wasm_f32x4_add(va, wasm_f32x4_mul(vw, vx)));
                }
                for (; f < freq; f++)
                    out_row[f] += w * in_row[f];
#else
                for (int f = 0; f < freq; f++)
                    out_row[f] += w * in_row[f];
#endif
            }
        }
        return;
    }

    /* Specialized k=3, pad=1 with SIMD inner loop */
    if (K == 3) {
        /* Init output with bias (SIMD) */
        for (int o = 0; o < Co; o++) {
            float *out_row = out + o * freq;
#if SIMD
            v128_t vb = wasm_f32x4_splat(c->bias[o]);
            int f = 0;
            for (; f + 3 < freq; f += 4)
                wasm_v128_store(out_row + f, vb);
            for (; f < freq; f++)
                out_row[f] = c->bias[o];
#else
            float b = c->bias[o];
            for (int f = 0; f < freq; f++) out_row[f] = b;
#endif
        }

        /* Accumulate per (o, i) */
        for (int o = 0; o < Co; o++) {
            float *out_row = out + o * freq;
            const float *w_base = c->weight + o * Ci * 3;

            for (int i = 0; i < Ci; i++) {
                const float w0 = w_base[i * 3 + 0];
                const float w1 = w_base[i * 3 + 1];
                const float w2 = w_base[i * 3 + 2];
                const float *in_row = in + i * freq;

                /* f=0: pad left */
                out_row[0] += w1 * in_row[0] + w2 * in_row[1];

                /* f=1..freq-2: main SIMD loop */
#if SIMD
                {
                    v128_t vw0 = wasm_f32x4_splat(w0);
                    v128_t vw1 = wasm_f32x4_splat(w1);
                    v128_t vw2 = wasm_f32x4_splat(w2);
                    int f = 1;
                    for (; f + 3 < freq - 1; f += 4) {
                        v128_t va = wasm_v128_load(out_row + f);
                        v128_t vi0 = wasm_v128_load(in_row + f - 1);
                        v128_t vi1 = wasm_v128_load(in_row + f);
                        v128_t vi2 = wasm_v128_load(in_row + f + 1);
                        va = wasm_f32x4_add(va, wasm_f32x4_mul(vw0, vi0));
                        va = wasm_f32x4_add(va, wasm_f32x4_mul(vw1, vi1));
                        va = wasm_f32x4_add(va, wasm_f32x4_mul(vw2, vi2));
                        wasm_v128_store(out_row + f, va);
                    }
                    for (; f < freq - 1; f++) {
                        out_row[f] += w0 * in_row[f - 1]
                                    + w1 * in_row[f]
                                    + w2 * in_row[f + 1];
                    }
                }
#else
                for (int f = 1; f < freq - 1; f++) {
                    out_row[f] += w0 * in_row[f - 1]
                                + w1 * in_row[f]
                                + w2 * in_row[f + 1];
                }
#endif

                /* f=freq-1: pad right */
                out_row[freq - 1] += w0 * in_row[freq - 2]
                                   + w1 * in_row[freq - 1];
            }
        }
        return;
    }

    /* General case (shouldn't be reached for this model) */
    {
        const int pad = (K - 1) / 2;
        for (int o = 0; o < Co; o++) {
            float b = c->bias[o];
            for (int f = 0; f < freq; f++) {
                float sum = b;
                for (int i = 0; i < Ci; i++) {
                    for (int k = 0; k < K; k++) {
                        int ff = f + k - pad;
                        if (ff >= 0 && ff < freq) {
                            sum += c->weight[(o * Ci + i) * K + k]
                                 * in[i * freq + ff];
                        }
                    }
                }
                out[o * freq + f] = sum;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  StridedConv1d: in[2, F=256] → out[C1, F1=64]                     */
/*  Original: Conv1d(2→C1, k=8, s=4, pad=2)                          */
/*  Reshaped: input[2, 256] → reshape to [2*4, 64] = [8, 64]         */
/*            then Conv1d(8→C1, k=2, pad=0)                           */
/*                                                                     */
/*  Optimization: use bit-shift for stride=4 (p>>2 instead of p/4,   */
/*  p&3 instead of p%4)                                                */
/* ------------------------------------------------------------------ */

void fe_strided_conv1d(const FeConv1d *c, const float *in, float *out,
                       int in_freq, int stride) {
    const int Ci_orig = c->in_ch >> 2; /* /stride, stride=4 */
    const int Co = c->out_ch;
    const int K  = c->kernel;
    const int pad_orig = (FE_ENC_K0 - stride) >> 1; /* /2 */

    const int F_padded = in_freq + 2 * pad_orig;
    const int Ci = Ci_orig << 2; /* *stride */
    const int F_new = F_padded >> 2; /* /stride */

    /* Temporary buffer for reshaped input [Ci, F_new]
     * Ci = in_ch_orig * stride, F_new = (F_in + 2*pad) / stride */
    float reshaped[8 * 65];  /* (2*4) × (256/4+1) — Tiny hardcoded */
    memset(reshaped, 0, Ci * F_new * sizeof(float));

    for (int ch = 0; ch < Ci_orig; ch++) {
        const float *in_ch = in + ch * in_freq;
        for (int p = 0; p < F_padded; p++) {
            int orig_p = p - pad_orig;
            if (orig_p >= 0 && orig_p < in_freq) {
                int new_ch = (p & 3) * Ci_orig + ch; /* p % stride */
                int new_f  = p >> 2;                   /* p / stride */
                reshaped[new_ch * F_new + new_f] = in_ch[orig_p];
            }
        }
    }

    /* Conv1d(Ci→Co, k=2, pad=0): accumulate loop */
    int out_freq = F_new - K + 1;

    /* Init output with bias */
    for (int o = 0; o < Co; o++) {
        float b = c->bias[o];
        float *out_row = out + o * out_freq;
#if SIMD
        v128_t vb = wasm_f32x4_splat(b);
        int f = 0;
        for (; f + 3 < out_freq; f += 4)
            wasm_v128_store(out_row + f, vb);
        for (; f < out_freq; f++)
            out_row[f] = b;
#else
        for (int f = 0; f < out_freq; f++) out_row[f] = b;
#endif
    }

    /* Accumulate with SIMD */
    for (int o = 0; o < Co; o++) {
        float *out_row = out + o * out_freq;
        for (int i = 0; i < Ci; i++) {
            const float *in_row = reshaped + i * F_new;
            const float *ww = c->weight + (o * Ci + i) * K;
            for (int k = 0; k < K; k++) {
                float wk = ww[k];
#if SIMD
                v128_t vwk = wasm_f32x4_splat(wk);
                const float *src = in_row + k;
                int f = 0;
                for (; f + 3 < out_freq; f += 4) {
                    v128_t va = wasm_v128_load(out_row + f);
                    v128_t vx = wasm_v128_load(src + f);
                    wasm_v128_store(out_row + f, wasm_f32x4_add(va, wasm_f32x4_mul(vwk, vx)));
                }
                for (; f < out_freq; f++)
                    out_row[f] += wk * src[f];
#else
                for (int f = 0; f < out_freq; f++) {
                    out_row[f] += wk * in_row[f + k];
                }
#endif
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  ConvTranspose1d: in[C1, F1=64] → out[2, F=256]                   */
/*  ConvTranspose1d(C1→2, k=8, s=4, pad=2)                           */
/*  out_freq = (F1 - 1) * stride - 2*pad + k                         */
/*                                                                     */
/*  Optimization: for Ci=24, Co=2, K=8, S=4, pad=2:                  */
/*    Most input positions (f=1..62) have all 8 taps valid.            */
/*    Split into valid-range (no branch) + boundary (with branch).    */
/* ------------------------------------------------------------------ */

void fe_conv_transpose1d(const FeConvT1d *c, const float *in, float *out,
                         int in_freq) {
    const int Ci = c->in_ch;
    const int Co = c->out_ch;
    const int K  = c->kernel;
    const int S  = c->stride;
    const int pad = (K - S) >> 1; /* /2, (8-4)/2 = 2 */
    const int out_freq = (in_freq - 1) * S - 2 * pad + K;

    /* Zero output + add bias in one pass */
    for (int o = 0; o < Co; o++) {
        float b = c->bias[o];
        float *row = out + o * out_freq;
#if SIMD
        v128_t vb = wasm_f32x4_splat(b);
        int f = 0;
        for (; f + 3 < out_freq; f += 4)
            wasm_v128_store(row + f, vb);
        for (; f < out_freq; f++)
            row[f] = b;
#else
        for (int f = 0; f < out_freq; f++)
            row[f] = b;
#endif
    }

    /* Pre-compute valid range for f where all K taps are in bounds:
     * base = f*S - pad, need base >= 0 and base+K-1 < out_freq
     * f >= pad/S (ceil), f <= (out_freq - K + pad) / S
     * For pad=2, S=4: f_start = 1, f_end depends on out_freq */
    const int f_safe_start = (pad + S - 1) / S; /* ceil(pad/S) */
    const int f_safe_end = (out_freq - K + pad) / S;

    /* Scatter: for each input position, spread to output via kernel */
    for (int i = 0; i < Ci; i++) {
        for (int o = 0; o < Co; o++) {
            const float *ww = c->weight + (i * Co + o) * K;
            float *out_row = out + o * out_freq;

            /* Boundary: f < f_safe_start */
            for (int f = 0; f < f_safe_start; f++) {
                float x = in[i * in_freq + f];
                int base = f * S - pad;
                for (int k = 0; k < K; k++) {
                    int out_pos = base + k;
                    if (out_pos >= 0 && out_pos < out_freq)
                        out_row[out_pos] += ww[k] * x;
                }
            }

            /* Safe range: no bounds check needed */
            for (int f = f_safe_start; f <= f_safe_end; f++) {
                float x = in[i * in_freq + f];
                int base = f * S - pad;
                /* Unroll K=8 (compile-time constant, compiler will unroll) */
                for (int k = 0; k < K; k++) {
                    out_row[base + k] += ww[k] * x;
                }
            }

            /* Boundary: f > f_safe_end */
            for (int f = f_safe_end + 1; f < in_freq; f++) {
                float x = in[i * in_freq + f];
                int base = f * S - pad;
                for (int k = 0; k < K; k++) {
                    int out_pos = base + k;
                    if (out_pos >= 0 && out_pos < out_freq)
                        out_row[out_pos] += ww[k] * x;
                }
            }
        }
    }
}
