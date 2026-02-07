/*
 * fe_attention.c — Multi-Head Self-Attention (frequency axis)
 *
 * Input:  [F2, C2]
 * Heads:  NUM_HEADS, head_dim = C2/NUM_HEADS
 *
 * 1. QKV = Linear(x) → [F2, 3*C2]
 * 2. Reshape to [NH, F2, head_dim] for Q, K, V each
 * 3. scores = Q @ K^T / sqrt(head_dim)  → [NH, F2, F2]
 * 4. attn = softmax(scores) @ V          → [NH, F2, head_dim]
 * 5. concat → [F2, C2]
 * 6. FC linear → [F2, C2]
 *
 * Optimizations:
 *   - Pure SIMD dot product and score×V for ANY head_dim
 *   - Tail mask: last SIMD vector masked to zero out padding lanes
 *   - No scalar fallback in hot loops — 100% SIMD
 */
#include "fastenhancer.h"
#include <math.h>
#include <string.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#define SIMD 1
#else
#define SIMD 0
#endif

#if SIMD
/* Compile-time tail mask for the last SIMD vector.
 * HD % 4 == 0: no mask needed (all lanes valid)
 * HD % 4 == 1: [valid, 0, 0, 0]
 * HD % 4 == 2: [valid, valid, 0, 0]
 * HD % 4 == 3: [valid, valid, valid, 0]
 *
 * We use this to zero out invalid lanes when loading the last
 * partial vector, so the SIMD dot product is exact without scalar tail. */
static inline v128_t fe_attn_tail_mask(void) {
    const int rem = FE_HEAD_DIM & 3;
    if (rem == 0) return wasm_f32x4_splat(1.0f); /* unused, all lanes valid */
    if (rem == 1) return wasm_f32x4_make(1.0f, 0.0f, 0.0f, 0.0f);
    if (rem == 2) return wasm_f32x4_make(1.0f, 1.0f, 0.0f, 0.0f);
    /*   rem == 3 */ return wasm_f32x4_make(1.0f, 1.0f, 1.0f, 0.0f);
}

/* Horizontal sum: reduce f32x4 to scalar */
static inline float hsum4(v128_t v) {
    v128_t s1 = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 2, 3, 0, 1));
    v128_t s2 = wasm_f32x4_add(s1, wasm_i32x4_shuffle(s1, s1, 1, 0, 3, 2));
    return wasm_f32x4_extract_lane(s2, 0);
}
#endif

/* Round up to multiple of 4 */
#define HD4 ((FE_HEAD_DIM + 3) & ~3)

void fe_mhsa(const FeAttention *a, const FeLinear *fc,
             const float *in, float *out,
             float *qkv_buf, float *score_buf, float *attn_buf,
             int freq) {
    const int C = FE_C2;
    const int NH = FE_NUM_HEADS;
    const int HD = FE_HEAD_DIM;
    const int C3 = 3 * C;

    const float scale = 1.0f / sqrtf((float)HD);

    /* 1. QKV projection: [F2, C2] → [F2, 3*C2] */
    if (a->qkv.bias) {
        fe_matmul_bias(in, a->qkv.weight, a->qkv.bias,
                       qkv_buf, freq, C3, C);
    } else {
        fe_matmul(in, a->qkv.weight, qkv_buf, freq, C3, C);
    }

    /* 2-4. Per-head attention */
#if SIMD
    const int hd4 = HD & ~3;          /* largest multiple of 4 <= HD */
    const int has_tail = (HD & 3) != 0;
    const v128_t tmask = fe_attn_tail_mask();
#endif
    const int freq_sq = freq * freq;
    const int HD3 = 3 * HD;           /* stride between heads in QKV */

    for (int h = 0; h < NH; h++) {
        const int h_base = h * HD3;
        const int k_off = h_base + HD;      /* K offset pre-computed */
        const int v_off = h_base + 2 * HD;  /* V offset pre-computed */
        float *scores = score_buf + h * freq_sq;

#if SIMD
        /* --- Dot product: score[f1,f2] = Q[f1,:] · K[f2,:] * scale --- */
        for (int f1 = 0; f1 < freq; f1++) {
            const float *q = qkv_buf + f1 * C3 + h_base;
            float *score_row = scores + f1 * freq;

            for (int f2 = 0; f2 < freq; f2++) {
                const float *k = qkv_buf + f2 * C3 + k_off;

                v128_t acc = wasm_f32x4_splat(0.0f);
                int d = 0;
                for (; d < hd4; d += 4) {
                    acc = wasm_f32x4_add(acc,
                        wasm_f32x4_mul(wasm_v128_load(q + d),
                                       wasm_v128_load(k + d)));
                }
                if (has_tail) {
                    /* Masked load: read 4 floats, zero out invalid lanes */
                    v128_t vq = wasm_f32x4_mul(wasm_v128_load(q + d), tmask);
                    v128_t vk = wasm_v128_load(k + d);
                    acc = wasm_f32x4_add(acc, wasm_f32x4_mul(vq, vk));
                }
                score_row[f2] = hsum4(acc) * scale;
            }
        }

        /* Softmax over each row */
        fe_softmax_rows(scores, freq, freq);

        /* --- Score × V accumulate ---
         * dst is within attn_buf[F2, C2], heads are packed at stride HD.
         * We CANNOT over-write past HD because it would clobber the next
         * head's data. So for the tail vector, we must use a read-mask-write
         * pattern that preserves the out-of-range lanes in dst. */
        for (int f1 = 0; f1 < freq; f1++) {
            float *dst = attn_buf + f1 * C + h * HD;

            /* Zero output: full vectors + tail.
             * For tail: invalid lanes get 0 written, but that's fine —
             * the next head will overwrite them with correct values.
             * The tail accumulate (V*tmask) ensures we only ADD to valid lanes. */
            for (int d = 0; d < hd4; d += 4)
                wasm_v128_store(dst + d, wasm_f32x4_splat(0.0f));
            if (has_tail)
                wasm_v128_store(dst + hd4, wasm_f32x4_splat(0.0f));

            const float *score_row = scores + f1 * freq;
            for (int f2 = 0; f2 < freq; f2++) {
                v128_t vs = wasm_f32x4_splat(score_row[f2]);
                const float *v = qkv_buf + f2 * C3 + v_off;
                int d = 0;
                for (; d < hd4; d += 4) {
                    v128_t vd = wasm_v128_load(dst + d);
                    v128_t vv = wasm_v128_load(v + d);
                    wasm_v128_store(dst + d, wasm_f32x4_add(vd, wasm_f32x4_mul(vs, vv)));
                }
                if (has_tail) {
                    /* Accumulate only valid lanes, preserve invalid lanes in dst */
                    v128_t vd = wasm_v128_load(dst + d);
                    v128_t vv = wasm_f32x4_mul(wasm_v128_load(v + d), tmask);
                    wasm_v128_store(dst + d, wasm_f32x4_add(vd, wasm_f32x4_mul(vs, vv)));
                }
            }
        }
#else
        /* Scalar fallback */
        for (int f1 = 0; f1 < freq; f1++) {
            const float *q = qkv_buf + f1 * C3 + h_base;
            float *score_row = scores + f1 * freq;
            for (int f2 = 0; f2 < freq; f2++) {
                const float *k = qkv_buf + f2 * C3 + k_off;
                float dot = 0.0f;
                for (int d = 0; d < HD; d++)
                    dot += q[d] * k[d];
                score_row[f2] = dot * scale;
            }
        }

        fe_softmax_rows(scores, freq, freq);

        for (int f1 = 0; f1 < freq; f1++) {
            float *dst = attn_buf + f1 * C + h * HD;
            for (int d = 0; d < HD; d++) dst[d] = 0.0f;

            const float *score_row = scores + f1 * freq;
            for (int f2 = 0; f2 < freq; f2++) {
                float s = score_row[f2];
                const float *v = qkv_buf + f2 * C3 + v_off;
                for (int d = 0; d < HD; d++)
                    dst[d] += s * v[d];
            }
        }
#endif
    }

    /* 5-6. FC projection */
    if (fc->bias) {
        fe_matmul_bias(attn_buf, fc->weight, fc->bias,
                       out, freq, C, C);
    } else {
        fe_matmul(attn_buf, fc->weight, out, freq, C, C);
    }
}
