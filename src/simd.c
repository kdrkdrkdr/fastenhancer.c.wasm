/*
 * fe_simd.c — WASM SIMD optimized vector / matrix primitives
 *
 * All hot-path operations vectorized with wasm_simd128.h (128-bit, f32x4).
 *
 * Optimizations:
 *   - 100% SIMD matmul/matmul_bias/gemv: tail mask eliminates scalar fallback
 *   - All vector ops handle arbitrary n (SIMD + scalar tail for small n)
 *   - Shuffle-based horizontal sum (hsum_f32x4) — no store-to-array
 *   - expf used for sigmoid (compiler -ffast-math optimizes it)
 */
#include "fastenhancer.h"

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#define SIMD 1
#else
#define SIMD 0
#endif

#include <math.h>
#include <string.h>

#if SIMD
/* Horizontal sum of f32x4 → scalar float using shuffle-based reduction.
 * Much faster than store-to-array + 4 scalar adds. */
static inline float hsum_f32x4(v128_t v) {
    v128_t s1 = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 2, 3, 0, 1));
    v128_t s2 = wasm_f32x4_add(s1, wasm_i32x4_shuffle(s1, s1, 1, 0, 3, 2));
    return wasm_f32x4_extract_lane(s2, 0);
}
#endif

/* ------------------------------------------------------------------ */
/*  Basic vector operations (with tail handling)                       */
/* ------------------------------------------------------------------ */

void fe_vec_add(float *dst, const float *src, int n) {
    int i = 0;
#if SIMD
    for (; i + 3 < n; i += 4) {
        v128_t a = wasm_v128_load(dst + i);
        v128_t b = wasm_v128_load(src + i);
        wasm_v128_store(dst + i, wasm_f32x4_add(a, b));
    }
#endif
    for (; i < n; i++) dst[i] += src[i];
}

void fe_vec_mul(float *dst, const float *src, int n) {
    int i = 0;
#if SIMD
    for (; i + 3 < n; i += 4) {
        v128_t a = wasm_v128_load(dst + i);
        v128_t b = wasm_v128_load(src + i);
        wasm_v128_store(dst + i, wasm_f32x4_mul(a, b));
    }
#endif
    for (; i < n; i++) dst[i] *= src[i];
}

void fe_vec_scale(float *dst, float s, int n) {
    int i = 0;
#if SIMD
    v128_t vs = wasm_f32x4_splat(s);
    for (; i + 3 < n; i += 4) {
        v128_t a = wasm_v128_load(dst + i);
        wasm_v128_store(dst + i, wasm_f32x4_mul(a, vs));
    }
#endif
    for (; i < n; i++) dst[i] *= s;
}

void fe_vec_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

void fe_vec_zero(float *dst, int n) {
    memset(dst, 0, n * sizeof(float));
}

/* ------------------------------------------------------------------ */
/*  Sigmoid and SiLU                                                   */
/*                                                                     */
/*  Using standard expf — -ffast-math on emcc/clang will optimize      */
/*  this to a fast approximation automatically.                        */
/*  Polynomial approximation was tested but introduces too much         */
/*  accumulated error in the tiny model (max_abs_diff ~0.01 vs 1e-7). */
/* ------------------------------------------------------------------ */

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void fe_vec_silu(float *x, int n) {
    /* SiLU = x * sigmoid(x) */
    for (int i = 0; i < n; i++) {
        x[i] = x[i] * sigmoid_f(x[i]);
    }
}

void fe_sigmoid(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = sigmoid_f(in[i]);
    }
}

/* ------------------------------------------------------------------ */
/*  Matrix multiply — 100% SIMD, no scalar fallback                    */
/*  C[M,N] = A[M,K] × B[N,K]^T + bias[N] (optional)                  */
/*  Tail mask: last SIMD vector masked to zero invalid lanes when      */
/*  K is not a multiple of 4. Same technique as fe_attention.c.        */
/* ------------------------------------------------------------------ */

void fe_matmul(const float *A, const float *B, float *C,
               int M, int N, int K) {
#if SIMD
    const int K4 = K & ~3; /* largest multiple of 4 <= K */
    const int has_tail = K & 3;
    /* Tail mask: zero out invalid lanes in the last SIMD vector */
    v128_t tmask;
    if (has_tail) {
        const int rem = K & 3;
        tmask = (rem == 1) ? wasm_f32x4_make(1.0f, 0.0f, 0.0f, 0.0f)
              : (rem == 2) ? wasm_f32x4_make(1.0f, 1.0f, 0.0f, 0.0f)
              :              wasm_f32x4_make(1.0f, 1.0f, 1.0f, 0.0f);
    }
    for (int m = 0; m < M; m++) {
        const float *a_row = A + m * K;
        for (int n = 0; n < N; n++) {
            const float *b_row = B + n * K;
            v128_t acc = wasm_f32x4_splat(0.0f);
            int k = 0;
            for (; k < K4; k += 4) {
                v128_t va = wasm_v128_load(a_row + k);
                v128_t vb = wasm_v128_load(b_row + k);
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(va, vb));
            }
            if (has_tail) {
                v128_t va = wasm_f32x4_mul(wasm_v128_load(a_row + k), tmask);
                v128_t vb = wasm_v128_load(b_row + k);
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(va, vb));
            }
            C[m * N + n] = hsum_f32x4(acc);
        }
    }
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * B[n * K + k];
            C[m * N + n] = sum;
        }
    }
#endif
}

void fe_matmul_bias(const float *A, const float *B, const float *bias,
                    float *C, int M, int N, int K) {
#if SIMD
    const int K4 = K & ~3;
    const int has_tail = K & 3;
    v128_t tmask;
    if (has_tail) {
        const int rem = K & 3;
        tmask = (rem == 1) ? wasm_f32x4_make(1.0f, 0.0f, 0.0f, 0.0f)
              : (rem == 2) ? wasm_f32x4_make(1.0f, 1.0f, 0.0f, 0.0f)
              :              wasm_f32x4_make(1.0f, 1.0f, 1.0f, 0.0f);
    }
    for (int m = 0; m < M; m++) {
        const float *a_row = A + m * K;
        for (int n = 0; n < N; n++) {
            const float *b_row = B + n * K;
            v128_t acc = wasm_f32x4_splat(0.0f);
            int k = 0;
            for (; k < K4; k += 4) {
                v128_t va = wasm_v128_load(a_row + k);
                v128_t vb = wasm_v128_load(b_row + k);
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(va, vb));
            }
            if (has_tail) {
                v128_t va = wasm_f32x4_mul(wasm_v128_load(a_row + k), tmask);
                v128_t vb = wasm_v128_load(b_row + k);
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(va, vb));
            }
            C[m * N + n] = hsum_f32x4(acc) + bias[n];
        }
    }
#else
    fe_matmul(A, B, C, M, N, K);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C[m * N + n] += bias[n];
#endif
}

/* General matrix-vector: y[out] = W[out,in] @ x[in] + bias[out] */
void fe_gemv(const float *W, const float *x, const float *bias,
             float *y, int out_dim, int in_dim) {
#if SIMD
    const int in4 = in_dim & ~3;
    const int has_tail = in_dim & 3;
    v128_t tmask;
    if (has_tail) {
        const int rem = in_dim & 3;
        tmask = (rem == 1) ? wasm_f32x4_make(1.0f, 0.0f, 0.0f, 0.0f)
              : (rem == 2) ? wasm_f32x4_make(1.0f, 1.0f, 0.0f, 0.0f)
              :              wasm_f32x4_make(1.0f, 1.0f, 1.0f, 0.0f);
    }
    for (int o = 0; o < out_dim; o++) {
        const float *w_row = W + o * in_dim;
        v128_t acc = wasm_f32x4_splat(0.0f);
        int i = 0;
        for (; i < in4; i += 4) {
            v128_t vw = wasm_v128_load(w_row + i);
            v128_t vx = wasm_v128_load(x + i);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(vw, vx));
        }
        if (has_tail) {
            v128_t vw = wasm_f32x4_mul(wasm_v128_load(w_row + i), tmask);
            v128_t vx = wasm_v128_load(x + i);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(vw, vx));
        }
        float sum = hsum_f32x4(acc);
        y[o] = bias ? sum + bias[o] : sum;
    }
#else
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; i++)
            sum += W[o * in_dim + i] * x[i];
        y[o] = bias ? sum + bias[o] : sum;
    }
#endif
}
