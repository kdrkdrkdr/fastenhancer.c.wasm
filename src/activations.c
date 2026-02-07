/*
 * fe_activations.c — Activation functions
 *
 * SiLU (x * sigmoid(x)) is in fe_simd.c for SIMD optimization.
 * This file contains softmax used in attention.
 *
 * Optimizations:
 *   - SIMD max/exp/normalize for cols that are multiples of 4
 */
#include "fastenhancer.h"
#include <math.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#define SIMD 1
#else
#define SIMD 0
#endif

/* ------------------------------------------------------------------ */
/*  Softmax over rows: x[rows, cols] — each row independently         */
/*  For attention: rows=cols=F2=16, so cols is always multiple of 4.  */
/* ------------------------------------------------------------------ */

void fe_softmax_rows(float *x, int rows, int cols) {
#if SIMD
    /* Fast path: cols is multiple of 4 (F2=16) */
    if ((cols & 3) == 0) {
        for (int r = 0; r < rows; r++) {
            float *row = x + r * cols;

            /* 1. Find max (SIMD horizontal max) */
            v128_t vmax = wasm_v128_load(row);
            for (int c = 4; c < cols; c += 4) {
                vmax = wasm_f32x4_max(vmax, wasm_v128_load(row + c));
            }
            /* Reduce 4 lanes to scalar max */
            vmax = wasm_f32x4_max(vmax,
                wasm_i32x4_shuffle(vmax, vmax, 2, 3, 0, 1));
            vmax = wasm_f32x4_max(vmax,
                wasm_i32x4_shuffle(vmax, vmax, 1, 0, 3, 2));
            /* vmax now has max in all 4 lanes */

            /* 2. exp(x - max) and sum */
            v128_t vsum = wasm_f32x4_splat(0.0f);
            for (int c = 0; c < cols; c += 4) {
                v128_t v = wasm_v128_load(row + c);
                v = wasm_f32x4_sub(v, vmax);
                /* expf per lane — no SIMD exp intrinsic, extract and compute */
                float e0 = expf(wasm_f32x4_extract_lane(v, 0));
                float e1 = expf(wasm_f32x4_extract_lane(v, 1));
                float e2 = expf(wasm_f32x4_extract_lane(v, 2));
                float e3 = expf(wasm_f32x4_extract_lane(v, 3));
                v128_t ve = wasm_f32x4_make(e0, e1, e2, e3);
                wasm_v128_store(row + c, ve);
                vsum = wasm_f32x4_add(vsum, ve);
            }
            /* Horizontal sum */
            vsum = wasm_f32x4_add(vsum,
                wasm_i32x4_shuffle(vsum, vsum, 2, 3, 0, 1));
            vsum = wasm_f32x4_add(vsum,
                wasm_i32x4_shuffle(vsum, vsum, 1, 0, 3, 2));
            /* vsum now has total sum in all 4 lanes */

            /* 3. Normalize: row[c] *= 1/sum */
            v128_t vinv = wasm_f32x4_div(wasm_f32x4_splat(1.0f), vsum);
            for (int c = 0; c < cols; c += 4) {
                wasm_v128_store(row + c,
                    wasm_f32x4_mul(wasm_v128_load(row + c), vinv));
            }
        }
        return;
    }
#endif
    /* Scalar fallback */
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;

        /* Find max for numerical stability */
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }

        /* exp and sum */
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }

        /* Normalize */
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}
