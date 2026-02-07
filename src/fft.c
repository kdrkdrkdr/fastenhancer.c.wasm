/*
 * fe_fft.c — 512-point real FFT / iFFT
 *
 * Uses half-complex trick: N-point real FFT via N/2-point complex FFT.
 * Twiddle factors pre-computed at init time.
 * Output: rfft → 257 complex (half-spectrum).
 *
 * Optimizations:
 *   - Full twiddle table for all k (no runtime cos/sin in rfft unpack)
 *   - Bit-reversal via table lookup
 *   - SIMD butterfly for stages with half >= 4
 *   - Per-stage pre-flattened twiddle tables (no gather in SIMD butterfly)
 *   - SIMD pack/unpack (deinterleave/interleave)
 *   - SIMD inverse scaling
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/*  Pre-computed tables                                                */
/* ------------------------------------------------------------------ */

#define HALF_N (FE_N_FFT / 2)  /* 256 */
#define LOG2_HALF 8             /* log2(256) */

/* Number of SIMD butterfly stages: stages where half >= 4
 * half = len/2, len = 8,16,32,64,128,256 → half = 4,8,16,32,64,128
 * That's 6 stages (stage indices 3..8 for log2 of len) */
#define NUM_SIMD_STAGES 6

static float tw_re[HALF_N];    /* twiddle for N/2-point FFT (original) */
static float tw_im[HALF_N];
static int   bit_rev_half[HALF_N];

/* Per-stage flattened twiddle tables for SIMD butterfly.
 * For stage s (len = 1<<(s+1), half = 1<<s, step = HALF_N/len):
 *   flattened[k] = tw[k * step]  for k = 0..half-1
 * Total elements across all SIMD stages:
 *   4 + 8 + 16 + 32 + 64 + 128 = 252 */
#define FLAT_TW_TOTAL (4 + 8 + 16 + 32 + 64 + 128)
static float flat_tw_re[FLAT_TW_TOTAL];
static float flat_tw_im[FLAT_TW_TOTAL];
/* Offset into flat_tw_re/im for each SIMD stage */
static int flat_tw_offset[NUM_SIMD_STAGES];

/* Post-processing twiddles for real FFT unpack: full table for all k */
static float post_re[HALF_N];
static float post_im[HALF_N];

static int fft_initialized = 0;

static int reverse_bits(int x, int log2n) {
    int result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void fe_fft_init(void) {
    if (fft_initialized) return;

    for (int i = 0; i < HALF_N; i++)
        bit_rev_half[i] = reverse_bits(i, LOG2_HALF);

    for (int k = 0; k < HALF_N; k++) {
        double angle = -2.0 * M_PI * k / HALF_N;
        tw_re[k] = (float)cos(angle);
        tw_im[k] = (float)sin(angle);
    }

    /* Build per-stage flattened twiddle tables.
     * SIMD stages: half = 4, 8, 16, 32, 64, 128
     * (len = 8, 16, 32, 64, 128, 256) */
    int flat_pos = 0;
    int stage_idx = 0;
    for (int len = 8; len <= HALF_N; len <<= 1) {
        int half = len >> 1;
        int step = HALF_N / len;
        flat_tw_offset[stage_idx] = flat_pos;
        for (int k = 0; k < half; k++) {
            flat_tw_re[flat_pos + k] = tw_re[k * step];
            flat_tw_im[flat_pos + k] = tw_im[k * step];
        }
        flat_pos += half;
        stage_idx++;
    }

    /* Full post-processing twiddle table for all k in [0, HALF_N) */
    for (int k = 0; k < HALF_N; k++) {
        double angle = -2.0 * M_PI * k / FE_N_FFT;
        post_re[k] = (float)cos(angle);
        post_im[k] = (float)sin(angle);
    }

    fft_initialized = 1;
}

/* ------------------------------------------------------------------ */
/*  N/2-point complex FFT (in-place, decimation-in-time)               */
/* ------------------------------------------------------------------ */

static void fft_half(float *re, float *im, int inverse) {
    const int N = HALF_N;

    /* Bit-reversal permutation */
    for (int i = 0; i < N; i++) {
        int j = bit_rev_half[i];
        if (i < j) {
            float tmp;
            tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
    }

    /* Butterfly stages */
    int simd_stage = 0; /* index into flat_tw_offset */
    for (int len = 2; len <= N; len <<= 1) {
        int half = len >> 1;
        int step = N / len;

#if SIMD
        if (half >= 4) {
            /* Use pre-flattened twiddle table: sequential load, no gather */
            const float *ftw_re = flat_tw_re + flat_tw_offset[simd_stage];
            const float *ftw_im = flat_tw_im + flat_tw_offset[simd_stage];
            simd_stage++;

            for (int i = 0; i < N; i += len) {
                int k = 0;
                for (; k + 3 < half; k += 4) {
                    v128_t vwr = wasm_v128_load(ftw_re + k);
                    v128_t vwi = wasm_v128_load(ftw_im + k);
                    if (inverse) vwi = wasm_f32x4_neg(vwi);

                    v128_t vre_b = wasm_v128_load(re + i + k + half);
                    v128_t vim_b = wasm_v128_load(im + i + k + half);
                    v128_t vre_a = wasm_v128_load(re + i + k);
                    v128_t vim_a = wasm_v128_load(im + i + k);

                    /* tr = re[b]*wr - im[b]*wi */
                    v128_t vtr = wasm_f32x4_sub(
                        wasm_f32x4_mul(vre_b, vwr),
                        wasm_f32x4_mul(vim_b, vwi));
                    /* ti = re[b]*wi + im[b]*wr */
                    v128_t vti = wasm_f32x4_add(
                        wasm_f32x4_mul(vre_b, vwi),
                        wasm_f32x4_mul(vim_b, vwr));

                    wasm_v128_store(re + i + k + half, wasm_f32x4_sub(vre_a, vtr));
                    wasm_v128_store(im + i + k + half, wasm_f32x4_sub(vim_a, vti));
                    wasm_v128_store(re + i + k, wasm_f32x4_add(vre_a, vtr));
                    wasm_v128_store(im + i + k, wasm_f32x4_add(vim_a, vti));
                }
                for (; k < half; k++) {
                    float wr = ftw_re[k];
                    float wi = inverse ? -ftw_im[k] : ftw_im[k];
                    int a = i + k, b = a + half;
                    float tr = re[b] * wr - im[b] * wi;
                    float ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr; im[b] = im[a] - ti;
                    re[a] = re[a] + tr; im[a] = im[a] + ti;
                }
            }
        } else
#endif
        {
            /* Scalar: small stages (half < 4: len=2,4) */
            for (int i = 0; i < N; i += len) {
                for (int k = 0; k < half; k++) {
                    int tw_idx = k * step;
                    float wr = tw_re[tw_idx];
                    float wi = inverse ? -tw_im[tw_idx] : tw_im[tw_idx];

                    int a = i + k;
                    int b = a + half;

                    float tr = re[b] * wr - im[b] * wi;
                    float ti = re[b] * wi + im[b] * wr;

                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] = re[a] + tr;
                    im[a] = im[a] + ti;
                }
            }
        }
    }

    if (inverse) {
        /* SIMD scaling: N=256, always multiple of 4 */
        fe_vec_scale(re, 1.0f / N, N);
        fe_vec_scale(im, 1.0f / N, N);
    }
}

/* ------------------------------------------------------------------ */
/*  Real FFT: N real → N/2+1 complex (half-complex trick)             */
/*  No runtime cos/sin: uses pre-computed post_re/post_im tables.     */
/* ------------------------------------------------------------------ */

void fe_rfft(const float *in, float *re_out, float *im_out, int n) {
    (void)n;

    float zr[HALF_N], zi[HALF_N];

    /* Pack: z[k] = x[2k] + j*x[2k+1]
     * SIMD deinterleave: load 8 floats (4 complex pairs), shuffle to separate re/im */
#if SIMD
    {
        int k = 0;
        for (; k + 3 < HALF_N; k += 4) {
            v128_t v0 = wasm_v128_load(in + k * 2);     /* r0,i0,r1,i1 */
            v128_t v1 = wasm_v128_load(in + k * 2 + 4); /* r2,i2,r3,i3 */
            wasm_v128_store(zr + k, wasm_i32x4_shuffle(v0, v1, 0, 2, 4, 6));
            wasm_v128_store(zi + k, wasm_i32x4_shuffle(v0, v1, 1, 3, 5, 7));
        }
        for (; k < HALF_N; k++) {
            zr[k] = in[2 * k];
            zi[k] = in[2 * k + 1];
        }
    }
#else
    for (int k = 0; k < HALF_N; k++) {
        zr[k] = in[2 * k];
        zi[k] = in[2 * k + 1];
    }
#endif

    fft_half(zr, zi, 0);

    /* Unpack: DC and Nyquist */
    re_out[0] = zr[0] + zi[0];
    im_out[0] = 0.0f;
    re_out[HALF_N] = zr[0] - zi[0];
    im_out[HALF_N] = 0.0f;

    /* Unpack: k = 1..HALF_N-1 — all from pre-computed table */
    for (int k = 1; k < HALF_N; k++) {
        int nk = HALF_N - k;

        float ar = zr[k], ai = zi[k];
        float br = zr[nk], bi = -zi[nk];

        float er = 0.5f * (ar + br);
        float ei = 0.5f * (ai + bi);
        float or_ = 0.5f * (ar - br);
        float oi = 0.5f * (ai - bi);

        float wr = post_re[k];
        float wi = post_im[k];
        float tr = wr * or_ - wi * oi;
        float ti = wr * oi + wi * or_;

        re_out[k] = er + ti;
        im_out[k] = ei - tr;
    }
}

/* ------------------------------------------------------------------ */
/*  Real iFFT: N/2+1 complex → N real                                */
/* ------------------------------------------------------------------ */

void fe_irfft(const float *re_in, const float *im_in, float *out, int n) {
    (void)n;

    float zr[HALF_N], zi[HALF_N];

    zr[0] = 0.5f * (re_in[0] + re_in[HALF_N]);
    zi[0] = 0.5f * (re_in[0] - re_in[HALF_N]);

    for (int k = 1; k < HALF_N; k++) {
        int nk = HALF_N - k;

        float xr = re_in[k], xi = im_in[k];
        float yr = re_in[nk], yi = im_in[nk];

        float er = 0.5f * (xr + yr);
        float ei = 0.5f * (xi - yi);
        float or_ = 0.5f * (xr - yr);
        float oi = 0.5f * (xi + yi);

        /* Inverse twiddle from table */
        float wr = post_re[k];
        float wi = -post_im[k]; /* conjugate */
        float tr = wr * or_ - wi * oi;
        float ti = wr * oi + wi * or_;

        zr[k] = er - ti;
        zi[k] = ei + tr;
    }

    fft_half(zr, zi, 1);

    /* Unpack: out[2k] = zr[k], out[2k+1] = zi[k]
     * SIMD interleave */
#if SIMD
    {
        int k = 0;
        for (; k + 3 < HALF_N; k += 4) {
            v128_t vr = wasm_v128_load(zr + k);   /* r0,r1,r2,r3 */
            v128_t vi = wasm_v128_load(zi + k);   /* i0,i1,i2,i3 */
            /* Interleave: [r0,i0,r1,i1] and [r2,i2,r3,i3] */
            v128_t lo = wasm_i32x4_shuffle(vr, vi, 0, 4, 1, 5);
            v128_t hi = wasm_i32x4_shuffle(vr, vi, 2, 6, 3, 7);
            wasm_v128_store(out + k * 2, lo);
            wasm_v128_store(out + k * 2 + 4, hi);
        }
        for (; k < HALF_N; k++) {
            out[2 * k]     = zr[k];
            out[2 * k + 1] = zi[k];
        }
    }
#else
    for (int k = 0; k < HALF_N; k++) {
        out[2 * k]     = zr[k];
        out[2 * k + 1] = zi[k];
    }
#endif
}
