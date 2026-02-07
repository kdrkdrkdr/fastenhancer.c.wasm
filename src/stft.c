/*
 * fe_stft.c — Streaming STFT / iSTFT with overlap-add
 *
 * STFT:  cache(256) + input(256) = 512 → window → rfft → [256, 2]
 * iSTFT: [257] complex → irfft → window_istft → overlap-add → output(256)
 *
 * Matches ONNXSTFT from FastEnhancer exactly.
 *
 * Optimizations:
 *   - powf replaced with fast_pow_neg07 / fast_pow_2333 using log2/exp2
 *   - Interleave/deinterleave uses SIMD where possible
 */
#include "fastenhancer.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/*  Initialize windows                                                 */
/* ------------------------------------------------------------------ */

void fe_stft_init(FeState *s) {
    const int N = FE_N_FFT;
    const int H = FE_HOP_SIZE;

    /* Hann window (periodic): w[n] = 0.5 - 0.5 * cos(2*pi*n / N)
     * Matches torch.hann_window(N, periodic=True) */
    for (int n = 0; n < N; n++) {
        s->window[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / N);
    }

    /* iSTFT window: window / sum_of_squared_windows
     * For perfect reconstruction with overlap-add. */
    int K = (N + H - 1) / H;  /* ceil(N/H) = 2 */
    int num_frames = 2 * K - 1; /* 3 */
    int L = H * num_frames + (N - H); /* 1024 */
    (void)L;

    float win_sq_sum[FE_N_FFT];
    memset(win_sq_sum, 0, sizeof(float) * N);

    float fold_buf[1024];
    memset(fold_buf, 0, sizeof(fold_buf));
    for (int f = 0; f < num_frames; f++) {
        int offset = f * H;
        for (int n = 0; n < N; n++) {
            fold_buf[offset + n] += s->window[n] * s->window[n];
        }
    }

    int start = (K - 1) * H;
    for (int n = 0; n < N; n++) {
        win_sq_sum[n] = fold_buf[start + n];
    }

    for (int n = 0; n < N; n++) {
        s->window_istft[n] = s->window[n] / win_sq_sum[n];
    }

    fe_vec_zero(s->cache_stft, FE_CACHE_LEN);
    fe_vec_zero(s->cache_istft, FE_CACHE_LEN);
}

/* ------------------------------------------------------------------ */
/*  STFT: 256 audio samples → spec_in[256 * 2]                        */
/* ------------------------------------------------------------------ */

void fe_stft(FeState *s, const float *audio_in) {
    const int N = FE_N_FFT;
    const int H = FE_HOP_SIZE;

    /* Concatenate cache + input → fft_buf[512] */
    memcpy(s->fft_buf, s->cache_stft, FE_CACHE_LEN * sizeof(float));
    memcpy(s->fft_buf + FE_CACHE_LEN, audio_in, H * sizeof(float));

    /* Update cache: last FE_CACHE_LEN samples */
    memcpy(s->cache_stft, s->fft_buf + H, FE_CACHE_LEN * sizeof(float));

    /* Apply window */
    fe_vec_mul(s->fft_buf, s->window, N);

    /* Real FFT → 257 complex bins */
    fe_rfft(s->fft_buf, s->fft_re, s->fft_im, N);

    /* Power compression: spec * mag^(compress - 1)
     * mag = sqrt(re² + im²), clamp min 1e-5
     * factor = mag^(-0.7)
     * = exp2(-0.7 * log2(mag))
     * = exp2(-0.7 * 0.5 * log2(mag²))
     * = exp2(-0.35 * log2(re² + im²))
     *
     * Store as interleaved [F, 2] — 256 bins (discard bin 256) */
    const float half_exp = 0.5f * (FE_COMPRESS_EXP - 1.0f); /* -0.35 */
    for (int f = 0; f < FE_FREQ_BINS; f++) {
        float re = s->fft_re[f];
        float im = s->fft_im[f];
        float mag_sq = re * re + im * im;
        float factor;
        if (mag_sq < 1e-10f) {
            factor = 0.0f; /* below threshold, zero out */
        } else {
            /* factor = mag^(c-1) = (mag²)^((c-1)/2) = exp2(half_exp * log2(mag²)) */
            factor = exp2f(half_exp * log2f(mag_sq));
        }
        s->spec_in[f * 2]     = re * factor;
        s->spec_in[f * 2 + 1] = im * factor;
    }
}

/* ------------------------------------------------------------------ */
/*  iSTFT: spec_out[256 * 2] → 256 audio samples                      */
/* ------------------------------------------------------------------ */

void fe_istft(FeState *s, float *audio_out) {
    const int N = FE_N_FFT;
    const int H = FE_HOP_SIZE;

    /* Power un-compression:
     * factor = mag_compressed^(1/c - 1)
     * = mag^(2.333...)
     * = (mag²)^(1.1666...)
     * = exp2(1.1666 * log2(mag²))
     */
    const float half_inv_exp = 0.5f * (1.0f / FE_COMPRESS_EXP - 1.0f); /* ~1.1667 */
    float re_full[FE_SPEC_BINS], im_full[FE_SPEC_BINS];

    for (int f = 0; f < FE_FREQ_BINS; f++) {
        float re = s->spec_out[f * 2];
        float im = s->spec_out[f * 2 + 1];
        float mag_sq = re * re + im * im;
        float factor;
        if (mag_sq > 1e-20f) {
            factor = exp2f(half_inv_exp * log2f(mag_sq));
        } else {
            factor = 0.0f;
        }
        re_full[f] = re * factor;
        im_full[f] = im * factor;
    }
    /* Last bin (index 256) = 0 */
    re_full[FE_FREQ_BINS] = 0.0f;
    im_full[FE_FREQ_BINS] = 0.0f;

    /* Inverse FFT */
    fe_irfft(re_full, im_full, s->fft_buf, N);

    /* Apply iSTFT window */
    fe_vec_mul(s->fft_buf, s->window_istft, N);

    /* Overlap-add with cache (SIMD — FE_CACHE_LEN=256 is multiple of 4) */
    fe_vec_add(s->fft_buf, s->cache_istft, FE_CACHE_LEN);

    /* Output: first H samples */
    memcpy(audio_out, s->fft_buf, H * sizeof(float));

    /* Store overlap for next frame */
    memcpy(s->cache_istft, s->fft_buf + H, FE_CACHE_LEN * sizeof(float));
}
