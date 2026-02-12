/*
 * dereverb.c — Two-stage dereverberation inference pipeline
 *
 * Processes 256-sample frames by splitting into two 128-hop STFT frames.
 * Each frame goes through:
 *   1. STFT (hop=128, sqrt_hann)
 *   2. DNN-WPE: LSTM → clean PSD → RLS-WPE
 *   3. DNN-PF:  LSTM → speech/interf PSD → Wiener post-filter
 *   4. iSTFT
 *
 * The dereverb STFT (hop=128, sqrt_hann) is separate from
 * the denoise STFT (hop=256, hann).
 */

#include "dereverb.h"
#include "../denoise/fastenhancer.h"  /* for fe_rfft, fe_irfft, fe_fft_init */
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================================================================== */
/*  Dereverb STFT (sqrt_hann, hop=128)                                  */
/* ================================================================== */

void dr_stft_init(DrSTFTState *s) {
    /* sqrt(hann) window */
    for (int i = 0; i < DR_N_FFT; i++) {
        float hann = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * i / DR_N_FFT));
        s->window[i] = sqrtf(hann);
        s->synth_window[i] = s->window[i]; /* for perfect reconstruction with OLA */
    }
    memset(s->cache, 0, sizeof(s->cache));
    memset(s->out_cache, 0, sizeof(s->out_cache));
}

void dr_stft(DrSTFTState *s, const float *audio_in,
             float *spec_re, float *spec_im, float *fft_buf) {
    /* Construct windowed frame: [cache(384) | new(128)] */
    memcpy(fft_buf, s->cache, DR_CACHE_LEN * sizeof(float));
    memcpy(fft_buf + DR_CACHE_LEN, audio_in, DR_HOP_SIZE * sizeof(float));

    /* Apply analysis window */
    for (int i = 0; i < DR_N_FFT; i++) {
        fft_buf[i] *= s->window[i];
    }

    /* Update cache: shift left by hop, append new samples */
    memmove(s->cache, s->cache + DR_HOP_SIZE,
            (DR_CACHE_LEN - DR_HOP_SIZE) * sizeof(float));
    memcpy(s->cache + DR_CACHE_LEN - DR_HOP_SIZE, audio_in,
           DR_HOP_SIZE * sizeof(float));

    /* FFT */
    fe_rfft(fft_buf, spec_re, spec_im, DR_N_FFT);
}

void dr_istft(DrSTFTState *s, const float *spec_re, const float *spec_im,
              float *audio_out, float *fft_buf) {
    /* iFFT */
    fe_irfft(spec_re, spec_im, fft_buf, DR_N_FFT);

    /* Apply synthesis window */
    for (int i = 0; i < DR_N_FFT; i++) {
        fft_buf[i] *= s->synth_window[i];
    }

    /* Overlap-add with cache */
    for (int i = 0; i < DR_CACHE_LEN; i++) {
        fft_buf[i] += s->out_cache[i];
    }

    /* Output first hop_size samples */
    memcpy(audio_out, fft_buf, DR_HOP_SIZE * sizeof(float));

    /* Save overlap for next frame */
    memcpy(s->out_cache, fft_buf + DR_HOP_SIZE, DR_CACHE_LEN * sizeof(float));
}

/* ================================================================== */
/*  Weight loading                                                      */
/* ================================================================== */

/*
 * Binary layout (all float32, little-endian):
 *
 * WPE DNN:
 *   W_ih        [2048 * 257]
 *   b_ih        [2048]
 *   W_hh        [2048 * 512]
 *   b_hh        [2048]
 *   clean_W     [257 * 512]
 *   (no interf_W for WPE DNN)
 *
 * WPE Stats:
 *   mean        [257]
 *   std         [257]
 *
 * PF DNN:
 *   W_ih        [2048 * 257]
 *   b_ih        [2048]
 *   W_hh        [2048 * 512]
 *   b_hh        [2048]
 *   clean_W     [257 * 512]
 *   interf_W    [257 * 512]
 *
 * PF Stats:
 *   mean        [257]
 *   std         [257]
 */

int derev_load_weights(DrWeights *w, const void *data, size_t size) {
    const float *p = (const float *)data;
    const int F = DR_LSTM_INPUT;    /* 257 */
    const int H = DR_LSTM_HIDDEN;   /* 512 */
    const int G4H = DR_LSTM_GATES * H; /* 2048 */

    /* WPE DNN weights */
    size_t wpe_dnn_size = (size_t)(G4H * F + G4H + G4H * H + G4H + F * H);
    /* WPE stats */
    size_t wpe_stats_size = (size_t)(F + F);
    /* PF DNN weights (has both clean + interf map) */
    size_t pf_dnn_size = (size_t)(G4H * F + G4H + G4H * H + G4H + F * H + F * H);
    /* PF stats */
    size_t pf_stats_size = (size_t)(F + F);

    size_t total = (wpe_dnn_size + wpe_stats_size + pf_dnn_size + pf_stats_size) * sizeof(float);
    if (size < total) return -1;

    /* WPE DNN */
    w->wpe_dnn.W_ih = p;     p += G4H * F;
    w->wpe_dnn.b_ih = p;     p += G4H;
    w->wpe_dnn.W_hh = p;     p += G4H * H;
    w->wpe_dnn.b_hh = p;     p += G4H;
    w->wpe_dnn.clean_W = p;  p += F * H;
    w->wpe_dnn.interf_W = NULL;  /* WPE DNN has no interference map */

    /* WPE stats */
    w->wpe_stats.mean = p;   p += F;
    w->wpe_stats.std = p;    p += F;

    /* PF DNN */
    w->pf_dnn.W_ih = p;      p += G4H * F;
    w->pf_dnn.b_ih = p;      p += G4H;
    w->pf_dnn.W_hh = p;      p += G4H * H;
    w->pf_dnn.b_hh = p;      p += G4H;
    w->pf_dnn.clean_W = p;   p += F * H;
    w->pf_dnn.interf_W = p;  p += F * H;

    /* PF stats */
    w->pf_stats.mean = p;    p += F;
    w->pf_stats.std = p;     p += F;

    return 0;
}

/* ================================================================== */
/*  Create / Destroy                                                    */
/* ================================================================== */

DrState *derev_create(const DrWeights *w) {
    (void)w;
    DrState *s = (DrState *)calloc(1, sizeof(DrState));
    if (!s) return NULL;

    /* Initialize STFT */
    dr_stft_init(&s->stft);

    /* Initialize WPE inv_R to identity matrix per frequency bin */
    for (int f = 0; f < DR_FREQ_BINS; f++) {
        for (int i = 0; i < DR_DK; i++) {
            s->wpe.inv_R_re[f * DR_DK * DR_DK + i * DR_DK + i] = 1.0;
        }
    }

    s->in_buf_valid = 0;
    s->out_buf_valid = 0;
    s->initialized = 1;
    return s;
}

void derev_destroy(DrState *s) {
    if (s) free(s);
}

/* ================================================================== */
/*  Process one 128-hop STFT frame                                      */
/* ================================================================== */

static void derev_process_frame(DrState *s, const DrWeights *w,
                                const float *audio_in, float *audio_out)
{
    const int F = DR_FREQ_BINS;

    /* STFT */
    dr_stft(&s->stft, audio_in, s->spec_re, s->spec_im, s->fft_buf);

    /* Compute magnitude for LSTM input */
    float mag[DR_FREQ_BINS];
    for (int f = 0; f < F; f++) {
        mag[f] = sqrtf(s->spec_re[f] * s->spec_re[f] +
                       s->spec_im[f] * s->spec_im[f]);
    }

    /* --- Stage 1: DNN-WPE --- */

    /* LSTM: predict clean mask */
    dr_lstm_step(&w->wpe_dnn, &w->wpe_stats,
                 mag, &s->wpe_lstm,
                 s->mask_clean, NULL,  /* no interference for WPE */
                 s->lstm_gates, s->lstm_scratch, s->lstm_input);

    /* Clean periodogram = (mag * clean_mask)^2 */
    for (int f = 0; f < F; f++) {
        float clean_mag = mag[f] * s->mask_clean[f];
        s->clean_psd[f] = clean_mag * clean_mag;
    }

    /* Convert spectrum to double for WPE */
    double Y_re_d[DR_FREQ_BINS], Y_im_d[DR_FREQ_BINS];
    for (int f = 0; f < F; f++) {
        Y_re_d[f] = (double)s->spec_re[f];
        Y_im_d[f] = (double)s->spec_im[f];
    }

    /* RLS-WPE step */
    dr_wpe_step(&s->wpe, Y_re_d, Y_im_d, s->clean_psd,
                s->wpe_out_re, s->wpe_out_im);

    /* --- Stage 2: DNN-PF (Wiener Post-Filter) --- */

    /* Compute magnitude of WPE output for PF LSTM input */
    float wpe_mag[DR_FREQ_BINS];
    for (int f = 0; f < F; f++) {
        wpe_mag[f] = (float)sqrt(s->wpe_out_re[f] * s->wpe_out_re[f] +
                                 s->wpe_out_im[f] * s->wpe_out_im[f]);
    }

    /* LSTM: predict speech + interference masks */
    dr_lstm_step(&w->pf_dnn, &w->pf_stats,
                 wpe_mag, &s->pf_lstm,
                 s->mask_clean, s->mask_interf,
                 s->lstm_gates, s->lstm_scratch, s->lstm_input);

    /* Speech periodogram = (wpe_mag * speech_mask)^2 */
    /* Interference periodogram = (wpe_mag * interf_mask)^2 */
    float speech_psd[DR_FREQ_BINS], interf_psd[DR_FREQ_BINS];
    for (int f = 0; f < F; f++) {
        float sm = wpe_mag[f] * s->mask_clean[f];
        float im = wpe_mag[f] * s->mask_interf[f];
        speech_psd[f] = sm * sm;
        interf_psd[f] = im * im;
    }

    /* Wiener post-filter */
    float pf_re[DR_FREQ_BINS], pf_im[DR_FREQ_BINS];
    dr_wiener_step(&s->wiener, s->wpe_out_re, s->wpe_out_im,
                   speech_psd, interf_psd, pf_re, pf_im);

    /* Zero DC and bin 1 (as in original 2sderev) */
    pf_re[0] = 0.0f; pf_im[0] = 0.0f;
    pf_re[1] = 0.0f; pf_im[1] = 0.0f;

    /* iSTFT */
    dr_istft(&s->stft, pf_re, pf_im, audio_out, s->fft_buf);
}

/* ================================================================== */
/*  Process 256-sample frame (splits into 2 × 128-hop frames)           */
/* ================================================================== */

void derev_process(DrState *s, const DrWeights *w,
                   const float *in, float *out)
{
    /* Process first 128 samples */
    derev_process_frame(s, w, in, out);

    /* Process second 128 samples */
    derev_process_frame(s, w, in + DR_HOP_SIZE, out + DR_HOP_SIZE);
}
