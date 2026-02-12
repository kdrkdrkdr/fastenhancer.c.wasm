/*
 * dereverb.h — Two-stage dereverberation (2sderev) header
 *
 * Based on: Lemercier et al. "A Computationally Efficient Online DNN-based
 * Dereverberation Algorithm" (2022)
 *
 * Simplified to single-channel (D=1) for real-time WASM use.
 * Architecture:
 *   Stage 1 (DNN-WPE): LSTM(257→512) → clean PSD → RLS-WPE
 *   Stage 2 (DNN-PF):  LSTM(257→512) → speech/interference PSD → Wiener PF
 *
 * STFT: 512-point, hop=128, sqrt(hann) window
 * (Different from denoise which uses hop=256, hann window)
 */
#ifndef DEREVERB_H
#define DEREVERB_H

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/*  Compile-time constants                                               */
/* ------------------------------------------------------------------ */

/* STFT parameters (different from denoise!) */
#define DR_N_FFT        512
#define DR_HOP_SIZE     128
#define DR_WIN_SIZE     512
#define DR_FREQ_BINS    (DR_N_FFT / 2 + 1)   /* 257 */
#define DR_CACHE_LEN    (DR_N_FFT - DR_HOP_SIZE)  /* 384 */

/* LSTM parameters */
#define DR_LSTM_INPUT   DR_FREQ_BINS          /* 257 */
#define DR_LSTM_HIDDEN  512
#define DR_LSTM_GATES   4                     /* i, f, g, o */

/* WPE parameters */
#define DR_WPE_TAPS     10
#define DR_WPE_DELAY    2
#define DR_WPE_ALPHA    0.99f
#define DR_WPE_EPS      1e-3f

/* Wiener post-filter parameters */
#define DR_PF_ALPHA_S   0.20f
#define DR_PF_ALPHA_N   0.20f
#define DR_PF_GMIN_DB   (-20.0f)
#define DR_PF_EPS       1e-8f

/* Derived (mono D=1) */
#define DR_DK           (1 * DR_WPE_TAPS)    /* 10 */
#define DR_BUF_LEN      (DR_WPE_DELAY + 1)   /* 3  */

/* ------------------------------------------------------------------ */
/*  Weight structures                                                    */
/* ------------------------------------------------------------------ */

/* LSTM weights (PyTorch convention: gates = [i, f, g, o]) */
typedef struct {
    const float *W_ih;        /* [4*hidden, input] = [2048, 257] */
    const float *b_ih;        /* [4*hidden] = [2048]             */
    const float *W_hh;        /* [4*hidden, hidden] = [2048, 512] */
    const float *b_hh;        /* [4*hidden] = [2048]             */
    const float *clean_W;     /* [257, 512] — clean map linear (no bias) */
    const float *interf_W;    /* [257, 512] — interference map (PF only, NULL for WPE) */
} DrLSTMWeights;

/* Normalization stats (per-frequency z-score) */
typedef struct {
    const float *mean;        /* [257] */
    const float *std;         /* [257] */
} DrNormStats;

/* Full model weights */
typedef struct {
    DrLSTMWeights wpe_dnn;    /* WPE LSTM */
    DrNormStats   wpe_stats;  /* WPE normalization */
    DrLSTMWeights pf_dnn;     /* Post-filter LSTM */
    DrNormStats   pf_stats;   /* PF normalization */
} DrWeights;

/* ------------------------------------------------------------------ */
/*  Runtime state                                                        */
/* ------------------------------------------------------------------ */

/* LSTM cell state */
typedef struct {
    float h[DR_LSTM_HIDDEN];  /* hidden state */
    float c[DR_LSTM_HIDDEN];  /* cell state */
} DrLSTMState;

/* WPE state (mono, uses double precision for stability) */
typedef struct {
    /* Y_buffer: delay line [F, delay+1] complex = [257, 3] × 2 (re,im) */
    double Y_buf_re[DR_FREQ_BINS * DR_BUF_LEN];
    double Y_buf_im[DR_FREQ_BINS * DR_BUF_LEN];

    /* Y_tilde: regression vector [F, DK] complex = [257, 10] × 2 */
    double Y_tilde_re[DR_FREQ_BINS * DR_DK];
    double Y_tilde_im[DR_FREQ_BINS * DR_DK];

    /* inv_R_WPE: inverse covariance [F, DK, DK] complex = [257, 10, 10] × 2 */
    double inv_R_re[DR_FREQ_BINS * DR_DK * DR_DK];
    double inv_R_im[DR_FREQ_BINS * DR_DK * DR_DK];

    /* G_WPE: filter taps [F, DK] complex = [257, 10] × 2 (D=1 so DK×D = DK×1) */
    double G_re[DR_FREQ_BINS * DR_DK];
    double G_im[DR_FREQ_BINS * DR_DK];
} DrWPEState;

/* Wiener post-filter state */
typedef struct {
    float clean_psd[DR_FREQ_BINS];       /* smoothed speech PSD */
    float interf_psd[DR_FREQ_BINS];      /* smoothed interference PSD */
} DrWienerState;

/* STFT state for dereverb (separate from denoise STFT) */
typedef struct {
    float cache[DR_CACHE_LEN];           /* overlap cache (384 samples) */
    float out_cache[DR_CACHE_LEN];       /* iSTFT overlap-add buffer */
    float window[DR_N_FFT];              /* sqrt(hann) analysis window */
    float synth_window[DR_N_FFT];        /* synthesis window */
} DrSTFTState;

/* Complete dereverb state */
typedef struct {
    DrSTFTState   stft;
    DrLSTMState   wpe_lstm;
    DrLSTMState   pf_lstm;
    DrWPEState    wpe;
    DrWienerState wiener;

    /* STFT workspace */
    float fft_buf[DR_N_FFT];
    float spec_re[DR_FREQ_BINS];
    float spec_im[DR_FREQ_BINS];

    /* LSTM scratch buffers */
    float lstm_gates[DR_LSTM_GATES * DR_LSTM_HIDDEN]; /* 2048 */
    float lstm_scratch[DR_LSTM_GATES * DR_LSTM_HIDDEN];
    float lstm_input[DR_LSTM_INPUT];     /* normalized input */
    float lstm_output[DR_LSTM_HIDDEN];   /* LSTM output */
    float mask_clean[DR_FREQ_BINS];      /* clean mask */
    float mask_interf[DR_FREQ_BINS];     /* interference mask */

    /* WPE intermediate results */
    float clean_psd[DR_FREQ_BINS];       /* clean periodogram */
    double wpe_out_re[DR_FREQ_BINS];     /* WPE output (complex) */
    double wpe_out_im[DR_FREQ_BINS];

    /* Input buffer for 256→128×2 adaptation */
    float in_buf[DR_HOP_SIZE];           /* second half of previous 256-sample input */
    int   in_buf_valid;                  /* 1 if in_buf has data from previous call */

    /* Output buffer for 128×2→256 adaptation */
    float out_buf[DR_HOP_SIZE];
    int   out_buf_valid;

    int initialized;
} DrState;

/* ------------------------------------------------------------------ */
/*  Public API                                                          */
/* ------------------------------------------------------------------ */

/* Create / destroy */
DrState *derev_create(const DrWeights *w);
void     derev_destroy(DrState *s);

/* Process one 256-sample frame (same frame size as denoise).
 * Internally processes as two 128-sample hop frames.
 * in and out may be the same pointer (in-place ok). */
void     derev_process(DrState *s, const DrWeights *w,
                       const float *in, float *out);

/* Load weights from flat binary blob */
int      derev_load_weights(DrWeights *w, const void *data, size_t size);

/* ------------------------------------------------------------------ */
/*  Internal functions                                                  */
/* ------------------------------------------------------------------ */

/* lstm.c */
void dr_lstm_step(const DrLSTMWeights *weights,
                  const DrNormStats *stats,
                  const float *mag_input,  /* [257] magnitude */
                  DrLSTMState *state,
                  float *clean_mask,       /* [257] output */
                  float *interf_mask,      /* [257] output or NULL */
                  float *scratch_gates,    /* [2048] workspace */
                  float *scratch_buf,      /* [2048] workspace */
                  float *norm_buf);        /* [257] workspace */

/* wpe.c */
void dr_wpe_step(DrWPEState *wpe,
                 const double *Y_re, const double *Y_im,  /* [257] input frame */
                 const float *clean_psd,                    /* [257] from LSTM */
                 double *X_re, double *X_im);               /* [257] output */

/* wiener.c */
void dr_wiener_step(DrWienerState *pf,
                    const double *X_re, const double *X_im,    /* [257] WPE output */
                    const float *speech_psd,                    /* [257] */
                    const float *interf_psd,                    /* [257] */
                    float *out_re, float *out_im);              /* [257] output */

/* stft (uses denoise FFT) */
void dr_stft_init(DrSTFTState *s);
void dr_stft(DrSTFTState *s, const float *audio_in,
             float *spec_re, float *spec_im, float *fft_buf);
void dr_istft(DrSTFTState *s, const float *spec_re, const float *spec_im,
              float *audio_out, float *fft_buf);

#endif /* DEREVERB_H */
