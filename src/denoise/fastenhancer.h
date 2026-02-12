/*
 * FastEnhancer-Tiny — C inference engine for WASM SIMD
 * Real-time streaming speech enhancement (16 kHz, 16 ms frame)
 *
 * Model: FastEnhancer-Tiny (22 K params)
 *   n_fft=512  hop=256  channels=24  rf_channels=20  rf_freq=16
 *   encoder: StridedConv(2→24,k8s4) + 2×Conv(24,k3)
 *   rnnformer: 2×{GRU(20)+MHSA(20,4heads)}
 *   decoder: 2×{Conv(48→24,k1)+Conv(24,k3)} + ConvT(24→2,k8s4)
 */
#ifndef FASTENHANCER_H
#define FASTENHANCER_H

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/*  Compile-time model constants (FastEnhancer-Tiny)                   */
/* ------------------------------------------------------------------ */

/* Audio / STFT */
#define FE_SAMPLE_RATE   16000
#define FE_N_FFT         512
#define FE_HOP_SIZE      256
#define FE_WIN_SIZE      512
#define FE_FREQ_BINS     (FE_N_FFT / 2)       /* 256 — input spec freqs  */
#define FE_SPEC_BINS     (FE_N_FFT / 2 + 1)   /* 257 — rfft output       */
#define FE_CACHE_LEN     (FE_N_FFT - FE_HOP_SIZE) /* 256 — overlap cache */
#define FE_COMPRESS_EXP  0.3f

/* Encoder / Decoder */
#define FE_STRIDE        4
#define FE_ENC_K0        8                     /* encoder pre-net kernel   */
#define FE_ENC_K         3                     /* encoder block kernel     */

/* Model parameters */
#define FE_C1            24                    /* encoder/decoder channels */
#define FE_ENC_BLOCKS    2                     /* number of encoder blocks */
#define FE_C2            20                    /* rnnformer channels       */
#define FE_F2            16                    /* rnnformer frequency bins */
#define FE_RF_BLOCKS     2                     /* number of rnnformer blks */
#define FE_NUM_HEADS     4                     /* attention heads          */

/* Derived constants */
#define FE_F1            (FE_FREQ_BINS / FE_STRIDE)  /* 64 — post-stride freq */
#define FE_HEAD_DIM      (FE_C2 / FE_NUM_HEADS) /* 5 — per-head dim      */

/* GRU dimensions (input_size == hidden_size == C2) */
#define FE_GRU_DIM       FE_C2
#define FE_GRU_GATES     3                     /* z, r, n */

/* Decoder (mirrors encoder) */
#define FE_DEC_BLOCKS    FE_ENC_BLOCKS
#define FE_DEC_OUT_CH    2                     /* real + imag output       */

/* ------------------------------------------------------------------ */
/*  Weight structures — all BN fused, weight_norm removed              */
/* ------------------------------------------------------------------ */

/* Conv1d with fused bias: y = W @ x + b */
typedef struct {
    const float *weight;   /* [out_ch, in_ch, kernel] row-major */
    const float *bias;     /* [out_ch]                          */
    int in_ch, out_ch, kernel;
} FeConv1d;

/* ConvTranspose1d with fused bias */
typedef struct {
    const float *weight;   /* [in_ch, out_ch, kernel] row-major */
    const float *bias;     /* [out_ch]                          */
    int in_ch, out_ch, kernel, stride;
} FeConvT1d;

/* Linear (weight + optional bias) */
typedef struct {
    const float *weight;   /* [out, in] row-major */
    const float *bias;     /* [out] or NULL       */
    int in_dim, out_dim;
} FeLinear;

/* GRU (single layer, batch_first=false, hidden_size == input_size) */
typedef struct {
    const float *W_ih;     /* [3*hidden, input]  — gates: z,r,n */
    const float *b_ih;     /* [3*hidden]                         */
    const float *W_hh;     /* [3*hidden, hidden]                 */
    const float *b_hh;     /* [3*hidden]                         */
    int hidden_size;
} FeGRU;

/* Multi-head self-attention */
typedef struct {
    FeLinear qkv;          /* [3*C2, C2]  — combined Q,K,V proj */
} FeAttention;

/* Single RNNFormer block */
typedef struct {
    FeGRU       gru;
    FeLinear    rnn_fc;     /* [C2, C2] bias=True (fused BN)   */
    FeAttention attn;
    FeLinear    attn_fc;    /* [C2, C2] bias=True (fused BN)   */
    const float *pe;        /* [F2, C2] positional embedding    */
    int         has_pe;     /* 1 for first block, 0 otherwise   */
} FeRNNFormerBlock;

/* Full model weights */
typedef struct {
    /* Encoder PreNet: StridedConv(2→C1, k8s4) → reshaped to Conv(8→C1, k2) */
    FeConv1d  enc_pre;     /* in=8, out=24, k=2  (after stride reshape) */

    /* Encoder blocks ×2 */
    FeConv1d  enc[FE_ENC_BLOCKS]; /* in=24, out=24, k=3 each */

    /* RNNFormer pre-net */
    FeLinear  rf_pre_lin;  /* [F1=64, F2=16] — freq downscale (no bias) */
    FeConv1d  rf_pre_conv; /* in=C1=24, out=C2=20, k=1                 */

    /* RNNFormer blocks ×2 */
    FeRNNFormerBlock rf[FE_RF_BLOCKS];

    /* RNNFormer post-net */
    FeLinear  rf_post_lin; /* [F2=16, F1=64] — freq upscale (no bias)   */
    FeConv1d  rf_post_conv;/* in=C2=20, out=C1=24, k=1                  */

    /* Decoder blocks ×2 */
    FeConv1d  dec_1x1[FE_DEC_BLOCKS]; /* in=48, out=24, k=1 */
    FeConv1d  dec_3x3[FE_DEC_BLOCKS]; /* in=24, out=24, k=3 */

    /* Decoder post-net */
    FeConv1d  dec_post_1x1; /* in=48, out=24, k=1 */
    FeConvT1d dec_post_up;  /* in=24, out=2, k=8, s=4 */
} FeWeights;

/* ------------------------------------------------------------------ */
/*  Runtime state (pre-allocated, zero-copy)                           */
/* ------------------------------------------------------------------ */

typedef struct {
    /* STFT / iSTFT cache */
    float cache_stft[FE_CACHE_LEN];   /* 256 — previous audio samples   */
    float cache_istft[FE_CACHE_LEN];  /* 256 — overlap-add buffer       */

    /* GRU hidden states: one per RNNFormer block [F2 * C2] */
    float gru_h[FE_RF_BLOCKS][FE_F2 * FE_C2];

    /* Pre-computed Hann window */
    float window[FE_N_FFT];
    float window_istft[FE_N_FFT];     /* window / sum(window^2) */

    /* FFT workspace */
    float fft_buf[FE_N_FFT];          /* time-domain frame */
    float fft_re[FE_SPEC_BINS];       /* real part of spectrum */
    float fft_im[FE_SPEC_BINS];       /* imag part of spectrum */

    /* Intermediate buffers — reused across layers */
    /* Main computation path: [C, F] layout (channels-first) */
    float buf_a[FE_C1 * FE_F1];      /* 24 × 64 = 1536 */
    float buf_b[FE_C1 * FE_F1];      /* 24 × 64 = 1536 */
    float buf_c[(FE_C1 * 2) * FE_F1];/* 48 × 64 = 3072 (concat buf) */

    /* Encoder skip connections [ENC_BLOCKS+1][C1 * F1] */
    float enc_skip[FE_ENC_BLOCKS + 1][FE_C1 * FE_F1];

    /* RNNFormer workspace [C2 * F2] */
    float rf_a[FE_C2 * FE_F2];       /* 20 × 16 = 320  */
    float rf_b[FE_C2 * FE_F2];       /* 20 × 16 = 320  */
    float rf_c[FE_C2 * FE_F2];       /* 20 × 16 = 320  */

    /* Attention workspace */
    float attn_qkv[3 * FE_C2 * FE_F2]; /* Q,K,V projected */
    float attn_scores[FE_NUM_HEADS * FE_F2 * FE_F2]; /* 4×16×16 */
    float attn_out[FE_C2 * FE_F2];

    /* Spec I/O [F=256, 2] (real/imag interleaved) */
    float spec_in[FE_FREQ_BINS * 2];
    float spec_out[FE_FREQ_BINS * 2];

    /* GRU scratch */
    float gru_gates[FE_GRU_GATES * FE_GRU_DIM]; /* 3 × 20 = 60  */
    float gru_scratch[FE_GRU_GATES * FE_GRU_DIM];

    /* Flag */
    int initialized;
} FeState;

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/* Create / destroy engine (weights must remain valid for lifetime) */
FeState *fe_create(const FeWeights *w);
void     fe_destroy(FeState *s);

/* Process one frame: 256 samples in → 256 samples out (in-place ok) */
void     fe_process(FeState *s, const FeWeights *w,
                    const float *in, float *out);

/* Load weights from a flat binary blob */
int      fe_load_weights(FeWeights *w, const void *data, size_t size);

/* ------------------------------------------------------------------ */
/*  Internal functions (exposed for testing)                           */
/* ------------------------------------------------------------------ */

/* fft.c */
void fe_fft_init(void);
void fe_rfft(const float *in, float *re, float *im, int n);
void fe_irfft(const float *re, const float *im, float *out, int n);

/* stft.c */
void fe_stft_init(FeState *s);
void fe_stft(FeState *s, const float *audio_in);
void fe_istft(FeState *s, float *audio_out);

/* conv.c */
void fe_conv1d(const FeConv1d *c, const float *in, float *out, int freq);
void fe_conv1d_relu_inplace(float *x, int n);
void fe_strided_conv1d(const FeConv1d *c, const float *in, float *out,
                       int in_freq, int stride);
void fe_conv_transpose1d(const FeConvT1d *c, const float *in, float *out,
                         int in_freq);

/* activations.c */
void fe_silu_inplace(float *x, int n);
void fe_sigmoid(const float *in, float *out, int n);
void fe_softmax_rows(float *x, int rows, int cols);

/* gru.c */
void fe_gru_step(const FeGRU *g, const float *x, float *h,
                 float *scratch, int freq);

/* attention.c */
void fe_mhsa(const FeAttention *a, const FeLinear *fc,
             const float *in, float *out,
             float *qkv_buf, float *score_buf, float *attn_buf,
             int freq);

/* simd.c — WASM SIMD primitives */
void fe_vec_add(float *dst, const float *src, int n);
void fe_vec_mul(float *dst, const float *src, int n);
void fe_vec_scale(float *dst, float s, int n);
void fe_vec_copy(float *dst, const float *src, int n);
void fe_vec_zero(float *dst, int n);
void fe_vec_silu(float *x, int n);
void fe_matmul(const float *A, const float *B, float *C,
               int M, int N, int K);
void fe_matmul_bias(const float *A, const float *B, const float *bias,
                    float *C, int M, int N, int K);
void fe_gemv(const float *W, const float *x, const float *bias,
             float *y, int out_dim, int in_dim);

#endif /* FASTENHANCER_H */
