/*
 * fastenhancer.c — Full model inference pipeline
 *
 * Streaming: one frame (256 samples) in → one frame out.
 *
 * Pipeline:
 *   STFT → power compress → encoder → rnnformer → decoder →
 *   complex mask → power uncompress → iSTFT
 *
 * Optimizations:
 *   - Pointer swap in encoder loop (no memcpy)
 *   - RNNFormer: eliminated redundant copies in GRU path
 *   - Complex mask: combined deinterleave + mask in one pass
 *   - Direct memcpy instead of fe_vec_copy for known sizes
 */
#include "fastenhancer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#define SIMD 1
#else
#define SIMD 0
#endif

/* ------------------------------------------------------------------ */
/*  Create / Destroy                                                   */
/* ------------------------------------------------------------------ */

FeState *fe_create(const FeWeights *w) {
    (void)w;
    FeState *s = (FeState *)calloc(1, sizeof(FeState));
    if (!s) return NULL;

    fe_fft_init();
    fe_stft_init(s);

    /* calloc already zeroed everything — no extra memset needed */
    s->initialized = 1;
    return s;
}

void fe_destroy(FeState *s) {
    if (s) free(s);
}

/* ------------------------------------------------------------------ */
/*  apply_freq_linear: [C, F_in] → [C, F_out]                        */
/*  Per-channel: out[c,fo] = sum_fi weight[fo,fi] * in[c,fi]         */
/* ------------------------------------------------------------------ */

static void apply_freq_linear(const FeLinear *lin,
                              const float *in, float *out,
                              int channels, int f_in, int f_out) {
    /* This IS a matmul: C[channels, f_out] = A[channels, f_in] @ W[f_out, f_in]^T
     * fe_matmul does exactly this with SIMD dot products. */
    fe_matmul(in, lin->weight, out, channels, f_out, f_in);
}

/* ------------------------------------------------------------------ */
/*  Fast matrix transpose for small dimensions                         */
/*  Transposes [rows, cols] → [cols, rows]                            */
/* ------------------------------------------------------------------ */

static void transpose(const float *in, float *out, int rows, int cols) {
#if SIMD
    /* 4×4 block transpose using SIMD shuffle.
     * Process in 4×4 tiles; handle remainder with scalar. */
    const int r4 = rows & ~3;
    const int c4 = cols & ~3;

    for (int r = 0; r < r4; r += 4) {
        int c = 0;
        for (; c < c4; c += 4) {
            /* Load 4 rows of 4 elements */
            v128_t row0 = wasm_v128_load(in + (r+0) * cols + c);
            v128_t row1 = wasm_v128_load(in + (r+1) * cols + c);
            v128_t row2 = wasm_v128_load(in + (r+2) * cols + c);
            v128_t row3 = wasm_v128_load(in + (r+3) * cols + c);

            /* Transpose 4×4:
             * Step 1: interleave pairs
             * t0 = [r0c0, r1c0, r0c1, r1c1]
             * t1 = [r0c2, r1c2, r0c3, r1c3]
             * t2 = [r2c0, r3c0, r2c1, r3c1]
             * t3 = [r2c2, r3c2, r2c3, r3c3] */
            v128_t t0 = wasm_i32x4_shuffle(row0, row1, 0, 4, 1, 5);
            v128_t t1 = wasm_i32x4_shuffle(row0, row1, 2, 6, 3, 7);
            v128_t t2 = wasm_i32x4_shuffle(row2, row3, 0, 4, 1, 5);
            v128_t t3 = wasm_i32x4_shuffle(row2, row3, 2, 6, 3, 7);

            /* Step 2: interleave quads → transposed columns
             * col0 = [r0c0, r1c0, r2c0, r3c0]
             * col1 = [r0c1, r1c1, r2c1, r3c1]
             * col2 = [r0c2, r1c2, r2c2, r3c2]
             * col3 = [r0c3, r1c3, r2c3, r3c3] */
            v128_t col0 = wasm_i32x4_shuffle(t0, t2, 0, 1, 4, 5);
            v128_t col1 = wasm_i32x4_shuffle(t0, t2, 2, 3, 6, 7);
            v128_t col2 = wasm_i32x4_shuffle(t1, t3, 0, 1, 4, 5);
            v128_t col3 = wasm_i32x4_shuffle(t1, t3, 2, 3, 6, 7);

            /* Store transposed 4×4 block */
            wasm_v128_store(out + (c+0) * rows + r, col0);
            wasm_v128_store(out + (c+1) * rows + r, col1);
            wasm_v128_store(out + (c+2) * rows + r, col2);
            wasm_v128_store(out + (c+3) * rows + r, col3);
        }
        /* Remainder columns */
        for (; c < cols; c++) {
            out[c * rows + r+0] = in[(r+0) * cols + c];
            out[c * rows + r+1] = in[(r+1) * cols + c];
            out[c * rows + r+2] = in[(r+2) * cols + c];
            out[c * rows + r+3] = in[(r+3) * cols + c];
        }
    }
    /* Remainder rows */
    for (int r = r4; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
#else
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
#endif
}

/* ------------------------------------------------------------------ */
/*  Process one frame                                                  */
/* ------------------------------------------------------------------ */

void fe_process(FeState *s, const FeWeights *w,
                const float *in, float *out) {
    /* ============================================================== */
    /* 1. STFT: audio_in[256] → spec_in[256, 2]                      */
    /* ============================================================== */
    fe_stft(s, in);

    /* Deinterleave [F, 2] → [2, F]
     * spec_in is interleaved (re, im, re, im, ...)
     * We need channels-first: [real_all, imag_all] */
    float spec_cf[2 * FE_FREQ_BINS]; /* [2, 256] */
#if SIMD
    {
        /* SIMD deinterleave: load 4 complex pairs (8 floats) at a time,
         * separate into re and im using shuffle */
        int f = 0;
        for (; f + 3 < FE_FREQ_BINS; f += 4) {
            /* Load 8 floats: [r0,i0, r1,i1, r2,i2, r3,i3] */
            v128_t v0 = wasm_v128_load(s->spec_in + f * 2);     /* r0,i0,r1,i1 */
            v128_t v1 = wasm_v128_load(s->spec_in + f * 2 + 4); /* r2,i2,r3,i3 */
            /* Shuffle to get [r0,r1,r2,r3] and [i0,i1,i2,i3] */
            v128_t re = wasm_i32x4_shuffle(v0, v1, 0, 2, 4, 6);
            v128_t im = wasm_i32x4_shuffle(v0, v1, 1, 3, 5, 7);
            wasm_v128_store(spec_cf + f, re);
            wasm_v128_store(spec_cf + FE_FREQ_BINS + f, im);
        }
        for (; f < FE_FREQ_BINS; f++) {
            spec_cf[f]                = s->spec_in[f * 2];
            spec_cf[FE_FREQ_BINS + f] = s->spec_in[f * 2 + 1];
        }
    }
#else
    for (int f = 0; f < FE_FREQ_BINS; f++) {
        spec_cf[f]                = s->spec_in[f * 2];
        spec_cf[FE_FREQ_BINS + f] = s->spec_in[f * 2 + 1];
    }
#endif

    /* ============================================================== */
    /* 2. Encoder PreNet: StridedConv1d(2→C1, k=8, s=4)              */
    /* ============================================================== */
    fe_strided_conv1d(&w->enc_pre, spec_cf, s->buf_a, FE_FREQ_BINS, FE_STRIDE);
    fe_vec_silu(s->buf_a, FE_C1 * FE_F1);

    /* Save skip connection 0 */
    memcpy(s->enc_skip[0], s->buf_a, FE_C1 * FE_F1 * sizeof(float));

    /* ============================================================== */
    /* 3. Encoder blocks: Conv1d(C1→C1, k=3) + SiLU                  */
    /* ============================================================== */
    {
        float *cur = s->buf_a, *nxt = s->buf_b;
        for (int i = 0; i < FE_ENC_BLOCKS; i++) {
            fe_conv1d(&w->enc[i], cur, nxt, FE_F1);
            fe_vec_silu(nxt, FE_C1 * FE_F1);

            memcpy(s->enc_skip[i + 1], nxt, FE_C1 * FE_F1 * sizeof(float));

            float *tmp = cur; cur = nxt; nxt = tmp;
        }
        if (cur != s->buf_a) {
            memcpy(s->buf_a, cur, FE_C1 * FE_F1 * sizeof(float));
        }
    }

    /* ============================================================== */
    /* 4. RNNFormer PreNet                                             */
    /* ============================================================== */
    float rf_temp[FE_C1 * FE_F2];
    apply_freq_linear(&w->rf_pre_lin, s->buf_a, rf_temp, FE_C1, FE_F1, FE_F2);
    fe_conv1d(&w->rf_pre_conv, rf_temp, s->rf_a, FE_F2);

    /* ============================================================== */
    /* 5. RNNFormer blocks                                             */
    /* ============================================================== */

    /* Transpose [C2, F2] → [F2, C2] */
    float rf_fc[FE_F2 * FE_C2];
    transpose(s->rf_a, rf_fc, FE_C2, FE_F2);

    for (int blk = 0; blk < FE_RF_BLOCKS; blk++) {
        const FeRNNFormerBlock *rb = &w->rf[blk];

        /* GRU path: input=rf_fc (const), updates gru_h in-place */
        fe_gru_step(&rb->gru, rf_fc, s->gru_h[blk], s->gru_scratch, FE_F2);

        /* FC on GRU output → rf_c */
        fe_matmul_bias(s->gru_h[blk], rb->rnn_fc.weight, rb->rnn_fc.bias,
                       s->rf_c, FE_F2, FE_C2, FE_C2);

        /* Residual: rf_fc += rf_c */
        fe_vec_add(rf_fc, s->rf_c, FE_F2 * FE_C2);

        /* Positional embedding (first block only) */
        if (rb->has_pe && rb->pe) {
            fe_vec_add(rf_fc, rb->pe, FE_F2 * FE_C2);
        }

        /* Attention path */
        memcpy(s->rf_b, rf_fc, FE_F2 * FE_C2 * sizeof(float));

        fe_mhsa(&rb->attn, &rb->attn_fc,
                rf_fc, s->rf_c,
                s->attn_qkv, s->attn_scores, s->attn_out,
                FE_F2);

        /* rf_fc = rf_b + rf_c (direct sum into rf_fc — no intermediate copy) */
        {
            const int len = FE_F2 * FE_C2;
            int j = 0;
#if SIMD
            for (; j + 3 < len; j += 4) {
                wasm_v128_store(rf_fc + j, wasm_f32x4_add(
                    wasm_v128_load(s->rf_b + j),
                    wasm_v128_load(s->rf_c + j)));
            }
#endif
            for (; j < len; j++)
                rf_fc[j] = s->rf_b[j] + s->rf_c[j];
        }
    }

    /* Transpose back [F2, C2] → [C2, F2] */
    transpose(rf_fc, s->rf_a, FE_F2, FE_C2);

    /* ============================================================== */
    /* 6. RNNFormer PostNet                                            */
    /* ============================================================== */
    float rf_temp2[FE_C2 * FE_F1];
    apply_freq_linear(&w->rf_post_lin, s->rf_a, rf_temp2, FE_C2, FE_F2, FE_F1);
    fe_conv1d(&w->rf_post_conv, rf_temp2, s->buf_a, FE_F1);

    /* ============================================================== */
    /* 7. Decoder blocks                                               */
    /* ============================================================== */
    for (int i = 0; i < FE_DEC_BLOCKS; i++) {
        int skip_idx = FE_ENC_BLOCKS - i;

        /* Concat [x, skip] along channel dim → [2*C1, F1] */
        memcpy(s->buf_c, s->buf_a, FE_C1 * FE_F1 * sizeof(float));
        memcpy(s->buf_c + FE_C1 * FE_F1,
               s->enc_skip[skip_idx], FE_C1 * FE_F1 * sizeof(float));

        fe_conv1d(&w->dec_1x1[i], s->buf_c, s->buf_b, FE_F1);
        fe_vec_silu(s->buf_b, FE_C1 * FE_F1);

        fe_conv1d(&w->dec_3x3[i], s->buf_b, s->buf_a, FE_F1);
        fe_vec_silu(s->buf_a, FE_C1 * FE_F1);
    }

    /* ============================================================== */
    /* 8. Decoder PostNet                                              */
    /* ============================================================== */
    memcpy(s->buf_c, s->buf_a, FE_C1 * FE_F1 * sizeof(float));
    memcpy(s->buf_c + FE_C1 * FE_F1, s->enc_skip[0], FE_C1 * FE_F1 * sizeof(float));

    fe_conv1d(&w->dec_post_1x1, s->buf_c, s->buf_b, FE_F1);
    fe_vec_silu(s->buf_b, FE_C1 * FE_F1);

    float mask_cf[2 * FE_FREQ_BINS];
    fe_conv_transpose1d(&w->dec_post_up, s->buf_b, mask_cf, FE_F1);

    /* ============================================================== */
    /* 9. Complex mask + interleave                                    */
    /*    spec_out[f,2] = spec_in[f,2] ⊗ mask_cf[2,F] (complex mul)  */
    /*    Fused: read interleaved spec_in, read channels-first mask,   */
    /*           complex mul, write interleaved spec_out.              */
    /* ============================================================== */
    {
        const float *mr = mask_cf;                /* mask real: [0..255] */
        const float *mi = mask_cf + FE_FREQ_BINS; /* mask imag: [256..511] */
        int f = 0;
#if SIMD
        for (; f + 3 < FE_FREQ_BINS; f += 4) {
            /* Load 4 complex spec_in pairs */
            v128_t s0 = wasm_v128_load(s->spec_in + f * 2);     /* sr0,si0,sr1,si1 */
            v128_t s1 = wasm_v128_load(s->spec_in + f * 2 + 4); /* sr2,si2,sr3,si3 */
            /* Deinterleave */
            v128_t sr = wasm_i32x4_shuffle(s0, s1, 0, 2, 4, 6); /* sr0,sr1,sr2,sr3 */
            v128_t si = wasm_i32x4_shuffle(s0, s1, 1, 3, 5, 7); /* si0,si1,si2,si3 */

            v128_t vmr = wasm_v128_load(mr + f);
            v128_t vmi = wasm_v128_load(mi + f);

            /* Complex mul: (sr+j*si)*(mr+j*mi) = (sr*mr-si*mi) + j*(sr*mi+si*mr) */
            v128_t out_r = wasm_f32x4_sub(wasm_f32x4_mul(sr, vmr), wasm_f32x4_mul(si, vmi));
            v128_t out_i = wasm_f32x4_add(wasm_f32x4_mul(sr, vmi), wasm_f32x4_mul(si, vmr));

            /* Interleave back to [r0,i0,r1,i1,...] */
            v128_t lo = wasm_i32x4_shuffle(out_r, out_i, 0, 4, 1, 5); /* r0,i0,r1,i1 */
            v128_t hi = wasm_i32x4_shuffle(out_r, out_i, 2, 6, 3, 7); /* r2,i2,r3,i3 */
            wasm_v128_store(s->spec_out + f * 2, lo);
            wasm_v128_store(s->spec_out + f * 2 + 4, hi);
        }
#endif
        for (; f < FE_FREQ_BINS; f++) {
            float sr = s->spec_in[f * 2];
            float si = s->spec_in[f * 2 + 1];
            s->spec_out[f * 2]     = sr * mr[f] - si * mi[f];
            s->spec_out[f * 2 + 1] = sr * mi[f] + si * mr[f];
        }
    }

    /* ============================================================== */
    /* 10. iSTFT                                                      */
    /* ============================================================== */
    fe_istft(s, out);
}

/* ------------------------------------------------------------------ */
/*  Weight loading from flat binary                                    */
/* ------------------------------------------------------------------ */

static const float *read_floats(const void *data, size_t *offset, int count) {
    const float *ptr = (const float *)((const char *)data + *offset);
    *offset += count * sizeof(float);
    return ptr;
}

static void load_conv1d(FeConv1d *c, const void *data, size_t *off,
                        int in_ch, int out_ch, int kernel) {
    c->in_ch = in_ch;
    c->out_ch = out_ch;
    c->kernel = kernel;
    c->weight = read_floats(data, off, out_ch * in_ch * kernel);
    c->bias   = read_floats(data, off, out_ch);
}

static void load_convt1d(FeConvT1d *c, const void *data, size_t *off,
                         int in_ch, int out_ch, int kernel, int stride) {
    c->in_ch = in_ch;
    c->out_ch = out_ch;
    c->kernel = kernel;
    c->stride = stride;
    c->weight = read_floats(data, off, in_ch * out_ch * kernel);
    c->bias   = read_floats(data, off, out_ch);
}

static void load_linear(FeLinear *l, const void *data, size_t *off,
                        int in_dim, int out_dim, int has_bias) {
    l->in_dim = in_dim;
    l->out_dim = out_dim;
    l->weight = read_floats(data, off, out_dim * in_dim);
    l->bias   = has_bias ? read_floats(data, off, out_dim) : NULL;
}

static void load_gru(FeGRU *g, const void *data, size_t *off, int dim) {
    g->hidden_size = dim;
    g->W_ih = read_floats(data, off, 3 * dim * dim);
    g->b_ih = read_floats(data, off, 3 * dim);
    g->W_hh = read_floats(data, off, 3 * dim * dim);
    g->b_hh = read_floats(data, off, 3 * dim);
}

int fe_load_weights(FeWeights *w, const void *data, size_t size) {
    size_t off = 0;

    load_conv1d(&w->enc_pre, data, &off, FE_STRIDE * 2, FE_C1, FE_ENC_K0 / FE_STRIDE);

    for (int i = 0; i < FE_ENC_BLOCKS; i++)
        load_conv1d(&w->enc[i], data, &off, FE_C1, FE_C1, FE_ENC_K);

    load_linear(&w->rf_pre_lin, data, &off, FE_F1, FE_F2, 0);
    load_conv1d(&w->rf_pre_conv, data, &off, FE_C1, FE_C2, 1);

    for (int i = 0; i < FE_RF_BLOCKS; i++) {
        FeRNNFormerBlock *rb = &w->rf[i];
        load_gru(&rb->gru, data, &off, FE_C2);
        load_linear(&rb->rnn_fc, data, &off, FE_C2, FE_C2, 1);
        load_linear(&rb->attn.qkv, data, &off, FE_C2, 3 * FE_C2, 0);
        load_linear(&rb->attn_fc, data, &off, FE_C2, FE_C2, 1);
        if (i == 0) {
            rb->pe = read_floats(data, &off, FE_F2 * FE_C2);
            rb->has_pe = 1;
        } else {
            rb->pe = NULL;
            rb->has_pe = 0;
        }
    }

    load_linear(&w->rf_post_lin, data, &off, FE_F2, FE_F1, 0);
    load_conv1d(&w->rf_post_conv, data, &off, FE_C2, FE_C1, 1);

    for (int i = 0; i < FE_DEC_BLOCKS; i++) {
        load_conv1d(&w->dec_1x1[i], data, &off, 2 * FE_C1, FE_C1, 1);
        load_conv1d(&w->dec_3x3[i], data, &off, FE_C1, FE_C1, FE_ENC_K);
    }

    load_conv1d(&w->dec_post_1x1, data, &off, 2 * FE_C1, FE_C1, 1);
    load_convt1d(&w->dec_post_up, data, &off, FE_C1, 2, FE_ENC_K0, FE_STRIDE);

    if (off > size) return -1;
    return 0;
}
