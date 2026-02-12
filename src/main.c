/*
 * main.c — Unified audio processing pipeline
 *
 * Pipeline: HPF → Input AGC → Dereverb → Denoise → Output AGC
 *
 * This replaces the old api.c with a full audio processing chain.
 * All modules are optional and can be toggled at runtime.
 */

#include "main.h"
#include "denoise/fastenhancer.h"
#include "agc/webrtc_agc.h"
#include "dereverb/dereverb.h"

#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

/* ================================================================== */
/*  Global state                                                        */
/* ================================================================== */

/* Denoise engine */
static FeWeights g_fe_weights;
static FeState  *g_fe_state = NULL;

/* Dereverb engine */
static DrWeights g_derev_weights;
static DrState  *g_derev_state = NULL;

/* AGC instances */
static WebRtcAgcState g_agc_in;
static WebRtcAgcState g_agc_out;

/* HPF state */
static float g_hpf_x1 = 0.0f, g_hpf_x2 = 0.0f;
static float g_hpf_y1 = 0.0f, g_hpf_y2 = 0.0f;

/* Feature enable flags */
static int g_hpf_enabled        = 1;  /* on by default */
static int g_agc_input_enabled  = 0;
static int g_agc_output_enabled = 0;
static int g_dereverb_enabled   = 0;

/* Intermediate buffer for pipeline stages */
static float g_tmp_buf[PIPE_FRAME_SIZE];

/* AGC sub-frame buffer (256→160+96 split processing) */
static float g_agc_carry[PIPE_AGC_FRAME];  /* carry-over buffer */
static int   g_agc_carry_in  = 0;          /* samples in carry (input AGC)  */
static int   g_agc_carry_out = 0;          /* samples in carry (output AGC) */

/* ================================================================== */
/*  HPF: 80 Hz Butterworth @ 16 kHz                                     */
/* ================================================================== */

static void hpf_process(float *buf, int len) {
    float x1 = g_hpf_x1, x2 = g_hpf_x2;
    float y1 = g_hpf_y1, y2 = g_hpf_y2;
    for (int i = 0; i < len; i++) {
        float x0 = buf[i];
        float y0 = HPF_B0*x0 + HPF_B1*x1 + HPF_B2*x2
                  - HPF_A1*y1 - HPF_A2*y2;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
        buf[i] = y0;
    }
    g_hpf_x1 = x1; g_hpf_x2 = x2;
    g_hpf_y1 = y1; g_hpf_y2 = y2;
}

/* ================================================================== */
/*  AGC helper: process arbitrary length by 160-sample sub-frames       */
/*                                                                       */
/*  WebRTC AGC needs exactly 160 samples (10ms @ 16kHz).                */
/*  Our frame is 256 samples. Strategy:                                  */
/*    - Maintain a carry buffer of remaining samples                     */
/*    - Process complete 160-sample chunks                               */
/* ================================================================== */

static void agc_process_flexible(WebRtcAgcState *agc,
                                  float *buf, int len,
                                  int *carry_count)
{
    /* We process 160 samples at a time in a scratch buffer */
    float scratch[PIPE_AGC_FRAME];
    int pos = 0;

    /* If there are carried-over samples from last call, prepend them */
    if (*carry_count > 0) {
        int need = PIPE_AGC_FRAME - *carry_count;
        if (need > len) need = len;

        /* Copy carry + new data into scratch */
        memcpy(scratch, g_agc_carry, *carry_count * sizeof(float));
        memcpy(scratch + *carry_count, buf, need * sizeof(float));

        int total = *carry_count + need;
        if (total >= PIPE_AGC_FRAME) {
            webrtc_agc_process(agc, scratch, PIPE_AGC_FRAME);
            /* Write back processed samples to buf */
            memcpy(buf, scratch + *carry_count, need * sizeof(float));
            pos = need;
            *carry_count = 0;
        } else {
            /* Still not enough — just accumulate */
            memcpy(g_agc_carry + *carry_count, buf, need * sizeof(float));
            *carry_count = total;
            return;
        }
    }

    /* Process remaining complete 160-sample frames */
    while (pos + PIPE_AGC_FRAME <= len) {
        webrtc_agc_process(agc, buf + pos, PIPE_AGC_FRAME);
        pos += PIPE_AGC_FRAME;
    }

    /* Save leftover to carry buffer */
    int leftover = len - pos;
    if (leftover > 0) {
        memcpy(g_agc_carry, buf + pos, leftover * sizeof(float));
        *carry_count = leftover;
    }
}

/* ================================================================== */
/*  Public API                                                          */
/* ================================================================== */

EXPORT int fe_init(const void *denoise_data, int denoise_size,
                   const void *dereverb_data, int dereverb_size)
{
    if (g_fe_state) return 0; /* already initialized */

    /* Load denoise weights */
    if (fe_load_weights(&g_fe_weights, denoise_data, (size_t)denoise_size) != 0)
        return -1;

    g_fe_state = fe_create(&g_fe_weights);
    if (!g_fe_state) return -1;

    /* Load dereverb weights if provided */
    if (dereverb_data && dereverb_size > 0) {
        if (derev_load_weights(&g_derev_weights, dereverb_data,
                                (size_t)dereverb_size) == 0) {
            g_derev_state = derev_create(&g_derev_weights);
            if (g_derev_state) {
                g_dereverb_enabled = 1;
            }
        }
    }

    return 0;
}

EXPORT void fe_run(const float *in, float *out) {
    if (!g_fe_state) return;

    /* Copy input to working buffer */
    float *buf = g_tmp_buf;
    memcpy(buf, in, PIPE_FRAME_SIZE * sizeof(float));

    /* ① HPF */
    if (g_hpf_enabled) {
        hpf_process(buf, PIPE_FRAME_SIZE);
    }

    /* ② Input AGC */
    if (g_agc_input_enabled) {
        agc_process_flexible(&g_agc_in, buf, PIPE_FRAME_SIZE,
                             &g_agc_carry_in);
    }

    /* ③ Dereverb */
    if (g_dereverb_enabled && g_derev_state) {
        derev_process(g_derev_state, &g_derev_weights, buf, buf);
    }

    /* ④ Denoise (FastEnhancer) */
    fe_process(g_fe_state, &g_fe_weights, buf, out);

    /* ⑤ Output AGC */
    if (g_agc_output_enabled) {
        agc_process_flexible(&g_agc_out, out, PIPE_FRAME_SIZE,
                             &g_agc_carry_out);
    }
}

EXPORT void fe_free(void) {
    if (g_fe_state) {
        fe_destroy(g_fe_state);
        g_fe_state = NULL;
    }
    if (g_derev_state) {
        derev_destroy(g_derev_state);
        g_derev_state = NULL;
    }
}

/* ---- Runtime toggles ---- */

EXPORT void fe_set_hpf(int enabled) {
    g_hpf_enabled = enabled ? 1 : 0;
    if (enabled) {
        g_hpf_x1 = g_hpf_x2 = 0.0f;
        g_hpf_y1 = g_hpf_y2 = 0.0f;
    }
}

EXPORT void fe_set_input_agc(int enabled) {
    g_agc_input_enabled = enabled ? 1 : 0;
}

EXPORT void fe_set_output_agc(int enabled) {
    g_agc_output_enabled = enabled ? 1 : 0;
}

EXPORT void fe_set_dereverb(int enabled) {
    g_dereverb_enabled = enabled ? 1 : 0;
}

/* ---- AGC configuration ---- */

EXPORT void fe_agc_init(void) {
    webrtc_agc_init(&g_agc_in);
    webrtc_agc_init(&g_agc_out);
    g_agc_carry_in = 0;
    g_agc_carry_out = 0;
}

EXPORT void fe_set_input_agc_compression(int gain_db) {
    webrtc_agc_set_compression_gain_db(&g_agc_in, gain_db);
}

EXPORT void fe_set_output_agc_compression(int gain_db) {
    webrtc_agc_set_compression_gain_db(&g_agc_out, gain_db);
}
