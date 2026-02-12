/*
 * WebRTC Legacy Digital AGC — Pure C single-header port.
 *
 * Ported from the official WebRTC source:
 *   webrtc.googlesource.com/src/modules/audio_processing/agc/legacy/digital_agc.cc
 *
 * This is the same algorithm that runs when Chrome's autoGainControl: true
 * is enabled via getUserMedia(). Mode: kAgcModeFixedDigital.
 *
 * Usage (16kHz):
 *   #include "webrtc_agc.h"
 *
 *   WebRtcAgcState agc;
 *   webrtc_agc_init(&agc);
 *   webrtc_agc_process(&agc, float_buf_160, 160);  // in-place, 16kHz
 *
 * Copyright (c) 2011 The WebRTC project authors. All Rights Reserved.
 * Use of this source code is governed by a BSD-style license.
 * Ported to standalone C for Emscripten/WASM by Knoc-Go project.
 */

#ifndef WEBRTC_AGC_H
#define WEBRTC_AGC_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* ====================================================================== */
/*  AGC Modes                                                              */
/* ====================================================================== */

#define kAgcModeAdaptiveAnalog  0
#define kAgcModeAdaptiveDigital 1
#define kAgcModeFixedDigital    2

/* ====================================================================== */
/*  SPL Helper Macros (from WebRTC signal_processing_library.h)            */
/* ====================================================================== */

#define WEBRTC_SPL_MUL_16_16(a, b) \
    ((int32_t)((int16_t)(a)) * ((int16_t)(b)))

#define WEBRTC_SPL_MUL_16_U16(a, b) \
    ((int32_t)((int16_t)(a)) * ((uint16_t)(b)))

#define WEBRTC_SPL_UMUL_32_16(a, b) \
    ((uint32_t)((uint32_t)(a) * (uint16_t)(b)))

#define WEBRTC_SPL_SHIFT_W32(x, c) \
    (((c) >= 0) ? ((x) * (1 << (c))) : ((x) >> (-(c))))

#define WEBRTC_SPL_ABS_W32(a) \
    (((a) < 0) ? -(a) : (a))

#define WEBRTC_SPL_MAX(A, B) (((A) > (B)) ? (A) : (B))
#define WEBRTC_SPL_MIN(A, B) (((A) < (B)) ? (A) : (B))

/* AGC-specific macros from digital_agc.cc */
#define AGC_MUL32(A, B) \
    (((B) >> 13) * (A) + (((0x0001FFF & (B)) * (A)) >> 13))

#define AGC_SCALEDIFF32(A, B, C) \
    ((C) + ((B) >> 16) * (A) + (((0x0000FFFF & (B)) * (A)) >> 16))

/* ====================================================================== */
/*  SPL Helper Functions (inlined)                                         */
/* ====================================================================== */

static inline int webrtc_spl_norm_u32(uint32_t v) {
    if (v == 0) return 0;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(v);
#else
    int n = 0;
    if (!(v & 0xFFFF0000)) { n += 16; v <<= 16; }
    if (!(v & 0xFF000000)) { n += 8;  v <<= 8;  }
    if (!(v & 0xF0000000)) { n += 4;  v <<= 4;  }
    if (!(v & 0xC0000000)) { n += 2;  v <<= 2;  }
    if (!(v & 0x80000000)) { n += 1; }
    return n;
#endif
}

static inline int webrtc_spl_norm_w32(int32_t v) {
    if (v == 0) return 0;
    if (v < 0) {
        return (v == (int32_t)0x80000000) ? 0 :
               webrtc_spl_norm_u32((uint32_t)(~v)) - 1;
    }
    return webrtc_spl_norm_u32((uint32_t)v) - 1;
}

static inline int32_t webrtc_spl_div_w32w16(int32_t num, int16_t den) {
    if (den == 0) return 0x7FFFFFFF;
    return num / den;
}

static inline int16_t webrtc_spl_div_w32w16_res_w16(int32_t num, int16_t den) {
    if (den == 0) return 0x7FFF;
    int32_t r = num / den;
    if (r > 32767) return 32767;
    if (r < -32768) return -32768;
    return (int16_t)r;
}

static inline int16_t webrtc_spl_add_sat_w16(int16_t a, int16_t b) {
    int32_t s = (int32_t)a + (int32_t)b;
    if (s > 32767) return 32767;
    if (s < -32768) return -32768;
    return (int16_t)s;
}

/* Integer square root — bit-by-bit method (matches WebRtcSpl_SqrtFloor) */
static inline int32_t webrtc_spl_sqrt(int32_t value) {
    if (value <= 0) return 0;
    uint32_t v = (uint32_t)value;
    uint32_t result = 0;
    uint32_t bit = 1u << 30;
    while (bit > v) bit >>= 2;
    while (bit != 0) {
        if (v >= result + bit) {
            v -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    return (int32_t)result;
}

/* ====================================================================== */
/*  Downsample by 2 (allpass polyphase from WebRTC)                        */
/* ====================================================================== */

/* Allpass filter coefficients from WebRTC kResampleAllpass* */
static const int16_t kAllpass1[3] = { 821, 6110, 12382 };
static const int16_t kAllpass2[3] = { 3050, 9368, 15063 };

static inline void webrtc_spl_down_by_2(
    const int16_t* in, int len, int16_t* out, int32_t* state)
{
    int32_t tmp0, tmp1, diff;
    int i;
    int len2 = len >> 1;

    for (i = 0; i < len2; i++) {
        /* upper allpass: process in[2*i] */
        tmp0 = ((int32_t)in[2 * i] << 15) + (1 << 14);
        diff = tmp0 - state[1];
        tmp1 = WEBRTC_SPL_MUL_16_16(kAllpass2[0], (int16_t)(diff >> 16))
             + (((int32_t)kAllpass2[0] * (0xFFFF & diff)) >> 16);
        tmp1 += state[0];
        state[0] = tmp0;
        diff = tmp1 - state[3];
        tmp0 = WEBRTC_SPL_MUL_16_16(kAllpass2[1], (int16_t)(diff >> 16))
             + (((int32_t)kAllpass2[1] * (0xFFFF & diff)) >> 16);
        tmp0 += state[2];
        state[2] = tmp1;
        diff = tmp0 - state[5];
        tmp1 = WEBRTC_SPL_MUL_16_16(kAllpass2[2], (int16_t)(diff >> 16))
             + (((int32_t)kAllpass2[2] * (0xFFFF & diff)) >> 16);
        tmp1 += state[4];
        state[4] = tmp0;
        state[5] = tmp1;

        /* lower allpass: process in[2*i + 1] */
        tmp0 = ((int32_t)in[2 * i + 1] << 15) + (1 << 14);
        diff = tmp0 - state[7];
        tmp1 = WEBRTC_SPL_MUL_16_16(kAllpass1[0], (int16_t)(diff >> 16))
             + (((int32_t)kAllpass1[0] * (0xFFFF & diff)) >> 16);
        tmp1 += state[6];
        state[6] = tmp0;
        diff = tmp1 - state[9];
        tmp0 = WEBRTC_SPL_MUL_16_16(kAllpass1[1], (int16_t)(diff >> 16))
             + (((int32_t)kAllpass1[1] * (0xFFFF & diff)) >> 16);
        tmp0 += state[8];
        state[8] = tmp1;
        diff = tmp0 - state[11];
        tmp1 = WEBRTC_SPL_MUL_16_16(kAllpass1[2], (int16_t)(diff >> 16))
             + (((int32_t)kAllpass1[2] * (0xFFFF & diff)) >> 16);
        tmp1 += state[10];
        state[10] = tmp0;
        state[11] = tmp1;

        /* output */
        out[i] = (int16_t)((state[5] + state[11] + (1 << 15)) >> 16);
    }
}

/* ====================================================================== */
/*  VAD (Voice Activity Detection) — from digital_agc.cc                   */
/* ====================================================================== */

#define kAvgDecayTime 250

typedef struct {
    int32_t downState[8];
    int16_t HPstate;
    int16_t counter;
    int16_t logRatio;           /* log( P(active) / P(inactive) ) Q10 */
    int16_t meanLongTerm;       /* Q10 */
    int32_t varianceLongTerm;   /* Q8  */
    int16_t stdLongTerm;        /* Q10 */
    int16_t meanShortTerm;      /* Q10 */
    int32_t varianceShortTerm;  /* Q8  */
    int16_t stdShortTerm;       /* Q10 */
} WebrtcAgcVad;

static inline void webrtc_agc_init_vad(WebrtcAgcVad* st) {
    memset(st, 0, sizeof(*st));
    st->meanLongTerm = 15 << 10;
    st->varianceLongTerm = 500 << 8;
    st->meanShortTerm = 15 << 10;
    st->varianceShortTerm = 500 << 8;
    st->counter = 3;
}

/*
 * Process VAD — direct port of WebRtcAgc_ProcessVad.
 * in: input samples at 16kHz, nrSamples = 160 (10ms).
 * Returns logRatio in Q10.
 */
static inline int16_t webrtc_agc_process_vad(
    WebrtcAgcVad* state, const int16_t* in, size_t nrSamples)
{
    uint32_t nrg;
    int32_t out, tmp32, tmp32b;
    uint16_t tmpU16;
    int16_t k, subfr, tmp16;
    int16_t buf1[8];
    int16_t buf2[4];
    int16_t HPstate;
    int16_t zeros, dB;
    int64_t tmp64;

    nrg = 0;
    HPstate = state->HPstate;

    for (subfr = 0; subfr < 10; subfr++) {
        /* downsample to 4 kHz */
        if (nrSamples == 160) {
            for (k = 0; k < 8; k++) {
                tmp32 = (int32_t)in[2 * k] + (int32_t)in[2 * k + 1];
                tmp32 >>= 1;
                buf1[k] = (int16_t)tmp32;
            }
            in += 16;
            webrtc_spl_down_by_2(buf1, 8, buf2, state->downState);
        } else {
            webrtc_spl_down_by_2(in, 8, buf2, state->downState);
            in += 8;
        }

        /* high pass filter and compute energy */
        for (k = 0; k < 4; k++) {
            out = buf2[k] + HPstate;
            tmp32 = 600 * out;
            HPstate = (int16_t)((tmp32 >> 10) - buf2[k]);
            nrg += out * (out / (1 << 6));
            nrg += out * (out % (1 << 6)) / (1 << 6);
        }
    }
    state->HPstate = HPstate;

    /* find number of leading zeros */
    if (!(0xFFFF0000 & nrg)) { zeros = 16; } else { zeros = 0; }
    if (!(0xFF000000 & (nrg << zeros))) { zeros += 8; }
    if (!(0xF0000000 & (nrg << zeros))) { zeros += 4; }
    if (!(0xC0000000 & (nrg << zeros))) { zeros += 2; }
    if (!(0x80000000 & (nrg << zeros))) { zeros += 1; }

    /* energy level (range {-32..30}) (Q10) */
    dB = (15 - zeros) * (1 << 11);

    if (state->counter < kAvgDecayTime) {
        state->counter++;
    }

    /* update short-term estimate of mean energy level (Q10) */
    tmp32 = state->meanShortTerm * 15 + dB;
    state->meanShortTerm = (int16_t)(tmp32 >> 4);

    /* update short-term estimate of variance in energy level (Q8) */
    tmp32 = (dB * dB) >> 12;
    tmp32 += state->varianceShortTerm * 15;
    state->varianceShortTerm = tmp32 / 16;

    /* update short-term estimate of standard deviation in energy level (Q10) */
    tmp32 = state->meanShortTerm * state->meanShortTerm;
    tmp32 = (state->varianceShortTerm << 12) - tmp32;
    state->stdShortTerm = (int16_t)webrtc_spl_sqrt(tmp32);

    /* update long-term estimate of mean energy level (Q10) */
    tmp32 = state->meanLongTerm * state->counter + dB;
    state->meanLongTerm = webrtc_spl_div_w32w16_res_w16(
        tmp32, webrtc_spl_add_sat_w16(state->counter, 1));

    /* update long-term estimate of variance in energy level (Q8) */
    tmp32 = (dB * dB) >> 12;
    tmp32 += state->varianceLongTerm * state->counter;
    state->varianceLongTerm = webrtc_spl_div_w32w16(
        tmp32, webrtc_spl_add_sat_w16(state->counter, 1));

    /* update long-term estimate of standard deviation in energy level (Q10) */
    tmp32 = state->meanLongTerm * state->meanLongTerm;
    tmp32 = (state->varianceLongTerm << 12) - tmp32;
    state->stdLongTerm = (int16_t)webrtc_spl_sqrt(tmp32);

    /* update voice activity measure (Q10) */
    tmp16 = 3 << 12;
    tmp32 = tmp16 * (int16_t)(dB - state->meanLongTerm);
    tmp32 = webrtc_spl_div_w32w16(tmp32, state->stdLongTerm);
    tmpU16 = (13 << 12);
    tmp32b = WEBRTC_SPL_MUL_16_U16(state->logRatio, tmpU16);
    tmp64 = tmp32;
    tmp64 += tmp32b >> 10;
    tmp64 >>= 6;

    if (tmp64 > 2048) {
        tmp64 = 2048;
    } else if (tmp64 < -2048) {
        tmp64 = -2048;
    }
    state->logRatio = (int16_t)tmp64;

    return state->logRatio;
}

/* ====================================================================== */
/*  Generator function table — from digital_agc.cc                         */
/*  y = log2(1 + e^x) in Q8, 128 entries                                  */
/* ====================================================================== */

static const uint16_t kGenFuncTable[128] = {
    256,   485,   786,   1126,  1484,  1849,  2217,  2586,
    2955,  3324,  3693,  4063,  4432,  4801,  5171,  5540,
    5909,  6279,  6648,  7017,  7387,  7756,  8125,  8495,
    8864,  9233,  9603,  9972,  10341, 10711, 11080, 11449,
    11819, 12188, 12557, 12927, 13296, 13665, 14035, 14404,
    14773, 15143, 15512, 15881, 16251, 16620, 16989, 17359,
    17728, 18097, 18466, 18836, 19205, 19574, 19944, 20313,
    20682, 21052, 21421, 21790, 22160, 22529, 22898, 23268,
    23637, 24006, 24376, 24745, 25114, 25484, 25853, 26222,
    26592, 26961, 27330, 27700, 28069, 28438, 28808, 29177,
    29546, 29916, 30285, 30654, 31024, 31393, 31762, 32132,
    32501, 32870, 33240, 33609, 33978, 34348, 34717, 35086,
    35456, 35825, 36194, 36564, 36933, 37302, 37672, 38041,
    38410, 38780, 39149, 39518, 39888, 40257, 40626, 40996,
    41365, 41734, 42104, 42473, 42842, 43212, 43581, 43950,
    44320, 44689, 45058, 45428, 45797, 46166, 46536, 46905
};

/* ====================================================================== */
/*  Digital AGC Core — from digital_agc.cc                                 */
/* ====================================================================== */

typedef struct {
    int32_t  capacitorSlow;
    int32_t  capacitorFast;
    int32_t  gain;
    int32_t  gainTable[32];
    int16_t  gatePrevious;
    int16_t  agcMode;
    WebrtcAgcVad vadNearend;
    WebrtcAgcVad vadFarend;
} WebrtcDigitalAgc;

/*
 * CalculateGainTable — direct port from WebRtcAgc_CalculateGainTable.
 */
static inline int32_t webrtc_agc_calculate_gain_table(
    int32_t* gainTable,
    int16_t digCompGaindB,
    int16_t targetLevelDbfs,
    uint8_t limiterEnable,
    int16_t analogTarget)
{
    uint32_t tmpU32no1, tmpU32no2, absInLevel, logApprox;
    int32_t inLevel, limiterLvl;
    int32_t tmp32, tmp32no1, tmp32no2, numFIX, den, y32;
    const uint16_t kLog10 = 54426;
    const uint16_t kLog10_2 = 49321;
    const uint16_t kLogE_1 = 23637;
    uint16_t constMaxGain;
    uint16_t tmpU16, intPart, fracPart;
    const int16_t kCompRatio = 3;
    int16_t limiterOffset = 0;
    int16_t limiterIdx, limiterLvlX;
    int16_t constLinApprox, maxGain, diffGain;
    int16_t i, tmp16, tmp16no1;
    int zeros, zerosScale;

    tmp32no1 = (digCompGaindB - analogTarget) * (kCompRatio - 1);
    tmp16no1 = analogTarget - targetLevelDbfs;
    tmp16no1 += webrtc_spl_div_w32w16_res_w16(
        tmp32no1 + (kCompRatio >> 1), kCompRatio);
    maxGain = WEBRTC_SPL_MAX(tmp16no1, (analogTarget - targetLevelDbfs));
    tmp32no1 = maxGain * kCompRatio;
    if ((digCompGaindB <= analogTarget) && (limiterEnable)) {
        limiterOffset = 0;
    }

    tmp32no1 = digCompGaindB * (kCompRatio - 1);
    diffGain = webrtc_spl_div_w32w16_res_w16(
        tmp32no1 + (kCompRatio >> 1), kCompRatio);
    if (diffGain < 0 || diffGain >= 128) {
        return -1;
    }

    limiterLvlX = analogTarget - limiterOffset;
    limiterIdx = 2 + webrtc_spl_div_w32w16_res_w16(
        (int32_t)limiterLvlX * (1 << 13), kLog10_2 / 2);
    tmp16no1 = webrtc_spl_div_w32w16_res_w16(
        limiterOffset + (kCompRatio >> 1), kCompRatio);
    limiterLvl = targetLevelDbfs + tmp16no1;

    constMaxGain = kGenFuncTable[diffGain];
    constLinApprox = 22817;
    den = WEBRTC_SPL_MUL_16_U16(20, constMaxGain);

    for (i = 0; i < 32; i++) {
        tmp16 = (int16_t)((kCompRatio - 1) * (i - 1));
        tmp32 = WEBRTC_SPL_MUL_16_U16(tmp16, kLog10_2) + 1;
        inLevel = webrtc_spl_div_w32w16(tmp32, kCompRatio);
        inLevel = (int32_t)diffGain * (1 << 14) - inLevel;
        absInLevel = (uint32_t)WEBRTC_SPL_ABS_W32(inLevel);

        intPart = (uint16_t)(absInLevel >> 14);
        fracPart = (uint16_t)(absInLevel & 0x00003FFF);

        if (intPart + 1 >= 128) {
            logApprox = kGenFuncTable[127] << 6;
        } else {
            tmpU16 = kGenFuncTable[intPart + 1] - kGenFuncTable[intPart];
            tmpU32no1 = tmpU16 * fracPart;
            tmpU32no1 += (uint32_t)kGenFuncTable[intPart] << 14;
            logApprox = tmpU32no1 >> 8;
        }

        if (inLevel < 0) {
            zeros = webrtc_spl_norm_u32(absInLevel);
            zerosScale = 0;
            if (zeros < 15) {
                tmpU32no2 = absInLevel >> (15 - zeros);
                tmpU32no2 = WEBRTC_SPL_UMUL_32_16(tmpU32no2, kLogE_1);
                if (zeros < 9) {
                    zerosScale = 9 - zeros;
                    tmpU32no1 >>= zerosScale;
                } else {
                    tmpU32no2 >>= zeros - 9;
                }
            } else {
                tmpU32no2 = WEBRTC_SPL_UMUL_32_16(absInLevel, kLogE_1);
                tmpU32no2 >>= 6;
            }
            logApprox = 0;
            if (tmpU32no2 < tmpU32no1) {
                logApprox = (tmpU32no1 - tmpU32no2) >> (8 - zerosScale);
            }
        }
        numFIX = (maxGain * constMaxGain) * (1 << 6);
        numFIX -= (int32_t)logApprox * diffGain;

        if (numFIX > (den >> 8) || -numFIX > (den >> 8)) {
            zeros = webrtc_spl_norm_w32(numFIX);
        } else {
            zeros = webrtc_spl_norm_w32(den) + 8;
        }
        numFIX *= 1 << zeros;

        tmp32no1 = WEBRTC_SPL_SHIFT_W32(den, zeros - 9);
        if (tmp32no1 == 0) tmp32no1 = 1;
        y32 = numFIX / tmp32no1;
        y32 = y32 >= 0 ? (y32 + 1) >> 1 : -(((-y32 + 1) >> 1));

        if (limiterEnable && (i < limiterIdx)) {
            tmp32 = WEBRTC_SPL_MUL_16_U16(i - 1, kLog10_2);
            tmp32 -= limiterLvl * (1 << 14);
            y32 = webrtc_spl_div_w32w16(tmp32 + 10, 20);
        }
        if (y32 > 39000) {
            tmp32 = (y32 >> 1) * kLog10 + 4096;
            tmp32 >>= 13;
        } else {
            tmp32 = y32 * kLog10 + 8192;
            tmp32 >>= 14;
        }
        tmp32 += 16 << 14;

        if (tmp32 > 0) {
            intPart = (int16_t)(tmp32 >> 14);
            fracPart = (uint16_t)(tmp32 & 0x00003FFF);
            if ((fracPart >> 13) != 0) {
                tmp16 = (2 << 14) - constLinApprox;
                tmp32no2 = (1 << 14) - fracPart;
                tmp32no2 *= tmp16;
                tmp32no2 >>= 13;
                tmp32no2 = (1 << 14) - tmp32no2;
            } else {
                tmp16 = constLinApprox - (1 << 14);
                tmp32no2 = (fracPart * tmp16) >> 13;
            }
            fracPart = (uint16_t)tmp32no2;
            gainTable[i] =
                (1 << intPart) +
                WEBRTC_SPL_SHIFT_W32((int32_t)fracPart, intPart - 14);
        } else {
            gainTable[i] = 0;
        }
    }
    return 0;
}

/*
 * InitDigital — direct port of WebRtcAgc_InitDigital.
 */
static inline void webrtc_agc_init_digital(
    WebrtcDigitalAgc* stt, int16_t agcMode)
{
    if (agcMode == kAgcModeFixedDigital) {
        stt->capacitorSlow = 0;
    } else {
        stt->capacitorSlow = 134217728; /* 0.125 * 32768^2 */
    }
    stt->capacitorFast = 0;
    stt->gain = 65536;
    stt->gatePrevious = 0;
    stt->agcMode = agcMode;
    webrtc_agc_init_vad(&stt->vadNearend);
    webrtc_agc_init_vad(&stt->vadFarend);
}

/*
 * ComputeDigitalGains — simplified from WebRtcAgc_ComputeDigitalGains.
 * Mono only, no far-end. Input: 160 samples at 16kHz.
 * Output: gains[11] (Q16, per-ms gain values).
 */
static inline int32_t webrtc_agc_compute_digital_gains(
    WebrtcDigitalAgc* stt,
    const int16_t* in,
    size_t nrSamples,
    int32_t gains[11])
{
    int32_t tmp32;
    int32_t env[10];
    int32_t max_nrg;
    int32_t cur_level;
    int32_t gain32;
    int16_t logratio;
    int16_t lower_thr, upper_thr;
    int16_t zeros_val = 0, zeros_fast, frac_val = 0;
    int16_t decay;
    int16_t gate, gain_adj;
    int16_t k;
    size_t n;
    const size_t L = 16; /* samples per ms at 16kHz */

    /* VAD for near end */
    logratio = webrtc_agc_process_vad(&stt->vadNearend, in, nrSamples);

    /* Determine decay factor depending on VAD */
    upper_thr = 1024; /* Q10 */
    lower_thr = 0;    /* Q10 */
    if (logratio > upper_thr) {
        decay = -65;
    } else if (logratio < lower_thr) {
        decay = 0;
    } else {
        tmp32 = (lower_thr - logratio) * 65;
        decay = (int16_t)(tmp32 >> 10);
    }

    /* adjust decay factor for long silence (adaptive modes only) */
    if (stt->agcMode != kAgcModeFixedDigital) {
        if (stt->vadNearend.stdLongTerm < 4000) {
            decay = 0;
        } else if (stt->vadNearend.stdLongTerm < 8096) {
            tmp32 = (stt->vadNearend.stdLongTerm - 4000) * decay;
            decay = (int16_t)(tmp32 >> 12);
        }
    }

    /* Find max amplitude per sub frame */
    for (k = 0; k < 10; k++) {
        max_nrg = 0;
        for (n = 0; n < L; n++) {
            int32_t nrg = in[k * L + n] * in[k * L + n];
            if (nrg > max_nrg) {
                max_nrg = nrg;
            }
        }
        env[k] = max_nrg;
    }

    /* Calculate gain per sub frame */
    gains[0] = stt->gain;
    for (k = 0; k < 10; k++) {
        /* Fast envelope follower (decay time ~131 ms) */
        stt->capacitorFast =
            AGC_SCALEDIFF32(-1000, stt->capacitorFast, stt->capacitorFast);
        if (env[k] > stt->capacitorFast) {
            stt->capacitorFast = env[k];
        }
        /* Slow envelope follower */
        if (env[k] > stt->capacitorSlow) {
            stt->capacitorSlow = AGC_SCALEDIFF32(
                500, (env[k] - stt->capacitorSlow), stt->capacitorSlow);
        } else {
            stt->capacitorSlow =
                AGC_SCALEDIFF32(decay, stt->capacitorSlow, stt->capacitorSlow);
        }

        /* use maximum of both capacitors as current level */
        if (stt->capacitorFast > stt->capacitorSlow) {
            cur_level = stt->capacitorFast;
        } else {
            cur_level = stt->capacitorSlow;
        }

        /* Translate signal level into gain using piecewise linear approx */
        zeros_val = webrtc_spl_norm_u32((uint32_t)cur_level);
        if (cur_level == 0) {
            zeros_val = 31;
        }
        tmp32 = (((uint32_t)cur_level << zeros_val) & 0x7FFFFFFF);
        frac_val = (int16_t)(tmp32 >> 19); /* Q12 */

        /* Interpolate between gainTable[zeros] and gainTable[zeros-1] */
        if (zeros_val > 0 && zeros_val <= 31) {
            tmp32 = (((int64_t)(stt->gainTable[zeros_val - 1] -
                                stt->gainTable[zeros_val]) *
                      frac_val) >> 12);
            gains[k + 1] = stt->gainTable[zeros_val] + tmp32;
        } else {
            gains[k + 1] = stt->gainTable[0];
        }
    }

    /* Gate processing (lower gain during absence of speech) */
    zeros_val = (zeros_val << 9) - (frac_val >> 3);
    zeros_fast = webrtc_spl_norm_u32((uint32_t)stt->capacitorFast);
    if (stt->capacitorFast == 0) {
        zeros_fast = 31;
    }
    tmp32 = (((uint32_t)stt->capacitorFast << zeros_fast) & 0x7FFFFFFF);
    zeros_fast <<= 9;
    zeros_fast -= (int16_t)(tmp32 >> 22);

    gate = 1000 + zeros_fast - zeros_val - stt->vadNearend.stdShortTerm;

    if (gate < 0) {
        stt->gatePrevious = 0;
    } else {
        tmp32 = stt->gatePrevious * 7;
        gate = (int16_t)((gate + tmp32) >> 3);
        stt->gatePrevious = gate;
    }

    if (gate > 0) {
        if (gate < 2500) {
            gain_adj = (2500 - gate) >> 5;
        } else {
            gain_adj = 0;
        }
        for (k = 0; k < 10; k++) {
            if ((gains[k + 1] - stt->gainTable[0]) > 8388608) {
                tmp32 = (gains[k + 1] - stt->gainTable[0]) >> 8;
                tmp32 *= 178 + gain_adj;
            } else {
                tmp32 = (gains[k + 1] - stt->gainTable[0]) * (178 + gain_adj);
                tmp32 >>= 8;
            }
            gains[k + 1] = stt->gainTable[0] + tmp32;
        }
    }

    /* Limit gain to avoid overload distortion */
    for (k = 0; k < 10; k++) {
        int zz = 10;
        if (gains[k + 1] > 474521559) {
            zz = 16 - webrtc_spl_norm_w32(gains[k + 1]);
        }
        gain32 = (gains[k + 1] >> zz) + 1;
        gain32 *= gain32;
        while (AGC_MUL32((env[k] >> 12) + 1, gain32) >
               WEBRTC_SPL_SHIFT_W32((int32_t)32767, 2 * (1 - zz + 10))) {
            if (gains[k + 1] > 8388607) {
                gains[k + 1] = (gains[k + 1] / 256) * 253;
            } else {
                gains[k + 1] = (gains[k + 1] * 253) / 256;
            }
            gain32 = (gains[k + 1] >> zz) + 1;
            gain32 *= gain32;
        }
    }

    /* gain reductions should be done 1 ms earlier than gain increases */
    for (k = 1; k < 10; k++) {
        if (gains[k] > gains[k + 1]) {
            gains[k] = gains[k + 1];
        }
    }

    /* save start gain for next frame */
    stt->gain = gains[10];

    return 0;
}

/*
 * ApplyDigitalGains48k — adapted from WebRtcAgc_ApplyDigitalGains.
 * Applies 11 gain values to 480 samples at 48kHz.
 * Each sub-frame = 48 samples (480/10), gains interpolated per sample.
 */
/*
 * Apply digital gains at 16kHz: 160 samples = 10 sub-frames × 16 samples.
 * gains[11] provides boundary values; linear interpolation within sub-frames.
 */
static inline void webrtc_agc_apply_digital_gains_16k(
    const int32_t gains[11],
    int16_t* buf,
    int len)
{
    const int L = 16; /* samples per sub-frame at 16kHz: 160/10 = 16 */
    int k;
    size_t n;

    for (k = 0; k < 10; k++) {
        int32_t delta = (gains[k + 1] - gains[k]) / L;
        int32_t gain32 = gains[k] * (1 << 4);
        int32_t delta_shifted = delta * (1 << 4);
        for (n = 0; n < (size_t)L; n++) {
            int idx = k * L + (int)n;
            int64_t tmp64 = ((int64_t)buf[idx]) * (gain32 >> 4);
            tmp64 >>= 16;
            if (tmp64 > 32767) {
                buf[idx] = 32767;
            } else if (tmp64 < -32768) {
                buf[idx] = -32768;
            } else {
                buf[idx] = (int16_t)tmp64;
            }
            gain32 += delta_shifted;
        }
    }

    (void)len;
}

/* ====================================================================== */
/*  High-Level Wrapper API                                                 */
/* ====================================================================== */

typedef struct {
    WebrtcDigitalAgc digital;
    int16_t  targetLevelDbfs;      /* 0-31, default 3  */
    int16_t  compressionGaindB;    /* 0-90, default 9  */
    uint8_t  limiterEnable;        /* default 1        */
    int      sampleRate;
    int16_t  tmpBuf[160];          /* float<->int16 buffer (16kHz) */
} WebRtcAgcState;

static inline void webrtc_agc_init(WebRtcAgcState* st) {
    memset(st, 0, sizeof(*st));
    st->sampleRate = 16000;
    st->targetLevelDbfs = 3;
    st->compressionGaindB = 9;
    st->limiterEnable = 1;
    webrtc_agc_init_digital(&st->digital, kAgcModeFixedDigital);
    webrtc_agc_calculate_gain_table(
        st->digital.gainTable,
        st->compressionGaindB,
        st->targetLevelDbfs,
        st->limiterEnable,
        st->targetLevelDbfs);
}

/*
 * Process 160 float samples in-place at 16kHz.
 * No downsampling needed — AGC core already operates at 16kHz.
 * 1. float -> int16
 * 2. compute gains on 160 samples
 * 3. apply gains to 160 samples
 * 4. int16 -> float
 */
static inline void webrtc_agc_process(
    WebRtcAgcState* st, float* buf, int len)
{
    int i;
    int32_t gains[11];

    /* 1. float -> int16 */
    for (i = 0; i < len; i++) {
        float s = buf[i] * 32767.0f;
        if (s > 32767.0f) s = 32767.0f;
        if (s < -32768.0f) s = -32768.0f;
        st->tmpBuf[i] = (int16_t)s;
    }

    /* 2. compute gains directly at 16kHz */
    webrtc_agc_compute_digital_gains(
        &st->digital, st->tmpBuf, 160, gains);

    /* 3. apply gains at 16kHz */
    webrtc_agc_apply_digital_gains_16k(gains, st->tmpBuf, len);

    /* 4. int16 -> float */
    for (i = 0; i < len; i++) {
        buf[i] = (float)st->tmpBuf[i] / 32767.0f;
    }
}

static inline void webrtc_agc_set_target_level_dbfs(
    WebRtcAgcState* st, int level)
{
    st->targetLevelDbfs = (int16_t)(level < 0 ? 0 : (level > 31 ? 31 : level));
    webrtc_agc_calculate_gain_table(
        st->digital.gainTable,
        st->compressionGaindB,
        st->targetLevelDbfs,
        st->limiterEnable,
        st->targetLevelDbfs);
}

static inline void webrtc_agc_set_compression_gain_db(
    WebRtcAgcState* st, int gain)
{
    st->compressionGaindB = (int16_t)(gain < 0 ? 0 : (gain > 90 ? 90 : gain));
    webrtc_agc_calculate_gain_table(
        st->digital.gainTable,
        st->compressionGaindB,
        st->targetLevelDbfs,
        st->limiterEnable,
        st->targetLevelDbfs);
}

#endif /* WEBRTC_AGC_H */
