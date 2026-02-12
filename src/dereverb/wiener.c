/*
 * wiener.c — Wiener post-filter for dereverberation
 *
 * Applies a gain mask derived from smoothed speech / interference PSDs.
 *   gain = speech_psd / (speech_psd + interference_psd + eps)
 *   gain = max(gain, gmin)
 *   output = gain * input
 *
 * gmin = 10^(DR_PF_GMIN_DB / 10) = 10^(-20/10) = 0.01
 */

#include "dereverb.h"
#include <math.h>

/* Pre-computed gain floor: 10^(-20/10) = 0.01 */
static const float GMIN = 0.01f;

void dr_wiener_step(DrWienerState *pf,
                    const double *X_re, const double *X_im,
                    const float *speech_psd,
                    const float *interf_psd,
                    float *out_re, float *out_im)
{
    const int F = DR_FREQ_BINS; /* 257 */
    const float alpha_s = DR_PF_ALPHA_S;  /* 0.20 */
    const float alpha_n = DR_PF_ALPHA_N;  /* 0.20 */
    const float one_minus_alpha_s = 1.0f - alpha_s;
    const float one_minus_alpha_n = 1.0f - alpha_n;
    const float eps = DR_PF_EPS;

    for (int f = 0; f < F; f++) {
        /* Recursive PSD smoothing */
        pf->clean_psd[f]  = alpha_s * pf->clean_psd[f]
                           + one_minus_alpha_s * speech_psd[f];
        pf->interf_psd[f] = alpha_n * pf->interf_psd[f]
                           + one_minus_alpha_n * interf_psd[f];

        /* Wiener gain */
        float gain = pf->clean_psd[f]
                   / (pf->clean_psd[f] + pf->interf_psd[f] + eps);

        /* Gain floor */
        if (gain < GMIN) gain = GMIN;

        /* Apply gain (real-valued, preserves phase) */
        out_re[f] = gain * (float)X_re[f];
        out_im[f] = gain * (float)X_im[f];
    }
}
