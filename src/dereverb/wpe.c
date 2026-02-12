/*
 * wpe.c — Recursive Least Squares Weighted Prediction Error (mono)
 *
 * Single-channel (D=1) simplification of the 2sderev RLS-WPE.
 * Uses double precision complex for numerical stability.
 *
 * Algorithm per frequency bin f:
 *   1. Buffer update: shift Y_buffer, append new frame
 *   2. Get delayed frame Y_delayed
 *   3. Update regression vector Y_tilde (shift left, append delayed)
 *   4. Kalman gain: K = (1-α) * inv_R @ conj(Y_tilde) / (α * psd + Y_tilde^H @ inv_R @ conj(Y_tilde))
 *   5. Update inv_R = (1/α) * (inv_R - K @ Y_tilde^H @ inv_R)
 *   6. Intermediate estimate: X_inter = Y - G^H @ Y_tilde
 *   7. Update taps: G = G + K * conj(X_inter)
 *   8. Final estimate: X = Y - G^H @ Y_tilde
 */

#include "dereverb.h"
#include <string.h>
#include <math.h>

/* Complex multiply: (a + bi)(c + di) = (ac-bd) + (ad+bc)i */
static inline void cmul(double ar, double ai, double br, double bi,
                        double *cr, double *ci) {
    *cr = ar * br - ai * bi;
    *ci = ar * bi + ai * br;
}

/* Complex multiply-accumulate: c += a * b */
static inline void cmac(double ar, double ai, double br, double bi,
                        double *cr, double *ci) {
    *cr += ar * br - ai * bi;
    *ci += ar * bi + ai * br;
}

/* Complex conjugate multiply: a * conj(b) */
static inline void cmul_conj(double ar, double ai, double br, double bi,
                             double *cr, double *ci) {
    *cr = ar * br + ai * bi;
    *ci = ai * br - ar * bi;
}

/* Complex divide: a / b */
static inline void cdiv(double ar, double ai, double br, double bi,
                        double *cr, double *ci) {
    double d = br * br + bi * bi;
    if (d < 1e-30) d = 1e-30;
    *cr = (ar * br + ai * bi) / d;
    *ci = (ai * br - ar * bi) / d;
}

void dr_wpe_step(DrWPEState *wpe,
                 const double *Y_re, const double *Y_im,
                 const float *clean_psd,
                 double *X_re, double *X_im)
{
    const int F = DR_FREQ_BINS;     /* 257 */
    const int K = DR_WPE_TAPS;     /* 10  */
    const int D = DR_BUF_LEN;      /* delay+1 = 3 */
    const double alpha = DR_WPE_ALPHA;
    const double one_minus_alpha = 1.0 - alpha;
    const double inv_alpha = 1.0 / alpha;
    const double eps = DR_WPE_EPS;

    for (int f = 0; f < F; f++) {
        /* --- 1. Buffer update: shift left, append new frame --- */
        /* Y_buf[f, 0..D-2] = Y_buf[f, 1..D-1] */
        for (int d = 0; d < D - 1; d++) {
            wpe->Y_buf_re[f * D + d] = wpe->Y_buf_re[f * D + d + 1];
            wpe->Y_buf_im[f * D + d] = wpe->Y_buf_im[f * D + d + 1];
        }
        /* Y_buf[f, D-1] = Y_update[f] */
        wpe->Y_buf_re[f * D + D - 1] = Y_re[f];
        wpe->Y_buf_im[f * D + D - 1] = Y_im[f];

        /* --- 2. Delayed frame = Y_buf[f, 0] (oldest = delay frames ago) --- */
        double Yd_re = wpe->Y_buf_re[f * D];
        double Yd_im = wpe->Y_buf_im[f * D];

        /* --- 3. Update regression vector: shift left by 1, append delayed --- */
        /* Y_tilde[f, 0..K-2] = Y_tilde[f, 1..K-1] */
        for (int k = 0; k < K - 1; k++) {
            wpe->Y_tilde_re[f * K + k] = wpe->Y_tilde_re[f * K + k + 1];
            wpe->Y_tilde_im[f * K + k] = wpe->Y_tilde_im[f * K + k + 1];
        }
        /* Y_tilde[f, K-1] = Y_delayed */
        wpe->Y_tilde_re[f * K + K - 1] = Yd_re;
        wpe->Y_tilde_im[f * K + K - 1] = Yd_im;

        /* PSD from DNN */
        double psd = (double)clean_psd[f];

        /* --- 4. Kalman gain --- */
        /* numerator = (1-α) * inv_R @ conj(Y_tilde)  [K] complex */
        double num_re[DR_WPE_TAPS];
        double num_im[DR_WPE_TAPS];
        for (int i = 0; i < K; i++) {
            double sr = 0.0, si = 0.0;
            for (int j = 0; j < K; j++) {
                /* inv_R[f, i, j] * conj(Y_tilde[f, j]) */
                double ir = wpe->inv_R_re[f * K * K + i * K + j];
                double ii = wpe->inv_R_im[f * K * K + i * K + j];
                double yr = wpe->Y_tilde_re[f * K + j];
                double yi = -wpe->Y_tilde_im[f * K + j]; /* conjugate */
                cmac(ir, ii, yr, yi, &sr, &si);
            }
            num_re[i] = one_minus_alpha * sr;
            num_im[i] = one_minus_alpha * si;
        }

        /* denominator = α * psd + Y_tilde^H @ numerator  [scalar complex] */
        double den_re = alpha * psd;
        double den_im = 0.0;
        for (int j = 0; j < K; j++) {
            /* conj(Y_tilde[j]) * num[j] ... actually Y_tilde^H @ num */
            double yr = wpe->Y_tilde_re[f * K + j];
            double yi = wpe->Y_tilde_im[f * K + j];
            /* Y_tilde^H means conj(Y_tilde)^T, so: conj(Y_tilde[j]) * num[j] */
            cmac(yr, -yi, num_re[j], num_im[j], &den_re, &den_im);
        }

        /* K_gain = numerator / (denominator + eps) */
        double K_re[DR_WPE_TAPS], K_im[DR_WPE_TAPS];
        den_re += eps;
        for (int i = 0; i < K; i++) {
            cdiv(num_re[i], num_im[i], den_re, den_im, &K_re[i], &K_im[i]);
        }

        /* --- 5. Update inv_R --- */
        /* tmp = Y_tilde^H @ inv_R  [K] complex */
        double tmp_re[DR_WPE_TAPS], tmp_im[DR_WPE_TAPS];
        for (int j = 0; j < K; j++) {
            double sr = 0.0, si = 0.0;
            for (int i = 0; i < K; i++) {
                double yr = wpe->Y_tilde_re[f * K + i];
                double yi = -wpe->Y_tilde_im[f * K + i]; /* conjugate */
                double ir = wpe->inv_R_re[f * K * K + i * K + j];
                double ii = wpe->inv_R_im[f * K * K + i * K + j];
                cmac(yr, yi, ir, ii, &sr, &si);
            }
            tmp_re[j] = sr;
            tmp_im[j] = si;
        }

        /* inv_R = (1/α) * (inv_R - K @ tmp)  [K×K] */
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                double kr, ki;
                cmul(K_re[i], K_im[i], tmp_re[j], tmp_im[j], &kr, &ki);
                int idx = f * K * K + i * K + j;
                wpe->inv_R_re[idx] = inv_alpha * (wpe->inv_R_re[idx] - kr);
                wpe->inv_R_im[idx] = inv_alpha * (wpe->inv_R_im[idx] - ki);
            }
        }

        /* --- 6. Intermediate estimate: X_inter = Y - G^H @ Y_tilde --- */
        double Xinter_re = Y_re[f];
        double Xinter_im = Y_im[f];
        for (int k = 0; k < K; k++) {
            /* G^H[k] = conj(G[k]) */
            double gr = wpe->G_re[f * K + k];
            double gi = -wpe->G_im[f * K + k]; /* conjugate */
            double yr = wpe->Y_tilde_re[f * K + k];
            double yi = wpe->Y_tilde_im[f * K + k];
            double pr, pi;
            cmul(gr, gi, yr, yi, &pr, &pi);
            Xinter_re -= pr;
            Xinter_im -= pi;
        }

        /* --- 7. Update taps: G = G + K * conj(X_inter) --- */
        for (int k = 0; k < K; k++) {
            double pr, pi;
            cmul(K_re[k], K_im[k], Xinter_re, -Xinter_im, &pr, &pi);
            wpe->G_re[f * K + k] += pr;
            wpe->G_im[f * K + k] += pi;
        }

        /* --- 8. Final estimate: X = Y - G^H @ Y_tilde --- */
        double Xfinal_re = Y_re[f];
        double Xfinal_im = Y_im[f];
        for (int k = 0; k < K; k++) {
            double gr = wpe->G_re[f * K + k];
            double gi = -wpe->G_im[f * K + k];
            double yr = wpe->Y_tilde_re[f * K + k];
            double yi = wpe->Y_tilde_im[f * K + k];
            double pr, pi;
            cmul(gr, gi, yr, yi, &pr, &pi);
            Xfinal_re -= pr;
            Xfinal_im -= pi;
        }

        X_re[f] = Xfinal_re;
        X_im[f] = Xfinal_im;
    }
}
