/*
 * fe_gru.c — GRU (Gated Recurrent Unit), single time-step
 *
 * PyTorch GRU equations (batch_first=False):
 *   r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)   — reset gate
 *   z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)   — update gate
 *   n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn)) — new gate
 *   h' = (1 - z) * n + z * h
 *
 * Weight layout (PyTorch convention — gate order is [r, z, n]):
 *   W_ih: [3*hidden, input]  — rows: [W_ir; W_iz; W_in]
 *   W_hh: [3*hidden, hidden] — rows: [W_hr; W_hz; W_hn]
 *
 * Optimization: batch all freq bins into 2 matmuls, then fuse gate
 * computation into a single pass (r, z, n, h' all at once per element).
 */
#include "fastenhancer.h"
#include <math.h>

#define GRU_BATCH_SIZE  (FE_F2 * FE_GRU_GATES * FE_GRU_DIM)

void fe_gru_step(const FeGRU *g, const float *x, float *h,
                 float *scratch, int freq) {
    const int D = g->hidden_size;
    const int D3 = D * 3;

    /* Batch matmul: IH[freq, 3D] and HH[freq, 3D] */
    float ih_batch[GRU_BATCH_SIZE];
    float hh_batch[GRU_BATCH_SIZE];

    fe_matmul_bias(x, g->W_ih, g->b_ih, ih_batch, freq, D3, D);
    fe_matmul_bias(h, g->W_hh, g->b_hh, hh_batch, freq, D3, D);

    /* Fused gate computation: r, z, n, h' in a single pass per bin.
     * No separate r[] and z[] arrays needed — compute and consume inline.
     * Gate offsets pre-computed outside inner loop. */
    const int D2 = D << 1;
    for (int f = 0; f < freq; f++) {
        float *hf = h + f * D;
        const float *ih = ih_batch + f * D3;
        const float *hh = hh_batch + f * D3;
        const float *ih_z = ih + D;   /* update gate offset */
        const float *hh_z = hh + D;
        const float *ih_n = ih + D2;  /* new gate offset */
        const float *hh_n = hh + D2;

        for (int i = 0; i < D; i++) {
            /* Reset gate */
            float r = 1.0f / (1.0f + expf(-(ih[i] + hh[i])));
            /* Update gate */
            float z = 1.0f / (1.0f + expf(-(ih_z[i] + hh_z[i])));
            /* New gate */
            float nv = tanhf(ih_n[i] + r * hh_n[i]);
            /* Hidden state update: h' = n + z * (h - n) — one less multiply */
            hf[i] = nv + z * (hf[i] - nv);
        }
    }
}
