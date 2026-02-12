/*
 * lstm.c — Single-layer LSTM cell for dereverberation
 *
 * PyTorch LSTM gate order: [input, forget, cell_candidate, output]
 * Each gate: hidden_size = 512, total = 4 * 512 = 2048
 *
 * LSTM step:
 *   gates = W_ih @ x + b_ih + W_hh @ h + b_hh
 *   i = sigmoid(gates[0:512])
 *   f = sigmoid(gates[512:1024])
 *   g = tanh(gates[1024:1536])
 *   o = sigmoid(gates[1536:2048])
 *   c = f * c_prev + i * g
 *   h = o * tanh(c)
 *
 * Then:
 *   clean_mask  = sigmoid(clean_W @ h)
 *   interf_mask = sigmoid(interf_W @ h)   (if interf_W != NULL)
 */

#include "dereverb.h"
#include "../denoise/fastenhancer.h"  /* for fe_gemv, fe_matmul_bias */
#include <math.h>
#include <string.h>

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float tanh_f(float x) {
    return tanhf(x);
}

void dr_lstm_step(const DrLSTMWeights *weights,
                  const DrNormStats *stats,
                  const float *mag_input,   /* [257] raw magnitude */
                  DrLSTMState *state,
                  float *clean_mask,        /* [257] output */
                  float *interf_mask,       /* [257] output or NULL */
                  float *scratch_gates,     /* [2048] workspace */
                  float *scratch_buf,       /* [2048] workspace */
                  float *norm_buf)          /* [257] workspace */
{
    const int H = DR_LSTM_HIDDEN;   /* 512 */
    const int F = DR_LSTM_INPUT;    /* 257 */
    const int G4 = DR_LSTM_GATES * H; /* 2048 */
    int i;

    /* Step 1: Z-score normalize input magnitude */
    for (i = 0; i < F; i++) {
        norm_buf[i] = (mag_input[i] - stats->mean[i]) / (stats->std[i] + 1e-8f);
    }

    /* Step 2: Compute gates = W_ih @ x + b_ih + W_hh @ h + b_hh */
    /* gates = W_ih @ x + b_ih */
    fe_gemv(weights->W_ih, norm_buf, weights->b_ih, scratch_gates, G4, F);

    /* scratch_buf = W_hh @ h + b_hh */
    fe_gemv(weights->W_hh, state->h, weights->b_hh, scratch_buf, G4, H);

    /* gates += scratch_buf */
    for (i = 0; i < G4; i++) {
        scratch_gates[i] += scratch_buf[i];
    }

    /* Step 3: Apply gate activations and update cell/hidden state */
    for (i = 0; i < H; i++) {
        float i_gate = sigmoid_f(scratch_gates[i]);           /* input gate */
        float f_gate = sigmoid_f(scratch_gates[H + i]);       /* forget gate */
        float g_gate = tanh_f(scratch_gates[2*H + i]);        /* cell candidate */
        float o_gate = sigmoid_f(scratch_gates[3*H + i]);     /* output gate */

        state->c[i] = f_gate * state->c[i] + i_gate * g_gate;
        state->h[i] = o_gate * tanh_f(state->c[i]);
    }

    /* Step 4: Compute clean mask = sigmoid(clean_W @ h) */
    /* clean_W: [257, 512], no bias */
    for (i = 0; i < F; i++) {
        float sum = 0.0f;
        const float *row = weights->clean_W + i * H;
        for (int j = 0; j < H; j++) {
            sum += row[j] * state->h[j];
        }
        clean_mask[i] = sigmoid_f(sum);
    }

    /* Step 5: Compute interference mask if requested */
    if (interf_mask && weights->interf_W) {
        for (i = 0; i < F; i++) {
            float sum = 0.0f;
            const float *row = weights->interf_W + i * H;
            for (int j = 0; j < H; j++) {
                sum += row[j] * state->h[j];
            }
            interf_mask[i] = sigmoid_f(sum);
        }
    }
}
