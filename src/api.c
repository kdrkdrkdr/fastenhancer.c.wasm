/*
 * fe_api.c — Simple WASM API wrapping all initialization
 *
 * Provides fe_init() / fe_run() / fe_free() that JS can call directly.
 * Weights are loaded from a pointer provided by JS (no filesystem needed).
 */
#include "fastenhancer.h"
#include <stdlib.h>

static FeWeights g_weights;
static FeState *g_state = NULL;

/* Initialize from weight data blob.
 * Returns 0 on success, -1 on failure.
 * weight_data must remain valid for the lifetime of the engine. */
int fe_init(const void *weight_data, int weight_size) {
    if (g_state) return 0; /* already initialized */

    if (fe_load_weights(&g_weights, weight_data, (size_t)weight_size) != 0)
        return -1;

    g_state = fe_create(&g_weights);
    return g_state ? 0 : -1;
}

/* Process one frame: 256 float samples in → 256 float samples out */
void fe_run(const float *in, float *out) {
    if (g_state) {
        fe_process(g_state, &g_weights, in, out);
    }
}

/* Free all resources */
void fe_free(void) {
    if (g_state) {
        fe_destroy(g_state);
        g_state = NULL;
    }
}
