/*
 * main.h — Unified audio processing pipeline header
 *
 * Pipeline: HPF → Input AGC → Dereverb → Denoise → Output AGC
 * Sample rate: 16 kHz, frame size: 256 samples (16 ms)
 */
#ifndef MAIN_H
#define MAIN_H

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/*  Pipeline constants                                                  */
/* ------------------------------------------------------------------ */

#define PIPE_SAMPLE_RATE    16000
#define PIPE_FRAME_SIZE     256     /* samples per frame (16ms @ 16kHz)   */
#define PIPE_AGC_FRAME      160     /* AGC sub-frame (10ms @ 16kHz)       */

/* ------------------------------------------------------------------ */
/*  HPF: 2nd-order Butterworth high-pass 80 Hz @ 16 kHz                 */
/*                                                                       */
/*  Python verification:                                                 */
/*    from scipy.signal import butter                                    */
/*    b, a = butter(2, 80/8000, btype='high')                           */
/* ------------------------------------------------------------------ */

#define HPF_B0  0.98532175f
#define HPF_B1 -1.97064350f
#define HPF_B2  0.98532175f
#define HPF_A1 -1.97009521f
#define HPF_A2  0.97119178f

/* ------------------------------------------------------------------ */
/*  Public API (WASM exports)                                           */
/* ------------------------------------------------------------------ */

/* Initialize engine from weight blobs.
 * denoise_data: FastEnhancer weights (required)
 * dereverb_data: 2sderev LSTM weights (optional, NULL to disable)
 * Returns 0 on success, -1 on failure. */
int fe_init(const void *denoise_data, int denoise_size,
            const void *dereverb_data, int dereverb_size);

/* Process one frame: 256 samples in → 256 samples out (in-place ok) */
void fe_run(const float *in, float *out);

/* Free all resources */
void fe_free(void);

/* Runtime toggles */
void fe_set_hpf(int enabled);
void fe_set_input_agc(int enabled);
void fe_set_output_agc(int enabled);
void fe_set_dereverb(int enabled);

/* AGC configuration */
void fe_agc_init(void);
void fe_set_input_agc_compression(int gain_db);
void fe_set_output_agc_compression(int gain_db);

#endif /* MAIN_H */
