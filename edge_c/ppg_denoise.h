#ifndef PPG_DENOISE_H
#define PPG_DENOISE_H

#include "ppg_denoise_weights.h"

typedef struct {
  float gain;
  float bias;
} PpgDenoiseState;

void ppg_denoise_init(PpgDenoiseState *state);
void ppg_denoise_run_window(
    const float input[PPG_WINDOW_LEN][PPG_INPUT_CHANNELS],
    float output[PPG_WINDOW_LEN],
    PpgDenoiseState *state);

#endif  // PPG_DENOISE_H

