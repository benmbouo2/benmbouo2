#include "ppg_denoise.h"

#include <math.h>

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

static void conv1d_same(
    const float *input, int in_ch, int out_ch, int kernel,
    const float *weights, const float *bias, float *output) {
  int pad = kernel / 2;
  for (int t = 0; t < PPG_WINDOW_LEN; ++t) {
    for (int oc = 0; oc < out_ch; ++oc) {
      float acc = bias[oc];
      for (int k = 0; k < kernel; ++k) {
        int ti = t + k - pad;
        if (ti < 0 || ti >= PPG_WINDOW_LEN) {
          continue;
        }
        for (int ic = 0; ic < in_ch; ++ic) {
          int in_idx = ti * in_ch + ic;
          int w_idx = ((k * in_ch + ic) * out_ch) + oc;
          acc += input[in_idx] * weights[w_idx];
        }
      }
      output[t * out_ch + oc] = acc;
    }
  }
}

void ppg_denoise_init(PpgDenoiseState *state) {
  state->gain = 1.0f;
  state->bias = 0.0f;
}

void ppg_denoise_run_window(
    const float input[PPG_WINDOW_LEN][PPG_INPUT_CHANNELS],
    float output[PPG_WINDOW_LEN],
    PpgDenoiseState *state) {
  static float norm_in[PPG_WINDOW_LEN * PPG_INPUT_CHANNELS];
  static float l1[PPG_WINDOW_LEN * ENC_CONV1_OUT];
  static float l2[PPG_WINDOW_LEN * ENC_CONV2_OUT];
  static float l3[PPG_WINDOW_LEN * DEC_CONV1_OUT];
  static float l4[PPG_WINDOW_LEN * DEC_CONV2_OUT];

  float motion_acc = 0.0f;
  float raw_ppg_mean = 0.0f;
  for (int t = 0; t < PPG_WINDOW_LEN; ++t) {
    raw_ppg_mean += input[t][0];
    float ax = input[t][1];
    float ay = input[t][2];
    float az = input[t][3];
    float gx = input[t][4];
    float gy = input[t][5];
    float gz = input[t][6];
    motion_acc += sqrtf(ax * ax + ay * ay + az * az + gx * gx + gy * gy + gz * gz);
    for (int c = 0; c < PPG_INPUT_CHANNELS; ++c) {
      norm_in[t * PPG_INPUT_CHANNELS + c] =
          (input[t][c] - ppg_feature_mean[c]) / ppg_feature_std[c];
    }
  }
  raw_ppg_mean /= (float)PPG_WINDOW_LEN;
  float motion_mean = motion_acc / (float)PPG_WINDOW_LEN;

  conv1d_same(norm_in, ENC_CONV1_IN, ENC_CONV1_OUT, ENC_CONV1_K, enc_conv1_kernel,
              enc_conv1_bias, l1);
  for (int i = 0; i < PPG_WINDOW_LEN * ENC_CONV1_OUT; ++i) {
    l1[i] = relu(l1[i]);
  }

  conv1d_same(l1, ENC_CONV2_IN, ENC_CONV2_OUT, ENC_CONV2_K, enc_conv2_kernel,
              enc_conv2_bias, l2);
  for (int i = 0; i < PPG_WINDOW_LEN * ENC_CONV2_OUT; ++i) {
    l2[i] = relu(l2[i]);
  }

  conv1d_same(l2, DEC_CONV1_IN, DEC_CONV1_OUT, DEC_CONV1_K, dec_conv1_kernel,
              dec_conv1_bias, l3);
  for (int i = 0; i < PPG_WINDOW_LEN * DEC_CONV1_OUT; ++i) {
    l3[i] = relu(l3[i]);
  }

  conv1d_same(l3, DEC_CONV2_IN, DEC_CONV2_OUT, DEC_CONV2_K, dec_conv2_kernel,
              dec_conv2_bias, l4);

  float pred_mean = 0.0f;
  for (int t = 0; t < PPG_WINDOW_LEN; ++t) {
    float pred = l4[t] * ppg_target_std + ppg_target_mean;
    pred_mean += pred;
    output[t] = state->gain * pred + state->bias;
  }
  pred_mean /= (float)PPG_WINDOW_LEN;

  // Tiny self-supervised adaptation:
  // when motion is low, raw PPG is used as pseudo-clean supervision.
  if (motion_mean < ppg_motion_clean_threshold) {
    float err = (state->gain * pred_mean + state->bias) - raw_ppg_mean;
    state->gain -= ppg_adapt_lr * err * pred_mean;
    state->bias -= ppg_adapt_lr * err;
  }
}

