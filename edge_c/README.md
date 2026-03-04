# Edge Integration Notes (PSoC63)

1. Generate `ppg_denoise_weights.h` from trained model:

```bash
python -m ppg_denoiser.export_c \
  --model artifacts/ppg_denoiser.keras \
  --normalization artifacts/normalization.npz \
  --output-header edge_c/ppg_denoise_weights.h
```

2. Add these files to firmware project:

- `ppg_denoise.h`
- `ppg_denoise.c`
- generated `ppg_denoise_weights.h`

3. Run inference per window (`PPG_WINDOW_LEN`, default 100 samples):

```c
PpgDenoiseState state;
ppg_denoise_init(&state);
ppg_denoise_run_window(sensor_window, ppg_out, &state);
```

4. Stream mode:

- collect overlapping windows (e.g., 100 samples, hop 25)
- denoise each window
- overlap-add average in application code if continuous output is required

