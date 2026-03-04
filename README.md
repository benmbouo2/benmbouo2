# Self-Supervised PPG Denoising (Python + Edge C)

This repository implements a **self-supervised 1D CNN autoencoder** to denoise PPG using:

- `PPG`
- `accelx`, `accely`, `accelz`
- `gyrx`, `gyry`, `gyrz`

from CSV files that may contain extra columns.

The pipeline enforces your constraints:

- uses only the **last 1250 samples**
- assumes `fs = 25 Hz` (~50 s)
- assumes ~20 samples/cardiac cycle
- requires at least **5 consecutive clean cardiac cycles** (>=100 samples) to keep a file
- mines clean and motion-corrupted windows from the **same recording** and uses equal-size pair counts

---

## 1) Setup

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas tensorflow
```

---

## 2) Train (self-supervised)

```bash
python -m ppg_denoiser.train \
  --csv-dir /path/to/csv_dir \
  --csv-pattern "*.csv" \
  --output-dir artifacts \
  --window-len 100 \
  --stride 25 \
  --epochs 80 \
  --batch-size 32
```

Artifacts:

- `artifacts/ppg_denoiser.keras`
- `artifacts/normalization.npz`
- `artifacts/train_report.json` (accepted/rejected files and reasons)
- `artifacts/train_history.json`

### Clean/noisy mining details

Each file is processed as:

1. load required columns, take last 1250 rows
2. estimate motion from accel+gyro magnitude
3. estimate PPG periodic quality (autocorr + spectral ratio)
4. mark clean windows: low motion + good periodicity
5. keep only clean runs >= 5 cycles (>=100 samples)
6. mark motion windows from high-motion regions
7. pair equal counts of clean/noisy windows from that same file

Files without >=5 consecutive clean cycles are eliminated automatically.

---

## 3) Python inference

```bash
python -m ppg_denoiser.infer \
  --input-csv /path/to/one_file.csv \
  --model artifacts/ppg_denoiser.keras \
  --normalization artifacts/normalization.npz \
  --output-csv artifacts/denoised_output.csv
```

Output CSV columns:

- `ppg_raw`
- `ppg_denoised`

---

## 4) Export trained model to C (PSoC63)

```bash
python -m ppg_denoiser.export_c \
  --model artifacts/ppg_denoiser.keras \
  --normalization artifacts/normalization.npz \
  --output-header edge_c/ppg_denoise_weights.h
```

Use these edge files in firmware:

- `edge_c/ppg_denoise.h`
- `edge_c/ppg_denoise.c`
- generated `edge_c/ppg_denoise_weights.h`

### Edge inference behavior

`ppg_denoise_run_window(...)` performs:

1. input normalization using exported feature stats
2. 4-layer Conv1D autoencoder forward pass
3. output de-normalization
4. lightweight self-supervised adaptation (gain+bias LMS update) when motion is below learned threshold

This keeps the on-device adaptation tiny and practical for embedded execution.

---

## 5) Expected input CSV schema

Required headers (case-insensitive aliases supported):

- `PPG`
- `accelx`, `accely`, `accelz`
- `gyrx`, `gyry`, `gyrz`

Extra columns are ignored.
