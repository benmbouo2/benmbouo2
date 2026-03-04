from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .data import load_last_1250_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PPG denoiser on one CSV file.")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("artifacts/ppg_denoiser.keras"))
    p.add_argument("--normalization", type=Path, default=Path("artifacts/normalization.npz"))
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--output-csv", type=Path, default=Path("artifacts/denoised_output.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model, compile=False)
    norm = np.load(args.normalization)

    window_len = int(norm["window_len"][0])
    feature_mean = norm["feature_mean"].astype(np.float32)
    feature_std = norm["feature_std"].astype(np.float32)
    target_mean = float(norm["target_mean"][0])
    target_std = float(norm["target_std"][0])

    df = load_last_1250_samples(args.input_csv)
    features = df[["ppg", "accelx", "accely", "accelz", "gyrx", "gyry", "gyrz"]].to_numpy(dtype=np.float32)
    ppg_raw = df["ppg"].to_numpy(dtype=np.float32)
    x = (features - feature_mean) / feature_std

    starts = np.arange(0, len(x) - window_len + 1, args.stride)
    windows = np.stack([x[s : s + window_len] for s in starts], axis=0)
    pred = model.predict(windows, verbose=0).squeeze(-1)
    pred = pred * target_std + target_mean

    recon = np.zeros_like(ppg_raw, dtype=np.float32)
    counts = np.zeros_like(ppg_raw, dtype=np.float32)
    for i, s in enumerate(starts):
        recon[s : s + window_len] += pred[i]
        counts[s : s + window_len] += 1.0
    counts[counts == 0] = 1.0
    ppg_clean = recon / counts

    out = pd.DataFrame({"ppg_raw": ppg_raw, "ppg_denoised": ppg_clean})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()

