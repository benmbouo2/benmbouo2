from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import build_self_supervised_dataset
from .model import build_denoising_autoencoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train self-supervised PPG denoiser.")
    p.add_argument("--csv-dir", type=Path, required=True, help="Directory with CSV files.")
    p.add_argument("--csv-pattern", type=str, default="*.csv", help="Glob for CSV files.")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Output artifact directory.")
    p.add_argument("--window-len", type=int, default=100, help="Training window length.")
    p.add_argument("--stride", type=int, default=25, help="Window stride.")
    p.add_argument("--epochs", type=int, default=80, help="Max training epochs.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    csv_files = sorted(args.csv_dir.glob(args.csv_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.csv_dir} matching {args.csv_pattern}")

    bundle = build_self_supervised_dataset(
        csv_files=csv_files,
        window_len=args.window_len,
        stride=args.stride,
        min_clean_cycles=5,
        cycle_samples=20,
    )
    x, y = bundle.x, bundle.y
    n = len(x)
    if n < 2:
        raise RuntimeError("Need at least 2 paired windows to split train/validation.")
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = max(1, min(n - 1, int(0.8 * n)))
    train_idx, val_idx = idx[:split], idx[split:]

    model = build_denoising_autoencoder(window_len=args.window_len, in_channels=x.shape[-1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
    ]
    history = model.fit(
        x[train_idx],
        y[train_idx],
        validation_data=(x[val_idx], y[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "ppg_denoiser.keras")
    np.savez(
        args.output_dir / "normalization.npz",
        feature_mean=bundle.feature_mean,
        feature_std=bundle.feature_std,
        target_mean=np.asarray([bundle.target_mean], dtype=np.float32),
        target_std=np.asarray([bundle.target_std], dtype=np.float32),
        motion_clean_threshold=np.asarray([bundle.motion_clean_threshold], dtype=np.float32),
        window_len=np.asarray([args.window_len], dtype=np.int32),
    )
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as f:
        json.dump(bundle.report, f, indent=2)
    with (args.output_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    print("Training complete.")
    print(f"Pairs used: {bundle.report['pairs']}")
    print(f"Accepted files: {len(bundle.report['accepted_files'])}")
    print(f"Rejected files: {len(bundle.report['rejected_files'])}")


if __name__ == "__main__":
    main()

