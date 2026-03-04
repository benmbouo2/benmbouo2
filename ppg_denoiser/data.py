from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = {
    "ppg": ("ppg",),
    "accelx": ("accelx", "accx", "ax"),
    "accely": ("accely", "accy", "ay"),
    "accelz": ("accelz", "accz", "az"),
    "gyrx": ("gyrx", "gyrox", "gx"),
    "gyry": ("gyry", "gyroy", "gy"),
    "gyrz": ("gyrz", "gyroz", "gz"),
}


@dataclass
class DatasetBundle:
    x: np.ndarray
    y: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    motion_clean_threshold: float
    report: dict[str, object]


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized = {col.strip().lower().replace("/", ""): col for col in df.columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            key = alias.strip().lower().replace("/", "")
            if key in normalized:
                resolved[canonical] = normalized[key]
                break
        if canonical not in resolved:
            raise ValueError(f"Missing required column for '{canonical}'.")
    return resolved


def load_last_1250_samples(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    cols = _resolve_columns(frame)
    out = pd.DataFrame({name: frame[src].to_numpy()[-1250:] for name, src in cols.items()})
    if len(out) != 1250:
        raise ValueError(f"{csv_path} has fewer than 1250 rows.")
    return out


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")


def _max_autocorr_peak(x: np.ndarray, lag_min: int, lag_max: int) -> float:
    centered = x - x.mean()
    denom = np.dot(centered, centered) + 1e-8
    best = 0.0
    for lag in range(lag_min, lag_max + 1):
        if lag >= len(centered):
            break
        score = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        best = max(best, score)
    return best


def _spectral_peak_ratio(x: np.ndarray, fs_hz: float, low_hz: float, high_hz: float) -> float:
    centered = x - x.mean()
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / fs_hz)
    power = np.abs(spectrum) ** 2
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return 0.0
    band_power = float(power[band].sum())
    total = float(power.sum()) + 1e-8
    return band_power / total


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def estimate_quality_masks(
    ppg: np.ndarray,
    imu: np.ndarray,
    fs_hz: float = 25.0,
    cycle_samples: int = 20,
    min_clean_cycles: int = 5,
    win_size: int = 100,
    step: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    motion_mag = np.linalg.norm(imu, axis=1)
    motion_mag = _rolling_mean(motion_mag.astype(np.float32), 5)

    ppg_hp = ppg - _rolling_mean(ppg, int(fs_hz))
    ppg_bp = _rolling_mean(ppg_hp, 3)

    motion_scores: list[float] = []
    quality_scores: list[float] = []
    starts = list(range(0, len(ppg) - win_size + 1, step))

    for s in starts:
        seg_ppg = ppg_bp[s : s + win_size]
        seg_motion = motion_mag[s : s + win_size]
        motion_score = float(np.mean(seg_motion))
        ac = _max_autocorr_peak(seg_ppg, lag_min=12, lag_max=40)
        spr = _spectral_peak_ratio(seg_ppg, fs_hz=fs_hz, low_hz=0.7, high_hz=3.2)
        quality = 0.6 * ac + 0.4 * spr
        motion_scores.append(motion_score)
        quality_scores.append(float(quality))

    motion_scores_arr = np.asarray(motion_scores, dtype=np.float32)
    quality_scores_arr = np.asarray(quality_scores, dtype=np.float32)
    motion_clean_thr = float(np.quantile(motion_scores_arr, 0.45))
    motion_noisy_thr = float(np.quantile(motion_scores_arr, 0.70))
    quality_thr = float(np.quantile(quality_scores_arr, 0.60))

    clean_win_mask = (motion_scores_arr <= motion_clean_thr) & (quality_scores_arr >= quality_thr)
    noisy_win_mask = motion_scores_arr >= motion_noisy_thr

    clean_mask = np.zeros_like(ppg, dtype=bool)
    noisy_mask = np.zeros_like(ppg, dtype=bool)
    for idx, s in enumerate(starts):
        if clean_win_mask[idx]:
            clean_mask[s : s + win_size] = True
        if noisy_win_mask[idx]:
            noisy_mask[s : s + win_size] = True

    min_clean_samples = min_clean_cycles * cycle_samples
    filtered_clean = np.zeros_like(clean_mask)
    for a, b in _contiguous_runs(clean_mask):
        if (b - a) >= min_clean_samples:
            filtered_clean[a:b] = True
    clean_mask = filtered_clean

    # Exclude clean sections from noisy candidates.
    noisy_mask = noisy_mask & ~clean_mask

    stats = {
        "motion_clean_thr": motion_clean_thr,
        "motion_noisy_thr": motion_noisy_thr,
        "quality_thr": quality_thr,
    }
    return clean_mask, noisy_mask, stats


def _valid_starts(mask: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    starts = np.arange(0, len(mask) - window_len + 1, stride)
    valid = []
    for s in starts:
        if mask[s : s + window_len].all():
            valid.append(s)
    return np.asarray(valid, dtype=np.int32)


def build_self_supervised_dataset(
    csv_files: Iterable[str | Path],
    window_len: int = 100,
    stride: int = 25,
    fs_hz: float = 25.0,
    min_clean_cycles: int = 5,
    cycle_samples: int = 20,
    seed: int = 7,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    accepted: list[str] = []
    rejected: list[str] = []
    clean_thresholds: list[float] = []

    for csv_path in csv_files:
        try:
            data = load_last_1250_samples(csv_path)
            ppg = data["ppg"].to_numpy(dtype=np.float32)
            imu = data[["accelx", "accely", "accelz", "gyrx", "gyry", "gyrz"]].to_numpy(dtype=np.float32)
            clean_mask, noisy_mask, stats = estimate_quality_masks(
                ppg=ppg,
                imu=imu,
                fs_hz=fs_hz,
                cycle_samples=cycle_samples,
                min_clean_cycles=min_clean_cycles,
                win_size=max(window_len, min_clean_cycles * cycle_samples),
                step=max(3, stride // 5),
            )
            clean_runs = _contiguous_runs(clean_mask)
            if not clean_runs:
                rejected.append(f"{csv_path}: no >=5-cycle clean region")
                continue

            clean_starts = _valid_starts(clean_mask, window_len=window_len, stride=stride)
            noisy_starts = _valid_starts(noisy_mask, window_len=window_len, stride=stride)
            pair_count = min(len(clean_starts), len(noisy_starts))
            if pair_count == 0:
                rejected.append(f"{csv_path}: cannot pair clean/noisy windows")
                continue

            clean_choice = rng.choice(clean_starts, size=pair_count, replace=False)
            noisy_choice = rng.choice(noisy_starts, size=pair_count, replace=False)

            features = data[["ppg", "accelx", "accely", "accelz", "gyrx", "gyry", "gyrz"]].to_numpy(dtype=np.float32)

            x_file = np.stack([features[s : s + window_len] for s in noisy_choice], axis=0)
            y_file = np.stack([ppg[s : s + window_len] for s in clean_choice], axis=0)[..., np.newaxis]
            x_chunks.append(x_file)
            y_chunks.append(y_file)
            clean_thresholds.append(stats["motion_clean_thr"])
            accepted.append(str(csv_path))
        except Exception as exc:  # noqa: BLE001 - include per-file failure reasons.
            rejected.append(f"{csv_path}: {exc}")

    if not x_chunks:
        raise RuntimeError(
            "No valid training pairs found. Check column headers and ensure clean/noisy sections exist."
        )

    x = np.concatenate(x_chunks, axis=0).astype(np.float32)
    y = np.concatenate(y_chunks, axis=0).astype(np.float32)

    feature_mean = x.mean(axis=(0, 1))
    feature_std = x.std(axis=(0, 1)) + 1e-6
    x = (x - feature_mean) / feature_std

    target_mean = float(y.mean())
    target_std = float(y.std() + 1e-6)
    y = (y - target_mean) / target_std

    report = {
        "accepted_files": accepted,
        "rejected_files": rejected,
        "pairs": int(x.shape[0]),
    }
    return DatasetBundle(
        x=x,
        y=y,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=target_mean,
        target_std=target_std,
        motion_clean_threshold=float(np.median(np.asarray(clean_thresholds))),
        report=report,
    )

