"""
Train a Random Forest regressor to predict blood glucose from PPG-derived features
and demographic information.

Expected CSV schema (flexible):
- One or more columns starting with "PPG" that hold the raw waveform per record.
  * If there is a single column named "PPG", each row can either be:
      - a JSON/CSV string with the waveform samples,
      - a Python-like list string, e.g. "[0.1, 0.2, ...]",
      - a path to a file containing comma-separated samples.
- Columns such as "age", "gender", and "glucose" (target).

Usage
-----
python src/glucose_prediction.py --data data/ppg_glucose.csv
python src/glucose_prediction.py --data data/ppg_glucose.csv --output-model artifacts/rf_glucose.joblib
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


def parse_ppg_sequence(value: object) -> np.ndarray:
    """Normalize any PPG column representation into a 1-D float array."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("PPG column contains missing values")

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=float)
    elif isinstance(value, (bytes, bytearray)):
        arr = _parse_ppg_string(value.decode("utf-8"))
    elif isinstance(value, (str, Path)):
        candidate = str(value).strip()
        if Path(candidate).is_file():
            arr = _parse_ppg_string(Path(candidate).read_text())
        else:
            arr = _parse_ppg_string(candidate)
    else:
        # Assume scalar numeric
        arr = np.array([float(value)], dtype=float)

    if arr.ndim != 1:
        arr = arr.ravel()
    if arr.size < 2:
        raise ValueError("PPG sequence must contain at least two samples")
    if not np.isfinite(arr).all():
        raise ValueError("PPG sequence contains non-finite values")
    return arr


def _parse_ppg_string(raw: str) -> np.ndarray:
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty PPG string encountered")

    # Try JSON/Python-like list first.
    if (raw[0] in "[{" and raw[-1] in "]}") or raw.startswith(("(", "{")):
        try:
            parsed = json.loads(raw.replace("(", "[").replace(")", "]"))
            return parse_ppg_sequence(parsed)
        except json.JSONDecodeError:
            pass

    tokens = re.split(r"[,\s;|]+", raw.strip("[](){}"))
    floats = [float(tok) for tok in tokens if tok]
    if not floats:
        raise ValueError(f"Could not parse PPG string: {raw[:50]}...")
    return np.asarray(floats, dtype=float)


def sample_entropy(signal: Sequence[float], m: int = 2, r: float | None = None) -> float:
    """Compute sample entropy (SampEn) with template length m.

    This implementation is quadratic in the signal length but robust for
    moderate wearable windows (hundreds of samples).
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n <= m + 1:
        return float("nan")
    std = np.std(x)
    if r is None:
        r = 0.2 * std if std > 0 else 1e-6

    def _count_matches(template_len: int) -> int:
        count = 0
        for i in range(n - template_len + 1):
            template = x[i : i + template_len]
            for j in range(i + 1, n - template_len + 1):
                if np.max(np.abs(template - x[j : j + template_len])) <= r:
                    count += 1
        return count

    matches_m = _count_matches(m)
    matches_m1 = _count_matches(m + 1)

    if matches_m == 0 or matches_m1 == 0:
        return float("nan")

    return -np.log(matches_m1 / matches_m)


def compute_ppg_features(df: pd.DataFrame, ppg_columns: List[str]) -> pd.DataFrame:
    """Derive statistical features (AUC, energy, entropy, etc.) per record."""
    feature_records = []
    for idx, row in df.iterrows():
        waveform = row_to_waveform(row, ppg_columns)
        feature_records.append(_summarize_waveform(waveform))

    feature_df = pd.DataFrame(feature_records, index=df.index)
    remaining = df.drop(columns=ppg_columns)
    return pd.concat([remaining, feature_df], axis=1)


def row_to_waveform(row: pd.Series, ppg_columns: List[str]) -> np.ndarray:
    if not ppg_columns:
        raise ValueError("No PPG columns detected in the dataset.")
    if len(ppg_columns) == 1:
        return parse_ppg_sequence(row[ppg_columns[0]])
    return row[ppg_columns].to_numpy(dtype=float)


def _summarize_waveform(waveform: np.ndarray) -> dict:
    auc = np.trapz(waveform)
    energy = np.sum(np.square(waveform)) / waveform.size
    sampen = sample_entropy(waveform)

    return {
        "ppg_length": waveform.size,
        "ppg_mean": float(np.mean(waveform)),
        "ppg_std": float(np.std(waveform, ddof=1) if waveform.size > 1 else 0.0),
        "ppg_min": float(np.min(waveform)),
        "ppg_max": float(np.max(waveform)),
        "ppg_range": float(np.ptp(waveform)),
        "ppg_auc": float(auc),
        "ppg_energy": float(energy),
        "ppg_sample_entropy": float(sampen),
    }


def detect_ppg_columns(df: pd.DataFrame) -> List[str]:
    candidates = [col for col in df.columns if col.lower().startswith("ppg")]
    if not candidates:
        raise ValueError("Dataset must contain at least one column starting with 'PPG'.")
    return candidates


def build_pipeline(X: pd.DataFrame, n_estimators: int, random_state: int) -> Pipeline:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )

    preprocess = ColumnTransformer(transformers, remainder="drop")
    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", regressor)])


def train_and_evaluate(
    dataset_path: Path,
    target_column: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
) -> tuple[Pipeline, dict]:
    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {df.columns}")

    ppg_columns = detect_ppg_columns(df)
    enriched = compute_ppg_features(df, ppg_columns)

    y = enriched[target_column].to_numpy(dtype=float)
    X = enriched.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline(X_train, n_estimators, random_state)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        "RMSE": float(mean_squared_error(y_test, preds, squared=False)),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
    }

    feature_rank = feature_importances(pipeline)
    if feature_rank:
        metrics["top_features"] = feature_rank[:10]
    return pipeline, metrics


def feature_importances(pipeline: Pipeline) -> List[tuple[str, float]]:
    model: RandomForestRegressor = pipeline.named_steps["model"]
    preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]

    try:
        feature_names = preprocess.get_feature_names_out()
    except AttributeError:
        feature_names = [
            f"f_{idx}" for idx in range(model.feature_importances_.shape[0])
        ]

    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    return pairs


def print_metrics(metrics: dict) -> None:
    print("\nEvaluation metrics:")
    for key in ("RMSE", "MAE", "R2"):
        if key in metrics:
            print(f"  {key:>4}: {metrics[key]:.4f}")

    if "top_features" in metrics:
        print("\nTop contributing features:")
        for name, score in metrics["top_features"]:
            print(f"  {name:<25} {score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPG-based glucose prediction with Random Forests.")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV file containing PPG, demo, glucose.")
    parser.add_argument("--target-column", default="glucose", help="Name of the ground-truth glucose column.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=500, help="Number of trees in the forest.")
    parser.add_argument("--output-model", type=Path, help="Optional path to persist the fitted pipeline (joblib).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline, metrics = train_and_evaluate(
        dataset_path=args.data,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    print_metrics(metrics)

    if args.output_model:
        if joblib is None:
            raise ImportError("joblib is required to persist the trained pipeline.")
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, args.output_model)
        print(f"\nModel saved to {args.output_model}")


if __name__ == "__main__":
    main()
