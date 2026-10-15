"""Versioned synthetic snapshots and strict evaluation boundaries."""

import numpy as np
import pandas as pd

GENERATOR_VERSION = "vessel-speed-synthetic-v1"
COLUMNS = ["record_id", "vessel_id", "observed_at", "speed_knots", "scenario", "is_anomalous"]


def generate_snapshots(rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate fictional independent snapshots, including deliberately ambiguous lows."""
    if not 1000 <= rows <= 100_000:
        raise ValueError("rows must be between 1,000 and 100,000")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    rng = np.random.default_rng(seed)
    anomalous = rng.random(rows) < 0.08
    ordinary_slow = rng.random(rows) < 0.12
    anomalous_fast = rng.random(rows) < 0.60
    speed = np.clip(rng.normal(12, 3, rows), 0, 24)
    low_speed = rng.uniform(0, 4, rows)
    fast_speed = np.clip(rng.normal(28, 6, rows), 10, 50)
    speed[ordinary_slow] = low_speed[ordinary_slow]
    speed[anomalous & ~anomalous_fast] = low_speed[anomalous & ~anomalous_fast]
    speed[anomalous & anomalous_fast] = fast_speed[anomalous & anomalous_fast]
    scenario = np.where(ordinary_slow, "ordinary_slow", "ordinary_transit")
    scenario = np.where(anomalous & ~anomalous_fast, "synthetic_slow_deviation", scenario)
    scenario = np.where(anomalous & anomalous_fast, "synthetic_fast_deviation", scenario)
    return pd.DataFrame(
        {
            "record_id": [f"synthetic-observation-{i:06d}" for i in range(rows)],
            "vessel_id": [f"synthetic-vessel-{i:06d}" for i in range(rows)],
            "observed_at": pd.date_range("2025-01-01", periods=rows, freq="min", tz="UTC").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "speed_knots": np.round(speed, 6),
            "scenario": scenario,
            "is_anomalous": anomalous.astype(int),
        }
    )


def validate_snapshots(frame: pd.DataFrame) -> None:
    if list(frame.columns) != COLUMNS or frame.empty or frame.isna().any().any():
        raise ValueError("Expected a nonempty snapshot table with the exact v1 schema and no nulls")
    for column in ("record_id", "vessel_id"):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"{column} must contain nonempty strings")
        if not frame[column].is_unique:
            raise ValueError(
                "Snapshot IDs must be unique; repeated-vessel tracks are not supported"
            )
    speed = frame["speed_knots"]
    if not pd.api.types.is_numeric_dtype(speed) or not np.isfinite(speed).all():
        raise ValueError("speed_knots must be finite numeric values")
    if not speed.between(0, 60).all():
        raise ValueError("speed_knots is outside the synthetic benchmark contract [0, 60]")
    if not frame["is_anomalous"].isin([0, 1]).all():
        raise ValueError("is_anomalous must contain only binary labels 0 and 1")
    if (
        not frame["observed_at"]
        .astype(str)
        .str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
        .all()
    ):
        raise ValueError("observed_at must use explicit UTC timestamps ending in Z")
    timestamps = pd.to_datetime(frame["observed_at"], utc=True, errors="raise")
    if not timestamps.is_unique:
        raise ValueError("Snapshot timestamps must be unique")


def assert_separated(earlier: pd.DataFrame, later: pd.DataFrame) -> None:
    for column in ("record_id", "vessel_id"):
        if set(earlier[column]) & set(later[column]):
            raise ValueError(f"Evaluation leakage: overlapping {column}")
    if earlier["observed_at"].max() >= later["observed_at"].min():
        raise ValueError("Evaluation periods must be strictly chronological and disjoint")


def split_snapshots(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split by timestamp, without inspecting labels to choose split membership."""
    validate_snapshots(frame)
    ordered = frame.sort_values("observed_at", kind="stable").reset_index(drop=True)
    first, second = int(len(ordered) * 0.6), int(len(ordered) * 0.8)
    splits = {
        "train": ordered.iloc[:first].copy(),
        "validation": ordered.iloc[first:second].copy(),
        "test": ordered.iloc[second:].copy(),
    }
    for name, split in splits.items():
        if split.empty or set(split["is_anomalous"]) != {0, 1}:
            raise ValueError(f"{name} needs both classes; do not resample to improve test results")
    assert_separated(splits["train"], splits["validation"])
    assert_separated(splits["validation"], splits["test"])
    assert_separated(splits["train"], splits["test"])
    return splits
