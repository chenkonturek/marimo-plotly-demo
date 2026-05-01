"""Synthetic feature stability simulation for the monitoring dashboard.

Generates per-batch statistical summaries (mean, std, null rate, drift score)
for model input features across simulated inference batches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_BATCHES: int = 60
BATCH_INTERVAL_SECONDS: int = 300       # 5-minute inference batches
BATCH_SIZE: int = 200                   # inference records per batch
ANOMALY_PROBABILITY: float = 0.03
ANOMALY_MAGNITUDE: float = 2.5          # spike size in units of ref_std
DEFAULT_SEED: int = 42
DRIFT_THRESHOLD_LOW: float = 0.20       # mild drift warning
DRIFT_THRESHOLD_HIGH: float = 0.50      # severe drift critical
NULL_RATE_MAX: float = 0.05             # 5% max acceptable null rate


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable specification for one monitored model input feature.

    Attributes:
        name: Feature identifier; used as column-name prefix and display label.
        ref_mean: Reference (training) distribution mean.
        ref_std: Reference (training) distribution standard deviation.
        clip_min: Hard floor applied to simulated batch values.
        clip_max: Hard ceiling applied to simulated batch values.
    """

    name: str
    ref_mean: float
    ref_std: float
    clip_min: float
    clip_max: float


@dataclass(frozen=True)
class DashboardConfig:
    """Immutable top-level configuration for the feature stability dashboard.

    Attributes:
        features: Ordered list of input feature specifications.
        total_batches: Number of inference batches to simulate.
        batch_interval_seconds: Simulated wall-clock seconds between batches.
        anomaly_probability: Per-batch per-feature probability of injecting drift.
        anomaly_magnitude: Drift spike size in multiples of ref_std.
        random_seed: Base seed for numpy's default_rng.
        drift_threshold_low: Normalised drift score for mild-drift warning.
        drift_threshold_high: Normalised drift score for severe-drift critical.
        null_rate_max: Maximum acceptable null rate fraction (0–1).
        color_normal: Plotly hex colour for in-range trace segments.
        color_warning: Plotly hex colour for mild-drift annotations.
        color_critical: Plotly hex colour for severe-drift annotations.
    """

    features: list[FeatureSpec]
    total_batches: int
    batch_interval_seconds: int
    anomaly_probability: float
    anomaly_magnitude: float
    random_seed: int
    drift_threshold_low: float
    drift_threshold_high: float
    null_rate_max: float
    color_normal: str
    color_warning: str
    color_critical: str


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

CONFIG: DashboardConfig = DashboardConfig(
    features=[
        FeatureSpec("age",               38.0,    11.0,    18.0,    85.0),
        FeatureSpec("income",            55000.0, 18000.0, 0.0,     500000.0),
        FeatureSpec("credit_score",      680.0,   65.0,    300.0,   850.0),
        FeatureSpec("loan_amount",       22000.0, 9000.0,  1000.0,  200000.0),
        FeatureSpec("employment_tenure", 5.5,     3.5,     0.0,     40.0),
        FeatureSpec("debt_ratio",        0.34,    0.10,    0.0,     1.0),
    ],
    total_batches=TOTAL_BATCHES,
    batch_interval_seconds=BATCH_INTERVAL_SECONDS,
    anomaly_probability=ANOMALY_PROBABILITY,
    anomaly_magnitude=ANOMALY_MAGNITUDE,
    random_seed=DEFAULT_SEED,
    drift_threshold_low=DRIFT_THRESHOLD_LOW,
    drift_threshold_high=DRIFT_THRESHOLD_HIGH,
    null_rate_max=NULL_RATE_MAX,
    color_normal="#5b9bd5",
    color_warning="#e67e22",
    color_critical="#e74c3c",
)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_feature_stats(config: DashboardConfig) -> pd.DataFrame:
    """Generate synthetic batch-level statistics for each model input feature.

    For each of config.total_batches time steps, produces four statistics per
    feature: batch mean, batch std, null rate, and normalised drift score.
    Includes slow linear mean drift and injected anomaly spikes.

    Args:
        config: Full dashboard configuration including feature specs and thresholds.

    Returns:
        DataFrame with a "timestamp" column (datetime64[ns]) followed by four
        float64 columns per feature: {name}_mean, {name}_std, {name}_null_pct,
        {name}_drift. Shape: (total_batches, 1 + 4 * len(features)).

    Raises:
        ValueError: If config.total_batches < 2 or any feature has ref_std <= 0.
    """
    if config.total_batches < 2:
        raise ValueError(f"total_batches must be >= 2, got {config.total_batches}")
    for spec in config.features:
        if spec.ref_std <= 0.0:
            raise ValueError(f"FeatureSpec '{spec.name}' has ref_std <= 0: {spec.ref_std}")

    window_seed = int(time.time()) // config.batch_interval_seconds
    rng = np.random.default_rng(config.random_seed ^ window_seed)

    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("s"),
        periods=config.total_batches,
        freq=f"{config.batch_interval_seconds}s",
    )

    data: dict[str, np.ndarray] = {}
    for spec in config.features:
        drift_sign = rng.choice([-1.0, 1.0])
        mean_drift = np.linspace(0.0, spec.ref_std * 1.5, config.total_batches) * drift_sign

        # Batch mean: ref + slow drift + sampling noise (CLT: σ/√N)
        batch_means = (
            spec.ref_mean
            + mean_drift
            + rng.normal(0.0, spec.ref_std / np.sqrt(BATCH_SIZE), config.total_batches)
        )
        batch_means = np.clip(batch_means, spec.clip_min, spec.clip_max)

        # Batch std: ref_std with ~5% multiplicative noise
        batch_stds = np.clip(
            spec.ref_std * rng.normal(1.0, 0.05, config.total_batches),
            0.0,
            spec.ref_std * 3.0,
        )

        # Null rate: ~2% baseline (Beta) with occasional spikes
        null_pct = rng.beta(1.0, 50.0, config.total_batches)
        null_anomaly_mask = rng.random(config.total_batches) < config.anomaly_probability
        null_pct[null_anomaly_mask] = rng.uniform(0.08, 0.25, int(null_anomaly_mask.sum()))
        null_pct = np.clip(null_pct, 0.0, 1.0)

        # Drift spike anomalies injected into batch means
        anomaly_mask = rng.random(config.total_batches) < config.anomaly_probability
        for idx in np.where(anomaly_mask)[0]:
            batch_means[idx] = np.clip(
                spec.ref_mean + rng.choice([-1.0, 1.0]) * spec.ref_std * config.anomaly_magnitude,
                spec.clip_min,
                spec.clip_max,
            )

        drift_score = (batch_means - spec.ref_mean) / spec.ref_std

        data[f"{spec.name}_mean"]     = batch_means
        data[f"{spec.name}_std"]      = batch_stds
        data[f"{spec.name}_null_pct"] = null_pct
        data[f"{spec.name}_drift"]    = drift_score

    return pd.DataFrame({"timestamp": timestamps, **data})


# ---------------------------------------------------------------------------
# Instability detection
# ---------------------------------------------------------------------------


def detect_instability(df: pd.DataFrame, config: DashboardConfig) -> pd.DataFrame:
    """Return a summary DataFrame of feature instability events.

    Checks drift scores and null rates against configured thresholds. Returns
    one row per (feature, issue_type) pair where at least one violation exists.

    Args:
        df: Output of simulate_feature_stats — must contain "timestamp" and
            {name}_drift, {name}_null_pct columns for all features in config.
        config: Dashboard configuration providing threshold values.

    Returns:
        DataFrame with columns [feature, issue_type, count, latest_value,
        latest_timestamp, threshold], sorted by count descending. Features
        with zero issues are excluded.

    Raises:
        ValueError: If df is missing "timestamp" or any required column.
    """
    required: set[str] = {"timestamp"}
    for spec in config.features:
        required.add(f"{spec.name}_drift")
        required.add(f"{spec.name}_null_pct")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns: {sorted(missing)}")

    records: list[dict] = []
    for spec in config.features:
        drift_series = df[f"{spec.name}_drift"]
        null_series  = df[f"{spec.name}_null_pct"]

        high_mask = drift_series > config.drift_threshold_high
        if high_mask.any():
            last_idx = int(drift_series[high_mask].index[-1])
            records.append({
                "feature": spec.name,
                "issue_type": "HIGH_DRIFT",
                "count": int(high_mask.sum()),
                "latest_value": float(drift_series.iloc[last_idx]),
                "latest_timestamp": df["timestamp"].iloc[last_idx],
                "threshold": config.drift_threshold_high,
            })

        neg_mask = drift_series < -config.drift_threshold_high
        if neg_mask.any():
            last_idx = int(drift_series[neg_mask].index[-1])
            records.append({
                "feature": spec.name,
                "issue_type": "NEGATIVE_DRIFT",
                "count": int(neg_mask.sum()),
                "latest_value": float(drift_series.iloc[last_idx]),
                "latest_timestamp": df["timestamp"].iloc[last_idx],
                "threshold": -config.drift_threshold_high,
            })

        null_mask = null_series > config.null_rate_max
        if null_mask.any():
            last_idx = int(null_series[null_mask].index[-1])
            records.append({
                "feature": spec.name,
                "issue_type": "HIGH_NULL_RATE",
                "count": int(null_mask.sum()),
                "latest_value": float(null_series.iloc[last_idx]),
                "latest_timestamp": df["timestamp"].iloc[last_idx],
                "threshold": config.null_rate_max,
            })

    if not records:
        return pd.DataFrame(
            columns=["feature", "issue_type", "count", "latest_value", "latest_timestamp", "threshold"]
        )

    return pd.DataFrame(records).sort_values("count", ascending=False).reset_index(drop=True)
