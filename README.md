# Feature Stability Monitor

A [Marimo](https://marimo.io) + [Plotly](https://plotly.com/python/) dashboard for monitoring the statistical stability of model input features over simulated inference batches.

## What it does

Each refresh generates 60 synthetic inference batches (5-minute intervals). For the selected feature the dashboard shows:

- **Drift score** — normalised deviation from the training-time reference mean, with warning/critical threshold bands
- **Null rate (%)** — fraction of missing values per batch, with a max-acceptable threshold line
- **Batch mean** — rolling per-batch mean overlaid with a ±1σ reference band
- **Batch std** — rolling per-batch standard deviation with a reference line

A KPI row and an instability summary table highlight any breaches across all features regardless of which one is currently selected.

## Project structure

```
dashboard.py     Marimo reactive notebook (UI + chart logic)
simulation.py    Synthetic data layer (FeatureSpec, DashboardConfig,
                 simulate_feature_stats, detect_instability)
pyproject.toml   Dependencies
```

## Running

```bash
# Production mode — code hidden, app only
marimo run dashboard.py

# Development mode — editable notebook
marimo edit dashboard.py
```

Both commands open the dashboard at `http://localhost:2718` by default.

## Monitored features

| Feature | Ref mean | Ref std |
|---|---|---|
| age | 38 | 11 |
| income | 55 000 | 18 000 |
| credit_score | 680 | 65 |
| loan_amount | 22 000 | 9 000 |
| employment_tenure | 5.5 | 3.5 |
| debt_ratio | 0.34 | 0.10 |

## Configuration

All thresholds and simulation parameters live in `DashboardConfig` / `CONFIG` inside `simulation.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `drift_threshold_low` | 0.20 | Mild-drift warning level |
| `drift_threshold_high` | 0.50 | Severe-drift critical level |
| `null_rate_max` | 0.05 | Max acceptable null rate (5%) |
| `anomaly_probability` | 0.03 | Per-batch anomaly injection rate |
| `total_batches` | 60 | Simulated batch history length |

## Smoke test

```bash
python -c "
import simulation
df = simulation.simulate_feature_stats(simulation.CONFIG)
print(df.shape)          # (60, 25)
print(list(df.columns))
"
```

## Requirements

- Python 3.12+
- marimo
- plotly
- pandas
- numpy
- rich
