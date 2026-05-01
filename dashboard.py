"""Marimo dashboard for model input feature stability monitoring.

Run with:
    marimo run dashboard.py      # production mode (code hidden)
    marimo edit dashboard.py     # development mode (editable notebook)
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full", app_title="Feature Stability Monitor")


@app.cell
def imports():
    """Import all third-party and local libraries."""
    import marimo as mo
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from rich.console import Console
    import simulation

    return Console, go, make_subplots, mo, simulation


@app.cell
def config_cell(simulation):
    """Expose the module-level dashboard configuration."""
    CONFIG = simulation.CONFIG
    return (CONFIG,)


@app.cell
def header_cell(mo):
    """Display the dashboard title and description."""
    mo.output.replace(
        mo.vstack(
            [
                mo.md("# Model Input Feature Stability Monitoring"),
                mo.md(
                    "_Simulated inference batch data · "
                    "drift thresholds configurable in `DashboardConfig`_"
                ),
            ]
        )
    )
    return


@app.cell
def refresh_control(mo):
    """Create and display the auto-refresh timer widget."""
    refresh_ticker = mo.ui.refresh(
        options=["5s", "10s", "30s", "1m"],
        default_interval="10s",
        label="Auto-refresh interval",
    )
    return (refresh_ticker,)


@app.cell
def raw_data(CONFIG, Console, refresh_ticker, simulation):
    """Regenerate the full synthetic feature statistics DataFrame on each refresh tick."""
    _console = Console()
    _tick = refresh_ticker.value  # establishes reactive dependency on every tick
    df = simulation.simulate_feature_stats(CONFIG)
    _console.log(f"[cyan]Data refreshed[/cyan]: {len(df)} batches, tick={_tick!r}")
    return (df,)


@app.cell
def controls(CONFIG, mo):
    """Create and display the feature dropdown and the time-window slider."""
    feature_dropdown = mo.ui.dropdown(
        options=[v.name for v in CONFIG.features],
        value=CONFIG.features[0].name,
        label="Feature",
    )
    time_slider = mo.ui.range_slider(
        start=0,
        stop=CONFIG.total_batches - 1,
        step=1,
        value=[0, CONFIG.total_batches - 1],
        debounce=True,
        show_value=False,
        label="Time window (batch index)",
        full_width=True,
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.hstack([feature_dropdown], justify="start"),
                time_slider,
            ],
            gap=0.5,
        )
    )
    return feature_dropdown, time_slider


@app.cell
def filtered_data(CONFIG, df, feature_dropdown, simulation, time_slider):
    """Slice the DataFrame to the selected time window and feature; detect instability."""
    lo, hi = time_slider.value
    selected_names: list[str] = [feature_dropdown.value]
    _window_df = df.iloc[lo : hi + 1].copy()
    stat_suffixes = ("_mean", "_std", "_null_pct", "_drift")
    stat_cols = [f"{name}{s}" for name in selected_names for s in stat_suffixes]
    filtered_df = _window_df[["timestamp"] + stat_cols].copy()
    selected_specs = [v for v in CONFIG.features if v.name in selected_names]
    instability_df = simulation.detect_instability(_window_df, CONFIG)
    return filtered_df, instability_df, selected_specs


@app.cell
def kpi_stats_cell(CONFIG, filtered_df, instability_df, mo, selected_specs):
    """Compute and display summary KPI statistics for feature stability."""
    unstable_names: set[str] = (
        set(instability_df["feature"].tolist()) if not instability_df.empty else set()
    )
    stable_count = sum(1 for s in selected_specs if s.name not in unstable_names)

    drift_cols = [
        f"{s.name}_drift"
        for s in selected_specs
        if f"{s.name}_drift" in filtered_df.columns
    ]
    max_drift = float(filtered_df[drift_cols].abs().max().max()) if drift_cols else 0.0

    t0 = filtered_df["timestamp"].iloc[0]
    t1 = filtered_df["timestamp"].iloc[-1]
    window_hrs = (t1 - t0).total_seconds() / 3600.0

    drift_direction = "increase" if max_drift > CONFIG.drift_threshold_high else None

    mo.output.replace(
        mo.hstack(
            [
                mo.stat(
                    value=f"{stable_count}/{len(selected_specs)}",
                    label="Stable Features",
                    direction="decrease" if stable_count < len(selected_specs) else None,
                    target_direction="increase",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{max_drift:.3f}",
                    label="Max Drift Score",
                    direction=drift_direction,
                    target_direction="decrease",
                    bordered=True,
                ),
                mo.stat(value=f"{window_hrs:.1f} hr", label="Window Duration", bordered=True),
                mo.stat(value=str(len(filtered_df)), label="Batches Monitored", bordered=True),
            ],
            widths="equal",
        )
    )
    return


@app.cell
def charts_cell(CONFIG, filtered_df, go, make_subplots, mo, selected_specs):
    """Build and display one 2×2 subplot figure per selected feature."""

    def _make_feature_chart(spec) -> go.Figure:
        ts        = filtered_df["timestamp"]
        drift_vals = filtered_df[f"{spec.name}_drift"]
        null_vals  = filtered_df[f"{spec.name}_null_pct"] * 100.0
        mean_vals  = filtered_df[f"{spec.name}_mean"]
        std_vals   = filtered_df[f"{spec.name}_std"]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Drift Score",
                "Null Rate (%)",
                f"Batch Mean  (ref={spec.ref_mean:g})",
                f"Batch Std  (ref={spec.ref_std:g})",
            ),
            shared_xaxes=False,
            vertical_spacing=0.18,
            horizontal_spacing=0.12,
        )

        # --- (1,1) Drift score ---
        fig.add_trace(go.Scatter(
            x=ts, y=drift_vals, mode="lines",
            line={"color": CONFIG.color_normal, "width": 1.5},
            showlegend=False,
        ), row=1, col=1)

        critical_mask = drift_vals.abs() > CONFIG.drift_threshold_high
        if critical_mask.any():
            fig.add_trace(go.Scatter(
                x=ts[critical_mask], y=drift_vals[critical_mask], mode="markers",
                marker={"color": CONFIG.color_critical, "size": 7, "symbol": "circle"},
                showlegend=False,
            ), row=1, col=1)

        _drift_y_pad = max(abs(drift_vals).max() * 0.3, 0.3)
        fig.add_hrect(
            y0=CONFIG.drift_threshold_high,
            y1=drift_vals.max() + _drift_y_pad,
            fillcolor=CONFIG.color_critical, opacity=0.10, layer="below", line_width=0,
            row=1, col=1,
        )
        fig.add_hrect(
            y0=drift_vals.min() - _drift_y_pad,
            y1=-CONFIG.drift_threshold_high,
            fillcolor=CONFIG.color_critical, opacity=0.10, layer="below", line_width=0,
            row=1, col=1,
        )
        fig.add_hrect(
            y0=CONFIG.drift_threshold_low, y1=CONFIG.drift_threshold_high,
            fillcolor=CONFIG.color_warning, opacity=0.08, layer="below", line_width=0,
            row=1, col=1,
        )
        fig.add_hrect(
            y0=-CONFIG.drift_threshold_high, y1=-CONFIG.drift_threshold_low,
            fillcolor=CONFIG.color_warning, opacity=0.08, layer="below", line_width=0,
            row=1, col=1,
        )
        fig.add_hline(
            y=CONFIG.drift_threshold_high, line_dash="dash", line_color=CONFIG.color_critical,
            annotation_text=f"Critical ±{CONFIG.drift_threshold_high}",
            annotation_position="top left", row=1, col=1,
        )
        fig.add_hline(
            y=-CONFIG.drift_threshold_high, line_dash="dash", line_color=CONFIG.color_critical,
            row=1, col=1,
        )
        fig.add_hline(
            y=CONFIG.drift_threshold_low, line_dash="dot", line_color=CONFIG.color_warning,
            annotation_text=f"Warn ±{CONFIG.drift_threshold_low}",
            annotation_position="bottom left", row=1, col=1,
        )
        fig.add_hline(
            y=-CONFIG.drift_threshold_low, line_dash="dot", line_color=CONFIG.color_warning,
            row=1, col=1,
        )

        # --- (1,2) Null rate ---
        null_threshold_pct = CONFIG.null_rate_max * 100.0
        fig.add_trace(go.Scatter(
            x=ts, y=null_vals, mode="lines",
            line={"color": CONFIG.color_normal, "width": 1.5},
            showlegend=False,
        ), row=1, col=2)

        null_high_mask = null_vals > null_threshold_pct
        if null_high_mask.any():
            fig.add_trace(go.Scatter(
                x=ts[null_high_mask], y=null_vals[null_high_mask], mode="markers",
                marker={"color": CONFIG.color_critical, "size": 7, "symbol": "circle"},
                showlegend=False,
            ), row=1, col=2)

        _null_y_pad = max(null_vals.max() * 0.3, 2.0)
        fig.add_hrect(
            y0=null_threshold_pct, y1=null_vals.max() + _null_y_pad,
            fillcolor=CONFIG.color_critical, opacity=0.10, layer="below", line_width=0,
            row=1, col=2,
        )
        fig.add_hline(
            y=null_threshold_pct, line_dash="dash", line_color=CONFIG.color_critical,
            annotation_text=f"Max: {null_threshold_pct:.0f}%",
            annotation_position="top left", row=1, col=2,
        )

        # --- (2,1) Batch mean ---
        fig.add_trace(go.Scatter(
            x=ts, y=mean_vals, mode="lines",
            line={"color": CONFIG.color_normal, "width": 1.5},
            showlegend=False,
        ), row=2, col=1)

        fig.add_hrect(
            y0=spec.ref_mean - spec.ref_std, y1=spec.ref_mean + spec.ref_std,
            fillcolor=CONFIG.color_normal, opacity=0.10, layer="below", line_width=0,
            row=2, col=1,
        )
        fig.add_hline(
            y=spec.ref_mean, line_dash="dash", line_color=CONFIG.color_normal,
            annotation_text=f"Ref: {spec.ref_mean:g}",
            annotation_position="top left", row=2, col=1,
        )

        # --- (2,2) Batch std ---
        fig.add_trace(go.Scatter(
            x=ts, y=std_vals, mode="lines",
            line={"color": CONFIG.color_normal, "width": 1.5},
            showlegend=False,
        ), row=2, col=2)

        fig.add_hline(
            y=spec.ref_std, line_dash="dash", line_color=CONFIG.color_normal,
            annotation_text=f"Ref σ: {spec.ref_std:g}",
            annotation_position="top left", row=2, col=2,
        )

        fig.update_layout(
            title={"text": spec.name.replace("_", " ").title(), "font": {"size": 14}},
            template="plotly_dark",
            height=420,
            margin={"l": 50, "r": 20, "t": 60, "b": 40},
            showlegend=False,
            hovermode="x unified",
        )
        return fig

    mo.output.replace(
        mo.ui.plotly(_make_feature_chart(selected_specs[0]))
        if selected_specs
        else mo.md("_No feature selected._")
    )
    return


@app.cell
def breach_table_cell(instability_df, mo):
    """Build and display the feature instability summary table."""
    if instability_df.empty:
        _content = mo.callout(
            mo.md("**No feature instability detected** in the current time window."),
            kind="success",
        )
    else:
        display_df = instability_df.assign(
            latest_value=instability_df["latest_value"].map(lambda v: f"{v:.4f}"),
            threshold=instability_df["threshold"].map(lambda v: f"{v:.4f}"),
        )
        _content = mo.vstack(
            [
                mo.md("### Feature Instability Summary"),
                mo.ui.table(data=display_df, page_size=10),
            ]
        )
    mo.output.replace(_content)
    return


if __name__ == "__main__":
    app.run()
