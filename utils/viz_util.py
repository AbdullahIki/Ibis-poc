import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# ── Global plot style ────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi":        120,
    "figure.figsize":    (12, 4),
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "lines.linewidth":   1.4,
    "font.family":       "sans-serif",
})


def plot_series(
    df,
    series_list,
    time_col="date",
    value_col="quantity",
    identifier_cols=None,
    title="Series Overview",
    figsize=(14, 4),
):
    """Plot one or more raw time series on a single axes.

    Parameters
    ----------
    df : DataFrame
        The full dataset containing all series.
    series_list : list[dict | Series]
        Each element identifies one series via the identifier columns.
        Can be a dict, a pandas Series, or a DataFrame row.
    time_col, value_col : str
        Column names for time axis and value axis.
    identifier_cols : list[str]
        Columns that uniquely identify a series.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    """
    if identifier_cols is None:
        identifier_cols = ["product", "sku", "region"]

    fig, ax = plt.subplots(figsize=figsize)
    for entry in series_list:
        sf = dict(entry) if not isinstance(entry, dict) else entry
        subset = _filter_series(df, identifier_cols, sf, time_col, value_col)
        subset = subset.sort_values(time_col)
        ax.plot(
            pd.to_datetime(subset[time_col]),
            subset[value_col],
            linewidth=1.2,
            label=_series_label(sf),
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    return fig


def _select_series(df, identifier_cols, series_to_plot=None, max_series=4):
    """Return a list of filter-dicts for the series to visualize."""
    if series_to_plot is not None:
        return series_to_plot

    groups = df.groupby(identifier_cols).size().reset_index().drop(columns=0)
    selected = groups.head(max_series)
    return [dict(row) for _, row in selected.iterrows()]


def _filter_series(df, identifier_cols, series_filter, time_col=None, value_col=None):
    """Filter df to a single series and sum duplicates per date if present."""
    mask = pd.Series(True, index=df.index)
    for col in identifier_cols:
        mask &= df[col] == series_filter[col]
    result = df[mask].copy()
    if time_col in result.columns and value_col in result.columns:
        group_cols = [time_col] + identifier_cols
        existing = [c for c in group_cols if c in result.columns]
        if result.duplicated(subset=existing).any():
            result = result.groupby(existing, as_index=False).agg(
                {value_col: "sum", **{c: "first" for c in result.columns
                                      if c not in existing + [value_col]}}
            )
    return result


def _series_label(series_filter):
    return " | ".join(str(v) for v in series_filter.values())


def plot_imputation(
    df_before,
    df_after,
    time_col="date",
    value_col="quantity",
    identifier_cols=None,
    series_to_plot=None,
    max_series=4,
):
    """Before/after comparison highlighting imputed (previously missing) values."""

    series_list = _select_series(
        df_after, identifier_cols, series_to_plot, max_series
    )
    n = len(series_list)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)

    for i, sf in enumerate(series_list):
        ax = axes[i, 0]
        before = _filter_series(df_before, identifier_cols, sf, time_col, value_col).sort_values(time_col)
        after = _filter_series(df_after, identifier_cols, sf, time_col, value_col).sort_values(time_col)

        ax.plot(
            pd.to_datetime(after[time_col]),
            after[value_col],
            color="tab:blue",
            linewidth=1,
            label="After (imputed)",
        )
        ax.plot(
            pd.to_datetime(before[time_col]),
            before[value_col],
            color="tab:orange",
            linewidth=1.2,
            linestyle="--",
            label="Before (original)",
        )

        ax.set_title(f"Imputation — {_series_label(sf)}", fontsize=11)
        ax.set_ylabel(value_col)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axes[-1, 0].set_xlabel(time_col)
    fig.tight_layout()
    return fig


def plot_anomalies(
    df_before,
    df_after,
    time_col="date",
    value_col="quantity",
    identifier_cols=None,
    anomaly_col=None,
    series_to_plot=None,
    max_series=4,
):
    """Plot time series with detected anomalies highlighted."""
    if identifier_cols is None:
        identifier_cols = ["product", "sku", "region"]

    if anomaly_col is None:
        candidate_cols = [c for c in df_after.columns
                          if "anomal" in c.lower() or "outlier" in c.lower()]
        anomaly_col = candidate_cols[0] if candidate_cols else "Anomaly"

    series_list = _select_series(
        df_after, identifier_cols, series_to_plot, max_series
    )
    n = len(series_list)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)

    for i, sf in enumerate(series_list):
        ax = axes[i, 0]
        after = _filter_series(df_after, identifier_cols, sf, time_col, value_col).sort_values(time_col)
        times = pd.to_datetime(after[time_col])

        ax.plot(times, after[value_col], color="tab:blue", linewidth=1, label="Series")

        if anomaly_col in after.columns:
            anom_mask = after[anomaly_col].astype(bool)
            ax.scatter(
                times[anom_mask],
                after.loc[anom_mask, value_col],
                color="tab:red",
                zorder=5,
                s=40,
                label=f"Anomaly ({anom_mask.sum()})",
            )

        ax.set_title(f"Anomaly Detection — {_series_label(sf)}", fontsize=11)
        ax.set_ylabel(value_col)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axes[-1, 0].set_xlabel(time_col)
    fig.tight_layout()
    return fig


def plot_change_points(
    df_before,
    df_after,
    time_col="date",
    value_col="quantity",
    identifier_cols=None,
    changepoint_col="Changepoint",
    series_to_plot=None,
    max_series=4,
):
    """Plot time series with vertical lines at detected change points."""
    if identifier_cols is None:
        identifier_cols = ["product", "sku", "region"]

    series_list = _select_series(
        df_after, identifier_cols, series_to_plot, max_series
    )
    n = len(series_list)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)

    for i, sf in enumerate(series_list):
        ax = axes[i, 0]
        after = _filter_series(df_after, identifier_cols, sf, time_col, value_col).sort_values(time_col)
        times = pd.to_datetime(after[time_col])

        ax.plot(times, after[value_col], color="tab:blue", linewidth=1, label="Series")

        if changepoint_col in after.columns:
            cp_mask = after[changepoint_col].astype(bool)
            for t in times[cp_mask]:
                ax.axvline(t, color="tab:red", linestyle="--", alpha=0.7)
            ax.plot([], [], color="tab:red", linestyle="--",
                    label=f"Change points ({cp_mask.sum()})")

        ax.set_title(f"Change Point Detection — {_series_label(sf)}", fontsize=11)
        ax.set_ylabel(value_col)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axes[-1, 0].set_xlabel(time_col)
    fig.tight_layout()
    return fig


def plot_decomposition(
    df_result,
    time_col="date",
    value_col="quantity",
    identifier_cols=None,
    decomp_col="Decomposition",
    components=None,
    series_to_plot=None,
    max_series=2,
    df_original=None,
):
    """Multi-panel decomposition plot.

    Handles the long-format output from Ikigai where the *decomp_col*
    column contains the component name and *value_col* holds the value.
    If df_original is provided, the first row shows the original series.
    """
    if identifier_cols is None:
        identifier_cols = ["product", "sku", "region"]

    # Detect long format (Decomposition column present)
    is_long = decomp_col in df_result.columns

    if is_long:
        all_components = df_result[decomp_col].unique().tolist()
        if components is None:
            components = all_components
        else:
            components = [c for c in components if c in all_components]
            if not components:
                components = all_components

        first_comp = df_result[df_result[decomp_col] == components[0]]
        series_list = _select_series(
            first_comp, identifier_cols, series_to_plot, max_series
        )
    else:
        available = [c for c in (components or []) if c in df_result.columns]
        if not available:
            available = [c for c in df_result.columns
                         if c not in identifier_cols + [time_col, value_col]
                         and not c.startswith("embedding")][:5]
        components = available
        series_list = _select_series(
            df_result, identifier_cols, series_to_plot, max_series
        )

    show_original = df_original is not None
    n_rows = len(components) + (1 if show_original else 0)
    n_cols = len(series_list)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 3 * n_rows), squeeze=False
    )

    for j, sf in enumerate(series_list):
        label = _series_label(sf)
        row_offset = 0

        if show_original:
            orig = _filter_series(df_original, identifier_cols, sf, time_col, value_col).sort_values(time_col)
            times = pd.to_datetime(orig[time_col])
            ax = axes[0, j]
            ax.plot(times, orig[value_col], color="tab:blue", linewidth=1)
            ax.set_title(f"Original — {label}", fontsize=10)
            ax.set_ylabel(value_col)
            row_offset = 1

        for k, comp in enumerate(components):
            ax = axes[k + row_offset, j]
            if is_long:
                comp_data = df_result[df_result[decomp_col] == comp]
                data = _filter_series(comp_data, identifier_cols, sf, time_col, value_col).sort_values(time_col)
                times = pd.to_datetime(data[time_col])
                ax.plot(times, data[value_col], color=f"C{k + 1}", linewidth=1)
            else:
                data = _filter_series(df_result, identifier_cols, sf, time_col, value_col).sort_values(time_col)
                times = pd.to_datetime(data[time_col])
                ax.plot(times, data[comp], color=f"C{k + 1}", linewidth=1)
            ax.set_title(f"{comp} — {label}", fontsize=10)
            ax.set_ylabel(value_col if is_long else comp)

        axes[-1, j].set_xlabel(time_col)
        for row_ax in axes[:, j]:
            row_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    return fig


def plot_cohorts(
    df_result,
    identifier_cols=None,
    cohort_col="cohort",
    max_clusters=8,
    **_kwargs,
):
    """Visualize cohort assignments: cluster bar chart + 2D embedding scatter.

    The cohorts output is one row per identifier (not per time step), so we
    show distribution and an embedding-space scatter rather than time series.
    Extra kwargs are accepted and ignored for call-site compatibility.
    """
    if identifier_cols is None:
        identifier_cols = ["product", "sku", "region"]

    if cohort_col not in df_result.columns:
        candidate = [c for c in df_result.columns if "cluster" in c.lower()
                     or "cohort" in c.lower()]
        cohort_col = candidate[0] if candidate else cohort_col

    clusters = sorted(df_result[cohort_col].unique())[:max_clusters]
    n_clusters = len(clusters)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_clusters, 3)))

    embed_cols = [c for c in df_result.columns if c.startswith("embedding_")]
    has_embeddings = len(embed_cols) >= 2

    fig, axes = plt.subplots(
        1, 2 if has_embeddings else 1,
        figsize=(14 if has_embeddings else 8, 5),
        squeeze=False,
    )

    # --- Bar chart of cluster sizes ---
    cluster_counts = df_result[cohort_col].value_counts().sort_index()
    ax0 = axes[0, 0]
    ax0.bar(
        [str(c) for c in cluster_counts.index],
        cluster_counts.values,
        color=[colors[clusters.index(c) % len(colors)] for c in cluster_counts.index],
    )
    ax0.set_title("Cluster Distribution (series per cohort)", fontsize=11)
    ax0.set_xlabel("Cohort")
    ax0.set_ylabel("Count")
    for bar_i, (cid, cnt) in enumerate(cluster_counts.items()):
        members = df_result[df_result[cohort_col] == cid]
        labels = [_series_label(dict(row)) for _, row in
                  members[identifier_cols].iterrows()]
        ax0.annotate(
            "\n".join(labels),
            xy=(bar_i, cnt),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # --- 2D embedding scatter ---
    if has_embeddings:
        ax1 = axes[0, 1]
        x_col, y_col = embed_cols[0], embed_cols[1]
        for ci, cid in enumerate(clusters):
            mask = df_result[cohort_col] == cid
            ax1.scatter(
                df_result.loc[mask, x_col],
                df_result.loc[mask, y_col],
                color=colors[ci % len(colors)],
                s=80,
                label=f"Cohort {cid}",
                edgecolors="black",
                linewidths=0.5,
            )
            for _, row in df_result[mask].iterrows():
                ax1.annotate(
                    _series_label({c: row[c] for c in identifier_cols}),
                    (row[x_col], row[y_col]),
                    fontsize=5,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title("Embedding Space (first 2 dims)", fontsize=11)
        ax1.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_cohorts_timeseries(
    input_df,
    cohorts_df,
    time_col="date",
    value_col="quantity",
    identifier_cols=None,
    cohort_col="cohort",
):
    """Plot timeseries merged with cohorts, with one subplot per cohort colored by cohort.

    Merges input_df with cohorts_df on identifier columns, then plots each series
    in its cohort subplot with cohort-based coloring.
    """
    if identifier_cols is None:
        identifier_cols = ["product", "sku", "region"]

    merged = input_df.merge(
        cohorts_df[identifier_cols + [cohort_col]],
        on=identifier_cols,
        how="inner",
    )

    cohorts_sorted = sorted(merged[cohort_col].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(cohorts_sorted), 1)))
    color_map = dict(zip(cohorts_sorted, colors))

    fig, axes = plt.subplots(
        len(cohorts_sorted), 1,
        figsize=(14, 4 * len(cohorts_sorted)),
        squeeze=False,
    )
    for i, cid in enumerate(cohorts_sorted):
        ax = axes[i, 0]
        subset = merged[merged[cohort_col] == cid]
        groups = list(subset.groupby(identifier_cols))
        n_series = len(groups)
        palette = plt.cm.tab20(np.linspace(0, 1, max(n_series, 1)))
        for j, (_, grp) in enumerate(groups):
            ax.plot(
                pd.to_datetime(grp[time_col]),
                grp[value_col],
                color=palette[j],
                alpha=0.8,
                linewidth=1,
            )
        ax.set_title(
            f"Cohort {cid} ({subset.groupby(identifier_cols).ngroups} series)",
            fontsize=11,
        )
        ax.set_ylabel(value_col)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1, 0].set_xlabel(time_col)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# AICast Forecast Visualisation
# ---------------------------------------------------------------------------


def plot_forecast(
    df_forecast,
    time_col,
    value_col,
    identifier_cols,
    series_to_plot=None,
    best_model_only=True,
    actuals_lookback=None,
    filter_holdout=False,
):
    """Plot AICast forecast with actuals, prediction, and confidence intervals.

    Parameters
    ----------
    df_forecast : pd.DataFrame
        AICast result with columns: identifier cols, *time_col*, *value_col*,
        ``prediction_type`` (real | predicted_value | lower_bound | upper_bound),
        ``real/pred``, ``model``, ``model_ranking``, ``identifier_ranking``.
    series_to_plot : list[dict] | None
        Explicit series filters.  When *None* the topline (total across all
        identifiers) is plotted.
    best_model_only : bool
        If True keep only model_ranking == 1 for each series.
    actuals_lookback : int | None
        Number of most recent actual data points to display before the
        forecast.  If *None* all actuals are shown.
    """
    df = df_forecast.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Keep best model per series when requested
    if best_model_only and "model_ranking" in df.columns:
        df = df[(df["real/pred"] == "real") | (df["model_ranking"] == 1) | (df["model_ranking"] == "1")]
    max_real_date = df.loc[df["prediction_type"] == "real", time_col].max()

    # Default: plot the topline (sum across all identifiers)
    plot_topline = series_to_plot is None
    # always aggregate up to ID + date level to handle any duplicates, but only keep the prediction_type breakdown if plotting topline
    df = df.groupby(
        [time_col, "prediction_type"] + identifier_cols ,
        as_index=False
    ).agg({value_col: "sum"})
    if plot_topline:
        # Aggregate to topline per date + prediction_type
        agg_cols = [time_col, "prediction_type"]
        topline = df.groupby(agg_cols, as_index=False).agg({value_col: "sum"})

        series_to_plot_internal = [{"_topline": True}]
        series_data = {0: topline}
    else:
        series_to_plot_internal = series_to_plot
        series_data = {}

    n = len(series_to_plot_internal)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n), squeeze=False)

    for i, sf in enumerate(series_to_plot_internal):
        ax = axes[i, 0]

        if plot_topline:
            sub = series_data[i]
        else:
            sub = _filter_series(df, identifier_cols, sf)

        sub = sub.sort_values(time_col)

        # Actuals — optionally trim to last N points
        actuals = sub[sub["prediction_type"] == "real"].copy()
        if actuals_lookback is not None and len(actuals) > actuals_lookback:
            actuals = actuals.tail(actuals_lookback)

        ax.plot(
            actuals[time_col], actuals[value_col],
            color="tab:blue", linewidth=1.2, label="Actual",
        )

        # Predicted values
        if filter_holdout:
            sub = sub[sub[time_col] > max_real_date]  # only show predictions after last actual date
        preds = sub[sub["prediction_type"] == "predicted_value"]
        if not preds.empty:
            model_name = preds["model"].iloc[0] if "model" in preds.columns else "Forecast"
            ax.plot(
                preds[time_col], preds[value_col],
                color="tab:orange", linewidth=1.2, linestyle="--",
                label=f"Forecast",
            )

        # Confidence interval
        lower = sub[sub["prediction_type"] == "lower_bound"].sort_values(time_col)
        upper = sub[sub["prediction_type"] == "upper_bound"].sort_values(time_col)
        if not lower.empty and not upper.empty:
            ax.fill_between(
                lower[time_col],
                lower[value_col],
                upper[value_col],
                alpha=0.18,
                color="tab:orange",
                label="Confidence interval",
            )

        if plot_topline:
            ax.set_title("Forecast — Topline (all series)", fontsize=11)
        else:
            ax.set_title(f"Forecast — {_series_label(sf)}", fontsize=11)
        ax.set_ylabel(value_col)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        # set y_lim to start from 0
        ax.set_ylim(bottom=0)

    axes[-1, 0].set_xlabel(time_col)
    fig.tight_layout()
    plt.show()


def plot_forecast_comparison(
    df_baseline,
    df_whatif,
    time_col,
    value_col,
    identifier_cols,
    best_model_only=True,
    actuals_lookback=30,
    series_to_plot=None,
):
    """Plot baseline vs what-if forecast comparison (topline or selected series).

    Parameters
    ----------
    df_baseline : pd.DataFrame
        AICast baseline result.
    df_whatif : pd.DataFrame
        AICast what-if result.
    time_col : str
        Name of the date column.
    value_col : str
        Name of the value column.
    identifier_cols : list[str]
        Identifier columns.
    best_model_only : bool
        If True keep only model_ranking == 1 for each series.
    actuals_lookback : int | None
        Number of most recent actual data points to display. If None, show all.
    series_to_plot : list[dict] | None
        Explicit series filters. When None, plot topline (sum across all identifiers).

    Returns
    -------
    matplotlib.figure.Figure
    """
    def _prepare(df):
        d = df.copy()
        d[time_col] = pd.to_datetime(d[time_col])
        if best_model_only and "model_ranking" in d.columns:
            d = d[(d["real/pred"] == "real") | (d["model_ranking"] == 1) | (d["model_ranking"] == "1")]
        return d

    base = _prepare(df_baseline)
    whatif = _prepare(df_whatif)

    plot_topline = series_to_plot is None

    if plot_topline:
        agg_cols = [time_col, "prediction_type"]
        base_top = base.groupby(agg_cols, as_index=False).agg({value_col: "sum"})
        whatif_top = whatif.groupby(agg_cols, as_index=False).agg({value_col: "sum"})
        series_list = [{"_topline": True}]
    else:
        series_list = series_to_plot

    n = len(series_list)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n), squeeze=False)

    for i, sf in enumerate(series_list):
        ax = axes[i, 0]

        if plot_topline:
            sub_base = base_top
            sub_whatif = whatif_top
        else:
            sub_base = _filter_series(base, identifier_cols, sf)
            sub_whatif = _filter_series(whatif, identifier_cols, sf)

        sub_base = sub_base.sort_values(time_col)
        sub_whatif = sub_whatif.sort_values(time_col)

        # Actuals from baseline
        actuals = sub_base[sub_base["prediction_type"] == "real"].copy()
        if actuals_lookback is not None and len(actuals) > actuals_lookback:
            actuals = actuals.tail(actuals_lookback)
        if not actuals.empty:
            ax.plot(
                actuals[time_col], actuals[value_col],
                color="tab:gray", linewidth=1, alpha=0.8, label="Actual (baseline)",
            )
        max_real_date = actuals[time_col].max() if not actuals.empty else None
        # Baseline forecast
        sub_base = sub_base[sub_base[time_col] > max_real_date] if max_real_date is not None else sub_base
        preds_base = sub_base[sub_base["prediction_type"] == "predicted_value"]
        
        if not preds_base.empty:
            ax.plot(
                preds_base[time_col], preds_base[value_col],
                color="tab:blue", linewidth=1.2, label="Baseline forecast",
            )

        # What-if forecast
        sub_whatif = sub_whatif[sub_whatif[time_col] > max_real_date] if max_real_date is not None else sub_whatif
        preds_whatif = sub_whatif[sub_whatif["prediction_type"] == "predicted_value"]
        if not preds_whatif.empty:
            ax.plot(
                preds_whatif[time_col], preds_whatif[value_col],
                color="tab:orange", linewidth=1.2, linestyle="--", label="What-if forecast",
            )

        if plot_topline:
            ax.set_title("Forecast Comparison — Topline (all series)", fontsize=11)
        else:
            ax.set_title(f"Forecast Comparison — {_series_label(sf)}", fontsize=11)
        ax.set_ylabel(value_col)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_ylim(bottom=0)

    axes[-1, 0].set_xlabel(time_col)
    fig.tight_layout()
    return fig


def forecast_table(
    df_forecast,
    time_col,
    value_col,
    identifier_cols,
    best_model_only=True,
):
    """Pivot table of forecast results.

    Rows are identifier combinations.  Columns are forecasted dates with
    sub-columns for ``actual`` and ``forecast``.  Only dates that have a
    predicted value are included (i.e. the forecast horizon, not the full
    history).

    Parameters
    ----------
    best_model_only : bool
        If True keep only model_ranking == 1 for predicted rows.
    """
    df = df_forecast.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Keep best model only for predictions
    if best_model_only and "model_ranking" in df.columns:
        df = df[
            (df["prediction_type"] == "real")
            | (df["model_ranking"] == 1)
            | (df["model_ranking"] == "1")
        ]

    # Only keep actual and forecast prediction types
    keep_types = {"real": "actual", "predicted_value": "forecast"}
    df = df[df["prediction_type"].isin(keep_types.keys())].copy()
    df["prediction_type"] = df["prediction_type"].map(keep_types)

    # Restrict to forecasted dates only (dates that have a predicted value)
    forecast_dates = df.loc[df["prediction_type"] == "forecast", time_col].unique()
    df = df[df[time_col].isin(forecast_dates)]

    # Pivot: rows = identifier cols, columns = (date, prediction_type)
    table = df.pivot_table(
        index=identifier_cols,
        columns=[time_col, "prediction_type"],
        values=value_col,
        aggfunc="first",
    )

    # Sort date columns chronologically and keep MultiIndex
    table = table.sort_index(axis=1, level=0)

    # Rename date level to formatted strings for readability
    table.columns = pd.MultiIndex.from_tuples(
        [(dt.strftime("%Y-%m-%d"), ptype) for dt, ptype in table.columns],
        names=[time_col, "type"],
    )

    return table.reset_index()


# ---------------------------------------------------------------------------
# AI Predict visualizations
# ---------------------------------------------------------------------------


def _extract_feature_importance(fi_df, max_features=15):
    """Extract feature names and importance values from long or wide format.

    Returns (names, values) arrays sorted ascending by importance.
    Long format has columns: model, metric_name, feature_name, metric_value.
    """
    if "feature_name" in fi_df.columns and "metric_value" in fi_df.columns:
        agg = (
            fi_df.groupby("feature_name")["metric_value"]
            .mean()
            .sort_values(ascending=True)
            .tail(max_features)
        )
        return agg.index.astype(str).tolist(), agg.values
    name_col = [c for c in fi_df.columns if "feature" in c.lower()]
    val_col = fi_df.select_dtypes(include="number").columns.tolist()
    if name_col and val_col:
        fi_sorted = fi_df.sort_values(val_col[0], ascending=True).tail(max_features)
        return fi_sorted[name_col[0]].astype(str).tolist(), fi_sorted[val_col[0]].values
    fi_numeric = fi_df.select_dtypes(include="number")
    if not fi_numeric.empty:
        means = fi_numeric.mean().sort_values(ascending=True).tail(max_features)
        return means.index.astype(str).tolist(), means.values
    return [], []


def _get_pred_and_actual_cols(predictions_df):
    """Pick prediction and actual columns from the predictions table.

    The predictions table has one column per model plus a 'reals' column.
    Returns (pred_col, actual_col) or (None, None).
    """
    actual_col = "reals" if "reals" in predictions_df.columns else None
    model_cols = [c for c in predictions_df.columns if c != "reals"]
    pred_col = model_cols[0] if model_cols else None
    return pred_col, actual_col


def plot_classification_results(
    predictions_df,
    confusion_matrix_df=None,
    feature_importance_df=None,
    target_column="Survived",
    max_features=15,
):
    """Dashboard for classification: feature importance and probability histogram."""
    prob_cols = [c for c in predictions_df.columns if c.startswith("probability_")]

    panels = 0
    if confusion_matrix_df is not None and not confusion_matrix_df.empty:
        panels += 1
    if feature_importance_df is not None and not feature_importance_df.empty:
        panels += 1
    if prob_cols:
        panels += 1
    if panels == 0:
        panels = 1

    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 5), squeeze=False)
    ax_idx = 0

    # --- Confusion matrix heatmap ---
    if confusion_matrix_df is not None and not confusion_matrix_df.empty:
        ax = axes[0, ax_idx]
        cm = confusion_matrix_df.copy()
        label_col = [c for c in cm.columns
                     if c not in cm.select_dtypes(include="number").columns]
        if label_col:
            cm = cm.set_index(label_col[0])
        numeric_cols = cm.select_dtypes(include="number").columns
        cm_vals = cm[numeric_cols].values.astype(float)
        im = ax.imshow(cm_vals, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(cm.index)))
        ax.set_yticklabels(cm.index, fontsize=9)
        for r in range(cm_vals.shape[0]):
            for c_i in range(cm_vals.shape[1]):
                ax.text(
                    c_i, r, f"{cm_vals[r, c_i]:.0f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_vals[r, c_i] > cm_vals.max() / 2 else "black",
                )
        ax.set_title("Confusion Matrix", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax_idx += 1

    # --- Feature importance ---
    if feature_importance_df is not None and not feature_importance_df.empty:
        ax = axes[0, ax_idx]
        names, values = _extract_feature_importance(
            feature_importance_df, max_features,
        )
        if len(names):
            ax.barh(names, values, color="steelblue")
        ax.set_title("Feature Importance (avg across models)", fontsize=12)
        ax.set_xlabel("Importance")
        ax_idx += 1

    # --- Probability histogram ---
    if prob_cols:
        ax = axes[0, ax_idx]
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(prob_cols), 3)))
        for i, col in enumerate(prob_cols):
            class_name = col.replace("probability_", "")
            ax.hist(
                predictions_df[col].dropna(), bins=30,
                alpha=0.5, color=colors[i], edgecolor="white",
                label=f"Class {class_name}",
            )
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        ax.set_title("Predicted Probability Distribution", fontsize=12)
        ax.legend(fontsize=8)
        ax_idx += 1

    fig.suptitle(f"Classification — target: {target_column}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_regression_results(
    predictions_df,
    feature_importance_df=None,
    target_column="Fare",
    max_features=15,
):
    """Dashboard for regression: actual vs predicted, residuals, feature importance."""
    panels = 2
    if feature_importance_df is not None and not feature_importance_df.empty:
        panels += 1

    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 5), squeeze=False)
    ax_idx = 0

    pred_col, actual_col = _get_pred_and_actual_cols(predictions_df)
    if pred_col is None:
        for candidate in [
            "prediction_label", "Prediction", "prediction",
        ]:
            if candidate in predictions_df.columns:
                pred_col = candidate
                break
    if actual_col is None:
        actual_col = target_column if target_column in predictions_df.columns else None

    has_actual = actual_col is not None and actual_col in predictions_df.columns
    has_pred = pred_col is not None and pred_col in predictions_df.columns

    has_intervals = (
        "lower_bound" in predictions_df.columns
        and "upper_bound" in predictions_df.columns
    )

    # --- Actual vs Predicted scatter ---
    ax = axes[0, ax_idx]
    if has_actual and has_pred:
        actual = pd.to_numeric(predictions_df[actual_col], errors="coerce")
        predicted = pd.to_numeric(predictions_df[pred_col], errors="coerce")
        valid = actual.notna() & predicted.notna()

        if has_intervals:
            lower = pd.to_numeric(predictions_df["lower_bound"], errors="coerce")
            upper = pd.to_numeric(predictions_df["upper_bound"], errors="coerce")
            yerr_lower = (predicted[valid] - lower[valid]).clip(lower=0)
            yerr_upper = (upper[valid] - predicted[valid]).clip(lower=0)
            ax.errorbar(
                actual[valid], predicted[valid],
                yerr=[yerr_lower, yerr_upper],
                fmt="none", ecolor="lightcoral", alpha=0.4, elinewidth=1.2,
                capsize=0, label="Prediction interval",
            )

        ax.scatter(actual[valid], predicted[valid], alpha=0.5, s=18, color="steelblue", zorder=3)
        lims = [
            min(actual[valid].min(), predicted[valid].min()),
            max(actual[valid].max(), predicted[valid].max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
        ax.set_xlabel(f"Actual ({actual_col})")
        ax.set_ylabel(f"Predicted ({pred_col})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Columns not found", transform=ax.transAxes, ha="center")
    ax.set_title("Actual vs Predicted", fontsize=12)
    ax_idx += 1

    # --- Residuals ---
    ax = axes[0, ax_idx]
    if has_actual and has_pred:
        residuals = predicted[valid] - actual[valid]
        ax.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (pred - actual)")
        ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution", fontsize=12)
    ax_idx += 1

    # --- Feature importance ---
    if feature_importance_df is not None and not feature_importance_df.empty:
        ax = axes[0, ax_idx]
        names, values = _extract_feature_importance(
            feature_importance_df, max_features,
        )
        if len(names):
            ax.barh(names, values, color="steelblue")
        ax.set_title("Feature Importance (avg across models)", fontsize=12)
        ax.set_xlabel("Importance")

    fig.suptitle(f"Regression — target: {target_column}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_forecast_edits(
    dataframes,
    time_col="date",
    value_col="quantity",
    filter_dict=None,
    title="Forecast: Original vs Edits",
    prediction_type_col="prediction_type",
    pred_value="real",
    figsize=(10, 5),
):
    """Compare multiple forecast DataFrames on a single chart.

    Parameters
    ----------
    dataframes : list[tuple[DataFrame, str, str]]
        Each element is ``(df, label, linestyle)``.
    time_col, value_col : str
        Column names for the time axis and value axis.
    filter_dict : dict | None
        Optional column→value filters applied to every DataFrame
        (e.g. ``{"region": "West"}``).
    title : str
        Plot title.
    prediction_type_col : str
        Column that distinguishes actuals from predictions.
    pred_value : str
        The value in *prediction_type_col* that marks actuals (rows to exclude).
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    first_df = dataframes[0][0]
    max_real = first_df.loc[first_df[prediction_type_col] == pred_value, time_col].max()
    # filter holdout if max_real exists
    if max_real:
        dataframes = [
            (df[df[time_col] > max_real], label, ls)
            for df, label, ls in dataframes
        ]
    for df, label, ls in dataframes:
        subset = df[df[prediction_type_col] != pred_value].copy()
        if filter_dict:
            for col, val in filter_dict.items():
                subset = subset[subset[col] == val]
        grp = subset.groupby(time_col)[value_col].sum()
        ax.plot(grp.index, grp.values, label=label, linestyle=ls, linewidth=1.5)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EDA — Tabular visualizations
# ---------------------------------------------------------------------------


def plot_numeric_statistics(numeric_stats_df):
    """Grouped bar chart of numeric column statistics.

    Handles the platform output format where rows are statistics
    (Statistic column) and each numeric column is a feature.
    Also handles the transposed layout where features are rows.
    """
    df = numeric_stats_df.copy()

    stat_col = None
    for candidate in ["Statistic", "statistic", "stat"]:
        if candidate in df.columns:
            stat_col = candidate
            break

    if stat_col:
        display_rows = ["mean", "std", "min", "max", "median", "25%", "50%", "75%"]
        mask = df[stat_col].isin(display_rows)
        if mask.sum() == 0:
            sub = df.head(8)
        else:
            sub = df[mask]

        feature_cols = [c for c in df.columns if c != stat_col]
        numeric_features = []
        for c in feature_cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                numeric_features.append(c)

        n_features = len(numeric_features)
        n_stats = len(sub)
        x = np.arange(n_features)
        width = 0.8 / max(n_stats, 1)

        fig, ax = plt.subplots(figsize=(max(10, n_features * 1.5), 5))
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_stats, 3)))

        for i, (_, row) in enumerate(sub.iterrows()):
            vals = [pd.to_numeric(row[c], errors="coerce") for c in numeric_features]
            vals = [0 if pd.isna(v) else v for v in vals]
            ax.bar(x + i * width, vals, width, label=row[stat_col], color=colors[i % len(colors)])

        ax.set_xticks(x + width * (n_stats - 1) / 2)
        ax.set_xticklabels(numeric_features, rotation=45, ha="right", fontsize=9)
        ax.set_title("Numeric Column Statistics", fontsize=13)
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    name_col = df.columns[0]
    stat_cols = df.select_dtypes(include="number").columns.tolist()
    display_stats = [c for c in ["mean", "std", "min", "max", "median"] if c in stat_cols]
    if not display_stats:
        display_stats = stat_cols[:6]

    n_cols = len(df)
    n_stats = len(display_stats)
    x = np.arange(n_cols)
    width = 0.8 / max(n_stats, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.2), 5))
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_stats, 3)))

    for i, stat in enumerate(display_stats):
        vals = pd.to_numeric(df[stat], errors="coerce").fillna(0)
        ax.bar(x + i * width, vals, width, label=stat, color=colors[i])

    ax.set_xticks(x + width * (n_stats - 1) / 2)
    ax.set_xticklabels(df[name_col], rotation=45, ha="right", fontsize=9)
    ax.set_title("Numeric Column Statistics", fontsize=13)
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_heatmap_df):
    """Heatmap from the EDA correlation_heatmap output.

    Handles both wide format (square matrix) and long format
    (Column Name, variable, value) as returned by the platform.
    """
    df = corr_heatmap_df.copy()

    row_col = None
    var_col = None
    val_col = None
    for candidate in ["Column Name", "column_name", "Column", "column", "index"]:
        if candidate in df.columns:
            row_col = candidate
            break
    if row_col and "variable" in df.columns and "value" in df.columns:
        var_col = "variable"
        val_col = "value"

    if row_col and var_col and val_col:
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        pivot = df.pivot(index=row_col, columns=var_col, values=val_col)
        labels = pivot.index.tolist()
        col_labels = pivot.columns.tolist()
        vals = pivot.values.astype(float)
    else:
        if row_col:
            labels = df[row_col].astype(str).tolist()
            matrix = df.drop(columns=[row_col]).select_dtypes(include="number")
        else:
            labels = df.columns.tolist()
            matrix = df.select_dtypes(include="number")
        col_labels = matrix.columns.tolist()
        vals = matrix.values.astype(float)

    n_rows, n_cols = vals.shape

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.9), max(6, n_rows * 0.8)))
    im = ax.imshow(vals, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=9)

    for r in range(n_rows):
        for c in range(n_cols):
            if not np.isnan(vals[r, c]):
                ax.text(c, r, f"{vals[r, c]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(vals[r, c]) > 0.5 else "black")

    ax.set_title("Correlation Heatmap", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Correlation")
    fig.tight_layout()
    return fig


def plot_numeric_histograms(numeric_hist_df, max_columns=6):
    """Plot histogram data from the EDA numeric_histogram output.

    Handles the platform format (Column Name, value) where raw values
    are listed per column. Groups by Column Name and plots a histogram
    for each numeric column.
    """
    df = numeric_hist_df.copy()

    name_col = None
    for candidate in ["Column Name", "column_name", "Column", "column", "feature"]:
        if candidate in df.columns:
            name_col = candidate
            break

    val_col = None
    for candidate in ["value", "Value"]:
        if candidate in df.columns:
            val_col = candidate
            break

    if name_col and val_col:
        columns = df[name_col].unique()[:max_columns]
        n = len(columns)
        cols_per_row = min(n, 3)
        rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows), squeeze=False)

        for i, col_name in enumerate(columns):
            ax = axes[i // cols_per_row, i % cols_per_row]
            subset = df[df[name_col] == col_name]
            vals = pd.to_numeric(subset[val_col], errors="coerce").dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=min(30, len(vals)), color="steelblue", edgecolor="white")
            ax.set_title(col_name, fontsize=10)
            ax.set_ylabel("Count")

        for j in range(i + 1, rows * cols_per_row):
            axes[j // cols_per_row, j % cols_per_row].set_visible(False)
    elif name_col and "count" in df.columns:
        columns = df[name_col].unique()[:max_columns]
        n = len(columns)
        cols_per_row = min(n, 3)
        rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows), squeeze=False)

        for i, col_name in enumerate(columns):
            ax = axes[i // cols_per_row, i % cols_per_row]
            subset = df[df[name_col] == col_name]
            y = pd.to_numeric(subset["count"], errors="coerce")
            ax.bar(range(len(y)), y, color="steelblue", edgecolor="white")
            ax.set_title(col_name, fontsize=10)
            ax.set_ylabel("Count")

        for j in range(i + 1, rows * cols_per_row):
            axes[j // cols_per_row, j % cols_per_row].set_visible(False)
    else:
        numeric_cols = df.select_dtypes(include="number").columns[:max_columns]
        n = len(numeric_cols)
        cols_per_row = min(n, 3)
        rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows), squeeze=False)

        for i, col in enumerate(numeric_cols):
            ax = axes[i // cols_per_row, i % cols_per_row]
            ax.hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="white")
            ax.set_title(col, fontsize=10)
            ax.set_ylabel("Count")

        for j in range(i + 1, rows * cols_per_row):
            axes[j // cols_per_row, j % cols_per_row].set_visible(False)

    fig.suptitle("Numeric Histograms", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_text_statistics(text_stats_df):
    """Bar chart of text column statistics.

    Handles the platform format where rows are statistics (Statistic column)
    and text columns are separate DataFrame columns.
    Shows number_of_distinct for each text column.
    """
    df = text_stats_df.copy()

    stat_col = None
    for candidate in ["Statistic", "statistic", "stat"]:
        if candidate in df.columns:
            stat_col = candidate
            break

    if stat_col:
        feature_cols = [c for c in df.columns if c != stat_col]
        distinct_row = df[df[stat_col].str.contains("distinct", case=False, na=False)]
        if len(distinct_row) > 0:
            row = distinct_row.iloc[0]
            vals = [pd.to_numeric(row[c], errors="coerce") for c in feature_cols]
            vals = [0 if pd.isna(v) else v for v in vals]

            fig, ax = plt.subplots(figsize=(max(8, len(feature_cols) * 1.2), 5))
            ax.barh(feature_cols, vals, color="steelblue")
            ax.set_xlabel("Distinct Count")
            ax.set_title("Text Columns — Number of Distinct Values", fontsize=13)
            fig.tight_layout()
            return fig

    name_col = df.columns[0]
    unique_col = None
    for candidate in ["unique", "unique_count", "nunique", "distinct"]:
        if candidate in df.columns:
            unique_col = candidate
            break

    if unique_col:
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.0), 5))
        vals = pd.to_numeric(df[unique_col], errors="coerce").fillna(0)
        ax.barh(df[name_col].astype(str), vals, color="steelblue")
        ax.set_xlabel("Unique Count")
        ax.set_title("Text Column — Unique Value Counts", fontsize=13)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    stat_cols = df.select_dtypes(include="number").columns.tolist()[:4]
    if stat_cols:
        x = np.arange(len(df))
        width = 0.8 / max(len(stat_cols), 1)
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(stat_cols), 3)))
        for i, col in enumerate(stat_cols):
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            ax.barh(x + i * width, vals, width, label=col, color=colors[i])
        ax.set_yticks(x + width * (len(stat_cols) - 1) / 2)
        ax.set_yticklabels(df[name_col].astype(str), fontsize=9)
        ax.legend(fontsize=8)
    ax.set_title("Text Column Statistics", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EDA — Time-Series ACF visualization
# ---------------------------------------------------------------------------


def plot_acf(acf_df, max_series=4):
    """Plot autocorrelation function (ACF) from the EDA time-series output.

    Expects columns: acf_values, lags, plus an identifier column (e.g. sku)
    and optionally acf_confidence_lower_bound / acf_confidence_upper_bound.
    """
    df = acf_df.copy()

    lag_col = None
    for candidate in ["lags", "lag", "Lag"]:
        if candidate in df.columns:
            lag_col = candidate
            break

    acf_col = None
    for candidate in ["acf_values", "acf", "ACF", "autocorrelation"]:
        if candidate in df.columns:
            acf_col = candidate
            break

    conf_lower = "acf_confidence_lower_bound" if "acf_confidence_lower_bound" in df.columns else None
    conf_upper = "acf_confidence_upper_bound" if "acf_confidence_upper_bound" in df.columns else None

    id_col = None
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        id_col = non_numeric[0]

    if lag_col is None or acf_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            acf_col = numeric_cols[0]
            lag_col = numeric_cols[-1]

    if id_col:
        series_ids = df[id_col].unique()[:max_series]
        n = len(series_ids)
    else:
        series_ids = [None]
        n = 1

    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), squeeze=False)

    for i, sid in enumerate(series_ids):
        ax = axes[i, 0]
        subset = df[df[id_col] == sid] if id_col and sid is not None else df
        subset = subset.sort_values(lag_col) if lag_col else subset

        if lag_col:
            lags = pd.to_numeric(subset[lag_col], errors="coerce")
        else:
            lags = np.arange(len(subset))

        values = pd.to_numeric(subset[acf_col], errors="coerce")

        ax.bar(lags, values, width=0.8, color="steelblue", edgecolor="white")

        if conf_lower and conf_upper:
            lower = pd.to_numeric(subset[conf_lower], errors="coerce")
            upper = pd.to_numeric(subset[conf_upper], errors="coerce")
            ax.fill_between(lags, lower, upper, alpha=0.15, color="red", label="95% confidence")
        else:
            n_obs = len(subset)
            if n_obs > 0:
                conf = 1.96 / np.sqrt(n_obs)
                ax.axhline(conf, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.axhline(-conf, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.axhline(0, color="black", linewidth=0.5)

        title = f"ACF — {sid}" if sid is not None else "Autocorrelation Function (ACF)"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        if conf_lower:
            ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
