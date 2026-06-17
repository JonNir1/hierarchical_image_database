"""
Pilot analysis figures for the SpAM task.

All functions accept DataFrames returned by analysis.pilot.parser.load_pilot_data
and return plotly Figure objects.

Usage (from repo root):
    from analysis.pilot.parser import load_pilot_data
    from analysis.pilot.figures import (
        fig_completion_status, fig_trial_duration_per_subject,
        fig_moves_per_subject, fig_duration_progression,
        fig_moves_progression, fig_duration_vs_moves,
        fig_within_subject_variability, fig_demographics,
        fig_pairwise_distance_distribution,
    )
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, pearsonr

# ---------------------------------------------------------------------------
# Version colour palette
# ---------------------------------------------------------------------------

_VERSION_PALETTE: dict[float, dict[str, str]] = {
    1.0: {
        "main":         "#1a5296",
        "violin":       "rgba(100,149,237,0.35)",
        "violin_line":  "rgba(100,149,237,0.9)",
        "subj":         "rgba(100,149,237,0.2)",
        "dot":          "rgba(30,80,160,0.9)",
        "dot_err":      "rgba(30,80,160,0.6)",
    },
    2.0: {
        "main":         "#922b21",
        "violin":       "rgba(220,80,60,0.35)",
        "violin_line":  "rgba(220,80,60,0.9)",
        "subj":         "rgba(220,80,60,0.2)",
        "dot":          "rgba(160,40,30,0.9)",
        "dot_err":      "rgba(160,40,30,0.6)",
    },
}

_PALETTE_FALLBACK: dict[str, str] = {
    "main":         "#555555",
    "violin":       "rgba(85,85,85,0.35)",
    "violin_line":  "rgba(85,85,85,0.9)",
    "subj":         "rgba(85,85,85,0.2)",
    "dot":          "rgba(85,85,85,0.9)",
    "dot_err":      "rgba(85,85,85,0.6)",
}


def _vc(version) -> dict[str, str]:
    return _VERSION_PALETTE.get(float(version), _PALETTE_FALLBACK)


def _vlabel(version) -> str:
    return f"v{version:g}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RT_TO_S = 1 / 1000  # milliseconds → seconds


def _rt_s(df: pd.DataFrame) -> pd.Series:
    return df["rt"] * _RT_TO_S


def _subject_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-subject mean and SD for *col*."""
    return (
        df.groupby("participant_id")[col]
        .agg(mean="mean", std="std")
        .reset_index()
    )


def _subject_se(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-subject mean and SE for *col*."""
    grouped = df.groupby("participant_id")[col]
    return (
        grouped.agg(mean="mean", std="std", n="count")
        .assign(se=lambda x: x["std"] / np.sqrt(x["n"]))
        .reset_index()
    )


def _parse_pairwise(pw_json: str) -> dict[tuple[str, str], float]:
    """Parse one trial's pairwise_distances JSON into {sorted_pair: distance}."""
    if pd.isna(pw_json) or pw_json == "":
        return {}
    try:
        items = json.loads(pw_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return {
        tuple(sorted([item["src1"], item["src2"]])): item["distance"]
        for item in items
    }


def _repeated_pair_distances(df_subject: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    For a single subject's trials, find pairs observed in ≥ 2 trials.
    Returns (d_first, d_second) — lists of paired distance values for Pearson r.
    Pairs observed > 2 times: take first two observations.
    """
    pair_obs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for pw_json in df_subject["pairwise_distances"]:
        for pair, dist in _parse_pairwise(pw_json).items():
            pair_obs[pair].append(dist)

    d1, d2 = [], []
    for obs in pair_obs.values():
        if len(obs) >= 2:
            d1.append(obs[0])
            d2.append(obs[1])
    return d1, d2


def _sorted_versions(df: pd.DataFrame) -> list[float]:
    return sorted(df["task_version"].unique().tolist())


# ---------------------------------------------------------------------------
# Fig 1 — Completion status distribution
# ---------------------------------------------------------------------------


def fig_completion_status(df_status: pd.DataFrame) -> go.Figure:
    counts = df_status["completion_status"].value_counts()
    total = counts.sum()
    pcts = (counts / total * 100).round(1)

    label_map = {
        "completed": "Completed",
        "revoked_consent": "Revoked consent",
        "erroneous_completion": "Erroneous completion",
    }
    color_map = {
        "completed": "#2ecc71",
        "revoked_consent": "#e74c3c",
        "erroneous_completion": "#f39c12",
    }

    categories = [k for k in label_map if k in counts.index]
    fig = go.Figure(go.Bar(
        x=[pcts.get(k, 0) for k in categories],
        y=[label_map[k] for k in categories],
        orientation="h",
        text=[f"{pcts.get(k, 0):.1f}% (n={counts.get(k, 0)})" for k in categories],
        textposition="outside",
        marker_color=[color_map[k] for k in categories],
    ))
    fig.update_layout(
        title="Participant completion status",
        xaxis_title="% of participants",
        xaxis_range=[0, 100],
        yaxis={"autorange": "reversed"},
        height=250,
        margin={"l": 160, "r": 80, "t": 50, "b": 50},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 2 — Trial duration per subject (horizontal violin + dots ± SD)
# ---------------------------------------------------------------------------


def fig_trial_duration_per_subject(df_trials: pd.DataFrame) -> go.Figure:
    versions = _sorted_versions(df_trials)
    multi = len(versions) > 1
    fig = go.Figure()

    for i, v in enumerate(versions):
        vc = _vc(v)
        vdf = df_trials[df_trials["task_version"] == v]
        rt_s = _rt_s(vdf)
        stats = _subject_stats(vdf.assign(rt_s=rt_s), "rt_s")
        stats = stats.sort_values("mean").reset_index(drop=True)
        y_dots = i + np.linspace(-0.15, 0.15, len(stats))

        fig.add_trace(go.Violin(
            x=rt_s,
            y0=i,
            orientation="h",
            side="both",
            fillcolor=vc["violin"],
            line_color=vc["violin_line"],
            name=_vlabel(v),
            showlegend=multi,
            width=0.6,
            points=False,
        ))
        fig.add_trace(go.Scatter(
            x=stats["mean"],
            y=y_dots.tolist(),
            mode="markers",
            marker={"color": vc["dot"], "size": 8, "symbol": "circle"},
            error_x={"type": "data", "array": stats["std"].tolist(), "visible": True,
                     "color": vc["dot_err"], "thickness": 1.5, "width": 4},
            showlegend=False,
            name=f"{_vlabel(v)} subject mean ± SD",
        ))

    y_range = [-0.55, len(versions) - 0.45]
    y_axis: dict = {"range": y_range}
    if multi:
        y_axis.update({
            "tickvals": list(range(len(versions))),
            "ticktext": [_vlabel(v) for v in versions],
        })
    else:
        y_axis["visible"] = False

    fig.update_layout(
        title="Trial duration per subject",
        xaxis_title="Trial duration (s)",
        yaxis=y_axis,
        height=300 + 100 * len(versions),
        margin={"l": 60, "r": 40, "t": 50, "b": 60},
        showlegend=multi,
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 3 — Moves per subject (horizontal violin + dots ± SD)
# ---------------------------------------------------------------------------


def fig_moves_per_subject(df_trials: pd.DataFrame) -> go.Figure:
    versions = _sorted_versions(df_trials)
    multi = len(versions) > 1
    fig = go.Figure()

    for i, v in enumerate(versions):
        vc = _vc(v)
        vdf = df_trials[df_trials["task_version"] == v]
        stats = _subject_stats(vdf, "n_moves")
        stats = stats.sort_values("mean").reset_index(drop=True)
        y_dots = i + np.linspace(-0.15, 0.15, len(stats))

        fig.add_trace(go.Violin(
            x=vdf["n_moves"],
            y0=i,
            orientation="h",
            side="both",
            fillcolor=vc["violin"],
            line_color=vc["violin_line"],
            name=_vlabel(v),
            showlegend=multi,
            width=0.6,
            points=False,
        ))
        fig.add_trace(go.Scatter(
            x=stats["mean"],
            y=y_dots.tolist(),
            mode="markers",
            marker={"color": vc["dot"], "size": 8, "symbol": "circle"},
            error_x={"type": "data", "array": stats["std"].tolist(), "visible": True,
                     "color": vc["dot_err"], "thickness": 1.5, "width": 4},
            showlegend=False,
            name=f"{_vlabel(v)} subject mean ± SD",
        ))

    y_range = [-0.55, len(versions) - 0.45]
    y_axis: dict = {"range": y_range}
    if multi:
        y_axis.update({
            "tickvals": list(range(len(versions))),
            "ticktext": [_vlabel(v) for v in versions],
        })
    else:
        y_axis["visible"] = False

    fig.update_layout(
        title="Number of moves per subject",
        xaxis_title="Moves per trial",
        yaxis=y_axis,
        height=300 + 100 * len(versions),
        margin={"l": 60, "r": 40, "t": 50, "b": 60},
        showlegend=multi,
    )
    return fig


# ---------------------------------------------------------------------------
# Shared helper for progression plots (Figs 4 & 5)
# ---------------------------------------------------------------------------


def _fig_progression(
    df_trials: pd.DataFrame,
    col: str,
    y_label: str,
    title: str,
) -> go.Figure:
    """Line plot of *col* over trial_number with mean±SE per task_version."""
    df = df_trials.copy()
    df["_y"] = df[col]
    versions = _sorted_versions(df)
    multi = len(versions) > 1

    fig = go.Figure()
    for v in versions:
        vc = _vc(v)
        vdf = df[df["task_version"] == v]

        # Individual subject traces
        for _pid, grp in vdf.groupby("participant_id"):
            grp_sorted = grp.sort_values("trial_number")
            fig.add_trace(go.Scatter(
                x=grp_sorted["trial_number"],
                y=grp_sorted["_y"],
                mode="lines",
                line={"color": vc["subj"], "width": 1},
                showlegend=False,
                hoverinfo="skip",
            ))

        # Mean ± SE
        agg = (
            vdf.groupby("trial_number")["_y"]
            .agg(mean="mean", std="std", n="count")
            .assign(se=lambda x: x["std"] / np.sqrt(x["n"]))
            .reset_index()
        )
        x = agg["trial_number"].tolist()
        mean = agg["mean"].tolist()
        se = agg["se"].tolist()

        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[m + s for m, s in zip(mean, se)] + [m - s for m, s in zip(mean[::-1], se[::-1])],
            fill="toself",
            fillcolor=vc["main"],
            opacity=0.2,
            line={"color": "rgba(0,0,0,0)"},
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            line={"color": vc["main"], "width": 3},
            marker={"size": 7, "color": vc["main"]},
            name=f"{_vlabel(v)} mean ± SE" if multi else "Mean ± SE",
        ))

    fig.update_layout(
        title=title,
        xaxis={"title": "Trial number", "tickvals": list(range(1, 11))},
        yaxis_title=y_label,
        height=400,
        margin={"l": 60, "r": 40, "t": 50, "b": 60},
        showlegend=multi,
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 4 — Trial duration over task progression
# ---------------------------------------------------------------------------


def fig_duration_progression(df_trials: pd.DataFrame) -> go.Figure:
    df = df_trials.assign(rt_s=_rt_s(df_trials))
    return _fig_progression(df, "rt_s", "Trial duration (s)", "Trial duration over task progression")


# ---------------------------------------------------------------------------
# Fig 5 — Moves over task progression
# ---------------------------------------------------------------------------


def fig_moves_progression(df_trials: pd.DataFrame) -> go.Figure:
    return _fig_progression(
        df_trials, "n_moves", "Moves per trial", "Number of moves over task progression",
    )


# ---------------------------------------------------------------------------
# Fig 6 — Duration vs moves scatter (per subject, coloured by version)
# ---------------------------------------------------------------------------


def fig_duration_vs_moves(df_trials: pd.DataFrame) -> go.Figure:
    df = df_trials.assign(rt_s=_rt_s(df_trials))
    versions = _sorted_versions(df)
    multi = len(versions) > 1

    fig = go.Figure()
    for v in versions:
        vc = _vc(v)
        vdf = df[df["task_version"] == v]
        stats = (
            vdf.groupby("participant_id")
            .agg(
                mean_rt=("rt_s", "mean"),
                sd_rt=("rt_s", "std"),
                mean_moves=("n_moves", "mean"),
                sd_moves=("n_moves", "std"),
            )
            .reset_index()
        )
        fig.add_trace(go.Scatter(
            x=stats["mean_moves"],
            y=stats["mean_rt"],
            mode="markers",
            name=_vlabel(v) if multi else "subjects",
            marker={"size": 9, "color": vc["dot"], "opacity": 0.85},
            error_x={"type": "data", "array": stats["sd_moves"].tolist(),
                     "visible": True, "color": vc["dot_err"], "thickness": 1.5},
            error_y={"type": "data", "array": stats["sd_rt"].tolist(),
                     "visible": True, "color": vc["dot_err"], "thickness": 1.5},
            text=stats["participant_id"],
            hovertemplate="<b>%{text}</b><br>Moves: %{x:.1f}<br>Duration: %{y:.1f}s<extra></extra>",
        ))

    fig.update_layout(
        title="Trial duration vs. number of moves (per subject, mean ± SD)",
        xaxis_title="Moves per trial",
        yaxis_title="Trial duration (s)",
        height=450,
        margin={"l": 70, "r": 40, "t": 50, "b": 60},
        showlegend=multi,
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 7 — Within-subject variability + reliability (coloured by version)
# ---------------------------------------------------------------------------


def fig_within_subject_variability(df_trials: pd.DataFrame) -> go.Figure:
    """
    Subplot A — SNR per subject: σ_d / mean(|Δd|).
    Subplot B — Within-subject reliability: Pearson r between d1 and d2 of repeated pairs.
    Both subplots colour subjects by task_version.
    """
    subjects = sorted(df_trials["participant_id"].unique())
    subj_short = {pid: f"S{i+1:02d}" for i, pid in enumerate(subjects)}

    records: list[dict] = []
    for pid in subjects:
        df_s = df_trials[df_trials["participant_id"] == pid]
        d1, d2 = _repeated_pair_distances(df_s)
        if len(d1) == 0:
            continue

        all_dists = [
            dist
            for pw_json in df_s["pairwise_distances"]
            for dist in _parse_pairwise(pw_json).values()
        ]
        sigma_d = float(np.std(all_dists)) if len(all_dists) > 1 else 1.0

        abs_diffs = np.abs(np.array(d1) - np.array(d2))
        mean_abs_diff = float(np.mean(abs_diffs))
        sd_abs_diff = float(np.std(abs_diffs))
        snr = sigma_d / mean_abs_diff if mean_abs_diff > 0 else np.nan
        snr_hi = sigma_d / (mean_abs_diff - sd_abs_diff) if mean_abs_diff > sd_abs_diff else np.nan
        snr_lo = sigma_d / (mean_abs_diff + sd_abs_diff) if (mean_abs_diff + sd_abs_diff) > 0 else np.nan

        r, _ = pearsonr(d1, d2)
        v = float(df_s["task_version"].iloc[0])
        records.append({
            "label":         subj_short[pid],
            "pid":           pid,
            "version":       v,
            "snr":           snr,
            "snr_err_plus":  snr_hi - snr if not np.isnan(snr_hi) else 0.0,
            "snr_err_minus": snr - snr_lo if not np.isnan(snr_lo) else 0.0,
            "r":             r,
        })

    versions = sorted({rec["version"] for rec in records})
    multi = len(versions) > 1

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=[
            "Signal-to-noise ratio per subject",
            "Within-subject reliability (Pearson r)",
        ],
    )

    # Subplot A: forest-plot SNR dots, coloured by version
    shown_versions_A: set[float] = set()
    for rec in records:
        vc = _vc(rec["version"])
        show_leg = multi and rec["version"] not in shown_versions_A
        shown_versions_A.add(rec["version"])
        fig.add_trace(
            go.Scatter(
                x=[rec["snr"]],
                y=[rec["label"]],
                mode="markers",
                marker={"color": vc["dot"], "size": 8},
                error_x={
                    "type": "data",
                    "array": [rec["snr_err_plus"]],
                    "arrayminus": [rec["snr_err_minus"]],
                    "visible": True,
                    "color": vc["dot_err"],
                    "thickness": 1.5,
                    "width": 5,
                },
                showlegend=show_leg,
                name=_vlabel(rec["version"]) if show_leg else None,
                legendgroup=_vlabel(rec["version"]),
                hovertemplate=f"<b>{rec['pid']}</b><br>SNR = %{{x:.3f}}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Subplot B: one box per version + coloured dots
    rng = np.random.default_rng(42)
    n_versions = len(versions)
    for j, v in enumerate(versions):
        vc = _vc(v)
        vrecs = [rec for rec in records if rec["version"] == v]
        rs = [rec["r"] for rec in vrecs]
        pids = [rec["pid"] for rec in vrecs]
        labels = [rec["label"] for rec in vrecs]
        x_pos = j / max(n_versions - 1, 1) if n_versions > 1 else 0

        fig.add_trace(
            go.Box(
                y=rs,
                x=[x_pos] * len(rs),
                boxpoints=False,
                name=_vlabel(v),
                marker_color=vc["main"],
                line_color=vc["main"],
                fillcolor=vc["violin"],
                showlegend=False,
                legendgroup=_vlabel(v),
                hoverinfo="skip",
                width=0.25,
            ),
            row=1, col=2,
        )
        x_jitter = (x_pos + rng.uniform(-0.08, 0.08, size=len(rs))).tolist()
        fig.add_trace(
            go.Scatter(
                x=x_jitter,
                y=rs,
                mode="markers",
                marker={"color": vc["dot"], "size": 9, "opacity": 0.8},
                showlegend=False,
                legendgroup=_vlabel(v),
                text=[f"{s} ({p})" for s, p in zip(labels, pids)],
                hovertemplate="<b>%{text}</b><br>r = %{y:.3f}<extra></extra>",
            ),
            row=1, col=2,
        )

    all_rs = [rec["r"] for rec in records]
    r_min, r_max = min(all_rs), max(all_rs)
    r_pad = max((r_max - r_min) * 0.2, 0.05)
    fig.update_yaxes(range=[r_min - r_pad, r_max + r_pad], row=1, col=2)

    if multi and n_versions > 1:
        fig.update_xaxes(
            tickvals=[j / (n_versions - 1) for j in range(n_versions)],
            ticktext=[_vlabel(v) for v in versions],
            row=1, col=2,
        )
    else:
        fig.update_xaxes(showticklabels=False, row=1, col=2)

    fig.update_xaxes(title_text="SNR = σ_d / mean(|Δd|)", row=1, col=1)
    fig.update_yaxes(title_text="Subject", row=1, col=1)
    fig.update_yaxes(title_text="Pearson r", row=1, col=2)
    fig.update_layout(
        title="Within-subject variability and reliability of repeated pairs",
        height=500,
        margin={"l": 70, "r": 40, "t": 70, "b": 60},
        showlegend=multi,
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 8 — Participant demographics
# ---------------------------------------------------------------------------


def fig_demographics(df_trials: pd.DataFrame) -> go.Figure:
    df_subj = df_trials.drop_duplicates("participant_id")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Age distribution", "Sex", "Ethnicity", "Country of residence"],
    )

    ages = df_subj["age"].dropna()
    fig.add_trace(
        go.Histogram(x=ages, nbinsx=10, marker_color="#2ecc71", showlegend=False, name="Age"),
        row=1, col=1,
    )

    sex_counts = df_subj["sex"].dropna().value_counts()
    fig.add_trace(
        go.Bar(x=sex_counts.values.tolist(), y=sex_counts.index.tolist(),
               orientation="h", marker_color="#3498db", showlegend=False, name="Sex"),
        row=1, col=2,
    )

    eth_counts = df_subj["ethnicity"].dropna().value_counts()
    fig.add_trace(
        go.Bar(x=eth_counts.values.tolist(), y=eth_counts.index.tolist(),
               orientation="h", marker_color="#9b59b6", showlegend=False, name="Ethnicity"),
        row=2, col=1,
    )

    country_counts = df_subj["country_of_residence"].dropna().value_counts().head(10)
    fig.add_trace(
        go.Bar(x=country_counts.values.tolist(), y=country_counts.index.tolist(),
               orientation="h", marker_color="#e67e22", showlegend=False, name="Country"),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Count", row=2, col=2)

    fig.update_layout(
        title="Participant demographics (completed submissions)",
        height=600,
        margin={"l": 120, "r": 40, "t": 80, "b": 60},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig C — Pairwise distance distribution (overlaid by version + optional null)
# ---------------------------------------------------------------------------


def fig_pairwise_distance_distribution(
    df_trials: pd.DataFrame,
    null_distribution: np.ndarray | None = None,
) -> go.Figure:
    """
    Parameters
    ----------
    df_trials:
        Trials DataFrame from load_pilot_data.
    null_distribution:
        Optional 1-D array of distances from the random-placement null
        (e.g. from simulate_null_distances.simulate). When provided,
        renders a two-panel figure: top panel = KDE density curves for each
        version + null as a grey filled area; bottom panel = per-version
        deviation from null (KDE_version − KDE_null).
    """
    versions = _sorted_versions(df_trials)
    multi = len(versions) > 1

    # Collect per-version distances
    version_dists: dict[float, list[float]] = {}
    total_obs = 0
    for v in versions:
        vdf = df_trials[df_trials["task_version"] == v]
        dists = [
            dist
            for pw_json in vdf["pairwise_distances"]
            for dist in _parse_pairwise(pw_json).values()
        ]
        version_dists[v] = dists
        total_obs += len(dists)

    # Common evaluation grid over the observed range
    x = np.linspace(0.0, 1.0, 500)

    # KDE for each version and for the null
    version_kde: dict[float, np.ndarray] = {
        v: gaussian_kde(dists)(x)
        for v, dists in version_dists.items()
        if len(dists) > 1
    }
    null_kde: np.ndarray | None = (
        gaussian_kde(null_distribution)(x) if null_distribution is not None else None
    )

    two_panel = null_kde is not None
    title = f"Distribution of pairwise SpAM distances (n={total_obs:,} pair observations)"

    if two_panel:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.60, 0.40],
            shared_xaxes=True,
            vertical_spacing=0.10,
        )

        # Null filled area (bottom layer — add first so version lines render on top)
        fig.add_trace(go.Scatter(
            x=x.tolist(), y=null_kde.tolist(),
            fill="tozeroy",
            fillcolor="rgba(160,160,160,0.30)",
            line={"color": "rgba(100,100,100,0.70)", "width": 1.5},
            name="Null (random placement)",
        ), row=1, col=1)

        # Version KDE lines
        for v in versions:
            if v not in version_kde:
                continue
            vc = _vc(v)
            fig.add_trace(go.Scatter(
                x=x.tolist(), y=version_kde[v].tolist(),
                line={"color": vc["main"], "width": 2.5},
                name=_vlabel(v) if multi else "SpAM distance",
                legendgroup=f"v{v:g}",
            ), row=1, col=1)

        # Bottom panel: deviation from null
        for v in versions:
            if v not in version_kde:
                continue
            vc = _vc(v)
            dev = (version_kde[v] - null_kde).tolist()
            fig.add_trace(go.Scatter(
                x=x.tolist(), y=dev,
                fill="tozeroy",
                fillcolor=vc["violin"],
                line={"color": vc["main"], "width": 1.5},
                name=_vlabel(v),
                legendgroup=f"v{v:g}",
                showlegend=False,
            ), row=2, col=1)

        # Zero reference line in deviation panel
        fig.add_trace(go.Scatter(
            x=[0.0, 1.0], y=[0.0, 0.0],
            line={"color": "rgba(80,80,80,0.55)", "width": 1.2, "dash": "dot"},
            showlegend=False,
        ), row=2, col=1)

        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Δ density vs null", row=2, col=1)
        fig.update_xaxes(title_text="Normalised distance", row=2, col=1)
        fig.update_layout(
            title=title,
            height=560,
            margin={"l": 80, "r": 40, "t": 60, "b": 60},
            showlegend=True,
        )

    else:
        fig = go.Figure()
        for v in versions:
            if v not in version_kde:
                continue
            vc = _vc(v)
            fig.add_trace(go.Scatter(
                x=x.tolist(), y=version_kde[v].tolist(),
                line={"color": vc["main"], "width": 2.5},
                name=_vlabel(v) if multi else "SpAM distance",
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Normalised distance",
            yaxis_title="Density",
            height=400,
            margin={"l": 70, "r": 40, "t": 50, "b": 60},
            showlegend=True,
        )

    return fig
