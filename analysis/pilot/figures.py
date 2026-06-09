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
from scipy.stats import pearsonr

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
    rt_s = _rt_s(df_trials)
    stats = _subject_stats(df_trials.assign(rt_s=rt_s), "rt_s")
    stats = stats.sort_values("mean").reset_index(drop=True)
    y_vals = np.linspace(-0.15, 0.15, len(stats))  # slight jitter on y

    fig = go.Figure()
    fig.add_trace(go.Violin(
        x=rt_s,
        orientation="h",
        side="both",
        fillcolor="rgba(100,149,237,0.3)",
        line_color="rgba(100,149,237,0.8)",
        showlegend=False,
        name="All trials",
        y0=0,
        width=0.6,
        points=False,
    ))
    fig.add_trace(go.Scatter(
        x=stats["mean"],
        y=y_vals,
        mode="markers",
        marker={"color": "rgba(30,80,160,0.9)", "size": 8, "symbol": "circle"},
        error_x={"type": "data", "array": stats["std"].tolist(), "visible": True,
                 "color": "rgba(30,80,160,0.6)", "thickness": 1.5, "width": 4},
        showlegend=False,
        name="Subject mean ± SD",
    ))
    fig.update_layout(
        title="Trial duration per subject",
        xaxis_title="Trial duration (s)",
        yaxis={"visible": False, "range": [-0.5, 0.5]},
        height=350,
        margin={"l": 40, "r": 40, "t": 50, "b": 60},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 3 — Moves per subject (horizontal violin + dots ± SD)
# ---------------------------------------------------------------------------


def fig_moves_per_subject(df_trials: pd.DataFrame) -> go.Figure:
    stats = _subject_stats(df_trials, "n_moves")
    stats = stats.sort_values("mean").reset_index(drop=True)
    y_vals = np.linspace(-0.15, 0.15, len(stats))

    fig = go.Figure()
    fig.add_trace(go.Violin(
        x=df_trials["n_moves"],
        orientation="h",
        side="both",
        fillcolor="rgba(180,120,200,0.3)",
        line_color="rgba(140,80,180,0.8)",
        showlegend=False,
        name="All trials",
        y0=0,
        width=0.6,
        points=False,
    ))
    fig.add_trace(go.Scatter(
        x=stats["mean"],
        y=y_vals,
        mode="markers",
        marker={"color": "rgba(100,40,140,0.9)", "size": 8, "symbol": "circle"},
        error_x={"type": "data", "array": stats["std"].tolist(), "visible": True,
                 "color": "rgba(100,40,140,0.6)", "thickness": 1.5, "width": 4},
        showlegend=False,
        name="Subject mean ± SD",
    ))
    fig.update_layout(
        title="Number of moves per subject",
        xaxis_title="Moves per trial",
        yaxis={"visible": False, "range": [-0.5, 0.5]},
        height=350,
        margin={"l": 40, "r": 40, "t": 50, "b": 60},
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
    color_mean: str = "#1a5296",
    color_subj: str = "rgba(100,149,237,0.2)",
) -> go.Figure:
    """Line plot of *col* over trial_number (1–10) with mean±SE + individual traces."""
    df = df_trials.copy()
    df["_y"] = df[col]

    # Individual subject traces
    fig = go.Figure()
    for pid, grp in df.groupby("participant_id"):
        grp_sorted = grp.sort_values("trial_number")
        fig.add_trace(go.Scatter(
            x=grp_sorted["trial_number"],
            y=grp_sorted["_y"],
            mode="lines",
            line={"color": color_subj, "width": 1},
            showlegend=False,
            hoverinfo="skip",
        ))

    # Mean ± SE
    agg = (
        df.groupby("trial_number")["_y"]
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
        fillcolor=color_mean,
        opacity=0.2,
        line={"color": "rgba(0,0,0,0)"},
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=mean,
        mode="lines+markers",
        line={"color": color_mean, "width": 3},
        marker={"size": 7, "color": color_mean},
        name="Mean ± SE",
    ))
    fig.update_layout(
        title=title,
        xaxis={"title": "Trial number", "tickvals": list(range(1, 11))},
        yaxis_title=y_label,
        height=400,
        margin={"l": 60, "r": 40, "t": 50, "b": 60},
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
        color_mean="#7b2d8b",
        color_subj="rgba(180,120,200,0.2)",
    )


# ---------------------------------------------------------------------------
# Fig 6 — Duration vs moves scatter (per subject)
# ---------------------------------------------------------------------------


def fig_duration_vs_moves(df_trials: pd.DataFrame) -> go.Figure:
    df = df_trials.assign(rt_s=_rt_s(df_trials))
    stats = (
        df.groupby("participant_id")
        .agg(
            mean_rt=("rt_s", "mean"),
            sd_rt=("rt_s", "std"),
            mean_moves=("n_moves", "mean"),
            sd_moves=("n_moves", "std"),
        )
        .reset_index()
    )

    fig = go.Figure(go.Scatter(
        x=stats["mean_moves"],
        y=stats["mean_rt"],
        mode="markers",
        marker={"size": 9, "color": "#1a5296", "opacity": 0.85},
        error_x={"type": "data", "array": stats["sd_moves"].tolist(),
                 "visible": True, "color": "rgba(26,82,150,0.4)", "thickness": 1.5},
        error_y={"type": "data", "array": stats["sd_rt"].tolist(),
                 "visible": True, "color": "rgba(26,82,150,0.4)", "thickness": 1.5},
        text=stats["participant_id"],
        hovertemplate="<b>%{text}</b><br>Moves: %{x:.1f}<br>Duration: %{y:.1f}s<extra></extra>",
    ))
    fig.update_layout(
        title="Trial duration vs. number of moves (per subject, mean ± SD)",
        xaxis_title="Moves per trial",
        yaxis_title="Trial duration (s)",
        height=450,
        margin={"l": 70, "r": 40, "t": 50, "b": 60},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 7 — Within-subject variability + reliability
# ---------------------------------------------------------------------------


def fig_within_subject_variability(df_trials: pd.DataFrame) -> go.Figure:
    subjects = sorted(df_trials["participant_id"].unique())

    all_abs_diffs: list[float] = []
    all_subj_labels: list[str] = []
    pearson_rs: list[float] = []
    subj_short = {pid: f"S{i+1:02d}" for i, pid in enumerate(subjects)}

    for pid in subjects:
        df_s = df_trials[df_trials["participant_id"] == pid]
        d1, d2 = _repeated_pair_distances(df_s)
        if len(d1) == 0:
            continue
        abs_diffs = np.abs(np.array(d1) - np.array(d2)).tolist()
        all_abs_diffs.extend(abs_diffs)
        all_subj_labels.extend([subj_short[pid]] * len(abs_diffs))
        r, _ = pearsonr(d1, d2)
        pearson_rs.append(r)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=["Within-subject |Δd| per pair", "Within-subject reliability (Pearson r)"],
    )

    # Subplot A: violin (pooled) + scatter per subject
    fig.add_trace(
        go.Violin(
            x=all_abs_diffs,
            y=all_subj_labels,
            orientation="h",
            side="positive",
            fillcolor="rgba(100,149,237,0.25)",
            line_color="rgba(100,149,237,0.7)",
            showlegend=False,
            points=False,
            width=0.8,
            name="|Δd|",
        ),
        row=1, col=1,
    )
    # Per-subject mean ± SD dots
    for i, pid in enumerate(subjects):
        mask = [l == subj_short[pid] for l in all_subj_labels]
        vals = [v for v, m in zip(all_abs_diffs, mask) if m]
        if not vals:
            continue
        fig.add_trace(
            go.Scatter(
                x=[float(np.mean(vals))],
                y=[subj_short[pid]],
                mode="markers",
                marker={"color": "#1a5296", "size": 7},
                error_x={"type": "data", "array": [float(np.std(vals))],
                         "visible": True, "color": "rgba(26,82,150,0.5)", "thickness": 1.5},
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Subplot B: box + strip of per-subject Pearson r
    fig.add_trace(
        go.Box(
            y=pearson_rs,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            marker={"color": "#1a5296", "size": 8},
            line_color="#1a5296",
            fillcolor="rgba(100,149,237,0.25)",
            showlegend=False,
            name="Pearson r",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="|Δd|", row=1, col=1)
    fig.update_yaxes(title_text="Subject", row=1, col=1)
    fig.update_yaxes(title_text="Pearson r", row=1, col=2, range=[-0.1, 1.05])
    fig.update_layout(
        title="Within-subject variability and reliability of repeated pairs",
        height=500,
        margin={"l": 70, "r": 40, "t": 70, "b": 60},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 8 — Participant demographics
# ---------------------------------------------------------------------------


def fig_demographics(df_trials: pd.DataFrame) -> go.Figure:
    # One row per subject (completed only)
    df_subj = df_trials.drop_duplicates("participant_id")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Age distribution", "Sex", "Ethnicity", "Country of residence"],
    )

    # Age histogram
    ages = df_subj["age"].dropna()
    fig.add_trace(
        go.Histogram(x=ages, nbinsx=10, marker_color="#2ecc71", showlegend=False, name="Age"),
        row=1, col=1,
    )

    # Sex
    sex_counts = df_subj["sex"].dropna().value_counts()
    fig.add_trace(
        go.Bar(x=sex_counts.values.tolist(), y=sex_counts.index.tolist(),
               orientation="h", marker_color="#3498db", showlegend=False, name="Sex"),
        row=1, col=2,
    )

    # Ethnicity
    eth_counts = df_subj["ethnicity"].dropna().value_counts()
    fig.add_trace(
        go.Bar(x=eth_counts.values.tolist(), y=eth_counts.index.tolist(),
               orientation="h", marker_color="#9b59b6", showlegend=False, name="Ethnicity"),
        row=2, col=1,
    )

    # Country of residence
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
# Fig C — Pairwise distance distribution
# ---------------------------------------------------------------------------


def fig_pairwise_distance_distribution(df_trials: pd.DataFrame) -> go.Figure:
    all_distances: list[float] = []
    for pw_json in df_trials["pairwise_distances"]:
        for dist in _parse_pairwise(pw_json).values():
            all_distances.append(dist)

    fig = go.Figure(go.Histogram(
        x=all_distances,
        nbinsx=50,
        histnorm="probability density",
        marker_color="rgba(100,149,237,0.6)",
        marker_line={"color": "rgba(100,149,237,1.0)", "width": 0.5},
        name="SpAM distance",
    ))
    fig.update_layout(
        title=f"Distribution of pairwise SpAM distances (n={len(all_distances):,} pair observations)",
        xaxis_title="Normalised distance",
        yaxis_title="Density",
        height=400,
        margin={"l": 70, "r": 40, "t": 50, "b": 60},
    )
    return fig
