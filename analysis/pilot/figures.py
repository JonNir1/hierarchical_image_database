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
from scipy.stats import gaussian_kde, spearmanr

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
    3.0: {
        "main":         "#1e7a45",
        "violin":       "rgba(46,160,90,0.35)",
        "violin_line":  "rgba(46,160,90,0.9)",
        "subj":         "rgba(46,160,90,0.2)",
        "dot":          "rgba(20,110,60,0.9)",
        "dot_err":      "rgba(20,110,60,0.6)",
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
    v1/v2 reliability measure: find individual image pairs that incidentally recur
    across ≥ 2 distinct trials (a side-effect of the old unique_images_per_subject
    design, where some images were shown in 2 of a subject's trials).

    Returns (d_first, d_second) — lists of paired distance values for test-retest
    analysis. Pairs observed > 2 times: take first two observations.
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


def _repeated_trial_distances(df_subject: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    v3+ reliability measure: match each verbatim trial repeat (is_trial_repeat=True)
    to its original trial via repeat_of_trial_number, then pair up their
    per-image-pair distances (image identity, not position, since presentation
    order is reshuffled in the repeat).

    Returns (d_original, d_repeat) — lists of paired distance values for test-retest
    analysis. Subjects/versions with no trial repeats yield two empty lists.
    """
    if "is_trial_repeat" not in df_subject.columns or "repeat_of_trial_number" not in df_subject.columns:
        return [], []

    pw_by_trial_number = dict(zip(df_subject["trial_number"], df_subject["pairwise_distances"]))

    d_orig: list[float] = []
    d_repeat: list[float] = []
    repeats = df_subject[df_subject["is_trial_repeat"] & df_subject["repeat_of_trial_number"].notna()]
    for _, row in repeats.iterrows():
        orig_pw_json = pw_by_trial_number.get(int(row["repeat_of_trial_number"]))
        if orig_pw_json is None:
            continue
        orig_dists = _parse_pairwise(orig_pw_json)
        repeat_dists = _parse_pairwise(row["pairwise_distances"])
        for pair, dist in repeat_dists.items():
            if pair in orig_dists:
                d_orig.append(orig_dists[pair])
                d_repeat.append(dist)

    return d_orig, d_repeat


def _reliability_pair_distances(df_subject: pd.DataFrame) -> tuple[list[float], list[float], str]:
    """
    Dispatch to whichever reliability measure has data for this subject:
    trial repeats (v3+) if present, else incidentally-repeated image pairs (v1/v2).
    Returns (d1, d2, measure_label).
    """
    d1, d2 = _repeated_trial_distances(df_subject)
    if len(d1) > 0:
        return d1, d2, "trial repeat"
    d1, d2 = _repeated_pair_distances(df_subject)
    return d1, d2, "image repeat"


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

    # 60 s timer reference line
    fig.add_vline(
        x=60,
        line={"color": "rgba(120,120,120,0.55)", "width": 1.5, "dash": "dash"},
    )

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
    Subplot B — Within-subject reliability: Spearman r between d1 and d2 of repeated
    image pairs (v1/v2) or repeated trials (v3+).
    Both subplots colour subjects by task_version.

    v1/v2 lack the whole-trial-repeat mechanism, so their reliability is measured
    via individual image pairs that incidentally recur across distinct trials
    (the old unique_images_per_subject design). v3+ repeats whole trials verbatim
    instead. Both yield a (d1, d2) pair of matched distances per subject, so SNR
    and Spearman r are computed identically once that pair is obtained — but the
    two measures are not strictly comparable in absolute magnitude (see subtitle).
    """
    subjects = sorted(df_trials["participant_id"].unique())
    subj_short = {pid: f"S{i+1:02d}" for i, pid in enumerate(subjects)}

    records: list[dict] = []
    for pid in subjects:
        df_s = df_trials[df_trials["participant_id"] == pid]
        d1, d2, measure = _reliability_pair_distances(df_s)
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

        r, _ = spearmanr(d1, d2)
        v = float(df_s["task_version"].iloc[0])
        records.append({
            "label":         subj_short[pid],
            "pid":           pid,
            "version":       v,
            "measure":       measure,
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
            "Within-subject reliability (Spearman r)",
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
                hovertemplate=f"<b>{rec['pid']}</b><br>SNR = %{{x:.3f}}<br>measure: {rec['measure']}<extra></extra>",
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
        measures = [rec["measure"] for rec in vrecs]
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
                text=[f"{s} ({p}, {m})" for s, p, m in zip(labels, pids, measures)],
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
    fig.update_yaxes(title_text="Spearman r", row=1, col=2)
    fig.update_layout(
        title={
            "text": (
                "Within-subject variability and reliability"
                "<br><sup>v1/v2 use incidentally-repeated image pairs; v3+ uses verbatim trial repeats - "
                "the two measures are not directly comparable in absolute magnitude.</sup>"
            ),
        },
        height=540,
        margin={"l": 70, "r": 40, "t": 90, "b": 60},
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
# Fig B — Temporal engagement profile (cumulative moves + move rate)
# ---------------------------------------------------------------------------


def fig_move_temporal_profile(df_trials: pd.DataFrame) -> go.Figure:
    """
    2x2 grid figure of temporal move patterns per cohort.

    Top row (colspan 2): time of last move per subject — violin + dots ± SD,
                         styled like fig_trial_duration_per_subject.
    Bottom left:  average cumulative-move curve (time fraction vs. move fraction)
                  with ±1 SE ribbon. Diagonal = uniform activity.
    Bottom right: average move rate (moves/s, 5 s bins) over absolute time
                  with ±1 SE ribbon and a 60 s reference line.
    """
    versions = _sorted_versions(df_trials)
    n_versions = len(versions)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=[
            "Time of last move per subject",
            "Cumulative moves within trial",
            "Move rate over absolute time",
        ],
        row_heights=[0.45, 0.55],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    x_grid = np.linspace(0.0, 1.0, 300)
    bin_s = 5
    bin_ms = bin_s * 1000
    max_ms = 120_000
    bins = np.arange(0, max_ms + bin_ms, bin_ms)
    bin_centers_s = (bins[:-1] + bins[1:]) / 2 / 1000

    for i, v in enumerate(versions):
        vc = _vc(v)
        vdf = df_trials[df_trials["task_version"] == v]

        t_last_records: list[dict] = []
        cum_curves: list[np.ndarray] = []
        trial_rates: list[np.ndarray] = []

        for _, row in vdf.iterrows():
            try:
                moves = json.loads(row["moves"])
            except (json.JSONDecodeError, TypeError):
                continue
            ts = [m["t"] for m in moves if isinstance(m.get("t"), (int, float))]
            rt = row["rt"]
            if not ts or pd.isna(rt) or rt <= 0:
                continue

            t_last_records.append({
                "participant_id": row["participant_id"],
                "t_last_s": max(ts) / 1000,
            })

            # Cumulative curve
            ts_sorted = np.minimum(np.sort(ts), rt)
            ts_norm = np.concatenate([[0.0], ts_sorted / rt, [1.0]])
            n = len(ts)
            cum_y = np.concatenate([[0.0], np.arange(1, n + 1) / n, [1.0]])
            cum_curves.append(np.interp(x_grid, ts_norm, cum_y))

            # Rate histogram
            ts_clipped = [t for t in ts if t <= max_ms]
            counts, _ = np.histogram(ts_clipped, bins=bins)
            trial_rates.append(counts / bin_s)

        # --- Top panel: last-move-time violin + subject dots ---
        if t_last_records:
            df_last = pd.DataFrame(t_last_records)
            stats = (
                df_last.groupby("participant_id")["t_last_s"]
                .agg(mean="mean", std="std")
                .reset_index()
                .sort_values("mean")
                .reset_index(drop=True)
            )
            stats["std"] = stats["std"].fillna(0)
            y_dots = i + np.linspace(-0.15, 0.15, len(stats))

            fig.add_trace(go.Violin(
                x=df_last["t_last_s"].tolist(),
                y0=i,
                orientation="h",
                side="both",
                fillcolor=vc["violin"],
                line_color=vc["violin_line"],
                name=_vlabel(v),
                showlegend=True,
                width=0.6,
                points=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=stats["mean"].tolist(),
                y=y_dots.tolist(),
                mode="markers",
                marker={"color": vc["dot"], "size": 8, "symbol": "circle"},
                error_x={
                    "type": "data", "array": stats["std"].tolist(),
                    "visible": True, "color": vc["dot_err"],
                    "thickness": 1.5, "width": 4,
                },
                showlegend=False,
                name=f"{_vlabel(v)} subject mean ± SD",
            ), row=1, col=1)

        # --- Bottom left: cumulative move curve ---
        if cum_curves:
            mat = np.array(cum_curves)
            mean_c = mat.mean(axis=0)
            se_c = mat.std(axis=0) / np.sqrt(len(mat))

            fig.add_trace(go.Scatter(
                x=x_grid.tolist() + x_grid.tolist()[::-1],
                y=(mean_c + se_c).tolist() + (mean_c - se_c).tolist()[::-1],
                fill="toself", fillcolor=vc["violin"],
                line={"width": 0}, showlegend=False,
                legendgroup=f"v{v:g}", hoverinfo="skip",
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=x_grid.tolist(), y=mean_c.tolist(),
                line={"color": vc["main"], "width": 2.5},
                name=_vlabel(v), legendgroup=f"v{v:g}", showlegend=False,
            ), row=2, col=1)

        # --- Bottom right: move rate over absolute time ---
        if trial_rates:
            rmat = np.array(trial_rates)
            mean_r = rmat.mean(axis=0)
            se_r = rmat.std(axis=0) / np.sqrt(len(rmat))

            fig.add_trace(go.Scatter(
                x=bin_centers_s.tolist() + bin_centers_s.tolist()[::-1],
                y=(mean_r + se_r).tolist() + (mean_r - se_r).tolist()[::-1],
                fill="toself", fillcolor=vc["violin"],
                line={"width": 0}, showlegend=False,
                legendgroup=f"v{v:g}", hoverinfo="skip",
            ), row=2, col=2)
            fig.add_trace(go.Scatter(
                x=bin_centers_s.tolist(), y=mean_r.tolist(),
                line={"color": vc["main"], "width": 2.5},
                name=_vlabel(v), legendgroup=f"v{v:g}", showlegend=False,
            ), row=2, col=2)

    # Diagonal reference line (uniform activity) in cumulative panel
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0], y=[0.0, 1.0], mode="lines",
        line={"color": "rgba(80,80,80,0.40)", "width": 1.2, "dash": "dot"},
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    # 60 s reference lines: last-move panel (axis 1) and rate panel (axis 3).
    # Use yref in domain coordinates so the shape spans the subplot height
    # without polluting the y-axis autorange.
    _ref_line = {"color": "rgba(80,80,80,0.55)", "width": 1.2, "dash": "dash"}
    fig.add_shape(type="line", x0=60, x1=60, y0=0, y1=1,
                  xref="x", yref="y domain", line=_ref_line)
    fig.add_shape(type="line", x0=60, x1=60, y0=0, y1=1,
                  xref="x3", yref="y3 domain", line=_ref_line)

    # y-axis for top panel
    y_range = [-0.55, n_versions - 0.45]
    if n_versions > 1:
        fig.update_yaxes(
            range=y_range,
            tickvals=list(range(n_versions)),
            ticktext=[_vlabel(v) for v in versions],
            row=1, col=1,
        )
    else:
        fig.update_yaxes(range=y_range, visible=False, row=1, col=1)

    fig.update_xaxes(title_text="Time of last move (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time fraction (t / RT)", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Cumulative move fraction", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Time from trial start (s)", range=[0, max_ms / 1000], row=2, col=2)
    fig.update_yaxes(title_text="Moves / s (avg. over trials)", row=2, col=2)
    fig.update_layout(
        title="Temporal engagement profile during trials",
        height=620,
        margin={"l": 70, "r": 40, "t": 80, "b": 60},
        showlegend=True,
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
