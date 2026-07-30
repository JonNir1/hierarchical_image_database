"""
Pilot analysis figures for the SpAM task.

All functions accept DataFrames returned by analysis.utils.parser.load_data
("participants" and/or "trials") and return plotly Figure objects.

Usage (from repo root):
    from analysis.utils.parser import load_data
    from analysis.pilot.figures import (
        fig_completion_status, fig_demographics,
        fig_trial_duration, fig_moves, fig_reliability,
        fig_duration_vs_moves, fig_within_subject_variability,
        fig_pairwise_distance_distribution, fig_reliability_vs_distance,
        fig_move_temporal_profile,
    )
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, ks_2samp, spearmanr

from analysis.utils.parser import parse_pairwise_distances

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


def _major_version(version) -> float:
    """Floor a task_version to its major version for display grouping -- e.g.
    3.0 and 3.06 both group as 3.0 (shown as "v3"), 4.0 stays 4.0 ("v4")."""
    return float(int(version))


def _with_major_version(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with task_version replaced by its major-version
    group (see _major_version), so figures show one trace/colour per major
    version (e.g. v3.0 and v3.06 merge into a single "v3") instead of one per
    exact sub-version. Does not mutate the input; unrelated to the raw
    task_version values used elsewhere (QC tables, cohort-comparison tests).
    Rows with a non-finite task_version (e.g. revoked-consent/missing-data
    participants with no resolvable session file) are dropped -- they can't
    be attributed to any version."""
    df = df[df["task_version"].notna()]
    return df.assign(task_version=df["task_version"].apply(_major_version))


def _sorted_versions(df: pd.DataFrame) -> list[float]:
    return sorted(df["task_version"].dropna().unique().tolist())


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
        for pair, dist in parse_pairwise_distances(pw_json).items():
            pair_obs[pair].append(dist)

    d1, d2 = [], []
    for obs in pair_obs.values():
        if len(obs) >= 2:
            d1.append(obs[0])
            d2.append(obs[1])
    return d1, d2


def _repeated_trial_distances(df_subject: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    v3+ reliability measure: match each verbatim trial repeat (repeat_of_trial notna)
    to its original trial via trial_id, then pair up their per-image-pair distances
    (image identity, not position, since presentation order is reshuffled in the
    repeat).

    Returns (d_original, d_repeat) — lists of paired distance values for test-retest
    analysis. Subjects/versions with no trial repeats yield two empty lists.
    """
    if "repeat_of_trial" not in df_subject.columns:
        return [], []

    pw_by_trial_id = dict(zip(df_subject["trial_id"], df_subject["pairwise_distances"]))

    d_orig: list[float] = []
    d_repeat: list[float] = []
    repeats = df_subject[df_subject["repeat_of_trial"].notna()]
    for _, row in repeats.iterrows():
        orig_pw_json = pw_by_trial_id.get(int(row["repeat_of_trial"]))
        if orig_pw_json is None:
            continue
        orig_dists = parse_pairwise_distances(orig_pw_json)
        repeat_dists = parse_pairwise_distances(row["pairwise_distances"])
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


# ---------------------------------------------------------------------------
# Fig 1 — Completion status distribution, faceted by major version
# ---------------------------------------------------------------------------

_STATUS_LABEL_MAP = {
    "full data": "Full data",
    "screened out": "Screened out",
    "revoked consent": "Revoked consent",
    "missing data": "Missing data",
}
_STATUS_COLOR_MAP = {
    "full data": "#2ecc71",
    "screened out": "#f39c12",
    "revoked consent": "#e74c3c",
    "missing data": "#95a5a6",
}


def fig_completion_status(df_participants: pd.DataFrame) -> go.Figure:
    """
    Participant status distribution, one horizontal-bar subplot per major version
    (v3.0/v3.06 merge into a single "v3" subplot), each subplot title annotated
    with that version's total N. Participants with no resolvable session file (so
    no derivable task_version -- revoked consent before ever starting, or a
    totally missing file, e.g. a Prolific/Pavlovia handoff that never reached us)
    can't be attributed to a version; rather than silently dropping them, they get
    one extra "Unknown version" subplot so they stay visible in this figure.
    """
    df = _with_major_version(df_participants)
    versions = _sorted_versions(df)
    unknown_df = df_participants[df_participants["task_version"].isna()]
    has_unknown = not unknown_df.empty
    n_panels = len(versions) + (1 if has_unknown else 0)
    categories = list(_STATUS_LABEL_MAP)

    subplot_titles = [f"{_vlabel(v)} (N={int((df['task_version'] == v).sum())})" for v in versions]
    if has_unknown:
        subplot_titles.append(f"Unknown version (N={len(unknown_df)})")

    fig = make_subplots(
        rows=1, cols=n_panels,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
    )

    def _add_status_bar(vdf: pd.DataFrame, col_idx: int) -> None:
        counts = vdf["status"].value_counts()
        total = counts.sum()
        pcts = (counts / total * 100).round(1)
        cats_present = [c for c in categories if c in counts.index]
        fig.add_trace(go.Bar(
            x=[pcts.get(c, 0) for c in cats_present],
            y=[_STATUS_LABEL_MAP[c] for c in cats_present],
            orientation="h",
            width=0.55,
            text=[f"{pcts.get(c, 0):.1f}% (n={counts.get(c, 0)})" for c in cats_present],
            textposition="auto",
            textfont={"size": 10},
            insidetextfont={"color": "white", "size": 10},
            cliponaxis=False,
            marker_color=[_STATUS_COLOR_MAP[c] for c in cats_present],
            showlegend=False,
        ), row=1, col=col_idx)
        fig.update_xaxes(
            range=[0, 100], tickvals=[0, 25, 50, 75, 100],
            title_text="% of participants", row=1, col=col_idx,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=col_idx)

    for col_idx, v in enumerate(versions, start=1):
        _add_status_bar(df[df["task_version"] == v], col_idx)
    if has_unknown:
        _add_status_bar(unknown_df, n_panels)

    fig.update_layout(
        title="Participant status by version",
        height=280,
        width=max(700, 260 * n_panels),
        margin={"l": 140, "r": 60, "t": 70, "b": 50},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 2 — Trial duration per subject + progression (violins top, progression bottom)
# ---------------------------------------------------------------------------


def _add_violin_dots_panel(fig, df_trials, col, versions, row, col_idx=1, xaxis_title=""):
    """Shared body for the 'violin + per-subject mean±SD dots, one row per version'
    panel used by fig_trial_duration/fig_moves' top row."""
    for i, v in enumerate(versions):
        vc = _vc(v)
        vdf = df_trials[df_trials["task_version"] == v]
        stats = _subject_stats(vdf, col)
        stats = stats.sort_values("mean").reset_index(drop=True)
        y_dots = i + np.linspace(-0.15, 0.15, len(stats))

        fig.add_trace(go.Violin(
            x=vdf[col],
            y0=i,
            orientation="h",
            side="both",
            fillcolor=vc["violin"],
            line_color=vc["violin_line"],
            name=_vlabel(v),
            legendgroup=_vlabel(v),
            showlegend=True,
            width=0.6,
            points=False,
        ), row=row, col=col_idx)
        fig.add_trace(go.Scatter(
            x=stats["mean"],
            y=y_dots.tolist(),
            mode="markers",
            marker={"color": vc["dot"], "size": 8, "symbol": "circle"},
            error_x={"type": "data", "array": stats["std"].tolist(), "visible": True,
                     "color": vc["dot_err"], "thickness": 1.5, "width": 4},
            showlegend=False,
            legendgroup=_vlabel(v),
            name=f"{_vlabel(v)} subject mean ± SD",
        ), row=row, col=col_idx)

    y_range = [-0.55, len(versions) - 0.45]
    fig.update_yaxes(
        range=y_range,
        tickvals=list(range(len(versions))),
        ticktext=[_vlabel(v) for v in versions],
        row=row, col=col_idx,
    )
    fig.update_xaxes(title_text=xaxis_title, row=row, col=col_idx)


def _add_progression_panel(fig, df_trials, col, versions, row, col_idx=1, yaxis_title=""):
    """Shared body for the 'mean±SE line over trial_id + thin per-subject lines'
    panel used by fig_trial_duration/fig_moves' bottom row."""
    for v in versions:
        vc = _vc(v)
        vdf = df_trials[df_trials["task_version"] == v]

        for _pid, grp in vdf.groupby("participant_id"):
            grp_sorted = grp.sort_values("trial_id")
            fig.add_trace(go.Scatter(
                x=grp_sorted["trial_id"],
                y=grp_sorted[col],
                mode="lines",
                line={"color": vc["subj"], "width": 1},
                showlegend=False,
                hoverinfo="skip",
            ), row=row, col=col_idx)

        agg = (
            vdf.groupby("trial_id")[col]
            .agg(mean="mean", std="std", n="count")
            .assign(se=lambda x: x["std"] / np.sqrt(x["n"]))
            .reset_index()
        )
        x = agg["trial_id"].tolist()
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
        ), row=row, col=col_idx)
        fig.add_trace(go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            line={"color": vc["main"], "width": 3},
            marker={"size": 7, "color": vc["main"]},
            name=f"{_vlabel(v)} mean ± SE",
            legendgroup=_vlabel(v),
            showlegend=False,
        ), row=row, col=col_idx)

    fig.update_xaxes(title_text="Trial ID", row=row, col=col_idx)
    fig.update_yaxes(title_text=yaxis_title, row=row, col=col_idx)


def fig_trial_duration(df_trials: pd.DataFrame) -> go.Figure:
    """
    Trial duration, merged 2-row figure: top row = per-subject horizontal violin
    + mean±SD dots (one row per major version); bottom row = duration over trial_id
    progression (mean±SE band + thin per-subject lines). Replaces the old separate
    fig_trial_duration_per_subject / fig_duration_progression.
    """
    df = _with_major_version(df_trials).assign(rt_s=lambda d: _rt_s(d))
    versions = _sorted_versions(df)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.35 + 0.1 * len(versions), 0.65 - 0.1 * len(versions)],
        subplot_titles=["Trial duration per subject", "Trial duration over task progression"],
        vertical_spacing=0.2,
    )
    _add_violin_dots_panel(fig, df, "rt_s", versions, row=1, xaxis_title="Trial duration (s)")
    _add_progression_panel(fig, df, "rt_s", versions, row=2, yaxis_title="Trial duration (s)")

    fig.add_vline(x=60, line={"color": "rgba(120,120,120,0.55)", "width": 1.5, "dash": "dash"}, row=1, col=1)

    fig.update_layout(
        title="Trial duration",
        height=620,
        margin={"l": 60, "r": 40, "t": 70, "b": 60},
        showlegend=True,
        legend={"groupclick": "togglegroup"},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 3 — Moves per subject + progression (violins top, progression bottom)
# ---------------------------------------------------------------------------


def fig_moves(df_trials: pd.DataFrame) -> go.Figure:
    """
    Number of moves, merged 2-row figure: top row = per-subject horizontal violin
    + mean±SD dots (one row per major version); bottom row = moves over trial_id
    progression (mean±SE band + thin per-subject lines). Replaces the old separate
    fig_moves_per_subject / fig_moves_progression.
    """
    df = _with_major_version(df_trials)
    versions = _sorted_versions(df)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.35 + 0.1 * len(versions), 0.65 - 0.1 * len(versions)],
        subplot_titles=["Number of moves per subject", "Number of moves over task progression"],
        vertical_spacing=0.2,
    )
    _add_violin_dots_panel(fig, df, "num_moves", versions, row=1, xaxis_title="Moves per trial")
    _add_progression_panel(fig, df, "num_moves", versions, row=2, yaxis_title="Moves per trial")

    fig.update_layout(
        title="Number of moves",
        height=620,
        margin={"l": 60, "r": 40, "t": 70, "b": 60},
        showlegend=True,
        legend={"groupclick": "togglegroup"},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 4 — Reliability: per-version-group violin (top) + progression (bottom)
# ---------------------------------------------------------------------------

# Aggregated by major version group (v3.* pools 3.0/3.06, v4.* pools 4.0/...),
# not exact version -- the point of this figure is comparing the two repeat
# designs (3 repeats/session for v3.x, 4 for v4.x), not individual sub-versions.
_GROUP_COLORS: dict[str, dict[str, str]] = {
    "v3.*": {"main": "#1e7a45", "subj": "rgba(46,160,90,0.25)", "violin": "rgba(46,160,90,0.30)"},
    "v4.*": {"main": "#8e44ad", "subj": "rgba(142,68,173,0.25)", "violin": "rgba(142,68,173,0.30)"},
}
_GROUP_COLOR_FALLBACK: dict[str, str] = {
    "main": "#555555", "subj": "rgba(85,85,85,0.25)", "violin": "rgba(85,85,85,0.30)",
}


def _repeat_reliability_by_index(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Per-subject, per-repeat-ordinal Spearman r (v3+ trial-repeat mechanism only;
    v1/v2 have no verbatim trial repeats). repeat_index is 1-based, ordered by
    ascending trial_id within each subject's session -- repeat #1 is the earliest
    repeat trial encountered, etc. v3.x sessions have 3 repeats; v4.x sessions have
    4 (2 in the screening stage, 2 in the experimental stage). Reliability values
    are read directly from the trials' precomputed `reliability` column (the parser
    already computes Spearman r per repeat row) rather than recomputed here.

    Returns one row per (participant_id, repeat_index): participant_id,
    task_version, version_group ("v3.*"/"v4.*"/...), repeat_index, spearman_r.
    """
    df_v3plus = df_trials[df_trials["task_version"] >= 3.0]
    rows = []
    for pid, df_s in df_v3plus.groupby("participant_id"):
        version = float(df_s["task_version"].iloc[0])
        repeats = (
            df_s[df_s["repeat_of_trial"].notna() & df_s["reliability"].notna()]
            .sort_values("trial_id")
        )
        for i, (_, row) in enumerate(repeats.iterrows(), start=1):
            rows.append({
                "participant_id": pid,
                "task_version": version,
                "version_group": f"v{int(version)}.*",
                "repeat_index": i,
                "spearman_r": float(row["reliability"]),
            })
    return pd.DataFrame(
        rows,
        columns=["participant_id", "task_version", "version_group", "repeat_index", "spearman_r"],
    )


def fig_reliability(df_trials: pd.DataFrame) -> go.Figure:
    """
    Test-retest reliability (Spearman r), v3+ only, merged 2-row figure: top row =
    per-version-group (v3.*/v4.*) horizontal violin + per-subject mean±SE dots of
    all repeat-trial r values; bottom row = r vs. repeat # progression (mean±SE
    band + thin per-subject lines). Replaces the old standalone
    fig_reliability_progression.
    """
    r_df = _repeat_reliability_by_index(df_trials)
    if r_df.empty:
        raise ValueError("fig_reliability: no v3+ trial repeats found in df_trials")

    groups = sorted(r_df["version_group"].unique())
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=["Test-retest reliability by version", "Reliability over repeat trials"],
        vertical_spacing=0.2,
    )

    # --- Top row: violin + per-subject mean±SE dots ---
    for i, group in enumerate(groups):
        gc = _GROUP_COLORS.get(group, _GROUP_COLOR_FALLBACK)
        gdf = r_df[r_df["version_group"] == group]
        stats = _subject_se(gdf, "spearman_r")
        stats = stats.sort_values("mean").reset_index(drop=True)
        y_dots = i + np.linspace(-0.15, 0.15, len(stats))

        fig.add_trace(go.Violin(
            x=gdf["spearman_r"],
            y0=i,
            orientation="h",
            side="both",
            fillcolor=gc["violin"],
            line_color=gc["main"],
            name=group,
            legendgroup=group,
            showlegend=True,
            width=0.6,
            points=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=stats["mean"],
            y=y_dots.tolist(),
            mode="markers",
            marker={"color": gc["main"], "size": 8, "symbol": "circle"},
            error_x={"type": "data", "array": stats["se"].tolist(), "visible": True,
                     "color": gc["main"], "thickness": 1.5, "width": 4},
            showlegend=False,
            legendgroup=group,
            name=f"{group} subject mean ± SE",
        ), row=1, col=1)

    fig.update_yaxes(
        range=[-0.55, len(groups) - 0.45],
        tickvals=list(range(len(groups))),
        ticktext=groups,
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Spearman r (repeat vs. original)", row=1, col=1)

    # --- Bottom row: progression ---
    for group in groups:
        gc = _GROUP_COLORS.get(group, _GROUP_COLOR_FALLBACK)
        gdf = r_df[r_df["version_group"] == group]

        for _pid, grp in gdf.groupby("participant_id"):
            grp_sorted = grp.sort_values("repeat_index")
            fig.add_trace(go.Scatter(
                x=grp_sorted["repeat_index"],
                y=grp_sorted["spearman_r"],
                mode="lines",
                line={"color": gc["subj"], "width": 1},
                showlegend=False,
                hoverinfo="skip",
            ), row=2, col=1)

        agg = (
            gdf.groupby("repeat_index")["spearman_r"]
            .agg(mean="mean", std="std", n="count")
            .assign(se=lambda x: x["std"] / np.sqrt(x["n"]))
            .reset_index()
        )
        x = agg["repeat_index"].tolist()
        mean = agg["mean"].tolist()
        se = agg["se"].tolist()

        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[m + s for m, s in zip(mean, se)] + [m - s for m, s in zip(mean[::-1], se[::-1])],
            fill="toself",
            fillcolor=gc["main"],
            opacity=0.2,
            line={"color": "rgba(0,0,0,0)"},
            showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            line={"color": gc["main"], "width": 3},
            marker={"size": 8, "color": gc["main"]},
            name=f"{group} mean ± SE",
            legendgroup=group,
            showlegend=False,
        ), row=2, col=1)

    max_repeat = int(r_df["repeat_index"].max())
    fig.update_xaxes(title_text="Repeat #", tickvals=list(range(1, max_repeat + 1)), row=2, col=1)
    fig.update_yaxes(title_text="Spearman r (repeat vs. original)", row=2, col=1)

    fig.update_layout(
        title="Test-retest reliability (Spearman r)",
        height=620,
        margin={"l": 60, "r": 40, "t": 70, "b": 60},
        showlegend=True,
        legend={"groupclick": "togglegroup"},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 5 — Duration vs moves scatter (per subject, coloured by version)
# ---------------------------------------------------------------------------


def fig_duration_vs_moves(df_trials: pd.DataFrame) -> go.Figure:
    df = _with_major_version(df_trials).assign(rt_s=_rt_s(df_trials))
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
                mean_moves=("num_moves", "mean"),
                sd_moves=("num_moves", "std"),
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
# Fig 6 — Within-subject variability + reliability (coloured by version)
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
    df_trials = _with_major_version(df_trials)
    version_by_subject = df_trials.groupby("participant_id")["task_version"].first()
    subjects = sorted(df_trials["participant_id"].unique(), key=lambda pid: (version_by_subject[pid], pid))
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
            for dist in parse_pairwise_distances(pw_json).values()
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
# Fig 7 — Participant demographics
# ---------------------------------------------------------------------------


def fig_demographics(df_participants: pd.DataFrame) -> go.Figure:
    """2x2 grid: age histogram, sex/ethnicity/country bar charts. Scoped to
    participants with real trial content (status full data / screened out) --
    revoked-consent / missing-data participants are excluded, matching the old
    figure's "completed submissions" scope."""
    df = df_participants[df_participants["status"].isin(["full data", "screened out"])]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Age distribution", "Sex", "Ethnicity", "Country of residence"],
    )

    ages = df["age"].dropna()
    fig.add_trace(
        go.Histogram(x=ages, nbinsx=10, marker_color="#2ecc71", showlegend=False, name="Age"),
        row=1, col=1,
    )

    sex_counts = df["sex"].dropna().value_counts()
    fig.add_trace(
        go.Bar(x=sex_counts.values.tolist(), y=sex_counts.index.tolist(),
               orientation="h", marker_color="#3498db", showlegend=False, name="Sex"),
        row=1, col=2,
    )

    eth_counts = df["ethnicity"].dropna().value_counts()
    fig.add_trace(
        go.Bar(x=eth_counts.values.tolist(), y=eth_counts.index.tolist(),
               orientation="h", marker_color="#9b59b6", showlegend=False, name="Ethnicity"),
        row=2, col=1,
    )

    country_counts = df["country_of_residence"].dropna().value_counts().head(10)
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
        title="Participant demographics (full data + screened out)",
        height=600,
        margin={"l": 120, "r": 40, "t": 80, "b": 60},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 8 — Temporal engagement profile (cumulative moves + move rate)
# ---------------------------------------------------------------------------


def fig_move_temporal_profile(df_trials: pd.DataFrame) -> go.Figure:
    """
    2x2 grid figure of temporal move patterns per cohort.

    Top row (colspan 2): time of last move per subject — violin + dots ± SD,
                         styled like fig_trial_duration.
    Bottom left:  average cumulative-move curve (time fraction vs. move fraction)
                  with ±1 SE ribbon. Diagonal = uniform activity.
    Bottom right: average move rate (moves/s, 5 s bins) over absolute time
                  with ±1 SE ribbon and a 60 s reference line.
    """
    df_trials = _with_major_version(df_trials)
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
                legendgroup=_vlabel(v),
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
                legendgroup=_vlabel(v),
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
        legend={"groupclick": "togglegroup"},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 9 — Pairwise distance distribution + KS distance from null (2x2)
# ---------------------------------------------------------------------------


def trial_ks_distance(pw_json: str, null_distribution: np.ndarray) -> float:
    """KS D statistic of one trial's pairwise distances against a null distribution.
    Shared by fig_pairwise_distance_distribution and the notebook's KS-vs-null
    cohort test."""
    dists = list(parse_pairwise_distances(pw_json).values())
    if len(dists) < 2:
        return np.nan
    D, _ = ks_2samp(dists, null_distribution)
    return D


def fig_pairwise_distance_distribution(
    df_trials: pd.DataFrame,
    null_distribution: np.ndarray,
) -> go.Figure:
    """
    2x2 grid: top (spans both cols) = per-version KDE of pairwise SpAM distances
    vs. a random-placement null; bottom-left = per-version deviation from null
    (KDE_version − KDE_null); bottom-right = per-subject KS distance (D statistic)
    from the null, violin + dots ± SE (same layout as the old standalone
    fig_ks_distance_per_subject, now merged in here since both panels share the
    same null_distribution input and distance-distribution theme).
    """
    df_trials = _with_major_version(df_trials)
    versions = _sorted_versions(df_trials)
    multi = len(versions) > 1

    version_dists: dict[float, list[float]] = {}
    total_obs = 0
    for v in versions:
        vdf = df_trials[df_trials["task_version"] == v]
        dists = [
            dist
            for pw_json in vdf["pairwise_distances"]
            for dist in parse_pairwise_distances(pw_json).values()
        ]
        version_dists[v] = dists
        total_obs += len(dists)

    x = np.linspace(0.0, 1.0, 500)
    version_kde: dict[float, np.ndarray] = {
        v: gaussian_kde(dists)(x)
        for v, dists in version_dists.items()
        if len(dists) > 1
    }
    null_kde = gaussian_kde(null_distribution)(x)

    df_ks = df_trials.assign(
        ks_D=df_trials["pairwise_distances"].apply(trial_ks_distance, null_distribution=null_distribution)
    )

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=[
            f"Distribution of pairwise SpAM distances (n={total_obs:,} pair observations)",
            "Δ density vs. null",
            "KS distance from null per subject",
        ],
        row_heights=[0.45, 0.55],
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )

    # Top: null area + version KDE lines
    fig.add_trace(go.Scatter(
        x=x.tolist(), y=null_kde.tolist(),
        fill="tozeroy",
        fillcolor="rgba(160,160,160,0.30)",
        line={"color": "rgba(100,100,100,0.70)", "width": 1.5},
        name="Null (random placement)",
    ), row=1, col=1)
    for v in versions:
        if v not in version_kde:
            continue
        vc = _vc(v)
        fig.add_trace(go.Scatter(
            x=x.tolist(), y=version_kde[v].tolist(),
            line={"color": vc["main"], "width": 2.5},
            name=_vlabel(v) if multi else "SpAM distance",
            legendgroup=_vlabel(v),
        ), row=1, col=1)

    # Bottom-left: deviation from null
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
            legendgroup=_vlabel(v),
            showlegend=False,
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0], y=[0.0, 0.0],
        line={"color": "rgba(80,80,80,0.55)", "width": 1.2, "dash": "dot"},
        showlegend=False,
    ), row=2, col=1)

    # Bottom-right: KS distance per subject (violin + dots ± SE)
    for i, v in enumerate(versions):
        vc = _vc(v)
        vdf = df_ks[df_ks["task_version"] == v]
        stats = _subject_se(vdf, "ks_D")
        stats = stats.sort_values("mean").reset_index(drop=True)
        y_dots = i + np.linspace(-0.15, 0.15, len(stats))
        fig.add_trace(go.Violin(
            x=vdf["ks_D"], y0=i, orientation="h", side="both",
            fillcolor=vc["violin"], line_color=vc["violin_line"],
            name=_vlabel(v), legendgroup=_vlabel(v), showlegend=False,
            width=0.6, points=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=stats["mean"], y=y_dots.tolist(), mode="markers",
            marker={"color": vc["dot"], "size": 8, "symbol": "circle"},
            error_x={"type": "data", "array": stats["se"].tolist(), "visible": True,
                     "color": vc["dot_err"], "thickness": 1.5, "width": 4},
            showlegend=False, legendgroup=_vlabel(v),
            name=f"{_vlabel(v)} subject mean ± SE",
        ), row=2, col=2)

    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Normalised distance", row=1, col=1)
    fig.update_yaxes(title_text="Δ density vs null", row=2, col=1)
    fig.update_xaxes(title_text="Normalised distance", row=2, col=1)
    fig.update_xaxes(title_text="KS D (vs. null)", row=2, col=2)

    y_range = [-0.55, len(versions) - 0.45]
    if multi:
        fig.update_yaxes(range=y_range, tickvals=list(range(len(versions))),
                          ticktext=[_vlabel(v) for v in versions], row=2, col=2)
    else:
        fig.update_yaxes(range=y_range, visible=False, row=2, col=2)

    fig.update_layout(
        height=680,
        margin={"l": 80, "r": 40, "t": 60, "b": 60},
        showlegend=True,
        legend={"groupclick": "togglegroup"},
    )
    return fig


# ---------------------------------------------------------------------------
# Fig 10 — Reliability vs. distance (v3+ only)
# ---------------------------------------------------------------------------

_MINOR_VERSION_PALETTE = pcolors.qualitative.Set2
_OVERALL_COLOR = "#333333"


def _minor_version_color(idx: int) -> str:
    return _MINOR_VERSION_PALETTE[idx % len(_MINOR_VERSION_PALETTE)]


def _reliability_vs_distance_data(df_trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-repeat-trial-pair (trial-level) and per-image-pair (pair-level) tables
    relating test-retest reliability to distance magnitude, v3+ only.

    trial_df: participant_id, task_version, spearman_r, mean_dist, sd_dist
        (mean_dist/sd_dist computed over that repeat-pair's combined original+repeat
        distances)
    pair_df: task_version, pair_mean_dist ((d_orig+d_repeat)/2), sq_diff
        ((d_orig-d_repeat)^2, the input to the RMSE aggregation)
    """
    df_v3plus = df_trials[df_trials["task_version"] >= 3.0]
    trial_records = []
    pair_records = []

    for pid, df_s in df_v3plus.groupby("participant_id"):
        version = float(df_s["task_version"].iloc[0])
        pw_by_trial_id = dict(zip(df_s["trial_id"], df_s["pairwise_distances"]))
        repeats = df_s[df_s["repeat_of_trial"].notna() & df_s["reliability"].notna()]
        for _, row in repeats.iterrows():
            orig_pw_json = pw_by_trial_id.get(int(row["repeat_of_trial"]))
            if orig_pw_json is None:
                continue
            orig_dists = parse_pairwise_distances(orig_pw_json)
            repeat_dists = parse_pairwise_distances(row["pairwise_distances"])
            common = [p for p in repeat_dists if p in orig_dists]
            if len(common) < 3:
                continue
            d1 = np.array([orig_dists[p] for p in common])
            d2 = np.array([repeat_dists[p] for p in common])
            all_d = np.concatenate([d1, d2])

            trial_records.append({
                "participant_id": pid,
                "task_version": version,
                "spearman_r": float(row["reliability"]),
                "mean_dist": all_d.mean(),
                "sd_dist": all_d.std(ddof=1),
            })

            pair_mean = (d1 + d2) / 2
            sq_diff = (d1 - d2) ** 2
            for pm, sq in zip(pair_mean, sq_diff):
                pair_records.append({"task_version": version, "pair_mean_dist": pm, "sq_diff": sq})

    trial_df = pd.DataFrame(
        trial_records, columns=["participant_id", "task_version", "spearman_r", "mean_dist", "sd_dist"]
    )
    pair_df = pd.DataFrame(pair_records, columns=["task_version", "pair_mean_dist", "sq_diff"])
    return trial_df, pair_df


def _add_trend_line(fig, x, y, color, row, col, label, showlegend):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.ptp(x) == 0:
        return
    coeffs = np.polyfit(x, y, deg=1)
    x_line = np.array([x.min(), x.max()])
    y_line = np.polyval(coeffs, x_line)
    fig.add_trace(go.Scatter(
        x=x_line.tolist(), y=y_line.tolist(), mode="lines",
        line={"color": color, "width": 2.5},
        name=label, legendgroup=label, showlegend=showlegend,
        hoverinfo="skip",
    ), row=row, col=col)


def _binned_rmse(fig, df, color, label, showlegend, bins, show_raw_scatter):
    """RMSE = sqrt(mean(sq_diff)) per 0.05-wide bin of pair_mean_dist. SEM of the
    RMSE is approximated via the delta method from the SEM of the underlying
    squared residuals: SEM(sqrt(X)) ≈ SEM(X) / (2*sqrt(mean(X)))."""
    d = df.copy()
    d["bin"] = pd.cut(d["pair_mean_dist"], bins, include_lowest=True)
    grouped = d.groupby("bin", observed=True)["sq_diff"]
    mean_sq = grouped.mean()
    rmse = np.sqrt(mean_sq)
    n = grouped.count()
    sem_sq = grouped.std() / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem_rmse = (sem_sq / (2 * rmse)).fillna(0.0)
    centers = [interval.mid for interval in rmse.index]

    if show_raw_scatter:
        fig.add_trace(go.Scatter(
            x=d["pair_mean_dist"], y=np.sqrt(d["sq_diff"]), mode="markers",
            marker={"color": color, "size": 4, "opacity": 0.06},
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=centers, y=rmse.tolist(),
        error_y={"type": "data", "array": sem_rmse.tolist(), "visible": True},
        mode="lines+markers",
        line={"color": color, "width": 2.5}, marker={"size": 6, "color": color},
        name=label, legendgroup=label, showlegend=showlegend,
    ), row=2, col=1)


def fig_reliability_vs_distance(df_trials: pd.DataFrame) -> go.Figure:
    """
    3-row x 2-col grid relating test-retest reliability to distance magnitude,
    v3+ only. Each MINOR version gets its own color (dots + line), plus an
    "overall" trace pooling all v3+ versions together.

    Top-left:  per-trial-pair Spearman r vs. that pair's mean distance (raw
               scatter + simple linear trend line per version).
    Top-right: per-trial-pair Spearman r vs. that pair's distance SD (same).
    Bottom (spans rows 2-3, both cols): per-image-pair test-retest RMSE
               (sqrt(mean((d_orig-d_repeat)^2))), binned into 0.05-wide
               intervals of (d_orig+d_repeat)/2 spanning the full [0, 1] range,
               mean ± SEM per bin. Background scatter shows raw per-pair
               sqrt(sq_diff) values (per version only, not duplicated for
               "overall", to avoid overplotting the same ~34k points twice).
    """
    trial_df, pair_df = _reliability_vs_distance_data(df_trials)
    if trial_df.empty:
        raise ValueError("fig_reliability_vs_distance: no v3+ trial repeats found in df_trials")

    minor_versions = sorted(trial_df["task_version"].unique())
    bins = np.arange(0.0, 1.05, 0.05)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{}, {}], [{"rowspan": 2, "colspan": 2}, None], [None, None]],
        subplot_titles=[
            "Reliability vs. mean distance", "Reliability vs. distance SD",
            "Test-retest RMSE vs. distance magnitude",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # --- Top-left & top-right: raw per-trial-pair scatter + linear trend, per version ---
    for idx, v in enumerate(minor_versions):
        color = _minor_version_color(idx)
        vdf = trial_df[trial_df["task_version"] == v]
        label = f"v{v:g}"
        fig.add_trace(go.Scatter(
            x=vdf["mean_dist"], y=vdf["spearman_r"], mode="markers",
            marker={"color": color, "size": 7, "opacity": 0.75},
            name=label, legendgroup=label,
        ), row=1, col=1)
        _add_trend_line(fig, vdf["mean_dist"], vdf["spearman_r"], color, 1, 1, label, showlegend=False)

        fig.add_trace(go.Scatter(
            x=vdf["sd_dist"], y=vdf["spearman_r"], mode="markers",
            marker={"color": color, "size": 7, "opacity": 0.75},
            name=label, legendgroup=label, showlegend=False,
        ), row=1, col=2)
        _add_trend_line(fig, vdf["sd_dist"], vdf["spearman_r"], color, 1, 2, label, showlegend=False)

    # Overall pooled trace (dots + trend line), on top of the per-version colours
    fig.add_trace(go.Scatter(
        x=trial_df["mean_dist"], y=trial_df["spearman_r"], mode="markers",
        marker={"color": _OVERALL_COLOR, "size": 5, "opacity": 0.4, "symbol": "circle-open"},
        name="overall", legendgroup="overall",
    ), row=1, col=1)
    _add_trend_line(fig, trial_df["mean_dist"], trial_df["spearman_r"], _OVERALL_COLOR, 1, 1, "overall", showlegend=False)

    fig.add_trace(go.Scatter(
        x=trial_df["sd_dist"], y=trial_df["spearman_r"], mode="markers",
        marker={"color": _OVERALL_COLOR, "size": 5, "opacity": 0.4, "symbol": "circle-open"},
        name="overall", legendgroup="overall", showlegend=False,
    ), row=1, col=2)
    _add_trend_line(fig, trial_df["sd_dist"], trial_df["spearman_r"], _OVERALL_COLOR, 1, 2, "overall", showlegend=False)

    # --- Bottom: pair-level RMSE, binned into 0.05-wide intervals spanning [0,1] ---
    for idx, v in enumerate(minor_versions):
        color = _minor_version_color(idx)
        vdf = pair_df[pair_df["task_version"] == v]
        _binned_rmse(fig, vdf, color, f"v{v:g}", showlegend=False, bins=bins, show_raw_scatter=True)
    _binned_rmse(fig, pair_df, _OVERALL_COLOR, "overall", showlegend=False, bins=bins, show_raw_scatter=False)

    fig.update_xaxes(title_text="Trial mean distance", row=1, col=1)
    fig.update_yaxes(title_text="Spearman r (test-retest)", row=1, col=1)
    fig.update_xaxes(title_text="Trial SD of distance", row=1, col=2)
    fig.update_yaxes(title_text="Spearman r (test-retest)", row=1, col=2)
    fig.update_xaxes(title_text="Pair mean distance ((d_orig + d_repeat) / 2)", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Test-retest RMSE", row=2, col=1)

    fig.update_layout(
        title="Reliability vs. distance (v3+)",
        height=680,
        margin={"l": 70, "r": 40, "t": 80, "b": 60},
        showlegend=True,
        legend={"groupclick": "togglegroup"},
    )
    return fig
