"""
Pre- vs post-SHINE cohort comparison, data-quality layer (analysis/behavioral/
cohort_data_quality.ipynb): within-subject test-retest reliability (overall and
over repeat index) and pairwise-distance distributions vs the random-placement
null (Wasserstein primary, KS secondary).

Usage (from repo root):
    from analysis.behavioral.cohort_quality import (
        subject_reliability, pairwise_distance_values, subject_wasserstein_distance,
        subject_ks_distance, fig_reliability, fig_distance_vs_null,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, wasserstein_distance

from analysis.behavioral.cohort_comparison import COHORT_COLORS, COHORT_LABEL, COHORTS, cohort_colors, cohort_label
from analysis.pilot.figures import trial_ks_distance
from analysis.utils.parser import parse_pairwise_distances
from analysis.utils.plotting import add_violin_dots_panel


def subject_reliability(df_trials: pd.DataFrame, participant_ids: pd.Series | list) -> pd.Series:
    """Per-participant within-subject reliability: mean Spearman r across that
    subject's verbatim trial repeats (the 'reliability' column, precomputed
    per repeat trial by analysis/utils/parser.py). Index is participant_id."""
    df = df_trials[df_trials["participant_id"].isin(participant_ids)]
    return df.groupby("participant_id")["reliability"].mean().dropna()


def pairwise_distance_values(df_trials_main: pd.DataFrame) -> list[float]:
    """Flatten every main trial's pairwise_distances into one list of
    normalised SpAM distances (same unit as the null distribution simulated
    by analysis.pilot.simulate_null_distances.simulate())."""
    return [
        dist
        for pw_json in df_trials_main["pairwise_distances"]
        for dist in parse_pairwise_distances(pw_json).values()
    ]


def trial_wasserstein_distance(pw_json: str, null_distribution: np.ndarray) -> float:
    """Earth mover's (Wasserstein-1) distance between one trial's pairwise
    distances and *null_distribution*. Unlike the KS D statistic (max gap
    between CDFs, shape-only), Wasserstein integrates over the full gap
    between distributions, so it's sensitive to *how far* the trial's
    distances sit from the null, not just whether the two shapes differ --
    the more appropriate primary measure for a magnitude comparison here.
    Companion to analysis.pilot.figures.trial_ks_distance(), which is kept as
    a secondary/shape-robustness check."""
    dists = list(parse_pairwise_distances(pw_json).values())
    if len(dists) < 2:
        return np.nan
    return wasserstein_distance(dists, null_distribution)


def subject_wasserstein_distance(df_trials_main: pd.DataFrame, null_distribution: np.ndarray) -> pd.Series:
    """Per-participant mean Wasserstein distance of each main trial's
    pairwise-distance distribution from *null_distribution* -- primary
    distance-from-null measure (see trial_wasserstein_distance())."""
    w = df_trials_main["pairwise_distances"].apply(trial_wasserstein_distance, null_distribution=null_distribution)
    return w.groupby(df_trials_main["participant_id"]).mean()


def subject_ks_distance(df_trials_main: pd.DataFrame, null_distribution: np.ndarray) -> pd.Series:
    """Per-participant mean KS distance (D statistic) of each main trial's
    pairwise-distance distribution against *null_distribution* -- reuses
    analysis.pilot.figures.trial_ks_distance(), the same measure as
    analysis/pilot/analysis.ipynb's Test 3 (KS distance from null). Secondary
    to subject_wasserstein_distance() here (see its docstring); grouped by
    SHINE cohort instead of task version."""
    ks = df_trials_main["pairwise_distances"].apply(trial_ks_distance, null_distribution=null_distribution)
    return ks.groupby(df_trials_main["participant_id"]).mean()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_reliability(rel_by_cohort: dict[str, pd.Series]) -> go.Figure:
    """Within-subject reliability (per-subject mean Spearman r across trial
    repeats), one horizontal violin + individual subject dots per cohort --
    via analysis.utils.plotting.add_violin_dots_panel(), styled like
    analysis/pilot/figures.py's fig_reliability violin panel."""
    fig = make_subplots(rows=1, cols=1)
    violins = {c: rel_by_cohort[c].values for c in COHORTS}
    dots = {c: pd.DataFrame({"value": np.sort(rel_by_cohort[c].values)}) for c in COHORTS}
    add_violin_dots_panel(
        fig, violins, dots, row=1, colors=COHORT_COLORS, labels=COHORT_LABEL,
        xaxis_title="Within-subject reliability (Spearman r)",
    )
    fig.update_layout(
        template="plotly_white",
        title="Within-subject reliability by cohort",
        height=380, margin={"l": 100, "r": 40, "t": 60, "b": 60},
        showlegend=True,
    )
    return fig


def fig_distance_vs_null(
    dists_by_cohort: dict[str, list[float]],
    null_distribution: np.ndarray,
    wasserstein_by_cohort: dict[str, pd.Series],
    ks_by_cohort: dict[str, pd.Series],
) -> go.Figure:
    """2x3 grid: top row (spans all 3 cols) = per-cohort KDE of pairwise SpAM
    distances vs. the random-placement null; bottom-left = per-cohort
    deviation from null (KDE_cohort - KDE_null); bottom-middle = per-subject
    Wasserstein distance from null, violin + dots (primary distance-from-null
    measure, see trial_wasserstein_distance()); bottom-right = the same for
    KS distance (secondary/shape-robustness check). Adapted from
    analysis/pilot/figures.py's fig_pairwise_distance_distribution, grouped by
    SHINE cohort instead of task version."""
    x = np.linspace(0.0, 1.0, 500)
    null_kde = gaussian_kde(null_distribution)(x)
    cohort_kde = {c: gaussian_kde(dists_by_cohort[c])(x) for c in COHORTS}
    total_obs = sum(len(v) for v in dists_by_cohort.values())

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        subplot_titles=[
            f"Pairwise SpAM distance distribution (n={total_obs:,} pair observations)",
            "Δ density vs. null",
            "Wasserstein distance from null per subject (primary)",
            "KS distance from null per subject (secondary)",
        ],
        row_heights=[0.42, 0.58], vertical_spacing=0.18, horizontal_spacing=0.1,
    )

    fig.add_trace(go.Scatter(
        x=x.tolist(), y=null_kde.tolist(),
        fill="tozeroy", fillcolor="rgba(160,160,160,0.30)",
        line={"color": "rgba(100,100,100,0.70)", "width": 1.5},
        name="Null (random placement)",
    ), row=1, col=1)
    for cohort in COHORTS:
        cc = cohort_colors(cohort)
        fig.add_trace(go.Scatter(
            x=x.tolist(), y=cohort_kde[cohort].tolist(),
            line={"color": cc["main"], "width": 2.5},
            name=cohort_label(cohort), legendgroup=cohort,
        ), row=1, col=1)
        dev = (cohort_kde[cohort] - null_kde).tolist()
        fig.add_trace(go.Scatter(
            x=x.tolist(), y=dev,
            fill="tozeroy", fillcolor=cc["violin"],
            line={"color": cc["main"], "width": 1.5},
            name=cohort_label(cohort), legendgroup=cohort, showlegend=False,
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0], y=[0.0, 0.0],
        line={"color": "rgba(80,80,80,0.55)", "width": 1.2, "dash": "dot"},
        showlegend=False,
    ), row=2, col=1)

    for values_by_cohort, col_idx, xaxis_title in (
        (wasserstein_by_cohort, 2, "Wasserstein distance (vs. null)"),
        (ks_by_cohort, 3, "KS D (vs. null)"),
    ):
        violins = {c: values_by_cohort[c].values for c in COHORTS}
        dots = {c: pd.DataFrame({"value": np.sort(values_by_cohort[c].values)}) for c in COHORTS}
        add_violin_dots_panel(
            fig, violins, dots, row=2, col_idx=col_idx, colors=COHORT_COLORS, labels=COHORT_LABEL,
            xaxis_title=xaxis_title, show_violin_legend=False,
        )

    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Normalised distance", row=1, col=1)
    fig.update_yaxes(title_text="Δ density vs null", row=2, col=1)
    fig.update_xaxes(title_text="Normalised distance", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=680, width=1200,
        margin={"l": 80, "r": 40, "t": 60, "b": 60},
        showlegend=True,
    )
    return fig
