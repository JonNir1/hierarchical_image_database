"""
Pre- vs post-SHINE cohort comparison: data-prep and figure-building helpers for
retained (production, task_version>=4.0, status=="full data") SpAM participants --
same retained-subject definition analysis.rdms.behavioral.build_behavioral_rdm()
uses to build the pre_shine/post_shine RDMs these comparisons ultimately feed.

Statistical tests are deliberately NOT in this module -- they're notebook work,
see analysis/behavioral/cohort_comparison.ipynb, matching this repo's existing
notebook-as-analysis-surface convention (analysis/pilot/analysis.ipynb pairs with
analysis/pilot/figures.py the same way).

MDS-embedding logic (classical eigenspectrum, the cached weighted-SMACOF sweep,
cross-embedding comparison) and general statistical helpers (masked rank
correlation, Mann-Whitney effect size) live in analysis/rdms/mds.py and
analysis/utils/stats.py respectively -- shared with analysis/behavioral/
generalizability.py, analysis/rdms/validate_rdms.py, and
analysis/rsatoolbox_eval/mds_deep_dive.py rather than reimplemented here.

Usage (from repo root):
    from analysis.behavioral.cohort_comparison import (
        retained_participants, pair_coverage, subject_reliability,
        main_trials, subject_trial_stats, pairwise_distance_values,
        subject_wasserstein_distance, subject_ks_distance,
        fig_reliability, fig_trial_time_moves, fig_trial_progression,
        fig_distance_vs_null, fig_demographics, fig_retention_status,
        fig_eigenvalue_scree, fig_stress_scree, fig_cross_cohort_agreement,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, wasserstein_distance

from analysis.pilot.figures import trial_ks_distance
from analysis.rdms.behavioral import _build_trials, _mean_combine, _select_participants
from analysis.rdms.common import REPO_ROOT, image_manifest_index
from analysis.utils.parser import parse_pairwise_distances
from analysis.utils.plotting import add_progression_panel, add_violin_dots_panel, fig_group_status_bar, role_palette

MDS_CACHE_DIR = REPO_ROOT / "analysis" / "results" / "behavioral" / "mds_cache"

# ---------------------------------------------------------------------------
# Cohort labels / colours
# ---------------------------------------------------------------------------

_COHORTS = ("pre", "post")
_COHORT_LABEL = {"pre": "Pre-SHINE", "post": "Post-SHINE"}

# Same validated colorblind-safe blue/orange pair used for v1.0/v2.0 in
# analysis/pilot/figures.py -- reused here via the shared palette generator
# rather than re-validating a new pair.
_COHORT_COLORS: dict[str, dict[str, str]] = role_palette(_COHORTS)


def _cc(cohort: str) -> dict[str, str]:
    return _COHORT_COLORS[cohort]


def _label(cohort: str) -> str:
    return _COHORT_LABEL[cohort]


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def retained_participants(df_participants: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Retained subjects for *cohort* ('pre' or 'post'): production cohort,
    status == 'full data', task_version >= 4.0 (analysis.rdms.behavioral's
    _select_participants(), pre-filtered to cohort == 'production' here so
    the caller doesn't have to pass an already prod_only-filtered frame)."""
    if cohort not in _COHORTS:
        raise ValueError(f"cohort must be one of {_COHORTS}, got {cohort!r}")
    prod = df_participants[df_participants["cohort"] == "production"]
    return _select_participants(prod, f"{cohort}_shine")


def pair_coverage(df_trials: pd.DataFrame, participant_ids: pd.Series | list, cohort: str) -> dict:
    """Coverage of the full 725-image pair space by >=1 SpAM judgement from
    *participant_ids*, via the same unweighted trial aggregation
    analysis.rdms.behavioral._mean_combine() uses (screening + experimental
    trials pooled -- see the project's screening-pass-subjects-use-full-data
    convention).

    Returns {"pct_covered", "mean_per_pair", "sem_per_pair", "n_obs_per_pair"};
    mean/sem are computed over covered pairs only (n_obs_per_pair > 0).
    """
    image_to_idx, n_conds = image_manifest_index()
    trials = _build_trials(df_trials, participant_ids, f"{cohort}_shine", image_to_idx, n_conds)
    _, n_obs_per_pair = _mean_combine(trials, n_conds)
    covered = n_obs_per_pair[n_obs_per_pair > 0]
    return {
        "pct_covered":    100.0 * len(covered) / len(n_obs_per_pair),
        "mean_per_pair":  float(covered.mean()),
        "sem_per_pair":   float(covered.std(ddof=1) / np.sqrt(len(covered))),
        "n_obs_per_pair": n_obs_per_pair,
    }


def subject_reliability(df_trials: pd.DataFrame, participant_ids: pd.Series | list) -> pd.Series:
    """Per-participant within-subject reliability: mean Spearman r across that
    subject's verbatim trial repeats (the 'reliability' column, precomputed
    per repeat trial by analysis/utils/parser.py). Index is participant_id."""
    df = df_trials[df_trials["participant_id"].isin(participant_ids)]
    return df.groupby("participant_id")["reliability"].mean().dropna()


def main_trials(df_trials: pd.DataFrame, participant_ids: pd.Series | list) -> pd.DataFrame:
    """Non-catch ('main') trials for *participant_ids*, with trial duration in
    seconds added as rt_s. Catch trials use a different images_per_trial (10
    vs. 20, see SpAM_Task/task_config.json) so mixing them in would confound
    the move-count / pairwise-distance comparisons below."""
    df = df_trials[df_trials["participant_id"].isin(participant_ids) & ~df_trials["is_catch"]]
    return df.assign(rt_s=df["rt"] / 1000)


def subject_trial_stats(df_trials_main: pd.DataFrame) -> pd.DataFrame:
    """Per-participant mean trial duration (s) and mean number of moves, over
    an already-filtered main_trials() frame."""
    return (
        df_trials_main.groupby("participant_id")
        .agg(rt_s=("rt_s", "mean"), num_moves=("num_moves", "mean"))
        .reset_index()
    )


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
    violins = {c: rel_by_cohort[c].values for c in _COHORTS}
    dots = {c: pd.DataFrame({"value": np.sort(rel_by_cohort[c].values)}) for c in _COHORTS}
    add_violin_dots_panel(
        fig, violins, dots, row=1, colors=_COHORT_COLORS, labels=_COHORT_LABEL,
        xaxis_title="Within-subject reliability (Spearman r)",
    )
    fig.update_layout(
        template="plotly_white",
        title="Within-subject reliability by cohort",
        height=380, margin={"l": 100, "r": 40, "t": 60, "b": 60},
        showlegend=True,
    )
    return fig


def fig_trial_time_moves(
    trials_by_cohort: dict[str, pd.DataFrame], subj_stats_by_cohort: dict[str, pd.DataFrame],
) -> go.Figure:
    """3-panel figure comparing trial dynamics between cohorts (main trials
    only): trial-duration distribution, number-of-moves distribution (both
    per-trial violins), and per-subject mean moves vs. mean duration scatter --
    styled like analysis/pilot/figures.py's fig_duration_vs_moves."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Trial duration", "Number of moves", "Moves vs. duration (per subject)"],
    )
    for cohort in _COHORTS:
        cc = _cc(cohort)
        tdf = trials_by_cohort[cohort]
        fig.add_trace(go.Violin(
            y=tdf["rt_s"], x0=_label(cohort), name=_label(cohort), legendgroup=cohort,
            marker_color=cc["main"], fillcolor=cc["violin"], line_color=cc["violin_line"],
            box_visible=True, meanline_visible=True, points=False, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Violin(
            y=tdf["num_moves"], x0=_label(cohort), name=_label(cohort), legendgroup=cohort,
            marker_color=cc["main"], fillcolor=cc["violin"], line_color=cc["violin_line"],
            box_visible=True, meanline_visible=True, points=False, showlegend=False,
        ), row=1, col=2)

        sdf = subj_stats_by_cohort[cohort]
        fig.add_trace(go.Scatter(
            x=sdf["num_moves"], y=sdf["rt_s"], mode="markers",
            marker={"color": cc["dot"], "size": 8, "opacity": 0.8},
            name=_label(cohort), legendgroup=cohort, showlegend=False,
            hovertemplate="Moves: %{x:.1f}<br>Duration: %{y:.1f}s<extra></extra>",
        ), row=1, col=3)

    fig.update_yaxes(title_text="Trial duration (s)", row=1, col=1)
    fig.update_yaxes(title_text="Moves per trial", row=1, col=2)
    fig.update_xaxes(title_text="Moves per trial (subject mean)", row=1, col=3)
    fig.update_yaxes(title_text="Trial duration (s, subject mean)", row=1, col=3)
    fig.update_layout(
        template="plotly_white",
        title="Trial dynamics by cohort (main trials)",
        height=460, width=1200,
        margin={"l": 60, "r": 40, "t": 70, "b": 60},
        showlegend=True,
    )
    return fig


def fig_trial_progression(trials_by_cohort: dict[str, pd.DataFrame]) -> go.Figure:
    """2-row figure: mean +/- SE trial duration (top) and number of moves
    (bottom) over trial number within the session (main trials only), with
    thin per-subject lines underneath -- via
    analysis.utils.plotting.add_progression_panel(), same progression-panel
    style as analysis/pilot/figures.py's fig_trial_duration/fig_moves bottom
    rows, grouped by SHINE cohort instead of task version."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Trial duration over task progression", "Number of moves over task progression"],
        vertical_spacing=0.15,
    )
    add_progression_panel(
        fig, trials_by_cohort, "rt_s", "trial_id", row=1, colors=_COHORT_COLORS, labels=_COHORT_LABEL,
        xaxis_title="Trial number", yaxis_title="Trial duration (s)", show_legend=True,
    )
    add_progression_panel(
        fig, trials_by_cohort, "num_moves", "trial_id", row=2, colors=_COHORT_COLORS, labels=_COHORT_LABEL,
        xaxis_title="Trial number", yaxis_title="Moves per trial", show_legend=False,
    )

    fig.update_layout(
        template="plotly_white",
        title="Trial duration & moves over task progression, by cohort",
        height=620, margin={"l": 60, "r": 40, "t": 70, "b": 60},
        showlegend=True, legend={"groupclick": "togglegroup"},
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
    cohort_kde = {c: gaussian_kde(dists_by_cohort[c])(x) for c in _COHORTS}
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
    for cohort in _COHORTS:
        cc = _cc(cohort)
        fig.add_trace(go.Scatter(
            x=x.tolist(), y=cohort_kde[cohort].tolist(),
            line={"color": cc["main"], "width": 2.5},
            name=_label(cohort), legendgroup=cohort,
        ), row=1, col=1)
        dev = (cohort_kde[cohort] - null_kde).tolist()
        fig.add_trace(go.Scatter(
            x=x.tolist(), y=dev,
            fill="tozeroy", fillcolor=cc["violin"],
            line={"color": cc["main"], "width": 1.5},
            name=_label(cohort), legendgroup=cohort, showlegend=False,
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
        violins = {c: values_by_cohort[c].values for c in _COHORTS}
        dots = {c: pd.DataFrame({"value": np.sort(values_by_cohort[c].values)}) for c in _COHORTS}
        add_violin_dots_panel(
            fig, violins, dots, row=2, col_idx=col_idx, colors=_COHORT_COLORS, labels=_COHORT_LABEL,
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


def fig_demographics(participants_by_cohort: dict[str, pd.DataFrame]) -> go.Figure:
    """Age (grouped histogram) and sex (grouped bar) distributions, one color
    per cohort, restricted to retained subjects."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Age distribution", "Sex"])
    sexes = sorted({s for df in participants_by_cohort.values() for s in df["sex"].dropna().unique()})

    for cohort in _COHORTS:
        cc = _cc(cohort)
        df = participants_by_cohort[cohort]
        fig.add_trace(go.Histogram(
            x=df["age"].dropna(), nbinsx=12, marker_color=cc["main"],
            name=_label(cohort), legendgroup=cohort,
        ), row=1, col=1)
        sex_counts = df["sex"].value_counts()
        fig.add_trace(go.Bar(
            x=sexes, y=[int(sex_counts.get(s, 0)) for s in sexes],
            marker_color=cc["main"], name=_label(cohort), legendgroup=cohort, showlegend=False,
        ), row=1, col=2)

    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        title="Participant demographics by cohort (retained subjects)",
        barmode="group", height=420, margin={"l": 60, "r": 40, "t": 70, "b": 60},
    )
    return fig


_STATUS_LABEL_MAP = {
    "full data":       "Full data",
    "screened out":    "Screened out",
    "false_positive":  "False positive (full-session recheck)",
    "revoked consent": "Revoked consent",
    "missing data":    "Missing data",
}
_STATUS_ORDER = list(_STATUS_LABEL_MAP)


def fig_retention_status(participants_v4plus: pd.DataFrame) -> go.Figure:
    """Participant status distribution (full data / screened out / false
    positive / revoked / missing) for all task_version>=4.0 production
    participants, one horizontal-bar panel per SHINE cohort -- via
    analysis.utils.plotting.fig_group_status_bar(), styled like
    analysis/pilot/figures.py's fig_completion_status, grouped by
    shine_variant instead of task-version major-version."""
    panels = {c: participants_v4plus[participants_v4plus["shine_variant"] == c] for c in _COHORTS}
    return fig_group_status_bar(
        panels, "status", status_order=_STATUS_ORDER, status_labels=_STATUS_LABEL_MAP,
        bar_colors={c: _cc(c)["main"] for c in _COHORTS}, panel_titles={c: _label(c) for c in _COHORTS},
        title="Screening outcome by cohort (all task_version>=4.0 production participants)",
        height=320, width=900,
    )


def fig_eigenvalue_scree(eig_by_cohort: dict[str, np.ndarray], max_dims: int) -> go.Figure:
    """Line+marker scree plot of the first *max_dims* classical-MDS
    eigenvalues, one trace per cohort. See
    analysis.rdms.mds.classical_mds_eigenspectrum()'s caveat -- qualitative
    view only, not a validated dimensionality estimate at this RDM's
    coverage level."""
    fig = go.Figure()
    x = list(range(1, max_dims + 1))
    for cohort in _COHORTS:
        cc = _cc(cohort)
        fig.add_trace(go.Scatter(
            x=x, y=eig_by_cohort[cohort][:max_dims].tolist(),
            mode="lines+markers", line={"color": cc["main"], "width": 2.5},
            marker={"size": 8, "color": cc["main"]}, name=_label(cohort),
        ))
    fig.update_xaxes(title_text="Component", tickvals=x)
    fig.update_yaxes(title_text="Eigenvalue")
    fig.update_layout(
        template="plotly_white",
        title=f"Classical-MDS eigenvalue scree (mean-imputed -- qualitative only, first {max_dims} components)",
        height=420, margin={"l": 70, "r": 40, "t": 70, "b": 60},
    )
    return fig


def fig_stress_scree(mds_results_by_cohort: dict[str, dict[int, dict]], dims: list[int]) -> go.Figure:
    """Line+marker scree plot of weighted-SMACOF stress across embedding
    dimensionality, one trace per cohort."""
    fig = go.Figure()
    for cohort in _COHORTS:
        cc = _cc(cohort)
        stresses = [mds_results_by_cohort[cohort][d]["stress"] for d in dims]
        fig.add_trace(go.Scatter(
            x=dims, y=stresses, mode="lines+markers",
            line={"color": cc["main"], "width": 2.5}, marker={"size": 8, "color": cc["main"]},
            name=_label(cohort),
        ))
    fig.update_xaxes(title_text="Embedding dimensionality", tickvals=list(dims))
    fig.update_yaxes(title_text="Stress-1 (SMACOF)")
    fig.update_layout(
        template="plotly_white",
        title="Weighted-SMACOF stress vs. embedding dimensionality",
        height=420, margin={"l": 70, "r": 40, "t": 70, "b": 60},
    )
    return fig


_AGREEMENT_COLORS = {"kendall": "#2a78d6", "spearman": "#1baf7a", "one_minus_m2": "#4a3aa7"}


def fig_cross_cohort_agreement(agreement_by_x: dict[str, dict], dims: list[int]) -> go.Figure:
    """3 traces (Kendall tau, Spearman rho, 1 - Procrustes m^2) of pre- vs.
    post-cohort structural agreement on a shared canvas, x-axis = 'raw RDM'
    plus each embedding dimensionality. Procrustes has no value at 'raw RDM'
    (undefined on an incomplete RDM -- only Kendall/Spearman, restricted to
    jointly-covered pairs, are defined there)."""
    x_labels = ["raw RDM"] + [str(d) for d in dims]
    kendall = [agreement_by_x[k]["kendall"] for k in x_labels]
    spearman = [agreement_by_x[k]["spearman"] for k in x_labels]
    one_minus_m2 = [
        (1 - agreement_by_x[k]["procrustes_m2"]) if agreement_by_x[k].get("procrustes_m2") is not None else None
        for k in x_labels
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=kendall, mode="lines+markers", name="Kendall tau", line={"color": _AGREEMENT_COLORS["kendall"], "width": 2.5}, marker={"size": 8}))
    fig.add_trace(go.Scatter(x=x_labels, y=spearman, mode="lines+markers", name="Spearman rho", line={"color": _AGREEMENT_COLORS["spearman"], "width": 2.5}, marker={"size": 8}))
    fig.add_trace(go.Scatter(x=x_labels, y=one_minus_m2, mode="lines+markers", name="1 - Procrustes m2", line={"color": _AGREEMENT_COLORS["one_minus_m2"], "width": 2.5}, marker={"size": 8}))

    fig.update_xaxes(title_text="Comparison", type="category")
    fig.update_yaxes(title_text="Agreement (bigger = better)")
    fig.update_layout(
        template="plotly_white",
        title="Pre- vs. post-SHINE structural agreement",
        height=420, margin={"l": 70, "r": 40, "t": 70, "b": 60},
        showlegend=True,
    )
    return fig
