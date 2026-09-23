"""
Pre- vs post-SHINE cohort comparison: data-prep and figure-building helpers for
retained (production, task_version>=4.0, status=="full data") SpAM participants --
same retained-subject definition analysis.rdms.behavioral.build_behavioral_rdm()
uses to build the pre_shine/post_shine RDMs these comparisons ultimately feed.

Statistical tests are deliberately NOT in this module -- they're notebook work,
see analysis/behavioral/cohort_comparison.ipynb, matching this repo's existing
notebook-as-analysis-surface convention (analysis/pilot/analysis.ipynb pairs with
analysis/pilot/figures.py the same way).

Usage (from repo root):
    from analysis.behavioral.cohort_comparison import (
        retained_participants, pair_coverage, subject_reliability,
        main_trials, subject_trial_stats, pairwise_distance_values,
        subject_ks_distance, mwu_two_group,
        fig_reliability, fig_trial_time_moves, fig_distance_vs_null,
        fig_demographics, fig_retention_status,
    )
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, mannwhitneyu

from analysis.pilot.figures import trial_ks_distance
from analysis.rdms.behavioral import _build_trials, _mean_combine, _select_participants
from analysis.rdms.common import load_manifest
from analysis.utils.parser import parse_pairwise_distances

# ---------------------------------------------------------------------------
# Cohort labels / colours
# ---------------------------------------------------------------------------

_COHORTS = ("pre", "post")
_COHORT_LABEL = {"pre": "Pre-SHINE", "post": "Post-SHINE"}

# Same validated colorblind-safe blue/orange pair used for v1.0/v2.0 in
# analysis/pilot/figures.py -- reused here rather than re-validating a new pair.
_COHORT_COLORS: dict[str, dict[str, str]] = {
    "pre": {
        "main":        "#2a78d6",
        "violin":      "rgba(42,120,214,0.35)",
        "violin_line": "rgba(42,120,214,0.9)",
        "dot":         "rgba(42,120,214,0.9)",
        "dot_err":     "rgba(42,120,214,0.6)",
    },
    "post": {
        "main":        "#eb6834",
        "violin":      "rgba(235,104,52,0.35)",
        "violin_line": "rgba(235,104,52,0.9)",
        "dot":         "rgba(235,104,52,0.9)",
        "dot_err":     "rgba(235,104,52,0.6)",
    },
}


def _cc(cohort: str) -> dict[str, str]:
    return _COHORT_COLORS[cohort]


def _label(cohort: str) -> str:
    return _COHORT_LABEL[cohort]


@lru_cache(maxsize=1)
def _manifest_index() -> tuple[dict[str, int], int]:
    manifest = load_manifest()
    canonical = [p.replace("\\", "/") for p in manifest["curated_path"]]
    return {img: i for i, img in enumerate(canonical)}, len(canonical)


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
    image_to_idx, n_conds = _manifest_index()
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


def subject_ks_distance(df_trials_main: pd.DataFrame, null_distribution: np.ndarray) -> pd.Series:
    """Per-participant mean KS distance (D statistic) of each main trial's
    pairwise-distance distribution against *null_distribution* -- reuses
    analysis.pilot.figures.trial_ks_distance(), the same measure as
    analysis/pilot/analysis.ipynb's Test 3 (KS distance from null), here
    grouped by SHINE cohort instead of task version."""
    ks = df_trials_main["pairwise_distances"].apply(trial_ks_distance, null_distribution=null_distribution)
    return ks.groupby(df_trials_main["participant_id"]).mean()


def mwu_two_group(values_a, values_b, label_a: str, label_b: str) -> dict:
    """Two-sided Mann-Whitney U between two independent samples, with
    rank-biserial effect size r = 1 - 2U/(n_A*n_B) (same formula as
    analysis/pilot/analysis.ipynb's mwu_compare()). Only two cohorts are ever
    compared here, so no multiple-comparison correction is applied."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - 2 * U / (len(a) * len(b))
    return {
        "comparison": f"{label_a} vs {label_b}",
        "n (A)": len(a), "n (B)": len(b),
        "median (A)": round(float(np.median(a)), 4), "median (B)": round(float(np.median(b)), 4),
        "mean (A)": round(float(np.mean(a)), 4), "mean (B)": round(float(np.mean(b)), 4),
        "std (A)": round(float(np.std(a, ddof=1)), 4), "std (B)": round(float(np.std(b, ddof=1)), 4),
        "U": round(float(U), 1), "p": float(p), "r": round(float(r), 4),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_reliability(rel_by_cohort: dict[str, pd.Series]) -> go.Figure:
    """Within-subject reliability (per-subject mean Spearman r across trial
    repeats), one horizontal violin + individual subject dots per cohort --
    styled like analysis/pilot/figures.py's fig_reliability violin panel."""
    fig = go.Figure()
    for i, cohort in enumerate(_COHORTS):
        cc = _cc(cohort)
        series = rel_by_cohort[cohort]
        fig.add_trace(go.Violin(
            x=series.values, y0=i, orientation="h", side="both",
            fillcolor=cc["violin"], line_color=cc["violin_line"],
            name=_label(cohort), legendgroup=cohort, showlegend=True,
            width=0.6, points=False,
        ))
        y_dots = i + np.linspace(-0.15, 0.15, len(series))
        fig.add_trace(go.Scatter(
            x=np.sort(series.values), y=y_dots, mode="markers",
            marker={"color": cc["dot"], "size": 7, "opacity": 0.8},
            showlegend=False, legendgroup=cohort,
            name=f"{_label(cohort)} subjects",
        ))

    fig.update_yaxes(
        range=[-0.6, len(_COHORTS) - 0.4],
        tickvals=list(range(len(_COHORTS))), ticktext=[_label(c) for c in _COHORTS],
    )
    fig.update_xaxes(title_text="Within-subject reliability (Spearman r)")
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


def fig_distance_vs_null(
    dists_by_cohort: dict[str, list[float]],
    null_distribution: np.ndarray,
    ks_by_cohort: dict[str, pd.Series],
) -> go.Figure:
    """2x2 grid: top (spans both cols) = per-cohort KDE of pairwise SpAM
    distances vs. the random-placement null; bottom-left = per-cohort
    deviation from null (KDE_cohort - KDE_null); bottom-right = per-subject
    KS distance from null, violin + dots. Adapted from
    analysis/pilot/figures.py's fig_pairwise_distance_distribution, grouped by
    SHINE cohort instead of task version."""
    x = np.linspace(0.0, 1.0, 500)
    null_kde = gaussian_kde(null_distribution)(x)
    cohort_kde = {c: gaussian_kde(dists_by_cohort[c])(x) for c in _COHORTS}
    total_obs = sum(len(v) for v in dists_by_cohort.values())

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=[
            f"Pairwise SpAM distance distribution (n={total_obs:,} pair observations)",
            "Δ density vs. null", "KS distance from null per subject",
        ],
        row_heights=[0.45, 0.55], vertical_spacing=0.16, horizontal_spacing=0.12,
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

    for i, cohort in enumerate(_COHORTS):
        cc = _cc(cohort)
        ks = ks_by_cohort[cohort]
        fig.add_trace(go.Violin(
            x=ks.values, y0=i, orientation="h", side="both",
            fillcolor=cc["violin"], line_color=cc["violin_line"],
            name=_label(cohort), legendgroup=cohort, showlegend=False,
            width=0.6, points=False,
        ), row=2, col=2)
        y_dots = i + np.linspace(-0.15, 0.15, len(ks))
        fig.add_trace(go.Scatter(
            x=np.sort(ks.values), y=y_dots, mode="markers",
            marker={"color": cc["dot"], "size": 7, "opacity": 0.8},
            showlegend=False, legendgroup=cohort,
        ), row=2, col=2)

    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Normalised distance", row=1, col=1)
    fig.update_yaxes(title_text="Δ density vs null", row=2, col=1)
    fig.update_xaxes(title_text="Normalised distance", row=2, col=1)
    fig.update_xaxes(title_text="KS D (vs. null)", row=2, col=2)
    fig.update_yaxes(
        range=[-0.6, len(_COHORTS) - 0.4],
        tickvals=list(range(len(_COHORTS))), ticktext=[_label(c) for c in _COHORTS],
        row=2, col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=680, margin={"l": 80, "r": 40, "t": 60, "b": 60},
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
    participants, one horizontal-bar panel per SHINE cohort -- styled like
    analysis/pilot/figures.py's fig_completion_status, grouped by
    shine_variant instead of task-version major-version."""
    fig = make_subplots(rows=1, cols=len(_COHORTS), subplot_titles=[_label(c) for c in _COHORTS])
    for col, cohort in enumerate(_COHORTS, start=1):
        cc = _cc(cohort)
        df = participants_v4plus[participants_v4plus["shine_variant"] == cohort]
        counts = df["status"].value_counts()
        total = counts.sum()
        cats = [c for c in _STATUS_ORDER if c in counts.index]
        pcts = [100 * counts[c] / total for c in cats]
        fig.add_trace(go.Bar(
            x=pcts, y=[_STATUS_LABEL_MAP[c] for c in cats], orientation="h",
            text=[f"{p:.1f}% (n={counts[c]})" for p, c in zip(pcts, cats)],
            textposition="auto", marker_color=cc["main"], showlegend=False,
        ), row=1, col=col)
        fig.update_xaxes(range=[0, 100], title_text="% of participants", row=1, col=col)
        fig.update_yaxes(autorange="reversed", row=1, col=col)

    fig.update_layout(
        template="plotly_white",
        title="Screening outcome by cohort (all task_version>=4.0 production participants)",
        height=320, width=900, margin={"l": 190, "r": 40, "t": 70, "b": 50},
    )
    return fig
