"""
Pre- vs post-SHINE cohort comparison, shared layer: cohort labels/colours, the
retained-subject loader, and the data-prep and figure helpers for the "do the
cohorts behave the same?" notebook (analysis/behavioral/cohort_comparison.ipynb).
Retained subjects are production, task_version>=4.0, status=="full data" -- the
same definition analysis.rdms.behavioral.build_behavioral_rdm() uses.

Companion modules, one per notebook question:
    cohort_quality.py    -- is the data equally good? (reliability, distance vs null)
    cohort_structure.py  -- do they give the same similarity structure? (MDS, permutation)

Statistical tests are deliberately NOT in these modules -- they are notebook
work, matching this repo's notebook-as-analysis-surface convention
(analysis/pilot/analysis.ipynb pairs with analysis/pilot/figures.py the same way).
General statistical helpers live in analysis/utils/stats.py and
analysis/utils/equivalence.py.

Usage (from repo root):
    from analysis.behavioral.cohort_comparison import (
        COHORTS, COHORT_LABEL, COHORT_COLORS, load_retained_cohorts,
        retained_participants, pair_coverage, main_trials, subject_trial_stats,
        fig_trial_time_moves, fig_trial_progression, fig_demographics, fig_retention_status,
    )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.rdms.behavioral import _build_trials, _mean_combine, _select_participants
from analysis.rdms.common import image_manifest_index
from analysis.utils.load_data import load_data
from analysis.utils.plotting import add_progression_panel, fig_group_status_bar, role_palette

# ---------------------------------------------------------------------------
# Cohort labels / colours
# ---------------------------------------------------------------------------

COHORTS = ("pre", "post")
COHORT_LABEL = {"pre": "Pre-SHINE", "post": "Post-SHINE"}

# Same validated colorblind-safe blue/orange pair used for v1.0/v2.0 in
# analysis/pilot/figures.py -- reused here via the shared palette generator
# rather than re-validating a new pair.
COHORT_COLORS: dict[str, dict[str, str]] = role_palette(COHORTS)


def cohort_colors(cohort: str) -> dict[str, str]:
    return COHORT_COLORS[cohort]


def cohort_label(cohort: str) -> str:
    return COHORT_LABEL[cohort]


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def retained_participants(df_participants: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Retained subjects for *cohort* ('pre' or 'post'): production cohort,
    status == 'full data', task_version >= 4.0 (analysis.rdms.behavioral's
    _select_participants(), pre-filtered to cohort == 'production' here so
    the caller doesn't have to pass an already prod_only-filtered frame)."""
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}, got {cohort!r}")
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


def load_retained_cohorts(cache_dir: Path) -> dict:
    """Load the parsed-data cache and derive the per-cohort retained-subject
    views every cohort notebook starts from. Returns df_participants,
    df_trials, and per-cohort dicts participants / participant_ids /
    trials_main (keyed by COHORTS)."""
    data = load_data(cache_dir)
    df_participants, df_trials = data["participants"], data["trials"]
    participants = {c: retained_participants(df_participants, c) for c in COHORTS}
    participant_ids = {c: participants[c]["participant_id"] for c in COHORTS}
    trials_main = {c: main_trials(df_trials, participant_ids[c]) for c in COHORTS}
    return {
        "df_participants": df_participants, "df_trials": df_trials, "participants": participants,
        "participant_ids": participant_ids, "trials_main": trials_main,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


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
    for cohort in COHORTS:
        cc = cohort_colors(cohort)
        tdf = trials_by_cohort[cohort]
        fig.add_trace(go.Violin(
            y=tdf["rt_s"], x0=cohort_label(cohort), name=cohort_label(cohort), legendgroup=cohort,
            marker_color=cc["main"], fillcolor=cc["violin"], line_color=cc["violin_line"],
            box_visible=True, meanline_visible=True, points=False, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Violin(
            y=tdf["num_moves"], x0=cohort_label(cohort), name=cohort_label(cohort), legendgroup=cohort,
            marker_color=cc["main"], fillcolor=cc["violin"], line_color=cc["violin_line"],
            box_visible=True, meanline_visible=True, points=False, showlegend=False,
        ), row=1, col=2)

        sdf = subj_stats_by_cohort[cohort]
        fig.add_trace(go.Scatter(
            x=sdf["num_moves"], y=sdf["rt_s"], mode="markers",
            marker={"color": cc["dot"], "size": 8, "opacity": 0.8},
            name=cohort_label(cohort), legendgroup=cohort, showlegend=False,
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
        fig, trials_by_cohort, "rt_s", "trial_id", row=1, colors=COHORT_COLORS, labels=COHORT_LABEL,
        xaxis_title="Trial number", yaxis_title="Trial duration (s)", show_legend=True,
    )
    add_progression_panel(
        fig, trials_by_cohort, "num_moves", "trial_id", row=2, colors=COHORT_COLORS, labels=COHORT_LABEL,
        xaxis_title="Trial number", yaxis_title="Moves per trial", show_legend=False,
    )

    fig.update_layout(
        template="plotly_white",
        title="Trial duration & moves over task progression, by cohort",
        height=620, margin={"l": 60, "r": 40, "t": 70, "b": 60},
        showlegend=True, legend={"groupclick": "togglegroup"},
    )
    return fig


def fig_demographics(participants_by_cohort: dict[str, pd.DataFrame]) -> go.Figure:
    """Age (grouped histogram) and sex (grouped bar) distributions, one color
    per cohort, restricted to retained subjects."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Age distribution", "Sex"])
    sexes = sorted({s for df in participants_by_cohort.values() for s in df["sex"].dropna().unique()})

    for cohort in COHORTS:
        cc = cohort_colors(cohort)
        df = participants_by_cohort[cohort]
        fig.add_trace(go.Histogram(
            x=df["age"].dropna(), nbinsx=12, marker_color=cc["main"],
            name=cohort_label(cohort), legendgroup=cohort,
        ), row=1, col=1)
        sex_counts = df["sex"].value_counts()
        fig.add_trace(go.Bar(
            x=sexes, y=[int(sex_counts.get(s, 0)) for s in sexes],
            marker_color=cc["main"], name=cohort_label(cohort), legendgroup=cohort, showlegend=False,
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
    panels = {c: participants_v4plus[participants_v4plus["shine_variant"] == c] for c in COHORTS}
    return fig_group_status_bar(
        panels, "status", status_order=_STATUS_ORDER, status_labels=_STATUS_LABEL_MAP,
        bar_colors={c: cohort_colors(c)["main"] for c in COHORTS}, panel_titles={c: cohort_label(c) for c in COHORTS},
        title="Screening outcome by cohort (all task_version>=4.0 production participants)",
        height=320, width=900,
    )
