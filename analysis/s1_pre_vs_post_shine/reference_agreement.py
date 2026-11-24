"""
Pre- vs post-SHINE cohort comparison, RDM-level agreement with a reference RDM (analysis/s1_pre_vs_post_shine/
4_semantic_hierarchy.ipynb and 5_ground_truth.ipynb): does each cohort's behavioral
RDM rank-correlate with a reference (the Kiani directory-tree RDM, or an external ratings/recordings
RDM on a shared image subset) beyond chance, how stable is that across participants, and are the two
cohorts equally similar to it?

Why not rsatoolbox.inference.eval_bootstrap_rdm(): each participant's own RDM covers ~1.3% of pairs
(disjoint per-trial image sets), and rsatoolbox's compare() demands identical NaN positions in both
RDMs, so it raises on this data. The same quantities are computed here directly. Kendall tau-a is the
primary statistic (the reference RDMs are heavily tied), Spearman rho the secondary one.

  (The image-relabeling permutation test, restricted to the pairs the behavioral RDM observed, is the
      shared analysis.utils.stats.label_shuffle_test.)
  jackknife_combined_fit_multi -- delete-one participant jackknife of the combined RDM's fit to several
      references (SE and CI via analysis.utils.stats.jackknife_se / jackknife_ci, centred on the estimate).
  loo_model_fit -- Nili et al. (2014) style: each left-out participant's own RDM scored against the
      reference, next to the same participant's lower-bound score against the combined others.
  jackknife_cohort_difference + cohort_difference_equivalence -- pre minus post fit, each cohort
      jackknifed on its own, with a TOST on the jackknife SE (analysis.utils.equivalence.equivalence_test_se_effect,
      the project's 80%-power bound, closed form).

A participant bootstrap was used before; on these sparse RDMs its draws are attenuated (a resample has about
63% distinct participants), so its CIs sat below the estimates. See analysis.s0_build_rdms.behavior.resampling.
  bh_within_cohort -- Benjamini-Hochberg within each cohort, never pooled across cohorts.

Every expensive call takes `cache_path` (analysis.utils.disk_cache): results are reused only under a
matching fingerprint of the inputs.

Usage (from repo root):
    from analysis.s1_pre_vs_post_shine.reference_agreement import (
        jackknife_combined_fit_multi, loo_model_fit,
        jackknife_cohort_difference, cohort_difference_equivalence, bh_within_cohort,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.stats.multitest import multipletests

from analysis.s0_build_rdms.behavior.resampling import participant_jackknife
from analysis.s0_build_rdms.behavior.spam_ratings import _combine, _mean_combine
from analysis.s0_build_rdms.common import REPO_ROOT, subset_condensed
from analysis.s1_pre_vs_post_shine.demographics_and_behavior import COHORTS, cohort_colors, cohort_label, cohort_symbol
from analysis.utils.disk_cache import cached_frame, fingerprint
from analysis.utils.equivalence import equivalence_test_se_effect
from analysis.utils.stats import jackknife_ci, jackknife_se, masked_rank_correlation, percentile_ci

REFERENCE_CACHE_DIR = REPO_ROOT / "analysis" / "results" / "pre_vs_post_shine" / "reference_cache"

TrialsByParticipant = dict[str, list[tuple[np.ndarray, np.ndarray]]]

_STAT_KEYS = ("kendall_a", "spearman")
_STAT_LABEL = {"kendall_a": "Kendall tau-a", "spearman": "Spearman rho"}


def split_by_cohort(
    trials_by_subject: TrialsByParticipant, subject_ids: Mapping[str, Sequence[str]],
) -> dict[str, TrialsByParticipant]:
    """{cohort: {subject: trials}} from geometry.parse_subject_trials()'s pooled {subject: trials}."""
    return {c: {pid: trials_by_subject[pid] for pid in subject_ids[c]} for c in subject_ids}


def _digest(trials: TrialsByParticipant) -> list[tuple]:
    """Cheap content fingerprint of parsed trials: (subject, n trials, sum of values)."""
    return [(pid, len(ts), round(float(sum(v.sum() for _, v in ts)), 6)) for pid, ts in trials.items()]


def _stats(perceptual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    result = masked_rank_correlation(perceptual, reference, tau_a=True)
    return {"kendall_a": result["kendall_a"], "spearman": result["spearman"], "n_overlap": result["n_overlap"]}


# ---------------------------------------------------------------------------
# Participant bootstraps and per-participant fit
# ---------------------------------------------------------------------------


def _scored(rdm: np.ndarray, subset: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray:
    rdm = subset_condensed(rdm, subset) if subset is not None else rdm
    return rdm if mask is None else np.where(mask, rdm, np.nan)


def jackknife_combined_fit_multi(
    trials_by_participant: TrialsByParticipant, references: dict[str, np.ndarray], n_conds: int,
    *, aggregation: str = "simple", subset: np.ndarray | None = None, n_jobs: int = 6, cache_path: Path | None = None,
) -> pd.DataFrame:
    """Delete-one participant jackknife of the combined RDM's correlation with several reference RDMs at once.

    Each participant is left out in turn, the others' trials combined with `aggregation` once and scored
    against every reference; the jackknife SE and CI (analysis.utils.stats.jackknife_se / jackknife_ci) come
    from these values, centred on the full-sample estimate. One row per (left-out participant, reference),
    columns left_out, model, kendall_a, spearman, n_overlap. With `subset` (manifest row indices) the
    combined RDM is restricted to those images first, so every reference must be that subset's RDM.
    See analysis.s0_build_rdms.behavior.resampling for why this replaces a participant bootstrap.
    """
    def score(rdm: np.ndarray) -> list[dict]:
        scored = _scored(rdm, subset, None)
        return [{"model": name, **_stats(scored, ref)} for name, ref in references.items()]

    def run() -> pd.DataFrame:
        return participant_jackknife(trials_by_participant, n_conds, score, aggregation=aggregation, n_jobs=n_jobs)

    key = fingerprint(_digest(trials_by_participant), list(references), *references.values(), n_conds,
                      aggregation, subset, "jackknife")
    return cached_frame(cache_path, key, run)


def _difference_score(side: str, other: np.ndarray, references: dict[str, np.ndarray], subset: np.ndarray | None,
                      masks: dict[str, np.ndarray | None]):
    """Scorer for one cohort's leave-one-out RDMs against the other cohort's full RDM."""
    def score(rdm: np.ndarray) -> list[dict]:
        rdm_a, rdm_b = (rdm, other) if side == "a" else (other, rdm)
        rows = []
        for pairs, mask in masks.items():
            scored_a, scored_b = _scored(rdm_a, subset, mask), _scored(rdm_b, subset, mask)
            for name, ref in references.items():
                stats_a, stats_b = _stats(scored_a, ref), _stats(scored_b, ref)
                row = {"side": side, "pairs": pairs, "model": name,
                       "n_overlap_a": stats_a["n_overlap"], "n_overlap_b": stats_b["n_overlap"]}
                for k in _STAT_KEYS:
                    row.update({f"{k}_a": stats_a[k], f"{k}_b": stats_b[k], f"{k}_diff": stats_a[k] - stats_b[k]})
                rows.append(row)
        return rows

    return score


def jackknife_cohort_difference(
    trials_a: TrialsByParticipant, trials_b: TrialsByParticipant, references: dict[str, np.ndarray], n_conds: int,
    *, aggregation: str = "simple", subset: np.ndarray | None = None,
    pair_masks: dict[str, np.ndarray | None] | None = None, n_jobs: int = 6, cache_path: Path | None = None,
) -> pd.DataFrame:
    """Delete-one jackknife of (cohort A - cohort B) in each cohort's combined-RDM correlation with each reference.

    The cohorts are disjoint participants, so each is jackknifed on its own with the other at full size;
    the SE of the difference is the root of the sum of the two jackknife variances (difference_se()).
    `pair_masks` names the pair sets to score on (default {"own": None}, each cohort on the pairs it
    observed); a boolean mask over the subset's condensed pairs restricts both cohorts to those pairs.
    One row per (side, left-out participant, pair set, reference): `side` says which cohort lost a
    participant, with `<stat>_a`, `<stat>_b` and `<stat>_diff` for each statistic.
    """
    masks = {"own": None} if pair_masks is None else pair_masks

    def run() -> pd.DataFrame:
        full = {side: _combine([t for ts in trials.values() for t in ts], n_conds, aggregation)[0]
                for side, trials in (("a", trials_a), ("b", trials_b))}
        frames = [
            participant_jackknife(trials, n_conds, _difference_score(side, full["b" if side == "a" else "a"],
                                                                     references, subset, masks),
                                  aggregation=aggregation, n_jobs=n_jobs)
            for side, trials in (("a", trials_a), ("b", trials_b))
        ]
        return pd.concat(frames, ignore_index=True)

    key = fingerprint(_digest(trials_a), _digest(trials_b), list(references), *references.values(), n_conds,
                      aggregation, subset, list(masks), *masks.values(), "jackknife")
    return cached_frame(cache_path, key, run)


def difference_se(draws: pd.DataFrame, column: str) -> float:
    """SE of an A - B difference from jackknife_cohort_difference()-style rows: the root of the sum of the
    jackknife variances over cohort A's leave-one-out rows and cohort B's."""
    return float(np.sqrt(sum(jackknife_se(draws.loc[draws["side"] == side, column]) ** 2 for side in ("a", "b"))))


def loo_model_fit(
    trials_by_participant: TrialsByParticipant, reference: np.ndarray, n_conds: int,
    *, aggregation: str = "simple", min_overlap: int = 20, cache_path: Path | None = None,
) -> pd.DataFrame:
    """Per-participant fit to `reference` next to that participant's lower noise-ceiling bound.

    Both scores use the same left-out side (that participant's own plain-mean RDM, as in
    noise_ceiling.loo_reliability): `model_*` scores it against `reference`, `lower_*` against the
    combined RDM of everyone else. A reference at the lower bound explains what a single participant
    shares with the group; above it, more.
    """
    def run() -> pd.DataFrame:
        ids = list(trials_by_participant)
        rows = []
        for pid in ids:
            left, _ = _mean_combine(trials_by_participant[pid], n_conds)
            others = [t for other in ids if other != pid for t in trials_by_participant[other]]
            others_rdm, _ = _combine(others, n_conds, aggregation)
            model_stats = masked_rank_correlation(left, reference, min_overlap=min_overlap, tau_a=True)
            lower_stats = masked_rank_correlation(left, others_rdm, min_overlap=min_overlap, tau_a=True)
            rows.append({
                "participant_id": pid,
                "model_kendall_a": model_stats["kendall_a"], "model_spearman": model_stats["spearman"],
                "lower_kendall_a": lower_stats["kendall_a"], "lower_spearman": lower_stats["spearman"],
                "n_overlap_model": model_stats["n_overlap"], "n_overlap_lower": lower_stats["n_overlap"],
            })
        return pd.DataFrame(rows)

    key = fingerprint(_digest(trials_by_participant), reference, n_conds, aggregation, min_overlap)
    return cached_frame(cache_path, key, run)


def bootstrap_mean_ci(
    values: np.ndarray, *, n_boot: int = 5000, seed: int = 0, level: float = 0.95,
) -> tuple[float, float]:
    """Participant-bootstrap CI of the mean of a per-participant statistic (NaNs dropped)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return percentile_ci(means, level)


# ---------------------------------------------------------------------------
# Are the cohorts equally similar? TOST on pre - post, and multiplicity
# ---------------------------------------------------------------------------


def observed_cohort_fit(
    rdm_a: np.ndarray, rdm_b: np.ndarray, references: dict[str, np.ndarray], mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Full-data `<stat>_a` / `<stat>_b` per reference (index), on `mask`'s pairs if given: the point
    estimates the jackknife CIs of jackknife_cohort_difference() are centred on."""
    def scored(rdm: np.ndarray) -> np.ndarray:
        return rdm if mask is None else np.where(mask, rdm, np.nan)

    rows = {}
    for name, ref in references.items():
        a, b = _stats(scored(rdm_a), ref), _stats(scored(rdm_b), ref)
        rows[name] = {f"{k}_a": a[k] for k in _STAT_KEYS} | {f"{k}_b": b[k] for k in _STAT_KEYS}
    return pd.DataFrame.from_dict(rows, orient="index")


def cohort_difference_equivalence(
    observed: dict[str, pd.DataFrame], jack: pd.DataFrame, n_a: int, n_b: int,
) -> pd.DataFrame:
    """One TOST row per (pair set, reference, statistic) on D = A - B (here pre - post).

    `observed[pairs]` is observed_cohort_fit() for that pair set; `jack` is the matching
    jackknife_cohort_difference() output. The 90% CI is D +/- z * SE with the jackknife SE of the
    difference; bound rule: equivalence_test_se_effect()."""
    rows = []
    for (pairs, model), draws in jack.groupby(["pairs", "model"]):
        for key, label in _STAT_LABEL.items():
            effect = observed[pairs].at[model, f"{key}_a"] - observed[pairs].at[model, f"{key}_b"]
            result = equivalence_test_se_effect(
                effect, difference_se(draws, f"{key}_diff"), n_a=n_a, n_b=n_b, label=f"pre - post, {label}, {model}",
            )
            rows.append({"pairs": pairs, "target": model, "statistic": key, **result})
    return pd.DataFrame(rows)


def bh_within_cohort(tests: pd.DataFrame, *, p_col: str = "p_value") -> pd.DataFrame:
    """`tests` (columns cohort, test, <p_col>) with a `p_fdr` column: Benjamini-Hochberg within each
    cohort's own family, never pooled across cohorts (pre and post are distinct confirmatory claims)."""
    out = tests.copy()
    out["p_fdr"] = np.nan
    for _, index in out.groupby("cohort").groups.items():
        out.loc[index, "p_fdr"] = multipletests(out.loc[index, p_col], method="fdr_bh")[1]
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_TEMPLATE = "plotly_white"


def fig_agreement_with_ci(
    summary: pd.DataFrame, *, targets: Mapping[str, str], y_title: str = "Kendall tau-a",
    ceiling: tuple[str, float] | None = None,
) -> go.Figure:
    """Combined-RDM tau-a per reference (x) and cohort (colour) with its bootstrap CI. `summary`
    has columns cohort, target, tau_a, ci_lo, ci_hi; `ceiling=(target, value)` marks a noise ceiling
    (e.g. the external raters' leave-one-out agreement) on that reference."""
    fig = go.Figure()
    for cohort in COHORTS:
        rows = summary[summary["cohort"] == cohort].set_index("target").loc[list(targets)]
        color = cohort_colors(cohort)["violin_line"]
        fig.add_trace(go.Scatter(
            x=[targets[t] for t in rows.index], y=rows["tau_a"], mode="markers", name=cohort_label(cohort),
            marker={"color": color, "size": 11, "symbol": cohort_symbol(cohort)},
            error_y={"type": "data", "symmetric": False, "array": rows["ci_hi"] - rows["tau_a"],
                     "arrayminus": rows["tau_a"] - rows["ci_lo"], "color": color},
        ))
    if ceiling is not None:
        fig.add_trace(go.Scatter(
            x=[targets[ceiling[0]]], y=[ceiling[1]], mode="markers", name="external leave-one-out ceiling",
            marker={"color": "grey", "symbol": "line-ew", "size": 26, "line": {"width": 3}},
        ))
    fig.update_layout(template=_TEMPLATE, height=460, yaxis_title=y_title)
    return fig


def fig_fit_vs_lower_bound(loo_by_cohort: dict[str, pd.DataFrame]) -> go.Figure:
    """Per-participant tau-a against the reference next to the same participants' lower noise bound
    (loo_model_fit() output per cohort), one violin pair per cohort."""
    fig = go.Figure()
    for cohort in COHORTS:
        data = loo_by_cohort[cohort].dropna(subset=["model_kendall_a", "lower_kendall_a"])
        color = cohort_colors(cohort)["violin_line"]
        for label, column, opacity in (("reference", "model_kendall_a", 0.85), ("lower bound", "lower_kendall_a", 0.35)):
            fig.add_trace(go.Violin(
                y=data[column], x=[f"{cohort_label(cohort)}: {label}"] * len(data), name=label, box_visible=True,
                points="all", line_color=color, opacity=opacity, showlegend=False,
            ))
    fig.update_layout(template=_TEMPLATE, height=460, yaxis_title="per-participant Kendall tau-a")
    return fig
