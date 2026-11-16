"""
Pre- vs post-SHINE cohort comparison, RDM-level agreement with a reference RDM (analysis/
pre_vs_post_shine/4_semantic_hierarchy.ipynb and 5_ground_truth.ipynb): does each cohort's behavioral
RDM rank-correlate with a reference (the Kiani directory-tree RDM, or an external ratings/recordings
RDM on a shared image subset) beyond chance, how stable is that across participants, and are the two
cohorts equally similar to it?

Why not rsatoolbox.inference.eval_bootstrap_rdm(): each participant's own RDM covers ~1.3% of pairs
(disjoint per-trial image sets), and rsatoolbox's compare() demands identical NaN positions in both
RDMs, so it raises on this data. The same quantities are computed here directly. Kendall tau-a is the
primary statistic (the reference RDMs are heavily tied), Spearman rho the secondary one.

  label_shuffle_test_masked -- image-relabeling permutation null restricted to the pairs the
      behavioral RDM observed (the repo's valid-p-value convention for RDM correlations).
  bootstrap_combined_fit_multi -- participant bootstrap of the combined RDM's fit to several references.
  loo_model_fit -- Nili et al. (2014) style: each left-out participant's own RDM scored against the
      reference, next to the same participant's lower-bound score against the combined others.
  bootstrap_cohort_difference + cohort_difference_equivalence -- pre minus post fit, with a TOST
      (analysis.utils.equivalence.equivalence_test_bootstrap_effect, the project's 80%-power bound, closed form).
  bh_within_cohort -- Benjamini-Hochberg within each cohort, never pooled across cohorts.

Every expensive call takes `cache_path` (analysis.utils.disk_cache): results are reused only under a
matching fingerprint of the inputs, n and seed.

Usage (from repo root):
    from analysis.pre_vs_post_shine.reference_agreement import (
        label_shuffle_test_masked, bootstrap_combined_fit_multi, loo_model_fit,
        bootstrap_cohort_difference, cohort_difference_equivalence, bh_within_cohort,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

from analysis.build_rdms.behavior.spam_ratings import _combine, _mean_combine
from analysis.build_rdms.common import REPO_ROOT, subset_condensed
from analysis.pre_vs_post_shine.demographics_and_behavior import COHORTS, cohort_colors, cohort_label
from analysis.utils.disk_cache import cached_array, cached_frame, fingerprint
from analysis.utils.equivalence import equivalence_test_bootstrap_effect
from analysis.utils.stats import masked_rank_correlation

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
# Permutation test
# ---------------------------------------------------------------------------


@dataclass
class MaskedShuffleResult:
    observed: dict[str, float]
    null: dict[str, np.ndarray]
    p_value: dict[str, float]

    def summary(self) -> pd.DataFrame:
        rows = [
            {"statistic": k, "observed": self.observed[k], "null_mean": float(self.null[k].mean()),
             "null_sd": float(self.null[k].std()), "p_value": self.p_value[k]}
            for k in _STAT_KEYS
        ]
        return pd.DataFrame(rows).set_index("statistic")


def label_shuffle_test_masked(
    perceptual: np.ndarray, reference: np.ndarray, *, n_perm: int = 1000, seed: int = 0,
    cache_path: Path | None = None,
) -> MaskedShuffleResult:
    """One-sided image-relabeling permutation test of the perceptual-vs-reference rank correlation.

    Image identity of the fully-observed `reference` RDM is permuted (rows and columns together, so
    its internal dependence structure survives); the statistic is recomputed over the pairs the
    perceptual RDM observed. p = (#null >= observed + 1) / (n_perm + 1).
    """
    n = squareform(reference, checks=False).shape[0]
    observed_pairs = ~np.isnan(perceptual)
    perceptual_obs = perceptual[observed_pairs]

    def draw_null() -> np.ndarray:
        rng = np.random.default_rng(seed)
        square = squareform(reference)
        draws = np.empty((n_perm, len(_STAT_KEYS)))
        for i in range(n_perm):
            p = rng.permutation(n)
            stats = _stats(perceptual_obs, squareform(square[np.ix_(p, p)], checks=False)[observed_pairs])
            draws[i] = [stats[k] for k in _STAT_KEYS]
        return draws

    draws = cached_array(cache_path, fingerprint(perceptual, reference, n_perm, seed), draw_null)
    null = {k: draws[:, j] for j, k in enumerate(_STAT_KEYS)}
    observed = _stats(perceptual, reference)
    p_value = {k: (float(np.sum(null[k] >= observed[k])) + 1.0) / (n_perm + 1.0) for k in _STAT_KEYS}
    return MaskedShuffleResult(observed, null, p_value)


# ---------------------------------------------------------------------------
# Participant bootstraps and per-participant fit
# ---------------------------------------------------------------------------


def bootstrap_combined_fit_multi(
    trials_by_participant: TrialsByParticipant, references: dict[str, np.ndarray], n_conds: int,
    *, aggregation: str = "simple", n_boot: int = 500, seed: int = 0, subset: np.ndarray | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Participant bootstrap of the combined RDM's correlation with several reference RDMs at once.

    Each draw resamples participants with replacement and re-combines their pooled trials with
    `aggregation` once, then scores that combined RDM against every reference, so the (expensive)
    combination is shared. One row per (draw, reference); percentile CIs via percentile_ci(). With
    `subset` (manifest row indices) the combined RDM is restricted to those images before scoring, so
    every reference must be that subset's RDM.
    """
    def run() -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        ids = list(trials_by_participant)
        rows = []
        for b in range(n_boot):
            drawn = rng.choice(len(ids), size=len(ids), replace=True)
            combined, _ = _combine([t for i in drawn for t in trials_by_participant[ids[i]]], n_conds, aggregation)
            if subset is not None:
                combined = subset_condensed(combined, subset)
            rows += [{"boot": b, "model": name, **_stats(combined, ref)} for name, ref in references.items()]
        return pd.DataFrame(rows)

    key = fingerprint(_digest(trials_by_participant), list(references), *references.values(), n_conds,
                      aggregation, n_boot, seed, subset)
    return cached_frame(cache_path, key, run)


def bootstrap_cohort_difference(
    trials_a: TrialsByParticipant, trials_b: TrialsByParticipant, references: dict[str, np.ndarray], n_conds: int,
    *, aggregation: str = "simple", n_boot: int = 1000, seed: int = 0, subset: np.ndarray | None = None,
    pair_masks: dict[str, np.ndarray | None] | None = None, cache_path: Path | None = None,
) -> pd.DataFrame:
    """Bootstrap of (cohort A - cohort B) in each cohort's combined-RDM correlation with each reference.

    The cohorts are disjoint participants, so each is resampled on its own; per draw both are
    re-combined with `aggregation` once and restricted to `subset`. `pair_masks` names the pair sets
    to score on (default {"own": None}, each cohort on the pairs it observed); a boolean mask over the
    subset's condensed pairs restricts both cohorts to those pairs, removing any difference in which
    pairs they happened to observe. One row per (draw, pair set, reference) with `<stat>_a`, `<stat>_b`,
    `<stat>_diff` and the overlap counts.
    """
    masks = {"own": None} if pair_masks is None else pair_masks

    def run() -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        ids_a, ids_b = list(trials_a), list(trials_b)

        def combined_rdm(trials: TrialsByParticipant, ids: list[str]) -> np.ndarray:
            drawn = rng.choice(len(ids), size=len(ids), replace=True)
            combined, _ = _combine([t for i in drawn for t in trials[ids[i]]], n_conds, aggregation)
            return subset_condensed(combined, subset) if subset is not None else combined

        rows = []
        for b in range(n_boot):
            rdm_a, rdm_b = combined_rdm(trials_a, ids_a), combined_rdm(trials_b, ids_b)
            for pairs, mask in masks.items():
                scored_a = rdm_a if mask is None else np.where(mask, rdm_a, np.nan)
                scored_b = rdm_b if mask is None else np.where(mask, rdm_b, np.nan)
                for name, ref in references.items():
                    stats_a, stats_b = _stats(scored_a, ref), _stats(scored_b, ref)
                    row = {"boot": b, "pairs": pairs, "model": name,
                           "n_overlap_a": stats_a["n_overlap"], "n_overlap_b": stats_b["n_overlap"]}
                    for key in _STAT_KEYS:
                        row.update({f"{key}_a": stats_a[key], f"{key}_b": stats_b[key],
                                    f"{key}_diff": stats_a[key] - stats_b[key]})
                    rows.append(row)
        return pd.DataFrame(rows)

    key = fingerprint(_digest(trials_a), _digest(trials_b), list(references), *references.values(), n_conds,
                      aggregation, n_boot, seed, subset, list(masks), *masks.values())
    return cached_frame(cache_path, key, run)


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


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile confidence interval of a bootstrap distribution, NaNs ignored."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    tail = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(finite, tail)), float(np.percentile(finite, 100.0 - tail))


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
    estimates the bootstrap draws of bootstrap_cohort_difference() are centred on."""
    def scored(rdm: np.ndarray) -> np.ndarray:
        return rdm if mask is None else np.where(mask, rdm, np.nan)

    rows = {}
    for name, ref in references.items():
        a, b = _stats(scored(rdm_a), ref), _stats(scored(rdm_b), ref)
        rows[name] = {f"{k}_a": a[k] for k in _STAT_KEYS} | {f"{k}_b": b[k] for k in _STAT_KEYS}
    return pd.DataFrame.from_dict(rows, orient="index")


def cohort_difference_equivalence(
    observed: dict[str, pd.DataFrame], boots: pd.DataFrame, n_a: int, n_b: int,
) -> pd.DataFrame:
    """One TOST row per (pair set, reference, statistic) on D = A - B (here pre - post).

    `observed[pairs]` is observed_cohort_fit() for that pair set; `boots` is the matching
    bootstrap_cohort_difference() output. Bound rule: equivalence_test_bootstrap_effect()."""
    rows = []
    for (pairs, model), draws in boots.groupby(["pairs", "model"]):
        for key, label in _STAT_LABEL.items():
            effect = observed[pairs].at[model, f"{key}_a"] - observed[pairs].at[model, f"{key}_b"]
            result = equivalence_test_bootstrap_effect(
                effect, draws[f"{key}_diff"], n_a=n_a, n_b=n_b, label=f"pre - post, {label}, {model}",
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
            marker={"color": color, "size": 11},
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


def fig_bootstrap_distributions(boots_by_cohort: dict[str, pd.DataFrame], column: str, *, x_title: str) -> go.Figure:
    """Overlaid participant-bootstrap histograms of `column`, one per cohort."""
    fig = go.Figure()
    for cohort in COHORTS:
        fig.add_trace(go.Histogram(
            x=boots_by_cohort[cohort][column], name=cohort_label(cohort), nbinsx=30, opacity=0.7,
            marker_color=cohort_colors(cohort)["violin_line"],
        ))
    fig.update_layout(template=_TEMPLATE, barmode="overlay", height=420, xaxis_title=x_title)
    return fig
