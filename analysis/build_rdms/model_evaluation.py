"""
Model-vs-perceptual RDM evaluation for sparse SpAM data (Q1, analysis 1a): does a fully-observed
model RDM (e.g. D_sem_km) rank-correlate with the behavioral RDM beyond chance, how stable is that
correlation across participants, and how does it compare to the noise ceiling?

Why not rsatoolbox.inference.eval_bootstrap_rdm(): each participant's own RDM covers ~1.3% of
pairs (disjoint per-trial image sets), and rsatoolbox's compare() demands identical NaN positions
in both RDMs, so it raises on this data. The same quantities are computed here directly:

  label_shuffle_test_masked -- image-relabeling permutation null (the repo's valid-p-value
      convention, see hierarchy_comparison.label_shuffle_test) restricted to the pairs the
      perceptual RDM observed. hierarchy_comparison's version assumes a complete vector.
  bootstrap_combined_fit -- participant bootstrap of the *combined* RDM's model correlation.
  loo_model_fit -- Nili et al. (2014) style: each left-out participant's own RDM scored against
      the model, next to the same participant's lower-bound score against the combined others.

Tau-a is the primary statistic (the KM RDM is heavily tied), Spearman is reported alongside.

Usage (from repo root):
    from analysis.build_rdms import model_evaluation as me
    tbp, n = me.load_trials_by_participant("pre_shine")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from analysis.behavioral.noise_ceiling import _DEFAULT_CACHE_DIR, _parse_trials_by_participant
from analysis.build_rdms.behavioral import _combine, _mean_combine, _select_participants
from analysis.build_rdms.common import image_manifest_index
from analysis.utils.load_data import load_data
from analysis.utils.stats import masked_rank_correlation

TrialsByParticipant = dict[str, list[tuple[np.ndarray, np.ndarray]]]

_STAT_KEYS = ("kendall_a", "spearman")


def load_trials_by_participant(
    variant: str, *, cache_dir: str | Path = _DEFAULT_CACHE_DIR,
) -> tuple[TrialsByParticipant, int]:
    """Parsed non-catch trials per retained participant, plus the image count."""
    data = load_data(cache_dir, prod_only=True)
    participants = _select_participants(data["participants"], variant)
    image_to_idx, n_conds = image_manifest_index()
    trials = _parse_trials_by_participant(
        data["trials"], list(participants["participant_id"]), variant, image_to_idx, n_conds,
    )
    return trials, n_conds


def _stats(perceptual: np.ndarray, model: np.ndarray) -> dict[str, float]:
    result = masked_rank_correlation(perceptual, model, tau_a=True)
    return {"kendall_a": result["kendall_a"], "spearman": result["spearman"], "n_overlap": result["n_overlap"]}


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
    perceptual: np.ndarray, model: np.ndarray, *, n_perm: int = 1000, seed: int = 0,
) -> MaskedShuffleResult:
    """One-sided image-relabeling permutation test of the perceptual-vs-model rank correlation.

    Image identity of the fully-observed `model` RDM is permuted (rows and columns together, so
    its internal dependence structure survives); the statistic is recomputed over the pairs the
    perceptual RDM observed. p = (#null >= observed + 1) / (n_perm + 1).
    """
    rng = np.random.default_rng(seed)
    n = squareform(model, checks=False).shape[0]
    observed_pairs = ~np.isnan(perceptual)
    perceptual_obs = perceptual[observed_pairs]
    square = squareform(model)
    null = {k: np.empty(n_perm) for k in _STAT_KEYS}
    for i in range(n_perm):
        p = rng.permutation(n)
        permuted = squareform(square[np.ix_(p, p)], checks=False)[observed_pairs]
        stats = _stats(perceptual_obs, permuted)
        for k in _STAT_KEYS:
            null[k][i] = stats[k]
    observed = _stats(perceptual, model)
    p_value = {k: (float(np.sum(null[k] >= observed[k])) + 1.0) / (n_perm + 1.0) for k in _STAT_KEYS}
    return MaskedShuffleResult(observed, null, p_value)


def bootstrap_combined_fit(
    trials_by_participant: TrialsByParticipant, model: np.ndarray, n_conds: int,
    *, aggregation: str = "simple", n_boot: int = 500, seed: int = 0,
) -> pd.DataFrame:
    """Participant bootstrap of the combined RDM's model correlation.

    Each draw resamples participants with replacement, re-combines their pooled trials with
    `aggregation`, and scores the result against `model`. One row per draw; percentile CIs are
    left to the caller (see percentile_ci()).
    """
    rng = np.random.default_rng(seed)
    ids = list(trials_by_participant)
    rows = []
    for b in range(n_boot):
        drawn = rng.choice(len(ids), size=len(ids), replace=True)
        trials = [t for i in drawn for t in trials_by_participant[ids[i]]]
        combined, _ = _combine(trials, n_conds, aggregation)
        rows.append({"boot": b, **_stats(combined, model)})
    return pd.DataFrame(rows)


def loo_model_fit(
    trials_by_participant: TrialsByParticipant, model: np.ndarray, n_conds: int,
    *, aggregation: str = "simple", min_overlap: int = 20,
) -> pd.DataFrame:
    """Per-participant model fit next to that participant's lower noise-ceiling bound.

    Both scores use the same left-out side (that participant's own plain-mean RDM, as in
    noise_ceiling.loo_reliability): `model_*` scores it against `model`, `lower_*` against the
    combined RDM of everyone else. A model at the lower bound explains what a single participant
    shares with the group; above it, more.
    """
    ids = list(trials_by_participant)
    rows = []
    for pid in ids:
        left, _ = _mean_combine(trials_by_participant[pid], n_conds)
        others = [t for other in ids if other != pid for t in trials_by_participant[other]]
        others_rdm, _ = _combine(others, n_conds, aggregation)
        model_stats = masked_rank_correlation(left, model, min_overlap=min_overlap, tau_a=True)
        lower_stats = masked_rank_correlation(left, others_rdm, min_overlap=min_overlap, tau_a=True)
        rows.append({
            "participant_id": pid,
            "model_kendall_a": model_stats["kendall_a"], "model_spearman": model_stats["spearman"],
            "lower_kendall_a": lower_stats["kendall_a"], "lower_spearman": lower_stats["spearman"],
            "n_overlap_model": model_stats["n_overlap"], "n_overlap_lower": lower_stats["n_overlap"],
        })
    return pd.DataFrame(rows)


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
