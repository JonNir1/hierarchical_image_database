"""
General statistical comparison helpers shared across analysis/ subdirectories:
Pearson r, Spearman rho (vector or row-wise) and Kendall tau-a, masked rank correlation between two (possibly incomplete) same-length vectors,
percentile CIs, delete-one jackknife SEs and CIs, an image bootstrap over k RDMs, rank-biserial effect size, independent-samples (Mann-Whitney U) and paired-samples (Wilcoxon) effect
sizes, an image-relabeling permutation test for RDM agreement, and a one-line significance-summary printer. None of this is specific
to RDMs or MDS -- see analysis/utils/mds.py for that layer.

Usage (from repo root):
    from analysis.utils.stats import (
        pearson_r, spearman_rho, kendall_tau_a, masked_rank_correlation, percentile_ci, image_bootstrap,
        mwu_effect_size, mwu_family, wilcoxon_effect_size, print_comparison_line, label_shuffle_test,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, mannwhitneyu, norm, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

from analysis.utils.disk_cache import cached_array, fingerprint


def _unit(x: np.ndarray, axis: int) -> np.ndarray:
    """`x` centred along `axis` and scaled to unit length (the dot product of two is their Pearson r)."""
    centred = np.asarray(x, dtype=np.float64)
    centred = centred - centred.mean(axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return centred / np.sqrt((centred ** 2).sum(axis=axis, keepdims=True))


def pearson_r(a: np.ndarray, b: np.ndarray, *, axis: int = -1) -> float | np.ndarray:
    """Pearson correlation of `a` and `b` along `axis` (a float for 1-D input, one value per row
    otherwise). NaN where either input is constant along the axis."""
    r = np.clip((_unit(a, axis) * _unit(b, axis)).sum(axis=axis), -1.0, 1.0)
    return float(r) if np.ndim(r) == 0 else r


def z_ranks(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Z-scored (tie-averaged) ranks along `axis`, so the mean of two such vectors' product is their
    Spearman rho."""
    ranks = rankdata(x, axis=axis)
    return _unit(ranks, axis) * np.sqrt(ranks.shape[axis])


def spearman_rho(a: np.ndarray, b: np.ndarray, *, axis: int = -1) -> float | np.ndarray:
    """Spearman rank correlation of `a` and `b` along `axis`: Pearson r of the tie-averaged ranks
    (so, like scipy.stats.spearmanr, ties are handled but not discounted). Use `axis=1` on two
    (N, M) matrices to get one rho per row."""
    return pearson_r(rankdata(a, axis=axis), rankdata(b, axis=axis), axis=axis)


def _tie_pairs(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    return float(np.sum(counts * (counts - 1) / 2))


def kendall_tau_a(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's tau-a, (C - D) / (n(n-1)/2), recovered exactly from scipy's tie-corrected
    tau-b: tau_b = (C - D) / sqrt((n0 - n1)(n0 - n2)) with n0 = total pairs and n1/n2 = tied pairs
    in each vector. Unlike tau-b, ties are not removed from the denominator, which is the variant
    Nili et al. (2014) recommend when the model RDM has many tied ranks."""
    n = len(a)
    if n < 2:
        return np.nan
    n0 = n * (n - 1) / 2
    tau_b = kendalltau(a, b).statistic
    return float(tau_b * np.sqrt((n0 - _tie_pairs(a)) * (n0 - _tie_pairs(b))) / n0)


def masked_rank_correlation(
    a: np.ndarray, b: np.ndarray, *, min_overlap: int | None = None, tau_a: bool = False,
) -> dict:
    """Kendall tau + Spearman rho between two (possibly incomplete) same-length
    vectors, restricted to positions non-NaN in both. Unifies
    analysis/s0_build_rdms/behavior/noise_ceiling.py's former _kendall_spearman(),
    analysis/s1_pre_vs_post_shine/demographics_and_behavior.py's former masked_kendall_spearman(),
    and the correlation analysis/s0_build_rdms/validate_rdms.py's
    report_behav_correlation() does inline.

    If `min_overlap` is given, an `"underpowered"` bool flag is added (True,
    with kendall/spearman set to NaN, when fewer than `min_overlap` positions
    are jointly non-NaN) -- otherwise `"underpowered"` is omitted entirely.

    If `tau_a` is True, a `"kendall_a"` entry (see kendall_tau_a()) is added; `"kendall"`
    stays tau-b either way.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    n = int(mask.sum())
    if min_overlap is not None and n < min_overlap:
        under = {"kendall": np.nan, "spearman": np.nan, "n_overlap": n, "underpowered": True}
        return {**under, "kendall_a": np.nan} if tau_a else under
    a_masked, b_masked = a[mask], b[mask]
    tau = float(kendalltau(a_masked, b_masked).statistic)
    rho = spearman_rho(a_masked, b_masked)
    result = {"kendall": tau, "spearman": rho, "n_overlap": n}
    if tau_a:
        result["kendall_a"] = kendall_tau_a(a_masked, b_masked)
    if min_overlap is not None:
        result["underpowered"] = False
    return result


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile confidence interval of a bootstrap distribution, NaNs ignored."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    tail = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(finite, tail)), float(np.percentile(finite, 100.0 - tail))


def jackknife_se(values: np.ndarray) -> float:
    """Delete-one jackknife standard error from the n leave-one-out values of a statistic:
    sqrt((n - 1) / n * sum((v_i - mean v)^2)). NaNs ignored."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.sqrt((len(v) - 1) / len(v) * np.sum((v - v.mean()) ** 2)))


def jackknife_bias(estimate: float, values: np.ndarray) -> float:
    """Delete-one jackknife bias estimate, (n - 1) * (mean of the leave-one-out values - estimate). NaNs ignored.

    For a statistic that grows with sample size (e.g. the agreement of a group RDM pooled over participants)
    this is large and says how far the estimate is from an infinitely large sample's value, not that the
    jackknife CI is off-centre: it is reported as a diagnostic, never subtracted."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float((len(v) - 1) * (v.mean() - estimate))


def jackknife_ci(estimate: float, values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Normal confidence interval around `estimate` with the jackknife standard error of `values`."""
    half = float(norm.ppf(0.5 + level / 2.0)) * jackknife_se(values)
    return float(estimate - half), float(estimate + half)


def image_bootstrap(
    rdms: Sequence[np.ndarray], statistic: Callable[..., Sequence[float]], *, n_boot: int = 1000,
    seed: int = 0, cache_path: Path | None = None, extra_key: tuple = (),
) -> np.ndarray:
    """Image-bootstrap draws of `statistic` over k condensed RDMs of the same images, shape (n_boot, n_stats).

    Each draw resamples the images with replacement and calls `statistic(*resampled)` on the pairs of the
    resampled set, one condensed-like vector per input RDM. A pair of an image with its own duplicate is
    dropped, so duplicates add repeated pairs but no artificial agreement. NaNs (unobserved pairs) are
    passed through, so `statistic` must mask them itself if any input can be incomplete. `cache_path`
    stores the draws; `extra_key` separates caches of different statistics over the same RDMs (the empty
    default keeps the fingerprint of caches written before this function existed).
    """
    def draw() -> np.ndarray:
        squares = [squareform(r, checks=False) for r in rdms]
        n = len(squares[0])
        iu = np.triu_indices(n, 1)
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            i, j = idx[iu[0]], idx[iu[1]]
            keep = i != j
            rows.append(np.atleast_1d(statistic(*(sq[i[keep], j[keep]] for sq in squares))))
        return np.asarray(rows, dtype=np.float64)

    key = fingerprint(*rdms, n_boot, seed, *extra_key)
    return cached_array(cache_path, key, draw, key="draws")


def rank_biserial(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Rank-biserial effect size r = 1 - 2U/(n_A*n_B) between two independent
    samples (the effect size mwu_effect_size() reports), without the p-value:
    positive when A tends to be smaller than B. Ranks are averaged over ties,
    matching scipy's Mann-Whitney U."""
    n_a, n_b = len(values_a), len(values_b)
    ranks = rankdata(np.concatenate([values_a, values_b]))
    u_a = ranks[:n_a].sum() - n_a * (n_a + 1) / 2
    return float(1 - 2 * u_a / (n_a * n_b))


def mwu_effect_size(values_a, values_b, *, label_a: str = "A", label_b: str = "B") -> dict:
    """Two-sided Mann-Whitney U between two independent samples, with
    rank-biserial effect size r = 1 - 2U/(n_A*n_B). Unifies
    analysis/s1_pre_vs_post_shine/demographics_and_behavior.py's former mwu_two_group() and
    analysis/_retired/pilot/analysis.ipynb's former inline mwu_compare()."""
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


def mwu_family(
    df: pd.DataFrame, value_col: str, group_col: str,
    *, correction: str = "bonferroni", alpha: float = 0.05,
    label_fn: Callable[[object], str] = str,
) -> pd.DataFrame:
    """Pairwise Mann-Whitney U for every pair of groups in `group_col`, with a
    multiple-comparison correction applied across the family. `correction`
    accepts any method string supported by
    statsmodels.stats.multitest.multipletests (e.g. "bonferroni", "holm",
    "fdr_bh"). Promotes analysis/_retired/pilot/analysis.ipynb's former inline
    mwu_family() to reusable code, built on mwu_effect_size(). `label_fn`
    formats each group value for the "comparison" column/index (default
    str(); pilot's original notebook used f"v{v:g}" for float task versions --
    pass that as label_fn to reproduce it)."""
    groups = sorted(df[group_col].dropna().unique())
    rows = []
    for a, b in combinations(groups, 2):
        values_a = df.loc[df[group_col] == a, value_col].dropna().astype(float).values
        values_b = df.loc[df[group_col] == b, value_col].dropna().astype(float).values
        result = mwu_effect_size(values_a, values_b, label_a=label_fn(a), label_b=label_fn(b))
        result["comparison"] = f"{label_fn(a)} vs {label_fn(b)}"
        rows.append(result)
    result_df = pd.DataFrame(rows).set_index("comparison")
    _, p_corrected, _, _ = multipletests(result_df["p"], alpha=alpha, method=correction)
    result_df["p_corrected"] = p_corrected
    return result_df[[
        "n (A)", "n (B)", "median (A)", "median (B)", "mean (A)", "mean (B)",
        "std (A)", "std (B)", "U", "p_corrected", "p", "r",
    ]]


def wilcoxon_effect_size(x: np.ndarray, y: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test + matched-pairs rank-biserial effect
    size (R+ - R-) / sum(ranks). Extracts
    analysis/s0_build_rdms/behavior/aggregation_comparison.ipynb's former inline
    _wilcoxon_row()."""
    diff = np.asarray(x) - np.asarray(y)
    diff_nz = diff[diff != 0]
    n = len(diff_nz)
    if n == 0:
        return {"n": 0, "statistic": np.nan, "p_value": np.nan, "effect_size": np.nan}
    stat, p = wilcoxon(x, y)
    ranks = pd.Series(np.abs(diff_nz)).rank().to_numpy()
    r_plus = ranks[diff_nz > 0].sum()
    r_minus = ranks[diff_nz < 0].sum()
    effect_size = (r_plus - r_minus) / ranks.sum()
    return {"n": n, "statistic": float(stat), "p_value": float(p), "effect_size": float(effect_size)}


def print_comparison_line(result: dict, *, alpha: float = 0.05, corrected: bool = False) -> None:
    """One-line significance summary for an mwu_effect_size()/mwu_family()-row
    result dict. `corrected` reads "p_corrected" instead of "p" (use True for
    an mwu_family() row). Unifies analysis/_retired/pilot/analysis.ipynb's former
    print_bottom_line() and analysis/s1_pre_vs_post_shine/1_demographics_and_behavior.ipynb's
    former _print_mwu()."""
    p_key = "p_corrected" if corrected else "p"
    p = result[p_key]
    sig = "significant" if p < alpha else "not significant"
    print(f"  {result['comparison']}: U={result['U']:.1f}  {p_key}={p:.4f} ({sig})  r={result['r']:.3f}")


# ---------------------------------------------------------------------------
# Image-relabeling permutation test for RDM agreement
# ---------------------------------------------------------------------------

_RDM_STATISTICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "kendall_a": kendall_tau_a,
    "spearman": spearman_rho,
}
_DEFAULT_STATISTICS = tuple(_RDM_STATISTICS)


@dataclass
class ShuffleResult:
    """Observed statistics, their permutation nulls and one-sided p-values, keyed by statistic name."""
    observed: dict[str, float]
    null: dict[str, np.ndarray]
    p_value: dict[str, float]

    def summary(self) -> pd.DataFrame:
        rows = [
            {"statistic": k, "observed": self.observed[k], "null_mean": float(self.null[k].mean()),
             "null_sd": float(self.null[k].std()), "p_value": self.p_value[k]}
            for k in self.observed
        ]
        return pd.DataFrame(rows).set_index("statistic")


def _rdm_statistic(name: str, a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    return _RDM_STATISTICS[name](a[mask], b[mask])


def label_shuffle_test(
    perceptual: np.ndarray, reference: np.ndarray, *, statistics: tuple[str, ...] = _DEFAULT_STATISTICS,
    n_perm: int = 1000, seed: int = 0, cache_path: Path | None = None,
) -> ShuffleResult:
    """One-sided image-relabeling permutation test of the rank agreement between two condensed RDMs.

    Image identity of the fully observed `reference` RDM is permuted (rows and columns together, so its
    internal dependence structure survives, which an entry-wise shuffle would not); each statistic
    ("kendall_a", "spearman") is recomputed over the pairs `perceptual` observed (NaN = unobserved, so
    incomplete RDMs are supported and complete ones need no special case). p = (#null >= observed + 1)
    / (n_perm + 1), so its floor is 1 / (n_perm + 1). `cache_path` stores the null draws.
    """
    n = squareform(reference, checks=False).shape[0]
    observed_pairs = ~np.isnan(perceptual)
    perceptual_obs = perceptual[observed_pairs]

    def draw_null() -> np.ndarray:
        rng = np.random.default_rng(seed)
        square = squareform(reference)
        draws = np.empty((n_perm, len(statistics)))
        for i in range(n_perm):
            p = rng.permutation(n)
            permuted = squareform(square[np.ix_(p, p)], checks=False)[observed_pairs]
            draws[i] = [_RDM_STATISTICS[k](perceptual_obs, permuted) for k in statistics]
        return draws

    # the default statistics keep the fingerprint caches written before `statistics` was a parameter used
    key_parts = (perceptual, reference, n_perm, seed) + (() if statistics == _DEFAULT_STATISTICS else (statistics,))
    draws = cached_array(cache_path, fingerprint(*key_parts), draw_null)
    null = {k: draws[:, j] for j, k in enumerate(statistics)}
    observed = {k: _rdm_statistic(k, perceptual, reference) for k in statistics}
    p_value = {k: (float(np.sum(null[k] >= observed[k])) + 1.0) / (n_perm + 1.0) for k in statistics}
    return ShuffleResult(observed, null, p_value)
