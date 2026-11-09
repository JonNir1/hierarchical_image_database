"""
General statistical comparison helpers shared across analysis/ subdirectories:
masked rank correlation between two (possibly incomplete) same-length vectors,
rank-biserial effect size, independent-samples (Mann-Whitney U) and paired-samples (Wilcoxon) effect
sizes, and a one-line significance-summary printer. None of this is specific
to RDMs or MDS -- see analysis/build_rdms/mds.py for that layer.

Usage (from repo root):
    from analysis.utils.stats import (
        masked_rank_correlation, mwu_effect_size, mwu_family,
        wilcoxon_effect_size, print_comparison_line,
    )
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, rankdata, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests


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
    analysis/behavioral/noise_ceiling.py's former _kendall_spearman(),
    analysis/behavioral/cohort_comparison.py's former masked_kendall_spearman(),
    and the correlation analysis/build_rdms/validate_rdms.py's
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
    rho = float(spearmanr(a_masked, b_masked).statistic)
    result = {"kendall": tau, "spearman": rho, "n_overlap": n}
    if tau_a:
        result["kendall_a"] = kendall_tau_a(a_masked, b_masked)
    if min_overlap is not None:
        result["underpowered"] = False
    return result


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
    analysis/behavioral/cohort_comparison.py's former mwu_two_group() and
    analysis/pilot/analysis.ipynb's former inline mwu_compare()."""
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
    "fdr_bh"). Promotes analysis/pilot/analysis.ipynb's former inline
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
    analysis/behavioral/aggregation_comparison.ipynb's former inline
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
    an mwu_family() row). Unifies analysis/pilot/analysis.ipynb's former
    print_bottom_line() and analysis/behavioral/cohort_comparison.ipynb's
    former _print_mwu()."""
    p_key = "p_corrected" if corrected else "p"
    p = result[p_key]
    sig = "significant" if p < alpha else "not significant"
    print(f"  {result['comparison']}: U={result['U']:.1f}  {p_key}={p:.4f} ({sig})  r={result['r']:.3f}")
