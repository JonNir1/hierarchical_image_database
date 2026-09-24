"""
Equivalence testing (TOST, in its confidence-interval form) for "no difference"
claims: a comparison is equivalent when the (1 - 2*alpha) CI of its effect size
lies entirely inside +/- `bound` (two one-sided tests at level alpha, Lakens 2017).

Continuous per-subject outcomes use the rank-biserial effect size (the same
r that analysis.utils.stats.mwu_effect_size() reports) with a bootstrap CI;
proportions (sex, retention) use the risk difference with a Newcombe hybrid
score CI. There is no pre-defined smallest effect size of interest for this
project, so `bound` is always passed explicitly by the caller.

Usage (from repo root):
    from analysis.utils.equivalence import (
        equivalence_test_rank_biserial, equivalence_test_risk_difference,
        equivalence_sensitivity, print_equivalence_line,
    )
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

from analysis.utils.stats import rank_biserial

# Project-chosen equivalence bounds (no pre-defined smallest effect size of
# interest exists). 0.3 rank-biserial is roughly d = 0.45, about what n = 77-79
# per group can distinguish from zero (see equivalence_sensitivity()); 0.15 is
# 15 percentage points for proportions.
RANK_BISERIAL_BOUND = 0.3
RISK_DIFFERENCE_BOUND = 0.15


def _bootstrap_rank_biserial(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Rank-biserial r of `n_boot` bootstrap resamples (each group resampled
    with replacement separately), vectorised over resamples."""
    n_a, n_b = len(a), len(b)
    boot_a = a[rng.integers(0, n_a, size=(n_boot, n_a))]
    boot_b = b[rng.integers(0, n_b, size=(n_boot, n_b))]
    ranks = rankdata(np.concatenate([boot_a, boot_b], axis=1), axis=1)
    u_a = ranks[:, :n_a].sum(axis=1) - n_a * (n_a + 1) / 2
    return 1 - 2 * u_a / (n_a * n_b)


def _rank_biserial_ci(a: np.ndarray, b: np.ndarray, *, ci_level: float, n_boot: int, seed: int) -> tuple[float, float]:
    boot = _bootstrap_rank_biserial(a, b, n_boot, np.random.default_rng(seed))
    tail = (1 - ci_level) / 2
    low, high = np.quantile(boot, [tail, 1 - tail])
    return float(low), float(high)


def equivalence_test_rank_biserial(
    values_a, values_b, *, bound: float, label_a: str = "A", label_b: str = "B",
    alpha: float = 0.05, n_boot: int = 5000, seed: int = 0,
) -> dict:
    """Equivalence test on the rank-biserial effect size between two independent
    samples: bootstrap (subjects resampled within group) (1 - 2*alpha) CI of r,
    equivalent when it lies inside [-bound, +bound]."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    r = rank_biserial(a, b)
    low, high = _rank_biserial_ci(a, b, ci_level=1 - 2 * alpha, n_boot=n_boot, seed=seed)
    return {
        "comparison": f"{label_a} vs {label_b}", "n (A)": len(a), "n (B)": len(b),
        "effect": round(r, 4), "ci_low": round(low, 4), "ci_high": round(high, 4),
        "bound": bound, "equivalent": bool(-bound < low and high < bound),
    }


def _wilson_interval(k: int, n: int, z: float) -> tuple[float, float]:
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def equivalence_test_risk_difference(
    k_a: int, n_a: int, k_b: int, n_b: int, *, bound: float, label_a: str = "A", label_b: str = "B",
    alpha: float = 0.05,
) -> dict:
    """Equivalence test on the risk difference p_A - p_B of two independent
    proportions (k successes out of n): Newcombe hybrid-score (1 - 2*alpha) CI,
    equivalent when it lies inside [-bound, +bound] (bound in proportion units)."""
    z = float(norm.ppf(1 - alpha))
    p_a, p_b = k_a / n_a, k_b / n_b
    low_a, high_a = _wilson_interval(k_a, n_a, z)
    low_b, high_b = _wilson_interval(k_b, n_b, z)
    diff = p_a - p_b
    low = diff - np.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2)
    high = diff + np.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2)
    return {
        "comparison": f"{label_a} vs {label_b}", "n (A)": n_a, "n (B)": n_b,
        "effect": round(float(diff), 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4),
        "bound": bound, "equivalent": bool(-bound < low and high < bound),
    }


def equivalence_sensitivity(
    n_a: int, n_b: int, *, power: float = 0.8, alpha: float = 0.05, n_sim: int = 500, n_boot: int = 500,
    seed: int = 0,
) -> float:
    """Smallest rank-biserial `bound` at which equivalence_test_rank_biserial()
    would declare equivalence with probability `power` when the groups truly do
    not differ, at group sizes (n_a, n_b). The rank-biserial null distribution
    is distribution-free for continuous data, so both groups are simulated as
    standard normal; the answer is the `power` quantile, across simulations, of
    the widest CI limit max(|low|, |high|) -- a bound below that would
    reject equivalence too often to be informative."""
    rng = np.random.default_rng(seed)
    ci_level = 1 - 2 * alpha
    tail = (1 - ci_level) / 2
    widest = np.empty(n_sim)
    for i in range(n_sim):
        a, b = rng.standard_normal(n_a), rng.standard_normal(n_b)
        boot = _bootstrap_rank_biserial(a, b, n_boot, rng)
        low, high = np.quantile(boot, [tail, 1 - tail])
        widest[i] = max(abs(low), abs(high))
    return float(np.quantile(widest, power))


def print_equivalence_line(result: dict) -> None:
    """One-line summary of an equivalence_test_*() result dict."""
    verdict = "equivalent" if result["equivalent"] else "not shown equivalent"
    print(
        f"  {result['comparison']}: effect={result['effect']:.3f}  90% CI [{result['ci_low']:.3f}, "
        f"{result['ci_high']:.3f}]  bound=+/-{result['bound']:g} ({verdict})"
    )
