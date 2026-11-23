"""Tests for analysis.s1_pre_vs_post_shine.decomposition: standardized NNLS, hierarchical delta R^2 and
the participant bootstrap of the pre vs post comparison."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import nnls

from analysis.s1_pre_vs_post_shine import decomposition as dec

_N_PAIRS = 400


def _z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / a.std()


def _synthetic(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """y = 2*x1 + x2 - x3 + noise: the third true weight is negative, so NNLS must zero it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(_N_PAIRS, 3))
    return 2 * X[:, 0] + X[:, 1] - X[:, 2] + rng.normal(scale=0.5, size=_N_PAIRS), X


def test_nnls_matches_scipy_on_the_observed_pairs_and_zeroes_a_negative_predictor():
    y, X = _synthetic()
    y[::7] = np.nan
    betas, r2 = dec.standardized_nnls(y, X)
    keep = np.isfinite(y)
    expected, residual = nnls(np.column_stack([_z(c[keep]) for c in X.T]), _z(y[keep]))
    assert betas == pytest.approx(expected, abs=1e-6)
    assert betas[2] == 0.0 and (betas >= 0).all()
    assert r2 == pytest.approx(1 - residual ** 2 / keep.sum(), abs=1e-6)


def test_betas_are_invariant_to_rescaling_and_shifting_the_inputs():
    y, X = _synthetic()
    betas, _ = dec.standardized_nnls(y, X)
    rescaled, _ = dec.standardized_nnls(5.0 * y + 3.0, X * np.array([10.0, 0.1, 2.0]) - 1.0)
    assert rescaled == pytest.approx(betas, abs=1e-8)


def test_hierarchical_delta_r2_sums_to_the_full_ols_r2():
    rng = np.random.default_rng(1)
    predictors = {t: rng.normal(size=_N_PAIRS) for t in dec.TERMS}
    y = predictors["pix"] + 0.5 * predictors["clip"] + 0.3 * predictors["sem_km"] + rng.normal(size=_N_PAIRS)
    fit = dec.fit_models(y, predictors)
    rows = fit[fit["model"] == "km+clip"]
    design = np.column_stack([_z(predictors[t]) for t in ("sem_km", "clip", "pix")])
    residual = _z(y) - design @ np.linalg.lstsq(design, _z(y), rcond=None)[0]
    assert rows["delta_r2"].sum() == pytest.approx(1 - residual.var(), abs=1e-8)
    assert fit[fit["model"] == "km+wn"]["delta_r2"].isna().all()


def test_bootstrap_is_seeded_and_its_difference_is_a_minus_b():
    rng = np.random.default_rng(2)
    n_conds, n_pairs = 8, 28
    predictors = {t: rng.normal(size=n_pairs) for t in dec.TERMS}

    def cohort() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
        return {f"p{i}": [(rng.choice(n_pairs, 20, replace=False), rng.random(20)) for _ in range(3)] for i in range(6)}

    args = (cohort(), cohort(), predictors, predictors, n_conds)
    first = dec.bootstrap_decomposition(*args, aggregation="mean", n_boot=5, seed=3)
    again = dec.bootstrap_decomposition(*args, aggregation="mean", n_boot=5, seed=3)
    assert first.equals(again)
    assert first["beta_diff"].to_numpy() == pytest.approx((first["beta_a"] - first["beta_b"]).to_numpy())
    assert first["boot"].nunique() == 5 and set(first["pairs"]) == {"own"}
