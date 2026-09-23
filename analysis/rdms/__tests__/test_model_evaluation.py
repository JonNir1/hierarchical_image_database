"""Tests for analysis.rdms.model_evaluation on small synthetic SpAM-like data."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from analysis.rdms import model_evaluation as me
from analysis.rdms.behavioral import _condensed_index

_N = 16
_GROUP = np.repeat(np.arange(4), 4)


def _model_rdm() -> np.ndarray:
    within = _GROUP[:, None] == _GROUP[None, :]
    return squareform(np.where(within, 1.0, 3.0) * (1 - np.eye(_N)), checks=False)


def _make_trials(rng: np.random.Generator, n_participants: int, noise: float) -> me.TrialsByParticipant:
    model = squareform(_model_rdm())
    out: me.TrialsByParticipant = {}
    for p in range(n_participants):
        trials = []
        for _ in range(6):
            members = np.sort(rng.choice(_N, size=6, replace=False))
            i, j = np.triu_indices(len(members), k=1)
            a, b = members[i], members[j]
            idx = _condensed_index(a, b, _N)
            values = model[a, b] + rng.normal(scale=noise, size=len(a))
            trials.append((idx, values))
        out[f"p{p}"] = trials
    return out


def test_label_shuffle_masked_detects_planted_structure():
    rng = np.random.default_rng(0)
    trials = _make_trials(rng, 20, noise=0.3)
    from analysis.rdms.behavioral import _combine
    combined, _ = _combine([t for ts in trials.values() for t in ts], _N, "mean")
    result = me.label_shuffle_test_masked(combined, _model_rdm(), n_perm=200, seed=1)
    assert result.observed["kendall_a"] > 0.3
    assert result.p_value["kendall_a"] < 0.05
    assert abs(result.null["kendall_a"].mean()) < 0.1


def test_label_shuffle_masked_null_is_centered_for_unrelated_rdm():
    rng = np.random.default_rng(2)
    unrelated = rng.normal(size=_N * (_N - 1) // 2)
    unrelated[rng.random(unrelated.size) < 0.3] = np.nan
    result = me.label_shuffle_test_masked(unrelated, _model_rdm(), n_perm=200, seed=3)
    assert abs(result.null["kendall_a"].mean()) < 0.05
    assert result.p_value["kendall_a"] > 0.05


def test_loo_model_fit_returns_one_row_per_participant_and_positive_fit():
    rng = np.random.default_rng(4)
    trials = _make_trials(rng, 12, noise=0.3)
    df = me.loo_model_fit(trials, _model_rdm(), _N, aggregation="mean", min_overlap=5)
    assert len(df) == 12
    assert df["model_kendall_a"].mean() > 0.2
    assert df["lower_kendall_a"].mean() > 0.2


def test_bootstrap_combined_fit_shape_and_range():
    rng = np.random.default_rng(5)
    trials = _make_trials(rng, 10, noise=0.3)
    boots = me.bootstrap_combined_fit(trials, _model_rdm(), _N, aggregation="mean", n_boot=25, seed=6)
    assert len(boots) == 25
    lo, hi = me.percentile_ci(boots["kendall_a"].to_numpy())
    assert lo <= hi
    assert boots["kendall_a"].mean() > 0.2


def test_bootstrap_mean_ci_brackets_the_mean():
    values = np.random.default_rng(7).normal(loc=0.5, scale=0.1, size=40)
    lo, hi = me.bootstrap_mean_ci(values, n_boot=500, seed=8)
    assert lo < values.mean() < hi


def test_percentile_ci_ignores_nans():
    lo, hi = me.percentile_ci(np.array([1.0, 2.0, 3.0, np.nan, 4.0]))
    assert lo == pytest.approx(1.075)
    assert hi == pytest.approx(3.925)
