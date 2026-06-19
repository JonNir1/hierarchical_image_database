"""Unit tests for metrics.coverage and metrics.spearman_correlation."""
from datetime import datetime

import numpy as np
import pytest
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import connected_components

from SpAM_Simulations.experiment import ExperimentResults
from SpAM_Simulations import metrics


def _results(num_obs_condensed, distances_condensed=None):
    if distances_condensed is None:
        distances_condensed = num_obs_condensed.astype(np.float32)
    return ExperimentResults(datetime.now(), distances_condensed, num_obs_condensed, np.array([0.0]))


def _reference_coverage(num_obs):
    """Independent re-implementation of the pre-refactor coverage logic."""
    n = num_obs.astype(float)
    num_pairs = n.shape[0]
    sq = squareform(n, checks=False)
    n_img_obs = np.sum(sq > 0, axis=0)
    adj = squareform(n > 0, checks=False).astype(int)
    ncomp, _ = connected_components(adj, directed=False)
    return {
        "num_images": sq.shape[0],
        "average_img_obs": np.mean(n_img_obs),
        "img_coverage": np.mean(n_img_obs > 0) * 100,
        "num_pairs": num_pairs,
        "average_pair_obs": np.mean(n),
        "pair_coverage": np.mean(n > 0) * 100,
        "num_connected_components": ncomp,
    }


@pytest.mark.parametrize("seed", range(5))
def test_coverage_matches_reference(seed):
    rng = np.random.default_rng(seed)
    N = 12
    num_obs = rng.integers(0, 4, size=N * (N - 1) // 2).astype(np.int8)
    cov = metrics.coverage(_results(num_obs))
    ref = _reference_coverage(num_obs)
    assert cov.keys() == ref.keys()
    for k in ref:
        assert np.isclose(cov[k], ref[k]), f"{k}: {cov[k]} != {ref[k]}"


def test_coverage_fully_connected_single_component():
    N = 6
    num_obs = np.ones(N * (N - 1) // 2, dtype=np.int8)  # every pair observed
    cov = metrics.coverage(_results(num_obs))
    assert cov["num_images"] == N
    assert cov["pair_coverage"] == 100.0
    assert cov["img_coverage"] == 100.0
    assert cov["num_connected_components"] == 1


def test_spearman_correlation_perfect_and_threshold():
    # two experiments with identical mean distances on overlapping (>=2 obs) pairs
    N = 8
    L = N * (N - 1) // 2
    dist = np.arange(1, L + 1, dtype=np.float32)
    obs = np.full(L, 2, dtype=np.int8)  # 2 observations each -> all valid
    e1 = _results(obs, dist * 2)  # mean = dist
    e2 = _results(obs, dist * 2)
    assert metrics.spearman_correlation(e1, e2) == pytest.approx(1.0)


def test_spearman_raises_on_insufficient_overlap():
    L = 6
    obs = np.zeros(L, dtype=np.int8)  # nothing observed -> all NaN means
    e = _results(obs, np.zeros(L, dtype=np.float32))
    with pytest.raises(ValueError):
        metrics.spearman_correlation(e, e)
