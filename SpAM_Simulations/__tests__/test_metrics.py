"""Unit tests for metrics.coverage and metrics.spearman_correlation."""
from datetime import datetime

import numpy as np
import pytest
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import connected_components

from SpAM_Simulations.models.experiment import ExperimentResults
from SpAM_Simulations.measures import metrics


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


# --------------------------------------------------------------------- effective_rank
def test_effective_rank_recovers_cube_dimensionality():
    """An isotropic D-cube of points has effective rank ~ D (each axis carries equal variance)."""
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng(0)
    for D in (2, 4, 6):
        pts = rng.normal(size=(300, D))
        er = metrics.effective_rank(pdist(pts))
        assert abs(er - D) < 0.6, f"D={D} -> effective_rank={er}"


def test_effective_rank_low_for_rank_two_configuration():
    """Points confined to a 2-D plane (embedded in 6-D) have effective rank ~ 2, well below 6."""
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng(1)
    flat = np.zeros((200, 6))
    flat[:, :2] = rng.normal(size=(200, 2))
    assert metrics.effective_rank(pdist(flat)) < 2.5


def test_effective_rank_handles_missing_pairs():
    from scipy.spatial.distance import pdist
    d = pdist(np.random.default_rng(2).normal(size=(40, 4)))
    d[::7] = np.nan  # scatter some unobserved pairs
    assert metrics.effective_rank(d) > 1.0  # mean-imputes, still returns a finite rank


# --------------------------------------------------------------------- closest-pair selection
#
# These moved here when the three duplicate implementations (pipeline._topk_similar_jaccard,
# gt_construction._topk_jaccard, recovery.topk_mask) were consolidated into metrics.

class TestTopKMask:
    def test_selects_the_smallest_entries(self):
        d = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        assert metrics.topk_mask(d, 0.4).tolist() == [False, True, False, True, False]

    def test_always_selects_at_least_one(self):
        assert metrics.topk_mask(np.arange(1000.0), 1e-9).sum() == 1

    def test_rejects_out_of_range_fracs(self):
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="frac must be in"):
                metrics.topk_mask(np.arange(10.0), bad)

    def test_selects_exactly_the_requested_count(self):
        assert metrics.topk_mask(np.arange(100.0), 0.25).sum() == 25


class TestTopKSimilarJaccard:
    def test_identical_vectors_give_one(self):
        v = np.random.default_rng(0).random(200)
        assert metrics.topk_similar_jaccard(v, v.copy(), 0.1) == 1.0

    def test_reversed_ranking_gives_zero(self):
        # smallest of v are the largest of -v -> the top-k closest sets are disjoint
        v = np.arange(200, dtype=float)
        assert metrics.topk_similar_jaccard(v, -v, 0.25) == 0.0

    def test_known_overlap(self):
        # a's 2 smallest = idx {0,1}; b's 2 smallest = idx {1,2}; Jaccard = 1/3
        a = np.array([0.0, 1.0, 2.0, 3.0])
        b = np.array([3.0, 0.0, 1.0, 2.0])
        assert metrics.topk_similar_jaccard(a, b, 0.5) == pytest.approx(1 / 3)

    def test_frac_rounds_to_at_least_one(self):
        v = np.arange(10, dtype=float)
        assert metrics.topk_similar_jaccard(v, v.copy(), 0.001) == 1.0   # k floored to 1

    def test_is_symmetric(self):
        rng = np.random.default_rng(1)
        a, b = rng.random(300), rng.random(300)
        assert metrics.topk_similar_jaccard(a, b, 0.1) == metrics.topk_similar_jaccard(b, a, 0.1)


def test_the_three_former_call_sites_agree():
    """The consolidation's whole point: one definition of 'the closest frac of pairs'."""
    from SpAM_Simulations.core import pipeline
    from SpAM_Simulations.empirical import gt_construction
    from SpAM_Simulations.measures import recovery
    assert gt_construction.topk_similar_jaccard is metrics.topk_similar_jaccard
    assert pipeline.topk_similar_jaccard is metrics.topk_similar_jaccard
    assert recovery.topk_mask is metrics.topk_mask
