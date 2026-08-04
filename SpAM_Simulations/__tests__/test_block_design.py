"""Tests for the balanced block-design generators.

Correctness is anchored on small designs whose optimum is known from combinatorial design theory
(Fano plane, affine/projective planes), since a greedy covering has no closed form at the sizes we
actually use. The greedy is a heuristic, so the assertions are "valid covering, and within a stated
factor of the known optimum" rather than exact block counts.

Nothing here needs R, and everything runs in seconds; the real n=725 timing is exercised separately
in test_design_comparison.py.
"""
import numpy as np
import pytest

from SpAM_Simulations.block_design import (
    best_of_greedy, design_stats, greedy_design, greedy_session_design, pair_counts,
    schonheim, session_design_stats,
)

# (n_images, k, lam, known optimal block count, name)
KNOWN_BIBDS = [
    (7, 3, 1, 7, "Fano plane PG(2,2)"),
    (9, 3, 1, 12, "affine plane AG(2,3)"),
    (13, 4, 1, 13, "projective plane PG(2,3)"),
    (25, 5, 1, 30, "affine plane AG(2,5)"),
    (15, 3, 1, 35, "BIBD(15,3,1)"),
]


def _rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------- covering correctness

@pytest.mark.parametrize("n,k,lam,known,name", KNOWN_BIBDS)
def test_greedy_produces_a_valid_covering(n, k, lam, known, name):
    """Every pair must be covered at least lam times - that is the defining property."""
    design = greedy_design(n, k, lam, _rng())
    counts = pair_counts(design, n)
    assert counts.min() >= lam, f"{name}: uncovered pairs"
    assert design.shape[1] == k


@pytest.mark.parametrize("n,k,lam,known,name", KNOWN_BIBDS)
def test_greedy_lands_near_the_known_optimum(n, k, lam, known, name):
    """Greedy is a heuristic, so it cannot beat the optimum and should not badly exceed it.

    The tiny designs are where it fares *worst*, having the fewest choices to make: Fano is the
    outlier at 9 blocks against an optimum of 7 (1.29x), while AG(2,3) is hit exactly. At the real
    n=725 the overhead is 1.53x at lambda=1 and falls to 1.10x by lambda=5, since the expensive part
    is the greedy tail mopping up the last few uncovered pairs.
    """
    design, _ = best_of_greedy(n, k, lam, m=20, rng=_rng(1))
    assert known <= design.shape[0] <= known * 1.30, (
        f"{name}: {design.shape[0]} blocks vs known optimum {known}"
    )


@pytest.mark.parametrize("n,k,lam,known,name", KNOWN_BIBDS)
def test_schonheim_bound_is_not_violated(n, k, lam, known, name):
    if lam == 1:
        assert schonheim(n, k) <= known


def test_higher_lambda_covers_every_pair_more_often():
    for lam in (1, 2, 3):
        counts = pair_counts(greedy_design(15, 3, lam, _rng()), 15)
        assert counts.min() >= lam


def test_best_of_greedy_returns_the_smallest_design():
    design, sizes = best_of_greedy(15, 3, 1, m=8, rng=_rng(2))
    assert design.shape[0] == min(sizes)
    assert len(sizes) == 8


# --------------------------------------------------------------------- session design

def test_session_design_never_repeats_an_image_within_a_session():
    """The invariant the deployed task guarantees, and every existing simulation assumes."""
    sessions = greedy_session_design(60, 5, trials_per_session=6, n_sessions=10, rng=_rng())
    for s in sessions:
        flat = s.ravel()
        assert len(np.unique(flat)) == flat.size


def test_session_design_shape_and_fill():
    sessions = greedy_session_design(60, 5, trials_per_session=6, n_sessions=4, rng=_rng())
    assert sessions.shape == (4, 6, 5)
    assert sessions.min() >= 0 and sessions.max() < 60


def test_session_design_is_deterministic_under_a_fixed_seed():
    a = greedy_session_design(60, 5, 6, 5, _rng(7))
    b = greedy_session_design(60, 5, 6, 5, _rng(7))
    np.testing.assert_array_equal(a, b)


def test_session_design_best_of_m_is_at_least_as_covering_as_a_single_draw():
    """`m` selects on coverage here, not on block count, which is fixed by n_sessions * t."""
    n, k, t, n_sess = 60, 5, 6, 8
    single = session_design_stats(greedy_session_design(n, k, t, n_sess, _rng(5), m=1), n)
    best = session_design_stats(greedy_session_design(n, k, t, n_sess, _rng(5), m=5), n)
    assert best["frac_pairs_covered"] >= single["frac_pairs_covered"]
    assert best["n_blocks"] == single["n_blocks"]


def test_session_design_rejects_oversized_sessions():
    """trials_per_session * k must fit in the image pool, since a session is image-disjoint."""
    with pytest.raises(ValueError, match="distinct images"):
        greedy_session_design(20, 5, trials_per_session=6, n_sessions=1, rng=_rng())


def test_session_design_keeps_covering_past_one_full_covering():
    """The deficit refills, so more sessions keep adding coverage rather than degenerating."""
    few = session_design_stats(greedy_session_design(40, 4, 5, 3, _rng()), 40)
    many = session_design_stats(greedy_session_design(40, 4, 5, 30, _rng()), 40)
    assert many["frac_pairs_covered"] > few["frac_pairs_covered"]
    assert many["min_pair_count"] >= few["min_pair_count"]


def test_session_design_beats_random_on_pair_coverage():
    """The whole point of the exercise, at a size small enough to run in a test."""
    n, k, t, n_sess = 60, 5, 6, 12
    designed = greedy_session_design(n, k, t, n_sess, _rng())
    rng = _rng(3)
    random = np.stack([rng.permutation(n)[:t * k].reshape(t, k) for _ in range(n_sess)])
    d = session_design_stats(designed, n)
    r = session_design_stats(random, n)
    assert d["frac_pairs_covered"] > r["frac_pairs_covered"]
    assert d["reps_per_image_sd"] < r["reps_per_image_sd"]


# --------------------------------------------------------------------- stats

def test_pair_counts_sum_matches_the_number_of_within_block_pairs():
    n, k = 30, 5
    design = greedy_design(n, k, 1, _rng())
    expected = design.shape[0] * k * (k - 1) // 2
    assert int(pair_counts(design, n).sum()) == expected


def test_design_stats_reports_a_full_covering_as_such():
    n = 25
    stats = design_stats(greedy_design(n, 5, 1, _rng()), n)
    assert stats["frac_pairs_covered"] == 1.0
    assert stats["min_pair_count"] >= 1
    assert stats["partners_per_image_mean"] == pytest.approx(n - 1)
