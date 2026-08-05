"""Tests for the Stage 2a design-only comparison.

Small n throughout so the suite stays fast; the real n=725 run is exercised once, separately, in
test_runs_at_the_deployed_scale (still ~seconds, no MDS involved).

No R needed.
"""
import numpy as np
import pytest

from SpAM_Simulations import design_comparison as dc


def _rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------- arm construction

@pytest.mark.parametrize("arm", dc.ARMS)
def test_every_arm_produces_the_requested_shape(arm):
    s = dc.build_sessions(arm, n_images=60, k=5, trials_per_session=4, n_sessions=6, rng=_rng())
    assert s.shape == (6, 4, 5)
    assert s.min() >= 0 and s.max() < 60


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm|arm must be"):
        dc.build_sessions("nonsense", 60, 5, 4, 6, _rng())


def test_only_the_unconstrained_arm_repeats_images_within_a_session():
    """The deployable arms must honour the task's one-image-per-subject guarantee."""
    kw = dict(n_images=60, k=5, trials_per_session=6, n_sessions=8)
    for arm in ("random", "designed"):
        rep = dc.design_report(dc.build_sessions(arm, rng=_rng(), **kw), 60)
        assert rep["within_session_duplicate_images"] == 0, arm


# --------------------------------------------------------------------- headline claim

def test_designed_covers_more_pairs_than_random_at_matched_cost():
    """The whole point of Stage 2a."""
    kw = dict(n_images=80, k=5, trials_per_session=6, n_sessions=10)
    d = dc.design_report(dc.build_sessions("designed", rng=_rng(1), **kw), 80)
    r = dc.design_report(dc.build_sessions("random", rng=_rng(1), **kw), 80)
    assert d["frac_pairs_covered"] > r["frac_pairs_covered"]


def test_designed_balances_image_replication_better_than_random():
    kw = dict(n_images=80, k=5, trials_per_session=6, n_sessions=10)
    d = dc.design_report(dc.build_sessions("designed", rng=_rng(1), **kw), 80)
    r = dc.design_report(dc.build_sessions("random", rng=_rng(1), **kw), 80)
    assert d["reps_per_image_sd"] < r["reps_per_image_sd"]


def test_designed_wastes_fewer_observations_on_already_covered_pairs():
    kw = dict(n_images=80, k=5, trials_per_session=6, n_sessions=10)
    d = dc.design_report(dc.build_sessions("designed", rng=_rng(1), **kw), 80)
    r = dc.design_report(dc.build_sessions("random", rng=_rng(1), **kw), 80)
    assert d["wasted_frac"] < r["wasted_frac"]


# --------------------------------------------------------------------- invariants

@pytest.mark.parametrize("arm", dc.ARMS)
def test_pair_counts_sum_to_the_number_of_within_trial_pairs(arm):
    """Every arm observes exactly t * C(k,2) pairs per session, however it chooses them."""
    n_images, k, t, n_sess = 60, 5, 4, 6
    s = dc.build_sessions(arm, n_images, k, t, n_sess, _rng())
    rep = dc.design_report(s, n_images)
    assert rep["mean_pair_count"] * rep["n_pairs"] == pytest.approx(n_sess * t * k * (k - 1) / 2)


def test_coverage_increases_with_cohort_size():
    kw = dict(n_images=100, k=5, trials_per_session=5)
    small = dc.design_report(dc.build_sessions("random", n_sessions=5, rng=_rng(), **kw), 100)
    large = dc.design_report(dc.build_sessions("random", n_sessions=40, rng=_rng(), **kw), 100)
    assert large["frac_pairs_covered"] > small["frac_pairs_covered"]


# --------------------------------------------------------------------- the sweep

def test_compare_designs_covers_the_full_grid():
    df = dc.compare_designs(n_images=60, k=5, trials_per_session=4, n_list=(5, 10), reps=2,
                            verbose=False)
    assert len(df) == len(dc.ARMS) * 2 * 2
    assert set(df["arm"]) == set(dc.ARMS)
    assert set(df["num_subjects"]) == {5, 10}


def test_compare_designs_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        dc.compare_designs(n_images=60, k=5, trials_per_session=4, n_list=(5,), reps=1,
                           arms=("random", "nope"), verbose=False)


def test_summarise_comparison_reduces_to_one_row_per_cell():
    df = dc.compare_designs(n_images=60, k=5, trials_per_session=4, n_list=(5, 10), reps=3,
                            verbose=False)
    summary = dc.summarise_comparison(df)
    assert len(summary) == len(dc.ARMS) * 2
    assert "frac_pairs_covered" in summary.columns


def test_reps_differ_so_the_random_arm_carries_allocation_variance():
    """Both arms must be resampled per rep, else their spreads are not comparable."""
    df = dc.compare_designs(n_images=60, k=5, trials_per_session=4, n_list=(8,), reps=5,
                            arms=("random",), verbose=False)
    assert df["frac_pairs_covered"].nunique() > 1


# --------------------------------------------------------------------- deployed scale

def test_runs_at_the_deployed_scale():
    """725 images, 20 per trial, 18 distinct trials - the real configuration, one rep per arm.

    Also pins the arithmetic quoted in the module docstring: 18*C(20,2) = 3420 pairs per subject of
    262,450, and random coverage tracking 1 - exp(-0.01303*N).
    """
    df = dc.compare_designs(n_list=(30,), reps=1, verbose=False)
    assert len(df) == len(dc.ARMS)
    assert (df["n_pairs"] == 725 * 724 // 2).all()
    np.testing.assert_allclose(df["mean_pair_count"] * df["n_pairs"], 30 * 18 * 190)

    random_cov = df[df.arm == "random"]["frac_pairs_covered"].iloc[0]
    assert random_cov == pytest.approx(1 - np.exp(-0.01303 * 30), abs=0.02)
    assert df[df.arm == "designed"]["frac_pairs_covered"].iloc[0] > random_cov
    # Connectivity saturates at these sizes, so it cannot discriminate - asserted, not assumed.
    assert df["single_component"].all()
