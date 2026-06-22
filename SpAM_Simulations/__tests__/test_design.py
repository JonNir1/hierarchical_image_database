"""Tests for the per-subject trial-allocation port of SpAM_Task/js/trial_generator.js.

Mirrors the contract asserted by SpAM_Task/__tests__/trial_generator.test.js: exactly t
trials of k images, no duplicates within a trial, exactly n_double images appearing in 2
trials (rest in 1), determinism from a seed, and a raise when the design is infeasible.
"""
from collections import Counter

import numpy as np
import pytest

from SpAM_Simulations.design import build_trial_lists, compute_design_counts


@pytest.fixture
def design():
    # r=1/3 -> n_unique = round(200 / 1.333) = 150; n_double = 200 - 150 = 50
    t, k, r = 10, 20, 1 / 3
    n_unique, n_double = compute_design_counts(t, k, r)
    return t, k, n_unique, n_double


def _make_trials(design, seed=42, pool_size=754):
    t, k, n_unique, n_double = design
    rng = np.random.default_rng(seed)
    active = rng.choice(pool_size, size=n_unique, replace=False)
    return build_trial_lists(active, t, k, n_double, rng)


class TestComputeDesignCounts:
    def test_matches_task_config_defaults(self):
        n_unique, n_double = compute_design_counts(10, 20, 1 / 3)
        assert n_unique == 150
        assert n_double == 50

    def test_zero_repeats(self):
        n_unique, n_double = compute_design_counts(3, 2, 0)
        assert n_unique == 6
        assert n_double == 0

    @pytest.mark.parametrize("r", [0.5, 0.6, 1.0])
    def test_rejects_r_at_or_above_half(self, r):
        with pytest.raises(AssertionError):
            compute_design_counts(10, 20, r)

    @pytest.mark.parametrize("r", [-0.1, -1.0])
    def test_rejects_negative_r(self, r):
        with pytest.raises(AssertionError):
            compute_design_counts(10, 20, r)


class TestBuildTrialLists:
    def test_returns_t_trials_each_with_exactly_k_images(self, design):
        t, k, _, _ = design
        trials = _make_trials(design)
        assert len(trials) == t
        for trial in trials:
            assert len(trial) == k

    def test_no_duplicates_within_a_single_trial(self, design):
        for trial in _make_trials(design):
            assert len(set(trial.tolist())) == len(trial)

    def test_exactly_n_double_images_appear_in_2_trials_rest_in_1(self, design):
        _, _, _, n_double = design
        counts = Counter()
        for trial in _make_trials(design):
            counts.update(trial.tolist())
        freq = list(counts.values())
        assert sum(1 for c in freq if c == 2) == n_double
        assert all(c in (1, 2) for c in freq)

    def test_deterministic_with_the_same_seed(self, design):
        t, k, n_unique, n_double = design
        rng_a = np.random.default_rng(7)
        active_a = rng_a.choice(754, size=n_unique, replace=False)
        trials_a = build_trial_lists(active_a, t, k, n_double, np.random.default_rng(99))

        rng_b = np.random.default_rng(7)
        active_b = rng_b.choice(754, size=n_unique, replace=False)
        trials_b = build_trial_lists(active_b, t, k, n_double, np.random.default_rng(99))

        for a, b in zip(trials_a, trials_b):
            np.testing.assert_array_equal(a, b)

    def test_throws_when_the_image_pool_cannot_fill_all_trials(self):
        # 2 images, 3 trials x 2 slots: impossible to fill every trial (each single image
        # can only be placed once), mirroring the JS test's r=0, N=6-but-pool-has-2 case.
        with pytest.raises(RuntimeError):
            build_trial_lists(np.array([0, 1]), t=3, k=2, n_double=0, rng=np.random.default_rng(0))

    def test_throws_when_too_few_eligible_trials_for_a_double_image(self):
        # t=1 trial: a double image needs 2 distinct trials, but only 1 exists.
        with pytest.raises(RuntimeError):
            build_trial_lists(np.array([0, 1, 2]), t=1, k=3, n_double=1, rng=np.random.default_rng(0))

    def test_zero_repeats_fills_trials_with_all_single_images(self):
        t, k, r = 4, 5, 0.0
        n_unique, n_double = compute_design_counts(t, k, r)
        assert n_double == 0
        rng = np.random.default_rng(1)
        active = rng.choice(100, size=n_unique, replace=False)
        trials = build_trial_lists(active, t, k, n_double, rng)
        counts = Counter()
        for trial in trials:
            counts.update(trial.tolist())
        assert all(c == 1 for c in counts.values())
