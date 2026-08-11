"""Tests for the per-subject trial-allocation port of SpAM_Task/js/trial_generator.js.

Mirrors the contract asserted by SpAM_Task/__tests__/trial_generator.test.js: exactly t
trials of k images, no duplicates within a trial, exactly n_double images appearing in 2
trials (rest in 1), determinism from a seed, and a raise when the design is infeasible.
"""
from collections import Counter

import numpy as np
import pytest

from SpAM_Simulations.models.design import (
    build_trial_lists, compute_design_counts, distinct_trial_count, select_repeat_trials
)


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


class TestDistinctTrialCount:
    @pytest.mark.parametrize("t, fr, expected", [
        (20, 0.0, 20),   # no repeats
        (20, 0.1, 18),   # round(2.0) = 2 repeats
        (10, 0.25, 8),   # round(2.5) = 2 (banker's rounding) -> 8 distinct
        (12, 0.25, 9),   # round(3.0) = 3 repeats
    ])
    def test_matches_formula(self, t, fr, expected):
        assert distinct_trial_count(t, fr) == expected

    @pytest.mark.parametrize("fr", [-0.1, 1.0, 1.5])
    def test_rejects_fr_out_of_range(self, fr):
        with pytest.raises(AssertionError):
            distinct_trial_count(10, fr)


class TestSelectRepeatTrials:
    def _singles_only_design(self, seed=3):
        # r=0 -> every image appears once -> all trials are singles-only candidates.
        t, k, r = 6, 5, 0.0
        n_unique, n_double = compute_design_counts(t, k, r)
        rng = np.random.default_rng(seed)
        active = rng.choice(100, size=n_unique, replace=False)
        return build_trial_lists(active, t, k, n_double, rng), rng

    def test_returns_n_repeats_distinct_indices(self):
        trials, rng = self._singles_only_design()
        chosen = select_repeat_trials(trials, 3, rng)
        assert len(chosen) == 3
        assert len(set(chosen)) == 3
        assert all(0 <= i < len(trials) for i in chosen)

    def test_zero_repeats_consumes_no_rng(self):
        # Early return on n_repeats == 0 must not advance the RNG (bit-exactness with v2.3).
        trials, _ = self._singles_only_design()
        rng = np.random.default_rng(123)
        before = rng.bit_generator.state
        assert select_repeat_trials(trials, 0, rng) == []
        assert rng.bit_generator.state == before

    def test_only_singles_only_trials_are_candidates(self):
        # r=1/3 -> doubled images saturate the trials; with k small there may be few or no
        # singles-only trials, and a repeat must never duplicate a trial holding a doubled image.
        t, k, r = 4, 6, 1 / 3
        n_unique, n_double = compute_design_counts(t, k, r)
        rng = np.random.default_rng(5)
        active = rng.choice(100, size=n_unique, replace=False)
        trials = build_trial_lists(active, t, k, n_double, rng)
        counts = Counter(int(i) for tr in trials for i in tr.tolist())
        singles_only = {j for j, tr in enumerate(trials)
                        if all(counts[int(i)] == 1 for i in tr.tolist())}
        n_avail = len(singles_only)
        if n_avail == 0:
            with pytest.raises(RuntimeError):
                select_repeat_trials(trials, 1, rng)
        else:
            chosen = select_repeat_trials(trials, n_avail, rng)
            assert set(chosen) <= singles_only

    def test_raises_when_too_few_singles_only_trials(self):
        trials, rng = self._singles_only_design()
        with pytest.raises(RuntimeError):
            select_repeat_trials(trials, len(trials) + 1, rng)

    def test_deterministic_with_same_seed(self):
        trials, _ = self._singles_only_design(seed=9)
        a = select_repeat_trials(trials, 2, np.random.default_rng(42))
        b = select_repeat_trials(trials, 2, np.random.default_rng(42))
        assert a == b
