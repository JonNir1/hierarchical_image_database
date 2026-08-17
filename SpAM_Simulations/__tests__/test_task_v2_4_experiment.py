"""Tests for the task-v2.4 simulation (task-v2.3 design + frac_trials_repeated whole-trial repeats)."""
import numpy as np
import pytest

from SpAM_Simulations.models.design import compute_design_counts, distinct_trial_count
from SpAM_Simulations.core.simulation import Simulation
from SpAM_Simulations.models.task_v2_3_experiment import (
    simulate_task_v2_3_experiment, TaskV2_3ExperimentParameters
)
from SpAM_Simulations.models.task_v2_4_experiment import (
    simulate_task_v2_4_experiment, simulate_task_v2_4_single_subject,
    TaskV2_4ExperimentParameters, _trial_test_retest,
)


def _make_params(**overrides):
    fields = dict(
        num_subjects=12, trials_per_subject=10, images_per_trial=20,
        subjects_noise_scale=0.4, subjects_noise_df=3,
        frac_images_repeated=0.0, frac_trials_repeated=0.2,
    )
    fields.update(overrides)
    return TaskV2_4ExperimentParameters(**fields)


class TestTrialTestRetest:
    def test_identical_vectors_give_one(self):
        v = np.array([1.0, 3.0, 2.0, 5.0])
        assert _trial_test_retest(v, v.copy()) == pytest.approx(1.0)

    def test_constant_vector_gives_nan(self):
        assert np.isnan(_trial_test_retest(np.ones(4), np.array([1.0, 2.0, 3.0, 4.0])))

    def test_monotone_reordering_below_one(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 4.0, 3.0])  # one swap -> Spearman < 1
        assert _trial_test_retest(a, b) < 1.0


class TestSimulateTaskV2_4SingleSubject:
    def _design(self, t=10, k=20, r=0.0, fr=0.3):
        t_distinct = distinct_trial_count(t, fr)
        n_unique, n_double = compute_design_counts(t_distinct, k, r)
        n_repeats = t - t_distinct
        return t_distinct, k, n_unique, n_double, n_repeats

    def test_output_shapes_and_types(self):
        sim = Simulation.make(200, 4, seed=1)
        t_distinct, k, n_unique, n_double, n_repeats = self._design()
        obs, n_obs, snr, retest = simulate_task_v2_4_single_subject(
            subject_noise=0.3, t_distinct=t_distinct, k=k, n_unique=n_unique,
            n_double=n_double, n_repeats=n_repeats,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(0), verbose=False,
        )
        assert obs.shape == sim.gt_distances.shape
        assert n_obs.shape == sim.gt_distances.shape
        assert isinstance(snr, float) and isinstance(retest, float)

    def test_repeated_trial_pairs_observed_twice(self):
        # r=0 -> distinct trials hold disjoint image sets, so the only pairs observed twice are
        # those of the n_repeats repeated trials: exactly n_repeats * C(k,2) condensed pairs.
        sim = Simulation.make(200, 4, seed=2)
        k, n_repeats = 20, 3
        t_distinct, _, n_unique, n_double, _ = self._design(t=10, k=k, r=0.0, fr=0.3)
        assert n_double == 0 and (10 - t_distinct) == n_repeats
        _, n_obs, _, _ = simulate_task_v2_4_single_subject(
            subject_noise=0.3, t_distinct=t_distinct, k=k, n_unique=n_unique,
            n_double=n_double, n_repeats=n_repeats,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(7), verbose=False,
        )
        pairs_per_trial = k * (k - 1) // 2
        assert int(np.sum(n_obs == 2)) == n_repeats * pairs_per_trial
        assert n_obs.max() == 2  # never observed more than twice (singles-only repeats)

    def test_repeat_redraws_distances_not_reuses_them(self):
        # If a repeat reused the original's distances, the two presentations would be identical
        # and the test-retest correlation would be exactly 1.0 even under noise. A value < 1
        # therefore proves the repeat drew fresh noise.
        sim = Simulation.make(200, 4, seed=3)
        t_distinct, k, n_unique, n_double, n_repeats = self._design()
        _, _, _, retest = simulate_task_v2_4_single_subject(
            subject_noise=0.5, t_distinct=t_distinct, k=k, n_unique=n_unique,
            n_double=n_double, n_repeats=n_repeats,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(11), verbose=False,
        )
        assert np.isfinite(retest) and retest < 1.0

    def test_noiseless_repeat_has_unit_reliability(self):
        sim = Simulation.make(200, 4, seed=4)
        t_distinct, k, n_unique, n_double, n_repeats = self._design()
        _, _, _, retest = simulate_task_v2_4_single_subject(
            subject_noise=0.0, t_distinct=t_distinct, k=k, n_unique=n_unique,
            n_double=n_double, n_repeats=n_repeats,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(13), verbose=False,
        )
        assert retest == pytest.approx(1.0)

    def test_reliability_nan_without_repeats(self):
        sim = Simulation.make(200, 4, seed=5)
        t_distinct, k, n_unique, n_double, _ = self._design(t=10, k=20, r=0.0, fr=0.0)
        _, _, _, retest = simulate_task_v2_4_single_subject(
            subject_noise=0.4, t_distinct=t_distinct, k=k, n_unique=n_unique,
            n_double=n_double, n_repeats=0,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(17), verbose=False,
        )
        assert np.isnan(retest)


class TestSimulateTaskV2_4Experiment:
    def test_reproducible_from_seed_only(self):
        sim = Simulation.make(200, 4, seed=10)
        params = _make_params()
        np.random.seed(0)
        _, r1 = simulate_task_v2_4_experiment(params, sim.gt_distances, np.random.default_rng(99), verbose=False)
        np.random.seed(123456)
        _, r2 = simulate_task_v2_4_experiment(params, sim.gt_distances, np.random.default_rng(99), verbose=False)
        np.testing.assert_array_equal(r1.distances, r2.distances)
        np.testing.assert_array_equal(r1.num_obs, r2.num_obs)
        np.testing.assert_array_equal(r1.subject_noises, r2.subject_noises)
        np.testing.assert_array_equal(r1.subject_test_retest, r2.subject_test_retest)

    def test_output_shapes(self):
        sim = Simulation.make(200, 4, seed=11)
        params = _make_params(num_subjects=9)
        _, res = simulate_task_v2_4_experiment(params, sim.gt_distances, np.random.default_rng(1), verbose=False)
        assert res.distances.shape == sim.gt_distances.shape
        assert res.num_obs.shape == sim.gt_distances.shape
        assert res.subject_noises.shape == (9,)
        assert res.subject_snr.shape == (9,)
        assert res.subject_test_retest.shape == (9,)

    def test_bit_exact_to_v2_3_when_frac_trials_repeated_zero(self):
        # frac_trials_repeated=0 consumes no extra RNG and runs no repeat trials, so the v2.4
        # simulation must reproduce v2.3 exactly (distances, observations, noises, SNR), and the
        # test-retest reliability is undefined for every subject.
        gt = Simulation.make(200, 4, seed=7).gt_distances
        common = dict(num_subjects=15, trials_per_subject=10, images_per_trial=20,
                      subjects_noise_scale=0.5, subjects_noise_df=3, frac_images_repeated=1 / 3)
        p3 = TaskV2_3ExperimentParameters(**common)
        p4 = TaskV2_4ExperimentParameters(**common, frac_trials_repeated=0.0)
        _, r3 = simulate_task_v2_3_experiment(p3, gt, np.random.default_rng(99), verbose=False)
        _, r4 = simulate_task_v2_4_experiment(p4, gt, np.random.default_rng(99), verbose=False)
        np.testing.assert_array_equal(np.nan_to_num(r3.distances), np.nan_to_num(r4.distances))
        np.testing.assert_array_equal(r3.num_obs, r4.num_obs)
        np.testing.assert_array_equal(r3.subject_noises, r4.subject_noises)
        np.testing.assert_array_equal(np.nan_to_num(r3.subject_snr, nan=-1),
                                      np.nan_to_num(r4.subject_snr, nan=-1))
        assert np.all(np.isnan(r4.subject_test_retest))

    def test_noiseless_experiment_has_unit_reliability(self):
        sim = Simulation.make(200, 4, seed=12)
        params = _make_params(num_subjects=6, subjects_noise_scale=0.0, frac_trials_repeated=0.3)
        _, res = simulate_task_v2_4_experiment(params, sim.gt_distances, np.random.default_rng(1), verbose=False)
        assert np.allclose(res.subject_test_retest, 1.0)

    @pytest.mark.parametrize("overrides", [
        dict(num_subjects=0),
        dict(trials_per_subject=0),
        dict(subjects_noise_scale=-0.1),
        dict(subjects_noise_df=0),
        dict(frac_trials_repeated=1.0),
        dict(frac_trials_repeated=-0.1),
    ])
    def test_rejects_invalid_parameters(self, overrides):
        sim = Simulation.make(200, 4, seed=14)
        params = _make_params(**overrides)
        with pytest.raises(AssertionError):
            simulate_task_v2_4_experiment(params, sim.gt_distances, np.random.default_rng(1), verbose=False)

    def test_infeasible_repeat_pool_raises(self):
        # frac_images_repeated=1/3 saturates the trials with doubled images, so no singles-only
        # trial is left to repeat: select_repeat_trials must raise (matching the JS task).
        sim = Simulation.make(725, 4, seed=15)
        params = _make_params(num_subjects=3, frac_images_repeated=1 / 3, frac_trials_repeated=0.2)
        with pytest.raises(RuntimeError):
            simulate_task_v2_4_experiment(params, sim.gt_distances, np.random.default_rng(1), verbose=False)

    def test_mean_reliability_decreases_with_noise_scale(self):
        """Statistical sanity check (not bit-exact): more subject noise should yield a lower
        mean test-retest reliability, averaged over several repetitions to smooth sampling noise."""
        sim = Simulation.make(200, 4, seed=16)

        def mean_retest_over_reps(scale, reps=6, seed0=1000):
            vals = []
            for i in range(reps):
                params = _make_params(num_subjects=20, subjects_noise_scale=scale,
                                      subjects_noise_df=5, frac_trials_repeated=0.3)
                _, res = simulate_task_v2_4_experiment(
                    params, sim.gt_distances, np.random.default_rng(seed0 + i), verbose=False
                )
                vals.append(np.nanmean(res.subject_test_retest))
            return np.nanmean(vals)

        assert mean_retest_over_reps(0.1) > mean_retest_over_reps(0.8)
