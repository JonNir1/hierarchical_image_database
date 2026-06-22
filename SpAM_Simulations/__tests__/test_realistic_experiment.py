"""Tests for the realistic (per-subject trial design) experiment simulation."""
import numpy as np
import pytest

from SpAM_Simulations.design import compute_design_counts
from SpAM_Simulations.realistic_experiment import (
    simulate_realistic_single_subject,
    simulate_realistic_experiment,
    RealisticExperimentParameters,
    _find_candidate_repeated_pairs,
    _compute_subject_snr,
)
from SpAM_Simulations.experiment import _condensed_pair_indices
from SpAM_Simulations.simulation import Simulation

TASK_DEFAULTS = dict(t=10, k=20, r=1 / 3)  # matches SpAM_Task/task_config.json


def _make_params(**overrides):
    fields = dict(
        num_subjects=12, trials_per_subject=10, images_per_trial=20,
        subjects_noise_scale=0.4, subjects_noise_df=3, frac_images_repeated=1 / 3,
    )
    fields.update(overrides)
    return RealisticExperimentParameters(**fields)


class TestFindCandidateRepeatedPairs:
    def test_finds_pairs_sharing_both_trials(self):
        N = 10
        trials = [
            np.array([1, 2, 3]),
            np.array([1, 2, 4]),
            np.array([5, 6, 7]),
        ]
        # images 1 and 2 both appear in trials 0 and 1 -> repeated pair (1, 2).
        # image 3 only appears in trial 0, image 4 only in trial 1 -> never repeated.
        candidates = _find_candidate_repeated_pairs(trials, N)
        expected = _condensed_pair_indices(np.array([1]), np.array([2]), N)
        np.testing.assert_array_equal(np.sort(candidates), np.sort(expected))

    def test_no_candidates_when_nothing_repeats(self):
        N = 10
        trials = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        candidates = _find_candidate_repeated_pairs(trials, N)
        assert candidates.size == 0

    def test_image_in_three_trials_is_not_a_candidate_source(self):
        # an image appearing in 3 trials isn't "double" by this subject's design (only
        # exactly-2-trial images count), so it must not generate candidate pairs.
        N = 10
        trials = [np.array([1, 2]), np.array([1, 3]), np.array([1, 4])]
        candidates = _find_candidate_repeated_pairs(trials, N)
        assert candidates.size == 0


class TestComputeSubjectSnr:
    def test_nan_without_repeated_pairs(self):
        obs = np.array([1.0, 2.0, 0.0])
        n_obs = np.array([1, 1, 0])
        assert np.isnan(_compute_subject_snr(obs, n_obs, {}))

    def test_ratio_matches_manual_calculation(self):
        obs = np.array([2.0, 4.0, 6.0])
        n_obs = np.array([1, 1, 1])
        repeated = {0: [1.0, 1.5], 1: [3.0, 5.0]}  # |delta| = 0.5, 2.0 -> mean = 1.25
        snr = _compute_subject_snr(obs, n_obs, repeated)
        sigma_d = np.std(obs / n_obs, ddof=1)
        assert snr == pytest.approx(sigma_d / 1.25)

    def test_zero_delta_gives_infinite_snr(self):
        obs = np.array([2.0, 4.0, 6.0])
        n_obs = np.array([1, 1, 1])
        repeated = {0: [1.0, 1.0]}  # identical repeats -> delta = 0
        assert np.isinf(_compute_subject_snr(obs, n_obs, repeated))

    def test_raises_if_a_pair_has_other_than_2_measurements(self):
        obs = np.array([2.0, 4.0])
        n_obs = np.array([1, 1])
        with pytest.raises(AssertionError):
            _compute_subject_snr(obs, n_obs, {0: [1.0, 1.5, 2.0]})


class TestSimulateRealisticSingleSubject:
    def test_output_shapes_match_condensed_gt(self):
        sim = Simulation.make(200, 4, seed=1)
        n_unique, n_double = compute_design_counts(**TASK_DEFAULTS)
        obs, n_obs, snr = simulate_realistic_single_subject(
            subject_noise=0.3, t=TASK_DEFAULTS["t"], k=TASK_DEFAULTS["k"],
            n_unique=n_unique, n_double=n_double,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(0), verbose=False,
        )
        assert obs.shape == sim.gt_distances.shape
        assert n_obs.shape == sim.gt_distances.shape
        assert isinstance(snr, float)

    def test_noiseless_subject_has_infinite_snr(self):
        sim = Simulation.make(200, 4, seed=2)
        n_unique, n_double = compute_design_counts(**TASK_DEFAULTS)
        _, _, snr = simulate_realistic_single_subject(
            subject_noise=0.0, t=TASK_DEFAULTS["t"], k=TASK_DEFAULTS["k"],
            n_unique=n_unique, n_double=n_double,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(3), verbose=False,
        )
        assert np.isinf(snr)

    def test_snr_is_nan_without_repeated_pairs(self):
        # frac_images_repeated=0 -> n_double=0 -> no subject can ever observe a repeated pair
        sim = Simulation.make(60, 4, seed=4)
        n_unique, n_double = compute_design_counts(t=4, k=5, r=0.0)
        assert n_double == 0
        _, _, snr = simulate_realistic_single_subject(
            subject_noise=0.3, t=4, k=5, n_unique=n_unique, n_double=n_double,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(5), verbose=False,
        )
        assert np.isnan(snr)

    def test_reproducible_independent_of_global_rng_state(self):
        sim = Simulation.make(200, 4, seed=6)
        n_unique, n_double = compute_design_counts(**TASK_DEFAULTS)

        np.random.seed(0)
        obs_a, nobs_a, snr_a = simulate_realistic_single_subject(
            subject_noise=0.4, t=TASK_DEFAULTS["t"], k=TASK_DEFAULTS["k"],
            n_unique=n_unique, n_double=n_double,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(42), verbose=False,
        )
        np.random.seed(99999)  # perturb the global RNG: must not affect the result
        obs_b, nobs_b, snr_b = simulate_realistic_single_subject(
            subject_noise=0.4, t=TASK_DEFAULTS["t"], k=TASK_DEFAULTS["k"],
            n_unique=n_unique, n_double=n_double,
            gt_distances=sim.gt_distances, rng=np.random.default_rng(42), verbose=False,
        )
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(nobs_a, nobs_b)
        assert snr_a == snr_b


class TestSimulateRealisticExperiment:
    def test_reproducible_from_seed_only(self):
        sim = Simulation.make(200, 4, seed=10)
        params = _make_params()

        np.random.seed(0)
        _, r1 = simulate_realistic_experiment(params, sim.gt_distances, np.random.default_rng(99), verbose=False)
        np.random.seed(123456)
        _, r2 = simulate_realistic_experiment(params, sim.gt_distances, np.random.default_rng(99), verbose=False)

        np.testing.assert_array_equal(r1.distances, r2.distances)
        np.testing.assert_array_equal(r1.num_obs, r2.num_obs)
        np.testing.assert_array_equal(r1.subject_noises, r2.subject_noises)
        np.testing.assert_array_equal(r1.subject_snr, r2.subject_snr)

        _, r3 = simulate_realistic_experiment(params, sim.gt_distances, np.random.default_rng(100), verbose=False)
        assert not np.array_equal(np.nan_to_num(r1.distances), np.nan_to_num(r3.distances))

    def test_output_shapes(self):
        sim = Simulation.make(200, 4, seed=11)
        params = _make_params(num_subjects=9)
        _, res = simulate_realistic_experiment(params, sim.gt_distances, np.random.default_rng(1), verbose=False)
        assert res.distances.shape == sim.gt_distances.shape
        assert res.num_obs.shape == sim.gt_distances.shape
        assert res.subject_noises.shape == (9,)
        assert res.subject_snr.shape == (9,)

    def test_unmeasured_pairs_are_nan(self):
        sim = Simulation.make(200, 4, seed=12)
        params = _make_params(num_subjects=3)
        _, res = simulate_realistic_experiment(params, sim.gt_distances, np.random.default_rng(1), verbose=False)
        assert np.isnan(res.distances).any()  # 3 subjects can't cover every pair of 200 images

    def test_rejects_n_unique_exceeding_pool(self):
        sim = Simulation.make(20, 4, seed=13)  # pool too small for the default design
        params = _make_params()
        with pytest.raises(AssertionError):
            simulate_realistic_experiment(params, sim.gt_distances, np.random.default_rng(1), verbose=False)

    @pytest.mark.parametrize("overrides", [
        dict(num_subjects=0),
        dict(trials_per_subject=0),
        dict(subjects_noise_scale=-0.1),
        dict(subjects_noise_df=0),
    ])
    def test_rejects_invalid_parameters(self, overrides):
        sim = Simulation.make(200, 4, seed=14)
        params = _make_params(**overrides)
        with pytest.raises(AssertionError):
            simulate_realistic_experiment(params, sim.gt_distances, np.random.default_rng(1), verbose=False)

    def test_mean_snr_decreases_with_noise_scale(self):
        """Statistical sanity check (not bit-exact): more subject noise should yield a
        lower mean SNR, averaged over several repetitions to smooth out sampling noise."""
        sim = Simulation.make(200, 4, seed=15)

        def mean_snr_over_reps(scale, reps=6, seed0=1000):
            vals = []
            for i in range(reps):
                params = _make_params(num_subjects=20, subjects_noise_scale=scale, subjects_noise_df=5)
                _, res = simulate_realistic_experiment(
                    params, sim.gt_distances, np.random.default_rng(seed0 + i), verbose=False
                )
                vals.append(np.nanmean(res.subject_snr))
            return np.nanmean(vals)

        low_noise_snr = mean_snr_over_reps(0.1)
        high_noise_snr = mean_snr_over_reps(0.8)
        assert low_noise_snr > high_noise_snr
