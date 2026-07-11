"""Tests for the task-v3 generative (coordinate-space) simulation.

These also encode the scientific-validity checks from the plan: the 2-D projection is loss-free
only when the ground truth already fits in 2-D, projection loss appears for higher-D ground truth,
per-subject perspective is a stable trait, and the aggregate is full-rank (not secretly rank-2).
"""
import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from SpAM_Simulations.simulation import Simulation, build_ground_truth_embeddings
from SpAM_Simulations.task_v3_experiment import (
    simulate_task_v3_experiment, simulate_task_v3_single_subject, TaskV3ExperimentParameters,
    project_2d, _draw_perspective_weights, _trial_test_retest,
)


def _make_params(**overrides):
    fields = dict(
        num_subjects=12, trials_per_subject=8, images_per_trial=10,
        subjects_noise_scale=0.4, subjects_noise_df=3,
        frac_trials_repeated=0.25, perspective_dispersion=0.3,
    )
    fields.update(overrides)
    return TaskV3ExperimentParameters(**fields)


def _mean_observed(res):
    """Mean distance per observed pair, with the observed-pair mask."""
    mask = ~np.isnan(res.distances)
    return res.distances[mask] / res.num_obs[mask], mask


class TestBuildGroundTruth:
    def test_shape_dtype(self):
        emb = build_ground_truth_embeddings(200, 6, seed=1)
        assert emb.shape == (200, 6) and emb.dtype == np.float32

    def test_isotropic_spectrum_flat(self):
        emb = build_ground_truth_embeddings(2000, 6, use_isotropic=True, seed=1)
        stds = emb.std(0)
        assert stds.max() / stds.min() < 1.2  # roughly equal variance across dims

    def test_anisotropic_spectrum_decays(self):
        emb = build_ground_truth_embeddings(4000, 6, use_isotropic=False, decay=0.6, seed=1)
        stds = emb.std(0)
        assert np.all(np.diff(stds) < 0)               # monotonically decreasing
        assert stds[-1] / stds[0] < 0.2                # steep decay

    def test_determinism(self):
        a = build_ground_truth_embeddings(100, 4, use_isotropic=False, seed=7)
        b = build_ground_truth_embeddings(100, 4, use_isotropic=False, seed=7)
        assert np.array_equal(a, b)

    def test_clusters_increase_spread(self):
        flat = build_ground_truth_embeddings(2000, 4, n_clusters=None, seed=3)
        clustered = build_ground_truth_embeddings(2000, 4, n_clusters=5, seed=3)
        assert clustered.std() > flat.std()            # cluster centres widen the cloud

    @pytest.mark.parametrize("bad", [dict(decay=0.0), dict(decay=1.5), dict(n_clusters=0)])
    def test_invalid_args(self, bad):
        with pytest.raises(ValueError):
            build_ground_truth_embeddings(50, 4, **bad)


class TestProject2D:
    def test_output_shape(self):
        Y = project_2d(np.random.default_rng(0).normal(size=(15, 7)))
        assert Y.shape == (15, 2)

    def test_loss_free_when_already_2d(self):
        X = np.random.default_rng(0).normal(size=(12, 2))
        assert np.allclose(pdist(X), pdist(project_2d(X)), atol=1e-5)

    def test_pads_when_one_dimensional(self):
        X = np.random.default_rng(0).normal(size=(8, 1))
        Y = project_2d(X)
        assert Y.shape == (8, 2) and np.allclose(Y[:, 1], 0.0)


class TestPerspectiveWeights:
    def test_zero_dispersion_is_identity(self):
        assert np.array_equal(_draw_perspective_weights(6, 0.0, np.random.default_rng(0)), np.ones(6))

    def test_positive_and_spread(self):
        w = _draw_perspective_weights(2000, 0.5, np.random.default_rng(0))
        assert np.all(w > 0) and w.std() > 0


class TestTrialTestRetest:
    def test_identical_gives_one(self):
        v = np.array([1.0, 3.0, 2.0, 5.0])
        assert _trial_test_retest(v, v.copy()) == pytest.approx(1.0)

    def test_constant_gives_nan(self):
        assert np.isnan(_trial_test_retest(np.ones(4), np.array([1.0, 2.0, 3.0, 4.0])))


class TestSingleSubject:
    def test_output_shapes(self):
        emb = build_ground_truth_embeddings(120, 5, seed=1)
        n_pairs = 120 * 119 // 2
        obs, n_obs, retest, retest_m2 = simulate_task_v3_single_subject(
            subject_noise=0.3, perspective_dispersion=0.2, t_distinct=6, k=10,
            n_unique=60, n_repeats=2, gt_embeddings=emb, rng=np.random.default_rng(0),
        )
        assert obs.shape == (n_pairs,) and n_obs.shape == (n_pairs,)
        assert isinstance(retest, float) and isinstance(retest_m2, float)
        assert 0.0 <= retest_m2 <= 1.0    # Procrustes M^2 disparity is normalised to [0, 1]

    def test_no_repeats_gives_nan_retest(self):
        emb = build_ground_truth_embeddings(120, 5, seed=1)
        _, _, retest, retest_m2 = simulate_task_v3_single_subject(
            subject_noise=0.3, perspective_dispersion=0.0, t_distinct=6, k=10,
            n_unique=60, n_repeats=0, gt_embeddings=emb, rng=np.random.default_rng(0),
        )
        assert np.isnan(retest) and np.isnan(retest_m2)


class TestSimulateExperiment:
    def test_determinism_same_seed(self):
        emb = build_ground_truth_embeddings(120, 5, seed=1)
        p = _make_params()
        _, r1 = simulate_task_v3_experiment(p, emb, np.random.default_rng(0), verbose=False)
        _, r2 = simulate_task_v3_experiment(p, emb, np.random.default_rng(0), verbose=False)
        assert np.array_equal(np.nan_to_num(r1.distances), np.nan_to_num(r2.distances))
        assert np.array_equal(r1.num_obs, r2.num_obs)

    def test_unmeasured_pairs_are_nan(self):
        emb = build_ground_truth_embeddings(120, 5, seed=1)
        _, res = simulate_task_v3_experiment(_make_params(num_subjects=3), emb,
                                             np.random.default_rng(0), verbose=False)
        assert np.all(np.isnan(res.distances[res.num_obs == 0]))
        assert np.all(~np.isnan(res.distances[res.num_obs > 0]))

    def test_pool_too_small_raises(self):
        emb = build_ground_truth_embeddings(30, 4, seed=1)  # t_distinct*k = 8*10 = 80 > 30
        with pytest.raises(AssertionError):
            simulate_task_v3_experiment(_make_params(frac_trials_repeated=0.0), emb,
                                        np.random.default_rng(0), verbose=False)


class TestScientificValidity:
    def test_loss_free_recovery_when_gt_is_2d(self):
        """D<=2, zero noise, zero dispersion -> observed distances exactly equal GT."""
        emb = build_ground_truth_embeddings(120, 2, use_isotropic=True, seed=1)
        params = _make_params(subjects_noise_scale=0.0, perspective_dispersion=0.0,
                              frac_trials_repeated=0.0, num_subjects=40)
        _, res = simulate_task_v3_experiment(params, emb, np.random.default_rng(0), verbose=False)
        mean_obs, mask = _mean_observed(res)
        assert np.abs(mean_obs - pdist(emb)[mask]).max() < 1e-4

    def test_projection_loss_when_gt_is_high_d(self):
        """D>2, zero noise -> structured projection loss (observed != GT)."""
        emb = build_ground_truth_embeddings(120, 6, use_isotropic=True, seed=1)
        params = _make_params(subjects_noise_scale=0.0, perspective_dispersion=0.0,
                              frac_trials_repeated=0.0, num_subjects=40)
        _, res = simulate_task_v3_experiment(params, emb, np.random.default_rng(0), verbose=False)
        mean_obs, mask = _mean_observed(res)
        assert np.abs(mean_obs - pdist(emb)[mask]).mean() > 0.1

    def test_noise_lowers_test_retest(self):
        emb = build_ground_truth_embeddings(120, 5, seed=1)
        _, lo = simulate_task_v3_experiment(_make_params(subjects_noise_scale=0.1), emb,
                                            np.random.default_rng(0), verbose=False)
        _, hi = simulate_task_v3_experiment(_make_params(subjects_noise_scale=1.0), emb,
                                            np.random.default_rng(0), verbose=False)
        assert np.nanmean(lo.subject_test_retest) > np.nanmean(hi.subject_test_retest)

    def test_procrustes_retest_rises_with_noise(self):
        """Procrustes M^2 test-retest is a disparity (lower=better), so MORE noise -> HIGHER M^2,
        and it stays within [0, 1]."""
        emb = build_ground_truth_embeddings(120, 5, seed=1)
        _, lo = simulate_task_v3_experiment(_make_params(subjects_noise_scale=0.1), emb,
                                            np.random.default_rng(0), verbose=False)
        _, hi = simulate_task_v3_experiment(_make_params(subjects_noise_scale=1.0), emb,
                                            np.random.default_rng(0), verbose=False)
        m2_lo, m2_hi = np.nanmean(lo.subject_test_retest_procrustes), np.nanmean(hi.subject_test_retest_procrustes)
        assert 0.0 <= m2_lo <= 1.0 and 0.0 <= m2_hi <= 1.0
        assert m2_hi > m2_lo                       # opposite direction to the Spearman test-retest

    def test_test_retest_decoupled_from_dispersion(self):
        """Canvas placement noise makes test-retest a pure placement effect: at fixed
        ``subjects_noise_scale`` it is ~invariant to ``perspective_dispersion`` (the whole point of
        placing noise *after* the deterministic projection). The old coordinate-space model coupled
        the two strongly (test-retest swung ~0.5 across this dispersion range), so this locks the fix.
        """
        emb = build_ground_truth_embeddings(120, 6, use_isotropic=True, seed=1)
        retests = []
        for disp in (0.0, 0.5, 1.4):
            params = _make_params(num_subjects=80, subjects_noise_scale=0.5, perspective_dispersion=disp)
            _, res = simulate_task_v3_experiment(params, emb, np.random.default_rng(0), verbose=False)
            retests.append(np.nanmedian(res.subject_test_retest))
        assert np.ptp(retests) < 0.1  # far below the ~0.5 swing of the pre-canvas-noise model

    def test_aggregate_is_full_rank_not_two(self):
        """Local per-trial projections must collectively span >2 dims (else MDS can't recover D)."""
        emb = build_ground_truth_embeddings(120, 6, use_isotropic=True, seed=1)
        params = _make_params(num_subjects=400, perspective_dispersion=0.4, subjects_noise_scale=0.3)
        _, res = simulate_task_v3_experiment(params, emb, np.random.default_rng(1), verbose=False)
        d = res.distances.copy()
        d[np.isnan(d)] = np.nanmean(d)
        Dsq = squareform(d) ** 2
        n = Dsq.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        eig = np.linalg.eigvalsh(-0.5 * J @ Dsq @ J)[::-1]
        eig = eig[eig > 0] / eig[eig > 0][0]
        # the 5th and 6th principal coordinates still carry real signal (>>0): not rank-2
        assert eig[5] > 0.1
