"""Tests for the task-v4 simulation (task-v3's generative model plus the v4.0 screening block).

The generative core is imported unchanged from ``task_v3_experiment`` and is covered by
``test_task_v3_experiment.py``; this suite therefore focuses on what v4 adds - the screening
block, the recruit-until-N loop, the pooling of screening data into the analysed aggregate - plus
a regression guard that v4 without screening still reproduces v3 exactly.
"""
import numpy as np
from typing import NamedTuple
import pytest

from SpAM_Simulations.core.simulation import build_ground_truth_embeddings
from SpAM_Simulations.models.task_v3_experiment import (
    simulate_task_v3_experiment, TaskV3ExperimentParameters,
)
from SpAM_Simulations.models.task_v4_experiment import (
    simulate_task_v4_experiment, simulate_task_v4_single_subject, TaskV4ExperimentParameters,
    _CandidateNoisePool, _passes_screening,
)

GT = build_ground_truth_embeddings(120, 5, seed=1)


def _params(**overrides):
    fields = dict(
        num_subjects=20, trials_per_subject=8, images_per_trial=10,
        subjects_noise_scale=0.6, subjects_noise_df=5,
        frac_trials_repeated=0.25, perspective_dispersion=0.2,
        screening_trials=4, screening_repeats=1, screening_min_reliability=-1.0,
        subjects_noise_lognormal_sigma=0.0,     # 0.0 = the historical |t(df)| family
    )
    fields.update(overrides)
    return TaskV4ExperimentParameters(**fields)


def _run(seed=0, **overrides):
    _, res = simulate_task_v4_experiment(_params(**overrides), GT,
                                         np.random.default_rng(seed), verbose=False)
    return res


class TestEquivalenceToTaskV3:
    """With the screening block switched off, v4 must be the v3 model - bit for bit.

    This is the guard that keeps every previously-run v3 sweep comparable to the v4 results: if
    the port drifted, the no-screening arm would no longer be a valid baseline.
    """

    COMMON = dict(num_subjects=15, trials_per_subject=8, images_per_trial=10,
                  subjects_noise_scale=0.5, subjects_noise_df=3,
                  frac_trials_repeated=0.25, perspective_dispersion=0.3)

    def _pair(self, seed=7):
        _, v3 = simulate_task_v3_experiment(
            TaskV3ExperimentParameters(**self.COMMON), GT, np.random.default_rng(seed), verbose=False)
        _, v4 = simulate_task_v4_experiment(
            TaskV4ExperimentParameters(**self.COMMON, screening_trials=0, screening_repeats=0,
                                       screening_min_reliability=-1.0,
                                       subjects_noise_lognormal_sigma=0.0),
            GT, np.random.default_rng(seed), verbose=False)
        return v3, v4

    def test_distances_and_counts_identical(self):
        v3, v4 = self._pair()
        np.testing.assert_array_equal(np.nan_to_num(v3.distances, nan=-1),
                                      np.nan_to_num(v4.distances, nan=-1))
        np.testing.assert_array_equal(v3.num_obs, v4.num_obs)

    def test_subject_level_diagnostics_identical(self):
        v3, v4 = self._pair()
        np.testing.assert_array_equal(v3.subject_noises, v4.subject_noises)
        np.testing.assert_allclose(np.nan_to_num(v3.subject_test_retest, nan=-9),
                                   np.nan_to_num(v4.subject_test_retest, nan=-9))
        np.testing.assert_allclose(np.nan_to_num(v3.subject_test_retest_procrustes, nan=-9),
                                   np.nan_to_num(v4.subject_test_retest_procrustes, nan=-9))

    def test_no_screening_means_no_rejections(self):
        _, v4 = self._pair()
        assert v4.n_candidates_screened == self.COMMON["num_subjects"]
        assert v4.screening_pass_rate == 1.0


class TestCandidateNoisePool:
    """The pool must preserve the |t(df)| heterogeneity that screening selects against."""

    def test_single_draws_are_not_all_identical(self):
        """Regression: drawing batches of 1 from `_draw_subject_noises` collapses every value to
        the mean (it divides by the mean of one number), destroying the heavy tail entirely."""
        pool = _CandidateNoisePool(df=5, scale=0.6, batch_size=32, rng=np.random.default_rng(0))
        values = [pool.next() for _ in range(64)]
        assert np.std(values) > 0.05

    def test_batch_mean_matches_the_configured_scale(self):
        pool = _CandidateNoisePool(df=5, scale=0.6, batch_size=50, rng=np.random.default_rng(0))
        assert np.mean([pool.next() for _ in range(50)]) == pytest.approx(0.6)

    def test_refills_beyond_the_first_batch(self):
        pool = _CandidateNoisePool(df=5, scale=0.6, batch_size=4, rng=np.random.default_rng(0))
        assert all(v >= 0 for v in (pool.next() for _ in range(17)))


class TestScreeningRule:
    """`_passes_screening` implements evaluateScreening's per-repeat MINIMUM rule."""

    def test_any_single_bad_repeat_excludes(self):
        assert not _passes_screening([0.9, 0.9, -0.1], 0.0)   # one bad repeat is enough
        assert _passes_screening([0.9, 0.9, 0.05], 0.0)

    def test_threshold_is_inclusive(self):
        assert _passes_screening([0.2], 0.2)                  # exactly at threshold passes

    def test_nan_repeats_are_skipped_not_failed(self):
        assert _passes_screening([np.nan, 0.5], 0.2)          # the usable repeat decides
        assert _passes_screening([np.nan], 0.9)               # no evidence -> retained


class TestScreeningEffect:
    def test_stricter_threshold_rejects_more_candidates(self):
        rates = [_run(screening_min_reliability=t).screening_pass_rate
                 for t in (-1.0, 0.2, 0.5)]
        assert rates[0] == 1.0
        assert rates[0] > rates[1] > rates[2]

    def test_stricter_threshold_retains_more_precise_subjects(self):
        noises = [_run(screening_min_reliability=t).subject_noises.mean()
                  for t in (-1.0, 0.2, 0.5)]
        assert noises[0] > noises[1] > noises[2]

    def test_stricter_threshold_raises_retained_reliability(self):
        rs = [np.nanmean(_run(screening_min_reliability=t).subject_test_retest)
              for t in (-1.0, 0.2, 0.5)]
        assert rs[0] < rs[1] < rs[2]

    def test_screening_truncates_the_heavy_tail(self):
        """Screening's mechanism is removing the worst subjects, not shifting everyone."""
        assert _run(screening_min_reliability=0.5).subject_noises.max() < \
               _run(screening_min_reliability=-1.0).subject_noises.max()

    def test_retained_cohort_is_always_exactly_num_subjects(self):
        """Recruit-until-N: the analysed cohort size is fixed, the recruitment cost varies."""
        for thr in (-1.0, 0.0, 0.3, 0.6):
            res = _run(screening_min_reliability=thr)
            assert res.subject_noises.size == 20
            assert res.n_candidates_screened >= 20

    def test_pass_rate_matches_the_candidate_count(self):
        res = _run(screening_min_reliability=0.4)
        assert res.screening_pass_rate == pytest.approx(20 / res.n_candidates_screened)


class TestScreeningFalsePositives:
    """Subjects the gate let through whose main stage fails the same rule.

    The deployed task evaluates once and never revisits, so it cannot see these. They are the
    simulated counterpart of a real participant who clears the screening block, is paid in full,
    and then stops trying - and the only part of prod's false-positive rate the model can match at
    all, since simulated subjects arrange every image and so can never fail the move-ratio check.
    """

    def test_rate_is_a_proportion_of_the_retained_cohort(self):
        res = _run(screening_min_reliability=0.0)
        assert 0.0 <= res.screening_false_positive_rate <= 1.0

    def test_nobody_is_removed_for_being_a_false_positive(self):
        """Diagnostic only: the deployed task does not drop them, so neither may the model."""
        res = _run(screening_min_reliability=0.3)
        assert res.subject_noises.size == 20
        assert res.screening_false_positive_rate > 0  # and yet the cohort is still full

    def test_a_stricter_gate_leaves_MORE_false_positives(self):
        """Regression to the mean, and it runs against intuition.

        The gate selects on one noisy estimate of reliability. The higher the bar, the more of the
        subjects who clear it did so partly by luck, and the more of them fall back below it when
        re-measured on the main stage. So tightening the threshold buys a better cohort *and* a
        larger share of survivors who do not hold up - which is exactly the quantity the deployed
        task is blind to.
        """
        lenient = np.mean([_run(seed=s, screening_min_reliability=0.0)
                           .screening_false_positive_rate for s in range(4)])
        strict = np.mean([_run(seed=s, screening_min_reliability=0.5)
                          .screening_false_positive_rate for s in range(4)])
        assert strict > lenient

    def test_no_gate_means_no_false_positives_by_definition(self):
        """A false positive is defined against the gate's own rule, and -1.0 excludes nobody."""
        res = _run(screening_min_reliability=-1.0)
        assert res.screening_pass_rate == 1.0
        assert res.screening_false_positive_rate == 0.0


class TestScreeningDataIsAnalysed:
    def test_screening_trials_add_observations(self):
        """A retained subject's screening trials are data, so they must reach the aggregate."""
        without = _run(screening_trials=0, screening_repeats=0)
        with_screening = _run(screening_trials=4, screening_repeats=1)
        assert with_screening.num_obs.sum() > without.num_obs.sum()

    def test_screening_repeats_feed_the_test_retest_summary(self):
        """Pooling both stages' repeats is why the per-subject values are returned unaggregated."""
        one_repeat = _run(screening_trials=4, screening_repeats=1)
        no_screening = _run(screening_trials=0, screening_repeats=0)
        assert np.isfinite(one_repeat.subject_test_retest).all()
        assert np.isfinite(no_screening.subject_test_retest).all()


class TestStagesUseDisjointImages:
    """`partitionIntoStages` guarantees no image crosses stages; the simulation must match.

    Drawing the two stages' pools independently would overlap them - at the deployed design about
    40 of a subject's 360 images - manufacturing within-subject cross-stage pair observations the
    real task cannot produce, and inflating coverage.
    """

    def test_no_pair_is_observed_across_both_stages(self):
        """With disjoint pools every within-subject pair count stays at 1 (2 for a repeated trial).

        A shared image across stages would put some pair in both stages' trials, so its count for
        a single subject would exceed what one stage alone can produce.
        """
        params = _params(num_subjects=1, trials_per_subject=6, images_per_trial=10,
                         frac_trials_repeated=0.0, screening_trials=3, screening_repeats=0,
                         screening_min_reliability=-1.0)
        _, res = simulate_task_v4_experiment(params, GT, np.random.default_rng(0), verbose=False)
        # 6 main + 3 screening trials of 10 images, all disjoint -> every observed pair seen once
        assert res.num_obs.max() == 1
        assert res.num_obs.sum() == 9 * (10 * 9 // 2)

    def test_pool_larger_than_the_image_set_is_rejected(self):
        """The two stages share one pool, so the check must be on their SUM."""
        with pytest.raises(AssertionError, match="on top of the main stage"):
            simulate_task_v4_experiment(
                _params(trials_per_subject=10, images_per_trial=10, frac_trials_repeated=0.0,
                        screening_trials=4, screening_repeats=0),
                build_ground_truth_embeddings(130, 5, seed=1),  # needs 40 + 100 = 140 > 130
                np.random.default_rng(0), verbose=False)


class TestSingleSubject:
    def test_returns_per_repeat_values_not_means(self):
        run = simulate_task_v4_single_subject(
            subject_noise=0.4, perspective_dispersion=0.2, t_distinct=6, k=10,
            n_unique=60, n_repeats=3, gt_embeddings=GT, rng=np.random.default_rng(0),
        )
        assert len(run.repeat_correlations) == 3 and len(run.repeat_procrustes) == 3
        assert all(0.0 <= m <= 1.0 for m in run.repeat_procrustes)

    def test_no_repeats_gives_empty_lists(self):
        run = simulate_task_v4_single_subject(
            subject_noise=0.4, perspective_dispersion=0.0, t_distinct=6, k=10,
            n_unique=60, n_repeats=0, gt_embeddings=GT, rng=np.random.default_rng(0),
        )
        assert run.repeat_correlations == [] and run.repeat_procrustes == []


class TestRecruitmentCap:
    """An unreachable screening threshold must fail fast, not spin forever.

    The loop retries until `num_subjects` candidates pass, so with a threshold nothing can reach it
    produces no output at all. On an unattended EC2 sweep that burns the whole run and surfaces only
    as a stalled log, which is why this is an error rather than a warning.
    """

    def test_unreachable_threshold_raises_rather_than_hanging(self):
        with pytest.raises(RuntimeError, match="screening recruited"):
            simulate_task_v4_experiment(
                _params(num_subjects=2, screening_trials=4, screening_repeats=1,
                        screening_min_reliability=1.0, subjects_noise_scale=0.8),
                GT, np.random.default_rng(0), verbose=False, max_recruit_per_subject=20,
            )

    def test_the_error_names_the_pass_rate_and_the_threshold(self):
        """The message has to be diagnosable from a log with no other context."""
        with pytest.raises(RuntimeError) as excinfo:
            simulate_task_v4_experiment(
                _params(num_subjects=2, screening_trials=4, screening_repeats=1,
                        screening_min_reliability=1.0, subjects_noise_scale=0.8),
                GT, np.random.default_rng(0), verbose=False, max_recruit_per_subject=20,
            )
        msg = str(excinfo.value)
        assert "pass rate" in msg
        assert "screening_min_reliability=1.0" in msg
        assert "subjects_noise_scale=0.8" in msg

    def test_the_cap_scales_with_num_subjects(self):
        """Absolute caps break real configurations: the deployed 0.4 threshold legitimately
        screens ~52 candidates per retained subject, so the budget must grow with the cohort."""
        params = _params(num_subjects=4, screening_trials=4, screening_repeats=1,
                         screening_min_reliability=1.0, subjects_noise_scale=0.8)
        with pytest.raises(RuntimeError, match="recruited 40 candidates"):
            simulate_task_v4_experiment(params, GT, np.random.default_rng(0), verbose=False,
                                        max_recruit_per_subject=10)

    def test_a_reachable_threshold_is_unaffected(self):
        res = _run(seed=1, screening_trials=4, screening_repeats=1, screening_min_reliability=-1.0)
        assert res.n_candidates_screened == 20   # nobody rejected, so no extra recruitment


class TestValidation:
    @pytest.mark.parametrize("bad", [
        dict(screening_trials=-1),
        dict(screening_trials=4, screening_repeats=4),      # every trial a repeat: no originals
        dict(screening_min_reliability=1.5),
        dict(screening_min_reliability=-2.0),
    ])
    def test_invalid_screening_params_raise(self, bad):
        with pytest.raises(AssertionError):
            simulate_task_v4_experiment(_params(**bad), GT, np.random.default_rng(0), verbose=False)

    def test_screening_pool_larger_than_image_set_raises(self):
        with pytest.raises(AssertionError, match="screening block needs"):
            simulate_task_v4_experiment(_params(screening_trials=20, screening_repeats=1),
                                        GT, np.random.default_rng(0), verbose=False)


def test_determinism_same_seed():
    a, b = _run(seed=3, screening_min_reliability=0.3), _run(seed=3, screening_min_reliability=0.3)
    np.testing.assert_array_equal(np.nan_to_num(a.distances, nan=-1),
                                  np.nan_to_num(b.distances, nan=-1))
    assert a.n_candidates_screened == b.n_candidates_screened


class TestNoisePopulationFamily:
    """The per-subject noise population's SHAPE is now a fitted parameter, not an assumption.

    Checking the simulation against 36 real subjects showed |t(df)| is the wrong family: its
    coefficient of variation cannot fall below ~0.756 (the half-normal limit) however large df
    gets, while the pilot's reliability distribution needs ~0.47. The lognormal option exists to
    span that range.
    """

    def test_sigma_zero_is_the_historical_t_family(self):
        """The 0.0 sentinel must reproduce |t(df)| exactly, so old parameter tuples keep meaning."""
        a = _run(seed=5, subjects_noise_lognormal_sigma=0.0)
        b = _run(seed=5, subjects_noise_lognormal_sigma=0.0, subjects_noise_df=3)
        assert not np.array_equal(a.subject_noises, b.subject_noises)   # df still matters
        c = _run(seed=5, subjects_noise_lognormal_sigma=0.0)
        np.testing.assert_array_equal(a.subject_noises, c.subject_noises)

    def test_lognormal_sigma_controls_dispersion(self):
        lo = _run(seed=5, subjects_noise_lognormal_sigma=0.2).subject_noises
        hi = _run(seed=5, subjects_noise_lognormal_sigma=0.9).subject_noises
        assert np.std(lo) / np.mean(lo) < np.std(hi) / np.mean(hi)

    def test_lognormal_reaches_below_the_t_family_floor(self):
        """The whole reason the family was added: |t| cannot express a concentrated cohort."""
        from SpAM_Simulations.models.noise_population import population_cv
        t_floor = min(population_cv("t", df) for df in (5, 30, 200))
        assert t_floor > 0.74                                   # the half-normal limit
        assert population_cv("lognormal", 0.45) < 0.6           # comfortably below it
        assert population_cv("lognormal", 0.15) < 0.2           # near-homogeneous is reachable

    def test_mean_scale_is_preserved_across_families(self):
        """`subjects_noise_scale` keeps its meaning (and its calibration) when the shape changes."""
        for sigma in (0.0, 0.3, 0.7):
            r = _run(seed=5, screening_min_reliability=-1.0, subjects_noise_lognormal_sigma=sigma)
            assert r.subject_noises.mean() == pytest.approx(0.6, rel=1e-6)

    def test_negative_sigma_rejected(self):
        with pytest.raises(AssertionError, match="lognormal_sigma"):
            simulate_task_v4_experiment(_params(subjects_noise_lognormal_sigma=-0.1), GT,
                                        np.random.default_rng(0), verbose=False)


class TestExcludeFalsePositives:
    """Modelling the ANALYSIS: drop a subject the gate let through but the experimental block fails.

    The flag lives on the v5 tuple, so v4 reads it defensively. These tests drive it through a
    stand-in params object rather than the v5 tuple, to keep the v4 suite independent of v5.
    """

    def _run_with_flag(self, flag, seed=0, **overrides):
        """v4 params carrying an `exclude_false_positives` attribute, as v5's tuple would."""
        base = _params(**overrides)
        fields = list(base._fields) + ["exclude_false_positives"]
        Extended = NamedTuple("Extended", [(f, float) for f in fields])
        params = Extended(*[getattr(base, f) for f in base._fields], float(flag))
        _, res = simulate_task_v4_experiment(params, GT, np.random.default_rng(seed), verbose=False)
        return res

    def test_default_is_a_no_op(self):
        """The v4 tuple has no such field at all, so the getattr default must keep v5 behaviour."""
        plain = _run(seed=3, screening_min_reliability=0.2)
        flagged = self._run_with_flag(0.0, seed=3, screening_min_reliability=0.2)
        assert flagged.n_candidates_screened == plain.n_candidates_screened
        assert flagged.n_false_positives_discarded == 0
        np.testing.assert_array_equal(flagged.subject_noises, plain.subject_noises)

    def test_excluding_them_still_returns_exactly_num_subjects(self):
        res = self._run_with_flag(1.0, seed=3, screening_min_reliability=0.2)
        assert res.subject_noises.size == 20
        assert res.subject_test_retest.size == 20

    def test_excluding_them_costs_extra_candidates(self):
        """Every discarded false positive has to be replaced, so recruitment rises."""
        keep = self._run_with_flag(0.0, seed=3, screening_min_reliability=0.2)
        drop = self._run_with_flag(1.0, seed=3, screening_min_reliability=0.2)
        assert drop.n_false_positives_discarded > 0
        assert drop.n_candidates_screened > keep.n_candidates_screened

    def test_the_retained_cohort_is_cleaner(self):
        """That is the entire point of paying for the replacements."""
        keep = self._run_with_flag(0.0, seed=3, screening_min_reliability=0.2)
        drop = self._run_with_flag(1.0, seed=3, screening_min_reliability=0.2)
        assert np.nanmean(drop.subject_test_retest) > np.nanmean(keep.subject_test_retest)

    def test_discarded_subjects_are_counted_separately_from_early_fails(self):
        """They are paid the FULL rate, unlike a screening rejection, so cost needs both counts."""
        res = self._run_with_flag(1.0, seed=3, screening_min_reliability=0.2)
        early_fails = res.n_candidates_screened - 20 - res.n_false_positives_discarded
        assert early_fails >= 0
        assert res.n_candidates_screened == 20 + early_fails + res.n_false_positives_discarded
