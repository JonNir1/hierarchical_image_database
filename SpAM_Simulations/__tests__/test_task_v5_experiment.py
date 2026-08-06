"""Tests for task-v5: the v4 screening model on a bounded canvas.

The property that matters is that v5 produces arrangements the deployed task could actually
produce. v4 did not: its median per-trial max distance was 1.39 on a scale whose ceiling is 1.0.
"""
import numpy as np
import pytest

from SpAM_Simulations import canvas as cv
from SpAM_Simulations.task_v4_experiment import (
    TaskV4ExperimentParameters, simulate_task_v4_experiment,
)
from SpAM_Simulations.task_v5_experiment import simulate_task_v5_experiment

N_IMAGES, N_DIMS = 120, 5


def _params(**over):
    base = dict(num_subjects=6, trials_per_subject=4, images_per_trial=12,
                subjects_noise_scale=0.08, subjects_noise_df=5, frac_trials_repeated=0.25,
                perspective_dispersion=0.3, screening_trials=3, screening_repeats=1,
                screening_min_reliability=-1.0, subjects_noise_lognormal_sigma=0.0)
    base.update(over)
    return TaskV4ExperimentParameters(**base)


def _gt(seed=0):
    return np.random.default_rng(seed).normal(size=(N_IMAGES, N_DIMS)).astype(np.float32)


def _observed(results):
    """Mean observed distance per judged pair."""
    counts = np.asarray(results.num_obs)
    obs = counts > 0
    return np.asarray(results.distances)[obs] / counts[obs]


def test_v5_distances_are_physically_possible_and_v4s_are_not():
    """The regression that motivated the model: 1.0 is the canvas diagonal, nothing may exceed it."""
    _, v5 = simulate_task_v5_experiment(_params(), _gt(), np.random.default_rng(1), verbose=False)
    assert _observed(v5).max() <= 1.0

    _, v4 = simulate_task_v4_experiment(_params(), _gt(), np.random.default_rng(1), verbose=False)
    assert _observed(v4).max() > 1.0            # the unbounded model exceeds the ceiling


def test_v5_leaves_v4_untouched():
    """The seam must be inert by default, or every published run silently changes."""
    a = simulate_task_v4_experiment(_params(), _gt(), np.random.default_rng(7), verbose=False)[1]
    b = simulate_task_v4_experiment(_params(), _gt(), np.random.default_rng(7), verbose=False,
                                    trial_simulator=None)[1]
    np.testing.assert_array_equal(a.distances, b.distances)


def test_v5_still_screens_and_still_records_test_retest():
    """The canvas replaces the placement step only; the v4 screening machinery is untouched."""
    _, res = simulate_task_v5_experiment(_params(screening_min_reliability=-1.0), _gt(),
                                         np.random.default_rng(2), verbose=False)
    tr = np.asarray(res.subject_test_retest)
    assert tr.shape == (6,)
    assert np.isfinite(tr).any()
    assert np.all(tr[np.isfinite(tr)] <= 1.0)


def test_a_fixed_canvas_can_be_pinned():
    spec = cv.CanvasSpec(aspect=0.6, fill=0.9, softness=6.0)
    _, res = simulate_task_v5_experiment(_params(), _gt(), np.random.default_rng(3),
                                         verbose=False, canvas=spec)
    assert _observed(res).max() <= 1.0


def test_sampling_the_canvas_changes_the_result_but_not_its_legality():
    """Note the two are not a paired comparison: `sample_spec` draws from the same RNG stream, so
    the runs diverge in which images land in which trial, not only in canvas shape."""
    fixed = simulate_task_v5_experiment(_params(), _gt(), np.random.default_rng(4), verbose=False,
                                        sample_canvas_per_trial=False)[1]
    sampled = simulate_task_v5_experiment(_params(), _gt(), np.random.default_rng(4), verbose=False,
                                          sample_canvas_per_trial=True)[1]
    assert not np.allclose(fixed.distances, sampled.distances)
    assert _observed(fixed).max() <= 1.0 and _observed(sampled).max() <= 1.0


def test_softness_is_a_sensitivity_axis_that_actually_moves_the_result():
    """If sweeping softness changed nothing, there would be no point sweeping it."""
    soft = simulate_task_v5_experiment(_params(), _gt(), np.random.default_rng(5), verbose=False,
                                       sample_canvas_per_trial=False, softness=3.0)[1]
    hard = simulate_task_v5_experiment(_params(), _gt(), np.random.default_rng(5), verbose=False,
                                       sample_canvas_per_trial=False, softness=12.0)[1]
    assert not np.allclose(_observed(soft), _observed(hard))
    # Harder walls let the periphery sit further out, so the largest distances grow.
    assert _observed(hard).max() >= _observed(soft).max()


def test_v5_is_reproducible_from_its_seed():
    a = simulate_task_v5_experiment(_params(), _gt(), np.random.default_rng(11), verbose=False)[1]
    b = simulate_task_v5_experiment(_params(), _gt(), np.random.default_rng(11), verbose=False)[1]
    np.testing.assert_array_equal(a.distances, b.distances)
