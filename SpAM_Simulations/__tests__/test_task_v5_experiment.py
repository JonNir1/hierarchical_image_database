"""Tests for task-v5: the v4 screening model on a bounded canvas.

The property that matters is that v5 produces arrangements the deployed task could actually
produce. v4 did not: its median per-trial max distance was 1.39 on a scale whose ceiling is 1.0.
"""
import numpy as np
import pytest

from SpAM_Simulations.models import canvas as cv
from SpAM_Simulations.models.task_v4_experiment import (
    TaskV4ExperimentParameters, simulate_task_v4_experiment,
)
from SpAM_Simulations.models.task_v5_experiment import simulate_task_v5_experiment

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


# --------------------------------------------------------------------- sweepability

def _v5_config(**over):
    from SpAM_Simulations.core.config import TaskV5SimulationConfig
    base = dict(gt_embeddings=_gt(), num_subjects=[5], trials_per_subject=[3],
                images_per_trial=[10], subjects_noise_scale=[0.08], subjects_noise_df=[5],
                frac_trials_repeated=[1 / 3], perspective_dispersion=[0.3], screening_trials=[2],
                screening_repeats=[1], screening_min_reliability=[-1.0], reps=2, seed=1)
    base.update(over)
    return TaskV5SimulationConfig(**base)


def test_canvas_softness_is_swept_and_reaches_the_metric_tables():
    """The whole reason it is a parameter: it must become a grouping column, not a hidden constant."""
    from SpAM_Simulations.core import pipeline
    cfg = _v5_config(canvas_softness=[3.0, 8.0])
    assert len(cfg.param_grid()) == 2
    sim = pipeline.generate_task_v5_simulation(cfg, verbose=False)
    cov = pipeline.compute_coverage_table(sim)
    assert "canvas_softness" in cov.columns
    assert sorted(cov["canvas_softness"].unique()) == [3.0, 8.0]
    assert len(cov) == 4                                  # 2 softness x 2 reps


def test_v5_params_survive_the_store_round_trip():
    """`_task_key` coerces every field with float(), so every v5 field must be numeric.

    Fields are located by NAME rather than by offset from the end: the key is
    ``(*params, rep, ndim)``, so appending a field to the tuple shifts every negative index and a
    positional assertion here would fail for a reason that has nothing to do with what it checks.
    """
    from SpAM_Simulations.core.pipeline import _task_key
    from SpAM_Simulations.models.task_v5_experiment import TaskV5ExperimentParameters
    params = TaskV5ExperimentParameters(*[1.0] * 12, 4.0)
    key = _task_key(params, rep=0, ndim=3)
    assert key[params._fields.index("canvas_softness")] == 4.0
    assert key[params._fields.index("exclude_false_positives")] == 0.0
    assert all(isinstance(v, float) for v in key[:-2])
    assert len(key) == len(params._fields) + 2            # + rep, ndim


def test_the_config_rejects_a_nonpositive_or_empty_softness_grid():
    with pytest.raises(ValueError, match="non-empty"):
        _v5_config(canvas_softness=[])
    with pytest.raises(ValueError, match="positive"):
        _v5_config(canvas_softness=[0.0])


def test_softness_defaults_to_the_value_carried_in_the_params():
    """A sweep varies softness through the grid, so the simulator must read it from there."""
    from SpAM_Simulations.models.task_v5_experiment import TaskV5ExperimentParameters
    fields = dict(zip(TaskV4ExperimentParameters._fields, _params()))
    soft = TaskV5ExperimentParameters(**fields, canvas_softness=3.0)
    hard = TaskV5ExperimentParameters(**fields, canvas_softness=12.0)
    a = simulate_task_v5_experiment(soft, _gt(), np.random.default_rng(5), verbose=False,
                                    sample_canvas_per_trial=False)[1]
    b = simulate_task_v5_experiment(hard, _gt(), np.random.default_rng(5), verbose=False,
                                    sample_canvas_per_trial=False)[1]
    assert not np.allclose(a.distances, b.distances)


def test_simulate_returns_the_v5_params_not_v4s_echo():
    """v4 rebuilds its own tuple; returning that would silently drop canvas_softness."""
    from SpAM_Simulations.models.task_v5_experiment import TaskV5ExperimentParameters
    fields = dict(zip(TaskV4ExperimentParameters._fields, _params()))
    p = TaskV5ExperimentParameters(**fields, canvas_softness=6.0)
    returned, _ = simulate_task_v5_experiment(p, _gt(), np.random.default_rng(0), verbose=False)
    assert returned is p
    assert returned.canvas_softness == 6.0
