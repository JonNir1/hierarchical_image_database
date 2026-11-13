"""Tests for the SimulationConfig / MDSSweepConfig dataclasses."""
import numpy as np
import pytest

from SpAM_Simulations.core.config import (
    SimulationConfig, TaskV2_3SimulationConfig, TaskV2_4SimulationConfig, TaskV3SimulationConfig,
    TaskV4SimulationConfig, TaskV5SimulationConfig, MDSSweepConfig
)
from SpAM_Simulations.core.pipeline import _task_key
from SpAM_Simulations.models.task_v2_3_experiment import TaskV2_3ExperimentParameters
from SpAM_Simulations.models.task_v2_4_experiment import TaskV2_4ExperimentParameters
from SpAM_Simulations.models.task_v4_experiment import TaskV4ExperimentParameters
from SpAM_Simulations.models.task_v5_experiment import TaskV5ExperimentParameters


def _base_grids(**over):
    g = dict(num_subjects=[10, 20], trials_per_subject=[8], images_per_trial=[6],
             subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1])
    g.update(over)
    return g


def test_random_ground_truth_ok():
    cfg = SimulationConfig(n_images=30, n_dims=4, reps=2, **_base_grids())
    assert cfg.uses_random_ground_truth
    # product: 2 * 1 * 1 * 2 * 1 = 4 combinations
    assert len(cfg.param_grid()) == 4


def test_embeddings_ground_truth_ok():
    emb = np.random.default_rng(0).random((15, 3))
    cfg = SimulationConfig(gt_embeddings=emb, **_base_grids())
    assert not cfg.uses_random_ground_truth


def test_must_pick_exactly_one_ground_truth_source():
    with pytest.raises(ValueError):  # neither
        SimulationConfig(**_base_grids())
    with pytest.raises(ValueError):  # both
        SimulationConfig(n_images=10, n_dims=2, gt_embeddings=np.zeros((4, 2)), **_base_grids())


def test_empty_grid_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(n_images=10, n_dims=2, **_base_grids(num_subjects=[]))


def test_nonpositive_reps_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(n_images=10, n_dims=2, reps=0, **_base_grids())


def test_target_dims_default_range():
    assert MDSSweepConfig(min_ndim=2).target_dims(gt_dimensions=5) == [2, 3, 4, 5]


def test_target_dims_explicit_override():
    assert MDSSweepConfig(ndims=[5, 7, 10]).target_dims(gt_dimensions=10) == [5, 7, 10]


def test_target_dims_empty_raises():
    with pytest.raises(ValueError):
        MDSSweepConfig(min_ndim=6).target_dims(gt_dimensions=5)


def test_task_v2_3_config_is_a_simulation_config():
    """Dataclass inheritance: gets the GT-source fields/validation for free."""
    cfg = TaskV2_3SimulationConfig(
        n_images=30, n_dims=4, frac_images_repeated=[1 / 3], **_base_grids()
    )
    assert isinstance(cfg, SimulationConfig)
    assert cfg.uses_random_ground_truth


def test_task_v2_3_config_inherits_ground_truth_validation():
    with pytest.raises(ValueError):  # neither GT source
        TaskV2_3SimulationConfig(frac_images_repeated=[1 / 3], **_base_grids())
    with pytest.raises(ValueError):  # both GT sources
        TaskV2_3SimulationConfig(
            n_images=10, n_dims=2, gt_embeddings=np.zeros((4, 2)),
            frac_images_repeated=[1 / 3], **_base_grids()
        )


def test_task_v2_3_config_rejects_empty_frac_images_repeated():
    with pytest.raises(ValueError):
        TaskV2_3SimulationConfig(n_images=10, n_dims=2, frac_images_repeated=[], **_base_grids())


def test_task_v2_3_config_param_grid():
    cfg = TaskV2_3SimulationConfig(
        n_images=30, n_dims=4, frac_images_repeated=[0.0, 1 / 3], **_base_grids()
    )
    grid = cfg.param_grid()
    # product: 2 (num_subjects) * 1 * 1 * 2 (noise_scale) * 1 * 2 (frac_images_repeated) = 8
    assert len(grid) == 8
    assert all(isinstance(p, TaskV2_3ExperimentParameters) for p in grid)
    assert {p.frac_images_repeated for p in grid} == {0.0, 1 / 3}


def test_task_v2_4_config_is_a_task_v2_3_config():
    """Dataclass inheritance: inherits the v2.3 levers/validation and the GT-source handling."""
    cfg = TaskV2_4SimulationConfig(
        n_images=30, n_dims=4, frac_images_repeated=[0.0],
        frac_trials_repeated=[0.0, 0.2], **_base_grids()
    )
    assert isinstance(cfg, TaskV2_3SimulationConfig)
    assert isinstance(cfg, SimulationConfig)
    assert cfg.uses_random_ground_truth


@pytest.mark.parametrize("bad", [
    dict(frac_images_repeated=[0.0], frac_trials_repeated=[]),
    dict(frac_images_repeated=[0.0], frac_trials_repeated=[1.0]),
    dict(frac_images_repeated=[0.0], frac_trials_repeated=[-0.1]),
    dict(frac_images_repeated=[0.0], frac_trials_repeated=[0.0, 1.5]),
    dict(frac_images_repeated=[], frac_trials_repeated=[0.0]),   # inherited from v2.3
], ids=["empty-trials", "trials-1.0", "trials-negative", "trials-above-1", "empty-images-inherited"])
def test_task_v2_4_config_rejects_invalid_repeat_grids(bad):
    with pytest.raises(ValueError):
        TaskV2_4SimulationConfig(n_images=10, n_dims=2, **bad, **_base_grids())


def test_task_v2_4_config_param_grid():
    cfg = TaskV2_4SimulationConfig(
        n_images=30, n_dims=4, frac_images_repeated=[0.0],
        frac_trials_repeated=[0.0, 0.2], **_base_grids()
    )
    grid = cfg.param_grid()
    # product: 2 (num_subjects) * 1 * 1 * 2 (noise_scale) * 1 * 1 * 2 (frac_trials_repeated) = 8
    assert len(grid) == 8
    assert all(isinstance(p, TaskV2_4ExperimentParameters) for p in grid)
    assert {p.frac_trials_repeated for p in grid} == {0.0, 0.2}


# --------------------------------------------------------------------------- task-v4 (screening)

def _v4_kwargs(**over):
    kw = dict(n_images=400, n_dims=4, frac_trials_repeated=[0.2], perspective_dispersion=[0.2],
              screening_trials=[8], screening_repeats=[2], screening_min_reliability=[-1.0, 0.2])
    kw.update(over)
    return kw


def test_task_v4_config_is_a_task_v3_config():
    cfg = TaskV4SimulationConfig(**_v4_kwargs(), **_base_grids())
    assert isinstance(cfg, TaskV3SimulationConfig)
    assert cfg.uses_random_ground_truth


def test_task_v4_config_param_grid():
    cfg = TaskV4SimulationConfig(**_v4_kwargs(), **_base_grids())
    grid = cfg.param_grid()
    # product: 2 (num_subjects) * 1 * 1 * 2 (noise_scale) * 1 * 1 * 1 * 1 * 1 * 2 (min_reliability)
    assert len(grid) == 8
    assert all(isinstance(p, TaskV4ExperimentParameters) for p in grid)
    assert {p.screening_min_reliability for p in grid} == {-1.0, 0.2}
    assert {p.screening_trials for p in grid} == {8}


@pytest.mark.parametrize("over, match", [
    (dict(screening_trials=[]), "screening_trials"),
    (dict(screening_repeats=[]), "screening_repeats"),
    (dict(screening_min_reliability=[]), "screening_min_reliability"),
    (dict(screening_min_reliability=[-1.5]), "screening_min_reliability"),
    (dict(screening_min_reliability=[1.5]), "screening_min_reliability"),
    # no un-repeated original trial: caught at config time, not hours into the sweep
    (dict(screening_trials=[4], screening_repeats=[4]), "at least one distinct trial"),
    # individually valid values whose Cartesian product contains an infeasible combination
    (dict(screening_trials=[8, 2], screening_repeats=[2]), "at least one distinct trial"),
    (dict(perspective_dispersion=[]), "perspective_dispersion"),   # inherited from v3
], ids=["empty-trials", "empty-repeats", "empty-min-reliability", "min-reliability-below",
        "min-reliability-above", "no-originals", "infeasible-product", "empty-dispersion-v3"])
def test_task_v4_config_rejects_invalid_grids(over, match):
    with pytest.raises(ValueError, match=match):
        TaskV4SimulationConfig(**_v4_kwargs(**over), **_base_grids())


def test_task_v4_config_allows_screening_trials_zero():
    """screening_trials=0 is the 'no screening block' arm and must stay legal."""
    cfg = TaskV4SimulationConfig(**_v4_kwargs(screening_trials=[0], screening_repeats=[0]),
                                 **_base_grids())
    assert all(p.screening_trials == 0 for p in cfg.param_grid())


# --------------------------------------------------------------------------- task-v5 (canvas)

def _v5_config(**over):
    return TaskV5SimulationConfig(**_v4_kwargs(**over), **_base_grids())


@pytest.mark.parametrize("over, match", [
    (dict(within_session_drift=[]), "non-empty"),
    (dict(within_session_drift=[0.0]), "positive"),
    (dict(canvas_softness=[]), "non-empty"),
    (dict(canvas_softness=[0.0]), "positive"),
], ids=["empty-drift", "nonpositive-drift", "empty-softness", "nonpositive-softness"])
def test_task_v5_config_rejects_empty_or_nonpositive_grid(over, match):
    with pytest.raises(ValueError, match=match):
        _v5_config(**over)


def test_drift_is_a_grid_axis():
    """It has to reach `param_grid`, or a swept sensitivity value would silently do nothing."""
    grid = _v5_config(within_session_drift=[1.0, 1.4]).param_grid()
    assert sorted({p.within_session_drift for p in grid}) == [1.0, 1.4]


def test_v5_params_survive_the_store_round_trip():
    """`_task_key` coerces every field with float(), so every v5 field must be numeric.

    Fields are located by NAME rather than by offset from the end: the key is
    ``(*params, rep, ndim)``, so appending a field to the tuple shifts every negative index and a
    positional assertion here would fail for a reason that has nothing to do with what it checks.
    """
    params = TaskV5ExperimentParameters(*[1.0] * 12, 4.0)
    key = _task_key(params, rep=0, ndim=3)
    assert key[params._fields.index("canvas_softness")] == 4.0
    assert key[params._fields.index("exclude_false_positives")] == 0.0
    assert key[params._fields.index("within_session_drift")] == 1.0
    assert all(isinstance(v, float) for v in key[:-2])
    assert len(key) == len(params._fields) + 2            # + rep, ndim
