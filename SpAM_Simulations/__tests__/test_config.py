"""Tests for the SimulationConfig / MDSSweepConfig dataclasses."""
import numpy as np
import pytest

from SpAM_Simulations.config import (
    SimulationConfig, TaskV2_3SimulationConfig, TaskV2_4SimulationConfig, MDSSweepConfig
)
from SpAM_Simulations.task_v2_3_experiment import TaskV2_3ExperimentParameters
from SpAM_Simulations.task_v2_4_experiment import TaskV2_4ExperimentParameters


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


def test_task_v2_4_config_rejects_empty_frac_trials_repeated():
    with pytest.raises(ValueError):
        TaskV2_4SimulationConfig(
            n_images=10, n_dims=2, frac_images_repeated=[0.0],
            frac_trials_repeated=[], **_base_grids()
        )


@pytest.mark.parametrize("bad", [[1.0], [-0.1], [0.0, 1.5]])
def test_task_v2_4_config_rejects_frac_trials_repeated_out_of_range(bad):
    with pytest.raises(ValueError):
        TaskV2_4SimulationConfig(
            n_images=10, n_dims=2, frac_images_repeated=[0.0],
            frac_trials_repeated=bad, **_base_grids()
        )


def test_task_v2_4_config_inherits_empty_frac_images_repeated_validation():
    with pytest.raises(ValueError):
        TaskV2_4SimulationConfig(
            n_images=10, n_dims=2, frac_images_repeated=[],
            frac_trials_repeated=[0.0], **_base_grids()
        )


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
