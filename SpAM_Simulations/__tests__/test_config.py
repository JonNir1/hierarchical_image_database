"""Tests for the SimulationConfig / MDSSweepConfig dataclasses."""
import numpy as np
import pytest

from SpAM_Simulations.config import SimulationConfig, MDSSweepConfig


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
