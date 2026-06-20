"""Tests for the non-MDS pipeline functions (no R required)."""
import numpy as np
import pytest

from SpAM_Simulations.config import SimulationConfig
from SpAM_Simulations import pipeline


def _config(**over):
    base = dict(n_images=25, n_dims=4, num_subjects=[15], trials_per_subject=[6],
                images_per_trial=[6], subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1],
                reps=3, seed=7)
    base.update(over)
    return SimulationConfig(**base)


def test_generate_simulation_reproducible():
    a = pipeline.generate_simulation(_config(), verbose=False)
    b = pipeline.generate_simulation(_config(), verbose=False)
    # same seed -> identical results across the whole grid
    for params in a._results:
        for ra, rb in zip(a._results[params], b._results[params]):
            np.testing.assert_array_equal(ra.distances, rb.distances)
            np.testing.assert_array_equal(ra.num_obs, rb.num_obs)


def test_generate_simulation_grid_and_reps():
    cfg = _config()
    sim = pipeline.generate_simulation(cfg, verbose=False)
    assert len(sim._results) == len(cfg.param_grid())  # 2 configurations
    assert all(len(v) == cfg.reps for v in sim._results.values())


def test_generate_simulation_from_embeddings():
    emb = np.random.default_rng(0).random((20, 3)).astype(np.float32)
    cfg = SimulationConfig(gt_embeddings=emb, num_subjects=[10], trials_per_subject=[5],
                           images_per_trial=[5], subjects_noise_scale=[0.3], subjects_noise_df=[1],
                           reps=2, seed=1)
    sim = pipeline.generate_simulation(cfg, verbose=False)
    assert sim.num_images == 20 and sim.gt_dimensions == 3


def test_coverage_table_shape_and_columns():
    cfg = _config()
    sim = pipeline.generate_simulation(cfg, verbose=False)
    df = pipeline.compute_coverage_table(sim)
    assert len(df) == len(cfg.param_grid()) * cfg.reps  # one row per (config, rep)
    for col in ["num_subjects", "rep", "num_images", "pair_coverage", "num_connected_components"]:
        assert col in df.columns


def test_stability_table_pairs():
    cfg = _config(reps=3)
    sim = pipeline.generate_simulation(cfg, verbose=False)
    df = pipeline.compute_stability_table(sim)
    # C(3,2)=3 rep-pairs per configuration
    assert len(df) == len(cfg.param_grid()) * 3
    assert {"rep_i", "rep_j", "spearman"} <= set(df.columns)
    # noisy configs should have finite, high correlations
    noisy = df[df["subjects_noise_scale"] == 0.5]["spearman"].dropna()
    assert (noisy > 0.5).all()
