"""Tests for the non-MDS pipeline functions (no R required)."""
import numpy as np
import pytest

from SpAM_Simulations.config import SimulationConfig, RealisticSimulationConfig
from SpAM_Simulations import pipeline


def _config(**over):
    base = dict(n_images=25, n_dims=4, num_subjects=[15], trials_per_subject=[6],
                images_per_trial=[6], subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1],
                reps=3, seed=7)
    base.update(over)
    return SimulationConfig(**base)


def _realistic_config(**over):
    base = dict(n_images=120, n_dims=4, num_subjects=[15], trials_per_subject=[6],
                images_per_trial=[6], subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1],
                frac_images_repeated=[1 / 3], reps=3, seed=7)
    base.update(over)
    return RealisticSimulationConfig(**base)


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


def test_run_mds_sweep_streams_payloads_lazily(tmp_path, monkeypatch):
    """Payloads (each a full dists+weights pair) must be built on demand and interleaved with
    execution - not all materialised up front, which for a large sweep would need tens of GB."""
    from SpAM_Simulations.config import MDSSweepConfig
    cfg = SimulationConfig(n_images=30, n_dims=3, num_subjects=[10], trials_per_subject=[5],
                           images_per_trial=[6], subjects_noise_scale=[0.5], subjects_noise_df=[1],
                           reps=3, seed=1)
    sim = pipeline.generate_simulation(cfg, verbose=False)
    sweep = MDSSweepConfig(ndims=[2, 3])
    L = sim.num_images * (sim.num_images - 1) // 2
    n_tasks = 1 * 3 * 2  # 1 config * 3 reps * 2 dims

    events = []
    real_build = pipeline._build_mds_payload

    def spy_build(task, sweep_config):
        events.append("build")
        return real_build(task, sweep_config)

    def fake_exec(payload):
        events.append("exec")
        meta = {**payload[0], "niter": 1.0, "stress": 0.0, "status": "success"}
        return meta, np.zeros(L, np.float32)

    monkeypatch.setattr(pipeline, "_build_mds_payload", spy_build)
    monkeypatch.setattr(pipeline, "_execute_mds_payload", fake_exec)

    store = pipeline.run_mds_sweep(sim, sweep, tmp_path / "s", verbose=False)

    assert events.count("build") == n_tasks
    assert events.count("exec") == n_tasks
    # streaming => build/exec alternate; the eager-list bug would emit all builds before any exec
    assert events[:4] == ["build", "exec", "build", "exec"]
    assert len(store) == n_tasks


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


# --------------------------------------------------------------------- realistic simulation
def test_generate_realistic_simulation_reproducible():
    a = pipeline.generate_realistic_simulation(_realistic_config(), verbose=False)
    b = pipeline.generate_realistic_simulation(_realistic_config(), verbose=False)
    for params in a._results:
        for ra, rb in zip(a._results[params], b._results[params]):
            np.testing.assert_array_equal(ra.distances, rb.distances)
            np.testing.assert_array_equal(ra.num_obs, rb.num_obs)
            np.testing.assert_array_equal(ra.subject_snr, rb.subject_snr)


def test_generate_realistic_simulation_grid_and_reps():
    cfg = _realistic_config()
    sim = pipeline.generate_realistic_simulation(cfg, verbose=False)
    assert len(sim._results) == len(cfg.param_grid())  # 2 configurations
    assert all(len(v) == cfg.reps for v in sim._results.values())


def test_realistic_coverage_table_includes_snr_columns():
    cfg = _realistic_config()
    sim = pipeline.generate_realistic_simulation(cfg, verbose=False)
    df = pipeline.compute_coverage_table(sim)
    assert len(df) == len(cfg.param_grid()) * cfg.reps
    for col in ["num_subjects", "rep", "pair_coverage", "mean_snr", "median_snr", "frac_nan_snr"]:
        assert col in df.columns
    # the old (non-realistic) coverage table must not gain these columns
    old_df = pipeline.compute_coverage_table(pipeline.generate_simulation(_config(), verbose=False))
    assert "mean_snr" not in old_df.columns


def test_realistic_run_mds_sweep_streams_payloads_lazily(tmp_path, monkeypatch):
    """Same lazy-streaming contract as the old simulation, but exercising the dynamically
    derived (rather than hardcoded) parameter fields used to build/read the store."""
    from SpAM_Simulations.config import MDSSweepConfig
    cfg = _realistic_config(n_images=30, n_dims=3, num_subjects=[10], trials_per_subject=[5],
                             images_per_trial=[6], subjects_noise_scale=[0.5],
                             subjects_noise_df=[1], frac_images_repeated=[1 / 3], reps=3, seed=1)
    sim = pipeline.generate_realistic_simulation(cfg, verbose=False)
    sweep = MDSSweepConfig(ndims=[2, 3])
    L = sim.num_images * (sim.num_images - 1) // 2
    n_tasks = 1 * 3 * 2  # 1 config * 3 reps * 2 dims

    events = []
    real_build = pipeline._build_mds_payload

    def spy_build(task, sweep_config):
        events.append("build")
        return real_build(task, sweep_config)

    def fake_exec(payload):
        events.append("exec")
        meta = {**payload[0], "niter": 1.0, "stress": 0.0, "status": "success"}
        return meta, np.zeros(L, np.float32)

    monkeypatch.setattr(pipeline, "_build_mds_payload", spy_build)
    monkeypatch.setattr(pipeline, "_execute_mds_payload", fake_exec)

    store = pipeline.run_mds_sweep(sim, sweep, tmp_path / "s", verbose=False)

    assert events.count("build") == n_tasks
    assert events.count("exec") == n_tasks
    assert events[:4] == ["build", "exec", "build", "exec"]
    assert len(store) == n_tasks
    # the realistic params' fields (including frac_images_repeated) must be in the metadata
    assert "frac_images_repeated" in store.metadata().columns
