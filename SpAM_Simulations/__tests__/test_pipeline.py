"""Tests for the non-MDS pipeline functions (no R required)."""
import numpy as np
import pytest

from SpAM_Simulations.config import (
    SimulationConfig, TaskV2_3SimulationConfig, TaskV2_4SimulationConfig, TaskV3SimulationConfig
)
from SpAM_Simulations import pipeline


def _task_v3_config(**over):
    base = dict(n_images=120, n_dims=4, num_subjects=[15], trials_per_subject=[8],
                images_per_trial=[6], subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1],
                frac_trials_repeated=[0.0, 0.25], perspective_dispersion=[0.0, 0.3],
                use_isotropic=True, reps=3, seed=7)
    base.update(over)
    return TaskV3SimulationConfig(**base)


def _config(**over):
    base = dict(n_images=25, n_dims=4, num_subjects=[15], trials_per_subject=[6],
                images_per_trial=[6], subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1],
                reps=3, seed=7)
    base.update(over)
    return SimulationConfig(**base)


def _task_v2_3_config(**over):
    base = dict(n_images=120, n_dims=4, num_subjects=[15], trials_per_subject=[6],
                images_per_trial=[6], subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1],
                frac_images_repeated=[1 / 3], reps=3, seed=7)
    base.update(over)
    return TaskV2_3SimulationConfig(**base)


def _task_v2_4_config(**over):
    # frac_images_repeated fixed at 0.0 so singles-only trials exist for the whole-trial repeats
    # (the two levers compete - see task_v2_4_experiment).
    base = dict(n_images=120, n_dims=4, num_subjects=[15], trials_per_subject=[8],
                images_per_trial=[6], subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1],
                frac_images_repeated=[0.0], frac_trials_repeated=[0.0, 0.25], reps=3, seed=7)
    base.update(over)
    return TaskV2_4SimulationConfig(**base)


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


# --------------------------------------------------------------------- task-v2.3 simulation
def test_generate_task_v2_3_simulation_reproducible():
    a = pipeline.generate_task_v2_3_simulation(_task_v2_3_config(), verbose=False)
    b = pipeline.generate_task_v2_3_simulation(_task_v2_3_config(), verbose=False)
    for params in a._results:
        for ra, rb in zip(a._results[params], b._results[params]):
            np.testing.assert_array_equal(ra.distances, rb.distances)
            np.testing.assert_array_equal(ra.num_obs, rb.num_obs)
            np.testing.assert_array_equal(ra.subject_snr, rb.subject_snr)


def test_generate_task_v2_3_simulation_grid_and_reps():
    cfg = _task_v2_3_config()
    sim = pipeline.generate_task_v2_3_simulation(cfg, verbose=False)
    assert len(sim._results) == len(cfg.param_grid())  # 2 configurations
    assert all(len(v) == cfg.reps for v in sim._results.values())


def test_task_v2_3_coverage_table_includes_snr_columns():
    cfg = _task_v2_3_config()
    sim = pipeline.generate_task_v2_3_simulation(cfg, verbose=False)
    df = pipeline.compute_coverage_table(sim)
    assert len(df) == len(cfg.param_grid()) * cfg.reps
    for col in ["num_subjects", "rep", "pair_coverage", "mean_snr", "median_snr", "frac_nan_snr"]:
        assert col in df.columns
    # the task-v0.1 coverage table must not gain these columns
    old_df = pipeline.compute_coverage_table(pipeline.generate_simulation(_config(), verbose=False))
    assert "mean_snr" not in old_df.columns


def test_task_v2_3_run_mds_sweep_streams_payloads_lazily(tmp_path, monkeypatch):
    """Same lazy-streaming contract as the task-v0.1 simulation, but exercising the dynamically
    derived (rather than hardcoded) parameter fields used to build/read the store."""
    from SpAM_Simulations.config import MDSSweepConfig
    cfg = _task_v2_3_config(n_images=30, n_dims=3, num_subjects=[10], trials_per_subject=[5],
                             images_per_trial=[6], subjects_noise_scale=[0.5],
                             subjects_noise_df=[1], frac_images_repeated=[1 / 3], reps=3, seed=1)
    sim = pipeline.generate_task_v2_3_simulation(cfg, verbose=False)
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
    # the task-v2.3 params' fields (including frac_images_repeated) must be in the metadata
    assert "frac_images_repeated" in store.metadata().columns


# --------------------------------------------------------------------- task-v2.4 simulation
def test_generate_task_v2_4_simulation_reproducible():
    a = pipeline.generate_task_v2_4_simulation(_task_v2_4_config(), verbose=False)
    b = pipeline.generate_task_v2_4_simulation(_task_v2_4_config(), verbose=False)
    for params in a._results:
        for ra, rb in zip(a._results[params], b._results[params]):
            np.testing.assert_array_equal(ra.distances, rb.distances)
            np.testing.assert_array_equal(ra.num_obs, rb.num_obs)
            np.testing.assert_array_equal(ra.subject_test_retest, rb.subject_test_retest)


def test_generate_task_v2_4_simulation_grid_and_reps():
    cfg = _task_v2_4_config()
    sim = pipeline.generate_task_v2_4_simulation(cfg, verbose=False)
    assert len(sim._results) == len(cfg.param_grid())  # 4 configurations
    assert all(len(v) == cfg.reps for v in sim._results.values())


def test_task_v2_4_coverage_table_includes_snr_and_test_retest_columns():
    cfg = _task_v2_4_config()
    sim = pipeline.generate_task_v2_4_simulation(cfg, verbose=False)
    df = pipeline.compute_coverage_table(sim)
    assert len(df) == len(cfg.param_grid()) * cfg.reps
    for col in ["num_subjects", "rep", "pair_coverage",
                "mean_snr", "frac_nan_snr",
                "mean_test_retest", "median_test_retest", "frac_nan_test_retest"]:
        assert col in df.columns
    # the frac_trials_repeated == 0 slice has no repeats -> reliability undefined for all subjects
    no_repeat = df[df["frac_trials_repeated"] == 0.0]
    assert (no_repeat["frac_nan_test_retest"] == 1.0).all()
    # the task-v0.1 coverage table must not gain the test-retest columns
    old_df = pipeline.compute_coverage_table(pipeline.generate_simulation(_config(), verbose=False))
    assert "mean_test_retest" not in old_df.columns


def test_task_v2_4_run_mds_sweep_store_roundtrips_seven_field_params(tmp_path, monkeypatch):
    """The store's metadata columns and completed-key roundtrip are derived from the params
    type, so they must handle the task-v2.4 tuple's extra frac_trials_repeated field."""
    from SpAM_Simulations.config import MDSSweepConfig
    cfg = _task_v2_4_config(n_images=60, n_dims=3, num_subjects=[10], trials_per_subject=[8],
                            images_per_trial=[6], subjects_noise_scale=[0.5], subjects_noise_df=[1],
                            frac_images_repeated=[0.0], frac_trials_repeated=[0.0, 0.25], reps=2, seed=1)
    sim = pipeline.generate_task_v2_4_simulation(cfg, verbose=False)
    sweep = MDSSweepConfig(ndims=[2, 3])
    L = sim.num_images * (sim.num_images - 1) // 2
    n_tasks = 2 * 2 * 2  # 2 configs * 2 reps * 2 dims

    def fake_exec(payload):
        meta = {**payload[0], "niter": 1.0, "stress": 0.0, "status": "success"}
        return meta, np.zeros(L, np.float32)

    monkeypatch.setattr(pipeline, "_execute_mds_payload", fake_exec)

    store = pipeline.run_mds_sweep(sim, sweep, tmp_path / "s", verbose=False)
    assert len(store) == n_tasks
    assert "frac_trials_repeated" in store.metadata().columns
    # resuming finds every task already complete (the 7-field key roundtrips through the store)
    store2 = pipeline.run_mds_sweep(sim, sweep, tmp_path / "s", verbose=False)
    assert len(store2) == n_tasks


# --------------------------------------------------------------------- task-v3 simulation
def test_generate_task_v3_simulation_reproducible():
    a = pipeline.generate_task_v3_simulation(_task_v3_config(), verbose=False)
    b = pipeline.generate_task_v3_simulation(_task_v3_config(), verbose=False)
    for params in a._results:
        for ra, rb in zip(a._results[params], b._results[params]):
            np.testing.assert_array_equal(np.nan_to_num(ra.distances), np.nan_to_num(rb.distances))
            np.testing.assert_array_equal(ra.num_obs, rb.num_obs)
            np.testing.assert_array_equal(ra.subject_test_retest, rb.subject_test_retest)


def test_generate_task_v3_simulation_grid_and_reps():
    cfg = _task_v3_config()
    sim = pipeline.generate_task_v3_simulation(cfg, verbose=False)
    assert len(sim._results) == len(cfg.param_grid())  # 2*2 = 4 configurations
    assert all(len(v) == cfg.reps for v in sim._results.values())


def test_task_v3_coverage_table_has_test_retest_not_snr():
    cfg = _task_v3_config()
    sim = pipeline.generate_task_v3_simulation(cfg, verbose=False)
    df = pipeline.compute_coverage_table(sim)
    assert len(df) == len(cfg.param_grid()) * cfg.reps
    for col in ["num_subjects", "perspective_dispersion", "pair_coverage",
                "mean_test_retest", "frac_nan_test_retest"]:
        assert col in df.columns
    assert "mean_snr" not in df.columns  # v3 drops the doubled-image SNR diagnostic
    # frac_trials_repeated == 0 -> no repeats -> reliability undefined for every subject
    no_repeat = df[df["frac_trials_repeated"] == 0.0]
    assert (no_repeat["frac_nan_test_retest"] == 1.0).all()


def test_task_v3_config_accepts_supplied_embeddings_and_pipeline_uses_them():
    # the pilot-calibrated GT is fed via gt_embeddings; generation must embed those coords, not synth
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 4)).astype(np.float32)
    cfg = TaskV3SimulationConfig(
        gt_embeddings=emb, num_subjects=[8], trials_per_subject=[5],
        images_per_trial=[6], subjects_noise_scale=[0.5], subjects_noise_df=[1],
        frac_trials_repeated=[0.0], perspective_dispersion=[0.2], reps=2, seed=1,
    )
    assert not cfg.uses_random_ground_truth
    sim = pipeline.generate_task_v3_simulation(cfg, verbose=False)
    np.testing.assert_array_equal(sim.gt_embeddings, emb)  # used the supplied embedding


def test_task_v3_config_still_requires_one_gt_source():
    with pytest.raises(ValueError, match="exactly one"):
        TaskV3SimulationConfig(  # both synthetic AND embeddings -> rejected by the base class
            n_images=10, n_dims=3, gt_embeddings=np.zeros((10, 3)),
            num_subjects=[5], trials_per_subject=[4], images_per_trial=[4],
            subjects_noise_scale=[0.5], subjects_noise_df=[1],
            frac_trials_repeated=[0.0], perspective_dispersion=[0.0],
        )


def test_task_v3_run_mds_sweep_store_roundtrips_seven_field_params(tmp_path, monkeypatch):
    """The store derives its metadata columns from the params type, so it must handle the v3
    tuple's perspective_dispersion field (and not the dropped frac_images_repeated)."""
    from SpAM_Simulations.config import MDSSweepConfig
    cfg = _task_v3_config(n_images=60, n_dims=3, num_subjects=[10], trials_per_subject=[8],
                          images_per_trial=[6], subjects_noise_scale=[0.5], subjects_noise_df=[1],
                          frac_trials_repeated=[0.0, 0.25], perspective_dispersion=[0.3], reps=2, seed=1)
    sim = pipeline.generate_task_v3_simulation(cfg, verbose=False)
    sweep = MDSSweepConfig(ndims=[2, 3])
    L = sim.num_images * (sim.num_images - 1) // 2
    n_tasks = 2 * 2 * 2  # 2 configs * 2 reps * 2 dims

    def fake_exec(payload):
        meta = {**payload[0], "niter": 1.0, "stress": 0.0, "status": "success"}
        return meta, np.zeros(L, np.float32)

    monkeypatch.setattr(pipeline, "_execute_mds_payload", fake_exec)

    store = pipeline.run_mds_sweep(sim, sweep, tmp_path / "s", verbose=False)
    assert len(store) == n_tasks
    cols = store.metadata().columns
    assert "perspective_dispersion" in cols and "frac_images_repeated" not in cols


class TestTopKSimilarJaccard:
    def test_identical_vectors_give_one(self):
        rng = np.random.default_rng(0)
        v = rng.random(200)
        assert pipeline._topk_similar_jaccard(v, v.copy(), 0.1) == 1.0

    def test_reversed_ranking_gives_zero(self):
        # smallest of v are the largest of -v -> the top-k closest sets are disjoint
        v = np.arange(200, dtype=float)
        assert pipeline._topk_similar_jaccard(v, -v, 0.25) == 0.0

    def test_known_overlap(self):
        # a's 2 smallest = idx {0,1}; b's 2 smallest = idx {1,2}; Jaccard = 1/3
        a = np.array([0.0, 1.0, 2.0, 3.0])
        b = np.array([3.0, 0.0, 1.0, 2.0])
        assert pipeline._topk_similar_jaccard(a, b, 0.5) == pytest.approx(1 / 3)

    def test_frac_rounds_to_at_least_one(self):
        v = np.arange(10, dtype=float)
        assert pipeline._topk_similar_jaccard(v, v.copy(), 0.001) == 1.0   # k floored to 1
