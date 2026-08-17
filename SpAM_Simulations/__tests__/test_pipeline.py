"""Tests for the non-MDS pipeline functions (no R required)."""
import numpy as np
import pytest

from SpAM_Simulations.core.config import (
    SimulationConfig, TaskV2_3SimulationConfig, TaskV2_4SimulationConfig, TaskV3SimulationConfig
)
from SpAM_Simulations.core import pipeline


def _fake_conf(sim, payload):
    """A stand-in MDS configuration for a monkeypatched `_execute_mds_payload`.

    Shaped `(n_images, ndim)` from the payload's own `ndim` so the store's shape validation and
    per-record padding are exercised; the values themselves are irrelevant to these tests.
    """
    return np.zeros((sim.num_images, int(payload[0]["ndim"])), np.float32)


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
    from SpAM_Simulations.core.config import MDSSweepConfig
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
        return meta, np.zeros(L, np.float32), _fake_conf(sim, payload)

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
    from SpAM_Simulations.core.config import MDSSweepConfig
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
        return meta, np.zeros(L, np.float32), _fake_conf(sim, payload)

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
    from SpAM_Simulations.core.config import MDSSweepConfig
    cfg = _task_v2_4_config(n_images=60, n_dims=3, num_subjects=[10], trials_per_subject=[8],
                            images_per_trial=[6], subjects_noise_scale=[0.5], subjects_noise_df=[1],
                            frac_images_repeated=[0.0], frac_trials_repeated=[0.0, 0.25], reps=2, seed=1)
    sim = pipeline.generate_task_v2_4_simulation(cfg, verbose=False)
    sweep = MDSSweepConfig(ndims=[2, 3])
    L = sim.num_images * (sim.num_images - 1) // 2
    n_tasks = 2 * 2 * 2  # 2 configs * 2 reps * 2 dims

    def fake_exec(payload):
        meta = {**payload[0], "niter": 1.0, "stress": 0.0, "status": "success"}
        return meta, np.zeros(L, np.float32), _fake_conf(sim, payload)

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
    from SpAM_Simulations.core.config import MDSSweepConfig
    cfg = _task_v3_config(n_images=60, n_dims=3, num_subjects=[10], trials_per_subject=[8],
                          images_per_trial=[6], subjects_noise_scale=[0.5], subjects_noise_df=[1],
                          frac_trials_repeated=[0.0, 0.25], perspective_dispersion=[0.3], reps=2, seed=1)
    sim = pipeline.generate_task_v3_simulation(cfg, verbose=False)
    sweep = MDSSweepConfig(ndims=[2, 3])
    L = sim.num_images * (sim.num_images - 1) // 2
    n_tasks = 2 * 2 * 2  # 2 configs * 2 reps * 2 dims

    def fake_exec(payload):
        meta = {**payload[0], "niter": 1.0, "stress": 0.0, "status": "success"}
        return meta, np.zeros(L, np.float32), _fake_conf(sim, payload)

    monkeypatch.setattr(pipeline, "_execute_mds_payload", fake_exec)

    store = pipeline.run_mds_sweep(sim, sweep, tmp_path / "s", verbose=False)
    assert len(store) == n_tasks
    cols = store.metadata().columns
    assert "perspective_dispersion" in cols and "frac_images_repeated" not in cols


# ------------------------------------------------------- configuration-space generalizability

def _conf_store(tmp_path, confs, n_images, max_ndim, ndim=None, statuses=None):
    """Build a ResultStore holding the given configurations, one per rep, in one config group."""
    from SpAM_Simulations.core.storage import ResultStore
    ndim = ndim if ndim is not None else max_ndim
    L = n_images * (n_images - 1) // 2
    cols = ["num_subjects", "rep", "ndim", "niter", "stress", "status"]
    store = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=cols,
                               n_images=n_images, max_ndim=max_ndim)
    for rep, conf in enumerate(confs):
        status = statuses[rep] if statuses else "success"
        meta = dict(num_subjects=10, rep=rep, ndim=ndim, niter=1.0, stress=0.0, status=status)
        keep = status in ("success", "max_iters")
        store.append(meta, np.zeros(L, np.float32) if keep else None,
                     np.asarray(conf, np.float32) if keep else None)
    store.close()
    return ResultStore.open(tmp_path / "s")


class TestEmbeddingGeneralizability:
    N, D = 12, 3

    def _random_conf(self, seed=0):
        return np.random.default_rng(seed).normal(size=(self.N, self.D))

    def test_identical_configurations_have_zero_disparity(self, tmp_path):
        c = self._random_conf()
        store = _conf_store(tmp_path, [c, c], self.N, self.D)
        out = pipeline.compute_embedding_generalizability(store, verbose=False)
        assert out["mean_procrustes_m2"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("transform", ["rotate", "reflect", "scale", "translate"])
    def test_disparity_is_invariant_to_mds_gauge_freedom(self, tmp_path, transform):
        """Position, scale, rotation and reflection are arbitrary in an MDS solution, so a
        configuration transformed by any of them is the SAME space and must score 0."""
        c = self._random_conf()
        if transform == "rotate":
            theta = 0.7
            rot = np.eye(self.D)
            rot[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            other = c @ rot
        elif transform == "reflect":
            other = c * np.array([-1.0] + [1.0] * (self.D - 1))
        elif transform == "scale":
            other = c * 4.2
        else:
            other = c + np.array([3.0] * self.D)
        store = _conf_store(tmp_path, [c, other], self.N, self.D)
        out = pipeline.compute_embedding_generalizability(store, verbose=False)
        assert out["mean_procrustes_m2"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_disparity_rises_with_divergence(self, tmp_path):
        c = self._random_conf()
        rng = np.random.default_rng(99)
        scores = []
        for i, jitter in enumerate((0.05, 0.5)):
            store = _conf_store(tmp_path / f"run{i}", [c, c + rng.normal(0, jitter, c.shape)],
                                self.N, self.D)
            scores.append(pipeline.compute_embedding_generalizability(store, verbose=False)
                          ["mean_procrustes_m2"].iloc[0])
        assert scores[0] < scores[1]

    def test_disparity_is_bounded_in_unit_interval(self, tmp_path):
        store = _conf_store(tmp_path, [self._random_conf(1), self._random_conf(2)], self.N, self.D)
        m2 = pipeline.compute_embedding_generalizability(store, verbose=False)["mean_procrustes_m2"].iloc[0]
        assert 0.0 <= m2 <= 1.0

    def test_padded_rows_are_trimmed_to_ndim(self, tmp_path):
        """A 2-D fit stored in a max_ndim=5 store must be compared as 2-D, not against zero pad."""
        c = np.random.default_rng(0).normal(size=(self.N, 2))
        store = _conf_store(tmp_path, [c, c], self.N, max_ndim=5, ndim=2)
        out = pipeline.compute_embedding_generalizability(store, verbose=False)
        assert out["mean_procrustes_m2"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_failed_runs_are_excluded(self, tmp_path):
        c = self._random_conf()
        store = _conf_store(tmp_path, [c, c, None], self.N, self.D,
                            statuses=["success", "success", "error"])
        out = pipeline.compute_embedding_generalizability(store, verbose=False)
        assert out["n_reps"].iloc[0] == 2

    def test_store_without_configurations_raises_clearly(self, tmp_path):
        from SpAM_Simulations.core.storage import ResultStore
        L = self.N * (self.N - 1) // 2
        cols = ["num_subjects", "rep", "ndim", "niter", "stress", "status"]
        s = ResultStore.create(tmp_path / "s", confdist_len=L, meta_columns=cols)
        s.append(dict(num_subjects=10, rep=0, ndim=3, niter=1.0, stress=0.0, status="success"),
                 np.zeros(L, np.float32))
        s.close()
        with pytest.raises(ValueError, match="no MDS configurations"):
            pipeline.compute_embedding_generalizability(ResultStore.open(tmp_path / "s"), verbose=False)


class TestItemGeneralizability:
    N, D = 12, 3

    def test_one_row_per_image(self, tmp_path):
        rng = np.random.default_rng(0)
        c = rng.normal(size=(self.N, self.D))
        store = _conf_store(tmp_path, [c, c + rng.normal(0, 0.2, c.shape)], self.N, self.D)
        out = pipeline.compute_item_generalizability(store, verbose=False)
        assert len(out) == self.N
        assert sorted(out["image_index"]) == list(range(self.N))

    def test_identical_configurations_give_zero_residuals(self, tmp_path):
        c = np.random.default_rng(0).normal(size=(self.N, self.D))
        store = _conf_store(tmp_path, [c, c], self.N, self.D)
        out = pipeline.compute_item_generalizability(store, verbose=False)
        np.testing.assert_allclose(out["mean_residual"], 0.0, atol=1e-8)

    @pytest.mark.parametrize("displacement", [0.3, 1.0, 3.0])
    def test_a_displaced_image_gets_the_largest_residual(self, tmp_path, displacement):
        """The point of the per-item table: locate WHICH stimuli fail to generalise."""
        c = np.random.default_rng(0).normal(size=(self.N, self.D))
        other = c.copy()
        other[7] += displacement
        store = _conf_store(tmp_path / f"d{displacement}", [c, other], self.N, self.D)
        out = pipeline.compute_item_generalizability(store, verbose=False)
        assert out.loc[out["mean_residual"].idxmax(), "image_index"] == 7

    def test_a_gross_outlier_smears_residuals_across_all_items(self, tmp_path):
        """Documents the limitation stated in the docstring rather than hiding it.

        Procrustes is a *global* fit that scales both configurations to unit norm, so one grossly
        displaced item dominates that norm and distorts the alignment for everything else: the
        residual is then no longer attributable to the item that actually moved. Here the
        displacement drives M^2 past ~0.5 (two barely-related spaces) and the argmax leaves 7.
        Cohort pairs in the real sweep sit far below that, so the per-item table stays usable -
        but it must be read alongside the group-level M^2, not on its own.
        """
        c = np.random.default_rng(0).normal(size=(self.N, self.D))
        other = c.copy()
        other[7] += 5.0
        store = _conf_store(tmp_path, [c, other], self.N, self.D)
        group_m2 = pipeline.compute_embedding_generalizability(store, verbose=False)
        out = pipeline.compute_item_generalizability(store, verbose=False)
        assert group_m2["mean_procrustes_m2"].iloc[0] > 0.5      # the spaces barely relate
        assert out.loc[out["mean_residual"].idxmax(), "image_index"] != 7


# --------------------------------------------------------------------- resume round-trip
# Regression on a bug that silently re-ran a finished sweep. pandas' default C parser is not
# round-trip exact for float64: csv writes 2/14 as 0.14285714285714285 and the default parser
# reads back 0.1428571428571428. Sweep resume rebuilds each completed key from those columns, so
# one differing last digit made every key miss and `completed` came back empty.

def test_metadata_reads_floats_back_exactly(tmp_path):
    """The read side: a float written by `csv` must survive the round trip through pandas."""
    from SpAM_Simulations.core.storage import ResultStore

    awkward = 2 / 14                      # 0.14285714285714285 - needs 17 significant digits
    cols = ["frac_trials_repeated", "rep", "ndim", "niter", "stress", "status"]
    store = ResultStore.create(tmp_path / "s", 10, cols)
    store.append({"frac_trials_repeated": awkward, "rep": 0, "ndim": 2,
                  "niter": 1, "stress": 0.1, "status": "success"},
                 confdist=np.zeros(10, dtype=np.float32))
    store.close()
    got = ResultStore.open(tmp_path / "s").metadata()["frac_trials_repeated"][0]
    assert got == awkward, f"{got!r} != {awkward!r}"


def test_task_key_is_insensitive_to_a_last_digit_difference():
    """The key side: rounding makes the comparison robust however the float reached disk."""
    from SpAM_Simulations.core.pipeline import _task_key

    from typing import NamedTuple
    Params = NamedTuple("Params", [("a", float), ("b", float)])
    exact = _task_key(Params(2 / 14, 0.3), rep=1, ndim=8)
    lossy = _task_key(Params(0.1428571428571428, 0.3), rep=1, ndim=8)
    assert exact == lossy


def test_a_resumed_sweep_skips_work_already_in_the_store(tmp_path):
    """End to end: the property the bug destroyed.

    `frac_trials_repeated=2/14` is exactly the value the deployed design uses, and exactly the one
    that failed to round-trip - so this is the real configuration, not a contrived float.
    """
    from SpAM_Simulations.core.config import MDSSweepConfig, SimulationConfig
    from SpAM_Simulations.core.pipeline import _completed_keys, _param_type, _task_key, mds_tasks
    from SpAM_Simulations.core.storage import ResultStore

    cfg = SimulationConfig(n_images=25, n_dims=3, num_subjects=[8], trials_per_subject=[4],
                           images_per_trial=[6], subjects_noise_scale=[2 / 14],
                           subjects_noise_df=[5], reps=1, seed=0)
    sim = pipeline.generate_simulation(cfg, verbose=False)
    pt = _param_type(sim)
    sweep = MDSSweepConfig(ndims=[2], max_iters=5, precalc_init=False)
    task = next(iter(mds_tasks(sim, sweep)))
    n_pairs = sim.num_images * (sim.num_images - 1) // 2

    store = ResultStore.create(
        tmp_path / "s", n_pairs,
        list(pt._fields) + ["rep", "ndim", "niter", "stress", "status"])
    store.append({**task[0]._asdict(), "rep": task[1], "ndim": 2, "niter": 3,
                  "stress": 0.1, "status": "success"},
                 confdist=np.zeros(n_pairs, dtype=np.float32))
    store.close()

    completed = _completed_keys(ResultStore.open(tmp_path / "s"), pt)
    assert _task_key(task[0], task[1], task[3]) in completed, "a resumed sweep would redo this fit"
