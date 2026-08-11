"""End-to-end MDS-sweep tests. Requires a working R + rpy2 + smacof.

Skipped automatically when the R bridge cannot be imported (e.g. CI without R). To run it
the R environment must be discoverable; see the repo notes on setting R_HOME / R_LIBS_USER
and giving the Python process a clean Windows PATH so rpy2 falls back to bin/x64.
"""
import numpy as np
import pytest

# rpy2's R initialisation can fail with errors other than ImportError when R is misconfigured.
try:
    from SpAM_Simulations.core.multi_dimensional_scaling import run_mds  # noqa: F401
    _MDS_AVAILABLE = True
    _MDS_SKIP_REASON = ""
except (ImportError, IndexError, RuntimeError, OSError, AttributeError) as e:
    _MDS_AVAILABLE = False
    _MDS_SKIP_REASON = f"R/rpy2/smacof unavailable: {type(e).__name__}: {e}"

pytestmark = pytest.mark.skipif(not _MDS_AVAILABLE, reason=_MDS_SKIP_REASON)

from SpAM_Simulations.core.config import SimulationConfig, MDSSweepConfig
from SpAM_Simulations.core import pipeline
from SpAM_Simulations.core.storage import ResultStore


def _small_sim():
    cfg = SimulationConfig(
        n_images=30, n_dims=4, num_subjects=[40], trials_per_subject=[10],
        images_per_trial=[12], subjects_noise_scale=[0.3], subjects_noise_df=[1],
        reps=3, seed=5,
    )
    return pipeline.generate_simulation(cfg, verbose=False), cfg


def test_run_mds_sweep_and_metadata(tmp_path):
    sim, _ = _small_sim()
    sweep = MDSSweepConfig(ndims=[3, 4], max_iters=150, convergence_tol=1e-5, precalc_init=False)
    store = pipeline.run_mds_sweep(sim, sweep, tmp_path / "store", verbose=False)

    df = store.metadata()
    # 1 config * 3 reps * 2 dims = 6 tasks
    assert len(df) == 6
    assert set(df["ndim"]) == {3, 4}
    assert (df["status"].isin(["success", "max_iters"])).all()
    # successful runs carry a confdist of the right length
    L = sim.num_images * (sim.num_images - 1) // 2
    for row in df.itertuples():
        if row.confdist_row >= 0:
            assert store.confdist(int(row.confdist_row)).shape == (L,)


def test_embedding_stability(tmp_path):
    sim, _ = _small_sim()
    sweep = MDSSweepConfig(ndims=[4], max_iters=200, convergence_tol=1e-5, precalc_init=False)
    store = pipeline.run_mds_sweep(sim, sweep, tmp_path / "store", verbose=False)
    stab = pipeline.compute_embedding_stability(store)
    assert len(stab) == 1  # one (config, ndim) group
    row = stab.iloc[0]
    assert row["n_reps"] == 3
    # recovering the same GT at the true dimensionality => high cross-rep agreement
    assert row["mean_spearman"] > 0.5


def test_run_mds_sweep_parallel(tmp_path):
    sim, _ = _small_sim()
    sweep = MDSSweepConfig(ndims=[3, 4], max_iters=150, convergence_tol=1e-5, precalc_init=False)
    store = pipeline.run_mds_sweep(sim, sweep, tmp_path / "store", parallel=True, n_jobs=2, verbose=False)
    df = store.metadata()
    assert len(df) == 6  # same task set as the serial sweep
    assert set(df["ndim"]) == {3, 4}
    assert df["status"].isin(["success", "max_iters"]).all()
    # every successful run stored a confdist of the right length
    L = sim.num_images * (sim.num_images - 1) // 2
    assert all(store.confdist(int(r)).shape == (L,) for r in df["confdist_row"] if r >= 0)


def test_sweep_resumes(tmp_path):
    sim, _ = _small_sim()
    store_path = tmp_path / "store"
    # first pass: only ndim=3
    pipeline.run_mds_sweep(sim, MDSSweepConfig(ndims=[3], max_iters=100), store_path, verbose=False)
    n_after_first = len(ResultStore.open(store_path))
    assert n_after_first == 3
    # second pass over a superset (3 and 4): resumes, only adds the 3 new ndim=4 tasks
    store = pipeline.run_mds_sweep(sim, MDSSweepConfig(ndims=[3, 4], max_iters=100), store_path, verbose=False)
    df = store.metadata()
    assert len(df) == 6
    assert sorted(df["ndim"]) == [3, 3, 3, 4, 4, 4]  # no duplicate ndim=3 tasks
