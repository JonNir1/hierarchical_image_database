"""Reusable orchestration for SpAM simulations and MDS sweeps.

This module lifts the heavy compute logic out of ``evaluation.ipynb`` so a new study can be
run from a config instead of by editing notebook cells. The notebook keeps only the plotting.

Functions:
* ``generate_simulation``        - build a Simulation from a ``SimulationConfig``.
* ``compute_coverage_table``     - per-run coverage metrics as a tidy DataFrame.
* ``compute_stability_table``    - pre-MDS pairwise Spearman reliability between repetitions.
* ``run_mds_sweep``              - fit MDS across configurations/reps/dimensions, streaming
                                   results to a ``ResultStore`` (resumable).
* ``compute_embedding_stability``- post-MDS Spearman agreement of reconstructed distances
                                   across repetitions.

``run_mds`` (and therefore R) is imported lazily inside ``run_mds_sweep`` so the rest of the
module is usable without an R/smacof installation.
"""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from SpAM_Simulations.config import SimulationConfig, MDSSweepConfig
from SpAM_Simulations.experiment import ExperimentParameters, ExperimentResults
from SpAM_Simulations.metrics import coverage, spearman_correlation, _calculate_mean_distances
from SpAM_Simulations.simulation import Simulation
from SpAM_Simulations.storage import ResultStore

logger = logging.getLogger(__name__)

_PARAM_FIELDS: List[str] = list(ExperimentParameters._fields)
_SWEEP_META_COLUMNS: List[str] = _PARAM_FIELDS + ["rep", "ndim", "niter", "stress", "status"]
_SUCCESS_STATUSES = ("success", "max_iters")


# --------------------------------------------------------------------------- generation
def generate_simulation(config: SimulationConfig, verbose: bool = True) -> Simulation:
    """Build a Simulation (random or real-data ground truth) and run its full parameter grid.

    Uses the bit-exact serial path: experiments are run sequentially through the simulation's
    seeded Generator, so the result is fully reproducible from ``config.seed``.
    """
    if config.uses_random_ground_truth:
        sim = Simulation.make(config.n_images, config.n_dims, config.seed)
    else:
        sim = Simulation.from_embeddings(config.gt_embeddings, config.seed)
    schedule = config.param_grid() * config.reps
    for params in tqdm(schedule, desc="Running experiments", disable=not verbose):
        sim.run_experiment(params, verbose=False)
    return sim


# --------------------------------------------------------------------------- metric tables
def compute_coverage_table(sim: Simulation) -> pd.DataFrame:
    """One row per (configuration, repetition) with all coverage metrics."""
    rows = [
        {**params._asdict(), "rep": rep, **coverage(res)}
        for params, results in sim._results.items()
        for rep, res in enumerate(results)
    ]
    return pd.DataFrame(rows)


def compute_stability_table(sim: Simulation) -> pd.DataFrame:
    """Pre-MDS reliability: Spearman correlation of mean distances between every pair of reps."""
    rows = []
    for params, results in sim._results.items():
        for (i, e1), (j, e2) in combinations(enumerate(results), 2):
            try:
                corr = spearman_correlation(e1, e2)
            except ValueError:
                corr = np.nan  # too few overlapping observed pairs
            rows.append({**params._asdict(), "rep_i": i, "rep_j": j, "spearman": corr})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- MDS sweep
def mds_tasks(
    sim: Simulation, sweep_config: MDSSweepConfig
) -> Iterator[Tuple[ExperimentParameters, int, ExperimentResults, int]]:
    """Yield every (params, rep, results, ndim) MDS task implied by the sweep config."""
    dims = sweep_config.target_dims(sim.gt_dimensions)
    for params, results in sim._results.items():
        for rep, res in enumerate(results):
            for ndim in dims:
                yield params, rep, res, ndim


def _prepare_mds_inputs(res: ExperimentResults) -> Tuple[np.ndarray, np.ndarray]:
    """Mean observed distances + a 0/1 weight mask (missing pairs get zero weight)."""
    mean = _calculate_mean_distances(res)
    weights = (~np.isnan(mean)).astype(float)
    dists = np.nan_to_num(mean, nan=0.0)
    return dists, weights


def _task_key(params: ExperimentParameters, rep: int, ndim: int) -> tuple:
    return (*(float(v) for v in params), int(rep), int(ndim))


def _completed_keys(store: ResultStore) -> set:
    df = store.metadata()
    keys = set()
    for _, row in df.iterrows():
        params = ExperimentParameters(*(row[f] for f in _PARAM_FIELDS))
        keys.add(_task_key(params, int(row["rep"]), int(row["ndim"])))
    return keys


def _build_mds_payload(task, sweep_config: MDSSweepConfig) -> tuple:
    """Build a small, picklable payload for one MDS task (for serial or worker execution)."""
    params, rep, res, ndim = task
    dists, weights = _prepare_mds_inputs(res)
    meta_base = {**params._asdict(), "rep": rep, "ndim": ndim}
    return (meta_base, dists, weights, ndim,
            sweep_config.max_iters, sweep_config.convergence_tol, sweep_config.precalc_init)


def _execute_mds_payload(payload: tuple) -> Tuple[dict, Optional[np.ndarray]]:
    """Run one MDS payload, returning (metadata, confdist-or-None). Failures are recorded, not raised.

    Module-level and picklable so it can run in a joblib worker process. ``run_mds`` (and R)
    is imported here so each worker initialises its own R instance exactly once.
    """
    from SpAM_Simulations.multi_dimensional_scaling import run_mds
    meta_base, dists, weights, ndim, max_iters, tol, precalc = payload
    meta = {**meta_base, "niter": np.nan, "stress": np.nan, "status": "success"}
    try:
        out = run_mds(dists=dists, weights=weights, ndim=ndim,
                      max_iters=max_iters, convergence_tol=tol, precalc_init=precalc)
    except RuntimeError as e:
        meta["status"] = "disconnected" if "connected components" in str(e) else "error"
        logger.warning("MDS task (ndim=%s, rep=%s) failed: %s", ndim, meta_base.get("rep"), e)
        return meta, None
    meta["niter"] = float(out["niter"])
    meta["stress"] = float(out["stress"])
    meta["status"] = "max_iters" if out["needs_more_iters"] else "success"
    return meta, np.asarray(out["confdist"], dtype=np.float32)


def run_mds_sweep(
    sim: Simulation,
    sweep_config: MDSSweepConfig,
    store_path: str | Path,
    *,
    parallel: bool = False,
    n_jobs: Optional[int] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> ResultStore:
    """Fit MDS across all (configuration, repetition, dimension) tasks, streaming to disk.

    Results are appended to a ``ResultStore`` at ``store_path``. If a store already exists
    (and ``overwrite`` is False) the sweep resumes, skipping tasks already recorded.

    Set ``parallel=True`` to distribute the independent MDS runs across ``n_jobs`` processes
    (default: all cores) via joblib's loky backend; results are streamed back as each worker
    finishes, so peak memory stays bounded. MDS itself is unaffected numerically - each run is
    independent - so the parallel path produces statistically equivalent results to serial
    (exactness depends only on the solver's own init, not on the scheduling).
    """
    store_path = Path(store_path)
    confdist_len = sim.num_images * (sim.num_images - 1) // 2
    if (store_path / "store_info.json").exists() and not overwrite:
        store = ResultStore.open(store_path)
        completed = _completed_keys(store)
    else:
        store = ResultStore.create(store_path, confdist_len, _SWEEP_META_COLUMNS, overwrite=overwrite)
        completed = set()

    payloads = [
        _build_mds_payload(task, sweep_config)
        for task in mds_tasks(sim, sweep_config)
        if _task_key(task[0], task[1], task[3]) not in completed
    ]
    try:
        if parallel and payloads:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(_execute_mds_payload)(p) for p in payloads
            )
            for meta, confdist in tqdm(results, total=len(payloads), desc="Running MDS", disable=not verbose):
                store.append(meta, confdist)
        else:
            for payload in tqdm(payloads, desc="Running MDS", disable=not verbose):
                meta, confdist = _execute_mds_payload(payload)
                store.append(meta, confdist)
    finally:
        store.close()
    return store


# --------------------------------------------------------------------------- post-MDS stability
def compute_embedding_stability(
    store: ResultStore, group_fields: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Post-MDS reliability: mean Spearman agreement of reconstructed distances across reps.

    Groups successful results by ``group_fields`` (default: the parameters + ndim) and
    correlates the stored confdist vectors across repetitions within each group.
    """
    if group_fields is None:
        group_fields = _PARAM_FIELDS + ["ndim"]
    group_fields = list(group_fields)

    df = store.metadata()
    df = df[df["status"].isin(_SUCCESS_STATUSES) & (df["confdist_row"] >= 0)]

    rows = []
    for key, grp in df.groupby(group_fields):
        confdists = [store.confdist(int(r)) for r in grp["confdist_row"]]
        corrs = [spearmanr(a, b).statistic for a, b in combinations(confdists, 2)]
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(group_fields, key_tuple)),
            "n_reps": len(confdists),
            "mean_spearman": float(np.mean(corrs)) if corrs else np.nan,
            "sem_spearman": (float(np.std(corrs, ddof=1) / np.sqrt(len(corrs)))
                             if len(corrs) > 1 else np.nan),
        })
    return pd.DataFrame(rows)
