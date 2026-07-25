"""Reusable orchestration for SpAM simulations and MDS sweeps.

This module lifts the heavy compute logic out of ``evaluation_v0_1.ipynb`` so a new study can be
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
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from SpAM_Simulations.config import (
    SimulationConfig, TaskV2_3SimulationConfig, TaskV2_4SimulationConfig, TaskV3SimulationConfig,
    TaskV4SimulationConfig, MDSSweepConfig
)
from SpAM_Simulations.experiment import ExperimentParameters, ExperimentResults
from SpAM_Simulations.metrics import (
    coverage, snr_summary, test_retest_summary, screening_summary, spearman_correlation,
    _calculate_mean_distances
)
from SpAM_Simulations.simulation import Simulation, build_ground_truth_embeddings
from SpAM_Simulations.storage import ResultStore

logger = logging.getLogger(__name__)

_SUCCESS_STATUSES = ("success", "max_iters")


def _param_type(sim: Simulation) -> type:
    """The `ExperimentParameters`/`TaskV2_3ExperimentParameters` type this sim's results are
    keyed by - derived from an existing result rather than assumed, so the MDS sweep below
    works unmodified for either simulation type."""
    return type(next(iter(sim._results)))


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


def generate_task_v2_3_simulation(config: TaskV2_3SimulationConfig, verbose: bool = True) -> Simulation:
    """Same as `generate_simulation`, but for the task-v2.3 (per-subject trial design) experiment."""
    if config.uses_random_ground_truth:
        sim = Simulation.make(config.n_images, config.n_dims, config.seed)
    else:
        sim = Simulation.from_embeddings(config.gt_embeddings, config.seed)
    schedule = config.param_grid() * config.reps
    for params in tqdm(schedule, desc="Running experiments", disable=not verbose):
        sim.run_task_v2_3_experiment(params, verbose=False)
    return sim


def generate_task_v2_4_simulation(config: TaskV2_4SimulationConfig, verbose: bool = True) -> Simulation:
    """Same as `generate_simulation`, but for the task-v2.4 experiment (v2.3 + whole-trial repeats)."""
    if config.uses_random_ground_truth:
        sim = Simulation.make(config.n_images, config.n_dims, config.seed)
    else:
        sim = Simulation.from_embeddings(config.gt_embeddings, config.seed)
    schedule = config.param_grid() * config.reps
    for params in tqdm(schedule, desc="Running experiments", disable=not verbose):
        sim.run_task_v2_4_experiment(params, verbose=False)
    return sim


def generate_task_v3_simulation(config: TaskV3SimulationConfig, verbose: bool = True) -> Simulation:
    """Build a Simulation for the task-v3 generative model and run its full parameter grid.

    Unlike the earlier generators, ground truth is built with an explicit eigenvalue spectrum
    (``simulation.build_ground_truth_embeddings`` with the config's ``use_isotropic``/``decay``/
    ``n_clusters``) so the coordinate-space observation model has a meaningful PC basis to reweight.
    """
    if config.uses_random_ground_truth:
        embeddings = build_ground_truth_embeddings(
            config.n_images, config.n_dims, use_isotropic=config.use_isotropic,
            decay=config.decay, n_clusters=config.n_clusters, seed=config.seed
        )
    else:
        embeddings = config.gt_embeddings  # e.g. the pilot-calibrated embedding
    sim = Simulation.from_embeddings(embeddings, config.seed)
    schedule = config.param_grid() * config.reps
    for params in tqdm(schedule, desc="Running experiments", disable=not verbose):
        sim.run_task_v3_experiment(params, verbose=False)
    return sim


def generate_task_v4_simulation(config: TaskV4SimulationConfig, verbose: bool = True) -> Simulation:
    """Same as `generate_task_v3_simulation`, but for task-v4 (v3 model + the screening block).

    Ground truth is built identically (explicit eigenvalue spectrum, or a supplied
    ``gt_embeddings`` such as the pilot-calibrated embedding); only the per-experiment model
    differs. Note that a screened configuration simulates *more* subjects than ``num_subjects`` -
    rejected candidates are generated and discarded - so generation is slower than v3 at the same
    grid size, by roughly the reciprocal of the pass rate.
    """
    if config.uses_random_ground_truth:
        embeddings = build_ground_truth_embeddings(
            config.n_images, config.n_dims, use_isotropic=config.use_isotropic,
            decay=config.decay, n_clusters=config.n_clusters, seed=config.seed
        )
    else:
        embeddings = config.gt_embeddings
    sim = Simulation.from_embeddings(embeddings, config.seed)
    schedule = config.param_grid() * config.reps
    for params in tqdm(schedule, desc="Running experiments", disable=not verbose):
        sim.run_task_v4_experiment(params, verbose=False)
    return sim


# --------------------------------------------------------------------------- metric tables
def compute_coverage_table(sim: Simulation) -> pd.DataFrame:
    """One row per (configuration, repetition) with all coverage metrics.

    For a task-v2.3/v2.4 simulation (whose results carry a `subject_snr` field) also includes
    the SNR summary stats from `metrics.snr_summary`; for task-v2.4 (which additionally carries
    `subject_test_retest`) it includes the test-retest summary from `metrics.test_retest_summary`;
    and for task-v4 (which carries `n_candidates_screened`) the screening/recruitment-cost stats
    from `metrics.screening_summary`.
    """
    rows = [
        {**params._asdict(), "rep": rep, **coverage(res),
         **(snr_summary(res) if hasattr(res, "subject_snr") else {}),
         **(test_retest_summary(res) if hasattr(res, "subject_test_retest") else {}),
         **(screening_summary(res) if hasattr(res, "n_candidates_screened") else {})}
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
    """Mean observed distances + a 0/1 weight mask (missing pairs get zero weight).

    Both arrays are float32 - the values (distances and a 0/1 mask) need no more precision, and
    for a large sim each pair is ``N(N-1)/2`` long, so float32 halves the per-task footprint.
    """
    mean = _calculate_mean_distances(res)
    weights = (~np.isnan(mean)).astype(np.float32)
    dists = np.nan_to_num(mean, nan=0.0).astype(np.float32, copy=False)
    return dists, weights


def _task_key(params: ExperimentParameters, rep: int, ndim: int) -> tuple:
    return (*(float(v) for v in params), int(rep), int(ndim))


def _completed_keys(store: ResultStore, param_type: type) -> set:
    df = store.metadata()
    keys = set()
    for _, row in df.iterrows():
        params = param_type(*(row[f] for f in param_type._fields))
        keys.add(_task_key(params, int(row["rep"]), int(row["ndim"])))
    return keys


def _build_mds_payload(task, sweep_config: MDSSweepConfig) -> tuple:
    """Build a small, picklable payload for one MDS task (for serial or worker execution)."""
    params, rep, res, ndim = task
    dists, weights = _prepare_mds_inputs(res)
    meta_base = {**params._asdict(), "rep": rep, "ndim": ndim}
    return (meta_base, dists, weights, ndim,
            sweep_config.max_iters, sweep_config.convergence_tol, sweep_config.precalc_init)


def _pending_tasks(sim: Simulation, sweep_config: MDSSweepConfig, completed: set):
    """Yield the (params, rep, res, ndim) tasks not already recorded in the store."""
    for task in mds_tasks(sim, sweep_config):
        if _task_key(task[0], task[1], task[3]) not in completed:
            yield task


def _iter_pending_payloads(sim: Simulation, sweep_config: MDSSweepConfig, completed: set):
    """Lazily build one payload per pending task.

    Crucially this is a generator, not a list: each payload holds a full-length dists+weights
    pair, so for a large sim with thousands of tasks materialising them all at once would need
    tens of GB. Building on demand keeps only a handful resident at a time.
    """
    for task in _pending_tasks(sim, sweep_config, completed):
        yield _build_mds_payload(task, sweep_config)


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
    param_type = _param_type(sim)
    if (store_path / "store_info.json").exists() and not overwrite:
        store = ResultStore.open(store_path)
        completed = _completed_keys(store, param_type)
    else:
        meta_columns = list(param_type._fields) + ["rep", "ndim", "niter", "stress", "status"]
        store = ResultStore.create(store_path, confdist_len, meta_columns, overwrite=overwrite)
        completed = set()

    # Count pending tasks for the progress bar without building any payloads (cheap: iterating
    # mds_tasks only yields references to the existing results, it allocates nothing).
    total = sum(1 for _ in _pending_tasks(sim, sweep_config, completed))
    payloads = _iter_pending_payloads(sim, sweep_config, completed)  # lazy generator
    try:
        if parallel and total:
            from joblib import Parallel, delayed
            # joblib pulls from the generator up to `pre_dispatch` ahead, so only a handful of
            # payloads are resident at once; results stream back as each worker finishes.
            results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(_execute_mds_payload)(payload) for payload in payloads
            )
            for meta, confdist in tqdm(results, total=total, desc="Running MDS", disable=not verbose):
                store.append(meta, confdist)
        else:
            for payload in tqdm(payloads, total=total, desc="Running MDS", disable=not verbose):
                meta, confdist = _execute_mds_payload(payload)
                store.append(meta, confdist)
    finally:
        store.close()
    return store


# --------------------------------------------------------------------------- post-MDS stability
def compute_embedding_stability(
    store: ResultStore, group_fields: Optional[Sequence[str]] = None, verbose: bool = True
) -> pd.DataFrame:
    """Post-MDS reliability: mean Spearman agreement of reconstructed distances across reps.

    Groups successful results by ``group_fields`` (default: the parameters + ndim) and
    correlates the stored confdist vectors across repetitions within each group. Each group
    does a few pairwise Spearman correlations over full-length confdist vectors, so for large
    sweeps this loop can take minutes with no other output - hence the progress bar.
    """
    df = store.metadata()
    if group_fields is None:
        # Every metadata column is a swept parameter or `ndim` except these fixed,
        # solver-outcome columns - this works for any params type (task-v0.1 or task-v2.3).
        _non_param_columns = {"rep", "niter", "stress", "status", "confdist_row"}
        group_fields = [c for c in df.columns if c not in _non_param_columns]
    group_fields = list(group_fields)

    df = df[df["status"].isin(_SUCCESS_STATUSES) & (df["confdist_row"] >= 0)]

    grouped = df.groupby(group_fields)
    rows = []
    for key, grp in tqdm(grouped, total=grouped.ngroups, desc="Embedding stability", disable=not verbose):
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


def _topk_similar_jaccard(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    """Jaccard overlap of the *smallest* ``frac`` fraction of two distance vectors.

    The smallest distances are the most-similar (closest) pairs - the 'too-similar' candidates. Both
    vectors index the same pair set, so we compare which pairs each rep flags as closest. Returns
    ``|A n B| / |A u B|`` (equal set sizes, so 0..1; 1 = identical closest-pair set).
    """
    n = a.shape[0]
    k = max(1, int(round(frac * n)))
    ma = np.zeros(n, dtype=bool); ma[np.argpartition(a, k - 1)[:k]] = True
    mb = np.zeros(n, dtype=bool); mb[np.argpartition(b, k - 1)[:k]] = True
    union = int(np.count_nonzero(ma | mb))
    return int(np.count_nonzero(ma & mb)) / union if union else np.nan


def compute_topk_similar_pair_stability(
    store: ResultStore, top_fracs: Sequence[float] = (0.05, 0.1, 0.25),
    group_fields: Optional[Sequence[str]] = None, verbose: bool = True,
) -> pd.DataFrame:
    """Reproducibility of the *most-similar pairs* across reps - the decision-relevant reliability
    when the goal is flagging items that are 'too similar'.

    For each configuration group and each fraction ``f`` in ``top_fracs`` (e.g. 0.05, 0.25 = the top
    5%/25% closest pairs), takes the ``f``-smallest entries of each rep's reconstructed distance vector
    and reports the mean pairwise **Jaccard** overlap of those closest-pair sets across reps. Unlike the
    full-RDM ``mean_spearman`` this ignores the noisy mid/far range and measures only whether the near
    neighbourhood is stable. Returns one row per (group, ``top_frac``) with ``mean_jaccard``/``sem``.

    Jaccard (not precision/recall/F1) because the two rep sets are the same size (top-``f``), so
    precision = recall = F1 = the overlap coefficient - all monotone transforms of Jaccard; there is
    no ground-truth 'positive' set in a rep-vs-rep comparison, so a d-prime/SDT framing doesn't apply
    (that would need recovery-vs-GT, a different, simulation-only question).
    """
    fracs = [float(top_fracs)] if isinstance(top_fracs, (int, float)) else [float(x) for x in top_fracs]
    if any(not (0 < f <= 1) for f in fracs):
        raise ValueError(f"top_fracs must be in (0, 1], got {fracs}")
    df = store.metadata()
    if group_fields is None:
        _non_param_columns = {"rep", "niter", "stress", "status", "confdist_row"}
        group_fields = [c for c in df.columns if c not in _non_param_columns]
    group_fields = list(group_fields)
    df = df[df["status"].isin(_SUCCESS_STATUSES) & (df["confdist_row"] >= 0)]

    grouped = df.groupby(group_fields)
    rows = []
    for key, grp in tqdm(grouped, total=grouped.ngroups, desc="Top-k pair stability", disable=not verbose):
        confdists = [store.confdist(int(r)) for r in grp["confdist_row"]]
        key_tuple = key if isinstance(key, tuple) else (key,)
        for f in fracs:
            js = [_topk_similar_jaccard(a, b, f) for a, b in combinations(confdists, 2)]
            js = [j for j in js if not np.isnan(j)]
            rows.append({
                **dict(zip(group_fields, key_tuple)), "top_frac": f, "n_reps": len(confdists),
                "mean_jaccard": float(np.mean(js)) if js else np.nan,
                "sem_jaccard": (float(np.std(js, ddof=1) / np.sqrt(len(js))) if len(js) > 1 else np.nan),
            })
    return pd.DataFrame(rows)
