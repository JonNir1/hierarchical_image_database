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
* ``compute_embedding_generalizability`` - post-MDS Procrustes disparity between the *spaces*
                                   two independent cohorts recover.
* ``compute_item_generalizability``      - the same, resolved per image.

``run_mds`` (and therefore R) is imported lazily inside ``run_mds_sweep`` so the rest of the
module is usable without an R/smacof installation.
"""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from tqdm import tqdm

from SpAM_Simulations.core.config import (
    SimulationConfig, TaskV2_3SimulationConfig, TaskV2_4SimulationConfig, TaskV3SimulationConfig,
    TaskV4SimulationConfig, MDSSweepConfig
)
from SpAM_Simulations.models.experiment import ExperimentParameters, ExperimentResults
from SpAM_Simulations.measures.metrics import (
    coverage, snr_summary, test_retest_summary, screening_summary, spearman_correlation,
    topk_similar_jaccard, _calculate_mean_distances
)
from SpAM_Simulations.core.simulation import Simulation, build_ground_truth_embeddings
from SpAM_Simulations.core.storage import ResultStore

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


def generate_task_v4_simulation(config: TaskV4SimulationConfig, verbose: bool = True,
                                allocator_factory=None) -> Simulation:
    """Same as `generate_task_v3_simulation`, but for task-v4 (v3 model + the screening block).

    Ground truth is built identically (explicit eigenvalue spectrum, or a supplied
    ``gt_embeddings`` such as the pilot-calibrated embedding); only the per-experiment model
    differs. Note that a screened configuration simulates *more* subjects than ``num_subjects`` -
    rejected candidates are generated and discarded - so generation is slower than v3 at the same
    grid size, by roughly the reciprocal of the pass rate.

    ``allocator_factory(params, rep) -> allocator | None`` supplies the image-to-trial allocation
    for each scheduled cell, which is how the ``allocation_mode`` lever is realised: the mode is a
    number inside ``params`` (so it survives the ``ResultStore`` round-trip and shows up as a
    grouping column), while the allocator object it selects is built here. ``rep`` is passed so a
    caller can hand each repetition its own design; sharing one design across reps would leave the
    designed arm with zero allocation variance while the random arm carries it, making the two
    arms' spreads incomparable.
    """
    if config.uses_random_ground_truth:
        embeddings = build_ground_truth_embeddings(
            config.n_images, config.n_dims, use_isotropic=config.use_isotropic,
            decay=config.decay, n_clusters=config.n_clusters, seed=config.seed
        )
    else:
        embeddings = config.gt_embeddings
    sim = Simulation.from_embeddings(embeddings, config.seed)
    grid = config.param_grid()
    schedule = [(params, rep) for rep in range(config.reps) for params in grid]
    for params, rep in tqdm(schedule, desc="Running experiments", disable=not verbose):
        allocator = allocator_factory(params, rep) if allocator_factory is not None else None
        sim.run_task_v4_experiment(params, verbose=False, allocator=allocator)
    return sim


def generate_task_v5_simulation(config, verbose: bool = True, allocator_factory=None) -> Simulation:
    """Same as `generate_task_v4_simulation`, but on the bounded canvas (task-v5).

    Identical scheduling and allocator handling; only the observation model differs, and it differs
    inside `simulate_task_v5_experiment` rather than here. `canvas_softness` rides in the parameter
    tuple, so it lands in the store and becomes a grouping column in every compute_* table - which
    is the point, since it is swept as a sensitivity axis rather than calibrated.
    """
    if config.uses_random_ground_truth:
        embeddings = build_ground_truth_embeddings(
            config.n_images, config.n_dims, use_isotropic=config.use_isotropic,
            decay=config.decay, n_clusters=config.n_clusters, seed=config.seed
        )
    else:
        embeddings = config.gt_embeddings
    sim = Simulation.from_embeddings(embeddings, config.seed)
    grid = config.param_grid()
    schedule = [(params, rep) for rep in range(config.reps) for params in grid]
    for params, rep in tqdm(schedule, desc="Running experiments", disable=not verbose):
        allocator = allocator_factory(params, rep) if allocator_factory is not None else None
        sim.run_task_v5_experiment(params, verbose=False, allocator=allocator)
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


# Parameters are rounded before they become a key. `storage.metadata` already reads with
# round-trip precision, so this is belt-and-braces against any future path that formats a float
# differently on the way to disk - the failure it guards is silent and expensive: one differing
# last digit makes every key miss, `completed` comes back empty, and a resumed sweep re-runs work
# it already has. 12 decimals is far finer than any swept grid and far coarser than float noise.
_KEY_DECIMALS = 12


def _task_key(params: ExperimentParameters, rep: int, ndim: int) -> tuple:
    return (*(round(float(v), _KEY_DECIMALS) for v in params), int(rep), int(ndim))


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


def _execute_mds_payload(payload: tuple) -> Tuple[dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """Run one MDS payload, returning (metadata, confdist, conf). Failures are recorded, not raised.

    ``conf`` is SMACOF's fitted configuration - the ``(n_images, ndim)`` coordinates. It is
    returned alongside ``confdist`` because comparing two cohorts' embedding *spaces* (Procrustes)
    needs coordinates, which cannot be recovered from a distance vector; both are ``None`` for a
    failed run. Module-level and picklable so it can run in a joblib worker process. ``run_mds``
    (and R) is imported here so each worker initialises its own R instance exactly once.
    """
    from SpAM_Simulations.core.multi_dimensional_scaling import run_mds
    meta_base, dists, weights, ndim, max_iters, tol, precalc = payload
    meta = {**meta_base, "niter": np.nan, "stress": np.nan, "status": "success"}
    try:
        out = run_mds(dists=dists, weights=weights, ndim=ndim,
                      max_iters=max_iters, convergence_tol=tol, precalc_init=precalc)
    except RuntimeError as e:
        meta["status"] = "disconnected" if "connected components" in str(e) else "error"
        logger.warning("MDS task (ndim=%s, rep=%s) failed: %s", ndim, meta_base.get("rep"), e)
        return meta, None, None
    meta["niter"] = float(out["niter"])
    meta["stress"] = float(out["stress"])
    meta["status"] = "max_iters" if out["needs_more_iters"] else "success"
    conf = np.asarray(out["conf"], dtype=np.float32).reshape(-1, ndim)
    return meta, np.asarray(out["confdist"], dtype=np.float32), conf


def run_mds_sweep(
    sim: Simulation,
    sweep_config: MDSSweepConfig,
    store_path: str | Path,
    *,
    parallel: bool = False,
    n_jobs: Optional[int] = None,
    overwrite: bool = False,
    store_conf: bool = True,
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

    ``store_conf=True`` (the default) also records each fit's configuration, which
    :func:`compute_embedding_generalizability` and :func:`compute_item_generalizability` need;
    it costs a few percent of the store's size (see ``storage``). It is ignored when resuming a
    store that was created without configurations, so a resumed pre-existing sweep keeps its
    original format rather than desynchronising mid-run.
    """
    store_path = Path(store_path)
    confdist_len = sim.num_images * (sim.num_images - 1) // 2
    param_type = _param_type(sim)
    if (store_path / "store_info.json").exists() and not overwrite:
        store = ResultStore.open(store_path)
        completed = _completed_keys(store, param_type)
    else:
        meta_columns = list(param_type._fields) + ["rep", "ndim", "niter", "stress", "status"]
        conf_kwargs = ({"n_images": sim.num_images,
                        "max_ndim": max(sweep_config.target_dims(sim.gt_dimensions))}
                       if store_conf else {})
        store = ResultStore.create(store_path, confdist_len, meta_columns, overwrite=overwrite,
                                   **conf_kwargs)
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
            for meta, confdist, conf in tqdm(results, total=total, desc="Running MDS", disable=not verbose):
                store.append(meta, confdist, conf if store.stores_conf else None)
        else:
            for payload in tqdm(payloads, total=total, desc="Running MDS", disable=not verbose):
                meta, confdist, conf = _execute_mds_payload(payload)
                store.append(meta, confdist, conf if store.stores_conf else None)
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

    See :func:`compute_embedding_generalizability` for the configuration-space counterpart: this
    function compares the *rank order* of the pairwise distances, that one compares the recovered
    *spaces*. Both are reported; they can disagree.
    """
    grouped, group_fields = _grouped_successful(store, group_fields)
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


def _grouped_successful(store: ResultStore, group_fields: Optional[Sequence[str]]):
    """Shared grouping for the post-MDS metrics: successful, confdist-bearing records by config."""
    df = store.metadata()
    if group_fields is None:
        # Every metadata column is a swept parameter or `ndim` except these fixed,
        # solver-outcome columns - this works for any params type.
        _non_param_columns = {"rep", "niter", "stress", "status", "confdist_row"}
        group_fields = [c for c in df.columns if c not in _non_param_columns]
    group_fields = list(group_fields)
    df = df[df["status"].isin(_SUCCESS_STATUSES) & (df["confdist_row"] >= 0)]

    # ONE ROW PER (configuration, rep). A store is append-only, so a resume that fails to recognise
    # already-completed work appends a SECOND copy of every fit it redoes - which is exactly what
    # the pre-round-trip resume bug did: 48 of 1728 groups in the task-v5 stage-2 store hold 20 rows
    # for 10 reps, with bit-identical stress between the copies.
    #
    # Left in, those duplicates enter the C(n, 2) pair loop as self-comparisons of an identical
    # cohort - VI exactly 0, ARI exactly 1 - so every rep-pair metric is biased upward in precisely
    # the affected cells. The duplicates are the same cohort refitted deterministically, so keeping
    # the first is a free choice.
    before = len(df)
    df = df.drop_duplicates(subset=group_fields + ["rep"], keep="first")
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "dropped %d duplicate (configuration, rep) fits of %d: an append-only store that was "
            "resumed without recognising completed work holds more than one copy of them. Keeping "
            "the first of each; leaving them in would compare cohorts against themselves.",
            dropped, before)
    return df.groupby(group_fields), group_fields


# --------------------------------------------------------------------------- parallel group work
# The post-MDS cluster metrics are all the same shape: independent work per configuration group,
# with the O(reps^2) pair comparisons inside. Nothing crosses a group boundary, so groups are the
# natural parallel axis - and they need to be, because the analysis is otherwise single-threaded
# and a 17,729-fit store takes hours of wall clock on one core.
#
# Workers reopen the store from its PATH rather than receiving it. A ResultStore holds open file
# handles and memmaps, which do not survive pickling to a spawned process; reopening is cheap and
# the store is append-only and read-only here.
_WORKER_STORES: Dict[str, ResultStore] = {}


def _worker_store(path: str) -> ResultStore:
    """The store for this worker process, opened once and cached for the rest of its life."""
    store = _WORKER_STORES.get(path)
    if store is None:
        store = _WORKER_STORES[path] = ResultStore.open(path)
    return store


def group_tasks(store: ResultStore, group_fields: Optional[Sequence[str]] = None,
                min_size: int = 1) -> List[tuple]:
    """One ``(base, confdist_rows, ndim)`` task per configuration group.

    ``min_size=2`` drops groups with a single successful rep, which have nothing to compare against.
    Built in the parent from metadata alone, so no configuration is read until a worker asks for it.
    """
    grouped, resolved = _grouped_successful(store, group_fields)
    tasks = []
    for key, grp in grouped:
        if len(grp) < min_size:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        tasks.append((dict(zip(resolved, key_tuple)),
                      [int(r) for r in grp["confdist_row"]],
                      int(grp["ndim"].iloc[0])))
    return tasks


def map_groups(store: ResultStore, tasks: Sequence[tuple], worker, *, n_jobs: int = -1,
               desc: str = "", verbose: bool = True, **kwargs) -> List[Dict]:
    """Run ``worker(store_path, base, rows, ndim, **kwargs)`` over every task, flattening the rows.

    ``n_jobs=1`` runs in-process, which is what the tests use: identical results, no spawn cost, and
    a traceback that points at the real line rather than at a worker boundary.

    With a parallel backend the progress bar tracks DISPATCH, not completion. joblib pre-dispatches
    only a small multiple of ``n_jobs``, so the two stay close, but a bar at 100% means the last
    task has been handed out rather than finished.
    """
    path = str(store.path)
    if n_jobs == 1:
        return [row for task in tqdm(tasks, desc=desc, disable=not verbose)
                for row in worker(path, *task, **kwargs)]
    from joblib import Parallel, delayed
    chunks = Parallel(n_jobs=n_jobs)(
        delayed(worker)(path, *task, **kwargs)
        for task in tqdm(tasks, desc=desc, disable=not verbose))
    return [row for chunk in chunks for row in chunk]


def map_groups_multi(store: ResultStore, tasks: Sequence[tuple], worker, *, n_jobs: int = -1,
                     desc: str = "", verbose: bool = True, **kwargs) -> Dict[str, List[Dict]]:
    """As :func:`map_groups`, but the worker returns ``{table_name: rows}`` for several tables.

    This is what lets one traversal feed metrics that would otherwise each rebuild the same
    per-cohort clustering. Rebuilding it is not cheap - linkage, twelve cuts and a cophenetic
    ranking per (fit, linkage) - and three separate passes over the same store paid for it three
    times.
    """
    path = str(store.path)
    if n_jobs == 1:
        chunks = [worker(path, *task, **kwargs)
                  for task in tqdm(tasks, desc=desc, disable=not verbose)]
    else:
        from joblib import Parallel, delayed
        chunks = Parallel(n_jobs=n_jobs)(
            delayed(worker)(path, *task, **kwargs)
            for task in tqdm(tasks, desc=desc, disable=not verbose))
    merged: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        for name, rows in chunk.items():
            merged.setdefault(name, []).extend(rows)
    return merged


def _require_conf_store(store: ResultStore) -> None:
    if not store.stores_conf:
        raise ValueError(
            "this store has no MDS configurations, so embedding *spaces* cannot be compared. "
            "Re-run the sweep with `run_mds_sweep(..., store_conf=True)` (the default); stores "
            "written before configurations were recorded only support `compute_embedding_stability`."
        )


def _aligned_pair(store: ResultStore, row_a: int, row_b: int, ndim: int):
    """Procrustes-align two cohorts' configurations, returning ``(a, b, m2)``.

    ``scipy.spatial.procrustes`` centres both configurations, scales each to unit Frobenius norm
    and applies the optimal orthogonal map to the second. That is exactly the invariance an MDS
    solution has - position, scale, rotation and reflection are all arbitrary - so what remains is
    the genuine disagreement in relative geometry.
    """
    a, b = store.conf(row_a, ndim), store.conf(row_b, ndim)
    std_a, std_b, m2 = procrustes(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return std_a, std_b, float(m2)


def compute_embedding_generalizability(
    store: ResultStore, group_fields: Optional[Sequence[str]] = None, verbose: bool = True
) -> pd.DataFrame:
    """How similar are the embedding **spaces** recovered by two independent cohorts of N subjects?

    Each ``rep`` in a sweep is an independently simulated cohort, so the pairwise comparison within
    a configuration group answers the study-planning question directly: *if I ran this study twice,
    would I get the same space?* For every pair of reps this Procrustes-aligns the two
    configurations and reports the residual disparity ``M^2`` in ``[0, 1]`` - **0 = identical
    shape, lower is better**, the opposite direction to :func:`compute_embedding_stability`'s
    ``mean_spearman``.

    The two metrics are complementary, not redundant, and both are worth reporting.
    ``mean_spearman`` correlates *distance vectors*, so it only asks whether the pairs are ordered
    the same way; ``M^2`` compares the *configurations*, so it is sensitive to metric distortion
    that leaves the rank order intact. A configuration pair can rank-correlate well while sitting
    in a measurably different space.

    Returns one row per group with ``mean_procrustes_m2``/``sem_procrustes_m2``/``n_reps``.
    Requires a store written with configurations (see :func:`run_mds_sweep`).
    """
    _require_conf_store(store)
    grouped, group_fields = _grouped_successful(store, group_fields)
    rows = []
    for key, grp in tqdm(grouped, total=grouped.ngroups, desc="Embedding generalizability",
                         disable=not verbose):
        ndim = int(grp["ndim"].iloc[0])
        m2s = [_aligned_pair(store, int(i), int(j), ndim)[2]
               for i, j in combinations(grp["confdist_row"], 2)]
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(group_fields, key_tuple)),
            "n_reps": len(grp),
            "mean_procrustes_m2": float(np.mean(m2s)) if m2s else np.nan,
            "sem_procrustes_m2": (float(np.std(m2s, ddof=1) / np.sqrt(len(m2s)))
                                  if len(m2s) > 1 else np.nan),
        })
    return pd.DataFrame(rows)


def compute_item_generalizability(
    store: ResultStore, group_fields: Optional[Sequence[str]] = None, verbose: bool = True
) -> pd.DataFrame:
    """Per-**image** contribution to the between-cohort Procrustes disparity.

    Same alignment as :func:`compute_embedding_generalizability`, but instead of collapsing to one
    number per group it keeps each image's residual distance between the two aligned
    configurations, averaged over rep-pairs. Images with large residuals are the ones whose
    position does not generalise across cohorts - useful for flagging unstable stimuli, and the
    item-level Procrustes diagnostic listed as exploratory in the pre-registration.

    Returns one row per (group, ``image_index``) with ``mean_residual``/``sem_residual``. Note this
    is ``n_images`` rows per group, so it is much larger than the group-level table; pass
    ``group_fields`` to restrict it to the configurations you actually want to inspect.

    **Read this alongside the group-level ``M^2``, not on its own.** Procrustes is a *global* fit
    that scales both configurations to unit norm, so residuals are only attributable to individual
    images while the two spaces broadly agree. Once they diverge badly (empirically around
    ``M^2 > 0.5``) a single grossly displaced image dominates the scaling and distorts the
    alignment of every other image, at which point the largest residual no longer identifies the
    image that actually moved. Cohort pairs in a calibrated sweep sit far below that threshold.
    """
    _require_conf_store(store)
    grouped, group_fields = _grouped_successful(store, group_fields)
    rows = []
    for key, grp in tqdm(grouped, total=grouped.ngroups, desc="Item generalizability",
                         disable=not verbose):
        ndim = int(grp["ndim"].iloc[0])
        residuals = []
        for i, j in combinations(grp["confdist_row"], 2):
            std_a, std_b, _ = _aligned_pair(store, int(i), int(j), ndim)
            residuals.append(np.linalg.norm(std_a - std_b, axis=1))
        if not residuals:
            continue
        stacked = np.vstack(residuals)
        mean = stacked.mean(axis=0)
        sem = (stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
               if stacked.shape[0] > 1 else np.full(stacked.shape[1], np.nan))
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_fields, key_tuple))
        rows.extend({**base, "image_index": idx, "n_pairs": stacked.shape[0],
                     "mean_residual": float(mean[idx]), "sem_residual": float(sem[idx])}
                    for idx in range(stacked.shape[1]))
    return pd.DataFrame(rows)


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
    grouped, group_fields = _grouped_successful(store, group_fields)
    rows = []
    for key, grp in tqdm(grouped, total=grouped.ngroups, desc="Top-k pair stability", disable=not verbose):
        confdists = [store.confdist(int(r)) for r in grp["confdist_row"]]
        key_tuple = key if isinstance(key, tuple) else (key,)
        for f in fracs:
            js = [topk_similar_jaccard(a, b, f) for a, b in combinations(confdists, 2)]
            js = [j for j in js if not np.isnan(j)]
            rows.append({
                **dict(zip(group_fields, key_tuple)), "top_frac": f, "n_reps": len(confdists),
                "mean_jaccard": float(np.mean(js)) if js else np.nan,
                "sem_jaccard": (float(np.std(js, ddof=1) / np.sqrt(len(js))) if len(js) > 1 else np.nan),
            })
    return pd.DataFrame(rows)


def compute_recovery_vs_gt(
    store: ResultStore, gt_condensed: np.ndarray,
    fracs: Sequence[float] = (0.01, 0.05, 0.10),
    group_fields: Optional[Sequence[str]] = None, verbose: bool = True,
) -> pd.DataFrame:
    """How well each rep recovers the **ground truth's** closest pairs (simulation only).

    The counterpart to `compute_topk_similar_pair_stability`: that one asks whether two cohorts
    agree with each other (reproducibility), this one asks whether a cohort finds the pairs that
    are genuinely closest in the space the data were generated from (recovery). They can diverge -
    two cohorts can agree on the wrong answer - so both are reported.

    Scored per rep and then averaged within a configuration group, giving one row per
    (group, `top_frac`) with the mean and SEM of `recall`, `dprime`, `separation_dprime` and `auc`.
    See `recovery` for why both a thresholded and a threshold-free d-prime are carried.
    """
    from SpAM_Simulations.measures.recovery import recovery_summary

    gt_condensed = np.asarray(gt_condensed, dtype=np.float64)
    fracs = [float(fracs)] if isinstance(fracs, (int, float)) else [float(x) for x in fracs]
    if any(not (0 < f <= 1) for f in fracs):
        raise ValueError(f"fracs must be in (0, 1], got {fracs}")
    grouped, group_fields = _grouped_successful(store, group_fields)
    metrics = ("recall", "dprime", "separation_dprime", "auc")
    rows = []
    for key, grp in tqdm(grouped, total=grouped.ngroups, desc="Recovery vs GT", disable=not verbose):
        key_tuple = key if isinstance(key, tuple) else (key,)
        per_frac: dict = {f: [] for f in fracs}
        for row in grp["confdist_row"]:
            confdist = store.confdist(int(row))
            if confdist.shape != gt_condensed.shape:
                raise ValueError(
                    f"stored confdist has {confdist.shape[0]} pairs but the supplied ground truth "
                    f"has {gt_condensed.shape[0]}; they must index the same image set"
                )
            for summary in recovery_summary(confdist, gt_condensed, fracs):
                per_frac[summary["top_frac"]].append(summary)
        for f in fracs:
            entries = per_frac[f]
            out = {**dict(zip(group_fields, key_tuple)), "top_frac": f, "n_reps": len(entries)}
            for m in metrics:
                vals = np.array([e[m] for e in entries], dtype=float)
                vals = vals[np.isfinite(vals)]
                out[f"mean_{m}"] = float(vals.mean()) if vals.size else np.nan
                out[f"sem_{m}"] = (float(vals.std(ddof=1) / np.sqrt(vals.size))
                                   if vals.size > 1 else np.nan)
            rows.append(out)
    return pd.DataFrame(rows)
