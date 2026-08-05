from __future__ import annotations

import os
from datetime import datetime
from functools import cached_property
from typing import List, Dict, Optional
from itertools import product
import pickle as pkl

import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm

from SpAM_Simulations.experiment import simulate_experiment, ExperimentParameters, ExperimentResults
from SpAM_Simulations.task_v2_3_experiment import (
    simulate_task_v2_3_experiment, TaskV2_3ExperimentParameters, TaskV2_3ExperimentResults
)
from SpAM_Simulations.task_v2_4_experiment import (
    simulate_task_v2_4_experiment, TaskV2_4ExperimentParameters, TaskV2_4ExperimentResults
)
from SpAM_Simulations.task_v3_experiment import (
    simulate_task_v3_experiment, TaskV3ExperimentParameters, TaskV3ExperimentResults
)
from SpAM_Simulations.task_v4_experiment import (
    simulate_task_v4_experiment, TaskV4ExperimentParameters, TaskV4ExperimentResults
)

_SimulationResults = Dict[ExperimentParameters, List[ExperimentResults]]


def build_ground_truth_embeddings(
        N: int,
        D: int,
        use_isotropic: bool = True,
        decay: float = 0.7,
        n_clusters: Optional[int] = None,
        cluster_separation: float = 3.0,
        seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic ground-truth embedding in a PC-aligned basis.

    Columns are independent with monotonically non-increasing variance, so the coordinate axes
    *are* the principal components - this is what the task-v3 observation model reweights
    per-subject (see ``task_v3_experiment``). The eigenvalue spectrum is the lever that decides
    how hard each dimension is to resolve once the 2-D arrangement bottleneck is in play:

    * ``use_isotropic=True``  -> all D dims have unit variance (the deliberately hard,
      conservative "every dimension equally important" case; under the 2-D projection it is also
      the *easiest* to span because every dim is equally likely to surface in a trial's local
      top-2).
    * ``use_isotropic=False`` -> per-dim std follows a geometric decay ``std_i = decay ** i``
      (``0 < decay <= 1``), so low-variance dims rarely surface in any trial's 2-D slice and need
      far more subjects to recover (the realistic case).

    ``n_clusters`` (optional) overlays a hierarchical block structure: points are assigned to
    ``n_clusters`` well-separated centres (separation scaled by ``cluster_separation``) with
    within-cluster scatter following the same spectrum, mimicking the dataset's semantic
    hierarchy. Returns an ``(N, D)`` float32 array.
    """
    if N <= 0:
        raise ValueError(f"`N` must be positive (got {N})")
    if D <= 0:
        raise ValueError(f"`D` must be positive (got {D})")
    if not (0 < decay <= 1):
        raise ValueError(f"`decay` must be in (0, 1] (got {decay})")
    if n_clusters is not None and not (0 < n_clusters <= N):
        raise ValueError(f"`n_clusters` must be in (0, N] (got {n_clusters})")

    rng = np.random.default_rng(seed)
    std = np.ones(D, dtype=np.float64) if use_isotropic else decay ** np.arange(D, dtype=np.float64)

    scatter = rng.normal(size=(N, D)) * std
    if n_clusters is None:
        embeddings = scatter
    else:
        centres = rng.normal(size=(n_clusters, D)) * std * cluster_separation
        assignment = rng.integers(0, n_clusters, size=N)
        embeddings = centres[assignment] + scatter
    return embeddings.astype(np.float32)


def create_simulation(
        n_images: int,
        n_dims: int,
        num_subjects: List[int],
        trials_per_subject: List[int],
        images_per_trial: List[int],
        subjects_noise_scale: List[float],
        subjects_noise_df: List[int],
        reps: int = 1,
        seed: int = 42,
        verbose: bool = True,
) -> Simulation:
    sim = Simulation.make(n_images, n_dims, seed)
    exp_params = [ExperimentParameters(*p) for p in product(
        num_subjects, trials_per_subject, images_per_trial, subjects_noise_scale, subjects_noise_df
    )]
    for exp in tqdm(exp_params * reps, desc="Running experiments", disable=not verbose):
        sim.run_experiment(exp, verbose=False)
    sim.to_pickle()
    return sim


def load_latest_simulation(sim_dir: str) -> Simulation:
    """
    Load the latest simulation from a directory.
    Assumes simulations are saved with the format "simulation_YYYYMMDD.pkl".
    """
    sim_files = [f for f in os.listdir(sim_dir) if f.startswith("simulation_") and f.endswith(".pkl")]
    if not sim_files:
        raise FileNotFoundError(f"No simulation files found in directory: {sim_dir}")
    latest_file = max(sim_files, key=lambda f: f.split("_")[1].split(".")[0])  # extract date and find max
    latest_path = os.path.join(sim_dir, latest_file)
    return Simulation.from_pickle(latest_path)


class Simulation:

    def __init__(
            self,
            gt_embeddings: np.ndarray,
            rng: np.random.Generator,
            start_time: datetime,
            results: _SimulationResults = None
    ):
        self.gt_embeddings = gt_embeddings
        self.rng = rng
        self.start_time = start_time
        self._results = results or dict()

    @staticmethod
    def make(N: int, D: int, seed: int = 42) -> "Simulation":
        if N <= 0:
            raise ValueError(f"`N` must be positive (got {N})")
        if D <= 0:
            raise ValueError(f"`D` must be positive (got {D})")
        rng = np.random.default_rng(seed)
        gt_embeddings = rng.normal(size=(N, D)).astype(np.float32)
        return Simulation(gt_embeddings, rng, datetime.now())

    @staticmethod
    def from_embeddings(gt_embeddings: np.ndarray, seed: int = 42) -> "Simulation":
        """Create a Simulation from a provided ground-truth embedding (e.g. real image features).

        Embeddings are cast to float32 to match the random-embedding path, so downstream
        ground-truth distances are computed at the same precision regardless of source.
        """
        gt_embeddings = np.asarray(gt_embeddings, dtype=np.float32)
        if gt_embeddings.ndim != 2:
            raise ValueError(f"`gt_embeddings` must be a 2-D (N, D) array, got shape {gt_embeddings.shape}")
        if gt_embeddings.shape[0] <= 0 or gt_embeddings.shape[1] <= 0:
            raise ValueError(f"`gt_embeddings` must have positive N and D, got shape {gt_embeddings.shape}")
        rng = np.random.default_rng(seed)
        return Simulation(gt_embeddings, rng, datetime.now())

    @property
    def num_images(self) -> int:
        return self.gt_embeddings.shape[0]

    @property
    def gt_dimensions(self) -> int:
        return self.gt_embeddings.shape[1]

    @cached_property
    def gt_distances(self) -> np.ndarray:
        """
        Calculate the pairwise Euclidean distances between embeddings.
        Returns a N(N-1)/2 vector of distances.

        Cached: embeddings are fixed for a Simulation's lifetime, so this is computed once
        and reused across every experiment run instead of on each access.
        """
        return pdist(self.gt_embeddings, metric="euclidean").astype(np.float32)

    def get_experiment(
            self, params: ExperimentParameters
    ) -> List[ExperimentResults]:
        return self._results.get(params, [])

    def run_experiment(
            self, params: ExperimentParameters, verbose: bool = True
    ) -> ExperimentResults:
        exp_params, exp_results = simulate_experiment(params, self.gt_distances, self.rng, verbose)
        # `_results` maps each parameter set to a list of repetition results; append this run.
        self._results.setdefault(exp_params, []).append(exp_results)
        return exp_results

    def run_task_v2_3_experiment(
            self, params: TaskV2_3ExperimentParameters, verbose: bool = True
    ) -> TaskV2_3ExperimentResults:
        """Same as `run_experiment`, but for the task-v2.3 (per-subject trial design) simulation."""
        exp_params, exp_results = simulate_task_v2_3_experiment(params, self.gt_distances, self.rng, verbose)
        self._results.setdefault(exp_params, []).append(exp_results)
        return exp_results

    def run_task_v2_4_experiment(
            self, params: TaskV2_4ExperimentParameters, verbose: bool = True
    ) -> TaskV2_4ExperimentResults:
        """Same as `run_experiment`, but for the task-v2.4 simulation (v2.3 + whole-trial repeats)."""
        exp_params, exp_results = simulate_task_v2_4_experiment(params, self.gt_distances, self.rng, verbose)
        self._results.setdefault(exp_params, []).append(exp_results)
        return exp_results

    def run_task_v3_experiment(
            self, params: TaskV3ExperimentParameters, verbose: bool = True
    ) -> TaskV3ExperimentResults:
        """Same as `run_experiment`, but for the task-v3 simulation.

        Unlike the earlier models (which consume the condensed ``gt_distances``), task-v3's
        observation model is generative in *coordinate* space - it reweights each subject's view
        of the PCs and projects every trial's items onto a local 2-D arrangement - so it is handed
        ``self.gt_embeddings`` (the N x D coordinates), not the distances.
        """
        exp_params, exp_results = simulate_task_v3_experiment(params, self.gt_embeddings, self.rng, verbose)
        self._results.setdefault(exp_params, []).append(exp_results)
        return exp_results

    def run_task_v4_experiment(
            self, params: TaskV4ExperimentParameters, verbose: bool = True, allocator=None
    ) -> TaskV4ExperimentResults:
        """Same as `run_task_v3_experiment`, but for the task-v4 simulation (v3 + screening block).

        Like v3 it consumes ``self.gt_embeddings`` (the N x D coordinates) rather than the
        condensed distances, since the observation model is generative in coordinate space.

        ``allocator`` overrides how images are assigned to trials (see ``allocation``); ``None``
        keeps the deployed per-subject random draw.
        """
        exp_params, exp_results = simulate_task_v4_experiment(
            params, self.gt_embeddings, self.rng, verbose, allocator=allocator)
        self._results.setdefault(exp_params, []).append(exp_results)
        return exp_results

    def get_or_run_experiments(
            self, exp_params: List[ExperimentParameters], reps: int = 1, verbose: bool = True
    ) -> _SimulationResults:
        results = dict()
        for params in tqdm(exp_params):
            existing_results = self.get_experiment(params) or []
            while len(existing_results) < reps:
                # extend existing results with new runs until we have enough repetitions
                new_res = self.run_experiment(params, False)
                existing_results.append(new_res)
            results[params] = existing_results[:reps]   # trim to the requested number of repetitions
        return results

    def to_pickle(self, path: str = ""):
        path = path or os.path.join(os.getcwd(), f"simulation_{self.start_time.strftime('%Y%m%d')}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def from_pickle(path: str) -> "Simulation":
        with open(path, "rb") as f:
            sim = pkl.load(f)
        return sim
