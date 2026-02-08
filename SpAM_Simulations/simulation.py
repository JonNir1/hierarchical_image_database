import os
from datetime import datetime
from typing import List, Dict
from itertools import product
import pickle as pkl

import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm

from SpAM_Simulations.experiment import simulate_experiment, ExperimentParameters, ExperimentResults

_SimulationResults = Dict[ExperimentParameters, List[ExperimentResults]]


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
        gt_embeddings = rng.normal(size=(N, D))
        return Simulation(gt_embeddings, rng, datetime.now())

    @property
    def num_images(self) -> int:
        return self.gt_embeddings.shape[0]

    @property
    def gt_dimensions(self) -> int:
        return self.gt_embeddings.shape[1]

    @property
    def gt_distances(self) -> np.ndarray:
        """
        Calculate the pairwise Euclidean distances between embeddings.
        Returns a N(N-1)/2 vector of distances.
        """
        return pdist(self.gt_embeddings, metric="euclidean")

    def get_experiment(
            self, params: ExperimentParameters
    ) -> List[ExperimentResults]:
        return self._results.get(params, [])

    def run_experiment(
            self, params: ExperimentParameters, verbose: bool = True
    ) -> ExperimentResults:
        exp_params, exp_results = simulate_experiment(params, self.gt_distances, self.rng, verbose)
        if exp_params in self._results.keys():
            # this experiment has already been run, convert to list and append
            existing_results = self._results[exp_params]
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
            existing_results.append(exp_results)
            self._results[exp_params] = existing_results
        else:
            self._results[exp_params] = [exp_results]
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
