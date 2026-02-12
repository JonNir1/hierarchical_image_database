from datetime import datetime
from typing import NamedTuple, Tuple

import numpy as np
from tqdm import trange
from scipy.spatial.distance import squareform

from SpAM_Simulations.helpers import convert_to_condensed

ExperimentParameters = NamedTuple("ExperimentParameters", [
    ("num_subjects", int),
    ("trials_per_subject", int),
    ("images_per_trial", int),
    ("subjects_noise_scale", float),
    ("subjects_noise_df", int),
])

ExperimentResults = NamedTuple("ExperimentResults", [
    ("run_time", datetime),
    ("distances", np.ndarray),
    ("num_obs", np.ndarray),
    ("subject_noises", np.ndarray),
])


def simulate_experiment(
        params: ExperimentParameters,
        gt_distances: np.ndarray,
        rng: np.random.Generator,
        verbose: bool = True,
) -> Tuple[ExperimentParameters, ExperimentResults]:
    """
    Simulates distance observations from multiple subjects based on ground truth distances.
    Each subject has their own noise level drawn from a scaled half-t distribution.
    Unmeasured distances are represented as NaN.
    """
    # validate parameters

    assert params.num_subjects > 0, f"`num_subjects` must be positive (got {params.num_subjects})"
    assert params.trials_per_subject > 0, f"`trials_per_subject` must be positive (got {params.trials_per_subject})"
    assert params.subjects_noise_scale > 0, f"`subjects_noise_scale` must be positive (got {params.subjects_noise_scale})"
    assert params.subjects_noise_df > 0, f"`subjects_noise_df` must be positive (got {params.subjects_noise_df})"
    # make sure distances are in condensed form
    gt_distances = convert_to_condensed(gt_distances)
    N = squareform(gt_distances, checks=False).shape[0]
    assert 0 < params.images_per_trial < N, f"`images_per_trial` must be between 0 and `N`(={N})"

    all_observations = np.zeros_like(gt_distances)
    all_n_obs = np.zeros_like(gt_distances)
    subject_noises = _draw_subject_noises(
        params.subjects_noise_df,
        params.subjects_noise_scale * gt_distances.std(),  # scale subject noise by GT noise to ensure reasonable range
        params.num_subjects,
        rng
    )
    for s in trange(params.num_subjects, desc="Simulating subjects", disable=not verbose):
        observations, n_obs = simulate_single_subject(
            subject_noise=subject_noises[s],
            num_trials=params.trials_per_subject,
            images_per_trial=params.images_per_trial,
            gt_distances=gt_distances,
            rng=rng,
            verbose=False
        )
        all_observations += observations
        all_n_obs += n_obs
    all_observations = np.where(     # ensure unmeasured distances are NaN
        all_observations > 0, all_observations, np.nan
    )
    results = ExperimentResults(datetime.now(), all_observations, all_n_obs, subject_noises)
    return params, results


def simulate_single_subject(
        subject_noise: float,
        num_trials: int,
        images_per_trial: int,
        gt_distances: np.ndarray,
        rng: np.random.Generator,
        verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates distance observations from a single subject based on ground truth distances, with added Gaussian noise.
    Unmeasured distances are represented as 0.
    """
    assert subject_noise > 0, "`subject_noise` must be positive"
    assert num_trials > 0, "`num_trials` must be positive"
    square_gt_distances = squareform(gt_distances)
    N = square_gt_distances.shape[0]
    assert 0 < images_per_trial < N, "`images_per_trial` must be between 0 and `N`"
    observations, n_obs = np.zeros_like(square_gt_distances), np.zeros_like(square_gt_distances)
    for _t in trange(num_trials, desc="Simulating trials", disable=not verbose):
        selected_indices = np.random.choice(N, size=images_per_trial, replace=False)
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                idx_i, idx_j = selected_indices[i], selected_indices[j]
                noisy_distance = square_gt_distances[idx_i, idx_j] + rng.normal(0, scale=subject_noise)
                noisy_distance = max(0, noisy_distance)     # ensure non-negativity
                observations[idx_i, idx_j] += noisy_distance
                observations[idx_j, idx_i] += noisy_distance
                n_obs[idx_i, idx_j] += 1
                n_obs[idx_j, idx_i] += 1
    return squareform(observations), squareform(n_obs)


def _draw_subject_noises(
        df: int, mu_noise: float, n_subjects: int, rng: np.random.Generator
) -> np.ndarray:
    assert df > 0, "`df` must be positive"
    assert mu_noise > 0, "`mu_noise` must be positive"
    assert n_subjects > 0, "`n_subjects` must be positive"
    raw_variability = np.abs(rng.standard_t(df, size=n_subjects))
    scaled_noises = raw_variability / np.mean(raw_variability) * mu_noise
    return scaled_noises
