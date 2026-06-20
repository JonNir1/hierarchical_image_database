from datetime import datetime
from typing import NamedTuple, Tuple

import numpy as np
from tqdm import trange
from scipy.spatial.distance import num_obs_y

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
    assert params.subjects_noise_scale >= 0, f"`subjects_noise_scale` must be non-negative (got {params.subjects_noise_scale})"
    assert params.subjects_noise_df > 0, f"`subjects_noise_df` must be positive (got {params.subjects_noise_df})"
    # make sure distances are in condensed form
    gt_distances = convert_to_condensed(gt_distances)
    N = num_obs_y(gt_distances)
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
    results = ExperimentResults(
        datetime.now(), all_observations, all_n_obs.astype(np.int8), subject_noises
    )
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
    Returns ``(observations, n_obs)`` as condensed (1-D) vectors. Unmeasured distances are represented as 0.
    """
    assert subject_noise >= 0, "`subject_noise` must be non-negative"
    assert num_trials > 0, "`num_trials` must be positive"
    if gt_distances.ndim != 1:  # accept a square matrix, but skip the copy for the common condensed case
        gt_distances = convert_to_condensed(gt_distances)
    N = num_obs_y(gt_distances)
    assert 0 < images_per_trial < N, "`images_per_trial` must be between 0 and `N`"
    # Work directly in condensed form: the buffers are ~half the size of the equivalent
    # square matrices and no per-trial squareform round-trip is needed. This is bit-for-bit
    # identical to accumulating into a square matrix and condensing afterwards, because
    # squareform only reshapes (it performs no arithmetic).
    observations = np.zeros_like(gt_distances)
    n_obs = np.zeros_like(gt_distances)
    # Pre-compute the upper-triangular pair ordering once; `np.triu_indices` enumerates
    # pairs in the same order as `itertools.combinations(range(k), 2)`, so drawing the
    # per-trial noise as a single block reproduces the scalar loop's RNG stream exactly:
    # NumPy's Generator fills `normal(size=n)` from the same sequence as `n` scalar calls.
    pair_rows, pair_cols = np.triu_indices(images_per_trial, k=1)
    n_pairs = pair_rows.size
    for _t in trange(num_trials, desc="Simulating trials", disable=not verbose):
        # Draw the trial's images from the simulation's seeded Generator so the whole
        # simulation is reproducible from its seed (the pre-refactor code used the global
        # `np.random`, which left image selection un-seeded and non-reproducible).
        selected_indices = rng.choice(N, size=images_per_trial, replace=False)
        cond_idx = _condensed_pair_indices(selected_indices[pair_rows], selected_indices[pair_cols], N)
        # Cast noise to float32 before adding. The scalar pre-refactor code drew noise via
        # `rng.normal(...)` (no size), which returns a Python float; NEP-50 weak promotion then
        # performs `float32 + python_float` in float32. A block draw returns a float64 array, so
        # we must down-cast it to reproduce the original float32 arithmetic bit-for-bit.
        noise = rng.normal(0, scale=subject_noise, size=n_pairs).astype(np.float32)
        noisy_distances = np.maximum(0, gt_distances[cond_idx] + noise)  # ensure non-negativity
        # Pairs within a trial are unique (distinct selected images), so each condensed index
        # appears at most once and these assignments accumulate identically to the scalar loop.
        observations[cond_idx] += noisy_distances
        n_obs[cond_idx] += 1
    return observations, n_obs


def _condensed_pair_indices(idx_i: np.ndarray, idx_j: np.ndarray, N: int) -> np.ndarray:
    """Map unordered image-index pairs to their position in a length-N condensed distance vector.

    Matches scipy's ``squareform`` ordering: for ``lo < hi`` the condensed index is
    ``lo*N - lo*(lo+1)//2 + (hi - lo - 1)``.
    """
    lo = np.minimum(idx_i, idx_j)
    hi = np.maximum(idx_i, idx_j)
    return lo * N - (lo * (lo + 1)) // 2 + (hi - lo - 1)


def _draw_subject_noises(
        df: int, mu_noise: float, n_subjects: int, rng: np.random.Generator
) -> np.ndarray:
    assert df > 0, "`df` must be positive"
    assert mu_noise >= 0, "`mu_noise` must be non-negative"
    assert n_subjects > 0, "`n_subjects` must be positive"
    if mu_noise == 0:
        return np.zeros(n_subjects)
    raw_variability = np.abs(rng.standard_t(df, size=n_subjects))
    scaled_noises = mu_noise * raw_variability / np.mean(raw_variability)
    return scaled_noises
