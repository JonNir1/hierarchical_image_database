"""Task-v2.3 SpAM experiment simulation, matching SpAM_Task's per-subject trial design.

Unlike `experiment.py` (which draws `images_per_trial` images uniformly at random from the
whole pool on every trial), each subject here is first restricted to their own random
`n_unique`-image subset, allocated into trials via `design.build_trial_lists` exactly like
the real task - including the `frac_images_repeated` mechanic that shows a fraction of
images in 2 distinct trials. The noise model and global condensed-index bookkeeping reuse
`experiment.py` unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.distance import num_obs_y

from SpAM_Simulations.design import build_trial_lists, compute_design_counts
from SpAM_Simulations.experiment import _condensed_pair_indices, _draw_subject_noises
from SpAM_Simulations.helpers import convert_to_condensed, mean_from_sum_and_count

TaskV2_3ExperimentParameters = NamedTuple("TaskV2_3ExperimentParameters", [
    ("num_subjects", int),
    ("trials_per_subject", int),
    ("images_per_trial", int),
    ("subjects_noise_scale", float),
    ("subjects_noise_df", int),
    ("frac_images_repeated", float),
])

TaskV2_3ExperimentResults = NamedTuple("TaskV2_3ExperimentResults", [
    ("run_time", datetime),
    ("distances", np.ndarray),
    ("num_obs", np.ndarray),
    ("subject_noises", np.ndarray),
    ("subject_snr", np.ndarray),
])


def simulate_task_v2_3_experiment(
        params: TaskV2_3ExperimentParameters,
        gt_distances: np.ndarray,
        rng: np.random.Generator,
        verbose: bool = True,
) -> Tuple[TaskV2_3ExperimentParameters, TaskV2_3ExperimentResults]:
    """
    Simulates distance observations from multiple subjects, each restricted to their own
    random `n_unique`-image subset allocated into trials per `design.build_trial_lists`.
    Each subject has their own noise level drawn from a scaled half-t distribution (reusing
    `experiment._draw_subject_noises` unchanged). Unmeasured distances are NaN.
    """
    assert params.num_subjects > 0, f"`num_subjects` must be positive (got {params.num_subjects})"
    assert params.trials_per_subject > 0, f"`trials_per_subject` must be positive (got {params.trials_per_subject})"
    assert params.subjects_noise_scale >= 0, f"`subjects_noise_scale` must be non-negative (got {params.subjects_noise_scale})"
    assert params.subjects_noise_df > 0, f"`subjects_noise_df` must be positive (got {params.subjects_noise_df})"
    gt_distances = convert_to_condensed(gt_distances)
    N = num_obs_y(gt_distances)
    n_unique, n_double = compute_design_counts(
        params.trials_per_subject, params.images_per_trial, params.frac_images_repeated
    )
    assert n_unique <= N, (
        f"`n_unique`(={n_unique}, derived from trials_per_subject/images_per_trial/"
        f"frac_images_repeated) exceeds the image pool size (N={N})"
    )

    all_observations = np.zeros_like(gt_distances)
    all_n_obs = np.zeros_like(gt_distances)
    subject_snr = np.empty(params.num_subjects, dtype=np.float64)
    subject_noises = _draw_subject_noises(
        params.subjects_noise_df,
        params.subjects_noise_scale * gt_distances.std(),  # scale subject noise by GT noise to ensure reasonable range
        params.num_subjects,
        rng
    )
    for s in trange(params.num_subjects, desc="Simulating subjects", disable=not verbose):
        observations, n_obs, snr = simulate_task_v2_3_single_subject(
            subject_noise=subject_noises[s],
            t=params.trials_per_subject,
            k=params.images_per_trial,
            n_unique=n_unique,
            n_double=n_double,
            gt_distances=gt_distances,
            rng=rng,
            verbose=False
        )
        all_observations += observations
        all_n_obs += n_obs
        subject_snr[s] = snr
    all_observations = np.where(     # ensure unmeasured distances are NaN
        all_observations > 0, all_observations, np.nan
    )
    results = TaskV2_3ExperimentResults(
        datetime.now(), all_observations, all_n_obs.astype(np.int8), subject_noises, subject_snr
    )
    return params, results


def simulate_task_v2_3_single_subject(
        subject_noise: float,
        t: int,
        k: int,
        n_unique: int,
        n_double: int,
        gt_distances: np.ndarray,
        rng: np.random.Generator,
        verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simulates one subject: draws their `n_unique`-image subset, allocates it into `t`
    trials of `k` images via `design.build_trial_lists`, and simulates noisy pairwise
    judgments per trial (same Gaussian noise model as `experiment.simulate_single_subject`).
    Returns ``(observations, n_obs, snr)``; the first two are condensed (1-D) vectors with
    unmeasured distances represented as 0, the third is this subject's SNR heuristic (NaN
    if they had no within-subject-repeated pairs to estimate it from).
    """
    assert subject_noise >= 0, "`subject_noise` must be non-negative"
    if gt_distances.ndim != 1:
        gt_distances = convert_to_condensed(gt_distances)
    N = num_obs_y(gt_distances)
    assert n_unique <= N, f"`n_unique`(={n_unique}) must not exceed the image pool size (N={N})"

    active_indices = rng.choice(N, size=n_unique, replace=False)
    trials = build_trial_lists(active_indices, t, k, n_double, rng)
    candidate_pairs = _find_candidate_repeated_pairs(trials, N)

    observations = np.zeros_like(gt_distances)
    n_obs = np.zeros_like(gt_distances)
    pair_rows, pair_cols = np.triu_indices(k, k=1)
    repeated_values: Dict[int, List[float]] = defaultdict(list)
    for trial_images in tqdm(trials, desc="Simulating trials", disable=not verbose):
        cond_idx = _condensed_pair_indices(trial_images[pair_rows], trial_images[pair_cols], N)
        noise = rng.normal(0, scale=subject_noise, size=cond_idx.size).astype(np.float32)
        noisy_distances = np.maximum(0, gt_distances[cond_idx] + noise)  # ensure non-negativity
        observations[cond_idx] += noisy_distances
        n_obs[cond_idx] += 1
        if candidate_pairs.size:
            matches = np.isin(cond_idx, candidate_pairs)
            if matches.any():
                for g, v in zip(cond_idx[matches], noisy_distances[matches]):
                    repeated_values[int(g)].append(float(v))

    snr = _compute_subject_snr(observations, n_obs, repeated_values)
    return observations, n_obs, snr


def _find_candidate_repeated_pairs(trials: List[np.ndarray], N: int) -> np.ndarray:
    """Global condensed indices of image pairs observed twice by this subject.

    Only "double" images (appearing in exactly 2 of the subject's trials) can repeat a
    pair - a "single" image, by construction, contributes to only one trial. Two double
    images that happen to share both of their assigned trials are therefore observed
    together twice, giving the independent noisy measurements the SNR heuristic needs.
    """
    appearances: Dict[int, List[int]] = defaultdict(list)
    for trial_idx, trial in enumerate(trials):
        for img in trial.tolist():
            appearances[img].append(trial_idx)

    trial_pair_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for img, trial_idxs in appearances.items():
        if len(trial_idxs) == 2:
            trial_pair_groups[tuple(sorted(trial_idxs))].append(img)

    img_i_list, img_j_list = [], []
    for imgs in trial_pair_groups.values():
        if len(imgs) >= 2:
            for img_i, img_j in combinations(imgs, 2):
                img_i_list.append(img_i)
                img_j_list.append(img_j)
    if not img_i_list:
        return np.empty(0, dtype=np.int64)
    return _condensed_pair_indices(np.asarray(img_i_list), np.asarray(img_j_list), N)


def _compute_subject_snr(
        observations: np.ndarray, n_obs: np.ndarray, repeated_values: Dict[int, List[float]]
) -> float:
    """``SNR = sigma_d / mean(|delta_d|)``.

    ``sigma_d``: std of this subject's own observed (noisy) per-pair mean distances.
    ``delta_d``: difference between the two independent noisy measurements of the same
    image pair, for pairs this subject observed in exactly two distinct trials (see
    `_find_candidate_repeated_pairs`). NaN if the subject has no such repeated pairs.
    """
    if not repeated_values:
        return np.nan
    deltas = []
    for values in repeated_values.values():
        assert len(values) == 2, f"expected exactly 2 repeated measurements, got {len(values)}"
        deltas.append(abs(values[0] - values[1]))
    # Keep these as numpy float64 (not Python float) so a zero denominator divides to `inf`
    # via numpy's IEEE-754 semantics instead of raising ZeroDivisionError.
    mean_abs_delta = np.mean(deltas)
    subject_mean_dists = mean_from_sum_and_count(observations, n_obs)
    with np.errstate(invalid='ignore'):
        sigma_d = np.nanstd(subject_mean_dists, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(sigma_d / mean_abs_delta)
