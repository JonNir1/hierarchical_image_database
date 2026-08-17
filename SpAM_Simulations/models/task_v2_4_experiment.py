"""Task-v2.4 SpAM experiment simulation, adding whole-trial repeats to the task-v2.3 design.

Task-v2.4 has two repetition levers (see ``SpAM_Task/js/trial_generator.js``):

* ``frac_images_repeated`` (``r``) - a fraction of each subject's images shown in two
  distinct-composition trials (cross-context reliability). Already modelled by
  ``task_v2_3_experiment.py`` and reused here unchanged.
* ``frac_trials_repeated`` (``fr``) - a fraction of each subject's *whole* trials shown again
  verbatim (same ``k``-image set), giving a direct within-subject **test-retest** reliability
  of the arrangement response itself. New in this module.

Each subject sees ``t = trials_per_subject`` trials total: ``t_distinct = t - round(fr*t)``
genuinely distinct trials plus ``n_repeats = round(fr*t)`` verbatim repeats. Repeats re-draw
their noisy distances (a fresh, independent arrangement), so the original/repeat pair gives an
independent test-retest measurement.

RNG-ordering contract (per subject): draw image subset -> ``build_trial_lists`` for
``t_distinct`` -> simulate the distinct trials and snapshot the SNR heuristic from that
distinct-only state -> *then* select which trials repeat -> simulate the repeats. Consequence:
with ``frac_trials_repeated = 0`` the output is bit-exact to ``task_v2_3_experiment`` (no extra
RNG is consumed and no repeats run), and ``subject_snr`` stays comparable to v2.3 as ``fr``
varies because it never sees the repeat-trial observations. Repeat trials *do* contribute to
the returned aggregate ``distances``/``num_obs`` (they are real completed trials).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from scipy.spatial.distance import num_obs_y
from scipy.stats import spearmanr
from tqdm import tqdm, trange

from SpAM_Simulations.models.design import (
    build_trial_lists, compute_design_counts, distinct_trial_count, select_repeat_trials
)
from SpAM_Simulations.models.experiment import _condensed_pair_indices, _draw_subject_noises
from SpAM_Simulations.core.helpers import convert_to_condensed
from SpAM_Simulations.models.task_v2_3_experiment import (
    _find_candidate_repeated_pairs, _compute_subject_snr
)

TaskV2_4ExperimentParameters = NamedTuple("TaskV2_4ExperimentParameters", [
    ("num_subjects", int),
    ("trials_per_subject", int),
    ("images_per_trial", int),
    ("subjects_noise_scale", float),
    ("subjects_noise_df", int),
    ("frac_images_repeated", float),
    ("frac_trials_repeated", float),
])

TaskV2_4ExperimentResults = NamedTuple("TaskV2_4ExperimentResults", [
    ("run_time", datetime),
    ("distances", np.ndarray),
    ("num_obs", np.ndarray),
    ("subject_noises", np.ndarray),
    ("subject_snr", np.ndarray),
    ("subject_test_retest", np.ndarray),
])


def simulate_task_v2_4_experiment(
        params: TaskV2_4ExperimentParameters,
        gt_distances: np.ndarray,
        rng: np.random.Generator,
        verbose: bool = True,
) -> Tuple[TaskV2_4ExperimentParameters, TaskV2_4ExperimentResults]:
    """Simulate multiple subjects under the task-v2.4 design (v2.3 + whole-trial repeats).

    Same per-subject image-subset/trial allocation and noise model as
    ``task_v2_3_experiment``, plus ``n_repeats = round(frac_trials_repeated * trials_per_subject)``
    verbatim trial repeats per subject. Returns the v2.3 fields plus ``subject_test_retest``
    (per-subject mean test-retest Spearman; NaN for subjects with no repeats). Unmeasured
    distances are NaN.
    """
    assert params.num_subjects > 0, f"`num_subjects` must be positive (got {params.num_subjects})"
    assert params.trials_per_subject > 0, f"`trials_per_subject` must be positive (got {params.trials_per_subject})"
    assert params.subjects_noise_scale >= 0, f"`subjects_noise_scale` must be non-negative (got {params.subjects_noise_scale})"
    assert params.subjects_noise_df > 0, f"`subjects_noise_df` must be positive (got {params.subjects_noise_df})"
    assert 0 <= params.frac_trials_repeated < 1, (
        f"`frac_trials_repeated` must be in [0, 1) (got {params.frac_trials_repeated})"
    )
    gt_distances = convert_to_condensed(gt_distances)
    N = num_obs_y(gt_distances)
    t_distinct = distinct_trial_count(params.trials_per_subject, params.frac_trials_repeated)
    n_repeats = params.trials_per_subject - t_distinct
    n_unique, n_double = compute_design_counts(
        t_distinct, params.images_per_trial, params.frac_images_repeated
    )
    assert n_unique <= N, (
        f"`n_unique`(={n_unique}, derived from t_distinct/images_per_trial/frac_images_repeated) "
        f"exceeds the image pool size (N={N})"
    )

    all_observations = np.zeros_like(gt_distances)
    all_n_obs = np.zeros_like(gt_distances)
    subject_snr = np.empty(params.num_subjects, dtype=np.float64)
    subject_test_retest = np.empty(params.num_subjects, dtype=np.float64)
    subject_noises = _draw_subject_noises(
        params.subjects_noise_df,
        params.subjects_noise_scale * gt_distances.std(),  # scale subject noise by GT noise to ensure reasonable range
        params.num_subjects,
        rng
    )
    for s in trange(params.num_subjects, desc="Simulating subjects", disable=not verbose):
        observations, n_obs, snr, test_retest = simulate_task_v2_4_single_subject(
            subject_noise=subject_noises[s],
            t_distinct=t_distinct,
            k=params.images_per_trial,
            n_unique=n_unique,
            n_double=n_double,
            n_repeats=n_repeats,
            gt_distances=gt_distances,
            rng=rng,
            verbose=False
        )
        all_observations += observations
        all_n_obs += n_obs
        subject_snr[s] = snr
        subject_test_retest[s] = test_retest
    all_observations = np.where(     # ensure unmeasured distances are NaN
        all_observations > 0, all_observations, np.nan
    )
    results = TaskV2_4ExperimentResults(
        datetime.now(), all_observations, all_n_obs.astype(np.int8),
        subject_noises, subject_snr, subject_test_retest
    )
    return params, results


def simulate_task_v2_4_single_subject(
        subject_noise: float,
        t_distinct: int,
        k: int,
        n_unique: int,
        n_double: int,
        n_repeats: int,
        gt_distances: np.ndarray,
        rng: np.random.Generator,
        verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Simulate one subject: ``t_distinct`` distinct trials plus ``n_repeats`` verbatim repeats.

    Distinct trials are allocated exactly as in ``task_v2_3_experiment`` (subset draw +
    ``build_trial_lists`` + per-trial Gaussian noise) and the SNR heuristic is computed from
    that distinct-only state. ``n_repeats`` singles-only trials are then re-presented with
    freshly drawn noise; each repeat's pairwise-distance vector is correlated (Spearman) with
    its original presentation to give the test-retest reliability.

    Returns ``(observations, n_obs, snr, test_retest)``: the first two are condensed (1-D)
    vectors with unmeasured pairs as 0 (repeat trials included in the accumulation); ``snr`` is
    the v2.3 within-subject SNR heuristic (NaN if no within-subject-repeated pairs);
    ``test_retest`` is the mean per-repeat Spearman correlation (NaN if ``n_repeats == 0``).
    """
    assert subject_noise >= 0, "`subject_noise` must be non-negative"
    assert n_repeats >= 0, "`n_repeats` must be non-negative"
    if gt_distances.ndim != 1:
        gt_distances = convert_to_condensed(gt_distances)
    N = num_obs_y(gt_distances)
    assert n_unique <= N, f"`n_unique`(={n_unique}) must not exceed the image pool size (N={N})"

    active_indices = rng.choice(N, size=n_unique, replace=False)
    trials = build_trial_lists(active_indices, t_distinct, k, n_double, rng)
    candidate_pairs = _find_candidate_repeated_pairs(trials, N)  # distinct trials only -> SNR

    observations = np.zeros_like(gt_distances)
    n_obs = np.zeros_like(gt_distances)
    pair_rows, pair_cols = np.triu_indices(k, k=1)
    repeated_values: Dict[int, List[float]] = defaultdict(list)
    distinct_noisy: List[np.ndarray] = []  # per-trial noisy vectors, indexed like `trials`
    for trial_images in tqdm(trials, desc="Simulating trials", disable=not verbose):
        cond_idx, noisy_distances = _simulate_trial(
            trial_images, pair_rows, pair_cols, N, gt_distances, subject_noise, observations, n_obs, rng
        )
        distinct_noisy.append(noisy_distances)
        if candidate_pairs.size:
            matches = np.isin(cond_idx, candidate_pairs)
            if matches.any():
                for g, v in zip(cond_idx[matches], noisy_distances[matches]):
                    repeated_values[int(g)].append(float(v))

    # SNR from the distinct-only state (snapshot before any repeat-trial observations land).
    snr = _compute_subject_snr(observations, n_obs, repeated_values)

    # Whole-trial repeats: re-present `n_repeats` singles-only trials with fresh noise.
    repeat_idxs = select_repeat_trials(trials, n_repeats, rng)
    retest_corrs: List[float] = []
    for orig_idx in repeat_idxs:
        trial_images = trials[orig_idx]
        _, repeat_noisy = _simulate_trial(
            trial_images, pair_rows, pair_cols, N, gt_distances, subject_noise, observations, n_obs, rng
        )
        # original and repeat share the same `trial_images` order, so their per-pair vectors
        # are aligned element-wise and can be correlated directly.
        retest_corrs.append(_trial_test_retest(distinct_noisy[orig_idx], repeat_noisy))

    test_retest = float(np.nanmean(retest_corrs)) if retest_corrs else np.nan
    return observations, n_obs, snr, test_retest


def _simulate_trial(
        trial_images: np.ndarray,
        pair_rows: np.ndarray,
        pair_cols: np.ndarray,
        N: int,
        gt_distances: np.ndarray,
        subject_noise: float,
        observations: np.ndarray,
        n_obs: np.ndarray,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one trial's noisy pairwise distances, accumulate them, and return them.

    Identical noise model to ``task_v2_3_experiment``/``experiment.simulate_single_subject``:
    a single block of ``float32`` Gaussian noise added to the ground-truth distances of this
    trial's within-trial pairs, clipped at 0. Mutates `observations`/`n_obs` in place and
    returns ``(cond_idx, noisy_distances)`` so the caller can do SNR / test-retest bookkeeping.
    """
    cond_idx = _condensed_pair_indices(trial_images[pair_rows], trial_images[pair_cols], N)
    noise = rng.normal(0, scale=subject_noise, size=cond_idx.size).astype(np.float32)
    noisy_distances = np.maximum(0, gt_distances[cond_idx] + noise)  # ensure non-negativity
    observations[cond_idx] += noisy_distances
    n_obs[cond_idx] += 1
    return cond_idx, noisy_distances


def _trial_test_retest(orig: np.ndarray, repeat: np.ndarray) -> float:
    """Spearman correlation between a trial's original and repeat pairwise-distance vectors.

    Returns NaN if either vector is constant (Spearman undefined), e.g. a degenerate single-pair
    trial. With noise, two independent presentations of the same images give a finite r < 1;
    with zero noise both vectors equal the ground-truth distances, so r == 1.
    """
    if np.ptp(orig) == 0 or np.ptp(repeat) == 0:
        return np.nan
    return float(spearmanr(orig, repeat).statistic)
