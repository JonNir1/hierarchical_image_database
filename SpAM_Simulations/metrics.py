from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from SpAM_Simulations.experiment import ExperimentResults
from SpAM_Simulations.helpers import convert_to_condensed, mean_from_sum_and_count


def coverage(exp_results: ExperimentResults) -> dict:
    """
    Calculates coverage metrics from the number of observations per image pair.
    Coverage metrics include:
        - Num images (N)
        - Average number of observations per image (average image observations)
        - Percentage of images with at least one observation (image coverage)
        - Num image pairs (N(N-1)/2)
        - Average number of observations per image pair (average pair observations)
        - Percentage of image pairs with at least one observation (pair coverage)
        - Number of connected components in the image pair graph (where edges exist if at least one observation)
    :param exp_results: ExperimentResults object containing the number of observations per image pair
    """
    # extract pairwise statistics
    n_pairwise_obs = convert_to_condensed(exp_results.num_obs)  # convert to condensed form if needed
    num_pairs = n_pairwise_obs.shape[0]
    avg_pairwise_obs = np.mean(n_pairwise_obs)
    percent_pairwise_obs = np.mean(n_pairwise_obs > 0) * 100

    # build the square form once and reuse it for the per-image stats and the adjacency graph
    observed = squareform(n_pairwise_obs, checks=False) > 0  # boolean image-pair adjacency
    n_images = observed.shape[0]
    n_img_obs = np.sum(observed, axis=0)  # count number of observed pairs for each image
    avg_img_obs = np.mean(n_img_obs)
    percent_img_obs = np.mean(n_img_obs > 0) * 100

    # Calculate number of connected components in the image pair graph
    num_components, _ = connected_components(observed.astype(int), directed=False)
    return {
        "num_images": n_images,
        "average_img_obs": avg_img_obs,
        "img_coverage": percent_img_obs,
        "num_pairs": num_pairs,
        "average_pair_obs": avg_pairwise_obs,
        "pair_coverage": percent_pairwise_obs,
        "num_connected_components": num_components
    }


def snr_summary(exp_results) -> dict:
    """Summary stats of a task-v2.3 experiment's per-subject SNR heuristic (`subject_snr`).

    `frac_nan_snr` surfaces subjects with no within-subject-repeated pairs (SNR undefined
    for them, see `task_v2_3_experiment._compute_subject_snr`); mean/median are computed
    over the remaining values (which may include `inf` for noiseless subjects).
    :param exp_results: a `TaskV2_3ExperimentResults` (has a `subject_snr` field)
    """
    snr = np.asarray(exp_results.subject_snr, dtype=np.float64)
    valid = snr[~np.isnan(snr)]
    return {
        "mean_snr": float(np.mean(valid)) if valid.size else np.nan,
        "median_snr": float(np.median(valid)) if valid.size else np.nan,
        "frac_nan_snr": float(np.mean(np.isnan(snr))),
    }


def test_retest_summary(exp_results) -> dict:
    """Summary stats of a task-v2.4 experiment's per-subject test-retest reliability.

    `subject_test_retest` holds each subject's mean Spearman correlation between the original
    and repeat presentations of their repeated trials (the `frac_trials_repeated` lever).
    `frac_nan_test_retest` surfaces subjects with no repeated trials (reliability undefined for
    them, see `task_v2_4_experiment._compute...`); mean/median are over the remaining values.
    :param exp_results: a `TaskV2_4ExperimentResults` (has a `subject_test_retest` field)
    """
    rel = np.asarray(exp_results.subject_test_retest, dtype=np.float64)
    valid = rel[~np.isnan(rel)]
    return {
        "mean_test_retest": float(np.mean(valid)) if valid.size else np.nan,
        "median_test_retest": float(np.median(valid)) if valid.size else np.nan,
        "frac_nan_test_retest": float(np.mean(np.isnan(rel))),
    }


def spearman_correlation(exp1: ExperimentResults, exp2: ExperimentResults) -> float:
    """
    Calculates the Spearman rank correlation between the mean distances of two experiments.
    :param exp1: First experiment results
    :param exp2: Second experiment results
    :return: Spearman rank correlation coefficient
    """
    mean_dists1 = _calculate_mean_distances(exp1)
    mean_dists2 = _calculate_mean_distances(exp2)
    # only consider pairs that have at least two observation in both experiments
    valid_mask = ~np.isnan(mean_dists1) & ~np.isnan(mean_dists2)
    if np.sum(valid_mask) < 2:
        raise ValueError("Experiments don't have enough overlapping observed pairs to calculate Spearman correlation.")
    corr, _ = spearmanr(mean_dists1[valid_mask], mean_dists2[valid_mask])
    return corr


def _calculate_mean_distances(exp_results: ExperimentResults) -> np.ndarray:
    dists = convert_to_condensed(exp_results.distances)
    n_obs = convert_to_condensed(exp_results.num_obs)
    return mean_from_sum_and_count(dists, n_obs)
