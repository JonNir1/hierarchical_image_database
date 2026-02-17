import numpy as np
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from SpAM_Simulations.experiment import ExperimentResults
from SpAM_Simulations.helpers import convert_to_condensed


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

    # extract per-image statistics
    sq_n_pairwise_obs = squareform(n_pairwise_obs, checks=False)  # convert to square form
    n_images = sq_n_pairwise_obs.shape[0]
    n_img_obs = np.sum(sq_n_pairwise_obs > 0, axis=0)  # count number of observed pairs for each image
    avg_img_obs = np.mean(n_img_obs)
    percent_img_obs = np.mean(n_img_obs > 0) * 100

    # Calculate number of connected components in the image pair graph
    adjacency_matrix = squareform(n_pairwise_obs > 0, checks=False).astype(int)  # convert to adjacency matrix
    num_components, _ = connected_components(adjacency_matrix, directed=False)
    return {
        "num_images": n_images,
        "average_img_obs": avg_img_obs,
        "img_coverage": percent_img_obs,
        "num_pairs": num_pairs,
        "average_pair_obs": avg_pairwise_obs,
        "pair_coverage": percent_pairwise_obs,
        "num_connected_components": num_components
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
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_dists = np.where(n_obs > 0, dists / n_obs, np.nan)  # avoid division by zero
    return mean_dists
