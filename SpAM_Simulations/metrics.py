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
        - Num image pairs (N(N-1)/2)
        - Average number of observations per image pair (average observations)
        - Proportion of image pairs with at least k observations for k=1,3,5 (coverage@k)
        - Number of connected components in the image pair graph (where edges exist if at least one observation)
    :param exp_results: ExperimentResults object containing the number of observations per image pair
    """
    n_obs = convert_to_condensed(exp_results.num_obs)  # convert to condensed form if needed
    num_pairs = n_obs.shape[0]
    avg_obs = np.mean(n_obs)
    coverage_at_1 = np.mean(n_obs > 0)
    coverage_at_3 = np.mean(n_obs >= 3)
    coverage_at_5 = np.mean(n_obs >= 5)
    # Calculate number of connected components in the image pair graph
    adjacency_matrix = squareform(n_obs > 0, checks=False).astype(int)  # convert to adjacency matrix
    num_components, _ = connected_components(adjacency_matrix, directed=False)
    return {
        "num_pairs": num_pairs,
        "average_obs_count": avg_obs,
        "coverage": coverage_at_1,
        "coverage@3": coverage_at_3,
        "coverage@5": coverage_at_5,
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
