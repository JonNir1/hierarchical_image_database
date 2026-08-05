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
    out = {
        "mean_test_retest": float(np.mean(valid)) if valid.size else np.nan,
        "median_test_retest": float(np.median(valid)) if valid.size else np.nan,
        "frac_nan_test_retest": float(np.mean(np.isnan(rel))),
    }
    # Task-v3 additionally reports a Procrustes M^2 test-retest (LOWER = better); absent for v2.4.
    m2 = getattr(exp_results, "subject_test_retest_procrustes", None)
    if m2 is not None:
        m2 = np.asarray(m2, dtype=np.float64)
        v2 = m2[~np.isnan(m2)]
        out["mean_test_retest_procrustes"] = float(np.mean(v2)) if v2.size else np.nan
        out["median_test_retest_procrustes"] = float(np.median(v2)) if v2.size else np.nan
    return out


def screening_summary(exp_results) -> dict:
    """Recruitment cost and retained-cohort precision for a task-v4 (screened) experiment.

    `n_candidates_screened` is how many candidates had to be simulated to retain
    `num_subjects` - the Prolific cost of the screening threshold, and the quantity that has to be
    weighed against the reliability it buys. `mean_subject_noise` describes the **retained**
    cohort's placement precision: screening truncates the heavy upper tail of the `|t(df)|` noise
    population, so a stricter threshold should push this below the configured
    `subjects_noise_scale`. `frac_nan_*` is absent here because both fields are always defined.

    :param exp_results: a `TaskV4ExperimentResults` (has `n_candidates_screened`)
    """
    noises = np.asarray(exp_results.subject_noises, dtype=np.float64)
    return {
        "n_candidates_screened": int(exp_results.n_candidates_screened),
        "screening_pass_rate": float(exp_results.screening_pass_rate),
        "mean_subject_noise": float(np.mean(noises)) if noises.size else np.nan,
        "median_subject_noise": float(np.median(noises)) if noises.size else np.nan,
        "max_subject_noise": float(np.max(noises)) if noises.size else np.nan,
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


def classical_mds_eigenvalues(condensed_distances: np.ndarray) -> np.ndarray:
    """Classical-MDS (PCoA) eigenvalues of a condensed distance vector, descending.

    Double-centres the squared distance matrix and returns the eigenvalues of the resulting
    Gram matrix - the variance carried by each principal coordinate. Missing pairs (NaN) are
    mean-imputed first so the matrix is complete; this is a coarse diagnostic, not a fit. Used by
    :func:`effective_rank` to check that the task-v3 aggregate spans more than the 2 dimensions of
    any single trial's arrangement (otherwise MDS could never recover the full ground-truth space).
    """
    d = convert_to_condensed(condensed_distances).astype(np.float64).copy()
    nan = np.isnan(d)
    if nan.any():
        d[nan] = np.nanmean(d)
    sq = squareform(d) ** 2
    n = sq.shape[0]
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring @ sq @ centring
    return np.sort(np.linalg.eigvalsh(gram))[::-1]


def effective_rank(condensed_distances: np.ndarray) -> float:
    """Effective rank (entropy of the normalised positive eigenvalue spectrum) of a distance set.

    ``exp(-sum p_i log p_i)`` over ``p_i = lambda_i / sum(lambda)`` for the positive classical-MDS
    eigenvalues. Equals the dimensionality for an isotropic ``k``-cube and collapses toward small
    values for a low-rank (e.g. rank-2) configuration - so a task-v3 aggregate whose effective rank
    rises toward the ground-truth ``D`` as subjects accumulate is direct evidence the per-trial 2-D
    slices are tiling the full space (plan verification #4).
    """
    eig = classical_mds_eigenvalues(condensed_distances)
    pos = eig[eig > 0]
    if pos.size == 0:
        return 0.0
    p = pos / pos.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


# --------------------------------------------------------------------------- closest pairs
def topk_mask(distances: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask of the ``frac`` **smallest** entries of a condensed distance vector.

    The smallest distances are the closest pairs, i.e. the 'too similar' candidates. ``k`` is
    floored at 1 so a tiny ``frac`` still selects something rather than returning an empty set.

    This is the single definition of "the closest ``frac`` of pairs" in the codebase. It used to be
    reimplemented in three places (rep-vs-rep Jaccard, split-half scoring, recovery-vs-GT), which
    risked them silently disagreeing about which pairs count as closest at a given fraction.
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    n = distances.shape[0]
    k = max(1, int(round(frac * n)))
    mask = np.zeros(n, dtype=bool)
    mask[np.argpartition(distances, k - 1)[:k]] = True
    return mask


def topk_similar_jaccard(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    """Jaccard overlap of the closest-``frac`` pair sets of two condensed distance vectors.

    Both vectors must index the same pair set, so this compares *which* pairs each side flags as
    closest. Returns ``|A n B| / |A u B|``; the two sets are the same size, so precision, recall and
    the overlap coefficient are all monotone transforms of it and only one number is reported.

    Note what this does **not** measure: if three images form one tight cluster, which of their three
    pairs is "closest" flips with noise, and every flip counts as disagreement here even though all
    of them support the same practical decision. It therefore understates usable structure whenever
    the question is about groups rather than pairs.
    """
    ma, mb = topk_mask(a, frac), topk_mask(b, frac)
    union = int(np.count_nonzero(ma | mb))
    return int(np.count_nonzero(ma & mb)) / union if union else np.nan
