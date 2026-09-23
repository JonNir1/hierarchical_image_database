from __future__ import annotations

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform, num_obs_y


def convert_to_condensed(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        # A 1-D input is already condensed; a vector is a valid condensed distance matrix iff
        # its length is a triangular number, which `num_obs_y` validates (raising otherwise).
        # Return a copy so callers may mutate the result without touching the input, matching
        # the copy semantics of the 2-D `squareform` path below.
        num_obs_y(matrix)
        return matrix.copy()
    elif matrix.ndim == 2:
        # convert to condensed form, ensuring it's square and symmetric
        if not (matrix.shape[0] == matrix.shape[1]):
            raise ValueError(f"Input matrix must be square, got shape {matrix.shape}")
        # `equal_nan=True` requires the finite entries to be close AND the NaN masks to align:
        # a NaN paired with a finite value (asymmetric NaN) is not "close", so it is rejected,
        # while a symmetric NaN pattern (NaN paired with NaN) is accepted.
        if not np.allclose(matrix, matrix.T, equal_nan=True):
            raise ValueError(f"Input matrix must be symmetric")
        return squareform(matrix, checks=False)
    else:
        raise ValueError(f"Input must be either a 1D vector or a 2D matrix, got shape {matrix.shape}")


def mean_from_sum_and_count(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Per-entry mean from accumulated sum/count arrays; NaN wherever `count` is 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(count > 0, total / count, np.nan)


def condensed_rdm_to_weighted_dists(condensed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A condensed RDM (may contain NaN for unobserved pairs) -> (dists, weights),
    both condensed 1-D arrays, ready for run_mds(): dists has NaN replaced by 0,
    weights is 1 where the pair was observed and 0 otherwise. No squareform
    round-trip is needed -- run_mds()'s own convert_to_condensed() accepts a 1-D
    condensed array directly."""
    weights = (~np.isnan(condensed)).astype(np.float64)
    dists = np.nan_to_num(condensed, nan=0.0)
    return dists, weights


def procrustes_m2(conf_a: np.ndarray, conf_b: np.ndarray) -> float:
    """Procrustes disparity M^2 between two point configurations of the same
    items, after optimal translation/rotation/scaling."""
    _, _, disparity = procrustes(conf_a, conf_b)
    return float(disparity)


def double_centered_gram(imputed_condensed: np.ndarray) -> np.ndarray:
    """Double-centred Gram matrix of a COMPLETE (no-NaN) condensed distance
    vector -- the classical (Torgerson) MDS input step. The caller is
    responsible for imputing/filling any missing pairs before calling this;
    this function has no NaN-handling of its own."""
    sq = squareform(imputed_condensed) ** 2
    n = sq.shape[0]
    centring = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * centring @ sq @ centring
