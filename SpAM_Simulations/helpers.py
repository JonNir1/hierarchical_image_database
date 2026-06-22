from __future__ import annotations

import numpy as np
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
