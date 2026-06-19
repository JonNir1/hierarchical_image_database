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
        if not (np.allclose(matrix, matrix.T) or (np.isnan(matrix) == np.isnan(matrix.T)).all()):
            raise ValueError(f"Input matrix must be symmetric")
        return squareform(matrix, checks=False)
    else:
        raise ValueError(f"Input must be either a 1D vector or a 2D matrix, got shape {matrix.shape}")
