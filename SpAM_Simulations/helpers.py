import numpy as np
from scipy.spatial.distance import squareform


def convert_to_condensed(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        # convert to matrix form to make sure it's square and symmetric
        return convert_to_condensed(squareform(matrix, checks=False))
    elif matrix.ndim == 2:
        # convert to condensed form, ensuring it's square and symmetric
        if not (matrix.shape[0] == matrix.shape[1]):
            raise ValueError(f"Input matrix must be square, got shape {matrix.shape}")
        if not (np.allclose(matrix, matrix.T) or (np.isnan(matrix) == np.isnan(matrix.T)).all()):
            raise ValueError(f"Input matrix must be symmetric")
        return squareform(matrix, checks=False)
    else:
        raise ValueError(f"Input must be either a 1D vector or a 2D matrix, got shape {matrix.shape}")
