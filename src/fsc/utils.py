"""Utility functions for FSC."""
import numpy as np

def validate_symmetric_matrix(matrix, name):
    """Check that a matrix is square, symmetric, and finite."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square.")
    if not np.allclose(matrix, matrix.T, equal_nan=True):
        raise ValueError(f"{name} must be symmetric.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix