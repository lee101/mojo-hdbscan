"""Mutual-reachability graph construction."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


def mutual_reachability(distance_matrix, min_points=5, alpha=1.0):
    """Compute the dense mutual-reachability matrix of a distance matrix."""
    matrix = f64(distance_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    n = matrix.shape[0]
    if n == 0:
        raise ValueError("distance_matrix must not be empty")
    if not np.isfinite(matrix).all():
        raise ValueError("distance_matrix must contain only finite values")
    try:
        requested = int(min_points)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("min_points must be an integer") from error
    if requested != min_points:
        raise ValueError("min_points must be an integer")
    kth = min(n - 1, requested)
    if kth < 0:
        kth += n
    if kth < 0:
        raise ValueError("min_points is out of bounds")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be a finite positive number")
    core = np.empty(n, dtype=np.float64)
    workers = min(n, 8) if n * n >= 262144 else 1
    scratch = np.empty(n * workers, dtype=np.float64)
    lib().mhdb_mutual_reachability(
        addr(matrix), addr(matrix), addr(core), addr(scratch), n, kth, alpha
    )
    return matrix


def sparse_mutual_reachability(
    lil_matrix, min_points=5, alpha=1.0, max_dist=0.0
):
    """Compute mutual reachability for a SciPy LIL sparse distance matrix."""
    from scipy.sparse import lil_matrix as sparse_matrix

    if len(lil_matrix.shape) != 2 or lil_matrix.shape[0] != lil_matrix.shape[1]:
        raise ValueError("lil_matrix must be square")
    if lil_matrix.shape[0] == 0:
        raise ValueError("lil_matrix must not be empty")
    if int(min_points) != min_points or min_points < 1:
        raise ValueError("min_points must be a positive integer")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be a finite positive number")
    result = sparse_matrix(lil_matrix.shape)
    core_distance = np.empty(lil_matrix.shape[0], dtype=np.float64)
    for i, row in enumerate(lil_matrix.data):
        sorted_row = sorted(row)
        if min_points - 1 < len(sorted_row):
            core_distance[i] = sorted_row[min_points - 1]
        else:
            core_distance[i] = np.inf
    scaled = lil_matrix / alpha if alpha != 1.0 else lil_matrix
    rows, cols = scaled.nonzero()
    for i, j in zip(rows, cols):
        distance = max(core_distance[i], core_distance[j], scaled[i, j])
        if np.isfinite(distance):
            result[i, j] = distance
        elif max_dist > 0:
            result[i, j] = max_dist
    return result.tocsr()
