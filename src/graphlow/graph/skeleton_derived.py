"""Derived skeletons: degree, normalized adjacency, Laplacian from adjacency.

Advanced API: these functions take and return scipy.sparse matrices on the
host. Use when you need scipy directly or want to avoid backend memory
(e.g. for scipy solvers, profiling, or custom sparse algebra).
"""

from __future__ import annotations

from enum import StrEnum, auto

import numpy as np
import scipy.sparse as sps


class DerivedMatrixName(StrEnum):
    """
    Derived matrix names.

    - DEGREE : Degree matrix (D)
    - NORMALIZED : Normalized adjacency matrix (D^{-1/2} A D^{-1/2})
    - LAPLACIAN : Laplacian matrix (D - A)
    """

    #: Degree matrix ``D``.
    DEGREE = auto()
    #: Normalized adjacency matrix ``D^{-1/2} A D^{-1/2}``.
    NORMALIZED = auto()
    #: Laplacian matrix ``D - A``.
    LAPLACIAN = auto()


def build_degree_matrix(adj: sps.csr_array) -> sps.csr_array:
    """
    Degree matrix from adjacency matrix.

    Advanced API: operates on scipy.sparse (host). Use when you need
    scipy directly or explicit control over host memory.

    Parameters
    ----------
    adj : scipy.sparse.csr_array
        Adjacency matrix (e.g. PP or CC).

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_points, n_points)`` or ``(n_cells, n_cells)``.
        dtype is np.float64.
    """
    degrees = adj.astype(np.float64).sum(axis=1).ravel()
    return sps.diags(degrees, format="csr")


def build_normalized_adjacency(adj: sps.csr_array) -> sps.csr_array:
    r"""
    Symmetric normalized adjacency matrix \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}.

    Advanced API: operates on scipy.sparse (host). Use when you need
    scipy directly or explicit control over host memory.

    Parameters
    ----------
    adj : scipy.sparse.csr_array
        Original adjacency matrix (e.g. PP or CC).

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_points, n_points)`` or ``(n_cells, n_cells)``.
        dtype is np.float64.
    """
    adj_hat = adj.astype(np.float64)
    degrees_diag = build_degree_matrix(adj_hat).diagonal()
    d_inv_sqrt_diag = np.zeros_like(degrees_diag)
    mask = degrees_diag > 0
    d_inv_sqrt_diag[mask] = 1.0 / np.sqrt(degrees_diag[mask])
    d_inv_sqrt = sps.diags(d_inv_sqrt_diag, format="csr")
    return d_inv_sqrt @ adj_hat @ d_inv_sqrt


def build_laplacian(adj: sps.csr_array) -> sps.csr_array:
    """
    Laplacian matrix from adjacency matrix.

    Advanced API: operates on scipy.sparse (host). Use when you need
    scipy directly or explicit control over host memory.

    Parameters
    ----------
    adj : scipy.sparse.csr_array
        Adjacency matrix (e.g. PP or CC).

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_points, n_points)`` or ``(n_cells, n_cells)``.
        dtype is np.float64.
    """
    d = build_degree_matrix(adj)
    return d - adj.astype(np.float64)
