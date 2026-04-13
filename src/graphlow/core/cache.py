"""BackendCache: backend-dependent sparse matrices and compiled functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import scipy.sparse as sps

if TYPE_CHECKING:
    from graphlow.core.backend.base import Backend, TensorLike

logger = logging.getLogger(__name__)


class BackendCache[T: TensorLike]:
    """
    Cache for backend-specific objects.

    Skeleton is built with scipy in MeshTopology (neutral cache); this layer
    materializes backend sparse from scipy.sparse and caches (torch sparse).

    Advanced API: direct use is for when you need to feed scipy skeletons
    into the backend or control backend (device) memory. Typical usage
    is via mesh.topology.incidence/adjacency; use bcache.sparse and
    invalidate/purge only if you need explicit control.
    """

    def __init__(self, backend: Backend[T]) -> None:
        self._backend = backend
        self._sparse: dict[str, T] = {}

    def sparse(
        self,
        skeleton: sps.csr_array,
        name: str,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Get or create backend sparse from skeleton (scipy-backed).

        Advanced API: you supply a scipy.sparse csr array and get back a
        backend tensor; the result is cached. Use when you need to
        materialize or control backend sparse memory explicitly.

        Parameters
        ----------
        skeleton : scipy.sparse.csr_array
            Scipy-backed skeleton from MeshTopology.
        name : str
            Sparse name.
        layout : Literal["csr", "coo"]
            "csr" or "coo".

        Returns
        -------
        T
            Backend sparse matrix in the requested layout.
        """
        key = self.make_key("sparse", name, layout)
        if key not in self._sparse:
            logger.debug("Materializing backend sparse %s", key)
            self._sparse[key] = self._backend.make_sparse_from_skeleton(
                skeleton, layout
            )
        return self._sparse[key]

    def make_key(
        self,
        kind: Literal["sparse"],
        name: str,
        layout: Literal["csr", "coo"],
    ) -> str:
        """
        Build a cache key for sparse objects.

        Returns
        -------
        str
            Cache key.
        """
        return f"{kind}:{name}:{layout}"

    def invalidate(self) -> None:
        """
        Clear all cached backend objects.

        Returns
        -------
        None
        """
        self._sparse.clear()

    def purge(self, key: str) -> None:
        """
        Purge a cached item by key.

        Parameters
        ----------
        key : str
            Cache key to purge.

        Returns
        -------
        None
        """
        del self._sparse[key]
