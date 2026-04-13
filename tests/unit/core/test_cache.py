"""Unit tests for graphlow.core.cache (BackendCache)."""

import numpy as np
import pytest
import scipy.sparse as sps

from graphlow.core.backend.base import Backend
from graphlow.core.cache import BackendCache


# =============================================================================
# BackendCache
# =============================================================================
class TestBackendCacheMakeKey:
    def test_make_key_returns_expected_format(self, backend: Backend):
        """make_key returns 'kind:name:layout'."""
        bcache = BackendCache(backend)
        assert bcache.make_key("sparse", "PC", "csr") == "sparse:PC:csr"
        assert bcache.make_key("sparse", "FC", "coo") == "sparse:FC:coo"


class TestBackendCacheSparse:
    def test_sparse_creates_and_caches(self, backend: Backend):
        """sparse() creates backend sparse from skeleton and caches it."""
        bcache = BackendCache(backend)
        skeleton = sps.csr_matrix(
            (
                np.array([1.0, 2.0]),
                (np.array([0, 1]), np.array([1, 0])),
            ),
            shape=(3, 3),
        )
        a = bcache.sparse(skeleton, "test", "csr")
        b = bcache.sparse(skeleton, "test", "csr")
        assert a is b
        assert a.shape == (3, 3)

    def test_sparse_different_name_different_entry(self, backend: Backend):
        """sparse() with different name stores separate cache entry."""
        bcache = BackendCache(backend)
        skeleton = sps.csr_matrix(
            (np.array([1.0]), (np.array([0]), np.array([0]))),
            shape=(2, 2),
        )
        a = bcache.sparse(skeleton, "A", "csr")
        b = bcache.sparse(skeleton, "B", "csr")
        assert a is not b
        assert a.shape == b.shape == (2, 2)

    def test_sparse_different_layout_different_entry(self, backend: Backend):
        """sparse() with different layout stores separate cache entry."""
        bcache = BackendCache(backend)
        skeleton = sps.csr_matrix(
            (np.array([1.0]), (np.array([0]), np.array([0]))),
            shape=(2, 2),
        )
        a = bcache.sparse(skeleton, "M", "csr")
        b = bcache.sparse(skeleton, "M", "coo")
        assert a is not b


class TestBackendCacheInvalidate:
    def test_invalidate_clears_cache(self, backend: Backend):
        """invalidate() clears all cached sparse; next sparse() creates new."""
        bcache = BackendCache(backend)
        skeleton = sps.csr_matrix(
            (np.array([1.0]), (np.array([0]), np.array([0]))),
            shape=(2, 2),
        )
        a = bcache.sparse(skeleton, "X", "csr")
        bcache.invalidate()
        b = bcache.sparse(skeleton, "X", "csr")
        assert a is not b


class TestBackendCacheDelItem:
    def test_purge_removes_one_entry(self, backend: Backend):
        """purge removes the named entry; next sparse() recreates it."""
        bcache = BackendCache(backend)
        skeleton = sps.csr_matrix(
            (np.array([1.0]), (np.array([0]), np.array([0]))), shape=(2, 2)
        )
        a = bcache.sparse(skeleton, "Y", "csr")
        key = bcache.make_key("sparse", "Y", "csr")
        bcache.purge(key)
        b = bcache.sparse(skeleton, "Y", "csr")
        assert a is not b

    def test_purge_nonexistent_raises(self, backend: Backend):
        """purge with unknown key raises KeyError."""
        bcache = BackendCache(backend)
        with pytest.raises(KeyError):
            bcache.purge("sparse:nosuch:csr")
