"""Unit tests for graph.mapping (dispatch and segment/sparse impl)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import phlower_tensor as pt
import pytest

import graphlow
from graphlow.core.backend.base import Backend
from graphlow.core.mesh import TensorMesh
from graphlow.graph import mapping


# =============================================================================
# Dispatch tests: map_point_to_cell calls correct internal function
# =============================================================================
class TestMapPointToCellDispatch:
    _SEGMENT_FNS = {
        "sum": "_segment_sum_map_point_to_cell",
        "mean": "_segment_mean_map_point_to_cell",
        "conservative": "_segment_conservative_map_point_to_cell",
    }
    _SPARSE_FNS = {
        "sum": "_sparse_sum_map_point_to_cell",
        "mean": "_sparse_mean_map_point_to_cell",
        "conservative": "_sparse_conservative_map_point_to_cell",
    }

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_dispatch_calls_correct_function(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: Literal["sum", "mean", "conservative"],
        method: Literal["segment", "sparse"],
    ):
        """
        For each (mode, method), map_point_to_cell calls the corresponding
        internal function exactly once with the expected arguments.
        """

        fn_name = (
            self._SEGMENT_FNS[mode]
            if method == "segment"
            else self._SPARSE_FNS[mode]
        )
        target = f"graphlow.graph.mapping.{fn_name}"
        backend = mix_poly_mesh.backend
        point_data = backend.ones(mix_poly_mesh.n_points)

        with patch(target) as mock_fn:
            _ = mapping.map_point_to_cell(
                mix_poly_mesh, point_data, mode=mode, method=method
            )

        assert mock_fn.call_count == 1
        if method == "segment":
            backend, topo, x = mock_fn.call_args[0]
            assert backend is mix_poly_mesh.backend
            assert topo is mix_poly_mesh.topology
            assert x is point_data
        elif method == "sparse":
            topo, x = mock_fn.call_args[0]
            assert topo is mix_poly_mesh.topology
            assert x is point_data

    @pytest.mark.parametrize("invalid_mode", ["invalid", "diff", "div"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_map_point_to_cell_invalid_mode_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        invalid_mode: Literal["invalid", "diff", "div"],
        method: Literal["segment", "sparse"],
    ):
        """map_point_to_cell with invalid mode raises ValueError."""
        backend = mix_poly_mesh.backend
        point_data = backend.ones(mix_poly_mesh.n_points)
        with pytest.raises(ValueError, match="Invalid mode"):
            mapping.map_point_to_cell(
                mix_poly_mesh, point_data, mode=invalid_mode, method=method
            )

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative"])
    @pytest.mark.parametrize("invalid_method", ["invalid"])
    def test_map_point_to_cell_invalid_method_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: Literal["sum", "mean", "conservative"],
        invalid_method: Literal["invalid"],
    ):
        """map_point_to_cell with invalid method raises ValueError."""
        backend = mix_poly_mesh.backend
        point_data = backend.ones(mix_poly_mesh.n_points)
        with pytest.raises(ValueError, match="method must be"):
            mapping.map_point_to_cell(
                mix_poly_mesh, point_data, mode=mode, method=invalid_method
            )


# =============================================================================
# Numerical test: map_point_to_cell
# =============================================================================
class TestMapPointToCellNumerical:
    """Numerical checks on the three-cell mixed mesh (polyhedron, hex, tet)."""

    @pytest.mark.parametrize(
        "np_point_data, np_expected",
        [
            (
                np.arange(13, dtype=np.float64),
                np.array([47.0, 36.0, 42.0], dtype=np.float64),
            ),
        ],
    )
    def test_sum(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_point_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        cell[0]: sum [0, 1, 3, 5, 8, 9, 10, 11] = 47
        cell[1]: sum [1, 2, 3, 4, 5, 6, 7, 8] = 36
        cell[2]: sum [9, 10, 11, 12] = 42
        """
        point_data = backend_phlower.as_tensor(
            np_point_data, dimension={"L": 1}
        )
        segment_result = mapping._segment_sum_map_point_to_cell(
            backend_phlower, mix_poly_mesh.topology, point_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_sum_map_point_to_cell(
            mix_poly_mesh.topology, point_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_point_data, np_expected",
        [
            (
                np.arange(13, dtype=np.float64),
                np.array(
                    [47.0 / 8.0, 36.0 / 8.0, 42.0 / 4.0], dtype=np.float64
                ),
            ),
        ],
    )
    def test_mean(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_point_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        cell[0]: mean [0, 1, 3, 5, 8, 9, 10, 11] = 47 / 8
        cell[1]: mean [1, 2, 3, 4, 5, 6, 7, 8] = 36 / 8
        cell[2]: mean [9, 10, 11, 12] = 42 / 4
        """
        point_data = backend_phlower.as_tensor(
            np_point_data, dimension={"L": 1}
        )
        segment_result = mapping._segment_mean_map_point_to_cell(
            backend_phlower, mix_poly_mesh.topology, point_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_mean_map_point_to_cell(
            mix_poly_mesh.topology, point_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_point_data, np_expected",
        [
            (
                np.ones(13, dtype=np.float64),
                np.array([4.5, 6.0, 2.5], dtype=np.float64),
            ),
        ],
    )
    def test_conservative(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_point_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        cell[0]: conservative [1/1, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2] = 4.5
        cell[1]: conservative [1/2, 1/1, 1/2, 1/1, 1/2, 1/1, 1/1, 1/2] = 6.0
        cell[2]: conservative [1/2, 1/2, 1/2, 1/1] = 2.5
        """
        point_data = backend_phlower.as_tensor(
            np_point_data, dimension={"L": 1}
        )
        segment_result = mapping._segment_conservative_map_point_to_cell(
            backend_phlower, mix_poly_mesh.topology, point_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_conservative_map_point_to_cell(
            mix_poly_mesh.topology, point_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)


# =============================================================================
# Dispatch tests: map_cell_to_point calls correct internal function
# =============================================================================
class TestMapCellToPointDispatch:
    _SEGMENT_FNS = {
        "sum": "_segment_sum_map_cell_to_point",
        "mean": "_segment_mean_map_cell_to_point",
        "conservative": "_segment_conservative_map_cell_to_point",
    }
    _SPARSE_FNS = {
        "sum": "_sparse_sum_map_cell_to_point",
        "mean": "_sparse_mean_map_cell_to_point",
        "conservative": "_sparse_conservative_map_cell_to_point",
    }

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_dispatch_calls_correct_function(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: Literal["sum", "mean", "conservative"],
        method: Literal["segment", "sparse"],
    ):
        """
        For each (mode, method), map_cell_to_point calls the corresponding
        internal function exactly once with the expected arguments.
        """
        fn_name = (
            self._SEGMENT_FNS[mode]
            if method == "segment"
            else self._SPARSE_FNS[mode]
        )
        target = f"graphlow.graph.mapping.{fn_name}"
        backend = mix_poly_mesh.backend
        cell_data = backend.ones(mix_poly_mesh.n_cells)

        with patch(target) as mock_fn:
            _ = mapping.map_cell_to_point(
                mix_poly_mesh, cell_data, mode=mode, method=method
            )

        assert mock_fn.call_count == 1
        if method == "segment":
            backend_arg, topo, x = mock_fn.call_args[0]
            assert backend_arg is mix_poly_mesh.backend
            assert topo is mix_poly_mesh.topology
            assert x is cell_data
        elif method == "sparse":
            topo, x = mock_fn.call_args[0]
            assert topo is mix_poly_mesh.topology
            assert x is cell_data

    @pytest.mark.parametrize("invalid_mode", ["invalid", "diff", "div"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_map_cell_to_point_invalid_mode_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        invalid_mode: Literal["invalid", "diff", "div"],
        method: Literal["segment", "sparse"],
    ):
        """map_cell_to_point with invalid mode raises ValueError."""
        backend = mix_poly_mesh.backend
        cell_data = backend.ones(mix_poly_mesh.n_cells)
        with pytest.raises(ValueError, match="Invalid mode"):
            mapping.map_cell_to_point(
                mix_poly_mesh, cell_data, mode=invalid_mode, method=method
            )

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative"])
    @pytest.mark.parametrize("invalid_method", ["invalid"])
    def test_map_cell_to_point_invalid_method_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: Literal["sum", "mean", "conservative"],
        invalid_method: Literal["invalid"],
    ):
        """map_cell_to_point with invalid method raises ValueError."""
        backend = mix_poly_mesh.backend
        cell_data = backend.ones(mix_poly_mesh.n_cells)
        with pytest.raises(ValueError, match="method must be"):
            mapping.map_cell_to_point(
                mix_poly_mesh, cell_data, mode=mode, method=invalid_method
            )


# =============================================================================
# Numerical test: map_cell_to_point
# =============================================================================
class TestMapCellToPointNumerical:
    """Numerical checks on the three-cell mixed mesh (polyhedron, hex, tet)."""

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.array([1, 2, 4], dtype=np.float64),
                np.array(
                    [1, 3, 2, 3, 2, 3, 2, 2, 3, 5, 5, 5, 4],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_sum(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        point[0]: sum [1]     = 1
        point[1]: sum [1, 2]  = 3
        point[2]: sum [2]     = 2
        point[3]: sum [1, 2]  = 3
        point[4]: sum [2]     = 2
        point[5]: sum [1, 2]  = 3
        point[6]: sum [2]     = 2
        point[7]: sum [2]     = 2
        point[8]: sum [1, 2]  = 3
        point[9]: sum [1, 4]  = 5
        point[10]: sum [1, 4] = 5
        point[11]: sum [1, 4] = 5
        point[12]: sum [4]    = 4
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_sum_map_cell_to_point(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_sum_map_cell_to_point(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.array([1, 2, 4], dtype=np.float64),
                np.array(
                    [
                        1.0,
                        1.5,
                        2.0,
                        1.5,
                        2.0,
                        1.5,
                        2.0,
                        2.0,
                        1.5,
                        2.5,
                        2.5,
                        2.5,
                        4.0,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_mean(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        point[0]: mean [1]     = 1/1
        point[1]: mean [1, 2]  = 3/2
        point[2]: mean [2]     = 2/1
        point[3]: mean [1, 2]  = 3/2
        point[4]: mean [2]     = 2/1
        point[5]: mean [1, 2]  = 3/2
        point[6]: mean [2]     = 2/1
        point[7]: mean [2]     = 2/1
        point[8]: mean [1, 2]  = 3/2
        point[9]: mean [1, 4]  = 5/2
        point[10]: mean [1, 4] = 5/2
        point[11]: mean [1, 4] = 5/2
        point[12]: mean [4]    = 4/1
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_mean_map_cell_to_point(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_mean_map_cell_to_point(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.array([1.0, 2.0, 4.0], dtype=np.float64),
                np.array(
                    [
                        1.0 / 8,
                        3.0 / 8,
                        2.0 / 8,
                        3.0 / 8,
                        2.0 / 8,
                        3.0 / 8,
                        2.0 / 8,
                        2.0 / 8,
                        3.0 / 8,
                        9.0 / 8,
                        9.0 / 8,
                        9.0 / 8,
                        8.0 / 8,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_conservative(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        point[0]: conservative [1/8]       = 1/8
        point[1]: conservative [1/8, 2/8]  = 3/8
        point[2]: conservative [2/8]       = 2/8
        point[3]: conservative [1/8, 2/8]  = 3/8
        point[4]: conservative [2/8]       = 2/8
        point[5]: conservative [1/8, 2/8]  = 3/8
        point[6]: conservative [2/8]       = 2/8
        point[7]: conservative [2/8]       = 2/8
        point[8]: conservative [1/8, 2/8]  = 3/8
        point[9]: conservative [1/8, 4/4]  = 9/8
        point[10]: conservative [1/8, 4/4] = 9/8
        point[11]: conservative [1/8, 4/4] = 9/8
        point[12]: conservative [4/4]      = 8/8
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_conservative_map_cell_to_point(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_conservative_map_cell_to_point(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)


# =============================================================================
# Dispatch tests: map_cell_to_face
# =============================================================================
class TestMapCellToFaceDispatch:
    _SEGMENT_FNS = {
        "sum": "_segment_sum_map_cell_to_face",
        "mean": "_segment_mean_map_cell_to_face",
        "conservative": "_segment_conservative_map_cell_to_face",
        "diff": "_segment_diff_map_cell_to_face",
    }
    _SPARSE_FNS = {
        "sum": "_sparse_sum_map_cell_to_face",
        "mean": "_sparse_mean_map_cell_to_face",
        "conservative": "_sparse_conservative_map_cell_to_face",
        "diff": "_sparse_diff_map_cell_to_face",
    }

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative", "diff"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_dispatch_calls_correct_function(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: Literal["sum", "mean", "conservative", "diff"],
        method: Literal["segment", "sparse"],
    ):
        fn_name = (
            self._SEGMENT_FNS[mode]
            if method == "segment"
            else self._SPARSE_FNS[mode]
        )
        target = f"graphlow.graph.mapping.{fn_name}"
        backend = mix_poly_mesh.backend
        cell_data = backend.ones(mix_poly_mesh.n_cells)
        n_faces = mix_poly_mesh.topology.face_registry().n_faces()

        with patch(target) as mock_fn:
            mock_fn.return_value = backend.ones(n_faces)
            _ = mapping.map_cell_to_face(
                mix_poly_mesh, cell_data, mode=mode, method=method
            )

        assert mock_fn.call_count == 1
        if method == "segment":
            backend_arg, topo, x = mock_fn.call_args[0]
            assert backend_arg is mix_poly_mesh.backend
            assert topo is mix_poly_mesh.topology
            assert x is cell_data
        else:
            topo, x = mock_fn.call_args[0]
            assert topo is mix_poly_mesh.topology
            assert x is cell_data

    @pytest.mark.parametrize("invalid_mode", ["invalid", "div"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_invalid_mode_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        invalid_mode: str,
        method: Literal["segment", "sparse"],
    ):
        backend = mix_poly_mesh.backend
        cell_data = backend.ones(mix_poly_mesh.n_cells)
        with pytest.raises(ValueError, match="Invalid mode"):
            mapping.map_cell_to_face(
                mix_poly_mesh, cell_data, mode=invalid_mode, method=method
            )

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative", "diff"])
    @pytest.mark.parametrize("invalid_method", ["invalid"])
    def test_invalid_method_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: str,
        invalid_method: Literal["invalid"],
    ):
        backend = mix_poly_mesh.backend
        cell_data = backend.ones(mix_poly_mesh.n_cells)
        with pytest.raises(ValueError, match="method must be"):
            mapping.map_cell_to_face(
                mix_poly_mesh, cell_data, mode=mode, method=invalid_method
            )


# =============================================================================
# Numerical test: map_cell_to_face
# =============================================================================
class TestMapCellToFaceNumerical:
    """
    mix_poly has poly, hex, tet.
    face[0]: [0, 1, 5, 10, 9]
    face[1]: [1, 3, 8, 5]
    face[2]: [0, 9, 11, 8, 3]
    face[3]: [5, 8, 11, 10]
    face[4]: [9, 10, 11]
    face[5]: [0, 3, 1]
    face[6]: [2, 4, 7, 6]
    face[7]: [1, 2, 6, 5]
    face[8]: [3, 8, 7, 4]
    face[9]: [1, 3, 4, 2]
    face[10]: [5, 6, 7, 8]
    face[11]: [9, 10, 12]
    face[12]: [10, 11, 12]
    face[13]: [11, 9, 12]
    """

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.array([1, 2, 4], dtype=np.float64),
                np.array(
                    [1, 3, 1, 1, 5, 1, 2, 2, 2, 2, 2, 4, 4, 4],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_sum(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        face[0]: sum [1]     = 1
        face[1]: sum [1, 2]  = 3
        face[2]: sum [1]     = 1
        face[3]: sum [1]     = 1
        face[4]: sum [1, 4]  = 5
        face[5]: sum [1]     = 1
        face[6]: sum [2]     = 2
        face[7]: sum [2]     = 2
        face[8]: sum [2]     = 2
        face[9]: sum [2]     = 2
        face[10]: sum [2]    = 2
        face[11]: sum [4]    = 4
        face[12]: sum [4]    = 4
        face[13]: sum [4]    = 4
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_sum_map_cell_to_face(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_sum_map_cell_to_face(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.array([1, 2, 4], dtype=np.float64),
                np.array(
                    [
                        1.0 / 1,
                        3.0 / 2,
                        1.0 / 1,
                        1.0 / 1,
                        5.0 / 2,
                        1.0 / 1,
                        2.0 / 1,
                        2.0 / 1,
                        2.0 / 1,
                        2.0 / 1,
                        2.0 / 1,
                        4.0 / 1,
                        4.0 / 1,
                        4.0 / 1,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_mean(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        face[0]: mean [1]     = 1/1
        face[1]: mean [1, 2]  = 3/2
        face[2]: mean [1]     = 1/1
        face[3]: mean [1]     = 1/1
        face[4]: mean [1, 4]  = 5/2
        face[5]: mean [1]     = 1/1
        face[6]: mean [2]     = 2/1
        face[7]: mean [2]     = 2/1
        face[8]: mean [2]     = 2/1
        face[9]: mean [2]     = 2/1
        face[10]: mean [2]    = 2/1
        face[11]: mean [4]    = 4/1
        face[12]: mean [4]    = 4/1
        face[13]: mean [4]    = 4/1
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_mean_map_cell_to_face(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_mean_map_cell_to_face(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.array([1.0, 2.0, 4.0], dtype=np.float64),
                np.array(
                    [
                        1.0 / 6,
                        3.0 / 6,
                        1.0 / 6,
                        1.0 / 6,
                        7.0 / 6,
                        1.0 / 6,
                        2.0 / 6,
                        2.0 / 6,
                        2.0 / 6,
                        2.0 / 6,
                        2.0 / 6,
                        6.0 / 6,
                        6.0 / 6,
                        6.0 / 6,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_conservative(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        face[0]: conservative [1/6]      = 1/6
        face[1]: conservative [1/6, 2/6] = 3/6
        face[2]: conservative [1/6]      = 1/6
        face[3]: conservative [1/6]      = 1/6
        face[4]: conservative [1/6, 4/4] = 7/6
        face[5]: conservative [1/6]      = 1/6
        face[6]: conservative [2/6]      = 2/6
        face[7]: conservative [2/6]      = 2/6
        face[8]: conservative [2/6]      = 2/6
        face[9]: conservative [2/6]      = 2/6
        face[10]: conservative [2/6]     = 2/6
        face[11]: conservative [4/4]     = 6/6
        face[12]: conservative [4/4]     = 6/6
        face[13]: conservative [4/4]     = 6/6
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_conservative_map_cell_to_face(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_conservative_map_cell_to_face(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.array([1, 2, 4], dtype=np.float64),
                np.array(
                    [1, -1, 1, 1, -3, 1, 2, 2, 2, 2, 2, 4, 4, 4],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_diff(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        face[0]: diff [1]     = 1
        face[1]: diff [1, -2]  = -1
        face[2]: diff [1]     = 1
        face[3]: diff [1]     = 1
        face[4]: diff [1, -4]  = -3
        face[5]: diff [1]     = 1
        face[6]: diff [2]     = 2
        face[7]: diff [2]     = 2
        face[8]: diff [2]     = 2
        face[9]: diff [2]     = 2
        face[10]: diff [2]    = 2
        face[11]: diff [4]    = 4
        face[12]: diff [4]    = 4
        face[13]: diff [4]    = 4
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_diff_map_cell_to_face(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_diff_map_cell_to_face(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)


# =============================================================================
# Dispatch tests: map_face_to_cell
# =============================================================================
class TestMapFaceToCellDispatch:
    _SEGMENT_FNS = {
        "sum": "_segment_sum_map_face_to_cell",
        "mean": "_segment_mean_map_face_to_cell",
        "conservative": "_segment_conservative_map_face_to_cell",
        "div": "_segment_div_map_face_to_cell",
    }
    _SPARSE_FNS = {
        "sum": "_sparse_sum_map_face_to_cell",
        "mean": "_sparse_mean_map_face_to_cell",
        "conservative": "_sparse_conservative_map_face_to_cell",
        "div": "_sparse_div_map_face_to_cell",
    }

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative", "div"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_dispatch_calls_correct_function(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: Literal["sum", "mean", "conservative", "div"],
        method: Literal["segment", "sparse"],
    ):
        fn_name = (
            self._SEGMENT_FNS[mode]
            if method == "segment"
            else self._SPARSE_FNS[mode]
        )
        target = f"graphlow.graph.mapping.{fn_name}"
        backend = mix_poly_mesh.backend
        n_faces = mix_poly_mesh.topology.face_registry().n_faces()
        face_data = backend.ones(n_faces)

        with patch(target) as mock_fn:
            mock_fn.return_value = backend.ones(mix_poly_mesh.n_cells)
            _ = mapping.map_face_to_cell(
                mix_poly_mesh, face_data, mode=mode, method=method
            )

        assert mock_fn.call_count == 1
        if method == "segment":
            backend_arg, topo, x = mock_fn.call_args[0]
            assert backend_arg is mix_poly_mesh.backend
            assert topo is mix_poly_mesh.topology
            assert x is face_data
        else:
            topo, x = mock_fn.call_args[0]
            assert topo is mix_poly_mesh.topology
            assert x is face_data

    @pytest.mark.parametrize("invalid_mode", ["invalid", "diff"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_invalid_mode_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        invalid_mode: str,
        method: Literal["segment", "sparse"],
    ):
        backend = mix_poly_mesh.backend
        n_faces = mix_poly_mesh.topology.face_registry().n_faces()
        face_data = backend.ones(n_faces)
        with pytest.raises(ValueError, match="Invalid mode"):
            mapping.map_face_to_cell(
                mix_poly_mesh, face_data, mode=invalid_mode, method=method
            )

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative", "div"])
    @pytest.mark.parametrize("invalid_method", ["invalid"])
    def test_invalid_method_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: str,
        invalid_method: Literal["invalid"],
    ):
        backend = mix_poly_mesh.backend
        n_faces = mix_poly_mesh.topology.face_registry().n_faces()
        face_data = backend.ones(n_faces)
        with pytest.raises(ValueError, match="method must be"):
            mapping.map_face_to_cell(
                mix_poly_mesh, face_data, mode=mode, method=invalid_method
            )


# =============================================================================
# Numerical test: map_face_to_cell
# =============================================================================
class TestMapFaceToCellNumerical:
    """
    mix_poly has poly, hex, tet.
    face[0]: [0, 1, 5, 10, 9]
    face[1]: [1, 3, 8, 5]
    face[2]: [0, 9, 11, 8, 3]
    face[3]: [5, 8, 11, 10]
    face[4]: [9, 10, 11]
    face[5]: [0, 3, 1]
    face[6]: [2, 4, 7, 6]
    face[7]: [1, 2, 6, 5]
    face[8]: [3, 8, 7, 4]
    face[9]: [1, 3, 4, 2]
    face[10]: [5, 6, 7, 8]
    face[11]: [9, 10, 12]
    face[12]: [10, 11, 12]
    face[13]: [11, 9, 12]
    """

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(14, dtype=np.float64),
                np.array(
                    [15, 41, 40],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_sum(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        cell[0]: sum [0, 1, 2, 3, 4, 5] = 15
        cell[1]: sum [1, 6, 7, 8, 9, 10] = 41
        cell[2]: sum [4, 11, 12, 13] = 40
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_sum_map_face_to_cell(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_sum_map_face_to_cell(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(14, dtype=np.float64),
                np.array(
                    [
                        15.0 / 6,
                        41.0 / 6,
                        40.0 / 4,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_mean(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        cell[0]: mean [0, 1, 2, 3, 4, 5] = 15/6
        cell[1]: mean [1, 6, 7, 8, 9, 10] = 41/6
        cell[2]: mean [4, 11, 12, 13] = 40/4
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_mean_map_face_to_cell(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_mean_map_face_to_cell(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(14, dtype=np.float64),
                np.array(
                    [
                        12.5,
                        40.5,
                        38.0,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_conservative(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        cell[0]: conservative [0/1, 1/2, 2/1, 3/1, 4/2, 5/1] = 12.5
        cell[1]: conservative [1/2, 6/1, 7/1, 8/1, 9/1, 10/1] = 40.5
        cell[2]: conservative [4/2, 11/1, 12/1, 13/1] = 38
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_conservative_map_face_to_cell(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_conservative_map_face_to_cell(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(14, dtype=np.float64),
                np.array(
                    [15, 39, 32],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_div(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        cell[0]: div [0, 1, 2, 3, 4, 5] = 15
        cell[1]: div [-1, 6, 7, 8, 9, 10] = 39
        cell[2]: div [-4, 11, 12, 13] = 32
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_div_map_face_to_cell(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_div_map_face_to_cell(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)


# =============================================================================
# Dispatch tests: map_face_to_point
# =============================================================================
class TestMapFaceToPointDispatch:
    _SEGMENT_FNS = {
        "sum": "_segment_sum_map_face_to_point",
        "mean": "_segment_mean_map_face_to_point",
        "conservative": "_segment_conservative_map_face_to_point",
    }
    _SPARSE_FNS = {
        "sum": "_sparse_sum_map_face_to_point",
        "mean": "_sparse_mean_map_face_to_point",
        "conservative": "_sparse_conservative_map_face_to_point",
    }

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_dispatch_calls_correct_function(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: Literal["sum", "mean", "conservative"],
        method: Literal["segment", "sparse"],
    ):
        fn_name = (
            self._SEGMENT_FNS[mode]
            if method == "segment"
            else self._SPARSE_FNS[mode]
        )
        target = f"graphlow.graph.mapping.{fn_name}"
        backend = mix_poly_mesh.backend
        n_faces = mix_poly_mesh.topology.face_registry().n_faces()
        face_data = backend.ones(n_faces)

        with patch(target) as mock_fn:
            mock_fn.return_value = backend.ones(mix_poly_mesh.n_points)
            _ = mapping.map_face_to_point(
                mix_poly_mesh, face_data, mode=mode, method=method
            )

        assert mock_fn.call_count == 1
        if method == "segment":
            backend_arg, topo, x = mock_fn.call_args[0]
            assert backend_arg is mix_poly_mesh.backend
            assert topo is mix_poly_mesh.topology
            assert x is face_data
        else:
            topo, x = mock_fn.call_args[0]
            assert topo is mix_poly_mesh.topology
            assert x is face_data

    @pytest.mark.parametrize("invalid_mode", ["invalid", "diff", "div"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_invalid_mode_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        invalid_mode: str,
        method: Literal["segment", "sparse"],
    ):
        backend = mix_poly_mesh.backend
        n_faces = mix_poly_mesh.topology.face_registry().n_faces()
        face_data = backend.ones(n_faces)
        with pytest.raises(ValueError, match="Invalid mode"):
            mapping.map_face_to_point(
                mix_poly_mesh, face_data, mode=invalid_mode, method=method
            )

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative"])
    @pytest.mark.parametrize("invalid_method", ["invalid"])
    def test_invalid_method_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: str,
        invalid_method: Literal["invalid"],
    ):
        backend = mix_poly_mesh.backend
        n_faces = mix_poly_mesh.topology.face_registry().n_faces()
        face_data = backend.ones(n_faces)
        with pytest.raises(ValueError, match="method must be"):
            mapping.map_face_to_point(
                mix_poly_mesh, face_data, mode=mode, method=invalid_method
            )


# =============================================================================
# Numerical test: map_face_to_point
# =============================================================================
class TestMapFaceToPointNumerical:
    """
    mix_poly has poly, hex, tet.
    face[0]: [0, 1, 5, 10, 9]
    face[1]: [1, 3, 8, 5]
    face[2]: [0, 9, 11, 8, 3]
    face[3]: [5, 8, 11, 10]
    face[4]: [9, 10, 11]
    face[5]: [0, 3, 1]
    face[6]: [2, 4, 7, 6]
    face[7]: [1, 2, 6, 5]
    face[8]: [3, 8, 7, 4]
    face[9]: [1, 3, 4, 2]
    face[10]: [5, 6, 7, 8]
    face[11]: [9, 10, 12]
    face[12]: [10, 11, 12]
    face[13]: [11, 9, 12]
    """

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(14, dtype=np.float64),
                np.array(
                    [7, 22, 22, 25, 23, 21, 23, 24, 24, 30, 30, 34, 36],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_sum(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        point[0]: sum [0, 2, 5]          = 7
        point[1]: sum [0, 1, 5, 7, 9]    = 22
        point[2]: sum [6, 7, 9]          = 22
        point[3]: sum [1, 2, 5, 8, 9]    = 25
        point[4]: sum [6, 8, 9]          = 23
        point[5]: sum [0, 1, 3, 7, 10]   = 21
        point[6]: sum [6, 7, 10]         = 23
        point[7]: sum [6, 8, 10]         = 24
        point[8]: sum [1, 2, 3, 8, 10]   = 24
        point[9]: sum [0, 2, 4, 11, 13]  = 30
        point[10]: sum [0, 3, 4, 11, 12] = 30
        point[11]: sum [2, 3, 4, 12, 13] = 34
        point[12]: sum [11, 12, 13]      = 36
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_sum_map_face_to_point(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_sum_map_face_to_point(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(14, dtype=np.float64),
                np.array(
                    [
                        7.0 / 3,
                        22.0 / 5,
                        22.0 / 3,
                        25.0 / 5,
                        23.0 / 3,
                        21.0 / 5,
                        23.0 / 3,
                        24.0 / 3,
                        24.0 / 5,
                        30.0 / 5,
                        30.0 / 5,
                        34.0 / 5,
                        36.0 / 3,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_mean(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        point[0]: mean [0, 2, 5]          = 7/3
        point[1]: mean [0, 1, 5, 7, 9]    = 22/5
        point[2]: mean [6, 7, 9]          = 22/3
        point[3]: mean [1, 2, 5, 8, 9]    = 25/5
        point[4]: mean [6, 8, 9]          = 23/3
        point[5]: mean [0, 1, 3, 7, 10]   = 21/5
        point[6]: mean [6, 7, 10]         = 23/3
        point[7]: mean [6, 8, 10]         = 24/3
        point[8]: mean [1, 2, 3, 8, 10]   = 24/5
        point[9]: mean [0, 2, 4, 11, 13]  = 30/5
        point[10]: mean [0, 3, 4, 11, 12] = 30/5
        point[11]: mean [2, 3, 4, 12, 13] = 34/5
        point[12]: mean [11, 12, 13]      = 36/3
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_mean_map_face_to_point(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_mean_map_face_to_point(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(14, dtype=np.float64),
                np.array(
                    [
                        124.0 / 60,
                        355.0 / 60,
                        330.0 / 60,
                        394.0 / 60,
                        345.0 / 60,
                        315.0 / 60,
                        345.0 / 60,
                        360.0 / 60,
                        354.0 / 60,
                        584.0 / 60,
                        585.0 / 60,
                        649.0 / 60,
                        720.0 / 60,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_conservative(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        point[0]: conservative [0/5, 2/5, 5/3]              = 124/60
        point[1]: conservative [0/5, 1/4, 5/3, 7/4, 9/4]    = 355/60
        point[2]: conservative [6/4, 7/4, 9/4]              = 330/60
        point[3]: conservative [1/4, 2/5, 5/3, 8/4, 9/4]    = 394/60
        point[4]: conservative [6/4, 8/4, 9/4]              = 345/60
        point[5]: conservative [0/5, 1/4, 3/4, 7/4, 10/4]   = 315/60
        point[6]: conservative [6/4, 7/4, 10/4]             = 345/60
        point[7]: conservative [6/4, 8/4, 10/4]             = 360/60
        point[8]: conservative [1/4, 2/5, 3/4, 8/4, 10/4]   = 354/60
        point[9]: conservative [0/5, 2/5, 4/3, 11/3, 13/3]  = 584/60
        point[10]: conservative [0/5, 3/4, 4/3, 11/3, 12/3] = 585/60
        point[11]: conservative [2/5, 3/4, 4/3, 12/3, 13/3] = 649/60
        point[12]: conservative [11/3, 12/3, 13/3]          = 720/60
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_conservative_map_face_to_point(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_conservative_map_face_to_point(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)


# =============================================================================
# Dispatch tests: map_point_to_face
# =============================================================================
class TestMapPointToFaceDispatch:
    _SEGMENT_FNS = {
        "sum": "_segment_sum_map_point_to_face",
        "mean": "_segment_mean_map_point_to_face",
        "conservative": "_segment_conservative_map_point_to_face",
    }
    _SPARSE_FNS = {
        "sum": "_sparse_sum_map_point_to_face",
        "mean": "_sparse_mean_map_point_to_face",
        "conservative": "_sparse_conservative_map_point_to_face",
    }

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_dispatch_calls_correct_function(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: Literal["sum", "mean", "conservative"],
        method: Literal["segment", "sparse"],
    ):
        fn_name = (
            self._SEGMENT_FNS[mode]
            if method == "segment"
            else self._SPARSE_FNS[mode]
        )
        target = f"graphlow.graph.mapping.{fn_name}"
        backend = mix_poly_mesh.backend
        point_data = backend.ones(mix_poly_mesh.n_points)
        n_faces = mix_poly_mesh.topology.face_registry().n_faces()

        with patch(target) as mock_fn:
            mock_fn.return_value = backend.ones(n_faces)
            _ = mapping.map_point_to_face(
                mix_poly_mesh, point_data, mode=mode, method=method
            )

        assert mock_fn.call_count == 1
        if method == "segment":
            backend_arg, topo, x = mock_fn.call_args[0]
            assert backend_arg is mix_poly_mesh.backend
            assert topo is mix_poly_mesh.topology
            assert x is point_data
        else:
            topo, x = mock_fn.call_args[0]
            assert topo is mix_poly_mesh.topology
            assert x is point_data

    @pytest.mark.parametrize("invalid_mode", ["invalid", "diff", "div"])
    @pytest.mark.parametrize("method", ["segment", "sparse"])
    def test_invalid_mode_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        invalid_mode: str,
        method: Literal["segment", "sparse"],
    ):
        backend = mix_poly_mesh.backend
        point_data = backend.ones(mix_poly_mesh.n_points)
        with pytest.raises(ValueError, match="Invalid mode"):
            mapping.map_point_to_face(
                mix_poly_mesh, point_data, mode=invalid_mode, method=method
            )

    @pytest.mark.parametrize("mode", ["sum", "mean", "conservative"])
    @pytest.mark.parametrize("invalid_method", ["invalid"])
    def test_invalid_method_raises(
        self,
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        mode: str,
        invalid_method: Literal["invalid"],
    ):
        backend = mix_poly_mesh.backend
        point_data = backend.ones(mix_poly_mesh.n_points)
        with pytest.raises(ValueError, match="method must be"):
            mapping.map_point_to_face(
                mix_poly_mesh, point_data, mode=mode, method=invalid_method
            )


# =============================================================================
# Numerical test: map_point_to_face
# =============================================================================
class TestMapPointToFaceNumerical:
    """
    mix_poly has poly, hex, tet.
    face[0]: [0, 1, 5, 10, 9]
    face[1]: [1, 3, 8, 5]
    face[2]: [0, 9, 11, 8, 3]
    face[3]: [5, 8, 11, 10]
    face[4]: [9, 10, 11]
    face[5]: [0, 3, 1]
    face[6]: [2, 4, 7, 6]
    face[7]: [1, 2, 6, 5]
    face[8]: [3, 8, 7, 4]
    face[9]: [1, 3, 4, 2]
    face[10]: [5, 6, 7, 8]
    face[11]: [9, 10, 12]
    face[12]: [10, 11, 12]
    face[13]: [11, 9, 12]
    """

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(13, dtype=np.float64),
                np.array(
                    [25, 17, 31, 34, 30, 4, 19, 14, 22, 10, 26, 31, 33, 32],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_sum(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        face[0]: sum [0, 1, 5, 10, 9] = 25
        face[1]: sum [1, 3, 8, 5]     = 17
        face[2]: sum [0, 9, 11, 8, 3] = 31
        face[3]: sum [5, 8, 11, 10]   = 34
        face[4]: sum [9, 10, 11]      = 30
        face[5]: sum [0, 3, 1]        = 4
        face[6]: sum [2, 4, 7, 6]     = 19
        face[7]: sum [1, 2, 6, 5]     = 14
        face[8]: sum [3, 8, 7, 4]     = 22
        face[9]: sum [1, 3, 4, 2]     = 10
        face[10]: sum [5, 6, 7, 8]    = 26
        face[11]: sum [9, 10, 12]     = 31
        face[12]: sum [10, 11, 12]    = 33
        face[13]: sum [11, 9, 12]     = 32
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_sum_map_point_to_face(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_sum_map_point_to_face(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_equal(np_segment_result, np_expected)
        np.testing.assert_array_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(13, dtype=np.float64),
                np.array(
                    [
                        25.0 / 5,
                        17.0 / 4,
                        31.0 / 5,
                        34.0 / 4,
                        30.0 / 3,
                        4.0 / 3,
                        19.0 / 4,
                        14.0 / 4,
                        22.0 / 4,
                        10.0 / 4,
                        26.0 / 4,
                        31.0 / 3,
                        33.0 / 3,
                        32.0 / 3,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_mean(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        face[0]: mean [0, 1, 5, 10, 9] = 25/5
        face[1]: mean [1, 3, 8, 5]     = 17/4
        face[2]: mean [0, 9, 11, 8, 3] = 31/5
        face[3]: mean [5, 8, 11, 10]   = 34/4
        face[4]: mean [9, 10, 11]      = 30/3
        face[5]: mean [0, 3, 1]        = 4/3
        face[6]: mean [2, 4, 7, 6]     = 19/4
        face[7]: mean [1, 2, 6, 5]     = 14/4
        face[8]: mean [3, 8, 7, 4]     = 22/4
        face[9]: mean [1, 3, 4, 2]     = 10/4
        face[10]: mean [5, 6, 7, 8]    = 26/4
        face[11]: mean [9, 10, 12]     = 31/3
        face[12]: mean [10, 11, 12]    = 33/3
        face[13]: mean [11, 9, 12]     = 32/3
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_mean_map_point_to_face(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_mean_map_point_to_face(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)

    @pytest.mark.parametrize(
        "np_cell_data, np_expected",
        [
            (
                np.arange(13, dtype=np.float64),
                np.array(
                    [
                        75.0 / 15,
                        51.0 / 15,
                        93.0 / 15,
                        102.0 / 15,
                        90.0 / 15,
                        12.0 / 15,
                        95.0 / 15,
                        58.0 / 15,
                        88.0 / 15,
                        42.0 / 15,
                        104.0 / 15,
                        117.0 / 15,
                        123.0 / 15,
                        120.0 / 15,
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_conservative(
        self,
        backend_phlower: Backend[pt.PhlowerTensor],
        mix_poly_mesh: TensorMesh[pt.PhlowerTensor],
        np_cell_data: np.ndarray,
        np_expected: np.ndarray,
    ):
        """
        face[0]: conservative [0/3, 1/5, 5/5, 10/5, 9/5] = 75/15
        face[1]: conservative [1/5, 3/5, 8/5, 5/5]       = 51/15
        face[2]: conservative [0/3, 9/5, 11/5, 8/5, 3/5] = 93/15
        face[3]: conservative [5/5, 8/5, 11/5, 10/5]     = 102/15
        face[4]: conservative [9/5, 10/5, 11/5]          = 90/15
        face[5]: conservative [0/3, 3/5, 1/5]            = 12/15
        face[6]: conservative [2/3, 4/3, 7/3, 6/3]       = 95/15
        face[7]: conservative [1/5, 2/3, 6/3, 5/5]       = 58/15
        face[8]: conservative [3/5, 8/5, 7/3, 4/3]       = 88/15
        face[9]: conservative [1/5, 3/5, 4/3, 2/3]       = 42/15
        face[10]: conservative [5/5, 6/3, 7/3, 8/5]      = 104/15
        face[11]: conservative [9/5, 10/5, 12/3]         = 117/15
        face[12]: conservative [10/5, 11/5, 12/3]        = 123/15
        face[13]: conservative [11/5, 9/5, 12/3]         = 120/15
        """
        cell_data = backend_phlower.as_tensor(np_cell_data, dimension={"L": 1})
        segment_result = mapping._segment_conservative_map_point_to_face(
            backend_phlower, mix_poly_mesh.topology, cell_data
        )
        assert segment_result.dimension == pt.phlower_dimension_tensor(
            {"L": 1}, device=backend_phlower.device
        )
        np_segment_result = backend_phlower.to_numpy(segment_result)
        sparse_result = mapping._sparse_conservative_map_point_to_face(
            mix_poly_mesh.topology, cell_data
        )
        np_sparse_result = backend_phlower.to_numpy(sparse_result)
        np.testing.assert_array_almost_equal(np_segment_result, np_expected)
        np.testing.assert_array_almost_equal(np_sparse_result, np_expected)


# =============================================================================
# Median
# =============================================================================
@pytest.mark.parametrize(
    "filename, input_data, n_hop, expected",
    [
        (
            Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                # 0  1  2  3  4  5  6  7  8  9 10 11
                [1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 5, 1]
            ),
            1,
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ),
        (
            Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                # 0  1  2  3  4  5  6  7  8  9 10 11
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            ),
            4,
            np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]),
        ),
    ],
)
def test_median_points(
    filename: Path,
    input_data: np.ndarray,
    n_hop: int,
    expected: np.ndarray,
) -> None:
    """median_points returns expected result."""
    mesh = graphlow.read(filename, backend="phlower")
    backend = mesh.backend
    x = backend.as_tensor(input_data, dimension={"Theta": 1})
    out = mapping.median_points(mesh, x, n_hop=n_hop)
    assert out.dimension == pt.phlower_dimension_tensor({"Theta": 1})
    np.testing.assert_array_almost_equal(out.numpy(), expected)


@pytest.mark.parametrize(
    "filename, input_data, n_hop, expected",
    [
        (
            Path("tests/data/vtu/complex/mesh.vtu"),
            np.array(
                # 0  1  2  3  4  5  6
                [1, 1, 1, 1, 3, 1, 1]
            ),
            1,
            np.array([1, 1, 1, 1, 1, 1, 1]),
        ),
        (
            Path("tests/data/vtu/complex/mesh.vtu"),
            np.array(
                # 0  1  2  3  4  5  6
                [0, 1, 2, 3, 4, 5, 6]
            ),
            3,
            np.array([3, 3, 3, 3, 3, 3, 3]),
        ),
    ],
)
def test_median_cells(
    filename: Path,
    input_data: np.ndarray,
    n_hop: int,
    expected: np.ndarray,
) -> None:
    """median_cells returns expected result."""
    mesh = graphlow.read(filename, backend="phlower")
    backend = mesh.backend
    x = backend.as_tensor(input_data, dimension={"Theta": 1})
    out = mapping.median_cells(mesh, x, n_hop=n_hop)
    assert out.dimension == pt.phlower_dimension_tensor({"Theta": 1})
    np.testing.assert_array_almost_equal(out.numpy(), expected)
