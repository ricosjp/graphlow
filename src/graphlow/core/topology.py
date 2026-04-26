"""Topology utilities for :class:`graphlow.core.mesh.TensorMesh`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyvista as pv
import scipy.sparse as sps

import graphlow.graph.skeleton_builder as skel_builder
import graphlow.graph.skeleton_derived as skel_derived
from graphlow.core.blocks import (
    CellBlock,
    FaceBlock,
    FixedCellBlock,
    JaggedCellBlock,
)
from graphlow.core.face_registry import (
    FaceRegistry,
    build_face_blocks_from_registry,
    build_face_registry,
)
from graphlow.graph import mapping as mapping_mod
from graphlow.graph.skeleton_builder import (
    AdjacencyName,
    IncidenceName,
)
from graphlow.graph.skeleton_derived import DerivedMatrixName
from graphlow.utils.topology_helper import (
    TopologyDim,
    gather_fixed_elements,
    gather_jagged_elements,
    get_cell_dimension,
    select_cells_by_type,
)

if TYPE_CHECKING:
    from graphlow.core.backend.base import TensorLike
    from graphlow.core.mesh import TensorMesh

logger = logging.getLogger(__name__)


class MeshTopology[T: TensorLike]:
    """
    Topology API for fixed-connectivity meshes.

    This class provides:

    - Access to basic connectivity and cell type information from ``pvmesh``.
    - Lazy construction and caching of incidence/adjacency skeletons
      (SciPy sparse) and their backend-materialized counterparts.
    - Block views (fixed-length vs jagged) for vectorized geometry/topology
      operations.
    - Mapping operations between point-, face-, and cell-associated fields.

    All topology-derived host-side data is backend-neutral and cached. Backend
    materialization is delegated to :class:`graphlow.core.cache.BackendCache`.

    Notes
    -----
    Caches assume that connectivity does not change. If connectivity changes,
    call :meth:`invalidate`.
    """

    def __init__(self, mesh: TensorMesh[T]) -> None:
        self._mesh = mesh
        self._unique_cell_types: np.ndarray | None = None
        self._mesh_dim: TopologyDim | None = None
        self._skeletons: dict[str, sps.csr_array] = {}
        self._cell_blocks: dict[pv.CellType, CellBlock] = {}
        self._built_face_registry: bool = False
        self._face_registry: FaceRegistry | None = None
        self._face_blocks: dict[pv.CellType, FaceBlock] = {}
        self._built_face_blocks: bool = False
        self._cell_tet_conn: np.ndarray | None = None

    # =========================================================================
    # Basic info
    # =========================================================================
    @property
    def n_points(self) -> int:
        """Number of points in the mesh."""
        return self._mesh.n_points

    @property
    def n_cells(self) -> int:
        """Number of cells in the mesh."""
        return self._mesh.n_cells

    def cell_conn(self) -> np.ndarray:
        """Cell connectivity array (VTK-style flattened connectivity)."""
        return self._mesh.pvmesh.cell_connectivity

    def cell_tet_conn(self) -> np.ndarray:
        """(n_cell, 4)-shaped cell connectivity array for tet cells."""
        if self._cell_tet_conn is not None:
            return self._cell_tet_conn
        if np.all(self._mesh.topology.unique_cell_types() != pv.CellType.TETRA):
            raise ValueError(
                "cell_tet_conn not supported for cell types: "
                f"{self._mesh.topology.unique_cell_types()}"
            )
        self._cell_tet_conn = self._mesh.topology.cell_conn().reshape(-1, 4)
        return self._cell_tet_conn

    def cell_offsets(self) -> np.ndarray:
        """Cell offsets array (VTK-style)."""
        return self._mesh.pvmesh.offset

    def cell_types(self) -> np.ndarray:
        """Cell types array (VTK cell type IDs)."""
        return self._mesh.pvmesh.celltypes

    def unique_cell_types(self) -> np.ndarray:
        """Unique cell type IDs present in the mesh."""
        if self._unique_cell_types is None:
            self._unique_cell_types = np.unique(self.cell_types())
        return self._unique_cell_types

    def mesh_dim(self) -> TopologyDim:
        """Maximum topological dimension present in the mesh."""
        if self._mesh_dim is None:
            types = self.unique_cell_types()
            self._mesh_dim = max(
                get_cell_dimension(pv.CellType(ct)) for ct in types
            )
        return self._mesh_dim

    def face_registry(self) -> FaceRegistry:
        """
        Build or return the cached face registry for a volume mesh.

        Advanced API: returns a FaceRegistry on the host and touches
        the internal face registry cache. Use when you need internal
        face data such as owner/neighbor/face_type/connectivity.

        Returns
        -------
        FaceRegistry
            Registry that can identify unique faces and boundary faces.

        Raises
        ------
        ValueError
            If the mesh dimension is less than volume.
        """
        if self.mesh_dim() < TopologyDim.VOLUME:
            raise ValueError(
                "Face registry is not available for less than volume dimension."
            )
        if not self._built_face_registry or self._face_registry is None:
            logger.debug("Building face registry")
            self._face_registry = build_face_registry(self._mesh.pvmesh)
            self._built_face_registry = True
        return self._face_registry

    def invalidate(self) -> None:
        """
        Drop all topology caches.

        Notes
        -----
        Call this after changing connectivity (cells, offsets, types). This also
        invalidates the backend cache via ``mesh.bcache.invalidate()``.
        """
        self._unique_cell_types = None
        self._cell_tet_conn = None
        self._mesh_dim = None
        self._skeletons.clear()
        self._cell_blocks.clear()
        self._face_registry = None
        self._built_face_registry = False
        self._face_blocks.clear()
        self._built_face_blocks = False
        self._mesh.bcache.invalidate()

    # =========================================================================
    # Blocks (Advanced API)
    # =========================================================================
    def cell_block(self, cell_type: pv.CellType) -> CellBlock | None:
        """
        Return cell block for specified cell type.

        Advanced API: returns a CellBlock on the host and touches
        the cell block cache. Use when you need cell connectivity data.

        Parameters
        ----------
        cell_type : pv.CellType
            VTK cell type to extract.

        Returns
        -------
        CellBlock | None
            If the corresponding cell does not exist, return None.
            If the cell is a fixed-length cell, return FixedCellBlock.
            If the cell is a variable-length cell, return JaggedCellBlock.
        """
        if cell_type not in self.unique_cell_types():
            return None

        if cell_type in self._cell_blocks:
            return self._cell_blocks[cell_type]

        block = self._extract_cell_block(cell_type)
        self._cell_blocks[cell_type] = block
        return block

    def face_block(
        self, face_type: pv.CellType, boundary_only: bool = False
    ) -> FaceBlock | None:
        """
        Return face block for specified face type.

        Advanced API: returns a FaceBlock on the host and touches
        the face block cache. Use when you need face connectivity data.

        Parameters
        ----------
        face_type : pv.CellType
            VTK cell type ID to extract faces.
        boundary_only : bool
            If True, only return boundary faces.
            If False, return all faces including internal faces.

        Returns
        -------
        FaceBlock | None
            If the corresponding face does not exist, return None.
            If the face is a fixed-length face, return FixedFaceBlock.
            If the face is a variable-length face, return JaggedFaceBlock.
        """
        if not self._built_face_blocks:
            logger.debug("Building face blocks")
            self._face_blocks = self._build_face_blocks()
            self._built_face_blocks = True

        face_block = self._face_blocks.get(face_type)
        if face_block is None:
            return None

        if boundary_only:
            return face_block.boundary_faces()

        return face_block

    # =========================================================================
    # Skeletons (Advanced API)
    # =========================================================================
    def _derived_key(
        self, derived: DerivedMatrixName, name: AdjacencyName
    ) -> str:
        return f"{derived}_{name}"

    def get_skeleton(
        self, name: IncidenceName | AdjacencyName
    ) -> sps.csr_array:
        """
        Get or build basic skeleton (CP, PC, CC, PP, FC, CF, FP, PF).
        The result is cached.

        Advanced API: returns a scipy.sparse matrix on the host and touches
        the skeleton cache. Use when you need scipy directly or
        explicit control over host memory.

        Parameters
        ----------
        name : IncidenceName | AdjacencyName
            Incidence or adjacency name (CP, PC, CC, PP, FC, CF, FP, PF).

        Returns
        -------
        scipy.sparse.csr_array
            CSR array.
            dtype: np.int64
        """
        if name in self._skeletons:
            return self._skeletons[name]

        logger.debug("Building skeleton %s", name)
        mat = skel_builder.build_skeleton(self, name)
        self._skeletons[name] = mat
        return mat

    def get_derived_skeleton(
        self,
        derived: DerivedMatrixName,
        name: AdjacencyName,
    ) -> sps.csr_array:
        """
        Get or build derived skeleton (Laplacian, Degree, etc.).
        The result is cached.

        Advanced API: returns a scipy.sparse matrix on the host and touches
        the skeleton cache. Use when you need scipy directly or
        explicit control over host memory.

        Parameters
        ----------
        derived : DerivedMatrixName
            Derived matrix name (Laplacian, Degree, etc.).
        name : AdjacencyName
            Adjacency name (CC, PP).

        Returns
        -------
        scipy.sparse.csr_array
            CSR array.
            dtype: np.float64
        """
        cache_key = self._derived_key(derived, name)

        if cache_key in self._skeletons:
            return self._skeletons[cache_key]

        logger.debug("Building derived skeleton %s(%s)", derived, name)
        adj = self.get_skeleton(name)

        match derived:
            case DerivedMatrixName.DEGREE:
                mat = skel_derived.build_degree_matrix(adj)
            case DerivedMatrixName.NORMALIZED:
                mat = skel_derived.build_normalized_adjacency(adj)
            case DerivedMatrixName.LAPLACIAN:
                mat = skel_derived.build_laplacian(adj)
            case _:
                raise ValueError(f"Unknown derived matrix name: {derived}")

        self._skeletons[cache_key] = mat
        return mat

    def free_host_skeletons(self, name: str | None = None) -> None:
        """
        Free host skeletons.

        Advanced API: for explicit control over host memory. Frees the
        scipy.sparse skeleton cache so that memory can be reclaimed. Use
        when you have finished using get_skeleton / get_derived_skeleton
        and want to release host memory.

        Parameters
        ----------
        name : str | None
            Skeleton name. If None, free all skeletons.
            - For incidence/adjacency, the name is like "CP", "CC".
            - For derived skeletons,
            the name is formatted as "[DerivedMatrixName]_[AdjacencyName]".
            e.g. "DEGREE_PP", "NORMALIZED_PP", "LAPLACIAN_CC".
        """
        if name is None:
            self._skeletons.clear()
        else:
            del self._skeletons[name]

    # =========================================================================
    # Sparse (backend materialize)
    # =========================================================================
    def _materialize_derived(
        self,
        derived: DerivedMatrixName,
        name: AdjacencyName,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """Materialize derived skeleton (degree, normalized_adj, laplacian)."""
        cache_key = self._derived_key(derived, name)
        skel = self.get_derived_skeleton(derived, name)
        return self._mesh.bcache.sparse(skel, cache_key, layout)

    def cell_point_incidence(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize cell-point incidence (CP).

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_cells, n_points)``. dtype is float
            with the precision set by the backend.
        """
        skeleton = self.get_skeleton(IncidenceName.CP)
        return self._mesh.bcache.sparse(skeleton, IncidenceName.CP, layout)

    def point_cell_incidence(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize point-cell incidence (PC).

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_points, n_cells)``. dtype is float
            with the precision set by the backend.
        """
        skeleton = self.get_skeleton(IncidenceName.PC)
        return self._mesh.bcache.sparse(skeleton, IncidenceName.PC, layout)

    def face_cell_incidence(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize face-cell incidence (FC).
        Its value is 1 for the owner cell and -1 for the neighbor cell.

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_faces, n_cells)``. dtype is float
            with the precision set by the backend.
        """
        skeleton = self.get_skeleton(IncidenceName.FC)
        return self._mesh.bcache.sparse(skeleton, IncidenceName.FC, layout)

    def cell_face_incidence(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize cell-face incidence (CF).
        Its value is 1 for the owner face and -1 for the neighbor face.

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_cells, n_faces)``. dtype is float
            with the precision set by the backend.
        """
        skeleton = self.get_skeleton(IncidenceName.CF)
        return self._mesh.bcache.sparse(skeleton, IncidenceName.CF, layout)

    def point_face_incidence(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize point-face incidence (PF).

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_points, n_faces)``. dtype is float
            with the precision set by the backend.

        Notes
        -----
        Here, "face" means a face of a volume mesh.
        This method does not apply to surface meshes.
        For surface meshes, use ``point_cell_incidence()`` instead.
        """
        skeleton = self.get_skeleton(IncidenceName.PF)
        return self._mesh.bcache.sparse(skeleton, IncidenceName.PF, layout)

    def face_point_incidence(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize face-point incidence (FP).

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_faces, n_points)``. dtype is float
            with the precision set by the backend.

        Notes
        -----
        Here, "face" means a face of a volume mesh.
        This method does not apply to surface meshes.
        For surface meshes, use ``cell_point_incidence()`` instead.
        """
        skeleton = self.get_skeleton(IncidenceName.FP)
        return self._mesh.bcache.sparse(skeleton, IncidenceName.FP, layout)

    def point_adjacency(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize point adjacency (PP).

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_points, n_points)``. dtype is float
            with the precision set by the backend.
        """
        skeleton = self.get_skeleton(AdjacencyName.PP)
        return self._mesh.bcache.sparse(skeleton, AdjacencyName.PP, layout)

    def cell_adjacency(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize cell adjacency (CC).

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_cells, n_cells)``. dtype is float
            with the precision set by the backend.
        """
        skeleton = self.get_skeleton(AdjacencyName.CC)
        return self._mesh.bcache.sparse(skeleton, AdjacencyName.CC, layout)

    def point_degree_matrix(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize point degree matrix.

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_points, n_points)``. dtype is float
            with the precision set by the backend.
        """
        return self._materialize_derived(
            DerivedMatrixName.DEGREE, AdjacencyName.PP, layout
        )

    def cell_degree_matrix(
        self,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """
        Materialize cell degree matrix.

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix. Default is "coo".

        Returns
        -------
        T
            Backend tensor of shape ``(n_cells, n_cells)``. dtype is float
            with the precision set by the backend.
        """
        return self._materialize_derived(
            DerivedMatrixName.DEGREE, AdjacencyName.CC, layout
        )

    def normalized_point_adjacency(
        self, layout: Literal["csr", "coo"] = "coo"
    ) -> T:
        """
        Materialize normalized point adjacency.

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix.

        Returns
        -------
        T
            Backend tensor of shape ``(n_points, n_points)``. dtype is float
            with the precision set by the backend.
        """
        return self._materialize_derived(
            DerivedMatrixName.NORMALIZED, AdjacencyName.PP, layout
        )

    def normalized_cell_adjacency(
        self, layout: Literal["csr", "coo"] = "coo"
    ) -> T:
        """
        Materialize normalized cell adjacency.

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix.

        Returns
        -------
        T
            Backend tensor of shape ``(n_cells, n_cells)``. dtype is float
            with the precision set by the backend.
        """
        return self._materialize_derived(
            DerivedMatrixName.NORMALIZED, AdjacencyName.CC, layout
        )

    def point_laplacian_matrix(
        self, layout: Literal["csr", "coo"] = "coo"
    ) -> T:
        """
        Materialize point Laplacian matrix.

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix.

        Returns
        -------
        T
            Backend tensor of shape ``(n_points, n_points)``. dtype is float
            with the precision set by the backend.
        """
        return self._materialize_derived(
            DerivedMatrixName.LAPLACIAN, AdjacencyName.PP, layout
        )

    def cell_laplacian_matrix(self, layout: Literal["csr", "coo"] = "coo") -> T:
        """
        Materialize cell Laplacian matrix.

        Parameters
        ----------
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix.

        Returns
        -------
        T
            Backend tensor of shape ``(n_cells, n_cells)``. dtype is float
            with the precision set by the backend.
        """
        return self._materialize_derived(
            DerivedMatrixName.LAPLACIAN, AdjacencyName.CC, layout
        )

    def free_backend_sparse(
        self, name: str | None = None, layout: Literal["csr", "coo"] = "coo"
    ) -> None:
        """
        Free backend sparse.

        Advanced API: for explicit control over backend (GPU/device) memory.
        Frees materialized sparse tensors from the backend cache. Use when
        you no longer need the sparse matrices and want to release
        device memory.

        Parameters
        ----------
        name : str | None
            Sparse name. If None, free all sparse.
            - For incidence/adjacency, the name is like "CP", "CC".
            - For derived skeletons,
            the name is formatted as "[DerivedMatrixName]_[AdjacencyName]".
            e.g. "DEGREE_PP", "NORMALIZED_PP", "LAPLACIAN_CC".
        layout : Literal["csr", "coo"]
            Layout of the sparse matrix.
        """
        if name is None:
            self._mesh.bcache.invalidate()
        else:
            key = self._mesh.bcache.make_key("sparse", name, layout)
            self._mesh.bcache.purge(key)

    # =========================================================================
    # Mapping
    # =========================================================================
    def map_point_to_cell(
        self,
        x_point: T,
        mode: Literal["sum", "mean", "conservative"],
        method: Literal["segment", "sparse"] = "segment",
    ) -> T:
        r"""
        Map point field to cell.

        Parameters
        ----------
        x_point : T
            Tensor of shape ``(n_points, ...)``.
        mode : "sum" | "mean" | "conservative"
            Mapping mode.

            ``"sum"``
                Sum of point values in each cell.

            ``"mean"``
                Row-normalized CP operator.

                The cell value is computed as the simple average of the
                values at the points connected to the cell:

                .. math::

                   x_{\text{cell}} =
                   \frac{1}{\operatorname{deg}(C)}
                   \sum_{p \in C} x_{\text{point}}

                where :math:`\operatorname{deg}(C)` is the number of points
                connected to cell :math:`C`.

                This corresponds to a local arithmetic averaging over the
                cell stencil. It treats each point in the cell equally,
                regardless of how many cells that point belongs to.
                In general, this method does not guarantee global conservation
                of the total quantity.

            ``"conservative"``
                Column-normalized CP operator.

                The cell value is obtained by distributing each point value
                equally among the cells connected to that point:

                .. math::

                   x_{\text{cell}} =
                   \sum_{p \in C}
                   \left(\frac{1}{\operatorname{deg}(p)}\right)
                   x_{\text{point}}

                where :math:`\operatorname{deg}(p)` is the number of cells
                connected to point :math:`p`.

                Each point contributes its value conservatively to its
                neighboring cells, and the total sum over cells equals
                the total sum over points,
                making this formulation suitable for conservative transfers.
        method : "segment" | "sparse"
            Mapping method. Default is "segment".

            ``"segment"``
                Computes the reduction
                without explicitly constructing the CP matrix.
                It is typically more memory efficient, especially on GPU.
                In most practical settings, this method is recommended.

            ``"sparse"``
                Computes the reduction by multiplying the CP matrix
                with the point values.
                It depends strongly on backend implementation quality but
                it can be beneficial when the CP matrix is used repeatedly.

        Returns
        -------
        T
            Tensor of shape ``(n_cells, ...)``.
        """
        return mapping_mod.map_point_to_cell(
            self._mesh, x_point, mode=mode, method=method
        )

    def map_cell_to_point(
        self,
        x_cell: T,
        mode: Literal["sum", "mean", "conservative"],
        method: Literal["segment", "sparse"] = "segment",
    ) -> T:
        r"""
        Map cell field to point.

        Parameters
        ----------
        x_cell : T
            Tensor of shape ``(n_cells, ...)``.
        mode : "sum" | "mean" | "conservative"
            Mapping mode.

            ``"sum"``
                Sum of cell values that are connected to the point.

            ``"mean"``
                Row-normalized PC operator.

                The point value is computed as the arithmetic average of
                the values of the cells connected to the point:

                .. math::

                   x_{\text{point}} =
                   \frac{1}{\operatorname{deg}(p)}
                   \sum_{C \ni p} x_{\text{cell}}

                where :math:`\operatorname{deg}(p)` is the number of cells
                connected to point :math:`p`.

                This corresponds to a local averaging over the cell stencil
                around each point. Each neighboring cell contributes equally,
                regardless of the number of points in that cell.
                In general, this method does not guarantee global conservation
                of the total quantity.

            ``"conservative"``
                Column-normalized PC operator.

                The point value is obtained by distributing each cell value
                equally among the points connected to that cell:

                .. math::

                   x_{\text{point}} =
                   \sum_{C \ni p}
                   \left(\frac{1}{\operatorname{deg}(C)}\right)
                   x_{\text{cell}}

                where :math:`\operatorname{deg}(C)` is the number of points
                in cell :math:`C`.

                Each cell contributes its value conservatively to its
                vertices, and the total sum over points equals
                the total sum over cells,
                making this formulation suitable for conservative transfers.
        method : "segment" | "sparse"
            Mapping method. Default is "segment".

            ``"segment"``
                Computes the reduction
                without explicitly constructing the PC matrix.
                It is typically more memory efficient, especially on GPU.
                In most practical settings, this method is recommended.

            ``"sparse"``
                Computes the reduction by multiplying the PC matrix
                with the cell values.
                It depends strongly on backend implementation quality but
                it can be beneficial when the PC matrix is used repeatedly.

        Returns
        -------
        T
            Tensor of shape ``(n_points, ...)``.
        """
        return mapping_mod.map_cell_to_point(
            self._mesh, x_cell, mode=mode, method=method
        )

    def map_cell_to_face(
        self,
        x_cell: T,
        mode: Literal["sum", "mean", "conservative", "diff"],
        method: Literal["segment", "sparse"] = "segment",
    ) -> T:
        r"""
        Map cell field to face.

        Parameters
        ----------
        x_cell : T
            Tensor of shape ``(n_cells, ...)``.
        mode : "sum" | "mean" | "conservative" | "diff"
            Mapping mode.

            ``"sum"``
                Sum of cell values that are connected to the face.
                For manifold meshes, this corresponds to:

                .. math::

                   x_{\text{face}} =
                   x_{\text{cell}}[\text{owner}] +
                   x_{\text{cell}}[\text{neighbor}]

            ``"mean"``
                The face value is computed as the arithmetic average of
                the values of the cells connected to the face.
                For manifold meshes, this corresponds to:

                .. math::

                   x_{\text{face}} =
                   \frac{1}{2}
                   \left(
                   x_{\text{cell}}[\text{owner}] +
                   x_{\text{cell}}[\text{neighbor}]
                   \right)

                In general, this method does not guarantee global conservation
                of the total quantity.

            ``"conservative"``
                The face value is obtained by distributing each cell value
                equally among the faces connected to that cell:

                .. math::

                   x_{\text{face}} =
                   \sum_{C \ni f}
                   \left(\frac{1}{\operatorname{deg}(C)}\right)
                   x_{\text{cell}}

                where :math:`\operatorname{deg}(C)` is the number of faces
                in cell :math:`C`.

                Each cell contributes its value conservatively to its
                faces, and the total sum over faces equals
                the total sum over cells,
                making this formulation suitable for conservative transfers.

            ``"diff"``
                Difference of the cell field.
                The face value is computed as the difference of the cell field:

                .. math::

                   x_{\text{face}} =
                   x_{\text{cell}}[\text{owner}] -
                   x_{\text{cell}}[\text{neighbor}]

                This corresponds to a local difference over the cell stencil.
        method : "segment" | "sparse"
            Mapping method. Default is "segment".

            ``"segment"``
                Computes the reduction
                without explicitly constructing the FC matrix.
                It is typically more memory efficient, especially on GPU.
                In most practical settings, this method is recommended.

            ``"sparse"``
                Computes the reduction by multiplying the FC matrix
                with the cell values.
                It depends strongly on backend implementation quality but
                it can be beneficial when the FC matrix is used repeatedly.

        Returns
        -------
        T
            Tensor of shape ``(n_faces, ...)``.
        """
        return mapping_mod.map_cell_to_face(
            self._mesh, x_cell, mode=mode, method=method
        )

    def map_face_to_cell(
        self,
        x_face: T,
        mode: Literal["sum", "mean", "conservative", "div"],
        method: Literal["segment", "sparse"] = "segment",
    ) -> T:
        r"""
        Map face field to cell.

        Parameters
        ----------
        x_face : T
            Tensor of shape ``(n_faces, ...)``.
        mode : "sum" | "mean" | "conservative" | "div"
            Mapping mode.

            ``"sum"``
                Sum of face values in cell.

            ``"mean"``
                Row-normalized CF operator.

                The cell value is computed as the arithmetic average of
                the values of the faces connected to the cell:

                .. math::

                   x_{\text{cell}} =
                   \frac{1}{\operatorname{deg}(C)}
                   \sum_{f \ni C} x_{\text{face}}

                where :math:`\operatorname{deg}(C)` is the number of faces
                connected to cell :math:`C`.

                This corresponds to a local arithmetic averaging over the
                cell stencil.
                In general, this method does not guarantee global conservation
                of the total quantity.

            ``"conservative"``
                Column-normalized CF operator.

                The cell value is obtained by distributing each face value
                equally among the cells connected to that face:

                .. math::

                   x_{\text{cell}} =
                   \sum_{f \ni C}
                   \left(\frac{1}{\operatorname{deg}(f)}\right)
                   x_{\text{face}}

                where :math:`\operatorname{deg}(f)` is the number of cells
                connected to face :math:`f`.

                Each face contributes its value conservatively to its
                neighboring cells, and the total sum over cells equals
                the total sum over faces,
                making this formulation suitable for conservative transfers.

            ``"div"``
                Divergence of the face field.
                The cell value is computed as the divergence of the face field:

                .. math::

                   x_{\text{cell}} =
                   \sum_{f \ni C} x_{\text{face}} \cdot n_f

                where :math:`n_f` is the normal vector of face :math:`f`.

                This corresponds to a local divergence over the cell stencil.
        method : "segment" | "sparse"
            Mapping method. Default is "segment".

            ``"segment"``
                Computes the reduction
                without explicitly constructing the CF matrix.
                It is typically more memory efficient, especially on GPU.
                In most practical settings, this method is recommended.

            ``"sparse"``
                Computes the reduction by multiplying the CF matrix
                with the face values.
                It depends strongly on backend implementation quality but
                it can be beneficial when the CF matrix is used repeatedly.

        Returns
        -------
        T
            Tensor of shape ``(n_cells, ...)``.
        """
        return mapping_mod.map_face_to_cell(
            self._mesh, x_face, mode=mode, method=method
        )

    def map_face_to_point(
        self,
        x_face: T,
        mode: Literal["sum", "mean", "conservative"],
        method: Literal["segment", "sparse"] = "segment",
    ) -> T:
        r"""
        Map face field to point.

        Parameters
        ----------
        x_face : T
            Tensor of shape ``(n_faces, ...)``.
        mode : "sum" | "mean" | "conservative"
            Mapping mode.

            ``"sum"``
                Sum of face values that are connected to the point.

            ``"mean"``
                Row-normalized PF operator.

                The point value is computed as the arithmetic average of
                the values of the faces connected to the point:

                .. math::

                   x_{\text{point}} =
                   \frac{1}{\operatorname{deg}(p)}
                   \sum_{f \ni p} x_{\text{face}}

                where :math:`\operatorname{deg}(p)` is the number of faces
                connected to point :math:`p`.

                This corresponds to a local averaging over the face stencil
                around each point. Each neighboring face contributes equally,
                regardless of the number of points in that face.
                In general, this method does not guarantee global conservation
                of the total quantity.

            ``"conservative"``
                Column-normalized PF operator.

                The point value is obtained by distributing each face value
                equally among the points connected to that face:

                .. math::

                   x_{\text{point}} =
                   \sum_{f \ni p}
                   \left(\frac{1}{\operatorname{deg}(f)}\right)
                   x_{\text{face}}

                where :math:`\operatorname{deg}(f)` is the number of points
                in face :math:`f`.

                Each face contributes its value conservatively to its
                vertices, and the total sum over points equals
                the total sum over faces,
                making this formulation suitable for conservative transfers.
        method : "segment" | "sparse"
            Mapping method. Default is "segment".

            ``"segment"``
                Computes the reduction
                without explicitly constructing the PF matrix.
                It is typically more memory efficient, especially on GPU.
                In most practical settings, this method is recommended.

            ``"sparse"``
                Computes the reduction by multiplying the PF matrix
                with the face values.
                It depends strongly on backend implementation quality but
                it can be beneficial when the PF matrix is used repeatedly.

        Returns
        -------
        T
            Tensor of shape ``(n_points, ...)``.
        """
        return mapping_mod.map_face_to_point(
            self._mesh, x_face, mode=mode, method=method
        )

    def map_point_to_face(
        self,
        x_point: T,
        mode: Literal["sum", "mean", "conservative"],
        method: Literal["segment", "sparse"] = "segment",
    ) -> T:
        r"""
        Map point field to face.

        Parameters
        ----------
        x_point : T
            Tensor of shape ``(n_points, ...)``.
        mode : "sum" | "mean" | "conservative"
            Mapping mode.

            ``"sum"``
                Sum of point values in each face.

            ``"mean"``
                Row-normalized FP operator.

                The face value is computed as the simple average of the
                values at the points connected to the face:

                .. math::

                   x_{\text{face}} =
                   \frac{1}{\operatorname{deg}(f)}
                   \sum_{p \in f} x_{\text{point}}

                where :math:`\operatorname{deg}(f)` is the number of points
                connected to face :math:`f`.

                This corresponds to a local arithmetic averaging over the
                face stencil. It treats each point in the face equally,
                regardless of how many faces that point belongs to.
                In general, this method does not guarantee global conservation
                of the total quantity.

            ``"conservative"``
                Column-normalized FP operator.

                The face value is obtained by distributing each point value
                equally among the faces connected to that point:

                .. math::

                   x_{\text{face}} =
                   \sum_{p \in f}
                   \left(\frac{1}{\operatorname{deg}(p)}\right)
                   x_{\text{point}}

                where :math:`\operatorname{deg}(p)` is the number of faces
                connected to point :math:`p`.

                Each point contributes its value conservatively to its
                neighboring faces, and the total sum over faces equals
                the total sum over points,
                making this formulation suitable for conservative transfers.
        method : "segment" | "sparse"
            Mapping method. Default is "segment".

            ``"segment"``
                Computes the reduction
                without explicitly constructing the FP matrix.
                It is typically more memory efficient, especially on GPU.
                In most practical settings, this method is recommended.

            ``"sparse"``
                Computes the reduction by multiplying the FP matrix
                with the point values.
                It depends strongly on backend implementation quality but
                it can be beneficial when the FP matrix is used repeatedly.

        Returns
        -------
        T
            Tensor of shape ``(n_faces, ...)``.
        """
        return mapping_mod.map_point_to_face(
            self._mesh, x_point, mode=mode, method=method
        )

    def median_points(
        self,
        x_point: T,
        n_hop: int = 1,
    ) -> T:
        """
        Compute median among n-hop point neighbors.

        Parameters
        ----------
        x_point : backend tensor of shape ``(n_points, ...)``.
        n_hop : int
            The number of adjacent hops to consider.

        Returns
        -------
        T
            Tensor of shape ``(n_points, ...)``.
        """
        return mapping_mod.median_points(self._mesh, x_point, n_hop=n_hop)

    def median_cells(
        self,
        x_cell: T,
        n_hop: int = 1,
    ) -> T:
        """
        Compute median among n-hop cell neighbors.

        Parameters
        ----------
        x_cell : backend tensor of shape ``(n_cells, ...)``.
        n_hop : int
            The number of adjacent hops to consider.

        Returns
        -------
        T
            Tensor of shape ``(n_cells, ...)``.
        """
        return mapping_mod.median_cells(self._mesh, x_cell, n_hop=n_hop)

    # =========================================================================
    # Degree
    # =========================================================================
    def _degree_from_skeleton(self, name: IncidenceName | AdjacencyName) -> T:
        """Return degree vector (row-wise) from a skeleton matrix."""
        skeleton = self.get_skeleton(name)
        degrees_np = np.diff(skeleton.indptr).astype(np.float64)
        return self._mesh.backend.as_tensor(degrees_np, dimension={})

    def degree_cp(self) -> T:
        """
        Degree of cell to point.

        Returns
        -------
        T
            Tensor of shape ``(n_cells,)``. dtype is float.
        """
        return self._degree_from_skeleton(IncidenceName.CP)

    def degree_pc(self) -> T:
        """
        Degree of point to cell.

        Returns
        -------
        T
            Tensor of shape ``(n_points,)``. dtype is float.
        """
        return self._degree_from_skeleton(IncidenceName.PC)

    def degree_cc(self) -> T:
        """
        Degree of cell to cell.

        Returns
        -------
        T
            Tensor of shape ``(n_cells,)``. dtype is float.
        """
        return self._degree_from_skeleton(AdjacencyName.CC)

    def degree_pp(self) -> T:
        """
        Degree of point to point.

        Returns
        -------
        T
            Tensor of shape ``(n_points,)``. dtype is float.
        """
        return self._degree_from_skeleton(AdjacencyName.PP)

    def degree_fc(self) -> T:
        """
        Degree of face to cell.

        Returns
        -------
        T
            Tensor of shape ``(n_faces,)``. dtype is float.
        """
        return self._degree_from_skeleton(IncidenceName.FC)

    def degree_cf(self) -> T:
        """
        Degree of cell to face.

        Returns
        -------
        T
            Tensor of shape ``(n_cells,)``. dtype is float.
        """
        return self._degree_from_skeleton(IncidenceName.CF)

    def degree_pf(self) -> T:
        """
        Degree of point to face.

        Returns
        -------
        T
            Tensor of shape ``(n_points,)``. dtype is float.
        """
        return self._degree_from_skeleton(IncidenceName.PF)

    def degree_fp(self) -> T:
        """
        Degree of face to point.

        Returns
        -------
        T
            Tensor of shape ``(n_faces,)``. dtype is float.
        """
        return self._degree_from_skeleton(IncidenceName.FP)

    # =========================================================================
    # Helper: Connectivity Extraction
    # =========================================================================
    def _extract_cell_block(self, cell_type: pv.CellType) -> CellBlock:
        """
        Extract the connectivity of the cell and generate a CellBlock.

        PyVista's `cells_dict` raises an error
        when a variable-length cell is included in the mesh, so it is not used.
        Instead, extract the connectivity from `cell_connectivity` and `offset`.

        Parameters
        ----------
        cell_type : pv.CellType
            The cell type to extract.

        Returns
        -------
        CellBlock
            Extracted cell block.
            If the cell is fixed-length, return FixedCellBlock.
            If the cell is variable-length, return JaggedCellBlock.
        """
        conn = self.cell_conn()
        global_indices, start_offsets, end_offsets = select_cells_by_type(
            self.cell_types(), self.cell_offsets(), cell_type
        )

        # for variable-length cells (Polygon, Polyhedron)
        if cell_type in (pv.CellType.POLYGON, pv.CellType.POLYHEDRON):
            new_conn, new_offsets = gather_jagged_elements(
                conn, start_offsets, end_offsets
            )
            return JaggedCellBlock(
                cell_type, global_indices, new_conn, new_offsets
            )

        # for fixed-length cells (Tri, Quad, Tet, Hex)
        new_conn = gather_fixed_elements(conn, start_offsets, end_offsets)
        return FixedCellBlock(cell_type, global_indices, new_conn)

    def _build_face_blocks(self) -> dict[pv.CellType, FaceBlock]:
        """
        Extract the 2D faces from the 3D mesh and generate a dict of FaceBlock.
        Resolve the relationship between the "Owner" (owning cell) and
        "Neighbor" (neighboring cell).

        Returns
        -------
        dict[pv.CellType, FaceBlock]
            Extracted face blocks.
        """
        if self.mesh_dim() < TopologyDim.VOLUME:
            raise ValueError(
                "The mesh dimension is less than the volume dimension."
            )

        registry = self.face_registry()
        return build_face_blocks_from_registry(registry)
