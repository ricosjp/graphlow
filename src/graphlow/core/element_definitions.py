"""Static VTK cell definitions: reference coordinates and face connectivity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

from graphlow.utils.topology_helper import TopologyDim

if TYPE_CHECKING:
    from graphlow.core.backend.base import Backend, TensorLike


def _face_patterns(
    patterns: dict[pv.CellType, list[list[int]]],
) -> dict[pv.CellType, np.ndarray]:
    """Build read-only face-pattern arrays keyed by face cell type."""
    result: dict[pv.CellType, np.ndarray] = {}
    for face_type, rows in patterns.items():
        array = np.asarray(rows, dtype=np.int64)
        array.setflags(write=False)
        result[face_type] = array
    return result


# =============================================================================
# Element Definitions
# =============================================================================
# reference_coords are nodal positions in reference space (VTK local-node
# order from GetParametricCoords). Tensor-product cells (QUAD / HEX /
# quadratic variants) use [-1, 1]^d here, while VTK stores [0, 1]^d;
# simplex cells match VTK coordinates as-is.
# Topology-only volume cells may omit reference_coords.


@dataclass(frozen=True)
class ElementDefinition:
    """Metadata for a VTK cell type.

    Parameters
    ----------
    n_nodes : int
        Number of element nodes.
    reference_dimension : TopologyDim
        Topological dimension of the element.
    reference_coords : tuple[tuple[float, ...], ...] or None
        Nodal coordinates in reference space, in VTK local-node order.
        ``None`` for topology-only definitions without FEM reference geometry.
    face_patterns : dict[pyvista.CellType, numpy.ndarray]
        Fixed face connectivities keyed by face cell type. Each array has
        shape ``(n_faces_of_type, n_face_nodes)``.
    """

    n_nodes: int
    reference_dimension: TopologyDim
    reference_coords: tuple[tuple[float, ...], ...] | None = None
    face_patterns: dict[pv.CellType, np.ndarray] = field(default_factory=dict)


_ELEMENTS: dict[pv.CellType, ElementDefinition] = {
    pv.CellType.TRIANGLE: ElementDefinition(
        n_nodes=3,
        reference_dimension=TopologyDim.SURFACE,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkTriangle.cxx#L39
        reference_coords=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    ),
    pv.CellType.QUAD: ElementDefinition(
        n_nodes=4,
        reference_dimension=TopologyDim.SURFACE,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkQuad.cxx#L36
        # Rescaled from VTK [0, 1]^2 to [-1, 1]^2.
        reference_coords=(
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        ),
    ),
    pv.CellType.TETRA: ElementDefinition(
        n_nodes=4,
        reference_dimension=TopologyDim.VOLUME,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkTetra.cxx#L40
        reference_coords=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkTetra.cxx#L58
        face_patterns=_face_patterns(
            {
                pv.CellType.TRIANGLE: [
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                    [0, 2, 1],
                ],
            }
        ),
    ),
    pv.CellType.VOXEL: ElementDefinition(
        n_nodes=8,
        reference_dimension=TopologyDim.VOLUME,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkVoxel.cxx#L74
        face_patterns=_face_patterns(
            {
                pv.CellType.PIXEL: [
                    [2, 0, 6, 4],
                    [1, 3, 5, 7],
                    [0, 1, 4, 5],
                    [3, 2, 7, 6],
                    [1, 0, 3, 2],
                    [4, 5, 6, 7],
                ],
            }
        ),
    ),
    pv.CellType.HEXAHEDRON: ElementDefinition(
        n_nodes=8,
        reference_dimension=TopologyDim.VOLUME,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkHexahedron.cxx#L45
        # Rescaled from VTK [0, 1]^3 to [-1, 1]^3.
        reference_coords=(
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ),
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkHexahedron.cxx#L73
        face_patterns=_face_patterns(
            {
                pv.CellType.QUAD: [
                    [0, 4, 7, 3],
                    [1, 2, 6, 5],
                    [0, 1, 5, 4],
                    [3, 7, 6, 2],
                    [0, 3, 2, 1],
                    [4, 5, 6, 7],
                ],
            }
        ),
    ),
    pv.CellType.WEDGE: ElementDefinition(
        n_nodes=6,
        reference_dimension=TopologyDim.VOLUME,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkWedge.cxx#L67
        face_patterns=_face_patterns(
            {
                pv.CellType.TRIANGLE: [
                    [0, 2, 1],
                    [3, 4, 5],
                ],
                pv.CellType.QUAD: [
                    [0, 1, 4, 3],
                    [1, 2, 5, 4],
                    [2, 0, 3, 5],
                ],
            }
        ),
    ),
    pv.CellType.PYRAMID: ElementDefinition(
        n_nodes=5,
        reference_dimension=TopologyDim.VOLUME,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkPyramid.cxx#L75
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkPyramid.cxx#L718
        face_patterns=_face_patterns(
            {
                pv.CellType.QUAD: [
                    [0, 3, 2, 1],
                ],
                pv.CellType.TRIANGLE: [
                    [0, 1, 4],
                    [1, 2, 4],
                    [2, 3, 4],
                    [3, 0, 4],
                ],
            }
        ),
    ),
    pv.CellType.QUADRATIC_TRIANGLE: ElementDefinition(
        n_nodes=6,
        reference_dimension=TopologyDim.SURFACE,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkQuadraticTriangle.cxx#L417
        reference_coords=(
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.0, 0.5),
        ),
    ),
    pv.CellType.QUADRATIC_QUAD: ElementDefinition(
        n_nodes=8,
        reference_dimension=TopologyDim.SURFACE,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkQuadraticQuad.cxx#L512
        # Rescaled from VTK [0, 1]^2 to [-1, 1]^2.
        reference_coords=(
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
        ),
    ),
    pv.CellType.QUADRATIC_TETRA: ElementDefinition(
        n_nodes=10,
        reference_dimension=TopologyDim.VOLUME,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkQuadraticTetra.cxx#L673
        reference_coords=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ),
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkQuadraticTetra.cxx#L62
        face_patterns=_face_patterns(
            {
                pv.CellType.QUADRATIC_TRIANGLE: [
                    [0, 1, 3, 4, 8, 7],
                    [1, 2, 3, 5, 9, 8],
                    [2, 0, 3, 6, 7, 9],
                    [0, 2, 1, 6, 5, 4],
                ],
            }
        ),
    ),
    pv.CellType.QUADRATIC_HEXAHEDRON: ElementDefinition(
        n_nodes=20,
        reference_dimension=TopologyDim.VOLUME,
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkQuadraticHexahedron.cxx#L737
        # Rescaled from VTK [0, 1]^3 to [-1, 1]^3.
        reference_coords=(
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (0.0, -1.0, -1.0),
            (1.0, 0.0, -1.0),
            (0.0, 1.0, -1.0),
            (-1.0, 0.0, -1.0),
            (0.0, -1.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (-1.0, 0.0, 1.0),
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
        # https://github.com/Kitware/VTK/blob/v9.7.0/Common/DataModel/vtkQuadraticHexahedron.cxx#L74
        face_patterns=_face_patterns(
            {
                pv.CellType.QUADRATIC_QUAD: [
                    [0, 4, 7, 3, 16, 15, 19, 11],
                    [1, 2, 6, 5, 9, 18, 13, 17],
                    [0, 1, 5, 4, 8, 17, 12, 16],
                    [3, 7, 6, 2, 19, 14, 18, 10],
                    [0, 3, 2, 1, 11, 10, 9, 8],
                    [4, 5, 6, 7, 12, 13, 14, 15],
                ],
            }
        ),
    ),
}


def get_element(cell_type: pv.CellType) -> ElementDefinition:
    """Return metadata for a supported VTK cell type.

    Parameters
    ----------
    cell_type : pyvista.CellType
        VTK cell type to look up.

    Returns
    -------
    ElementDefinition
        Cell metadata.

    Raises
    ------
    NotImplementedError
        If the cell type is not supported.
    """
    try:
        return _ELEMENTS[cell_type]
    except KeyError as error:
        raise NotImplementedError(
            f"Unsupported VTK cell type: {cell_type}"
        ) from error


def get_fixed_face_patterns(
    cell_type: pv.CellType,
) -> dict[pv.CellType, np.ndarray] | None:
    """
    Get the fixed face patterns for a given cell type.

    Parameters
    ----------
    cell_type : pv.CellType
        VTK cell type to query.

    Returns
    -------
    dict[pv.CellType, np.ndarray] or None
        Face patterns keyed by face type, or None if the cell type has no
        fixed face layout in this module.
    """
    element = _ELEMENTS.get(cell_type)
    if element is None or not element.face_patterns:
        return None
    return element.face_patterns


def supported_cell_types() -> tuple[pv.CellType, ...]:
    """Return VTK cell types with FEM reference coordinates.

    Returns
    -------
    tuple[pyvista.CellType, ...]
        Cell types that define ``reference_coords``, in definition order.
    """
    return tuple(
        cell_type
        for cell_type, element in _ELEMENTS.items()
        if element.reference_coords is not None
    )


def reference_coords[T: TensorLike](
    cell_type: pv.CellType,
    backend: Backend[T],
) -> T:
    """Return nodal coordinates in reference space.

    Parameters
    ----------
    cell_type : pyvista.CellType
        VTK cell type to look up.
    backend : Backend[T]
        Tensor backend used to materialize the coordinates.

    Returns
    -------
    T
        Coordinates with shape ``(n_nodes, reference_dimension)``,
        in VTK local-node order.

    Raises
    ------
    NotImplementedError
        If the cell type has no reference coordinates.
    """
    coords = get_element(cell_type).reference_coords
    if coords is None:
        raise NotImplementedError(
            f"No reference coordinates for VTK cell type: {cell_type}"
        )
    return backend.as_tensor(coords)
