from enum import Enum


# TODO: Use StrEnum after upgrading to Python 3.11
class SparseMatrixName(str, Enum):
    CELL_POINT_INCIDENCE = "cell_point_incidence"
    CELL_ADJACENCY = "cell_adjacency"
    POINT_ADJACENCY = "point_adjacency"
    POINT_DEGREE = "point_degree"
    CELL_DEGREE = "cell_degree"
    NORMALIZED_POINT_ADJ = "normalized_point_adj"
    NORMALIZED_CELL_ADJ = "normalized_cell_adj"
    FACET_CELL_INCIDENCE = "facet_cell_incidence"


class FeatureName(str, Enum):
    POINTS = "points"
    ORIGINAL_INDEX = "original_index"
    TIME_VALUE = "TimeValue"


class Extension(str, Enum):
    VTK = "vtk"
    VTM = "vtm"
    VTU = "vtu"
    VTP = "vtp"

    STL = "stl"
