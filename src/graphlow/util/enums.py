
from enum import Enum


# TODO: Use StrEnum after upgrading to Python 3.11
class SparseMatrixName(str, Enum):
    CELL_POINT_INCIDENCE = "cell_point_incidence"
    CELL_ADJACENCY = "cell_adjacency"
    POINT_ADJACENCY = "point_adjacency"


class FeatureName(str, Enum):
    POINTS = "points"
    ORIGINAL_INDEX = "original_index"
