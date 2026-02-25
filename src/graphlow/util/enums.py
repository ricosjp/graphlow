from enum import IntEnum, StrEnum, auto

import torch


class SparseMatrixName(StrEnum):
    CELL_POINT_INCIDENCE = auto()
    CELL_ADJACENCY = auto()
    POINT_ADJACENCY = auto()
    POINT_DEGREE = auto()
    CELL_DEGREE = auto()
    NORMALIZED_POINT_ADJ = auto()
    NORMALIZED_CELL_ADJ = auto()
    FACET_CELL_INCIDENCE = auto()


class FeatureName(StrEnum):
    POINTS = auto()
    TIME_VALUE = auto()
    ORIGINAL_INDEX = auto()


DEFAULT_DIMENSIONS = {
    FeatureName.POINTS: {"L": 1},
    FeatureName.TIME_VALUE: {"T": 1},
}


class Extension(StrEnum):
    VTK = auto()
    VTM = auto()
    VTU = auto()
    VTP = auto()
    STL = auto()


class FloatPrecision(IntEnum):
    FLOAT32 = 32
    FLOAT64 = 64


PRECISION_TO_DTYPE = {
    FloatPrecision.FLOAT32: torch.float32,
    FloatPrecision.FLOAT64: torch.float64,
}
