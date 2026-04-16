from enum import StrEnum, auto


class FeatureName(StrEnum):
    """Feature names for pyvista mesh."""

    #: Points data.
    POINTS = auto()

    #: Time value data.
    TIME_VALUE = auto()

    #: Original point index stored during surface extraction.
    #:
    #: When extracting a boundary surface from a volume mesh, the resulting
    #: surface mesh stores the corresponding volume point indices in
    #: ``surface.point_data[ORIGINAL_INDEX]``.
    #: To avoid name conflict with other features, the key is prefixed with an
    #: underscore.
    ORIGINAL_INDEX = "_original_index"


DEFAULT_DIMENSIONS = {
    FeatureName.POINTS: {"L": 1.0},
    FeatureName.TIME_VALUE: {"T": 1.0},
}


class Extension(StrEnum):
    """File extensions for pyvista mesh."""

    VTK = auto()
    VTM = auto()
    VTU = auto()
    VTP = auto()
    STL = auto()


UNSTRUCTURED_GRID_EXTENSIONS = {
    Extension.VTK,
    Extension.VTU,
}

POLYDATA_EXTENSIONS = {
    Extension.VTK,
    Extension.VTP,
    Extension.STL,
}
