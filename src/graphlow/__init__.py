from importlib.metadata import version

from graphlow.base.mesh import GraphlowMesh
from graphlow.io.io import read
from graphlow.util.enums import FloatPrecision

__version__ = version("graphlow")
__all__ = ["__version__"]
