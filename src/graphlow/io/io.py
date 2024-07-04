import pathlib

import pyvista as pv

from graphlow import GraphlowMesh


def read(file: str | pathlib.Path) -> GraphlowMesh:
    return GraphlowMesh(pv.read(file))
