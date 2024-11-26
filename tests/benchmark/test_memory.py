import pathlib

import pytest

import graphlow
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.with_memray
@pytest.mark.parametrize(
    "file_name",
    [pathlib.Path("tests/data/vtu/cube/large.vtu")],
)
def test_compute_volumes_memray(file_name: pathlib.Path):
    volmesh = graphlow.read(file_name)
    _ = volmesh.compute_volumes()
