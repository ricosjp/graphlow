import pathlib

import pytest

import graphlow
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.with_profile
def test_compute_volumes_memray():
    file_name = pathlib.Path("tests/data/vtu/cube/large.vtu")
    volmesh = graphlow.read(file_name)
    _ = volmesh.compute_volumes()
