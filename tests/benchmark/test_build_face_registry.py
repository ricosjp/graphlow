"""Benchmarks for build_face_registry (face extraction from 3D cells)."""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Generator

import pytest
import pyvista as pv

from graphlow.core.face_registry import FaceRegistry, build_face_registry

TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = TESTS_DIR / "data"


def benchmark_with_group(func: Callable) -> Callable:
    """Attach a pytest-benchmark group matching the function name."""
    return pytest.mark.benchmark(group=func.__name__)(func)


def _load_volume_grid(path: pathlib.Path) -> pv.UnstructuredGrid:
    """
    Load benchmark mesh data as an ``UnstructuredGrid``.

    Returns
    -------
    pv.UnstructuredGrid
        Loaded benchmark grid.
    """
    grid = pv.read(path)
    return grid.cast_to_unstructured_grid()


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "poly_ratio",
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
@benchmark_with_group
def test_build_face_registry_100x100_volumes(
    benchmark: Generator, poly_ratio: int
) -> None:
    """
    Benchmark face registry construction on 100x100 volume meshes.
    """
    path = (
        DATA_DIR
        / "benchmark_mesh"
        / "volumes"
        / f"100x100_{poly_ratio:03d}.vtu"
    )
    if not path.exists():
        pytest.skip(f"Benchmark data not found: {path}")
    grid = _load_volume_grid(path)

    def run() -> FaceRegistry:
        return build_face_registry(grid)

    result = benchmark(run)
    assert result.n_faces() >= 0
