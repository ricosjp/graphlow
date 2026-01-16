import pathlib
from collections import defaultdict

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from pydantic import BaseModel, computed_field

# naming a layout theme for future reference
pio.templates["google"] = go.layout.Template(
    layout_colorway=[
        "#4285F4",
        "#DB4437",
        "#F4B400",
        "#0F9D58",
        "#185ABC",
        "#B31412",
        "#EA8600",
        "#137333",
        "#d2e3fc",
        "#ceead6",
    ]
)

# setting Google color palette as default
pio.templates.default = "google"


class CommitInfo(BaseModel, frozen=True):
    id: str
    time: str
    author_time: str
    dirty: bool
    project: str
    branch: str


class Stats(BaseModel, frozen=True):
    data: list[float]


class BenchmarkEntry(BaseModel, frozen=True):
    group: str
    name: str
    params: dict | None
    stats: Stats

    @computed_field
    @property
    def data(self) -> list[float]:
        return self.stats.data


class BenchmarkFile(BaseModel, frozen=True):
    commit_info: CommitInfo
    version: str
    benchmarks: list[BenchmarkEntry]

    @computed_field
    @property
    def benchmark_groups(self) -> dict[str, list[BenchmarkEntry]]:
        groups = defaultdict(list)
        for benchmark in self.benchmarks:
            groups[benchmark.group].append(benchmark)
        return groups


def load_benchmark(path: pathlib.Path) -> BenchmarkFile:
    with open(path) as f:
        return BenchmarkFile.model_validate_json(f.read())


def plot_surface_polyratio_vs_time(
    benchmark_file: BenchmarkFile, output_dir: pathlib.Path
):
    fig = go.Figure()
    fig.update_layout(
        template="google",
        xaxis_title="Polygon Ratio",
        yaxis_title="Processing Time (s)",
        title="Time of surface area computation vs. Polygon ratio",
        width=1200,
        height=800,
    )
    records = []
    group_name = "test_compute_areas_100x100_benchmark"

    benchmarks = benchmark_file.benchmark_groups[group_name]
    for bench in benchmarks:
        if bench.params is None:
            continue
        records.append(
            {
                "poly_ratio": bench.params["poly_ratio"],
                "time": bench.data,
            }
        )

    df = pl.DataFrame(records)
    df = df.with_columns(
        [
            pl.col("time").list.mean().alias("mean"),
            pl.col("time").list.std(ddof=1).alias("std"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("mean") + pl.col("std")).alias("upper"),
            (pl.col("mean") - pl.col("std")).alias("lower"),
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=df["poly_ratio"],
            y=df["upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            name="upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["poly_ratio"],
            y=df["lower"],
            mode="lines",
            fill="tonexty",
            line={"width": 0},
            showlegend=False,
            name="lower",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["poly_ratio"],
            y=df["mean"],
            mode="lines+markers",
            name=group_name,
        )
    )
    fig.write_html(output_dir / "surface_poly_ratio_vs_time.html")


def plot_volume_polyratio_vs_time(
    benchmark_file: BenchmarkFile, output_dir: pathlib.Path
):
    fig = go.Figure()
    fig.update_layout(
        template="google",
        xaxis_title="Polygon Ratio",
        yaxis_title="Processing Time (s)",
        title="Time of volume computation vs. Polygon ratio",
        width=1200,
        height=800,
    )
    records = []
    group_name = "test_compute_volumes_100x100_benchmark"

    benchmarks = benchmark_file.benchmark_groups[group_name]
    for bench in benchmarks:
        if bench.params is None:
            continue
        records.append(
            {
                "poly_ratio": bench.params["poly_ratio"],
                "time": bench.data,
            }
        )

    df = pl.DataFrame(records)
    df = df.with_columns(
        [
            pl.col("time").list.mean().alias("mean"),
            pl.col("time").list.std(ddof=1).alias("std"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("mean") + pl.col("std")).alias("upper"),
            (pl.col("mean") - pl.col("std")).alias("lower"),
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=df["poly_ratio"],
            y=df["upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            name="upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["poly_ratio"],
            y=df["lower"],
            mode="lines",
            fill="tonexty",
            line={"width": 0},
            showlegend=False,
            name="lower",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["poly_ratio"],
            y=df["mean"],
            mode="lines+markers",
            name=group_name,
        )
    )
    fig.write_html(output_dir / "volume_poly_ratio_vs_time.html")


if __name__ == "__main__":
    benchmark_dir = pathlib.Path(
        "tests/outputs/benchmark/Linux-CPython-3.10-64bit"
    )
    output_dir = pathlib.Path("tests/outputs/benchmark")
    files = benchmark_dir.glob("*.json")
    benchmark_files = sorted(
        [load_benchmark(file) for file in files],
        key=lambda x: x.commit_info.id[:8],
    )
    plot_surface_polyratio_vs_time(benchmark_files[-1], output_dir)
    plot_volume_polyratio_vs_time(benchmark_files[-1], output_dir)
