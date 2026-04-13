"""
Plot compute_areas benchmark: poly_ratio (x) vs time (y) line chart.

Reads pytest-benchmark JSON and plots test_compute_areas_100x100_benchmark
results. When using `make benchmark`, the Makefile writes benchmark JSON to
tests/outputs/benchmark/latest.json and then runs the face_registry plot; run
this script manually for areas, or add it to the Makefile.

Standalone:

    uv run --group visualize python \
        tests/visualize/plot_compute_areas_benchmark.py [path/to/benchmark.json]

Output: tests/outputs/visualize/compute_areas_benchmark.html (optional .png).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go

TESTS_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_JSON_LATEST = TESTS_DIR / "outputs/benchmark/latest.json"
DEFAULT_JSON = BENCHMARK_JSON_LATEST
OUT_DIR = TESTS_DIR / "outputs/visualize"
GROUP_NAME = "test_compute_areas_100x100_benchmark"


def load_series(json_path: Path) -> tuple[list[int], list[float]]:
    """
    Load benchmark JSON and return (poly_ratios, mean_times_sec) for the
    compute_areas 100x100 benchmark.
    """
    with open(json_path) as f:
        data = json.load(f)

    poly_ratios: list[int] = []
    mean_times: list[float] = []

    for bench in data.get("benchmarks", []):
        if bench.get("group") != GROUP_NAME:
            continue
        params = bench.get("params") or {}
        if "poly_ratio" not in params:
            continue
        stats = bench.get("stats") or {}
        mean = stats.get("mean")
        if mean is None:
            continue
        poly_ratios.append(int(params["poly_ratio"]))
        mean_times.append(float(mean))

    pairs = sorted(zip(poly_ratios, mean_times, strict=True))
    if not pairs:
        return [], []
    poly_ratios, mean_times = zip(*pairs, strict=True)
    return list(poly_ratios), list(mean_times)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot compute_areas 100x100 benchmark: poly_ratio vs time"
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to pytest-benchmark JSON \
            (default: tests/outputs/benchmark/latest.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path \
            (default: tests/outputs/visualize/compute_areas_benchmark.html)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also save PNG (requires kaleido).",
    )
    args = parser.parse_args()

    if not args.json_path.exists():
        raise SystemExit(f"Benchmark file not found: {args.json_path}")

    poly_ratios, mean_times = load_series(args.json_path)
    if not poly_ratios:
        raise SystemExit(
            f"No '{GROUP_NAME}' benchmarks found in {args.json_path}"
        )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=poly_ratios,
                y=mean_times,
                mode="lines+markers",
                name="mean time",
                line={"width": 2},
                marker={"size": 8},
            )
        ],
        layout=go.Layout(
            title="face_areas (100x100 surfaces): poly_ratio vs mean time",
            xaxis={"title": "poly_ratio (%)", "dtick": 10},
            yaxis={"title": "Time (s)", "tickformat": ".4f"},
            template="plotly_white",
            height=450,
            margin={"l": 60, "r": 40, "t": 60, "b": 60},
        ),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_html = args.output or (OUT_DIR / "compute_areas_benchmark.html")
    fig.write_html(str(out_html))
    print(f"Wrote {out_html}")

    if args.png:
        try:
            out_png = out_html.with_suffix(".png")
            fig.write_image(str(out_png))
            print(f"Wrote {out_png}")
        except Exception as e:
            raise SystemExit(
                f"PNG export failed (install kaleido?): {e}"
            ) from e


if __name__ == "__main__":
    main()
