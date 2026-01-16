IN_PROJECT?=true

.PHONY: reset
reset:
	rm -r ./.venv || true
	rm uv.lock || true

.PHONY: install
install:
	uv sync --refresh --reinstall

.PHONY: dev-install
dev-install:
	uv sync --refresh --reinstall --group dev

.PHONY: lint
lint:
	uv run ruff check --output-format=full
	uv run ruff format --diff
	# uv run mypy src --check-untyped-defs

.PHONY: cpu-test
cpu-test:
	uv run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: gpu-test
gpu-test:
	uv run pytest tests -m with_device --gpu --cov=src --cov-report term-missing --durations 5

.PHONY: benchmark
benchmark:
	mkdir -p ./tests/outputs/benchmark
	uv sync --refresh --reinstall --group benchmark
	uv run pytest -v -m with_benchmark --benchmark-min-rounds=3 --benchmark-save-data --benchmark-time-unit='ms' --benchmark-storage=./tests/outputs/benchmark --benchmark-autosave
	uv run python visualization/benchmark.py

.PHONY: profile-time
profile-time:
	mkdir -p ./tests/outputs/profile/time/
	uv run pyinstrument -r html -o ./tests/outputs/profile/time/profile.html -m pytest -v -m with_profile

.PHONY: profile-memory
profile-memory:
	mkdir -p ./tests/outputs/profile/memory/
	uv run pytest -v -m with_profile --memray --memray-bin-path=./tests/outputs/profile/memory --memray-bin-prefix=graphlow
	uv run memray flamegraph -f ./tests/outputs/profile/memory/graphlow-tests-test_profile.py-test_compute_volumes_memray.bin

.PHONY: document
document:
	rm -rf public
	rm -rf docs/build docs/source/tutorials docs/source/modules sg_execution_times.rst
	uv run sphinx-apidoc -f -o docs/source/modules src
	sed -i "1s/^src$$/Module Reference/" ./docs/source/modules/modules.rst
	uv run sphinx-build -M html docs/source docs/build
