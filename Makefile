CUDA_TAG = cu124

.PHONY: reset
reset:
	rm -r ./.venv || true
	rm uv.lock || true

.PHONY: install
install:
	uv sync --refresh --reinstall --extra ${CUDA_TAG}

.PHONY: dev-install
dev-install:
	uv sync --refresh --reinstall --extra ${CUDA_TAG} --extra phlower --group dev

.PHONY: lint
lint:
	uv run ruff check --output-format=full
	uv run ruff format --diff

.PHONY: cpu-test
cpu-test:
	uv run pytest tests --device=cpu --cov=src --cov-report term-missing --durations 5 

.PHONY: gpu-test
gpu-test:
	uv run pytest tests --device=cuda --cov=src --cov-report term-missing --durations 5

.PHONY: slow-test
slow-test:
	uv run pytest tests -m slow --save

# For headless CI: install xvfb and run ``xvfb-run make document``.
.PHONY: document
document:
	rm -rf docs/build || true
	rm -rf docs/source/api_reference/generated/ || true
	rm -rf docs/source/example_gallery/auto_examples || true
	rm docs/source/sg_execution_times.rst || true
	uv run sphinx-build docs/source docs/build -b html

.PHONY: benchmark
benchmark:
	mkdir -p ./tests/outputs/benchmark
	uv run pytest -v -m benchmark \
		--benchmark-min-rounds=3 \
		--benchmark-save-data \
		--benchmark-time-unit=ms \
		--benchmark-json=./tests/outputs/benchmark/latest.json
		--benchmark-storage=./tests/outputs/benchmark
		--benchmark-autosave
	uv run --group visualize python tests/visualize/plot_face_registry_benchmark.py
	uv run --group visualize python tests/visualize/plot_compute_areas_benchmark.py
	uv run --group visualize python tests/visualize/plot_compute_volumes_benchmark.py

.PHONY: profile-time
profile-time:
	mkdir -p ./tests/outputs/profile/time/
	uv run pyinstrument -r html -o ./tests/outputs/profile/time/profile.html -m pytest -v -m profile

.PHONY: profile-memory
profile-memory:
	mkdir -p ./tests/outputs/profile/memory/
	uv run pytest -v -m profile --memray --memray-bin-path=./tests/outputs/profile/memory --memray-bin-prefix=graphlow
	uv run memray flamegraph -f ./tests/outputs/profile/memory/graphlow-tests-test_profile.py-test_compute_volumes_memray.bin

.PHONY: performance_check
performance_check: benchmark profile-time profile-memory

