IN_PROJECT?=true


.PHONY: init
init:
	rm -r ./.venv || true
	poetry config virtualenvs.in-project ${IN_PROJECT}
	poetry install


.PHONY: mypy
mypy:
	poetry run mypy src --check-untyped-defs

.PHONY: format
format:
	poetry run python3 -m ruff check --fix
	poetry run python3 -m ruff format

.PHONY: test
test:
	poetry run pytest tests --cov=src --cov-report term-missing --durations 5
	poetry run pytest tests -m with_device --gpu

.PHONY: lint
lint:
	poetry run python3 -m ruff check --output-format=full
	poetry run python3 -m ruff format --diff
	# $(MAKE) mypy

.PHONY: benchmark
benchmark:
	poetry run pytest -v -m with_benchmark --benchmark-save-data --benchmark-time-unit='ms' --benchmark-storage=./tests/outputs/benchmark/time --benchmark-save=geo
	poetry run pytest -v -m with_memray --memray --memray-bin-path=./tests/outputs/benchmark/memory --memray-bin-prefix=geo
	poetry run memray flamegraph ./tests/outputs/benchmark/memory/geo-tests-benchmark-test_memory.py-test_compute_volumes_memray[file_name0].bin

.PHONY: document
document:
	$(RM) -r public
	$(RM) -r docs/build docs/source/tutorials docs/source/modules sg_execution_times.rst
	poetry run sphinx-apidoc -f -o docs/source/modules src
	sed -i "1s/^src$$/Module Reference/" ./docs/source/modules/modules.rst
	poetry run sphinx-build -M html docs/source docs/build
