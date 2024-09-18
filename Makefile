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


.PHONY: lint
lint:
	poetry run python3 -m ruff check --output-format=full
	poetry run python3 -m ruff format --diff
	# $(MAKE) mypy


.PHONY: document
document:
	$(RM) public
	$(RUN) sphinx-build -M html docs/source docs/build
