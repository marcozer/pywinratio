.PHONY: help install test doctest lint format validate docs docs-fast build zenodo case-study check clean

PY ?= python3
VERSION := $(shell $(PY) -c "import re;print(re.search(r'\"(.+)\"', open('src/pywinratio/_version.py').read()).group(1))")

help:
	@echo "Targets:"
	@echo "  make install     Editable install with all extras"
	@echo "  make test        Unit, property and R-agreement tests"
	@echo "  make doctest     Docstring examples"
	@echo "  make lint        Ruff lint and format check"
	@echo "  make validate    Compare with the committed BuyseTest/WINS references"
	@echo "  make docs        Build the documentation (executes tutorials, warnings are errors)"
	@echo "  make build       Wheel and sdist, checked with twine"
	@echo "  make zenodo      Source archive + wheel + sdist for a manual Zenodo upload"
	@echo "  make case-study  Regenerate the synthetic case-study data and results"
	@echo "  make check       Everything above except zenodo and case-study"
	@echo "  make clean       Remove build products and caches"

install:
	$(PY) -m pip install -e ".[dev,docs,study]"

test:
	$(PY) -m pytest

doctest:
	$(PY) -m pytest --doctest-modules src/pywinratio

lint:
	ruff check src tests studies examples validation
	ruff format --check src tests studies examples validation

format:
	ruff format src tests studies examples validation
	ruff check --fix src tests studies examples validation

validate:
	$(PY) validation/compare.py

docs:
	$(PY) -m sphinx -W --keep-going -b html docs docs/_build/html

docs-fast:
	NB_EXECUTION_MODE=off $(PY) -m sphinx -b html docs docs/_build/html

build:
	rm -rf dist
	$(PY) -m build
	$(PY) -m twine check --strict dist/*

zenodo: build
	git archive --format=zip --prefix=pywinratio-$(VERSION)/ -o dist/pywinratio-$(VERSION).zip HEAD
	@ls -l dist

case-study:
	$(PY) -m studies.distal_pancreatectomy.export_public_data
	$(PY) -m studies.distal_pancreatectomy.run_public_analysis --n-boot 400

check: lint test doctest validate docs build

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist docs/_build docs/api/generated docs/jupyter_execute
