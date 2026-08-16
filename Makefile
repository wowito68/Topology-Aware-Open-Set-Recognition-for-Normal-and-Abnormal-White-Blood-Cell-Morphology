PYTHON ?= python3

.PHONY: install lint format-check type test smoke quality

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

format-check:
	ruff format --check .

type:
	mypy src

test:
	pytest

smoke:
	$(PYTHON) -m hemato_osr smoke --output-dir artifacts/smoke

quality: lint format-check type test smoke

