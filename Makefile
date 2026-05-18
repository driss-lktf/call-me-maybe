.PHONY: install run debug clean lint lint-strict test help

UV ?= $(shell which uv 2>/dev/null || echo /opt/pyenv/versions/3.11.15/bin/uv)
SRC := src

# Redirect large caches away from the small home partition
export HF_HOME          ?= /goinfre/$(USER)/huggingface
export UV_CACHE_DIR     ?= /goinfre/$(USER)/uv_cache
export UV_LINK_MODE     ?= copy

help:
	@echo "Available targets:"
	@echo "  install      - install project dependencies via uv sync"
	@echo "  run          - run the function-calling pipeline"
	@echo "  debug        - run the pipeline under pdb"
	@echo "  clean        - remove caches and build artefacts"
	@echo "  lint         - flake8 + mypy with the project flags"
	@echo "  lint-strict  - flake8 + mypy --strict (optional)"
	@echo "  test         - run the pytest test suite"

install:
	$(UV) sync

run:
	$(UV) run python -m $(SRC) $(ARGS)

verbose:
	$(UV) run python -m $(SRC) --verbose

debug:
	$(UV) run python -m pdb -m $(SRC)

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -exec rm -rf {} +
	rm -rf build dist

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

test:
	$(UV) run pytest tests/ -v
