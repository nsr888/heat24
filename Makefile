# Makefile for heat24 project

# Variables
PYTHON := uv run python
PIP := uv pip

.PHONY: annual
annual:
	$(PYTHON) annual_heat_index.py
	xdg-open annual_heat_index_plots.pdf

.PHONY: month
month:
	$(PYTHON) month_heat_index.py
	xdg-open month_heat_index_plots.pdf

.PHONY: last_week
last_week:
	$(PYTHON) last_week_heat_index.py
	xdg-open last_week_heat_index.pdf

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  run         - Run the main application"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"
	@echo "  typecheck   - Run type checking"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean cache files"
	@echo "  help        - Show this help message"

# Install dependencies
.PHONY: install
install:
	$(PIP) install -e .

# Run the application
.PHONY: run
run:
	$(PYTHON) main.py
	xdg-open heat_index_plots.pdf

# Lint code
.PHONY: lint
lint:
	ruff check .

# Format code
.PHONY: format
format:
	ruff format .

# Type checking
.PHONY: typecheck
typecheck:
	mypy .

# Run tests
.PHONY: test
test:
	pytest

# Clean cache files
.PHONY: clean
clean:
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Run all checks
.PHONY: check
check: lint format typecheck test
