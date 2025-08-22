# Crush Agent Instructions

## Project Overview

Python project using numpy, pandas, matplotlib for weather data processing.

## Commands

- Run project: `uv run main.py`
- Install dependencies: `uv pip install -e .`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type check: `mypy .`
- Run tests: `pytest`
- Run single test: `pytest path/to/test_file.py::test_function_name`

## Code Style

- Use ruff for linting and formatting
- Use mypy for type checking
- Follow PEP 8
- Use type hints for all functions
- Use descriptive variable names
- Keep functions small and focused
- Use docstrings for modules, classes, and functions

## Import Style

- Standard library imports first
- Third-party imports second
- Local imports last
- Alphabetical order within each group
- No wildcards

## Naming Conventions

- snake_case for variables and functions
- PascalCase for classes
- UPPER_CASE for constants

## Error Handling

- Use specific exception types
- Log errors appropriately
- Don't catch exceptions silently
