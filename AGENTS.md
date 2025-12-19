# Repository Guidelines

## Project Structure & Module Organization
- `src/`: main Python packages.
  - `src/game/`: game engine and rules.
  - `src/abstraction/`: action/card abstraction and bucketing.
  - `src/solver/`: MCCFR solvers and storage.
  - `src/training/` and `src/evaluation/`: training loop and evaluation tools.
  - `src/cli/`: CLI handlers used by scripts.
- `tests/`: pytest suite mirroring `src/` modules (e.g., `tests/solver/`).
- `scripts/`: runnable entry points (training, querying, charts).
- `config/`: YAML configs for training and abstractions.
- `data/`: generated checkpoints and artifacts; treat as output, not source.

## Build, Test, and Development Commands
- `uv sync`: install dependencies into the virtual environment.
- `uv run python scripts/train.py --iterations 1000`: run a training session.
- `uv run python scripts/train_and_show.py --iterations 100`: train and print a summary.
- `uv run pytest`: run the full test suite.
- `uv run pytest -m "not slow"`: skip slow tests.
- `uv run pytest --cov=src --cov-report=html`: generate coverage report.

## Coding Style & Naming Conventions
- Python, 4-space indentation, line length 100 (Ruff).
- Use `snake_case` for functions/variables and `PascalCase` for classes.
- Lint with `ruff` (`tool.ruff` in `pyproject.toml`).
- Keep module boundaries aligned with `src/` subpackages.

## Testing Guidelines
- Framework: `pytest` with `tests/` as root.
- File naming: `test_*.py` and `Test*` classes.
- Prefer adding focused unit tests near the module under test.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, sentence case (e.g., "Add type checking").
- PRs should include: concise description, test command run/results, and any config changes.
- If output data changes (e.g., `data/runs/`), call it out and avoid committing large artifacts.

## Configuration & Data Tips
- Training configs live in `config/training/` and abstractions in `config/abstractions/`.
- For reproducibility, document the config and CLI flags used for any reported results.
