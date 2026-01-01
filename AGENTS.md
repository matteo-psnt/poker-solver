# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the Python package, with core modules in `solver/`, `training/`,
`evaluation/`, `game/`, `bucketing/`, `actions/`, `utils/`, and `cli/`. Tests
live in `tests/` and generally mirror `src/` by feature area, with integration
coverage in `tests/integration/`. Configuration is stored in YAML under
`config/` (e.g., `config/training/quick_test.yaml`). Generated artifacts and
runtime outputs go in `data/` (`runs/`, `profiles/`, `combo_abstraction/`). The
frontend lives in `ui/` with React sources in `ui/src` and builds in `ui/dist`.

## Build, Test, and Development Commands
- `uv sync --group dev` installs Python dependencies from `uv.lock`.
- `uv run poker-solver` launches the CLI (or `poker-solver` if installed into
  your environment).
- `uv run pytest` runs the test suite.
- `uv run pytest -m "not slow"` runs the fast gate (default during iteration).
- `uv run pytest --durations=40 --durations-min=0.1 -q` prints the slowest
  tests to guide optimization and marker cleanup.
- `uv run ruff check .` runs linting and import sorting checks; `uv run ruff format .`
  applies formatting.
- `uv run ty check` runs static type checks.
- `cd ui && npm install` installs UI dependencies.
- `cd ui && npm run dev` starts the Vite dev server.
- `cd ui && npm run build` creates a production UI build.

## Coding Style & Naming Conventions
Python code targets 3.10+ with 4-space indentation and a 100-character line
limit enforced by Ruff. Use `snake_case` for functions/variables, `PascalCase`
for classes, and `UPPER_SNAKE_CASE` for constants. Keep modules focused and
prefer explicit, typed interfaces between solver, training, and evaluation
layers.
Imports must be placed at the top of the file unless absolutely necessary;
avoid importing inside functions.
This is a research-grade project; call out anything that does not meet that
bar, including bugs, correctness risks, or unelegant code that can be
optimized.

## Testing Guidelines
Tests are written with pytest and should follow the configured patterns:
`test_*.py`, `Test*` classes, and `test_*` functions.
Use this workflow:
- While developing, run focused tests first: `uv run pytest tests/<path>::<test_name>`.
- Before handing off changes, run the fast gate: `uv run pytest -m "not slow"`.
- Run full suite (`uv run pytest`) when your change impacts training, bucketing,
  evaluator logic, config loading, or shared infrastructure.

Marker and timeout policy:
- Mark expensive tests with `@pytest.mark.slow`.
- The default timeout is 5 seconds (configured in `pyproject.toml` via
  `pytest-timeout`).
- If a test is intentionally longer, add an explicit per-test timeout with
  `@pytest.mark.timeout(<seconds>)` and keep it as tight as possible.

Performance and determinism expectations:
- Keep tests deterministic (fixed seeds, no nondeterministic assertions).
- Avoid long loops and oversized fixtures in non-slow tests.
- When a test becomes unexpectedly slow, capture timings with
  `--durations` and either optimize or reclassify it with `@pytest.mark.slow`.
- Read duration output carefully: setup time and call time are reported
  separately.

## Commit Guidelines
Commit messages are short and imperative (sentence case is common); a
Conventional prefix like `feat:` is acceptable if used consistently. Call out
any new config files or data artifacts added under `config/` or `data/`.

## Configuration & Data Notes
Treat `config/` as the source of truth for training setups; keep new YAML files
named for their purpose (e.g., `full_training.yaml`). Avoid committing large
training outputs under `data/` unless explicitly required for reproducibility.
