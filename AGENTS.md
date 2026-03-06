# Repository Guidelines

## Layout & Architecture
`src/` is layered — `interfaces/` (cli, api, chart) → `pipeline/` (training,
evaluation, abstraction) → `engine/` (solver, search) → `core/` (game,
actions), with `shared/` importable by all layers. The layering is
hard-enforced by import-linter (`.importlinter`, run via pre-commit).

Tests in `tests/` mirror this layout. Config YAML lives under `config/`
(source of truth for training setups; name new files for their purpose).
Runtime artifacts go in `data/` (`runs/`, `profiles/`, `combo_abstraction/`,
`eval_ledger.jsonl`); avoid committing large training outputs. The React
frontend lives in `ui/` (`npm run dev` / `npm run build` from `ui/`).

## Commands
- `uv sync --group dev` — install dependencies.
- `uv run poker-solver` — interactive CLI.
- `uv run poker-solver-run` — headless entrypoint: `train`, `evaluate`,
  `ledger`, `compare`, `checkpoint-profile`. Use `ledger`/`compare` for eval
  bookkeeping instead of hand-transcribing scores.
- `modal_app.py` — cloud training/eval on Modal (`uv run modal run
  modal_app.py`); mirrors the local layout with `data/` on a Volume.
- `uv run pytest -m "not slow"` — fast gate; `uv run pytest` — full suite.
- `uv run pre-commit run --all-files` — full quality gate (ruff lint+format,
  ty, import-linter, deptry, vulture). Run before handing off changes.

## Code Style
Python 3.12+. Ruff enforces formatting and import sorting — don't hand-police
style. What is *not* enforced by tooling:
- Prefer explicit, typed interfaces between solver, training, and evaluation
  layers.
- Do not assume backward compatibility is required. Unless explicitly
  requested, prefer clean breaks over compatibility shims, aliases, or legacy
  import paths.
- Imports at the top of the file; avoid importing inside functions unless
  absolutely necessary.
- This is a research-grade project: call out anything that does not meet that
  bar — bugs, correctness risks, or inelegant code that can be optimized.

## Testing
- While developing, run focused tests: `uv run pytest tests/<path>::<test>`.
- Before handoff, run the fast gate (`-m "not slow"`). Run the full suite when
  a change touches training, abstraction/bucketing, evaluator logic, config
  loading, or shared infrastructure.
- Default timeout is 5s (pytest-timeout). Mark expensive tests
  `@pytest.mark.slow`; intentionally longer ones get an explicit tight
  `@pytest.mark.timeout(<seconds>)`.
- Keep tests deterministic (fixed seeds, no nondeterministic assertions) and
  fast; when one turns slow, check `--durations` (setup vs call time are
  separate) and optimize or reclassify.

## Commits
Short, imperative messages; a Conventional prefix like `feat:` is fine if used
consistently. Call out new files added under `config/` or `data/`.
