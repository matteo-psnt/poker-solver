# Repository Guidelines

## Layout & Architecture
`src/` is layered — `interfaces/` (cli) → `pipeline/` (training, evaluation,
abstraction) → `engine/` (solver, search) → `core/` (game, actions), with
`shared/` importable by all layers. The layering is hard-enforced by
import-linter (`.importlinter`, run via pre-commit).

**One solver backend: the statically-enumerated tree.** An infoset is
`(node_id, bucket)` — an index into a table allocated once at full size, so
memory is flat in iteration count. The old dynamic backend (hashed
`InfoSetKey`, discovered as it went) is gone, and with it every checkpoint it
wrote: those are unreadable at HEAD by design, not by accident. Runs are
identified as loadable by `STATIC_CHECKPOINT.json`.

Tests in `tests/` mirror this layout. Config YAML lives under `config/`
(source of truth for training setups; name new files for their purpose).
Runtime artifacts go in `data/` (`runs/`, `combo_abstraction/`,
`eval_ledger.jsonl`); avoid committing large training outputs.

## Commands
- `uv sync --group dev` — install dependencies.
- `uv run poker-solver` — interactive CLI.
- `uv run poker-solver-run` — headless entrypoint: `train-static`,
  `precompute`, `evaluate`, `curve`, `report`, `promote`, `ledger`, `compare`,
  `checkpoint-profile`. Every long-running operation is reachable here, so cloud
  jobs are shell invocations of this module rather than provider-specific
  reimplementations.
- **`train-static` covers both starting and continuing a run.** `--iterations`
  is an ABSOLUTE target and `--run <id>` continues an existing directory, so
  re-running past the target is a no-op. That is what makes a scheduler retry
  converge instead of training twice; there is no separate `resume`.
- **Experiment bookkeeping.** Tag runs with `--experiment`/`--arm`/`--parent`
  (`--set k=v` for config overrides); `report --experiment` pins every arm to the
  control's knob tier and pairs each variant against its control. `curve --run`
  is the within-run exploitability-vs-iteration artifact. `promote` moves the
  baseline. Never hand-transcribe scores — and never compare arms across knob
  tiers, which `compare`/`report` refuse by design.
- **Eval records are per-run files**, not ledger appends: `evaluate` writes the
  complete row into `<run_dir>/evals/record-*.json`, and `data/eval_ledger.jsonl`
  is a rebuildable cache (`ledger --rebuild`). This is what makes concurrent
  evaluation from several boxes safe.
- `infra/` — **fire-and-forget cloud training on Azure Batch**. `just submit
  <config> <absolute-iteration> [experiment] [arm] [k=v...]` queues a leg and
  returns; the pool scales 0→N→0 on its own. `just jobs` / `pool-status` /
  `fetch`. Terraform owns the account and pool; jobs and tasks are created at
  runtime by the justfile, never in HCL. `infra/store/` is a **separate**
  Terraform state holding the durable share, so `just destroy` cannot reach the
  experiment record. Active runs live on the node's `/mnt/work` data disk and are
  *published* to the share — never point `runs_dir` at the share. Constraints that
  look arbitrary but are measured (UserSubscription mode, `Dals_v6` not
  `Dalds_v6`, Gen2-only images, the SKU policy) are documented in
  `infra/README.md`; read it before changing pool config.
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
