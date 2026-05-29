# Repository Guidelines

A heads-up No-Limit Hold'em CFR solver. Training, evaluation and abstraction run
on Azure Batch; the laptop is for tests and dispatch.

## Commands

- `uv sync --group dev` — install dependencies.
- `uv run poker-solver <cmd>` — the single entrypoint. `--help` lists all of
  them, grouped: open a surface · dispatch and account for work · run on a node
  · read the record.
- `uv run pytest -m "not slow"` — fast gate. `uv run pytest` — full suite.
  Both carry `-n auto` via `addopts`; add `-n0` when debugging ONE test, where
  worker startup and interleaved output are pure cost.
- `uv run pre-commit run --all-files` — full quality gate (ruff lint+format, ty,
  import-linter, deptry, vulture). **Run before handing off changes.**
- `just` is Terraform lifecycle + `panic` + `credit-check` + a few aliases.
  Anything needing an unaliased flag goes through `uv run poker-solver <cmd>` or
  `just cli <cmd> [flags...]`.

## Architecture

`src/` is layered: `interfaces/` → `pipeline/` → `engine/` → `core/`, with
`shared/` importable by all. **The layering is hard-enforced by import-linter,
and `.importlinter` is the documentation** — nine contracts, four of which state
a rule the directory alone does not. Read it before moving anything.

Detail that only matters inside one subtree lives in `.claude/rules/`, scoped by
path so it loads when you open a matching file: `commands.md`, `console.md`,
`cloud.md`, `solver-and-evaluation.md`.

**There are TWO surfaces, split by who is asking.** `poker-solver` is the
scriptable one — what a cloud job, a shell, and an AI agent drive. The web
console is the one a human reads. The console reimplements nothing: **it may
COMPOSE command payloads, and it may not COMPUTE one.** New capability goes in a
command; the console gains it by calling one.

**There is no `data/` directory, and nothing recreates one.** Runs live on the
share and nowhere else. Regenerable caches live under `$POKER_SOLVER_CACHE`,
else `$XDG_CACHE_HOME`, else `~/.cache/poker-solver`;
`tests/shared/test_cache.py` fails if any module names a `data/cache` path
again. Anything under `data/` is a cache by definition: never a source of truth,
never worth backing up.

Config YAML lives under `config/` and is the source of truth for training
setups; name new files for their purpose. Tests in `tests/` mirror `src/`.

## Cost & where work runs

Compute budget is effectively unbounded — **money is not a constraint here.** If
a bigger box, more nodes, a longer task, more seeds, or more evaluation deals
would make a result arrive sooner or land with tighter error bars, propose that;
don't quietly pick the cheap option. Never trade statistical power or wall-clock
for dollars, and don't cite a cost estimate as a reason against something.

**Real work runs in the cloud, not on the laptop.** Local is for tests, the
pre-commit gate, and seconds-long probes — no training, precompute, or
evaluation, not even a shrunken "sanity" version. For a fast signal, run a short
one-node job on the pool.

That makes the cloud path the critical path: when it is awkward, **fix the infra
instead of routing around it locally.** Work on `src/interfaces/cloud/` and
`infra/` is in scope by default.

**Wall-clock is the scarce thing.** Probe short before committing long, and give
an ETA plus a go-ahead check before anything multi-hour.

## Code style

Python 3.13+ (ruff `target-version = "py313"`). The one exception is
`src/shared/cloudtask/`, which the node imports on the pinned image's system
`python3` — 3.10 — before `uv sync`; two tests enforce that floor.

Ruff enforces formatting and import sorting — don't hand-police style. What is
*not* enforced by tooling:

- Imports at the top of the file; avoid importing inside functions unless
  absolutely necessary.
- **Do not assume backward compatibility is required.** Unless explicitly
  requested, prefer clean breaks over compatibility shims, aliases, or legacy
  import paths.
- **A docstring says why the code has this shape, not what used to be there.**
  Deleted alternatives, the bug that prompted a fix, and what a module replaces
  belong in the commit message — not at the top of a file every reader pays for.
  Prose rots silently and nothing fails: `kinds.py` asserted for weeks that two
  task kinds no longer existed while four of their commands stayed registered.
- **A guard test pins a MEASURED failure, not a filing convention.** Read cost,
  the node's stdlib floor, contract staleness and the golden numbers each earn
  their place because they broke something expensive. A registry with no
  violations in it is documentation wearing a tripwire; write the documentation.
- This is a research-grade project: call out anything that does not meet that
  bar — bugs, correctness risks, or inelegant code that can be optimized.

## Commits

Short, imperative messages; a Conventional prefix like `feat:` is fine if used
consistently. Call out new files added under `config/`.
