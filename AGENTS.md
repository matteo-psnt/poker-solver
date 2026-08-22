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

Subtree detail lives in `.claude/rules/`, path-scoped so it loads with a
matching file: `commands.md`, `console.md`, `cloud.md`,
`solver-and-evaluation.md`.

**TWO surfaces, split by who is asking.** `poker-solver` is the scriptable one —
what a cloud job, a shell and an agent drive; the web console is the one a human
reads. New capability goes in a command; the console gains it by calling one.

**There is no `data/` directory, and nothing recreates one.** Runs live on the
share and nowhere else. Regenerable caches go under `$POKER_SOLVER_CACHE`, else
`$XDG_CACHE_HOME`, else `~/.cache/poker-solver` — never a source of truth,
never worth backing up.

Config YAML under `config/` is the source of truth for training setups; name new
files for their purpose. Tests in `tests/` mirror `src/`.

## Cost & where work runs

**Money is not a constraint.** If a bigger box, more nodes, a longer task, more
seeds or more evaluation deals would land a result sooner or with tighter error
bars, propose that. Never trade statistical power or wall-clock for dollars, and
never cite cost as a reason against something.

**Real work runs in the cloud, not on the laptop.** Local is for tests, the
pre-commit gate and seconds-long probes — no training, precompute or evaluation,
not even a shrunken "sanity" version. For a fast signal, run a short one-node job
on the pool.

That makes the cloud path the critical path: when it is awkward, **fix the infra
instead of routing around it locally** — `src/interfaces/cloud/` and `infra/` are
in scope by default.

**Wall-clock is the scarce thing.** Probe short before committing long, and give
an ETA plus a go-ahead check before anything multi-hour.

## Code style

Python 3.13+ (ruff `target-version = "py313"`) everywhere, node included — the
pool installs the interpreter, so there is no old-language floor to code around.
`src/shared/cloudtask/` and its closure are **stdlib only**: the node imports
them before `uv sync`. Ruff enforces formatting and import sorting — don't
hand-police style. What is *not* enforced by tooling:

- Imports at the top of the file; avoid importing inside a function.
- **Do not assume backward compatibility is required.** Prefer clean breaks over
  compatibility shims, aliases and legacy import paths.
- **A docstring says why the code has this shape, not what used to be there.**
  Deleted alternatives, the bug behind a fix and what a module replaces belong in
  the commit message. Prose rots silently and nothing fails.
- **Write the shortest docstring that leaves the reader able to act.** It earns
  its lines by carrying what the code cannot — an invariant a caller can violate,
  a unit, a sign convention, an array shape, a measurement that decides a knob.
  Never restate a signature `ty` already checks (`Args:`/`Returns:` blocks are
  noise) or a `.claude/rules/` bullet that loads anyway. **Three lines is the
  budget, and longer than the code below it is the smell.**
- **A string that is not the first statement of a scope is not a docstring** — it
  is discarded at import, read by nothing, and no ruff rule will tell you. The
  house style for a non-obvious attribute is a one-line comment after the field.
- **A guard test pins a MEASURED failure, not a filing convention.** A registry
  with no violations in it is documentation wearing a tripwire; write the
  documentation instead.
- This is a research-grade project: call out anything that does not meet that
  bar — bugs, correctness risks, or inelegant code that can be optimized.

## Commits

Short, imperative messages; a Conventional prefix like `feat:` is fine if used
consistently.
