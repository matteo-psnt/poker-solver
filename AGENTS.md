# Repository Guidelines

A heads-up No-Limit Hold'em CFR solver. Training, evaluation and abstraction run
on Azure Batch; the laptop is for tests and dispatch.

## Commands

- `uv sync --group dev` — install dependencies.
- `uv run poker-solver <cmd>` — the single entrypoint. `--help` lists all of
  them, grouped: open a surface · dispatch and account for work · run on a node
  · read the record.
- `uv run pytest -m "not slow"` — fast gate. `uv run pytest` — full suite.
  Both carry `-n auto` via `addopts`; add `-n0` when debugging ONE test.
- `uv run pre-commit run --all-files` — full quality gate (ruff lint+format, ty,
  import-linter, deptry, vulture). **Green before any merge.**
- `just` is Terraform lifecycle + `panic` + `credit-check` + a few aliases.
  Anything needing an unaliased flag goes through `uv run poker-solver <cmd>`.

## Architecture

`src/` is five packages. `interfaces/` → `pipeline/` → `engine/` → `core/` is
a strict layering; `adapters/` (Postgres) is reachable from `interfaces/` only
and may not import the work it stores; `shared/` is importable by all.
**import-linter enforces this with eleven contracts, and `.importlinter` is
the documentation** — read it before moving anything.

Subtree detail lives in `.claude/rules/`, path-scoped so it loads with a
matching file: `commands.md`, `console.md`, `cloud.md`,
`solver-and-evaluation.md`.

**Two surfaces, split by who is asking.** `poker-solver` is the scriptable one —
what a cloud job, a shell and an agent drive; the web console is the one a human
reads. New capability goes in a command; the console gains it by calling one.

**There is no `data/` directory, and nothing recreates one.** Runs live on the
share and nowhere else. Regenerable caches go under `$POKER_SOLVER_CACHE`, else
`$XDG_CACHE_HOME`, else `~/.cache/poker-solver`.

Config YAML under `config/` is the source of truth for training setups. Tests
in `tests/` mirror `src/`.

## How work happens

**Every change starts in a worktree.** Several sessions run against this repo
at once and `main` is the only shared thing. Follow the `worktree` skill:
branch from local `HEAD`, link skills and both tfstates, run the fast gate
before editing so a later failure is yours. Edit the primary checkout only
when told to.

The loop is **try → prove → clean → merge**. A session's job is to close it.

1. **Try.** Build the experiment or improvement on the branch. Rough is fine
   here; nothing on a branch needs to be pretty.
2. **Prove.** A change merges on evidence. A solver, abstraction or eval change
   needs a measured number against a control in the same knob tier, from a pool
   job. A refactor needs the full gate green and unchanged outputs — golden
   numbers, or a bit-identical claim actually checked. Unproven means the
   branch stays a branch; say what was measured and what was not.
3. **Clean.** The user does not read the code, so the branch is the review.
   Before merging, remove what the experiment needed and the result does not:
   probes, one-off flags, scratch knobs, dead branches, anything the change made
   unused. Then take the refactor the change made obvious — decouple what got
   tangled, name what got repeated, add an abstraction where one now pays for
   itself. A negative result leaves nothing behind except a memory entry.
4. **Merge.** Re-check `main` moved, rebase, `git merge --ff-only` from the
   primary checkout, then remove the worktree and branch.

## Deciding and asking

Stopping to ask is the expensive failure here, not a wrong routine call.

- **Decide and go** when the direction was already discussed, when the code,
  git history or memory answers it, or when the choice is cheap to reverse.
  State what you picked in the report instead of asking.
- **Ask only** when the readings lead to materially different work and nothing
  above settles it, or before a destructive action or a multi-hour dispatch
  (give the ETA).
- **A question carries its own context.** The user runs several sessions and
  will not remember this one. Give what you were doing, what you found, the
  options with a recommendation, and what each costs. Someone who has read
  nothing else must be able to answer it.
- Finish everything that does not depend on the answer first.

## Reporting

Every hand-off, mid-task or final, is written for a reader who saw none of it.
The user runs several sessions and will not remember this one.

- **Outcome first.** Merged, parked on a branch, or blocked — and on what.
- **What changed**, in words: behaviour and files, next to what was *not*
  verified.
- **What is left**: leftovers still on the branch, follow-ups, the worktree.
- **Decisions made** on the user's behalf, one line each.
- **Whether the session can end.** Say so when the loop has closed — merged,
  worktree removed, memory written — or when nothing more can happen without
  the user. Otherwise keep working: a long context or a finished sub-step is
  not a reason to stop.

**A number never stands alone.** Every figure says what it measures, its unit,
and what it is compared against — the control, the previous rung, the target —
so the reader knows what to conclude. "940 mbb" tells them nothing; "940 mbb
exploitable on the gate, 35% below PCS and still 1.9x the 500 target" does.
The same for counts, durations and sizes.

## Cost & where work runs

**Money is not a constraint.** If a bigger box, more nodes, a longer task, more
seeds or more evaluation deals would land a result sooner or with tighter error
bars, propose that. Never trade statistical power or wall-clock for dollars.

**Real work runs in the cloud, not on the laptop.** No training, precompute or
evaluation locally, not even a shrunken version — a short one-node pool job is
the fast signal. The laptop is for tests, the gate, one-off scripts and probes
that finish in seconds to a few minutes; use judgment on anything in between.
When the cloud path is awkward, **fix the infra instead of routing around it
locally** — `src/interfaces/cloud/` and `infra/` are in scope.

**Wall-clock is the scarce thing.** Probe short before committing long.

## Code style

Python 3.13+ everywhere, node included. `src/shared/cloudtask/` and its closure
are **stdlib only**: the node imports them before `uv sync`. Ruff enforces
formatting and import sorting. What tooling does not enforce:

- **No backward compatibility.** Clean breaks over shims, aliases and legacy
  import paths. Delete code that nothing calls, and the tests of it.
- **Refactor when a change shows the need**: split what it coupled, extract
  what it repeated. Do not add an abstraction for one caller, or for a caller
  that does not exist yet. Simpler wins.
- **A docstring says why the code has this shape, not what used to be there.**
  Deleted alternatives and the bug behind a fix belong in the commit message.
  Three lines is the budget; never restate a signature `ty` already checks.
- **A string that is not the first statement of a scope is not a docstring.**
  A non-obvious attribute gets a one-line comment after the field.
- **A guard test pins a MEASURED failure, not a filing convention.**
- This is a research-grade project: call out anything that does not meet that
  bar — bugs, correctness risks, or code that can be simplified.

## Commits

Short, imperative messages with a Conventional prefix (`feat:`, `fix:`,
`perf:`). Sessions share the primary checkout: stage by path (a hook refuses
`git add -A`), leave other sessions' dirty files alone, and confirm each
commit with `git show --stat HEAD` — a file can change under you between
edit and stage.
