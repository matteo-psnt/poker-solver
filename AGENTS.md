# Repository Guidelines

## Layout & Architecture
`src/` is layered — `interfaces/` (commands, plus the cli and web surfaces that
read through them) → `pipeline/` (training, evaluation,
abstraction) → `engine/` (solver, search) → `core/` (game, actions), with
`shared/` importable by all layers. The layering is hard-enforced by
import-linter (`.importlinter`, run via pre-commit) — **nine contracts, and they
are the documentation**: read them before moving anything, because four state a
rule the directory alone does not. `reference/` may not import the estimators it
validates (or the 1e-9 agreement is circular); Azure dispatch may not reach into
the solver; an estimator never imports the ledger; the play server never trains.

**Structure that import-linter cannot see is pinned in `tests/test_layout.py`.**
Direction is an edge and is enforced; "are these peers filed the same way" and
"does this name mean one thing" are not, and both drifted. They are declared
registries in the `records.REGISTRY` idiom — a new loose module beside
sub-packages, or a new duplicated basename, fails until someone declares it
*with the reason*. Adding to the list is fine; doing it without deciding is
what the failure message argues against. The same file holds two more guards:
`tests/` must mirror the src layout (a test whose src-imports all land in one
sub-package belongs in it), and every `src.`/`tests.` dotted path spelled
inside a STRING — monkeypatch targets, subprocess programs, docstring
citations — must resolve against the tree, because an import fails loudly
when a module moves and a string just goes stale. Peers are grouped: the four estimators
in `evaluation/estimators/`, infoset identity in `solver/infoset/`, strategy
lookup in `solver/policy/`. A service is named for what it DOES
(`services.scoring`, `services.bucketing`), so no one word names both a service
and the subsystem beneath it.

**One solver backend: the statically-enumerated tree.** An infoset is
`(node_id, bucket)` — an index into a table allocated once at full size, so
memory is flat in iteration count. The old dynamic backend (hashed
`InfoSetKey`, discovered as it went) is gone, and with it every checkpoint it
wrote: those are unreadable at HEAD by design, not by accident. Runs are
identified as loadable by `STATIC_CHECKPOINT.json`.

Tests in `tests/` mirror this layout. Config YAML lives under `config/`
(source of truth for training setups; name new files for their purpose).
**There is no `data/` directory, and nothing recreates one.** Runs live on the
share and nowhere else; the regenerable caches moved OUT of the working tree to
`src/shared/cache.py`'s root (`$POKER_SOLVER_CACHE`, else `$XDG_CACHE_HOME`,
else `~/.cache/poker-solver`), and `tests/shared/test_cache.py` fails if any
module names a `data/cache` path again — which is how the directory came back
after each of the two previous prunes. The node wrapper sets
`POKER_SOLVER_CACHE=/mnt/work/cache`, because a Batch task's `HOME` is its own
working directory and the default would re-canonicalise the river's 2.6M boards
(~1 min) on every task. `$CODE/data` is still symlinked on a node — that is
where `precompute` writes and where `runs_dir` resolves — but it is a runtime
path on `/mnt/work`, never a directory in the checkout. The 194 MB
`combo_abstraction` that survived earlier prunes was a fixture for exactly ONE
test, and while it was missing that test FAILED with a `FileNotFoundError`
pointing at `precompute` — an environment report dressed as a regression, on
an artifact that is gitignored, unversioned, and had already come back once
after being deleted. `tests/conftest.py::requires_card_abstraction` makes it a
skip that names the fix. Anything under `data/` is therefore a cache by
definition: never a source of truth, never worth backing up.

## Commands
- `uv sync --group dev` — install dependencies.
- **There are TWO surfaces, split by who is asking.** `poker-solver` is the
  scriptable one — what a cloud job, a shell, and an AI agent drive. The web
  console is the one a human reads. The console does not reimplement anything:
  every endpoint is a single `Command.invoke`, so the two cannot drift. The
  console is expected to GROW toward what the commands can do; the commands stay
  the complete surface, and anything the console gains it gains by calling one.
  **Coverage is complete and enforced.**
  `tests/interfaces/web/test_command_coverage.py` fails until a command has an
  endpoint or a declared reason: `NO_PAYLOAD` (`serve`, `blueprint-serve` never
  return one) or `NODE_ONLY` (`train-static`, `precompute`, `evaluate` are
  node compute). `status` is covered by the three panels it composes — do not
  add `/api/status`.
- **The console writes.** Seven dispatching endpoints (`submit`, `score`,
  `submit-precompute`, `push-code`, `push-data`, `compact-legs`, `promote`).
  One rule: an optional body field means *omitted*, and `web.app.given()` drops
  it so the command's own parser supplies the default — never re-declare a
  default in the request model. That is also what makes `compact-legs` default
  to the dry run. Guard flags (`--force`, `--delete`) are opt-in, never
  pre-checked. Writes use `TtlCache(0.0)`; a dispatch must not be memoised.
- **There is no interactive CLI.** `uv run poker-solver` — a questionary menu —
  was deleted (`src/interfaces/cli/{app,flows,ui}`, plus the `questionary`
  dependency). It had been hollowed out over time until every item was a wizard
  around flags `poker-solver` already accepted, and four abstraction-browsing
  items too specific to keep. Its one non-obvious rule survives and still binds
  anything that submits: **nothing on the laptop reads a card abstraction.**
  Submitting used to call `build_card_abstraction` first, loading ~773 MB to
  answer a question about the wrong machine — the node mounts the share, and the
  laptop's copy is irrelevant. Config overrides likewise go through `--set k=v`
  and nothing else: a task carries a config name plus `LegSpec.sets`.
- `uv run poker-solver` — the single entrypoint, in four groups, and the order
  of the `COMMANDS` tuple IS the grouping:
  - **open a surface** — `status`, `serve`, `blueprint-serve`, `serve-box`.
    One screen, or a server putting a surface on localhost: `serve` is the
    console, `blueprint-serve` is one trained run served for reading, and
    `serve-box` reports/wakes/stops the VM the latter runs on.
  - **dispatch and account for work** — `submit`, `score`, `submit-precompute`,
    `jobs`, `logs`, `tasks`, `cancel`, `pool-status`, `autoscale-check`,
    `push-code`, `push-data`, `compact-legs`.
    **`status` is the one screen for "what is the pool doing right now"** —
    it composes `pool-status` + `jobs` + `tasks` through `invoke()` and renders
    each with the command that owns it. Panels are fetched CONCURRENTLY and
    fail INDEPENDENTLY. **Read cost is a maintained property, not an accident**
    — `tests/interfaces/cloud/test_read_cost.py` pins it as call counts, since
    latency is invisible in a test and enormous in practice. Three rules:
    never list tasks for a job you will discard (`jobs` fetched all 44 to render
    2); issue independent round trips together (47 task records serially was
    9.1s); and never sync `keys-*` key tables, which are the deleted dynamic
    backend's and were 37 of the 38 MB a share read pulled. Measured warm
    before → after: `jobs` 11s → 2.5s, `tasks` 23s → 2.0s, `ledger` 20s → 4.5s,
    the whole status screen 22s → 2.0s. The interactive menu's "Cloud Status"
    is the same call; it used to be a second renderer that could disagree with
    `jobs` and could not see the task log at all.
  - **run on a node** — `train-static`, `precompute`, `evaluate`. These are
    invoked BY the node wrapper, on whichever box executes the task; not a
    local-compute door, and they keep `--runs-dir` because a node writes to
    `/mnt/work` before publishing.
  - **read the record** — `ledger`, `curve`, `cost`, `progress`, `runs`,
    `configs`, `runinfo`, `report`, `compare`, `promote`. `configs` lists the
    stems `submit`/`submit-precompute` accept; `runs` carries
    `experiment_id`/`arm`, since the listing is the only place the set of
    experiments exists; `activity` reads the local telemetry log (below).
    **There is no `--source` and no `--runs-dir`: every
    reader answers against the published record**, materialised into a temp
    tree and discarded. Nothing on a laptop is a source of truth about a run,
    so a local copy could only be a stale second answer. The eval index is
    REBUILT from the per-run documents on every read rather than stored — which
    is why `ledger --rebuild` is gone (every read is a rebuild) and
    `--migrate` is gone (it rewrites in place, and the tree is a throwaway).

  A subcommand is one module under `src/interfaces/commands/`, listed once
  in the `COMMANDS` tuple. **That package sits beside `cli/` and `web/`, not
  inside either** — `cli.headless` renders a command to a terminal and
  `web.app` invokes the same command and serves the payload, so a path under
  `cli/` named one owner for a seam with two callers. `cli/` now holds only
  `headless.py`: the genuinely terminal-specific half.
  The `Command` dataclass carries parser, handler AND
  renderer together on purpose: when those lived apart, `checkpoint-profile`
  borrowed evaluate's renderer and died on a missing key.
- **`Command.execute` is the seam both surfaces share, and it is observed.**
  Not `invoke` — the command line parses argv and calls the handler, so anything
  wrapped around `invoke` sees the console and misses the CLI. `execute` writes
  one row per invocation via `interfaces/telemetry.py`: command, surface,
  duration, outcome (`ok`/`refusal`/`error`), the exception's type name, and the
  arguments that differ from their defaults. `poker-solver activity` reads it
  (p50/p95/total, refusals apart from errors). It is **laptop-local and
  disposable** — under `$POKER_SOLVER_CACHE`, rotated at 8 MB — never the share:
  no atomic append, a per-document scheme would outgrow `legs/` in hours, and
  every write would add a round trip to the thing being measured. Writes are
  best-effort and must stay that way; `POKER_SOLVER_TELEMETRY=0` turns it off,
  which is what the test suite does.
- **A command is callable without a command line, and refusals are values.**
  `Command.invoke(**kwargs)` builds the arguments from the command's own parser
  — one declaration of what a command accepts, so a second surface cannot drift
  from it — and returns the payload unrendered. Anything the caller could have
  got right raises `CommandError` (`src/interfaces/errors.py`); a bug still
  tracebacks. Only `headless.py` turns a `CommandError` back into a message on
  stderr and exit 1, so the command line keeps its behaviour without imposing
  it: a surface that polls several commands greys out one panel instead of
  dying, which `raise SystemExit` at 16 sites made impossible. A guard test
  fails if a command module reintroduces it. `render()` is deliberately NOT
  abstracted — it is the terminal's renderer, and for any other surface the
  payload is the interface. **Which failures a surface may survive is decided
  once**, in `errors.attempt`: it runs a call and returns either the payload or
  a `Failure` classified `refusal` (understood, and the answer is no → 422) or
  `unavailable` (Azure did not answer → 503). The Azure SDK's
  `ClientAuthenticationError`/`HttpResponseError` have no chokepoint in
  `batch.py`, so they are caught THERE and nowhere else — `status` and the
  console carried the same three-arm ladder, and a third surface would have
  written it a third time. A guard test fails if anything under `interfaces/`
  names either exception again. The SDK imports lazily inside `attempt`,
  because `errors` is imported by every command and `azure.core.exceptions`
  costs 76ms against a 0.18s `--help`.
- **Both doors serialise through `shared.jsonio.dumps`** — `--json` and the
  console's `PayloadResponse` — so a payload carrying a numpy scalar or a
  `Path` cannot print fine on one surface and 500 the other, past the refusal
  ladder. They differ in one declared respect: the console keeps
  `allow_nan=False`, because `JSON.parse` rejects `NaN`.
- **Azure dispatch is Python, in `src/interfaces/cloud/`** — three sub-packages
  by what the code TALKS TO, not by what it is about:
  - `tasks/` — Batch. `spec.py` (pure, the testable core: what a task IS),
    `batch.py` (the SDK client), `dispatch.py` (queueing one).
  - `store/` — the SMB share. `share.py` (the file operations),
    `workspace.py` (materialising the published record a reader answers against).
  - `cost/` — what it all billed. `billing.py`, against Cost Management.

  Above them, because all three need them: `config.py` (credentials and resource
  ids) and `serve_box.py` (provisioning the play-server VM, which shares nothing
  with dispatching work or reading the bill). The package lives under
  `interfaces` so nothing in `pipeline`/`engine`/`core` can reach Azure.
  **Auth is `AzureCliCredential`, never `DefaultAzureCredential`** — the default
  chain probes the link-local IMDS address, which on a laptop hangs rather than
  refusing (measured: >120s vs 1.3s).
- `just` is **Terraform lifecycle + `panic` + `credit-check` + a few aliases**.
  `just panic <rg> <account> <pool>` is the one recipe that deliberately avoids
  the Python CLI and reads no Terraform state, so it works from a phone in Azure
  Cloud Shell. The aliases are passthroughs and nothing else — a guard test
  (`tests/interfaces/commands/test_justfile_aliases.py`) fails if one names a
  subcommand that does not exist, which is how a `just fetch` recipe outlived
  the command it called by weeks. Anything needing a flag that is not aliased
  goes through `uv run poker-solver <cmd>`, or `just cli <cmd> [flags...]`.
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
  complete row into `<run_dir>/evals/<slug>.json`. There is no stored index at
  all any more — `ledger` DERIVES one from the published documents on every
  read, which is what makes concurrent evaluation from several boxes safe.
  **`eval-*.json` and `record-*.json` are LEGACY** — two of the three
  pre-substrate shapes — and the rebuild skips both on purpose: a legacy record
  points at the old filename, so reading both enters one evaluation twice
  (measured: 63 rows became 110). A sparse `ledger` therefore means
  un-migrated legacy files on the share, NOT a broken rebuild.
- **There is no local run storage, and no local compute.** `data/runs`, the
  local eval ledger, `fetch` and the `ab` command are all deleted: `ab` trained
  on whatever box invoked it, which is the one thing this project does not do.
  Experiment arms go through `submit --experiment/--arm` and pair up under
  `report --experiment`. What a reader pulls is metadata only — `*.json`/
  `*.jsonl`, never `*.zarr` and never `keys-*` (the deleted dynamic backend's
  key tables, which were 37 of the 38 MB a share read used to fetch).
- `infra/` — **fire-and-forget cloud training on Azure Batch**.
  `poker-solver submit --config <c> --to <absolute-iteration>` queues a task
  and returns; the pool scales 0→N→0 on its own. Terraform owns the account and
  pool; jobs and tasks are created at runtime by `src/interfaces/cloud/`, never
  in HCL. **The cloud task is one subsystem, in `src/shared/cloudtask/`** —
  `kinds.py` (what KIND of work a task is, read from BOTH ends of the wire),
  `task_log.py` (WRITING the durable per-task account, on the node), and `node/`,
  the wrapper:
  `paths.py`, `process.py` (the timeout guard and the tee), `plan.py` (the
  `RUN_*` environment → the argv), `archive.py` (publish/fetch between the node
  disk and the SMB share), `progress.py` (the heartbeat and what a task
  achieved), `handlers.py` (what each kind DOES), `lifecycle.py` (one task start
  to finish, and the exit accounting), with `infra/run_task.py` as the entry
  point the task command line names. It lives under `shared` for two reasons:
  `pipeline` reads the task record, and `pipeline → interfaces` is forbidden; and
  the node runs it BEFORE `uv sync`, so **stdlib only** — a constraint enforced
  twice, by `tests/shared/cloudtask/node/test_node_interpreter.py` (which
  imports the node's whole closure on the interpreter `main.tf` installs) and by
  `tests/shared/cloudtask/test_imports.py` (fail-closed: nothing outside
  `records`/`jsonio`/`cache` may be reached).
  **READING that record is `src/shared/task_history.py`, deliberately outside
  this package** — the join, reconciliation against Batch, and what may be
  compacted. It runs only on a laptop, and the fail-closed guard above walks
  every file in `cloudtask/`, so keeping it there held ~260 lines to a stdlib
  floor they have no reason to meet. It cannot go under `interfaces` either,
  where most callers are: `pipeline.services.experiments` joins task rows into a
  run digest. The TABLE is neither — that is the terminal's renderer and lives
  on the `tasks` command, like every other `render()`.
  Two shell things REMAIN shell and should stay that way: `just panic` (must
  work from a phone in Cloud Shell, with no venv and no Terraform state) and
  `main.tf`'s `start_task` (disk discovery, `mkfs`, mount — it runs before any
  code snapshot exists on the node). `infra/store/` is a **separate**
  Terraform state holding the durable share, so `just destroy` cannot reach the
  experiment record. Active runs live on the node's `/mnt/work` data disk and are
  *published* to the share — never point `runs_dir` at the share. Constraints that
  look arbitrary but are measured (UserSubscription mode, `Dals_v6` not
  `Dalds_v6`, Gen2-only images, the SKU policy) are documented in
  `infra/README.md`; read it before changing pool config.
  **`poker-solver tasks` is how you find out why a task died** — the run log cannot
  record a death (the container is gone first), so the wrapper writes its own
  account to `<share>/legs/` and `tasks` reconciles the ones whose exit record
  never landed against Batch's view. 124 (the guard's deadline — a hang) and
  137 (SIGKILL from outside — the OOM killer) are DIFFERENT causes, and a wrong
  terminal one is permanent: it suppresses reconciliation.
- `uv run pytest -m "not slow"` — fast gate (25s); `uv run pytest` — full suite
  (39s). **Both run across 12 workers: `-n auto` is in `addopts`**, because 86%
  of the wall clock was 50 of the ~1460 tests and the rest were free (serial:
  2m46s / 1m27s). Add `-n0` to turn it off — worth doing when debugging ONE
  test, where worker startup and interleaved output are pure cost.
- `uv run pre-commit run --all-files` — full quality gate (ruff lint+format,
  ty, import-linter, deptry, vulture). Run before handing off changes.

## Cost & Where Work Runs
Compute budget is effectively unbounded — **money is not a constraint here.**
If a bigger box, more nodes, a longer task, more seeds, or more evaluation deals
would make a result arrive sooner or land with tighter error bars, propose that;
don't quietly pick the cheap option. Never trade statistical power or wall-clock
for dollars, and don't cite a cost estimate as if it were a reason against
something.

**Real work runs in the cloud, not on the laptop.** Local is for tests, the
pre-commit gate, and seconds-long probes — no training, precompute, or
evaluation, not even a shrunken "sanity" version. For a fast signal, run a short
one-node job on the pool.

That makes the cloud path the critical path: when it is awkward, **fix the infra
instead of routing around it locally.** Work on `src/interfaces/cloud/` and
`infra/` is in scope by default.

Still scarce is wall-clock: probe short before committing long, and give an ETA
plus a go-ahead check before anything multi-hour — a scheduling constraint, not
a budget one.

## Code Style
Python 3.13+ (`requires-python = ">=3.13"`, ruff `target-version = "py313"`).
The one exception is `src/shared/cloudtask/`, which the node imports on the
pinned image's system `python3` — 3.10 — before `uv sync`; two tests enforce
that floor. Ruff enforces formatting and import sorting — don't hand-police
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
- While developing, run focused tests: `uv run pytest -n0 tests/<path>::<test>`
  — `-n0` because `addopts` carries `-n auto`, and 12 workers for one test is
  slower to start and harder to read than running it in-process.
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
