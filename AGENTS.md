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
**There is no `data/` directory, and nothing recreates one.** Runs live on the
share and nowhere else; the regenerable caches moved OUT of the working tree to
`src/shared/cache.py`'s root (`$POKER_SOLVER_CACHE`, else `$XDG_CACHE_HOME`,
else `~/.cache/poker-solver`), and `tests/shared/test_cache.py` fails if any
module names a `data/cache` path again — which is how the directory came back
after each of the two previous prunes. The node wrapper sets
`POKER_SOLVER_CACHE=/mnt/work/cache`, because a Batch task's `HOME` is its own
working directory and the default would re-canonicalise the river's 2.6M boards
(~1 min) on every leg. `$CODE/data` is still symlinked on a node — that is
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
- `uv run poker-solver` — interactive CLI. **It is a cloud client**: its
  Train/Score/Precompute items build the same `LegSpec` the headless commands
  build and submit it to the pool. There is no local-compute door in the menu,
  and nothing in it reads a card abstraction — submitting used to call
  `build_card_abstraction` first, which loads ~773 MB and answered the question
  about the wrong machine (the node mounts the share; the laptop's copy is
  irrelevant). It also has no config *editor*: a leg carries a config name plus
  `LegSpec.sets`, so overrides go through `--set k=v` and nothing else.
- `uv run poker-solver-run` — the single entrypoint, in three groups:
  - **see and dispatch** — `status`, `submit`, `score`, `submit-precompute`,
    `jobs`, `logs`, `legs`, `cancel`, `pool-status`, `autoscale-check`,
    `repair-ladder`, `push-code`, `push-data`.
    **`status` is the one screen for "what is the pool doing right now"** —
    it composes `pool-status` + `jobs` + `legs` through `invoke()` and renders
    each with the command that owns it. Panels are fetched CONCURRENTLY and
    fail INDEPENDENTLY. **Read cost is a maintained property, not an accident**
    — `tests/interfaces/cloud/test_read_cost.py` pins it as call counts, since
    latency is invisible in a test and enormous in practice. Three rules:
    never list tasks for a job you will discard (`jobs` fetched all 44 to render
    2); issue independent round trips together (47 leg records serially was
    9.1s); and never sync `keys-*` key tables, which are the deleted dynamic
    backend's and were 37 of the 38 MB a share read pulled. Measured warm
    before → after: `jobs` 11s → 2.5s, `legs` 23s → 2.0s, `ledger` 20s → 4.5s,
    the whole status screen 22s → 2.0s. The interactive menu's "Cloud Status"
    is the same call; it used to be a second renderer that could disagree with
    `jobs` and could not see the leg log at all.
  - **run on a node** — `train-static`, `precompute`, `evaluate`. These are
    invoked BY the node wrapper, on whichever box executes the leg; not a
    local-compute door, and they keep `--runs-dir` because a node writes to
    `/mnt/work` before publishing.
  - **read the record** — `ledger`, `curve`, `progress`, `runinfo`, `report`,
    `compare`, `promote`. **There is no `--source` and no `--runs-dir`: every
    reader answers against the published record**, materialised into a temp
    tree and discarded. Nothing on a laptop is a source of truth about a run,
    so a local copy could only be a stale second answer. The eval index is
    REBUILT from the per-run documents on every read rather than stored — which
    is why `ledger --rebuild` is gone (every read is a rebuild) and
    `--migrate` is gone (it rewrites in place, and the tree is a throwaway).

  A subcommand is one module under `src/interfaces/cli/commands/`, listed once
  in the `COMMANDS` tuple. The `Command` dataclass carries parser, handler AND
  renderer together on purpose: when those lived apart, `checkpoint-profile`
  borrowed evaluate's renderer and died on a missing key.
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
  payload is the interface. Still unwrapped: the Azure SDK's
  `ClientAuthenticationError`/`HttpResponseError`, which have no chokepoint in
  `batch.py`, so a surface talking to Batch catches those by name too.
- **Azure dispatch is Python, in `src/interfaces/cloud/`** — `spec.py` (pure,
  the testable core: what a leg IS), `batch.py`, `share.py`, `dispatch.py`,
  `config.py`, `workspace.py` (what `--source share` materialises). It lives
  under `interfaces` so nothing in
  `pipeline`/`engine`/`core` can reach Azure. **Auth is `AzureCliCredential`,
  never `DefaultAzureCredential`** — the default chain probes the link-local
  IMDS address, which on a laptop hangs rather than refusing (measured: >120s
  vs 1.3s).
- `just` is **Terraform lifecycle + `panic` + `credit-check` + a few aliases**.
  `just panic <rg> <account> <pool>` is the one recipe that deliberately avoids
  the Python CLI and reads no Terraform state, so it works from a phone in Azure
  Cloud Shell. The aliases are passthroughs and nothing else — a guard test
  (`tests/interfaces/cli/test_justfile_aliases.py`) fails if one names a
  subcommand that does not exist, which is how a `just fetch` recipe outlived
  the command it called by weeks. Anything needing a flag that is not aliased
  goes through `uv run poker-solver-run <cmd>`, or `just cli <cmd> [flags...]`.
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
  `poker-solver-run submit --config <c> --to <absolute-iteration>` queues a leg
  and returns; the pool scales 0→N→0 on its own. Terraform owns the account and
  pool; jobs and tasks are created at runtime by `src/interfaces/cloud/`, never
  in HCL. **The node-side wrapper is Python, in `src/shared/node/`** —
  `archive.py` (publish/fetch between the node disk and the SMB share),
  `plan.py` (the `RUN_*` environment → the argv), `runner.py` (the timeout
  guard, the tee, the mid-run publisher, the exit accounting), with
  `infra/run_leg.py` as the entry point the task command line names. It lives
  under `shared` because it runs on the node BEFORE `uv sync`, under the node's
  system `python3` — **3.10 on the pinned Ubuntu 22.04 image, and stdlib
  only**. Both constraints are enforced by
  `tests/shared/node/test_node_interpreter.py`, which imports the whole package
  on a real 3.10 via `uv run --python 3.10 --no-project`; ruff's
  `target-version` is `py312`, so `pyproject.toml` disables `UP017` for these
  files or the formatter rewrites them into something the node cannot import.
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
  **`poker-solver-run legs` is how you find out why a leg died** — the run log cannot
  record a death (the container is gone first), so the wrapper writes its own
  account to `<share>/legs/` and `legs` reconciles the ones whose exit record
  never landed against Batch's view. 124 (the guard's deadline — a hang) and
  137 (SIGKILL from outside — the OOM killer) are DIFFERENT causes, and a wrong
  terminal one is permanent: it suppresses reconciliation.
- `uv run pytest -m "not slow"` — fast gate; `uv run pytest` — full suite.
- `uv run pre-commit run --all-files` — full quality gate (ruff lint+format,
  ty, import-linter, deptry, vulture). Run before handing off changes.

## Cost & Where Work Runs
Compute budget is effectively unbounded — **money is not a constraint here.**
If a bigger box, more nodes, a longer leg, more seeds, or more evaluation deals
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
