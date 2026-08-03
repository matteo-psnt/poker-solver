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
- `uv run poker-solver` — interactive CLI. **It is a cloud client**: its
  Train/Score/Precompute items build the same `LegSpec` the headless commands
  build and submit it to the pool. There is no local-compute door in the menu,
  and nothing in it reads a card abstraction — submitting used to call
  `build_card_abstraction` first, which loads ~773 MB and answered the question
  about the wrong machine (the node mounts the share; the laptop's copy is
  irrelevant). It also has no config *editor*: a leg carries a config name plus
  `LegSpec.sets`, so overrides go through `--set k=v` and nothing else.
- `uv run poker-solver-run` — the single entrypoint, in three groups:
  - **dispatch to the pool** — `ab`, `submit`, `score`, `submit-precompute`,
    `jobs`, `logs`, `legs`, `cancel`, `pool-status`, `autoscale-check`,
    `repair-ladder`, `fetch`, `push-code`, `push-data`
  - **run here** (what a node invokes) — `train-static`, `precompute`,
    `evaluate`
  - **read the record** — `ledger`, `curve`, `progress`, `runinfo`, `report`,
    `compare`, `promote`. Every one takes `--source local|share`: `local` reads
    the copy `fetch` left in `--runs-dir`, `share` answers against the published
    record without keeping one — the eval index is rebuilt from the per-run
    documents rather than read from a second writable file on a share that has
    no atomic append.

  A subcommand is one module under `src/interfaces/cli/commands/`, listed once
  in the `COMMANDS` tuple. The `Command` dataclass carries parser, handler AND
  renderer together on purpose: when those lived apart, `checkpoint-profile`
  borrowed evaluate's renderer and died on a missing key.
- **Azure dispatch is Python, in `src/interfaces/cloud/`** — `spec.py` (pure,
  the testable core: what a leg IS), `batch.py`, `share.py`, `dispatch.py`,
  `config.py`, `workspace.py` (what `--source share` materialises). It lives
  under `interfaces` so nothing in
  `pipeline`/`engine`/`core` can reach Azure. **Auth is `AzureCliCredential`,
  never `DefaultAzureCredential`** — the default chain probes the link-local
  IMDS address, which on a laptop hangs rather than refusing (measured: >120s
  vs 1.3s).
- `just` is now **Terraform lifecycle + `panic` + thin aliases**, ~175 lines.
  `just panic <rg> <account> <pool>` is the one recipe that deliberately avoids
  the Python CLI and reads no Terraform state, so it works from a phone in Azure
  Cloud Shell.
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
- **`fetch` defaults to metadata only.** Every analysis command reads nothing
  but small JSON, so the default pulls `*.json`/`*.jsonl` and leaves the ~540 MB
  zarr checkpoints on the share. `--full` / `--run <id> --full` pull checkpoint
  data, and both obey the manifest: only what `CHECKPOINT.json` names is fetched,
  because a killed task leaves partially-copied snapshot directories behind and
  an unnamed one is unfinished by construction.
- `infra/` — **fire-and-forget cloud training on Azure Batch**.
  `poker-solver-run submit --config <c> --to <absolute-iteration>` queues a leg
  and returns; the pool scales 0→N→0 on its own. Terraform owns the account and
  pool; jobs and tasks are created at runtime by `src/interfaces/cloud/`, never
  in HCL. `infra/run_leg.sh` stays shell on purpose — it does disk discovery,
  mount handling and publish-on-exit traps *around* the Python process, and it
  reads the leg's `RUN_*` environment (overrides arrive as `RUN_SETS_JSON`, a
  JSON array, decoded with `read -d ''` so a value containing `=`, a space or a
  newline survives). `infra/store/` is a **separate**
  Terraform state holding the durable share, so `just destroy` cannot reach the
  experiment record. Active runs live on the node's `/mnt/work` data disk and are
  *published* to the share — never point `runs_dir` at the share. Constraints that
  look arbitrary but are measured (UserSubscription mode, `Dals_v6` not
  `Dalds_v6`, Gen2-only images, the SKU policy) are documented in
  `infra/README.md`; read it before changing pool config.
  **`just legs` is how you find out why a leg died** — the run log cannot
  record a death (the container is gone first), so `run_leg.sh` writes its own
  account to `<share>/legs/` and `legs` reconciles the ones whose trap never ran
  against Batch's view.
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
