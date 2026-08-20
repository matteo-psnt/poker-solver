---
paths:
  - "src/pipeline/**"
  - "src/engine/**"
  - "src/core/**"
  - "tests/pipeline/**"
  - "tests/engine/**"
---

# Solver, training and evaluation

**One solver backend: the statically-enumerated tree.** An infoset is
`(node_id, bucket)` — an index into a table allocated once at full size, so
memory is flat in iteration count. Runs are loadable iff they carry
`STATIC_CHECKPOINT.json`; checkpoints from the deleted dynamic backend are
unreadable by design.

- **Never compare arms across knob tiers.** Nothing enforces this; check
  `base_seed` and every tier knob before putting two numbers together, and
  never hand-transcribe a score. `poker-solver arms --experiment X --control Y`
  reads the tags back and subtracts exact_br arms directly — the estimator is
  deterministic, so no p-value is needed.
- **Scores live in the record (Postgres), not in per-run files.** `evaluate`
  writes a row through the record sink; `ledger`, `curve` and `arms` read it.
  `<run_dir>/evals/*.json` is written only by resolver-match scoring now, so
  code that reads that directory to decide something sees almost nothing —
  `prune-checkpoints` protects scored rungs from it and cannot see a DB-scored
  rung. **Score before pruning**, and fix that reader before trusting it.
- **Experiment bookkeeping** goes through `--experiment`/`--arm`/`--parent`,
  with `--set k=v` for config overrides; the tags are recorded on every eval.
  `--set` flags are dropped on resume — check for a continuation boundary
  before reading a mid-ladder turn as a result.
- **`reference/` may not import the estimators it validates.** The oracles
  check the production estimators to 1e-9; an oracle importing what it
  validates makes that agreement circular while the test still passes. This is
  an import-linter contract, and the one that is about correctness.
- Training and evaluation may not import each other (contract). Keep the
  interfaces between solver, training and evaluation typed and explicit.
- Tests are deterministic: fixed seeds, no wall-clock assertions. Expensive
  tests are `@pytest.mark.slow`; the default timeout is 5s and a longer one is
  declared with `@pytest.mark.timeout(<seconds>)`.
- Run the **full** suite when a change touches training, abstraction/bucketing,
  evaluator logic, config loading, or shared infrastructure.
