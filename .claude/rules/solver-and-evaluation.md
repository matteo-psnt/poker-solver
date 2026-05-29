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
memory is flat in iteration count. The old dynamic backend (hashed `InfoSetKey`)
is gone, and every checkpoint it wrote is unreadable at HEAD by design. Runs are
loadable iff they carry `STATIC_CHECKPOINT.json`.

- **Never compare arms across knob tiers.** `compare`/`report` refuse by design.
  Never hand-transcribe scores.
- **Eval records are per-run files**, not ledger appends: `evaluate` writes the
  complete row into `<run_dir>/evals/<slug>.json`. There is no stored index —
  `ledger` DERIVES one on every read, which is what makes concurrent evaluation
  from several boxes safe.
- **`eval-*.json` and `record-*.json` are LEGACY** and the rebuild skips both on
  purpose: a legacy record points at the old filename, so reading both enters
  one evaluation twice (measured: 63 rows became 110). A sparse `ledger` means
  un-migrated legacy files on the share, NOT a broken rebuild.
- **`reference/` may not import the estimators it validates.** The oracles check
  the production estimators to 1e-9; an oracle importing what it validates makes
  that agreement circular while the test still passes. This is an import-linter
  contract and the one that is about correctness rather than tidiness.
- **Experiment bookkeeping** goes through `--experiment`/`--arm`/`--parent`
  (`--set k=v` for config overrides). `report --experiment` pins every arm to
  the control's knob tier and pairs each variant against its control.
  `curve --run` is the within-run exploitability-vs-iteration artifact.
- **Prefer explicit, typed interfaces** between solver, training and evaluation.
- Keep tests deterministic (fixed seeds, no nondeterministic assertions). Mark
  expensive tests `@pytest.mark.slow`; intentionally longer ones get a tight
  explicit `@pytest.mark.timeout(<seconds>)`. Default timeout is 5s.
- Run the **full** suite when a change touches training, abstraction/bucketing,
  evaluator logic, config loading, or shared infrastructure.
