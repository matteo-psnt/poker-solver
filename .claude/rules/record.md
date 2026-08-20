---
paths:
  - "src/adapters/**"
  - "src/shared/records.py"
  - "src/shared/ports/**"
  - "tests/adapters/**"
  - "src/interfaces/commands/ledger.py"
  - "src/interfaces/commands/runs.py"
  - "src/interfaces/commands/arms.py"
  - "src/interfaces/commands/curve.py"
  - "src/interfaces/commands/runinfo.py"
  - "src/interfaces/commands/evaluate.py"
  - "src/interfaces/commands/score.py"
  - "src/interfaces/commands/submit.py"
  - "src/interfaces/commands/prune_checkpoints.py"
---

# The record

The experiment record is Postgres, behind `src/adapters/postgres/`. The share
holds checkpoints, logs and legs; scores and run state do not live there.
`models.py` is the schema, and the adapter's README says why it is declarative
and not SQLAlchemy Core (`ty` checks `Eval.scor_mbb`; it cannot check
`evals.c.scor_mbb`).

- **`POKER_SOLVER_RECORD_DSN` is read from the shell at dispatch AND at read
  time.** `submit`/`score` refuse without it. `runs`, `ledger` and `runinfo`
  do not refuse — they answer from the share and silently omit every DB-only
  row. `eval "$(just record-env)"` first, in the same shell as the command.
- **A read costs round trips, not query time.** Measured 175 ms of RTT against
  3 ms of query. A fresh engine per call, or a transaction around a
  one-statement read, turns a screen into seconds without raising. One engine
  per process; the reader pool at least as wide as the widest screen's fan-out.
- **The node cannot import the driver.** The wrapper runs bare `python3.13`,
  so the stdlib-only node closure writes the record by shelling out to
  `poker-solver mirror-legs`, never by importing `adapters`.
- **The fake driver parses no SQL.** A statement can pass the whole suite and
  fail on the server. Before dispatching anything that writes, run the write
  against the live server from the laptop: seconds, and it exercises retry
  numbering better than a node does.
- **Assert the destination, not a count.** Four writers in one session
  reported success and wrote nowhere; their tests asserted a return value. A
  test of a write reads the row back, and any `--apply` is followed by a read.
- **A writer that is declared is not wired.** `RecordSink.flush` existed,
  was tested, and had no caller for a week. When adding a writer, grep for
  the call site in the same change.
