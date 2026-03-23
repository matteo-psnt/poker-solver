---
name: verify
description: Drive this solver's real surfaces to observe a change working — CLI entrypoints, and the multi-process static training path that has no CLI yet.
---

# Verifying changes in poker-solver

Runtime observation, not tests. `uv run pytest` is CI's job.

## Handle

`uv sync --group dev` once, then everything runs under `uv run`.

Surfaces, in order of preference:

| Change touches | Surface |
|---|---|
| anything reachable from a command | `uv run poker-solver-run <cmd>` (`train`, `resume`, `precompute`, `evaluate`, `curve`, `report`, `promote`, `ledger`, `compare`, `checkpoint-profile`) |
| interactive flows | `uv run poker-solver` |
| `src/pipeline/training/static_parallel.py`, `src/engine/solver/storage/static_array.py`, `static_solver.py` | **no CLI wiring yet** — see below |
| `ui/` | `npm run dev` from `ui/` |

## The static-tree path has no CLI entrypoint

As of the `worktree-static-tree-rebuild` line, nothing in `src/` calls
`train_static_parallel` — only tests do. Check first:

```bash
grep -rn "train_static_parallel" src/
```

If that is still empty, the outermost real surface is the function itself,
which spawns real OS processes and real POSIX shared memory. Drive it from a
scratchpad script (NOT pytest) so you can crash it, restart it, and inspect
segments:

```python
# scratchpad/drive.py  — run with:
#   PYTHONPATH=<repo>:<scratchpad> uv run python scratchpad/drive.py
from src.pipeline.training.static_parallel import train_static_parallel
result = train_static_parallel(
    config, num_iterations=600, num_workers=4,
    session_id="run-...", abstraction=MyBuckets(),
)
```

Gotchas:

- `mp.get_context("spawn")` — every helper class the run touches
  (`abstraction`, worker targets) must be **module-level and picklable**.
  A `python -c` one-liner cannot spawn: `__main__` is not importable, so
  children die with `AttributeError: Can't get attribute 'child'`. Write a file.
- A ~2,100-node test tree (`starting_stack=20`, 3 buckets/street) trains 600
  iterations across 4 workers in ~2s. Do not reach for production configs.
- **Never bucket on Python `hash()`** in a driver. It is per-process
  randomised, so under spawn every worker buckets the same hand differently and
  you will be measuring your own harness. Key on `Card.rank_eval7()` /
  `suit_eval7()` instead. (`Buckets` in
  `tests/pipeline/training/test_static_parallel.py` has this problem.)
- `Card.rank_eval7` and `suit_eval7` are **methods**, not properties.

## Shared-memory hygiene

Segment names are `sts_<4 letters>_<12-hex digest of session_id|tree fingerprint>`
(`StaticArrayStorage._shm_name`). The digest means you cannot guess a segment
name from the session id — print `storage._shm_name("regrets")` if you need it.

A SIGKILLed coordinator leaks all five segments; the next run of the same
session then dies with a bare `FileExistsError: [Errno 17] File exists:
'/sts_regr_<digest>'`. To clear:

```python
from multiprocessing import shared_memory
for a in ("regr", "stra", "reac", "cumu", "visi"):
    try:
        s = shared_memory.SharedMemory(name=f"sts_{a}_{digest}"); s.close(); s.unlink()
    except FileNotFoundError:
        pass
```

macOS caps POSIX shm names at 30 chars excluding the leading `/` (measured);
31 including it. Names are 21 chars, so there is headroom.

## Probes worth running on this subsystem

- Restart the same session twice; SIGKILL a coordinator mid-run and restart.
- Two sessions on the same tree concurrently — names must differ.
- A worker that builds a *different* tree (different `num_buckets`) — must
  fail to attach, not attach and corrupt.
- Same bucket **counts** but different bucket **assignment** — this still
  attaches; the tree fingerprint does not cover assignment.
