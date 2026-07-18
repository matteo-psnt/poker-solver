# Training System Architecture

## Overview

The training system orchestrates Monte Carlo Counterfactual Regret
Minimization (MCCFR) using parallel multiprocessing over hash-partitioned
shared memory (hogwild-style: owner-only writes, lock-free reads). It
includes Zarr-based checkpointing with resume, per-batch convergence
metrics, and run metadata with git provenance and per-attempt timelines.

---

## System Components

### 1. TrainingSession (`src/pipeline/training/trainer/session.py`)

Main orchestration class managing the complete training lifecycle:
component construction, the parallel training loop, checkpointing, metrics,
and run metadata.

**Key methods:**
```python
TrainingSession(config: Config, run_id: str | None = None,
                run_tracker: RunTracker | None = None)
    # Build all components from configuration; create the run directory.

train(num_iterations=None, num_workers=None, batch_size=None)
    # Parallel training over hash-partitioned shared memory.
    # num_workers defaults to CPU count; batch_size defaults to
    # iterations_per_worker * num_workers.

TrainingSession.resume(run_dir: Path, capacity_override: int | None = None)
    # Classmethod: load solver state from a checkpoint and return a
    # configured session; call .train() to continue. The iteration to
    # continue from comes from the checkpoint manifest (committed atomically
    # with the arrays), not from .run.json, which is written separately and
    # can lag the data when a leg is hard-killed.

evaluate(num_samples=10000, num_rollouts_per_infoset=100,
         use_average_strategy=True, seed=None)
    # Legacy rollout diagnostic only. For trustworthy numbers use the LBR
    # evaluator via `poker-solver-run evaluate` (see evaluation README).
```

**Attributes:**
```
TrainingSession
├── config: Config
├── run_dir: Path
├── solver: MCCFRSolver
├── action_model: ActionModel
├── card_abstraction: BucketingStrategy
├── storage: SharedArrayStorage
├── checkpoints: CheckpointManager
├── metrics: MetricsTracker
└── run_tracker: RunTracker
```

### 2. Component Builders (`src/pipeline/training/components.py`)

Centralized builders keep components independently testable and swappable:

- `build_card_abstraction(...)` — loads the precomputed combo-level
  abstraction (raises if missing).
- `resolve_card_abstraction_hash(...)` — resolves the abstraction hash that
  gets pinned into run metadata.
- `build_storage(...)` — always returns a `SharedArrayStorage`.
- `build_solver(...)` — assembles the MCCFR solver.
- `build_evaluation_solver(...)` / `evaluate_solver_exploitability(...)` —
  read-only solver construction for evaluation callers.

### 3. Parallel Training (`trainer/partitioned.py`, `parallel_manager/`)

`TrainingSession.train()` delegates to `train_partitioned()`, which runs a
persistent worker pool (`SharedArrayWorkerManager`) over shared memory:

```
Coordinator                              Workers (N)
│                                        │
├── Create shared arrays (ARRAY_SPECS)   │
├── Serialize abstractions once ───────→ │ attach to shared memory,
│   (pickle: action_model + card_abs)    │ deserialize abstractions
│                                        │
└── For each batch:                      │
    ├── Submit batch ──────────────────→ │ run K iterations each:
    │                                    │   owned infoset  → write in place
    │                                    │   remote infoset → lock-free read
    │                                    │                    (may be stale)
    ├── Exchange newly discovered IDs ←→ │
    ├── Append metrics row               │
    └── Checkpoint (if due, async)       │
```

**Ownership model** (`src/engine/solver/storage/shared_array/ownership.py`):

```python
owner = xxhash.xxh64(infoset_key).intdigest() % num_workers
```

- Deterministic, stable across processes (not Python's randomized `hash()`).
- Owner-only writes: no locks, no race conditions on updates.
- Any worker may read any partition; stale reads are acceptable for MCCFR
  sampling.
- **No merge step** — workers update shared memory directly. This
  eliminated the merge bottleneck of the earlier design.

**Why serialize abstractions?** One disk load + N fast deserializations
(~5MB pickle) instead of N × 5-10s disk loads.

**Why batches?** Amortizes checkpoint/metrics overhead and provides the
synchronization points at which workers exchange newly discovered infoset
IDs. Batch size defaults to `iterations_per_worker * num_workers`
(`iterations_per_worker` defaults to 1000; production sets 5000).

### 4. Storage (`src/engine/solver/storage/`)

**Shared arrays** are declared once in `ARRAY_SPECS`
(`src/engine/solver/storage/array_specs.py`):

| Array                     | dtype   | Checkpoint dataset   |
|---------------------------|---------|----------------------|
| `shared_regrets`          | float64 | `regrets`            |
| `shared_strategy_sum`     | float64 | `strategies`         |
| `shared_action_counts`    | int32   | `action_counts`      |
| `shared_reach_counts`     | int64   | `reach_counts`       |
| `shared_cumulative_utility` | float64 | `cumulative_utility` |

**Two storage implementations:**

- `SharedArrayStorage` — hash-partitioned flat NumPy views into shared
  memory; used for all training.
- `InMemoryStorage` — read-only dict-based storage for loading checkpoints
  (charts, analysis, evaluation); raises on any write.

### 5. Checkpointing (`trainer/checkpointing.py`, `storage/shared_array/checkpoint.py`)

`CheckpointManager` runs checkpoints on a background single-thread executor
with back-pressure (a new checkpoint waits for the previous one), so
training continues while a snapshot is written.

**Format is Zarr**, compressed with Blosc/zstd. A checkpoint directory
contains:

```
data/runs/<run_id>/checkpoints/<checkpoint_id>/
├── checkpoint.zarr/         # regrets, strategies, action_counts,
│                            # reach_counts, cumulative_utility
├── key_mapping.pkl          # infoset key → row index
└── action_signatures.pkl
```

Config knobs (all have defaults; production.yaml relies on them):

```yaml
storage:
  zarr_compression_level: 1   # benchmarked: fastest AND smallest
  zarr_chunk_size: 50000
training:
  checkpoint_frequency: 500000
  max_checkpoint_overhead: 0.1
```

On resume, all partitions are loaded and re-partitioned for the current
`num_workers`, so worker count may change between sessions.

### 6. Metrics (`metrics.py`, `metrics_history.py`)

- `MetricsTracker` — in-memory rolling stats: utility mean/std, infoset
  count, iterations/sec, plus convergence quality (regret/entropy) via
  `record_quality` / `compute_quality_from_arrays`.
- `MetricsHistoryWriter` — appends one JSON row per batch to
  `<run_dir>/metrics.jsonl` (utility, speed, regret and entropy
  convergence), giving each run a persistent convergence curve.

In self-play, utility should converge toward ~0; large swings indicate
non-convergence.

### 7. Run Tracking (`run_tracker.py`)

Three classes: `RunMetadata` (the schema), `AttemptRecord` (one training
session within a run), and `RunTracker` (lifecycle API:
`initialize`, `update`, `mark_completed`, `mark_failed`, `mark_resumed`,
`mark_interrupted`, `list_runs`).

**Metadata (`.run.json`):**
```json
{
  "run_id": "run-20260717_143022",
  "config_name": "production",
  "status": "running",
  "iterations": 500000,
  "runtime_seconds": 3842.5,
  "num_infosets": 125403,
  "storage_capacity": 4000000,
  "action_config_hash": "abc123",
  "card_abstraction_hash": "def456",
  "representation_version": 1,
  "git_commit": "993255a",
  "git_dirty": false,
  "started_at": "2026-07-17T14:30:22",
  "completed_at": null,
  "attempts": [ /* per-session AttemptRecords with git provenance */ ],
  "config": { /* full config dict */ }
}
```

The `attempts` list records each start/resume as its own timeline entry, so
a run's wall-clock history survives interruptions. The
`card_abstraction_hash` pins exactly which abstraction the run was trained
with; evaluation refuses runs without it.

---

## Configuration

Actual `config/training/production.yaml`:

```yaml
solver:
  iteration_weighting: dcfr

action_model:
  preflop_templates:
    sb_first_in: ["fold", "call", 2.5, 3.5, 5.0]
    bb_vs_open: ["fold", "call", "3.5x_open", "4.5x_open"]
    sb_vs_3bet: ["fold", "call", "2.3x_last", "jam"]
  postflop_templates:
    first_aggressive: [0.33, 0.66, 1.25]
    facing_bet: ["min_raise", "pot_raise", "jam"]
    after_one_raise: ["pot_raise", "jam"]
    after_two_raises: ["jam"]

resolver:
  max_raises_per_street: 5

card_abstraction:
  config: production

training:
  num_iterations: 1000000
  checkpoint_frequency: 500000
  iterations_per_worker: 5000

storage:
  initial_capacity: 4000000

system:
  config_name: "production"
```

Unset fields fall back to defaults in `src/shared/config.py`.

---

## CLI Integration

**Interactive** (`uv run poker-solver`, flows in
`src/interfaces/cli/flows/training.py`): train, resume, evaluate, and view
runs; flows delegate to `src/pipeline/services.py`.

**Headless** (`poker-solver-run`, `src/interfaces/cli/headless.py`):

```bash
poker-solver-run train --config production
poker-solver-run evaluate --run <id> --scorer lookahead
poker-solver-run ledger            # browse recorded evaluations
poker-solver-run compare --a <run> --b <run>   # paired comparison
```

---

## Quick Reference

```python
from pathlib import Path
from src.pipeline.training.trainer import TrainingSession
from src.shared.config_loader import load_config

# Start training
config = load_config("config/training/production.yaml")
trainer = TrainingSession(config)
trainer.train(num_workers=8)

# Resume from checkpoint
trainer = TrainingSession.resume(Path("data/runs/run-20260717_143022"))
trainer.train(num_iterations=500000)

# Single worker (sequential-like, still shared memory)
trainer.train(num_iterations=1000, num_workers=1)
```

---

## Testing

Tests live in `tests/pipeline/training/`: `test_trainer.py`,
`test_components.py`, `test_run_tracker.py`, `test_metrics.py`,
`test_metrics_history.py`, `test_checkpointing.py`, `test_resume.py`, and
others. Tests cover initialization, configuration, checkpoint round-trips,
and metadata tracking — not full training runs or MCCFR convergence (too
slow for the suite; mark anything expensive `@pytest.mark.slow`).

---

## Future Directions

1. **Distributed training:** extend hash-partitioning to multi-node
   (network-based partition exchange).
2. **Adaptive batching:** dynamic batch sizes based on checkpoint overhead.
3. **Incremental checkpointing:** delta snapshots of changed partitions.
4. **Live evaluation:** continuous LBR monitoring during training.

**Related docs:** `README.md`, `src/pipeline/evaluation/README.md`,
`src/pipeline/abstraction/postflop/README.md`
