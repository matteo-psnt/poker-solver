# Training System Architecture

## Overview

The training system orchestrates Monte Carlo Counterfactual Regret Minimization (MCCFR) to compute near-optimal poker strategies using parallel multiprocessing with hash-partitioned shared memory. Includes comprehensive checkpointing, metrics tracking, and configuration management.

**Key Features:**
- Parallel multiprocessing training with hash-partitioned infosets
- Automatic checkpointing and resume capability
- Real-time metrics tracking and reporting
- Configurable abstractions (card/action) and storage backends
- Production-ready with 100+ tests

---

## System Components

### 1. TrainingSession (`src/training/trainer.py`)

**Purpose:** Main orchestration class that manages the complete training lifecycle.

**Responsibilities:**
- Initialize solver components (abstractions, storage, solver)
- Run parallel training iterations with hash-partitioned shared memory
- Handle checkpointing and resume
- Track metrics and progress
- Manage run metadata

**Key Methods:**
```python
TrainingSession(config: Config, run_id: str | None = None)
    # Initialize trainer from configuration
    # Creates run directory and initializes components

train(num_iterations: int | None = None, num_workers: int | None = None, batch_size: int | None = None)
    # Main training loop - always uses parallel multiprocessing with hash-partitioned shared memory
    # num_workers: Number of parallel workers (default: CPU count, use 1 for sequential-like behavior)
    # batch_size: Iterations per batch (default: 80)

TrainingSession.resume(run_dir: Path) -> TrainingSession
    # Static method to resume training from checkpoint
    # Loads solver state from disk and returns configured TrainingSession
    # Call .train() on returned session to continue training

evaluate(num_hands: int = 1000)
    # Evaluate current strategy
    # Returns exploitability estimate
```

**Architecture:**
```
TrainingSession
├── config: Config                        # Training configuration
├── solver: MCCFRSolver                   # MCCFR solver instance
├── action_model: ActionModel             # Action space model
├── card_abstraction: BucketingStrategy   # Card bucketing strategy
├── storage: Storage                      # Regret/strategy storage
├── metrics: MetricsTracker               # Training metrics
└── run_tracker: RunTracker               # Run metadata
```

---

### 2. Component Builders (`src/training/components.py`)

**Purpose:** Centralized builder functions for creating solver components from configuration.

**Key Functions:**

```python
build_card_abstraction(config: Config, ...) -> BucketingStrategy
    # Loads precomputed combo-level abstraction
    # Uses suit isomorphism and board clustering
    # Raises FileNotFoundError if abstraction not found

build_storage(config: Config, run_dir: Path) -> Storage
    # Creates in-memory storage with optional disk checkpointing
    # Optionally saves periodic checkpoints to disk for persistence

build_solver(config: Config, ...) -> MCCFRSolver
    # Creates MCCFR solver with all components
    # Configures CFR+, sampling method, game parameters
```

**Why Separate Builders?**
- Makes components independently testable
- Allows easy swapping of implementations
- Clear separation of concerns

---

### 3. Parallel Training Architecture

**Hash-Partitioned Shared Memory Architecture:**
```
Coordinator Process                   Worker Processes (N workers)
│                                     │
├── Create shared memory arrays       │
│   (regrets, strategies, actions)    │
│   Partitioned by worker ownership   │
│                                     │
├── Serialize abstractions            │
│   (action_model + card_abs)         │
│                                     │
├── Start worker pool ─────────────→  │ Worker 0..N-1: attach to shared memory
│                                     │   └── Initialize solver with partitioned storage
│                                     │   └── Each worker owns hash(key) % N partition
│                                     │
└── For each batch:                   │
    ├── Submit batch to workers ────→ │ Worker: Run K iterations
    │                                 │   ├── For each game iteration:
    │                                 │   │   ├── Sample game & traverse tree
    │                                 │   │   ├── For each infoset encountered:
    │                                 │   │   │   ├── If owned: update directly in shared memory
    │                                 │   │   │   └── If remote: read from shared memory (may be stale)
    │                                 │   │   └── Continue CFR traversal
    │                                 │   └── Signal batch complete
    │
    ├── Exchange ID requests ←──────→ │ Workers share newly discovered infoset IDs
    │                                 │
    ├── Log metrics                   │
    └── Checkpoint (if needed)        │
```

**Training Code Path:**
```python
TrainingSession.train(num_workers=N, batch_size=B)
└── _train_partitioned()
    ├── Serialize abstractions (once)
    │   └── pickle.dumps(action_model, card_abstraction)
    ├── Start SharedArrayWorkerManager (persistent pool with shared memory)
    │   ├── Create shared memory arrays (regrets, strategies, action_counts)
    │   └── Each worker attaches and gets exclusive ID range
    │
    └── for each batch:
        ├── Submit batch to all workers (parallel execution)
        │   └── worker loop:
        │       ├── Run train_iteration() K times
        │       │   └── For each infoset: write to owned partition, read from any
        │       └── Signal completion
        │
        ├── Exchange ID requests between workers
        │   └── Workers share IDs for newly discovered remote infosets
        │
        ├── Log metrics from shared memory
        └── Checkpoint (if iteration % freq == 0)
            └── Each worker saves its partition to disk
```

**Key Design Decisions:**

**Why Hash-Partitioned Infosets?**
- Each worker owns `hash(infoset_key) % num_workers` partition
- No merge step - workers update their partition directly in shared memory
- Workers can read from any partition (lock-free, potentially stale)
- Eliminates expensive merge bottleneck (was 81% of coordinator time)

**Why Serialize Abstractions?**
- Loading card abstraction from disk takes ~5-10 seconds per worker
- Serialized abstractions are ~5MB, transferred once to all workers
- Workers deserialize in <0.1 seconds
- Massive speedup: 1 disk load + N fast deserializations vs N disk loads

**Why Shared Memory Arrays?**
- Direct NumPy views into shared memory (zero-copy between workers)
- Lock-free reads (stale data is acceptable for MCCFR sampling)
- Owner-only writes (no race conditions)
- Eliminates 164MB broadcast overhead per batch

**Why Batch Processing?**
- Amortizes checkpoint overhead across many iterations
- Allows periodic progress reporting
- Workers can exchange newly discovered infoset IDs between batches
- Default: batch_size = 80 iterations per batch

**Ownership Model:**
```python
def get_owner(infoset_key: InfoSetKey) -> int:
    # Deterministic, stable hash (xxhash, not Python's randomized hash())
    key_hash = xxhash.xxh64(infoset_key).intdigest()
    return key_hash % num_workers

# Worker decides: can I update this infoset?
if get_owner(key) == my_worker_id:
    # Owner: write directly to shared memory
    update_infoset_in_place(key, regrets, strategies)
else:
    # Non-owner: read current values (may be stale, that's OK for MCCFR)
    infoset = read_from_shared_memory(key)
```

**Performance Characteristics:**
- **Speedup:** ~3-5x throughput improvement over merge-based approach
- **Worker Idle:** Reduced from 90% to ~10-20%
- **Memory:** N × solver memory + shared arrays (~400MB for 2M infosets)
- **Best Use:** All training (use num_workers=1 for sequential-like behavior)
- **Overhead:** One-time worker pool startup + minimal batch sync

---

### 4. Storage System (`src/solver/storage.py`)

**Two Implementations:**

**SharedArrayStorage:** (Training)
```python
# Hash-partitioned flat NumPy arrays in shared memory
# Each worker owns hash(key) % num_workers partition
# Direct memory access (zero-copy between workers)
# Owner-only writes, lock-free reads

# Data layout:
# - shared_regrets: float32[max_infosets, max_actions]
# - shared_strategy_sum: float32[max_infosets, max_actions]
# - shared_action_counts: int32[max_infosets]

# Used for: all parallel training
# Performance: Lock-free, no merge overhead
```

**InMemoryStorage:** (Read-Only)
```python
# Simple dict-based storage for loading checkpoints
# Read-only - raises error on write attempts

# Used for: charts, analysis, debugging
# - Load checkpoint from disk (HDF5 files)
# - Provides .infosets property for iteration
# - Cannot create new infosets or train

# DO NOT use for training - use SharedArrayStorage instead
```

**Checkpointing:**
```python
# Periodic saves to HDF5 files (async in background thread)
# Each worker saves its partition independently

# Write strategy:
# - Workers update shared memory in-place during training
# - Periodic checkpoint: each worker saves its partition to HDF5
# - Async execution (non-blocking, training continues)
# - Controlled by checkpoint_frequency in training config

# Read strategy (on resume):
# - Load all partitions from HDF5
# - Re-partition keys based on current num_workers (supports resume with different worker count)
# - Continue training with partitioned updates

# Used for: production training with periodic persistence
# Trade-off: Brief coordinator pause to collect worker partition info
```

**Configuration:**
```yaml
storage:
  checkpoint_enabled: true  # Save periodic checkpoints to disk (recommended)

training:
  checkpoint_frequency: 2000  # Save checkpoint every N iterations
```

---

### 5. Metrics Tracking (`src/training/metrics.py`)

**MetricsTracker:**

**Tracked Metrics:**
- **Utility:** Player 0 EV per iteration (convergence indicator)
- **Infosets:** Number of unique game states discovered
- **Timing:** Iterations per second, total runtime
- **Convergence:** Rolling averages, standard deviation

**Key Methods:**
```python
log_iteration(iteration, utility, num_infosets)
    # Log metrics for an iteration

get_avg_utility() -> float
    # Rolling average utility (window_size=100)

get_iterations_per_second() -> float
    # Training throughput

get_summary() -> dict
    # Complete metrics summary
```

**Why Track Utility?**
- In self-play, utility should converge to ~0 (balanced strategies)
- Large swings indicate non-convergence
- Standard deviation tracks stability

**Console Output:**
```
Iteration 1000/10000 [===>    ] 10.0%
  Avg Utility: -0.03 ± 2.1
  Infosets: 45,213
  Speed: 12.5 iter/s
  ETA: 12m 3s
```

---

### 6. Run Tracking (`src/training/run_tracker.py`)

**Purpose:** Lightweight metadata tracking for training runs.

**RunTracker:**
```python
RunTracker(run_dir: Path, config_name: str, config: Config, action_config_hash: str)
    # Initialize tracker for a run

initialize()
    # Create run directory and .run.json file

update(iterations, runtime_seconds, num_infosets, storage_capacity)
    # Update progress metadata

mark_completed()
    # Mark run as successfully completed

mark_failed(cleanup_if_empty: bool = True)
    # Mark run as failed, optionally cleanup

@staticmethod
list_runs(runs_dir: Path) -> list[str]
    # List all runs in directory
```

**Metadata Format (`.run.json`):**
```json
{
  "run_id": "run-20251222_143022",
  "config_name": "production",
  "status": "running",  // "running", "completed", "failed"
  "iterations": 5000,
  "runtime_seconds": 3842.5,
  "num_infosets": 125403,
  "storage_capacity": 2000000,
  "action_config_hash": "abc123",
  "started_at": "2025-12-22T14:30:22",
  "resumed_at": null,
  "completed_at": null,
  "config": { /* full config dict */ }
}
```

**Use Cases:**
- CLI: View past runs, resume training
- Monitoring: Check run status
- Analysis: Compare configurations
- Cleanup: Identify failed/incomplete runs

---

### 8. MCCFR Solver (`src/solver/mccfr.py`)

**Core Algorithm Implementation:**

```python
MCCFRSolver.train_iteration() -> float:
    """Run one MCCFR iteration."""
    1. Deal cards (hole + board)
    2. Choose traversing player (alternates)
    3. Run external_sampling_cfr()
    4. Return utility
```

**External Sampling CFR:**
```python
def external_sampling_cfr(state, player, reach_prob):
    """Traverse game tree with external sampling."""

    if state.is_terminal():
        return state.get_payoff()

    if state.is_chance_node():
        # Sample next card
        return sample_chance_action()

    # Get information set (abstract state)
    infoset_key = encode_infoset_key(state, player, card_abstraction)
    infoset = storage.get_infoset(infoset_key)

    # Compute strategy from regrets
    strategy = infoset.get_strategy()

    if state.player == player:  # Traversing player
        # Explore all actions
        action_values = []
        for action in legal_actions:
            action_values.append(
                external_sampling_cfr(next_state, player, reach_prob * strategy[a])
            )

        # Update regrets
        value = sum(action_values[a] * strategy[a] for a in actions)
        for a in actions:
            regret = action_values[a] - value
            infoset.regrets[a] += reach_prob * regret

        return value

    else:  # Opponent
        # Sample single action
        action = sample_action(strategy)
        return external_sampling_cfr(next_state, player, reach_prob)
```

**CFR+ Enhancements:**
```python
# CFR+ uses positive regret floor
def get_strategy(regrets):
    positive_regrets = np.maximum(regrets, 0)  # Floor at 0

    if positive_regrets.sum() > 0:
        return positive_regrets / positive_regrets.sum()
    else:
        return uniform_strategy()
```

**Convergence:** CFR+ converges ~100x faster than vanilla CFR.

---

## Training Configuration

**Example Config (`config/training/production.yaml`):**
```yaml
training:
  num_iterations: 100000
  checkpoint_frequency: 1000
  verbose: true
  runs_dir: "data/runs"
  parallel_result_timeout_seconds: null  # null = wait indefinitely for workers

action_model:
  preflop_templates:
    sb_first_in: ["fold", "call", 2.5, 3.5, 5.0]
  postflop_templates:
    first_aggressive: [0.33, 0.66, 1.25]
    facing_bet: ["min_raise", "pot_raise", "jam"]
    after_one_raise: ["pot_raise", "jam"]
    after_two_raises: ["jam"]
  jam_spr_threshold: 2.0
  off_tree_mapping: "probabilistic"

resolver:
  enabled: true
  time_budget_ms: 300
  max_depth: 2
  max_raises_per_street: 4
  leaf_value_mode: "blueprint_rollout"
  range_update_mode: "bayes_light"
  policy_blend_alpha: 0.35

card_abstraction:
  config: "default"  # References precomputed abstraction

storage:
  backend: "disk"
  cache_size: 100000
  flush_frequency: 1000

solver:
  type: "mccfr"
  sampling_method: "external"
  cfr_plus: true
  linear_cfr: false

game:
  starting_stack: 200
  small_blind: 1
  big_blind: 2

system:
  seed: 42
  config_name: "production"
```

---

## Checkpointing & Resume

**Checkpoint Structure:**
```
data/runs/run-20251222_143022/
├── .run.json                    # Run metadata
├── action_counts.npy
├── action_signatures.pkl
├── cumulative_utility.npy
├── key_mapping.pkl
├── reach_counts.npy
├── regrets.npy
└── strategies.npy
```

**Checkpoint Tracking (`.run.json`):**
```json
{
  "iterations": 500000,
  "num_infosets": 1234,
  "storage_capacity": 2000000
}
```

**Resume Training:**
```python
# From CLI or code
trainer = TrainingSession.resume(run_dir="data/runs/run-20251222_143022")
trainer.train(num_iterations=10000)  # Continue for 10k more iterations
```

---

## Performance Characteristics

**Sequential Training:**
- Speed: ~10-30 iterations/second (depends on abstraction size)
- Memory: ~500MB - 2GB (depends on storage.cache_size)
- CPU: Single core saturated
- Best for: Testing, debugging, small experiments

**Parallel Training (8 workers):**
- Speed: ~60-180 iterations/second (6-7x speedup)
- Memory: ~3GB - 12GB (N × sequential memory)
- CPU: 8 cores utilized
- Overhead: ~10-15% from serialization/merging
- Best for: Production runs, large-scale experiments

**Bottlenecks:**
1. Card abstraction lookup (10-20% of time)
   - Mitigated by: Precomputation, efficient data structures
2. Storage I/O (disk backend)
   - Mitigated by: LRU cache, batched flushes
3. Worker pool startup + merge overhead (parallel)
   - Mitigated by: Persistent pool, larger batches

---

## Testing Strategy

**Test Coverage: ~65% overall**

**Test Files:**
- `tests/training/test_trainer.py` - TrainingSession tests (6 tests, fast)
- `tests/training/test_components.py` - Component builders (15 tests)
- `tests/training/test_run_tracker.py` - Metadata tracking (8 tests)
- `tests/training/test_metrics.py` - Metrics tracking (15 tests)

**Testing Philosophy:**
- ✅ **DO:** Test initialization, configuration, error handling
- ✅ **DO:** Test component creation and validation
- ✅ **DO:** Test metadata tracking and file operations
- ❌ **DON'T:** Run full training in tests (too slow)
- ❌ **DON'T:** Test MCCFR convergence (requires 1000s of iterations)

**Fast Test Approach:**
```python
# Don't do this (slow):
trainer.train(num_iterations=1000)  # 30+ seconds

# Do this instead (fast):
trainer = TrainingSession(config)  # Just test initialization
assert trainer.solver is not None
assert trainer.storage.num_infosets() == 0
```

---

## CLI Integration

**Training Commands:**
```bash
# Interactive CLI
$ uv run python -m src.cli
> Train Solver
  - Select config: production.yaml
  - Select card abstraction
  - Confirm and start

# View runs
> View Past Runs
  - Lists all runs with status
  - Shows iterations, time, infosets

# Resume training
> Resume Training
  - Select run to resume
  - Specify additional iterations
```

**CLI Implementation (`src/cli/flows/training.py`):**
```python
def train_solver():
    config = select_config(ctx)
    trainer = TrainingSession(config)
    trainer.train(num_workers=8)

def resume_training():
    run_dir = select_run()
    trainer = TrainingSession.resume(run_dir)
    trainer.train(num_iterations=ask_additional_iterations())
```

---

## Design Rationale

**Why Two Training Modes?**
- Sequential: Simplicity, debugging, reproducibility
- Parallel: Performance, production use
- Unified interface: Same API, easy to switch

**Why Component Builders?**
- DRY: Eliminates duplication between modes
- Testability: Test components independently
- Flexibility: Easy to swap implementations
- Clarity: Clear separation of concerns

**Why Not Use multiprocessing.Pool?**
- Need fine control over worker lifecycle
- Custom serialization strategy (abstractions)
- Custom result aggregation (merge regrets)
- Progress tracking during batch processing

**Why Batch Processing in Parallel?**
- Amortizes per-batch IPC/merge overhead
- Enables periodic checkpointing
- Better progress reporting
- Balances latency vs throughput

**Why Sum (not Average) Regrets in Merge?**
- Regrets are additive quantities that accumulate over iterations
- Each worker contributes independent samples from the MCCFR process
- Summing correctly accumulates counterfactual regret across all samples
- CFR convergence theory requires additive regret accumulation
- Workers may visit infosets with different frequencies/reach probabilities

**Why Handle Action Set Mismatches?**
- Workers may discover different legal actions due to sampling variance
- Action abstraction edge cases can cause inconsistent action sets
- Silently skipping infosets biases learning (some states never update)
- Aligning by action identity keeps regret updates consistent
- Missing actions contribute zero regret/strategy (equivalent to padding)

---

## Future Enhancements

**Potential Improvements:**
1. **Neural CFR:** Replace card abstraction with neural network
2. **Distributed Training:** Extend to multi-node (Ray, MPI) with network-based partition exchange
3. **Adaptive Sampling:** Dynamic batch sizes based on convergence metrics
4. **Incremental Checkpointing:** Only save changed partitions (delta checkpoints)
5. **GPU Acceleration:** Batch equity calculations on GPU
6. **Live Evaluation:** Continuous exploitability monitoring during training
7. **Dynamic Repartitioning:** Rebalance partitions based on actual infoset distribution

---

## Quick Reference

**Start Training:**
```python
from src.training.trainer import TrainingSession
from src.utils.config_loader import load_config

config = load_config("config/training/production.yaml")
trainer = TrainingSession(config)
# Parallel training (always) with 8 workers
results = trainer.train(num_iterations=10000, num_workers=8)
```

**Resume Training:**
```python
from pathlib import Path
# Resume from checkpoint (use static method)
trainer = TrainingSession.resume(Path("data/runs/run-20251222_143022"))
results = trainer.train(num_iterations=5000)  # Continue training
```

**Different Worker Counts:**
```python
# Single worker (sequential-like behavior, but still uses shared memory)
results = trainer.train(num_iterations=1000, num_workers=1)

# Multi-worker parallel (8 workers, batch_size=80)
results = trainer.train(num_iterations=1000, num_workers=8, batch_size=80)

# Auto-detect worker count (uses CPU count)
results = trainer.train(num_iterations=1000)  # num_workers defaults to CPU count
```

**Custom Configuration:**
```python
from src.utils.config import Config

config = Config.default().merge({
    "storage": {"checkpoint_enabled": False},
    "training": {"checkpoint_frequency": 500, "num_iterations": 5000},
})
```

## Architecture Review Questions

**Parallel correctness:**
1. ✅ ~~Should we use shared memory?~~ - Implemented hash-partitioned shared memory
2. ✅ ~~Is lock-free access theoretically sound?~~ - Yes, stale reads acceptable for MCCFR sampling
3. Does hash-based partitioning create load imbalance?

**Performance concerns:**
1. Is batch_size=80 optimal, or should it adapt based on checkpoint overhead?
2. Should we compress checkpoints to reduce I/O time?
3. Can we overlap computation and ID exchange to hide latency?

**Extensibility:**
1. How to extend hash-partitioning to multi-node (distributed hash table)?
2. Should we support dynamic abstraction refinement during training?
3. Can we use lock-free data structures to eliminate batch-level synchronization?

---

**Last Updated:** February 19, 2026
**Author:** Training system maintainer
**Related Docs:** ARCHITECTURE.md, README.md
