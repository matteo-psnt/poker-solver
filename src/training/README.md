# Training System Architecture

## Overview

The training system orchestrates Monte Carlo Counterfactual Regret Minimization (MCCFR) to compute near-optimal poker strategies. It supports both sequential and parallel training modes with comprehensive checkpointing, metrics tracking, and configuration management.

**Key Features:**
- Sequential and parallel (multiprocessing) training
- Automatic checkpointing and resume capability
- Real-time metrics tracking and reporting
- Configurable abstractions (card/action) and storage backends
- Production-ready with 100+ tests and ~65% coverage

---

## System Components

### 1. TrainingSession (`src/training/trainer.py`)

**Purpose:** Main orchestration class that manages the complete training lifecycle.

**Responsibilities:**
- Initialize solver components (abstractions, storage, solver)
- Run training iterations (sequential or parallel)
- Handle checkpointing and resume
- Track metrics and progress
- Manage run metadata

**Key Methods:**
```python
TrainingSession(config: Config, run_id: Optional[str] = None)
    # Initialize trainer from configuration
    # Creates run directory and initializes components

train(num_iterations: int, use_parallel: bool = True, num_workers: int = None, batch_size: int = None)
    # Main training loop
    # use_parallel=True: uses persistent worker pool (multiprocessing)
    #                  workers reset local storage per batch to avoid double counting
    # use_parallel=False: sequential execution

resume(from_checkpoint: Optional[int] = None)
    # Resume training from checkpoint
    # Loads solver state and continues iterations

evaluate(num_hands: int = 1000)
    # Evaluate current strategy
    # Returns exploitability estimate
```

**Architecture:**
```
TrainingSession
├── config: Config                        # Training configuration
├── solver: MCCFRSolver                   # MCCFR solver instance
├── action_abstraction: BettingActions    # Action space abstraction
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
build_action_abstraction(config: Config) -> BettingActions
    # Creates betting action abstraction from config
    # Defines available bet/raise sizes per street

build_card_abstraction(config: Config, ...) -> BucketingStrategy
    # Loads precomputed combo-level abstraction
    # Uses suit isomorphism and board clustering
    # Raises FileNotFoundError if abstraction not found

build_storage(config: Config, run_dir: Path) -> Storage
    # Creates in-memory storage with optional disk checkpointing
    # Optionally saves periodic checkpoints to disk for persistence

build_solver(config: Config, ...) -> BaseSolver
    # Creates MCCFR solver with all components
    # Configures CFR+, sampling method, game parameters
```

**Why Separate Builders?**
- Eliminates code duplication between sequential/parallel modes
- Makes components independently testable
- Allows easy swapping of implementations
- Clear separation of concerns

---

### 3. Sequential Training Flow

**Process:**
```
1. Initialize TrainingSession
   └── Build components (action_abstraction, card_abstraction, storage, solver)
   └── Create run directory and metadata
   └── Initialize metrics tracker

2. Run training loop
   for iteration in range(num_iterations):
       ├── solver.train_iteration()         # Run one MCCFR iteration
       ├── metrics.log_iteration(...)       # Track utility, infosets, timing
       ├── if iteration % checkpoint_freq:
       │   └── storage.save_checkpoint()    # Save regrets/strategies
       └── if iteration % log_freq:
           └── print progress metrics       # Show convergence progress

3. Complete training
   ├── storage.flush()                     # Persist all data
   ├── run_tracker.mark_completed()        # Update metadata
   └── return results dict
```

**Sequential Training Code Path:**
```python
TrainingSession.train(use_parallel=False)
└── _train_sequential()
    └── loop: solver.train_iteration()
        └── MCCFRSolver.train_iteration()
            ├── deal_cards()                    # Sample hole cards, board
            ├── external_sampling_cfr()         # Traverse game tree
            │   └── traverse recursively:
            │       ├── get_infoset()           # Abstract game state
            │       ├── get_strategy()          # Compute strategy from regrets
            │       ├── choose_action()         # Traverse/sample actions
            │       └── update_regrets()        # Backpropagate values
            └── storage.update_infoset()        # Store updated regrets
```

**When to Use Sequential:**
- Debugging (easier to trace execution)
- Small-scale experiments
- Baseline comparisons
- When overhead of multiprocessing > gains

---

### 4. Parallel Training Flow

**Architecture:**
```
Main Process                          Worker Processes (N workers)
│                                     │
├── Serialize abstractions            │
│   (action_abstraction + card_abs)   │
│                                     │
├── Start persistent worker pool ──→  │ Worker 0..N-1: initialize solver once
│                                     │   └── wait for jobs
│                                     │
└── For each batch:                   │
    ├── Submit jobs with iteration ─→ │ Worker: reset local storage + set iteration offset
    │   offsets + K iterations        │   ├── Run train_iteration() K times
    │                                 │   └── Return infoset data + utilities
    │
    ├── Collect results ←─────────────┤
    ├── Merge worker results (sum)    │
    ├── Update master storage         │
    ├── Log metrics                   │
    └── Checkpoint (if needed)        │
```

**Parallel Training Code Path:**
```python
TrainingSession.train(use_parallel=True, num_workers=N, batch_size=B)
└── _train_parallel()
    ├── Serialize abstractions (once)
    │   └── pickle.dumps(action_abstraction, card_abstraction)
    ├── Start WorkerManager (persistent pool)
    │
    └── for each batch:
        ├── Submit jobs with iteration offsets
        │   └── worker loop:
        │       ├── Reset local storage (per job)
        │       ├── Set solver.iteration = offset
        │       ├── Run train_iteration() K times
        │       └── Queue results {utilities, infoset_data}
        │
        ├── Collect results from queue
        │
        ├── Merge worker results (sum + action alignment)
        │   └── mark_dirty for disk-backed storage
        │
        ├── Log merged metrics
        └── Checkpoint (if iteration % freq == 0)
```

**Key Parallel Design Decisions:**

**Why Serialize Abstractions?**
- Loading card abstraction from disk takes ~5-10 seconds
- Serialized abstractions are ~5MB, fast to transfer
- Workers deserialize in <0.1 seconds
- Massive speedup: 1 disk load vs N disk loads

**Why In-Memory Storage for Workers?**
- Workers run independently with isolated storage
- No contention on shared storage
- Results merged after completion
- Main storage only updated after merge

**Why Batch Processing?**
- Amortizes merge/checkpoint overhead
- Allows periodic checkpointing
- Better progress reporting
- Default: batch_size = num_workers * 200

**Why Reset Worker Storage Each Batch?**
- Avoids double-counting when using persistent workers
- Keeps worker memory bounded
- Makes each batch an independent MCCFR sample

**Merge Strategy:**
```python
def _merge_worker_results(worker_results):
    for infoset_key in all_infosets:
        # Collect from all workers that visited this infoset
        worker_data = [w[key] for w in workers if key in w]

        # Build unified action list (master + any new worker actions)
        actions = union_actions(master_actions, [w.legal_actions for w in worker_data])
        action_index = {action: i for i, action in enumerate(actions)}

        # Sum regrets and strategies aligned by action identity
        merged_regrets = np.zeros(len(actions))
        merged_strategy = np.zeros(len(actions))
        for data in worker_data:
            for i, action in enumerate(data.legal_actions):
                idx = action_index[action]
                merged_regrets[idx] += data.regrets[i]
                merged_strategy[idx] += data.strategy_sum[i]

        # Accumulate into main storage
        infoset = storage.get_or_create_infoset(key, actions)
        infoset.regrets += merged_regrets
        infoset.strategy_sum += merged_strategy
        storage.mark_dirty(key)  # Required for disk-backed storage
```

**Performance Characteristics:**
- **Speedup:** ~0.7-0.9x per worker (overhead from serialization, merging)
- **Memory:** N × solver memory (each worker has full solver)
- **Best Use:** Long runs (>100 iterations), multi-core machines
- **Overhead:** One-time worker pool startup + per-batch IPC/merge costs

---

### 5. Storage System (`src/solver/storage.py`)

**Two Implementations:**

**InMemoryStorage:**
```python
# Simple dict-based storage
infosets: Dict[str, InfoSet] = {}

# Fast lookups, no I/O
# Used for: all training
```

**Checkpointing:**
```python
# Periodic saves to HDF5 files
# All operations stay in RAM (fast)
# Checkpoint saves current state to disk

# Write strategy:
# - All operations in-memory (fast dictionary access)
# - Periodic checkpoint: save all infosets to HDF5
# - Controlled by checkpoint_frequency in training config

# Read strategy (on resume):
# - Load all infosets from HDF5 into memory
# - Continue training in-memory

# Used for: production training with periodic persistence
# Trade-off: Brief I/O pause during checkpoint saves
```

**Configuration:**
```yaml
storage:
  checkpoint_enabled: true  # Save periodic checkpoints to disk (recommended)

training:
  checkpoint_frequency: 2000  # Save checkpoint every N iterations
```

---

### 6. Metrics Tracking (`src/training/metrics.py`)

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

### 7. Run Tracking (`src/training/run_tracker.py`)

**Purpose:** Lightweight metadata tracking for training runs.

**RunTracker:**
```python
RunTracker(run_dir: Path, config_name: str, config: dict)
    # Initialize tracker for a run

initialize()
    # Create run directory and .run.json file

update(iterations, runtime_seconds, num_infosets)
    # Update progress metadata

mark_completed()
    # Mark run as successfully completed

mark_failed(cleanup_if_empty: bool = True)
    # Mark run as failed, optionally cleanup

@staticmethod
list_runs(runs_dir: Path) -> List[str]
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
  "started_at": "2025-12-22T14:30:22",
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
    infoset_key = get_infoset_key(state, player)
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
  log_frequency: 100
  verbose: true
  runs_dir: "data/runs"
  parallel_result_timeout_seconds: null  # null = wait indefinitely for workers

action_abstraction:
  preflop_raises: [2.5, 3.0, 4.0, "allin"]
  flop_bets: [0.33, 0.5, 0.75, 1.0, "allin"]
  turn_bets: [0.5, 0.75, 1.0, "allin"]
  river_bets: [0.5, 0.75, 1.0, "allin"]

card_abstraction:
  config: "default_plus"  # References precomputed abstraction

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
├── checkpoints/
│   ├── checkpoint_1000.h5       # Iteration 1000 state
│   ├── checkpoint_2000.h5       # Iteration 2000 state
│   └── checkpoint_final.h5      # Final state
└── logs/
    └── training.log             # Detailed logs (if enabled)
```

**Checkpoint Contents (HDF5):**
```python
checkpoint.h5:
    /infosets/
        {infoset_key}/
            regrets: [N actions]
            strategy_sum: [N actions]
            legal_actions: [action_ids]
    /metadata/
        iteration: int
        timestamp: str
        num_infosets: int
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
$ uv run python -m scripts.cli
> Training Tools
> Start New Training
  - Select config: production.yaml
  - Select card abstraction
  - Confirm and start

# View runs
> Training Tools
> View Training Runs
  - Lists all runs with status
  - Shows iterations, time, infosets

# Resume training
> Training Tools
> Resume Training
  - Select run to resume
  - Specify additional iterations
```

**CLI Implementation (`src/cli/training_handler.py`):**
```python
def handle_start_training():
    config = select_config()
    trainer = TrainingSession(config)
    trainer.train(num_iterations=config.get("training.num_iterations"))

def handle_resume_training():
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
2. **Distributed Training:** Extend to multi-node (Ray, MPI)
3. **Adaptive Sampling:** Dynamic batch sizes based on convergence
4. **Better Parallelism:** Share storage via shared memory
5. **Incremental Checkpointing:** Only save changed infosets
6. **GPU Acceleration:** Batch equity calculations
7. **Live Evaluation:** Continuous exploitability monitoring

---

## Quick Reference

**Start Training:**
```python
from src.training.trainer import TrainingSession
from src.utils.config import Config

config = Config.load("production")
trainer = TrainingSession(config)
results = trainer.train(num_iterations=10000, use_parallel=True)
```

**Resume Training:**
```python
from pathlib import Path
trainer = TrainingSession.resume(Path("data/runs/run-20251222_143022"))
results = trainer.train(num_iterations=5000)
```

**Sequential vs Parallel:**
```python
# Sequential (single core)
results = trainer.train(num_iterations=1000, use_parallel=False)

# Parallel (8 workers, batch_size=80)
results = trainer.train(num_iterations=1000, use_parallel=True, num_workers=8, batch_size=80)
```

**Custom Configuration:**
```python
config = Config.default()
config.set("storage.backend", "memory")
config.set("training.checkpoint_frequency", 500)
config.set("training.num_iterations", 5000)
```

## Questions for Review

**Parallel correctness:**
1. Is per-batch reset + summation merge the right abstraction for MCCFR parallelism?
2. Should we support shared storage (shared memory) instead of merging?

**Performance concerns:**
1. Is the batch size heuristic (num_workers * 200) optimal for real runs?
2. Should we compress checkpoints to reduce I/O?

**Extensibility:**
1. How hard would it be to add distributed training (Ray/MPI)?
2. Should we support dynamic abstraction refinement during training?

---

**Last Updated:** December 22, 2025
**Author:** Training system maintainer
**Related Docs:** ARCHITECTURE.md, README.md
