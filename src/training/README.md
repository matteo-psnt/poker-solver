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
    # use_parallel=True: spawns worker processes
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
    # Creates storage backend (memory or disk)
    # Memory: InMemoryStorage - fast, no persistence
    # Disk: DiskBackedStorage - LRU cache + HDF5

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
├── Spawn worker processes ────────→  │ Worker 0: run K iterations
│                                     │   ├── Deserialize abstractions
│                                     │   ├── Create solver (in-memory storage)
│                                     │   ├── Run train_iteration() K times
│                                     │   └── Return infoset data + utilities
│                                     │
│                                     │ Worker 1: run K iterations
│                                     │   └── (same as Worker 0)
│                                     │
├── Collect results ←─────────────────┤ Worker N-1: run K iterations
│                                     │
├── Merge worker results              │
│   └── Average regrets/strategies    │
│                                     │
├── Update main solver storage        │
├── Log metrics                       │
└── Checkpoint (if needed)            │
```

**Parallel Training Code Path:**
```python
TrainingSession.train(use_parallel=True, num_workers=N, batch_size=B)
└── _train_parallel()
    ├── Serialize abstractions (once)
    │   └── pickle.dumps(action_abstraction, card_abstraction)
    │
    └── for each batch:
        ├── Spawn N worker processes
        │   └── _worker_process():
        │       ├── Deserialize abstractions
        │       ├── Create solver with in-memory storage
        │       ├── Run train_iteration() K times
        │       └── Queue results {utilities, infoset_data}
        │
        ├── Collect results from queue
        │
        ├── Merge worker results
        │   └── _merge_worker_results():
        │       ├── Average regrets across workers
        │       ├── Average strategy_sums across workers
        │       └── Update main storage
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
- Amortizes process spawn/join overhead
- Allows periodic checkpointing
- Better progress reporting
- Typical: batch_size = num_workers * 10

**Merge Strategy:**
```python
def _merge_worker_results(worker_results):
    for infoset_key in all_infosets:
        # Collect from all workers that visited this infoset
        all_regrets = [w[key].regrets for w in workers if key in w]
        all_strategies = [w[key].strategy_sum for w in workers if key in w]

        # Handle action set mismatches (due to sampling variance)
        if regrets have different shapes:
            # Pad smaller arrays to match largest action set
            # Fill missing actions with zeros
            all_regrets = pad_to_max_size(all_regrets)
            all_strategies = pad_to_max_size(all_strategies)

        # Sum regrets and strategies (CFR theory: additive accumulation)
        merged_regrets = np.sum(all_regrets, axis=0)
        merged_strategy = np.sum(all_strategies, axis=0)

        # Accumulate into main storage
        storage.infoset[key].regrets += merged_regrets
        storage.infoset[key].strategy_sum += merged_strategy
```

**Performance Characteristics:**
- **Speedup:** ~0.7-0.9x per worker (overhead from serialization, merging)
- **Memory:** N × solver memory (each worker has full solver)
- **Best Use:** Long runs (>100 iterations), multi-core machines
- **Overhead:** Process spawn ~0.5s, serialization ~0.1s, merge ~0.2s per batch

---

### 5. Storage System (`src/solver/storage.py`)

**Two Implementations:**

**InMemoryStorage:**
```python
# Simple dict-based storage
infosets: Dict[str, InfoSet] = {}

# Fast lookups, no I/O
# Used for: testing, parallel workers, short runs
# Limitation: RAM-bound, no persistence
```

**DiskBackedStorage:**
```python
# LRU cache + HDF5 backing
cache: OrderedDict[str, InfoSet] = OrderedDict()  # In-memory LRU
file: h5py.File                                   # Persistent storage

# Write strategy:
# - New infosets go to cache
# - Cache eviction → flush to HDF5
# - Periodic flush every N iterations

# Read strategy:
# - Check cache first
# - Cache miss → load from HDF5 → add to cache

# Used for: long training runs, large games
# Trade-off: I/O overhead for persistence
```

**Configuration:**
```yaml
storage:
  backend: "disk"           # "memory" or "disk"
  cache_size: 100000        # Number of infosets in LRU cache
  flush_frequency: 1000     # Flush to disk every N iterations
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
3. Process spawn overhead (parallel)
   - Mitigated by: Batch processing, abstraction serialization

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
- Amortizes overhead (spawn, join, merge)
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
- Padding with zeros maintains correctness (zero regret for unseen actions)
- Uses the most complete action set discovered across workers

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
tra✅ **FIXED:** Parallel training now correctly sums regrets (not averages)
2. ✅ **FIXED:** Action set mismatches are handled by padding (not skipping)
3. Is batch processing the right abstraction for parallel training?
4. Should workers use shared storage instead of merging?
5. Is the checkpoint frequency optimal (every 1000 iterations)?
6
## Questions for Review

**Is this approach sound?**
1. Is the parallel training merge strategy correct (averaging regrets)?
2. Is batch processing the right abstraction for parallel training?
3. Should workers use shared storage instead of merging?
4. Is the checkpoint frequency optimal (every 1000 iterations)?
5. Are there better ways to serialize/deserialize abstractions?

**Performance concerns:**
1. Is the overhead of process spawning acceptable?
2. Should we use threads instead of processes?
3. Is LRU cache the best strategy for disk storage?
4. Should we compress checkpoints?

**Extensibility:**
1. How hard would it be to add distributed training?
2. Can this support heterogeneous worker capabilities?
3. Is the config system flexible enough for experiments?
4. Should we support dynamic abstraction refinement?

---

**Last Updated:** December 22, 2025
**Author:** Training system maintainer
**Related Docs:** ARCHITECTURE.md, README.md
