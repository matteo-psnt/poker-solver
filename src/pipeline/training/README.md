# Training

Trains a blueprint by MCCFR over the **statically-enumerated betting tree** —
the only backend. An infoset is `(node_id, bucket)`, which is an array index,
so the table is allocated once at full size and **memory is flat in iteration
count**: 1M and 300M iterations cost the same.

## Why the shape is what it is

The approach this replaced discovered infosets as it went, so the space never
stopped growing and every worker held dicts proportional to it. Fitted against
a live run (`infosets ~ 1.96 * iters^1.058`):

| iters | infosets | shared GB | per-worker GB | 8w node GB |
| --- | --- | --- | --- | --- |
| 5,000,000 | 24,169,390 | 2.4 | 2.6 | 23.3 |
| 10,000,000 | 50,335,061 | 5.0 | 4.1 | 37.6 |
| 30,000,000 | 161,007,993 | 16.1 | 10.2 | 98.0 |

A 30M-iteration run needs ~98 GB on 8 workers. No node we have reaches that,
and no worker count fixes it, because the growth was in the **keying**, not the
parallelism. The public tree is small and finite — 57,604 decision nodes under
`config/training/production.yaml` — so enumerating it statically removes the
growth instead of budgeting for it: ~16.8M rows, ~1.7 GB, fixed.

Everything else here follows from that. Because the row index is a pure
function of the tree and the tree is a pure function of config, every process
computes the same answer independently — so there is no ownership map, no id
exchange, no capacity estimate and no resize path. Every worker can write every
infoset from the start. The layout this replaced needed cross-worker agreement
and dropped a measured 39–74% of update samples.

## Modules

    static_parallel.py       the multi-process training loop
    run_tracker/             what a run records about itself

Storage lives one layer down in `src/engine/solver/storage/`:
`static_array.py` (the arrays) and `static_checkpoint.py` (their snapshots).

Two modules that used to live here now sit beside this package, because
training stopped being their only consumer: `src/pipeline/blueprint/construction.py`
(builders for abstraction, tree, storage and solver) and
`src/pipeline/abstraction/resolver.py` (find and load a precomputed combo
abstraction).

### `static_parallel.py`

`train_static_parallel()` runs a worker pool and returns a
`StaticTrainingResult`. What crosses a process boundary is **the config, a seed
and an iteration count** — notably not the tree and not the abstraction. Each
worker rebuilds the tree from config (a pure function, ~1s) and attaches to the
shared arrays by name, so no index information is ever shipped or reconciled.
Shipping the abstraction would mean pickling ~773 MB per worker.

Concurrency is Hogwild: workers write shared memory lock-free and races are
tolerated. That is the same convergence argument as the previous backend; only
the addressing changed.

`worker_seed(base_seed, worker_id, batch_id)` and `worker_iteration_indices()`
are the determinism seam — a run's sampling is reproducible from its seed.

### `run_tracker/`

`RunMetadata` (the schema), `AttemptRecord` and `ExperimentTag`
(`attempts.py`), and `RunTracker` (`tracker.py`, the lifecycle API). A run
directory carries `.run.json` and `progress.jsonl`.

`.run.json` pins `card_abstraction_hash` — exactly which abstraction the run
trained with. Evaluation refuses a run without it. The `attempts` list records
each start and resume as its own timeline entry, so wall-clock history survives
interruption.

## Storage and checkpoints

`StaticArrayStorage` is a pair of flat, ragged arrays indexed by arithmetic:

    regrets / strategy_sum   flat, length tree.num_slots
    reach / utility          flat, length tree.num_rows

    infoset (node n, bucket b) owns slots
        [slot_offset[n] + b*num_actions[n], ... + num_actions[n])

Ragged rather than a dense `(num_infosets, max_actions)` rectangle: at the
production tree's mean of ~2.6 actions against `max_actions=10`, dense would
waste roughly 4x the memory.

A checkpoint is those arrays plus a **16-byte tree fingerprint**, and nothing
else — the tree already says which infoset each row is. (Carrying that mapping
explicitly was ~83% of the old write cost.)

    <dir>/STATIC_CHECKPOINT.json     manifest: current + retained ladder
    <dir>/static-<iteration>.zarr    the arrays

The fingerprint is load-bearing, not defensive: a checkpoint carries no
self-describing row identity, so loading one against a different tree would not
fail — it would silently reinterpret every row as a different infoset and train
on scrambled regrets. Refusing the load is the only way that failure is visible.

The manifest is published with an atomic `Path.replace` after the arrays are
written, so a snapshot is either current or absent, never half-current. The
retained ladder keeps at most one snapshot per `retain_every` band, which is
what makes a within-run exploitability curve computable after the fact.

`STATIC_CHECKPOINT.json` is also how a run is identified as loadable at all.
Checkpoints from the deleted dynamic backend are unreadable at HEAD by design.

## Running one

Training runs **on the pool**, never on a laptop. `submit` queues a task;
`train-static` is what the node then executes.

```bash
poker-solver submit --config production --to 30000000
poker-solver train-static --config production --workers 16 --iterations 1000000
```

`--iterations` is an **absolute target** and `--run <id>` continues an existing
directory, so re-running past the target is a no-op. That is what makes a
scheduler retry converge instead of training twice; there is no separate
`resume`.

`--workers` is a pure throughput knob here. Unlike the old backend it does not
raise memory: the table is shared and there are no per-worker key maps.

Experiment bookkeeping goes through `--experiment` / `--arm` / `--parent`, with
`--set key=value` for config overrides.

## Configuration

`config/training/production.yaml` is the source of truth; unset fields fall
back to defaults in `src/shared/config/schema.py`. Knobs worth knowing:

```yaml
solver:
  iteration_weighting: dcfr
card_abstraction:
  config: production
training:
  num_iterations: 1000000
storage:
  zarr_compression_level: 1   # benchmarked: fastest AND smallest
```

## Testing

```bash
uv run pytest tests/pipeline/training/
```

Tests cover construction, config, checkpoint round-trips and metadata — not
full runs or MCCFR convergence, which are too slow for the suite. Mark anything
expensive `@pytest.mark.slow`.

**Related:** `src/pipeline/evaluation/README.md`,
`src/pipeline/abstraction/postflop/README.md`
