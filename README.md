# Poker Solver

A research-grade Monte Carlo CFR implementation for computing near-optimal (GTO) strategies in Heads-Up No-Limit Texas Hold'em.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This solver uses **Monte Carlo Counterfactual Regret Minimization (MCCFR)** with modern regret-weighting schemes (CFR+, Linear CFR, Discounted CFR) to compute equilibrium strategies for heads-up no-limit hold'em. It features exact suit-isomorphism card abstraction, hogwild parallel training over shared memory, and rigorous exploitability evaluation via Local Best Response.

### Key Features

- **Advanced CFR Variants**: configurable iteration weighting (`none` | `linear` | `dcfr`) with optional CFR+ regret flooring (off by default — the floor hurts under Monte Carlo sampling); production training uses Discounted CFR
- **Suit Isomorphism Card Abstraction**: Exact, full-coverage combo-level abstraction that preserves suit relationships (flush draws, blockers) with no fallback path
- **Node-Template Action Model**: Context-aware preflop/postflop sizing with SPR-gated jam logic
- **Realtime Subgame Resolver**: Runtime local re-solving with configurable depth, rollout leaves, and conservative blueprint blending
- **Parallel Training**: Hash-partitioned shared memory with owner-only writes — lock-free, no merge step
- **Rigorous Evaluation**: Local Best Response (LBR) exploitability lower bounds with confidence intervals, recorded to an append-only eval ledger with paired run comparison
- **Production-Ready Checkpointing**: Async Zarr-based snapshots with resume capability
- **Reproducibility**: Runs record git provenance, config hashes, and the exact card-abstraction hash; evaluation auto-pins to it
- **Cloud-first execution**: training and evaluation run on an autoscale-to-zero Azure Batch pool, dispatched from Python (`src/interfaces/cloud/`). The laptop submits and reads JSON; it does not train.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/matteo-psnt/poker-solver.git
cd poker-solver

# Install dependencies with uv
uv sync --group dev
```

### Basic Usage

There are two surfaces, split by who is asking.

**`poker-solver` is the scriptable one** — what a cloud job, a shell and an
AI agent drive. It is the complete surface; everything the project can do is a
flag-driven subcommand of it:

```bash
uv run poker-solver submit --config production --to 25000000   # queue a leg
uv run poker-solver jobs                                       # what is running
uv run poker-solver score --run <id> --at 10000000,20000000    # one task per rung
uv run poker-solver ledger                                     # every evaluation, from the share
uv run poker-solver compare --a <run> --b <run>                # paired comparison with p-value
```

**The web console is the one a human reads** — `just console`, then
http://127.0.0.1:8765. It does not reimplement anything: every endpoint is a
single `Command.invoke` against the list above, so the two cannot disagree.

```bash
just console       # build, then serve
just console-dev   # Vite on :5173 with hot reload, proxying /api to :8765
```

### Training Your First Solver

1. `just create` once, then `just cli push-data` to publish the card abstractions.
2. `uv run poker-solver submit --config quick_test --to 3000`
3. Watch with `poker-solver jobs`; read the leg with
   `poker-solver logs --task <id>`.
4. `poker-solver ledger` shows the evaluations. It reads the share and
   derives the index on every call — there is nothing to fetch and no local copy.

The iteration target is **absolute**: re-submitting the same run with the same
number is a no-op, which is what makes a retry converge instead of training
twice. Continuing a run is `submit --run <id> --to <larger-number>`; there is no
separate resume.

`train-static`, `precompute` and `evaluate` still exist as local subcommands —
they are what a node invokes, and they remain useful for seconds-long probes.

## Architecture

The codebase follows a layered package layout with a single allowed dependency direction, enforced by import-linter:

- `src/interfaces` -> `src/pipeline` -> `src/engine` -> `src/core`
- `src/shared` is layer-neutral and can be imported by all layers
- Reverse imports across layers are forbidden

### Card Abstraction: Suit Isomorphism

The solver uses **combo-level abstraction** that preserves suit relationships to the board:

- A♠K♠ on T♠9♠8♣ (flush draw) → Different bucket than
- A♠K♠ on T♥9♥8♣ (no flush draw)

This is a significant improvement over naive 169-class abstractions that ignore suit coordination.

**Process (per street, computed exactly):**
1. **Canonicalize** boards and hands by suit isomorphism (22,100 → 1,755 unique flops)
2. **Compute exact equity** for every canonical hand class on every canonical board
3. **Bucket** hand classes by equity (weighted 1D k-means) into the configured count per street
4. **Result**: 12-19x state space reduction, full per-board coverage, no fallback path

See [Card Abstraction README](src/pipeline/abstraction/postflop/README.md) for details.

### Realtime Resolver (Runtime Search)

Decision-time play is handled by `BlueprintAgent` (`src/engine/search/agent.py`), which wraps any `Blueprint` (a protocol the MCCFR solver satisfies). When the resolver is enabled, `act()`:

- Builds a depth-limited local lookahead tree from the current state
- Estimates leaf values via blueprint rollouts
- Computes a local strategy and blends it with the blueprint policy (`policy_blend_alpha`)
- Applies a minimum strategy floor (`min_strategy_prob`) before normalization

Training still learns the blueprint policy; resolving happens only at decision time. Off-tree opponent actions are handled by the action model's `off_tree_mapping`.

## Configuration

Training configs in `config/training/` are sparse overrides on the schema defaults in `src/shared/config/schema.py` (`config/training/default.yaml` documents every default). The actual production config:

```yaml
# config/training/production.yaml
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

storage:
  initial_capacity: 4000000

system:
  config_name: "production"
```

Resolver defaults (`ResolverConfig`): `enabled: true`, `time_budget_ms: 300`, `max_depth: 2`, `leaf_rollouts: 8`, `policy_blend_alpha: 0.35`, `min_strategy_prob: 1.0e-6`.

Card abstraction configs live in `config/abstraction/`:

```yaml
# config/abstraction/default.yaml
buckets:            # equity buckets per street
  flop: 50
  turn: 100
  river: 200
flop_runouts: null  # null = exact (all 1,176 runouts)
equity_histogram_bins: 8
kmeans_max_iter: 300
kmeans_n_init: 10
num_workers: null
seed: 42
```

`config/training/default.yaml` documents every available setting and its default; new configs only need the keys they override. See the [Configuration Guide](config/README.md) for the full schema reference and workflows.

## Evaluation

The primary quality metric is **exploitability**, measured with **Local Best Response (LBR)**: an exploiter plays against the frozen blueprint and its winnings (in mbb/g) form a *lower bound* on true exploitability. Key knobs define the comparison tier:

- `--scorer lookahead` — depth-limited best-response scoring (the standard for on-tree evaluation; the default `myopic` scorer understates exploitability)
- `--opponent blueprint|deployed` — raw strategy table vs. blueprint + runtime resolver
- `--include-off-tree` — allow the exploiter off the trained action tree (shadow-state translation)

Every evaluation is written as its own document under `<run_dir>/evals/`, with git provenance and the pinned abstraction hash. There is no stored index: `poker-solver ledger` DERIVES one from the published documents on every read, which is what makes evaluating from several boxes at once safe. `poker-solver compare` runs a paired statistical comparison and refuses mismatched seeds or tiers.

An older rollout-based estimator (`compute_exploitability`) is retained as a fast smoke test only — it measures a one-ply deviation gain and is not a trustworthy exploitability figure.

See [Evaluation README](src/pipeline/evaluation/README.md) for methodology and best practices.

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run fast tests only (excludes slow integration tests)
uv run pytest -m "not slow"
```

### Code Quality

```bash
# Linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run ty check

# Layering/architecture contracts
uv run lint-imports
```

### Project Structure

```
poker-solver/
├── src/
│   ├── interfaces/      # commands/ (every subcommand) + cli/ and web/ that render them + cloud/ dispatch
│   ├── pipeline/        # Training, evaluation, abstraction workflows
│   ├── engine/          # Solver/search internals
│   ├── core/            # Poker domain foundations (game/actions)
│   └── shared/          # Cross-layer utilities (config/, cloudtask/ node wrapper, cache)
├── console/             # The web console (Vite + React); its own npm toolchain
├── tests/               # Mirrors src/ layout + integration tests
├── config/
│   ├── training/        # Training configuration presets
│   └── abstraction/     # Card abstraction presets
├── infra/               # Azure Batch substrate (Terraform) + run_task.py, the node's entry point
└── justfile             # Terraform lifecycle, `panic`, `credit-check`
```

**There is no `data/` directory and nothing recreates one.** Runs live on the
share; regenerable caches resolve through `src/shared/cache.py` to
`$POKER_SOLVER_CACHE`, else `$XDG_CACHE_HOME`, else `~/.cache/poker-solver`.
A `data/` path inside `src/` fails `tests/shared/test_cache.py` — which is how
the directory came back after each of the two previous prunes.

## License

MIT License - see LICENSE file for details.

---

**Note**: This is a research implementation for educational purposes. While the solver computes theoretically sound strategies, it should not be used for real-money poker without extensive additional testing and validation.
