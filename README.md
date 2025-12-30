# Poker Solver

A research-grade Monte Carlo CFR implementation for computing near-optimal (GTO) strategies in Heads-Up No-Limit Texas Hold'em.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This solver uses **Monte Carlo Counterfactual Regret Minimization (MCCFR)** with advanced optimizations (CFR+, Linear CFR) to compute equilibrium strategies for heads-up no-limit hold'em. It features sophisticated card and action abstractions, parallel training, and empirical exploitability evaluation.

### Key Features

- **Advanced CFR Variants**: CFR+ provides ~100x faster convergence than vanilla CFR, with Linear CFR adding an additional 2-3x speedup
- **Suit Isomorphism Card Abstraction**: Reduces state space by 12-19x while preserving strategic relevance (flush draws, suit coordination)
- **Configurable Action Abstraction**: Flexible bet sizing with street-specific sizing sets
- **Parallel Training**: Multi-core support with lock-free shared memory for efficient scaling
- **Comprehensive Evaluation**: Rollout-based exploitability estimation with confidence intervals
- **Production-Ready Checkpointing**: Efficient Zarr-based storage with resume capability
- **Interactive CLI & Web UI**: Train, evaluate, and visualize strategies through intuitive interfaces

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

```bash
# Launch the interactive CLI
uv run poker-solver
```

From the CLI, you can:
- **Train** a new solver with predefined configurations
- **Resume** training from checkpoints
- **Evaluate** trained strategies (exploitability estimation)
- **View** preflop GTO charts
- **Precompute** custom card abstractions

### Training Your First Solver

1. Launch the CLI: `uv run poker-solver`
2. Select "Train Solver"
3. Choose a configuration:
   - `quick_test`: Fast convergence test (~2 minutes, ~500 mbb/g)
   - `production`: Balanced quality (~2-3 hours, ~10-20 mbb/g)
4. Training runs with live progress updates and automatic checkpointing

## Architecture

### Card Abstraction: Suit Isomorphism

The solver uses **combo-level abstraction** that preserves suit relationships to the board:

- A♠K♠ on T♠9♠8♣ (flush draw) → Different bucket than
- A♠K♠ on T♥9♥8♣ (no flush draw)

This is a significant improvement over naive 169-class abstractions that ignore suit coordination.

**Process:**
1. **Canonicalize** boards by suit order (22,100 → 1,755 unique flops)
2. **Cluster** boards by texture (connectivity, pairing, suits)
3. **Bucket** hands within clusters by equity distributions
4. **Result**: 12-19x state space reduction with minimal strategic loss

See [Card Abstraction README](src/bucketing/postflop/README.md) for details.

## Configuration

Training behavior is controlled by YAML configs in `config/training/`:

```yaml
# config/training/production.yaml
game:
  starting_stack: 200  # BB units
  small_blind: 1
  big_blind: 2

action_abstraction:
  preflop_raises: [2.5, 3.5, 5.0]
  postflop:
    flop: [0.33, 0.66, 1.25]    # 1/3 pot, 2/3 pot, overbet
    turn: [0.50, 1.0, 1.5]
    river: [0.50, 1.0, 2.0]
  all_in_spr_threshold: 2.0
  max_raises_per_street: 5

solver:
  cfr_plus: true         # 100x faster convergence
  linear_cfr: true       # Additional 2-3x speedup
  sampling_method: "external"  # or "outcome"

training:
  num_iterations: 1000000  # 1M iterations
  checkpoint_frequency: 100000
```

Card abstraction configs live in `config/abstraction/`:

```yaml
# config/abstraction/default.yaml
board_clusters:
  flop: 50
  turn: 100
  river: 200

buckets:
  flop: 50
  turn: 100
  river: 200

equity_samples: 1000
```

See [Configuration Guide](docs/CONFIGURATION.md) for details on adding custom configs.

## Evaluation Metrics

### Exploitability

The primary quality metric is **exploitability**: the expected value a best-response opponent can achieve.

**Target Values** (in milli-big-blinds per game):
- `< 1 mbb/g`: Strong player
- `1-5 mbb/g`: Good player
- `5-20 mbb/g`: Decent player
- `20+ mbb/g`: Needs more training

**Implementation**: Monte Carlo rollout-based best response approximation (following Brown & Sandholm 2019). This is tractable for large games but provides empirical estimates rather than exact exploitability.

```python
results = compute_exploitability(
    solver,
    num_samples=10000,          # Game simulations per player
    num_rollouts_per_infoset=100,  # Rollouts for action value estimation
    use_average_strategy=True
)

# Output includes confidence intervals
print(f"{results['exploitability_mbb']:.2f} ± {results['std_error_mbb']:.2f} mbb/g")
print(f"95% CI: [{results['confidence_95_mbb'][0]:.2f}, {results['confidence_95_mbb'][1]:.2f}]")
```

See [Evaluation README](src/evaluation/README.md) for methodology and best practices.

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
```

### Chart Viewer Backend

The preflop chart viewer now serves data through FastAPI.

- FastAPI server (`/health`, `/api/meta`, `/api/chart`) + static UI from `ui/dist`

### Project Structure

```
poker-solver/
├── src/
│   ├── solver/          # MCCFR implementation
│   ├── training/        # Training loop, parallel workers, checkpointing
│   ├── evaluation/      # Exploitability, head-to-head evaluation
│   ├── bucketing/       # Card abstraction (preflop & postflop)
│   ├── actions/         # Action abstraction, bet sizing
│   ├── game/            # Game state, rules, hand evaluation
│   ├── cli/             # Interactive CLI application
│   └── utils/           # Configuration, helpers
├── tests/               # pytest test suite
├── config/
│   ├── training/        # Training configuration presets
│   └── abstraction/     # Card abstraction presets
├── data/
│   ├── runs/            # Training runs and checkpoints
│   └── combo_abstraction/  # Precomputed card abstractions
├── ui/                  # React web interface for charts
└── docs/                # Additional documentation
```

## License

MIT License - see LICENSE file for details.

---

**Note**: This is a research implementation for educational purposes. While the solver computes theoretically sound strategies, it should not be used for real-money poker without extensive additional testing and validation.
