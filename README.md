# Poker Solver: Heads-Up No-Limit Hold'em

A research-grade Heads-Up No-Limit Texas Hold'em (HUNLHE) solver using Monte Carlo Counterfactual Regret Minimization (MCCFR) with self-play training.

## Features

- **Exact HUNLHE Rules**: Complete implementation of heads-up Texas Hold'em with proper betting rounds
- **Monte Carlo CFR**: Scalable MCCFR with outcome sampling for efficient convergence
- **Pluggable Abstractions**: Modular action and card abstraction systems
- **Equity-Based Bucketing**: Monte Carlo equity calculation with k-means clustering
- **Disk-Backed Storage**: Efficient HDF5 storage with LRU caching for 10M+ iterations
- **Comprehensive Evaluation**: Head-to-head matches and exploitability analysis

## Architecture

The solver consists of six main components:

1. **Game Engine** (`src/game/`): Exact HUNLHE rules implementation
2. **Abstraction Layer** (`src/abstraction/`): Action and card abstractions
3. **Solver** (`src/solver/`): MCCFR algorithm implementation
4. **Training** (`src/training/`): Training loop and checkpointing
5. **Evaluation** (`src/evaluation/`): Strategy evaluation and analysis
6. **Utilities** (`src/utils/`): Configuration, logging, and RNG management

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Clone the repository
git clone https://github.com/yourusername/poker-solver.git
cd poker-solver

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-extras
```

## Quick Start

```bash
# Precompute equity buckets (one-time setup)
uv run python scripts/precompute_buckets.py --config config/abstractions/card_equity_50.yaml

# Train the solver
uv run python scripts/train.py --config config/default.yaml --iterations 100000

# Evaluate checkpoints
uv run python scripts/evaluate.py --checkpoint1 data/checkpoints/iter_50000 --checkpoint2 data/checkpoints/iter_100000
```

## Configuration

All solver parameters are configured via YAML files in `config/`:

- `default.yaml`: Main configuration (game params, abstractions, training)
- `abstractions/`: Action and card abstraction definitions
- `training/`: Training-specific configs (iteration counts, checkpointing)

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

## Project Status

**Current Phase**: Phase 1 - Core Game Engine Implementation

- [x] Project setup with uv
- [ ] Game primitives (Card, Action, Street)
- [ ] GameState implementation
- [ ] Abstraction layer
- [ ] MCCFR solver
- [ ] Training infrastructure
- [ ] Evaluation harness
- [ ] Optimization and scaling

## Research Goals

This solver aims to:

1. Train toward approximate Nash equilibrium in HUNLHE
2. Demonstrate scalability to 10M+ iterations
3. Provide a modular platform for poker AI research
4. Support future enhancements (subgame solving, neural abstractions)

## Technical Approach

- **Algorithm**: Monte Carlo CFR with outcome sampling
- **Card Abstraction**: Equity-based bucketing (8-50 buckets per street)
- **Action Abstraction**: Configurable bet sizing (e.g., 33%, 75%, all-in)
- **Storage**: Hybrid memory/disk with HDF5 and LRU caching
- **Scale**: Designed for 100K-10M training iterations

## License

MIT License

## References

- Zinkevich et al. (2007): "Regret Minimization in Games with Incomplete Information"
- Lanctot et al. (2009): "Monte Carlo Sampling for Regret Minimization in Extensive Games"
- Moravčík et al. (2017): "DeepStack: Expert-level artificial intelligence in heads-up no-limit poker"
- Brown & Sandholm (2019): "Superhuman AI for multiplayer poker"
