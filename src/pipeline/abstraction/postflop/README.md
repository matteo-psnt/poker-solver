# Card + Board Abstraction (Suit Isomorphism, Full Coverage)

This module implements **combo-level abstraction** for postflop poker using
**suit isomorphism** with **full per-board coverage**. It's the foundation for
tractable CFR training on Texas Hold'em.

## Overview

### The Problem

The number of raw postflop states is astronomically large (billions of
(hand, board) pairs). Storing strategies for every state is infeasible, so
similar states are grouped into a small number of **buckets** per street;
training learns one strategy per (bucket, betting line).

### The Solution

Two reductions are applied, both computed exactly:

1. **Suit isomorphism** collapses strategically identical states:
   A♠K♠ on T♠9♠8♣ ≡ A♥K♥ on T♥9♥8♣ (same flush draw, blockers, equity), but
   ≠ A♥K♥ on T♠9♠8♣ (no flush draw). This shrinks boards ~12-19x
   (1,755 canonical flops, 16,432 turns, 134,459 rivers) and hands on each
   board into a few hundred classes.
2. **Equity bucketing** groups hand classes with similar exact equity into
   the configured number of buckets per street (weighted 1D k-means).

Every canonical board on every street is covered — equity is computed on the
board itself by the exact range-vs-range engine
(`src/pipeline/abstraction/utils/equity.py`), never approximated from a
"similar" board. There is **no fallback path**: every legal (hand, board)
lookup resolves; an unresolvable lookup is a hard error.

## Pipeline

```
Precompute (per street)                       Runtime lookup
─────────────────────────                     ──────────────────────────
1. Enumerate canonical boards                 1. Canonicalize (hand, board)
2. Exact equity for every hand                2. Binary-search board row
   class on every board                       3. Index hand column
3. Weighted 1D k-means → buckets              4. Read bucket from matrix
4. Dense matrices + metadata
```

## Modules

- `suit_isomorphism.py` — canonicalization of boards/hands, canonical IDs
- `board_enumeration.py` — enumeration of all canonical boards per street
- `canonical_hands.py` — canonical hand classes per board (with multiplicity)
- `precompute.py` — `PostflopPrecomputer`: the offline pipeline
- `bucketer.py` — `DenseBucketer`: runtime lookup over dense matrices
- `quality.py` — abstraction quality metrics (variance explained, etc.)

## Storage Format

```
data/combo_abstraction/buckets-F50T100R200-rexact-{hash}/
├── metadata.json          # config, per-street stats + quality metrics
├── hand_id_to_col.npy     # canonical hand ID → matrix column (static)
├── flop_board_ids.npy     # sorted canonical board IDs (one row each)
├── flop_buckets.npy       # [n_boards, 1326] uint8/uint16 bucket matrix
├── turn_board_ids.npy / turn_buckets.npy
└── river_board_ids.npy / river_buckets.npy
```

Bucket matrices are loaded with `mmap_mode="r"`, so artifacts cost RAM only
for the pages actually touched. Cells for hand classes that can't exist on a
board hold the dtype's max value as a sentinel.

## Configuration (`config/abstraction/`)

```yaml
buckets:            # equity buckets per street
  flop: 50
  turn: 100
  river: 200
flop_runouts: null  # null = exact (all 1,176 runouts); turn/river always exact
kmeans_max_iter: 300
kmeans_n_init: 10
num_workers: null
seed: 42
```

## Usage

```python
from pathlib import Path
from src.core.game.state import Card, Street
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer

# Precompute (also available via CLI: "Combo Abstraction Tools")
config = PrecomputeConfig.from_yaml("default")
precomputer = PostflopPrecomputer(config)
precomputer.precompute_all()
precomputer.save(Path("data/combo_abstraction/my_abstraction"))

# Load and look up
abstraction = PostflopPrecomputer.load(Path("data/combo_abstraction/my_abstraction"))
bucket = abstraction.get_bucket(
    hole_cards=(Card.new("As"), Card.new("Ks")),
    board=(Card.new("Th"), Card.new("9h"), Card.new("8c")),
    street=Street.FLOP,
)
```

## Quality Metrics

Computed exactly at precompute time (combo-weighted) and stored in
`metadata.json`; view via CLI → "View Abstraction Quality":

- **equity_std** — equity spread available on the street
- **within_bucket_std** — equity spread forced to share a strategy
- **variance_explained** — share of equity variance the buckets preserve

Use these to choose bucket counts with evidence instead of guessing.

## Performance

| Street | Canonical boards | Precompute (12 cores, exact) |
|--------|------------------|------------------------------|
| Flop   | 1,755            | ~3 min (seconds with `flop_runouts: 200`) |
| Turn   | 16,432           | ~1.5 min |
| River  | 134,459          | ~2 min (dominated by board enumeration) |

Runtime lookup is two array indexes after canonicalization (~µs), memoized
across CFR traversals.

## References

- **Suit Isomorphism**: Waugh et al., "A Practical Use of Imperfect Recall" (2009)
- **Card Abstraction**: Johanson et al., "Evaluating State-Space Abstractions in Extensive-Form Games" (2013)
- **CFR+**: Tammelin et al., "Solving Large Imperfect Information Games Using CFR+" (2014)
