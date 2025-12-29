# Card + Board Abstraction (Suit Isomorphism)

This module implements **combo-level abstraction** for postflop poker using **suit isomorphism**. It's the foundation for tractable CFR training on Texas Hold'em.

## Overview

### The Problem

In Texas Hold'em, the number of possible game states is astronomically large:
- **Flop**: 22,100 possible boards × 1,176 hole card combos = ~26 million states
- **Turn**: 270,725 boards × 1,128 hole card combos = ~305 million states
- **River**: 2,598,960 boards × 1,081 hole card combos = ~2.8 billion states

Storing strategies for every state is infeasible. We need **abstraction** - grouping similar states together.

### The Solution: Suit Isomorphism

The key insight is that **suits are strategically interchangeable**. For example:
- A♠K♠ on T♠9♠8♣ (spade flush draw)
- A♥K♥ on T♥9♥8♣ (heart flush draw)

These are **strategically identical** - both have the same flush draw potential, same blockers, same equity. We can treat them as the same state.

However, A♠K♠ on T♠9♠8♣ is **different** from A♥K♥ on T♠9♠8♣ - the first has a flush draw, the second doesn't.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRECOMPUTATION PIPELINE                      │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Canonical   │───▶│    Board     │───▶│   Equity     │       │
│  │   Boards     │    │  Clustering  │    │ Calculation  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│        │                   │                    │               │
│        ▼                   ▼                    ▼               │
│  1,755 flops         50-400 clusters     Monte Carlo            │
│  16,432 turns        per street          equity samples         │
│  ~135k rivers                                                   │
│                                                                 │
│                           ┌──────────────┐                      │
│                           │   K-Means    │                      │
│                           │  Bucketing   │                      │
│                           └──────────────┘                      │
│                                 │                               │
│                                 ▼                               │
│                      Bucket assignments saved                   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RUNTIME LOOKUP                             │
│                                                                 │
│  Input: (A♠K♠, [T♠9♥8♣])                                        │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Canonicalize │───▶│   Predict    │───▶│   Lookup     │       │
│  │  (hand,board)│    │   Cluster    │    │   Bucket     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│           │                 │                   │               │
│           ▼                 ▼                   ▼               │
│     (A₀K₀, [T₀9₁8₂])   cluster_id=42      bucket_id=37          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

### 1. `suit_canonicalization.py` - Core Canonicalization

Converts concrete cards to canonical form by assigning suit labels in order of first appearance.

```python
# Example: Board establishes suit mapping
Board: [T♠ 9♥ 8♠]  →  Canonical: [T₀ 9₁ 8₀]
                       Mapping: {♠→0, ♥→1}

# Hands use board's mapping, extending if needed
Hand: [A♠ K♠]  →  [A₀ K₀]  (uses existing ♠→0)
Hand: [A♥ K♥]  →  [A₁ K₁]  (uses existing ♥→1)
Hand: [A♦ K♦]  →  [A₂ K₂]  (new suit, assigned 2)
```

**Key Types:**
- `CanonicalCard`: A card with rank index and canonical suit label
- `SuitMapping`: Tracks real suit → canonical label mapping
- `canonicalize_board()`: Establishes mapping from board
- `canonicalize_hand()`: Converts hand using board's mapping

### 2. `canonical_boards.py` - Board Enumeration

Enumerates all unique canonical boards per street.

| Street | Raw Boards | Canonical Boards | Reduction |
|--------|------------|------------------|-----------|
| Flop   | 22,100     | ~1,755           | 12.6x     |
| Turn   | 270,725    | ~16,432          | 16.5x     |
| River  | 2,598,960  | ~134,459         | 19.3x     |

**Key Class:**
- `CanonicalBoardEnumerator`: Generates and caches all canonical boards

### 3. `board_clustering.py` - Public State Abstraction

Clusters canonical boards by strategic texture to further reduce computation.

**Features extracted for each board:**
- **Suit distribution**: Monotone, two-tone, rainbow indicators
- **Rank pairing**: Paired, trips, two-pair, quads
- **Connectivity**: Gaps between cards, straight potential
- **High card strength**: Normalized rank values

**Key Class:**
- `BoardClusterer`: K-means clustering on board texture features

```python
# Example clusters (conceptual)
Cluster 0: Monotone, connected boards (flush + straight draws)
Cluster 1: Rainbow, paired boards (trips potential)
Cluster 2: Two-tone, unconnected (moderate texture)
...
```

### 4. `combo_abstraction.py` - Main Abstraction Class

The `PostflopBucketer` class provides the runtime interface:

```python
# Get bucket for a hand+board combination
bucket = abstraction.get_bucket(
    hole_cards=(Card.new("As"), Card.new("Ks")),
    board=(Card.new("Th"), Card.new("9h"), Card.new("8c")),
    street=Street.FLOP
)
```

**Pipeline:**
1. Canonicalize (hand, board) pair
2. Predict board cluster using trained K-means
3. Lookup bucket from `{cluster_id, hand_id} → bucket_id` table

**Storage Format (sparse):**
```python
{
    Street.FLOP: {
        cluster_0: {hand_id_1: bucket, hand_id_2: bucket, ...},
        cluster_1: {...},
        ...
    },
    Street.TURN: {...},
    Street.RIVER: {...}
}
```

### 5. `precompute.py` - Precomputation Pipeline

Handles the one-time computation of bucket assignments.

**Steps:**
1. **Enumerate** all canonical boards for each street
2. **Cluster** boards by texture (K-means on features)
3. **Select representatives** from each cluster
4. **Compute equity** for each (representative_board, hand) pair
5. **Bucket hands** using K-means on equity values
6. **Save** abstraction to disk

**Configuration (`config/abstraction/`):**
```yaml
# Example: default.yaml
board_clusters:
  flop: 50
  turn: 100
  river: 200

buckets:
  flop: 50
  turn: 100
  river: 200

equity_samples: 1000
representatives_per_cluster: 1
```

## Usage

### Precompute Abstraction

```bash
# Via CLI
uv run poker-solver
# Select "Combo Abstraction Tools" -> "Precompute Abstraction"

# Or directly
from src.bucketing.postflop.precompute import PostflopPrecomputer, PrecomputeConfig

config = PrecomputeConfig.from_yaml("default")
precomputer = PostflopPrecomputer(config)
precomputer.precompute_all()
precomputer.save(Path("data/combo_abstraction/my_abstraction"))
```

### Load and Use

```python
from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.game.state import Card, Street

# Load precomputed abstraction
abstraction = PostflopPrecomputer.load(Path("data/combo_abstraction/my_abstraction"))

# Get bucket for a game state
bucket = abstraction.get_bucket(
    hole_cards=(Card.new("As"), Card.new("Ks")),
    board=(Card.new("Th"), Card.new("9h"), Card.new("8c")),
    street=Street.FLOP
)
print(f"This hand is in bucket {bucket}")
```

## Key Design Decisions

### 1. Board Clustering BEFORE Equity Calculation

We cluster boards by texture features (not equity), then only compute equity for representative boards. This is critical for tractability:

| Approach | Computation |
|----------|-------------|
| Naive | 1,755 boards × 1,081 hands × 1,000 samples = 1.9B equity calcs |
| With clustering (50 clusters, 1 rep) | 50 × 1,081 × 1,000 = 54M equity calcs (35x reduction) |

### 2. Fallback Mechanism

Not every (cluster, hand) combination is seen during precomputation. When a lookup misses, we fall back to the **median bucket** for that cluster. This provides reasonable behavior while logging warnings for coverage analysis.

### 3. Sparse Storage

Only (cluster_id, hand_id) pairs actually seen during precomputation are stored. This saves memory compared to dense arrays.

## File Structure

```
src/abstraction/isomorphism/
├── __init__.py
├── suit_canonicalization.py  # Core canonicalization logic
├── canonical_boards.py       # Board enumeration
├── board_clustering.py       # K-means on board textures
├── combo_abstraction.py      # Main abstraction class
├── precompute.py             # Precomputation pipeline
└── README.md                 # This file

config/abstraction/
├── fast_test.yaml            # Quick testing (10/20/30 buckets)
├── default.yaml              # Balanced (50/100/200 buckets)
├── production.yaml           # High quality (100/300/600 buckets)
└── README.md

data/combo_abstraction/
├── buckets-F50T100R200-C50C100C200-s1000-{hash}/
│   ├── combo_abstraction.pkl  # Pickled PostflopBucketer
│   └── metadata.json          # Config and statistics
└── ...
```

## Performance Characteristics

### Precomputation Time

| Config | Board Clusters | Equity Samples | Time Estimate |
|--------|---------------|----------------|---------------|
| fast_test | 10/20/30 | 100 | ~5 minutes |
| default | 50/100/200 | 1,000 | ~20 minutes |
| production | 100/200/400 | 2,000 | ~2 hours |

### Runtime Lookup

- **O(1)** dictionary lookups after canonicalization
- Canonicalization: ~1μs per hand
- Total lookup: ~5-10μs per hand

### Training Results

From a 100-iteration test run with `fast_test` abstraction:
- **425,012 infosets** created
- **16.9% fallback rate** (hands not in precomputed coverage)
- **0.07 iterations/second** (limited by CFR traversal, not abstraction)

## Comparison: 169-Class vs Combo-Level

| Aspect | 169-Class | Combo-Level |
|--------|-----------|-------------|
| Preflop | Correct (169 classes) | Same |
| Postflop | Wrong (ignores suit relation to board) | Correct (preserves flush draws) |
| Storage | Smaller | Larger but sparse |
| AKs on Ts9s8c | Same bucket as AKs on Tc9c8s | Different buckets (flush vs no flush) |

## References

- **Suit Isomorphism**: Waugh et al., "A Practical Use of Imperfect Recall" (2009)
- **Card Abstraction**: Johanson et al., "Evaluating State-Space Abstractions in Extensive-Form Games" (2013)
- **CFR+**: Tammelin et al., "Solving Large Imperfect Information Games Using CFR+" (2014)
