# Evaluation Module

This module provides tools for evaluating poker solver performance, with emphasis on empirical exploitability estimation.

## What is Exploitability?

Exploitability measures how much an optimal opponent (best response) can gain against your strategy. It's the gold standard metric for evaluating CFR convergence in poker.

**Definition**: `Exploitability = (BR₀ + BR₁) / 2`

Where BR_i is the expected utility when player i plays best response against the strategy.

**Target Values (in milli-big-blinds per game):**
- `< 0.1 mbb/g`: Near-optimal (professional GTO play)
- `0.1-1 mbb/g`: Strong player
- `1-5 mbb/g`: Good player
- `5-20 mbb/g`: Decent player
- `20-100 mbb/g`: Weak player
- `100+ mbb/g`: Very exploitable

## Implementation: Rollout-Based Approximation

### Why Not Exact Best Response?

Exact BR computation requires:
- Full game tree traversal (millions+ nodes in No-Limit Hold'em)
- Expectation over ALL chance outcomes (no sampling allowed)
- Tracking belief distributions over opponent hands
- Dynamic programming over abstract game states

This is **computationally infeasible** for production poker games.

### Our Approach: Monte Carlo Rollout Sampling

Following modern poker research (Johanson et al. 2013, Brown & Sandholm 2019), we use rollout-based approximation:

1. **Freeze** the solver's average strategy σ
2. **For each player** as potential exploiter:
   - Simulate N complete games from random starting states
   - At exploiter's decision points:
     * Estimate action values via K Monte Carlo rollouts
     * Choose greedily (best estimated action)
   - At opponent's decision points:
     * Sample actions from frozen strategy σ
   - Record terminal utilities
3. **Compute** empirical mean and confidence intervals

**Key Properties:**
- ✅ Does NOT sample chance outcomes within BR (samples complete games)
- ✅ Scales to large games (linear in samples, not tree size)
- ✅ Provides confidence intervals (it's an empirical estimate)
- ✅ Correct in expectation with sufficient rollout budget
- ❌ NOT exact exploitability (it's an approximation)

## Functions

### `compute_exploitability(solver, num_samples, use_average_strategy, num_rollouts_per_infoset, seed)`

Computes empirical exploitability estimate via Monte Carlo rollout sampling.

**Parameters:**
- `solver`: Trained MCCFR solver
- `num_samples`: Number of game simulations per player (default: 10000)
- `use_average_strategy`: Use average strategy (True) or current (False)
- `num_rollouts_per_infoset`: Rollouts for action value estimation (default: 100)
- `seed`: Random seed for reproducibility

**Returns:**
```python
{
    'exploitability_mbb': float,      # Primary metric (milli-BB per game)
    'exploitability_bb': float,       # In big blinds per game
    'player_0_br_utility': float,     # P0's BR utility (chips)
    'player_1_br_utility': float,     # P1's BR utility (chips)
    'std_error_mbb': float,           # Standard error of estimate
    'confidence_95_mbb': (float, float),  # 95% confidence interval
    'num_samples': int,               # Samples used
}
```

**Example:**
```python
from src.pipeline.evaluation.exploitability import compute_exploitability

results = compute_exploitability(
    solver,
    num_samples=10000,
    num_rollouts_per_infoset=100,
    seed=42
)

print(f"Exploitability: {results['exploitability_mbb']:.2f} ± {results['std_error_mbb']:.2f} mbb/g")
print(f"95% CI: [{results['confidence_95_mbb'][0]:.2f}, {results['confidence_95_mbb'][1]:.2f}]")
```

### `compute_total_positive_regret(solver)`

Computes total positive regret across all information sets.

**IMPORTANT:** This is a training diagnostic, NOT a quality metric:
- ❌ NOT comparable across different abstractions
- ❌ NOT interpretable in big-blind terms
- ✅ Useful for monitoring convergence (same abstraction)
- ✅ Should decrease during training

**Returns:**
```python
{
    'total_positive_regret': float,
    'num_infosets': int,
    'avg_regret_per_infoset': float,
}
```

## Usage Guidelines

### For Research / Publication

**ALWAYS report:**
1. Confidence intervals, not just point estimates
2. Number of samples used
3. Number of rollouts per infoset
4. That this is an *empirical estimate*, not exact exploitability

**Example statement:**
> "We estimate exploitability at 12.3 ± 1.8 mbb/g (95% CI: [8.7, 15.9], N=10000 samples, 100 rollouts/infoset)."

### For Development

**Quick checks** (low accuracy, fast):
```python
results = compute_exploitability(solver, num_samples=100, num_rollouts_per_infoset=20)
```

**Production evaluation** (high accuracy, slow):
```python
results = compute_exploitability(solver, num_samples=10000, num_rollouts_per_infoset=200)
```

### Reducing Variance

Standard error decreases with √N:
- 100 samples: ±X mbb/g
- 400 samples: ±X/2 mbb/g
- 10000 samples: ±X/10 mbb/g

### Improving BR Quality

More rollouts → better action value estimates → better BR approximation:
- 10 rollouts: Very rough BR
- 50 rollouts: Reasonable BR
- 100 rollouts: Good BR (recommended)
- 200+ rollouts: Diminishing returns

## Performance Notes

Rollout-based approach is **much faster** than exact tree traversal:
- Scales linearly with num_samples
- Each sample is a single game simulation
- Typical: 10-100ms per sample (vs. minutes for exact)

For 10000 samples at 50ms each: ~8 minutes total

## Best Practices

1. **For Quick Testing:**
   - `num_samples=100`, `num_rollouts_per_infoset=20`
   - Iterate quickly during development

2. **For Final Evaluation:**
   - `num_samples=10000+`, `num_rollouts_per_infoset=100+`
   - Report confidence intervals
   - Use multiple random seeds

3. **For Training Monitoring:**
   - Use `compute_total_positive_regret()` for fast convergence checks
   - Run full exploitability evaluation at checkpoints only

4. **For Publications:**
   - Always state this is "empirical exploitability estimate"
   - Report all parameters (N, K, seed)
   - Include confidence intervals
   - Compare only strategies with same abstraction

## Example Usage

```python
from src.pipeline.evaluation.exploitability import (
    compute_exploitability,
    compute_total_positive_regret
)

# After training
solver = trainer.solver

# Quick training diagnostic (fast, < 1 second)
regret_stats = compute_total_positive_regret(solver)
print(f"Total positive regret: {regret_stats['total_positive_regret']:.2e}")
print(f"Avg per infoset: {regret_stats['avg_regret_per_infoset']:.2e}")

# Full exploitability evaluation (slower, ~5-10 minutes)
results = compute_exploitability(
    solver,
    num_samples=10000,
    num_rollouts_per_infoset=100,
    use_average_strategy=True,
    seed=42
)

print(f"\nExploitability: {results['exploitability_mbb']:.2f} ± {results['std_error_mbb']:.2f} mbb/g")
print(f"95% CI: [{results['confidence_95_mbb'][0]:.2f}, {results['confidence_95_mbb'][1]:.2f}]")

# Interpret
exp = results['exploitability_mbb']
if exp < 1.0:
    print("✅ Strong solution (< 1 mbb/g)")
elif exp < 5.0:
    print("✅ Good solution (< 5 mbb/g)")
elif exp < 20.0:
    print("⚠️  Decent solution (< 20 mbb/g)")
else:
    print("❌ Needs more training")
```

## Implementation Details

### Rollout-Based Best Response

The implementation uses **rollout simulation** instead of exact tree traversal:

**At BR player's decision points:**
```python
for action in legal_actions:
    # Estimate action value via K rollouts
    value = average([simulate_game(action) for _ in range(K)])

return max(values)  # Choose best
```

**At opponent's decision points:**
```python
strategy = solver.get_strategy(infoset)
action = sample(strategy)  # Sample from frozen strategy
return simulate_game(action)
```

This is **fundamentally different** from exact BR but is tractable for large games.

### Why This Works

- Each rollout provides an unbiased estimate of action value
- Averaging K rollouts reduces variance
- Greedy selection based on estimates approximates max EV
- In expectation over many samples, converges to true exploitability

## Testing

Tests in [test_exploitability_rollout.py](../../tests/evaluation/test_exploitability_rollout.py).

```bash
uv run pytest tests/evaluation/test_exploitability_rollout.py -v
```

## References

1. Johanson et al. "Evaluating State-Space Abstractions in Extensive-Form Games" (AAMAS 2013)
2. Brown & Sandholm "Solving Imperfect-Information Games via Discounted Regret Minimization" (AAAI 2019)
3. Bowling et al. "Heads-up Limit Hold'em Poker is Solved" (Science 2015)

## Future Enhancements

Potential improvements:
- Parallel sampling across multiple cores
- Adaptive rollout budgets based on game phase
- Importance sampling for rare situations
- Caching for repeated public states
