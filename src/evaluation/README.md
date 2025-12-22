# Exploitability Computation

This module implements exploitability computation for evaluating the quality of CFR-trained poker strategies.

## What is Exploitability?

Exploitability measures how much profit an optimal opponent (best response) can make against your strategy. It's the gold standard metric for evaluating poker solvers.

**Target Values (in milli-big-blinds per game):**
- `< 0.1 mbb/g`: Near-optimal (professional GTO play)
- `0.1-1 mbb/g`: Strong player
- `1-5 mbb/g`: Good player
- `5-20 mbb/g`: Decent player
- `20-100 mbb/g`: Weak player
- `100+ mbb/g`: Very exploitable

## Functions

### `compute_exploitability(solver, num_samples, use_average_strategy)`

Computes approximate exploitability via Monte Carlo sampling.

**Algorithm:**
1. Sample random hands (dealing cards)
2. For each player, compute their best response utility against the solver's strategy
3. Exploitability = average of both players' BR utilities

**Returns:**
```python
{
    'exploitability': float,  # mbb/g
    'player_0_br_utility': float,
    'player_1_br_utility': float,
    'nash_utility': 0.0
}
```

### `compute_nash_conv(solver)`

Computes Nash Convergence - the sum of positive regrets across all information sets.

Simpler than exploitability but less interpretable. Useful for tracking convergence during training.

## Performance Considerations

**WARNING:** Exploitability computation is VERY expensive for Heads-Up No-Limit Hold'em!

- A single sample requires traversing the entire game tree
- Can make **10M+ recursive calls** per sample
- With default action/card abstraction, one sample can take **minutes to hours**

### Current Performance

With standard configuration (200BB stacks, 3 bet sizes per street):
- **1 sample**: ~10-15 million function calls, 5-30 minutes
- **10 samples**: Hours
- **1000 samples** (for accurate exploitability): Days

### Recommendations

1. **For Testing:**
   - Use `num_samples=1` or very small values
   - Test with simpler games first

2. **For Production:**
   - Implement memoization/caching
   - Use depth limits
   - Sample-based approximations
   - Reduce game complexity:
     - Smaller stacks (20-50BB instead of 200BB)
     - Fewer bet sizes (2 instead of 3)
     - Coarser card abstraction

3. **Alternative Metrics:**
   - Use `compute_nash_conv()` for quick convergence checks
   - Head-to-head play against baseline strategies
   - Spot-checking common situations

## Future Optimizations

Potential speedups (not yet implemented):

1. **Memoization**: Cache BR values for visited states
2. **Depth limits**: Stop traversal after N streets
3. **Sampling**: Sample opponent actions instead of exploring all
4. **Parallel computation**: Distribute samples across cores
5. **Numba JIT**: Compile hot paths for 5-10x speedup
6. **Neural approximation**: Train a network to estimate BR values

## Example Usage

```python
from src.evaluation.exploitability import compute_exploitability, compute_nash_conv

# After training your solver
solver = trainer.solver

# Quick convergence check (fast)
nash_conv = compute_nash_conv(solver)
print(f"Nash convergence: {nash_conv:.2f}")

# Approximate exploitability (slow!)
result = compute_exploitability(
    solver,
    num_samples=1,  # Start with just 1 sample!
    use_average_strategy=True
)
print(f"Exploitability: {result['exploitability']:.2f} mbb/g")
```

## Implementation Details

The best response computation follows this recursion:

```
BR(state, player):
  if terminal:
    return payoff(player)

  if chance_node:
    sample cards and recurse

  if current_player == BR_player:
    # Best response: choose action with max EV
    return max(BR(next_state) for action in legal_actions)

  else:
    # Opponent: use their strategy
    strategy = solver.get_strategy(infoset)
    return sum(prob * BR(next_state) for action, prob in strategy)
```

This explores the full game tree when the BR player acts (trying all actions) and follows the solver's strategy when the opponent acts.

## Testing

Tests are in `tests/evaluation/test_exploitability.py`. They use very small sample sizes due to performance constraints.

To run tests:
```bash
uv run pytest tests/evaluation/test_exploitability.py -v
```

Note: Tests with exploitability computation may take several minutes to complete.
