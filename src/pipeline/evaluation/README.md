# Evaluation Module

This module measures blueprint quality. Two trustworthy metrics coexist:
**public-tree exact BR** (`public_tree_br.py`) — a deterministic,
zero-variance point value for comparing checkpoints — and **Local Best
Response (LBR)** exploitability, a Monte Carlo lower bound that also covers
off-tree play and the deployed (resolver) system. A legacy rollout estimator
is retained only as a fast smoke-test diagnostic.

## What is Exploitability?

Exploitability measures how much an optimal opponent (best response) can gain
against your strategy — the gold-standard metric for evaluating CFR
convergence in poker. Exact best response over full HUNL is computationally
infeasible (the generic implementation in `best_response.py` is validated on
toy games like Kuhn), so we approximate it two ways: exactly on a
deterministic board-sampled restriction of the game (`public_tree_br.py`),
and from below with LBR on the full game.

**Units:** milli-big-blinds per game (mbb/g). Lower is better. LBR is a
*lower bound* on true exploitability: it reports how much a specific,
tractable exploiter wins, so real exploitability is at least the reported
value.

## Public-Tree Exact BR (`public_tree_br.py`)

`compute_public_tree_br()` computes an **exact, full-lookahead best response**
— per-combo, range-vs-range, with exact card-removal — against the
blueprint's average strategy, on the blueprint's own betting tree with chance
restricted to a fixed, seed-deterministic board sample (`PublicBRConfig`:
`num_flops`/`num_turns`/`num_rivers`/`board_seed`). Its distinguishing
property is **zero evaluation variance**: the same checkpoint always scores
identically, and two checkpoints scored under one config are exactly paired —
a difference is pure signal, with no hand budget or p-value involved. The
scalar reference game in `tests/pipeline/evaluation/restricted_hunl.py`
validates the vectorized engine against the generic `best_response.py` to
1e-9.

Caveats: the absolute value is the exploitability of the board-restricted
game (it deflates as the sample thins), not a bound on full HUNL; and it is
on-tree only. Use it as the default cross-checkpoint gate; use LBR for
off-tree pressure and for measuring the deployed (resolver) system.

**Prefer running it on Modal** — production checkpoints are multi-GB loads
and the walk is CPU-heavy, so don't occupy a laptop with it:

    modal run modal_app.py::run_eval --run-id <run> --method exact-br \
        --br-flops 8 --br-turns 2 --br-rivers 2 --br-board-seed 7

This routes through the same `evaluate_and_record()` orchestrator and commits
the ledger row to the Volume. The local CLI form (`poker-solver-run evaluate
--method exact_br --br-flops N --br-turns K --br-rivers K --br-board-seed S`)
exists for small checkpoints and debugging only.

## Local Best Response (`hunl_local_best_response.py`)

`compute_lbr_exploitability()` plays the frozen blueprint against an LBR
exploiter over `num_hands` dealt hands and reports the exploiter's mean
winnings with standard error and a 95% confidence interval (`LBRResult`).

Key knobs (`LBRConfig`), all of which define the *comparison tier* of a
result:

- **`scorer`** — how the exploiter values its actions:
  - `"myopic"` (default): one-step equity-based action scoring.
  - `"lookahead"`: depth-limited best-response lookahead
    (`lookahead_scorer.py`), with `lookahead_depth` (default 2) and
    `lookahead_top_k` (default 3). This is the standard scorer for
    on-tree evaluation — the myopic scorer substantially understates
    exploitability.
- **`opponent`** — what the exploiter plays against:
  - `"blueprint"` (default): raw average-strategy table lookups.
  - `"deployed"`: blueprint + the runtime subgame resolver, i.e. the agent
    as actually deployed (`resolver_iterations` pins the resolver budget).
- **`include_off_tree`** (default `False`): allow the exploiter to take bet
  sizes outside the trained action tree. Implemented rigorously via
  `shadow_state.py` (`ShadowTracker`): a shadow on-tree `GameState` is
  carried alongside the real one, and off-tree bets are translated to
  on-tree proxies (pseudo-harmonic mapping) so opponent strategy lookups
  stay on the trained tree. When off, the shadow path never diverges and
  draws no RNG.
- `num_hands` (default 1000), `equity_runouts` (default 12), `allin_runouts`,
  `num_workers`, `base_seed`.

Results include per-hand samples (`pair_samples_mbb`), which enable *paired*
statistical comparison between runs evaluated with the same seed and tier.

## Orchestration & the Eval Ledger

All evaluation transports (headless CLI, Modal) route through one
orchestrator: `evaluate_and_record()` in `src/pipeline/services.py`. It:

1. Runs the requested method (`lbr` by default, or legacy `rollout`).
2. Pins the run's recorded `card_abstraction_hash` (refusing unhashed runs)
   so evals always use the abstraction the run was trained with.
3. Records git provenance (commit/dirty for both the run and the eval).
4. Appends a row to the append-only ledger at `data/eval_ledger.jsonl`, and
   writes the full payload (including per-hand samples) under
   `<run_dir>/evals/`.

### CLI (`poker-solver-run`)

```bash
# Evaluate a run (LBR, on-tree lookahead scorer, deployed opponent)
poker-solver-run evaluate --run <id> --hands 1000 --runouts 12 \
    --scorer lookahead --opponent deployed --seed 42

# Browse recorded evaluations
poker-solver-run ledger [--run <id>] [--limit N]

# Paired comparison between two runs (mean diff ± se, 95% CI, p-value)
poker-solver-run compare --a <run> --b <run>
```

`compare` computes a paired-sample test (`statistics.py`) and **refuses**
mismatched pairings: differing `base_seed`, `num_hands`, or any tier knob
(`scorer`, `opponent`, `include_off_tree`). `--force` overrides, but the
p-value is then untrustworthy. Use the ledger instead of hand-transcribing
scores.

## Legacy Rollout Estimator (`exploitability.py`)

`compute_exploitability(solver, num_samples, use_average_strategy,
num_rollouts_per_infoset, seed)` estimates a **one-ply deviation gain** via
Monte Carlo rollouts. It is **not** a trustworthy exploitability figure — it
severely understates true exploitability and is not a valid bound. It is
retained only as a fast smoke test (`--method rollout` in the CLI).

`compute_total_positive_regret(solver)` is a training-convergence
diagnostic: it should decrease during training, is not comparable across
abstractions, and is not interpretable in big-blind terms.

## Reporting Guidelines

1. Report confidence intervals, never bare point estimates.
2. State the full tier: scorer, opponent, off-tree flag, `num_hands`,
   `equity_runouts`, seed. Numbers from different tiers are not comparable.
3. Compare runs only via `poker-solver-run compare` (paired, same seed/tier).
4. State that LBR is a lower bound on exploitability, not the exact value.

## Module Map

- `hunl_local_best_response.py` — LBR evaluator (`compute_lbr_exploitability`)
- `lookahead_scorer.py` — depth-limited lookahead action scorer
- `shadow_state.py` — off-tree shadow-state translation
- `ledger.py` — eval ledger (rows, tiers, mismatch detection)
- `statistics.py` — paired-sample comparison
- `resolver_match.py` — resolver-in-eval machinery
- `exploitability.py` — legacy rollout diagnostic
- `best_response.py` / `tabular_cfr.py` — exact BR for toy-game validation
- `preflop_chart.py` — preflop strategy extraction for the chart viewer

## Testing

```bash
uv run pytest tests/pipeline/evaluation/
```

Key files: `test_hunl_local_best_response.py`, `test_lookahead_scorer.py`,
`test_shadow_state.py`, `test_ledger.py`, `test_statistics.py`,
`test_exploitability_rollout.py` (legacy path).

## References

1. Lisý & Bowling, "Equilibrium Approximation Quality of Current No-Limit
   Poker Bots" (2017) — Local Best Response
2. Johanson et al., "Evaluating State-Space Abstractions in Extensive-Form
   Games" (AAMAS 2013)
3. Bowling et al., "Heads-up Limit Hold'em Poker is Solved" (Science 2015)
