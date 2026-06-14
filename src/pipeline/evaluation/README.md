# Evaluation

This package measures blueprint quality. Three questions, three answers, and
they are not interchangeable:

| Question | Module | Property |
| --- | --- | --- |
| How far from optimal, exactly? | `public_tree_br.py` | zero variance, on-tree, board-restricted |
| How far from optimal, from below? | `lbr/` | Monte Carlo lower bound, off-tree and deployed play |
| Does A win chips off B? | `blueprint_match.py`, `resolver_match.py` | the only abstraction-crossing gate |

## Layout

    estimators/      the four things that put a number on a blueprint
      lbr/               Local Best Response against the HUNL blueprint
      public_tree_br.py  deterministic exact BR (the default checkpoint gate)
      blueprint_match.py duplicate-deal head-to-head
      resolver_match.py  resolver-in-eval machinery
    reference/       exact answers for SMALL games -- the harness, not a scorer
    ledger/          eval records: tiers, queries, the derived index
    statistics.py    paired-sample comparison
    units.py         chips -> bb/mbb, defined once

The estimators are peers: a caller picks exactly one, and which one is the
evaluation's identity, recorded on every row. They used to be filed two
different ways -- `lbr/` a package, `public_tree_br.py` loose beside the ledger
and the oracles -- so "which of these is a scorer" was not answerable from the
directory.

**`reference/` is the one that gets misread.** It holds `best_response.py`,
`game_tree.py`, `tabular_cfr.py` and `local_best_response.py` — engine-agnostic,
exact, and tractable only at Kuhn/Leduc scale. They exist to prove the
expensive evaluators right, not to score a blueprint:
`tests/pipeline/evaluation/restricted_hunl.py` validates the vectorised
public-tree engine against `reference/best_response.py` to 1e-9. Flat alongside
`public_tree_br.py`, a file called `best_response.py` reads like a third
production scorer, which is why it no longer sits there -- and why `reference/`
stayed OUT of `estimators/` when the scorers were gathered into it.

## What exploitability is

How much an optimal opponent can gain against your strategy — the standard
metric for CFR convergence. Exact BR over full HUNL is infeasible, so it is
approximated two ways: exactly on a deterministic board-sampled restriction of
the game (`public_tree_br.py`), and from below with LBR on the full game.

**Units:** milli-big-blinds per game (mbb/g), lower is better, via `units.py`.
LBR is a *lower bound*: it reports what one tractable exploiter wins, so true
exploitability is at least the reported value.

## Public-tree exact BR

`compute_public_tree_br()` computes an exact, full-lookahead best response —
per-combo, range-vs-range, with exact card removal — against the blueprint's
average strategy, on the blueprint's own betting tree with chance restricted to
a fixed, seed-deterministic board sample (`PublicBRConfig`:
`num_flops`/`num_turns`/`num_rivers`/`board_seed`).

Its distinguishing property is **zero evaluation variance**: the same
checkpoint always scores identically, and two checkpoints scored under one
config are exactly paired — a difference is pure signal, with no hand budget or
p-value involved.

Caveats: the absolute value is the exploitability of the board-restricted game
(it deflates as the sample thins), not a bound on full HUNL; and it is on-tree
only. Use it as the default cross-checkpoint gate; use LBR for off-tree
pressure and for the deployed system.

## Local Best Response (`lbr/`)

`compute_lbr_exploitability()` plays the frozen blueprint against an LBR
exploiter over `num_hands` dealt hands and reports the exploiter's mean
winnings with standard error and a 95% CI (`LBRResult`).

Every knob in `LBRConfig` defines the **comparison tier** of the result, and
`ledger/tiers.py` derives that tier from the same config the evaluation
consumed — so two numbers from different tiers can never be paired:

- **`scorer`** — how the exploiter values an action. `"myopic"` is one-step
  equity scoring; `"lookahead"` is depth-limited best-response lookahead
  (`lookahead_scorer.py`, knobs `lookahead_depth`, `lookahead_top_k`).
  **`lookahead` is the standard scorer** — myopic substantially understates
  exploitability.
- **`opponent`** — `"blueprint"` (raw average-strategy lookups) or
  `"deployed"` (blueprint plus the runtime subgame resolver, i.e. the agent as
  actually shipped; `resolver_iterations` pins the resolver budget).
- **`include_off_tree`** (default `False`) — let the exploiter bet sizes
  outside the trained action tree. Implemented via `shadow_state.py`
  (`ShadowTracker`): a shadow on-tree `GameState` is carried alongside the real
  one and off-tree bets are translated to on-tree proxies (pseudo-harmonic
  mapping), so opponent lookups stay on the trained tree. When off, the shadow
  path never diverges and draws no RNG.
- `num_hands`, `equity_runouts`, `allin_runouts`, `num_workers`, `base_seed`.

Results carry per-hand samples (`pair_samples_mbb`), which is what makes
*paired* comparison between runs at the same seed and tier possible.

## Head-to-head — the abstraction gate

`play_blueprint_match()` plays two blueprints on duplicate, seat-swapped deals
off a fixed deck and reports A's chip edge in mbb/hand (`a_mbb_per_hand`, with
SE, 95% CI, paired p-value). Each blueprint maps the real state through **its
own** card abstraction and action model, so this is the one metric that
compares blueprints trained on **different abstractions** — exact BR and
on-tree LBR both live on a single tree and cannot.

It answers a different question, and the two are complementary, not redundant:
head-to-head is **intransitive**, so a *more*-exploitable blueprint can still
beat a less-exploitable one. Use both.

**Scope.** Rigorous when the two blueprints differ only in their **card**
abstraction (bucket count/scheme, board-order) — they share the action grid, so
every state stays on a common action tree. An **action**-abstraction difference
would put an opponent's off-grid bet outside the other's tree and needs the
off-tree translation this harness does not implement.

**Variance — read each match's own CI.** Duplicate-deal CRN cancels card luck
only on pairs where the two blueprints *agree* (a self-match is provably
all-zero — see `test_blueprint_match.py`); where they disagree the pair carries
full pot variance. At 2000 deals the SE is ~240 mbb/hand *even for
near-identical strategies*, and a real cross-abstraction match is noisier. SE
falls as 1/sqrt(deals), so ~100 mbb resolution needs ~45k deals. **Never quote a
fixed threshold.** Play is cheap next to blueprint *load*, so raising `--deals`
is the right lever; a variance-reduction estimator is not worth its correctness
cost here, where samples are nearly free. (Measured: AIVAT came out *worse* —
do not enable it.)

## How a result gets recorded

All transports route through one orchestrator, `evaluate_and_record()` in
`src/pipeline/services/scoring/`. It runs the requested method, pins the
run's recorded `card_abstraction_hash` (refusing unhashed runs) so an eval
always uses the abstraction the run trained with, records git provenance, and
writes the complete row to `<run_dir>/evals/<slug>.json`.

**There is no stored index.** `ledger` DERIVES one from the published documents
on every read, which is what makes concurrent evaluation from several boxes
safe. `eval-*.json` and `record-*.json` are legacy shapes and are skipped
deliberately — reading both entered one evaluation twice. A sparse `ledger`
therefore means un-migrated legacy files on the share, not a broken rebuild.

## Running one

Evaluation runs **on the pool**, never on a laptop. `score` submits it, one
task per ladder rung; `evaluate` is what the node then executes.

```bash
# Score a published run on the pool
poker-solver score --run <id> --method lbr -- --scorer lookahead --opponent deployed

# Read the record back
poker-solver ledger [--run <id>] [--limit N]
poker-solver curve --run <id>
```

`--method` is `lbr` or `exact_br`. Arms are tagged with `--experiment`/`--arm`
at submit and the tags travel onto the eval documents, so `ledger --json` is
what groups them. Two numbers are only comparable within one tier — matching
`base_seed`, `num_hands` and every tier knob. Never hand-transcribe scores.

## Reporting guidelines

1. Report confidence intervals, never bare point estimates.
2. State the full tier: scorer, opponent, off-tree flag, `num_hands`,
   `equity_runouts`, seed. Numbers from different tiers are not comparable.
3. Compare only within one tier (same seed and knobs).
4. Say that LBR is a lower bound, not the exact value.

## Testing

```bash
uv run pytest tests/pipeline/evaluation/
```

Mirrors the source layout: `estimators/` (with `lbr/` inside it) and
`reference/` have their own test packages; the toy-game fixtures (`kuhn_poker.py`, `leduc_poker.py`) stay at the
top level because `tests/engine/` uses them too.

## References

1. Lisy & Bowling, "Equilibrium Approximation Quality of Current No-Limit
   Poker Bots" (2017) — Local Best Response
2. Johanson et al., "Evaluating State-Space Abstractions in Extensive-Form
   Games" (AAMAS 2013)
3. Johanson et al., "Accelerated Best Response Calculation in Large Extensive
   Games" (2011)
4. Bowling et al., "Heads-up Limit Hold'em Poker is Solved" (Science 2015)
