"""Seed a scalar run from a strategy another kernel already found.

The two kernels fail in opposite directions. External sampling is unbiased —
it plays the real game, exact card removal and all — but touches 88-95 rows per
iteration, so at 30M iterations most rows have seen single-digit updates. The
board-free kernel updates all 32,240,608 rows every iteration but solves a
bucket-transition approximation of the chance layer, so it converges accurately
to a game that is not quite ours.

Bias against variance. A warm start uses the cheap biased answer as a prior and
lets unbiased sampling correct it, which is worth trying precisely because
board-free at 200 iterations already scored in the same range as scalar at 30M.

## Warm-start from the STRATEGY, not from the regrets

The obvious move — copy the source's ``regrets`` across and keep training — is
the one Brown & Sandholm (*Strategy-Based Warm Starting for Regret Minimization
in Games*, AAAI-16) show is wrong. Those regrets are sums accumulated under a
different game; carried over, they assert a confidence the real game never
justified, and CFR's bound no longer holds.

What transfers is the strategy. Regret matching reads
``sigma(a) = R+(a) / sum(R+)``, so ANY non-negative regret vector proportional to
``sigma`` reproduces it exactly. The free constant is the interesting part: it
is how much accumulated regret the warm start CLAIMS, and therefore how many real
iterations it takes to talk the solver out of the prior.

That constant is ``effective_iterations`` here, and it is a genuine choice rather
than a detail:

* too small and the prior evaporates in the first few thousand iterations, so the
  warm start buys nothing;
* too large and the run spends its budget arguing with a strategy derived from
  the wrong chance layer, which is worse than starting cold.

It is exposed, recorded, and defaulted to something deliberately modest. Treat it
as the experiment's independent variable, not as a tuned constant.

## The average strategy does NOT transfer

``strategy_sum`` is the deliverable — the thing exploitability is measured on —
and the source's copy is an average over the wrong game. Seeding it would mix two
games' averages in a ratio nobody chose, so it starts at zero and every
contribution to the reported blueprint is earned on the real game.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint, save_checkpoint
from src.pipeline.training import components
from src.pipeline.training.run_tracker import RunTracker
from src.pipeline.training.run_tracker.attempts import ExperimentTag
from src.shared.config_loader import load_training_config
from src.shared.numeric import NORMALIZE_EPS

logger = logging.getLogger(__name__)

DEFAULT_EFFECTIVE_ITERATIONS = 1000


@dataclass(frozen=True)
class WarmStartOutput:
    """Portable summary of a seeded run."""

    run_id: str
    runs_dir: str
    config_name: str
    source_run_id: str
    effective_iterations: int
    num_rows: int
    seeded_rows: int
    seeded_fraction: float
    status: str


def _row_slot_starts(tree) -> np.ndarray:
    per_row = np.repeat(tree.num_actions, tree.buckets_per_node)
    starts = np.zeros(tree.num_rows, dtype=np.int64)
    np.cumsum(per_row[:-1], out=starts[1:])
    return starts


def regrets_encoding(
    strategy_sum: np.ndarray, row_starts: np.ndarray, slot_width: np.ndarray, weight: float
) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative regrets whose regret-matching is ``strategy_sum`` row-normalised.

    Pure, so the property that matters — matching these regrets reproduces the
    source strategy — is testable without a tree, a config or a disk.

    Returns the regrets and a mask of rows that carried any mass. A row with none
    is left at zero, which regret-matching reads as uniform; that is the honest
    answer for a row the source never gave one for.
    """
    strategy = np.asarray(strategy_sum, dtype=np.float64)
    totals = np.add.reduceat(strategy, row_starts)
    seeded = totals > NORMALIZE_EPS
    per_slot_total = np.repeat(totals, slot_width)
    normalised = np.divide(
        strategy, per_slot_total, out=np.zeros_like(strategy), where=per_slot_total > NORMALIZE_EPS
    )
    return normalised * float(weight), seeded


def warm_start_run(
    config_name: str,
    *,
    source_run: Path,
    run_id: str,
    effective_iterations: int = DEFAULT_EFFECTIVE_ITERATIONS,
    at_iteration: int | None = None,
    runs_dir: Path | None = None,
    experiment: ExperimentTag | None = None,
) -> WarmStartOutput:
    """Write a fresh run whose regrets encode ``source_run``'s average strategy.

    The result is an ordinary static checkpoint at iteration 0, so
    ``train-static --run <id>`` continues it with no special casing anywhere in
    the training path.

    Args:
        source_run: Run directory to take the strategy from. Its tree fingerprint
            must match, which ``load_checkpoint`` enforces.
        effective_iterations: How much accumulated regret the prior claims. The
            experiment's independent variable; see the module docstring.
        at_iteration: Source rung to seed from. Board-free quality is U-shaped,
            so the last rung is usually NOT the best one.
    """
    if effective_iterations <= 0:
        raise ValueError("effective_iterations must be positive; it scales the prior's weight.")

    config = load_training_config(config_name)
    base_dir = Path(runs_dir) if runs_dir is not None else Path(config.training.runs_dir)
    run_dir = base_dir / run_id
    if (run_dir / "STATIC_CHECKPOINT.json").exists():
        raise FileExistsError(
            f"{run_dir} already holds a checkpoint. Seeding over a run in progress would "
            "discard its training; pick a new run id."
        )

    action_model = ActionModel(config)
    abstraction = components.build_card_abstraction(config)
    abstraction_hash = components.resolve_card_abstraction_hash(config)
    rules = GameRules(config.game.small_blind, config.game.big_blind)
    tree = build_betting_tree(
        rules, action_model, abstraction, starting_stack=config.game.starting_stack
    )

    source = StaticArrayStorage(tree)
    # Verifies the tree fingerprint, so a strategy from a different tree is
    # refused rather than reinterpreted row-for-row.
    load_checkpoint(source, source_run, at_iteration=at_iteration)

    starts = _row_slot_starts(tree)
    slot_width = np.repeat(tree.num_actions, tree.buckets_per_node)
    # sigma(a) = R+(a)/sum(R+), so regrets proportional to the average strategy
    # reproduce it exactly. The scale is what the prior claims to have learned.
    regrets, seeded = regrets_encoding(
        source.strategy_sum, starts, slot_width, float(effective_iterations)
    )

    target = StaticArrayStorage(tree)
    target.regrets[:] = regrets.astype(target.regrets.dtype)
    # strategy_sum stays ZERO on purpose: the reported blueprint must be an
    # average over the real game only. See the module docstring.
    target.visited[:] = seeded

    tag = experiment or ExperimentTag()
    tracker = RunTracker(
        run_dir=run_dir,
        config_name=config.system.config_name,
        config=config,
        action_config_hash=action_model.get_config_hash(),
        card_abstraction_hash=abstraction_hash,
        experiment_id=tag.experiment_id,
        arm=tag.arm,
        parent_run_id=tag.parent_run_id or source_run.name,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker.initialize()
    save_checkpoint(target, run_dir, iteration=0, abstraction_id=abstraction_hash)
    tracker.update(
        iterations=0,
        runtime_seconds=0.0,
        num_infosets=int(seeded.sum()),
        storage_capacity=tree.num_rows,
    )

    fraction = float(seeded.mean())
    logger.info(
        "seeded %s from %s at weight %d: %.2f%% of rows carry a prior",
        run_id,
        source_run.name,
        effective_iterations,
        fraction * 100.0,
    )
    return WarmStartOutput(
        run_id=run_id,
        runs_dir=str(base_dir),
        config_name=config.system.config_name,
        source_run_id=source_run.name,
        effective_iterations=effective_iterations,
        num_rows=tree.num_rows,
        seeded_rows=int(seeded.sum()),
        seeded_fraction=fraction,
        status="seeded",
    )


__all__ = ("DEFAULT_EFFECTIVE_ITERATIONS", "WarmStartOutput", "warm_start_run")
