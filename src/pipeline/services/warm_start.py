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
from typing import TYPE_CHECKING

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint, save_checkpoint
from src.pipeline.blueprint import construction
from src.shared.numeric import NORMALIZE_EPS

if TYPE_CHECKING:
    from pathlib import Path

    from src.shared.config import Config

logger = logging.getLogger(__name__)

DEFAULT_EFFECTIVE_ITERATIONS = 1000

PRIOR_SHAPES = ("flat", "confidence")
"""How a prior's weight is spread across rows. See :func:`regrets_encoding`."""

SEEDED_MARKER = ".warm-started"
# Written once a run is seeded, so a resume can tell "carries a prior" from
# "asked for one and never got it" -- two 30M sweeps turned out to be controls.


def _row_slot_starts(tree) -> np.ndarray:
    per_row = np.repeat(tree.num_actions, tree.buckets_per_node)
    starts = np.zeros(tree.num_rows, dtype=np.int64)
    np.cumsum(per_row[:-1], out=starts[1:])
    return starts


def row_confidence(
    normalised: np.ndarray, row_starts: np.ndarray, slot_width: np.ndarray
) -> np.ndarray:
    """How decisive the prior is at each row, in ``[0, 1]``.

    ``1 - H/H_max``, where ``H`` is the Shannon entropy of the row's strategy.
    A row the prior has committed to (95/3/2) scores near 1; a row it is
    indifferent about (34/33/33) scores near 0.

    Why this is not cosmetic: a FLAT weight seeds a near-uniform row at full
    strength, and those units then have to be overcome before the solver can
    move away from uniform. That is not a neutral prior -- it is an active brake
    on exactly the rows where the prior had no opinion to contribute. Scaling by
    confidence lets those rows get out of the way.

    Rows of width 1 have no choice to be confident about; they score 1 so a
    forced action is never damped.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(normalised > 0.0, -normalised * np.log(normalised), 0.0)
    entropy = np.add.reduceat(terms, row_starts)
    widths = slot_width.astype(np.float64)
    max_entropy = np.log(np.maximum(widths, 2.0))
    return np.clip(1.0 - entropy / max_entropy, 0.0, 1.0)


def regrets_encoding(
    strategy_sum: np.ndarray,
    row_starts: np.ndarray,
    slot_width: np.ndarray,
    weight: float,
    *,
    shape: str = "flat",
    base_regrets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative regrets whose regret-matching is ``strategy_sum`` row-normalised.

    Pure, so the property that matters — matching these regrets reproduces the
    source strategy — is testable without a tree, a config or a disk.

    ``shape`` decides how the weight is DISTRIBUTED across rows; it never changes
    the strategy any row plays, only how hard that row resists being moved.

        flat        every row claims ``weight``. One number, one meaning
                    everywhere, and the shape every measured result so far used.
        confidence  a row claims ``weight * row_confidence``, so a prior with no
                    opinion does not brake the rows it has no opinion about.

    Returns the regrets and a mask of rows that carried any mass. A row with none
    is left at zero, which regret-matching reads as uniform; that is the honest
    answer for a row the source never gave one for.
    """
    if shape not in PRIOR_SHAPES:
        raise ValueError(
            f"unknown prior shape '{shape}'; expected one of {', '.join(PRIOR_SHAPES)}"
        )
    strategy = np.asarray(strategy_sum, dtype=np.float64)
    totals = np.add.reduceat(strategy, row_starts)
    seeded = totals > NORMALIZE_EPS
    per_slot_total = np.repeat(totals, slot_width)
    normalised = np.divide(
        strategy, per_slot_total, out=np.zeros_like(strategy), where=per_slot_total > NORMALIZE_EPS
    )
    scale = np.full(row_starts.shape[0], float(weight))
    if shape == "confidence":
        scale = scale * row_confidence(normalised, row_starts, slot_width)
    regrets = normalised * np.repeat(scale, slot_width)
    if base_regrets is not None:
        # ADDED, not replaced. The two priors answer different questions -- an
        # equity guess covers rows nothing reached, a trained prior covers rows
        # it has a real opinion about -- so each contributes its own confidence
        # and regret matching resolves the overlap. Replacing would discard the
        # guess exactly where the trained prior is weakest.
        base = np.asarray(base_regrets, dtype=np.float64)
        regrets = regrets + base
        seeded = seeded | (np.add.reduceat(base, row_starts) > NORMALIZE_EPS)
    return regrets, seeded


def seed_checkpoint(
    config: Config,
    *,
    source_run: Path,
    run_dir: Path,
    effective_iterations: int,
    abstraction_hash: str | None,
    at_iteration: int | None = None,
    shape: str = "flat",
    base_regrets: np.ndarray | None = None,
) -> int:
    """Write iteration-0 regrets encoding ``source_run``'s average strategy.

    The caller already owns ``run_dir`` and its tracker, which is what lets
    ``train_static`` seed and train in ONE leg rather than two -- a second leg
    could not be retried, since it would find a checkpoint already there.
    Returns the number of rows that carry a prior.
    """
    if effective_iterations <= 0:
        raise ValueError("effective_iterations must be positive; it scales the prior's weight.")

    action_model = ActionModel(config)
    abstraction = construction.build_card_abstraction(config)
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
    regrets, seeded = regrets_encoding(
        source.strategy_sum,
        starts,
        slot_width,
        float(effective_iterations),
        shape=shape,
        base_regrets=base_regrets,
    )

    target = StaticArrayStorage(tree)
    target.regrets[:] = regrets.astype(target.regrets.dtype)
    # strategy_sum stays ZERO: the reported blueprint must average the real game
    # only. See the module docstring.
    target.visited[:] = seeded
    save_checkpoint(target, run_dir, iteration=0, abstraction_id=abstraction_hash)

    logger.info(
        "seeded %s from %s at weight %d: %.2f%% of rows carry a prior",
        run_dir.name,
        source_run.name,
        effective_iterations,
        float(seeded.mean()) * 100.0,
    )
    return int(seeded.sum())


__all__ = (
    "DEFAULT_EFFECTIVE_ITERATIONS",
    "PRIOR_SHAPES",
    "SEEDED_MARKER",
    "regrets_encoding",
    "row_confidence",
    "seed_checkpoint",
)
