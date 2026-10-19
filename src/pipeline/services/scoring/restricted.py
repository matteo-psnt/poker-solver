"""Restricted-game convergence: the run's tree and abstraction, retrained on a fixed board pool.

The run supplies only its GAME -- config, tree, abstraction, DCFR knobs -- and
none of its strategy, so the curve answers what this recipe can reach, not what
the run did reach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.game.rules import GameRules
from src.engine.solver.vector import compile_tree
from src.pipeline.evaluation.estimators.restricted_cfr import RestrictedConfig, train_restricted
from src.pipeline.services.runs import load_run_metadata
from src.pipeline.services.scoring._shared import (
    EvaluationOutput,
    build_blueprint_for,
    effective_abstraction_hash,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.shared.ports.record import RecordSource

RESTRICTED_ESTIMATOR_LABEL = (
    "restricted_cfr (the run's tree and abstraction retrained on a fixed board pool, "
    "scored by that pool's EXACT in-abstraction best response; NOT the full game)"
)


def evaluate_run_restricted(
    run_dir: Path,
    config: RestrictedConfig,
    *,
    abstraction_hash: str | None = None,
    record_source: RecordSource | None = None,
) -> EvaluationOutput:
    metadata = load_run_metadata(run_dir, record_source)
    effective = effective_abstraction_hash(run_dir, metadata, abstraction_hash)
    solver, storage, _ = build_blueprint_for(run_dir, metadata, effective, None, play_only=True)
    run_config = metadata.config
    rules = GameRules(run_config.game.small_blind, run_config.game.big_blind)
    knobs = run_config.solver
    results = train_restricted(
        compile_tree(solver.tree, rules),
        solver.card_abstraction,
        config,
        weighting=knobs.iteration_weighting,
        dcfr_alpha=knobs.dcfr_alpha,
        dcfr_beta=knobs.dcfr_beta,
        dcfr_gamma=knobs.dcfr_gamma,
        cfr_plus=knobs.cfr_plus,
        big_blind=float(run_config.game.big_blind),
    )
    return EvaluationOutput(infosets=storage.num_infosets(), results=results)
