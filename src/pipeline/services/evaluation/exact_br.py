"""Exact best response on a sampled public tree: zero evaluation variance."""

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.pipeline.evaluation.estimators.public_tree_br import PublicBRConfig, compute_public_tree_br
from src.pipeline.services.evaluation._shared import (
    EvaluationOutput,
    _effective_abstraction_hash,
    _load_blueprint,
    build_blueprint_for,
)
from src.pipeline.services.runs import checkpoint_iteration_of, load_run_metadata

logger = logging.getLogger(__name__)

EXACT_BR_ESTIMATOR_LABEL = "public_tree_exact_br (deterministic exact BR on sampled public tree)"


def evaluate_run_exact_br(
    run_dir: Path,
    config: PublicBRConfig | None = None,
    *,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
    on_branch: Callable[[int, int], None] | None = None,
) -> EvaluationOutput:
    """Exact best response on the sampled public tree (deterministic point value).

    Zero evaluation variance: the same checkpoint under the same
    :class:`PublicBRConfig` always scores identically, so two checkpoints in one
    tier are exactly paired — a difference is pure signal, with no hand budget
    or p-value involved. The value is the exploitability of the board-sampled
    restricted game (see :mod:`~src.pipeline.evaluation.estimators.public_tree_br`), not of
    full HUNL: compare within a tier, don't quote it as a bound.

    ``at_iteration`` scores a retained ladder rung instead of the published
    snapshot. Because the value is deterministic, scoring several rungs of one run
    under an identical config gives an exactly-paired within-run convergence curve
    — the intended use. Hold the board plan fixed across rungs; shrinking the
    sample for cheap early points destroys the pairing.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid configuration or checkpoint state, or ``at_iteration``
            is not a retained rung (the error lists the available ones).
    """
    config = config or PublicBRConfig()
    metadata = load_run_metadata(run_dir)
    effective_hash = _effective_abstraction_hash(run_dir, metadata, abstraction_hash)
    solver, storage = build_blueprint_for(run_dir, metadata, effective_hash, at_iteration)
    # The four (seat, button) walks are independent; workers rebuild the
    # blueprint because the solver is not picklable. Same factory shape parallel
    # LBR uses, so only picklable args are captured.
    loader = _load_blueprint
    factory = (
        functools.partial(loader, metadata.config, run_dir, effective_hash, at_iteration)
        if config.num_workers > 1
        else None
    )
    result = compute_public_tree_br(
        solver,
        config,
        starting_stack=metadata.config.game.starting_stack,
        blueprint_factory=factory,
        # BOTH paths. Serial reports every flop branch; parallel reports each
        # walk as it lands, in the same unit, because a walk returns what it
        # cost. Coarser when parallel -- four steps, not hundreds -- and still
        # the difference between a bar that moves and one that cannot.
        on_branch=on_branch,
    )
    results: dict[str, Any] = {
        "exploitability_mbb": result.exploitability_mbb,
        "std_error_mbb": 0.0,
        "num_hands": 0,
        "seat_values_mbb": [
            {
                "br_seat": r.br_seat,
                "button": r.button,
                "value_mbb": r.value_mbb,
                "missing_policy_mass": r.missing_policy_mass,
            }
            for r in result.seat_results
        ],
        "missing_policy_mass": result.missing_policy_mass,
        "nodes_visited": result.nodes_visited,
        "num_flops": result.num_flops,
        "num_turns": config.num_turns,
        "num_rivers": config.num_rivers,
        "board_seed": config.board_seed,
        "elapsed_s": result.elapsed_s,
        "big_blind": metadata.config.game.big_blind,
    }
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir, at_iteration),
    )
