"""Exact best response on a sampled public tree: zero evaluation variance."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.pipeline.evaluation.estimators.public_tree_br import PublicBRConfig, compute_public_tree_br
from src.pipeline.evaluation.policy_profile import profile_policy
from src.pipeline.services.runs import checkpoint_iteration_of
from src.pipeline.services.scoring._shared import EvaluationOutput, prepare_blueprint

logger = logging.getLogger(__name__)

EXACT_BR_ESTIMATOR_LABEL = "public_tree_exact_br (deterministic exact BR on sampled public tree)"


def evaluate_run_exact_br(
    run_dir: Path,
    config: PublicBRConfig | None = None,
    *,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
    on_branch: Callable[[int, int], None] | None = None,
    policy_profile: bool = False,
) -> EvaluationOutput:
    """Exact best response on the sampled public tree (deterministic point value).

    Zero evaluation variance: the same checkpoint under the same
    :class:`PublicBRConfig` always scores identically, so two checkpoints in one
    tier are exactly paired — a difference is pure signal, with no hand budget
    or p-value involved. The value is the exploitability of the board-sampled
    restricted game (see :mod:`~src.pipeline.evaluation.estimators.public_tree_br`), not of
    full HUNL: compare within a tier, don't quote it as a bound.

    ``policy_profile`` attaches the checkpoint's per-street entropy and
    preflop tables (:mod:`~src.pipeline.evaluation.policy_profile`) to the
    record; it reads the arrays already loaded for the walk.

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
    # Workers rebuild the blueprint because the solver is not picklable.
    prepared = prepare_blueprint(
        run_dir,
        abstraction_hash,
        at_iteration,
        config.num_workers,
        config.policy_iterate,
        config.avg_window_from,
        config.avg_gamma,
    )
    metadata, solver, storage, factory = (
        prepared.metadata,
        prepared.solver,
        prepared.storage,
        prepared.factory,
    )
    result = compute_public_tree_br(
        solver,
        config,
        starting_stack=metadata.config.game.starting_stack,
        blueprint_factory=factory,
        # Both paths report every flop branch, serial as it is walked and
        # forked as its job lands.
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
                "self_play_mbb": r.self_play_mbb,
                "gain_mbb": r.gain_mbb,
            }
            for r in result.seat_results
        ],
        "missing_policy_mass": result.missing_policy_mass,
        "nodes_visited": result.nodes_visited,
        "num_flops": result.num_flops,
        "num_turns": config.num_turns,
        "num_rivers": config.num_rivers,
        "board_seed": config.board_seed,
        "conditional_chance": config.conditional_chance,
        "elapsed_s": result.elapsed_s,
        "big_blind": metadata.config.game.big_blind,
        "in_abstraction": config.in_abstraction,
        "policy_threshold": config.policy_threshold,
        "purify": config.purify,
        # Attribution only -- it changes what is TALLIED, never a value -- so it
        # is recorded to tell two rows apart and stays out of the knob tier.
        "decompose": config.decompose,
        # Which strategy of the checkpoint was measured. Empty for the plain
        # average, so every row written before this pairs exactly as it did.
        **prepared.policy_record,
    }
    if result.decomposition is not None:
        results["decomposition"] = result.decomposition
    if policy_profile:
        results["policy_profile"] = profile_policy(
            storage, solver.rules, solver.action_model, metadata.config.game.starting_stack
        )
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir, at_iteration),
        tree_fingerprint=solver.tree.fingerprint(),
    )
