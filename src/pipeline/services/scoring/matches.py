"""Head-to-head gates: run against run, and blueprint against deployed resolver."""

import functools
import logging
from pathlib import Path
from typing import Any

from src.pipeline.blueprint.construction import build_static_evaluation_solver
from src.pipeline.evaluation.estimators.blueprint_match import play_blueprint_match
from src.pipeline.evaluation.estimators.resolver_match import play_resolver_match
from src.pipeline.services.runs import checkpoint_iteration_of, load_run_metadata
from src.pipeline.services.scoring._shared import (
    EvaluationOutput,
    build_blueprint_for,
    effective_abstraction_hash,
    load_blueprint,
)
from src.shared.ports.record import RecordSource

logger = logging.getLogger(__name__)

RESOLVER_GATE_ESTIMATOR_LABEL = (
    "resolver_match (duplicate-deal chip edge of blueprint+resolver over the bare "
    "blueprint; NOT an exploitability bound)"
)

BLUEPRINT_MATCH_ESTIMATOR_LABEL = (
    "blueprint_match (duplicate-deal head-to-head chip edge; abstraction-safe, not a bound)"
)


def _with_resolver_overrides(
    config: Any,
    *,
    leaf_continuation_fraction: float | None,
    max_iterations: int | None,
    root_prior_weight: float | None,
    leaf_rollouts: int | None,
    blend_alpha: float | None,
) -> Any:
    """The run's config, with the two knobs an A/B over leaf valuation needs.

    Overridden HERE rather than at train time because a resolver knob is a
    property of how a checkpoint is PLAYED, not of how it was trained -- the
    same run has to be scored under both arms or the comparison is confounded
    by everything else that differs between two runs.

    ``max_iterations`` matters more than it looks. The resolver is normally
    WALL-CLOCK budgeted (``time_budget_ms``), so a busier or slower box does
    fewer CFR iterations and lands on a different strategy. Two arms measured
    that way differ by machine speed as much as by the knob under test. Pinning
    the iteration count is what makes the two arms comparable, and the config
    field already exists for exactly this ("a determinism knob for reproducible
    experiments and tests").
    """
    # Built as a dict rather than a chain of `and ... is None`: every knob added
    # here has had to edit that condition too, and forgetting it would silently
    # return the run's config while the caller believed it had overridden one.
    requested = {
        "leaf_continuation_fraction": leaf_continuation_fraction,
        "max_iterations": max_iterations,
        "root_prior_weight": root_prior_weight,
        "leaf_rollouts": leaf_rollouts,
        "policy_blend_alpha": blend_alpha,
    }
    updates: dict[str, Any] = {k: v for k, v in requested.items() if v is not None}
    if not updates:
        return config
    return config.model_copy(update={"resolver": config.resolver.model_copy(update=updates)})


def evaluate_run_resolver_gate(
    run_dir: Path,
    *,
    num_deals: int = 1000,
    time_budget_ms: int = 100,
    seed: int = 1,
    leaf_continuation_fraction: float | None = None,
    max_iterations: int | None = None,
    root_prior_weight: float | None = None,
    leaf_rollouts: int | None = None,
    blend_alpha: float | None = None,
    workers: int = 1,
    allin_runouts: int = 1,
    record_source: RecordSource | None = None,
) -> EvaluationOutput:
    """Head-to-head resolver gate on a run: blueprint+resolver vs bare blueprint.

    Duplicate deals (seat-swapped pairs off a fixed deck) cancel card luck, so the
    resolver's chip edge is measurable in ~1k deals. Positive edge means the
    resolver improves on the blueprint it wraps — the go/no-go signal for
    investing in the search path.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid configuration or checkpoint state.
    """
    metadata = load_run_metadata(run_dir, record_source)
    config = _with_resolver_overrides(
        metadata.config,
        leaf_continuation_fraction=leaf_continuation_fraction,
        max_iterations=max_iterations,
        root_prior_weight=root_prior_weight,
        leaf_rollouts=leaf_rollouts,
        blend_alpha=blend_alpha,
    )
    # Pinned like every other estimator: resolving by config NAME would silently
    # rebucket the checkpoint if the abstraction was recomputed since training.
    effective_hash = effective_abstraction_hash(run_dir, metadata, None)
    solver, storage, _ = build_static_evaluation_solver(
        config, checkpoint_dir=run_dir, abstraction_hash=effective_hash
    )
    # The factory carries the OVERRIDDEN config, not the run's. A worker built
    # from the stored config would resolve under different knobs than the
    # coordinator and the arms would silently stop being an A/B -- the same trap
    # `prepare_blueprint` documents for a mismatched abstraction hash.
    factory = (
        functools.partial(load_blueprint, config, run_dir, effective_hash) if workers > 1 else None
    )
    result = play_resolver_match(
        solver,
        num_deals=num_deals,
        time_budget_ms=time_budget_ms,
        seed=seed,
        workers=workers,
        allin_runouts=allin_runouts,
        blueprint_factory=factory,
    )
    results = {
        "resolver_mbb_per_hand": result.resolver_mbb_per_hand,
        "se_mbb": result.se_mbb,
        "confidence_95_mbb": result.confidence_95_mbb,
        "p_value": result.p_value,
        "num_deals": result.num_deals,
        "num_hands": result.num_hands,
        "resolver_decisions": result.resolver_decisions,
        "resolver_fallbacks": result.resolver_fallbacks,
        "time_budget_ms": time_budget_ms,
        "workers": workers,
        "root_prior_weight": config.resolver.root_prior_weight,
        "leaf_rollouts": config.resolver.leaf_rollouts,
        "seed": seed,
        "pair_samples_mbb": result.pair_samples_mbb,
        # The arm's identity, recorded WITH the number. Two resolver-gate rows
        # are otherwise indistinguishable, and the whole point of running this
        # twice is to subtract one from the other.
        "leaf_continuation_fraction": config.resolver.leaf_continuation_fraction,
        "resolver_max_iterations": config.resolver.max_iterations,
        "allin_runouts": allin_runouts,
        "resolver_blend_alpha": config.resolver.policy_blend_alpha,
    }
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir),
    )


def evaluate_blueprint_match(
    run_dir_a: Path,
    run_dir_b: Path,
    *,
    num_deals: int = 2000,
    seed: int = 1,
    at_a: int | None = None,
    at_b: int | None = None,
    record_source: RecordSource | None = None,
) -> EvaluationOutput:
    """Head-to-head match between two runs' blueprints on duplicate deals.

    Positive ``a_mbb_per_hand`` means run A's blueprint wins chips off run B's.
    Each blueprint is pinned to the card abstraction its run trained against.

    Both rungs are addressable because two arms are only comparable at matched
    TRAINING BUDGET: without ``at_b`` an arm that happened to be continued
    further plays its whole extra ladder, which is a different arm.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid checkpoint state, or mismatched game configurations.
    """
    metadata_a = load_run_metadata(run_dir_a, record_source)
    metadata_b = load_run_metadata(run_dir_b, record_source)
    if metadata_a.config.game != metadata_b.config.game:
        raise ValueError(
            f"Game configs differ between runs ({metadata_a.config.game} vs "
            f"{metadata_b.config.game}); a chip match would be meaningless."
        )

    # `play_only`: a match never reads regrets, and two blueprints are resident
    # at once here where every other estimator holds one.
    solver_a, storage_a, _ = build_blueprint_for(
        run_dir_a,
        metadata_a,
        effective_abstraction_hash(run_dir_a, metadata_a, None),
        at_a,
        play_only=True,
    )
    solver_b, storage_b, _ = build_blueprint_for(
        run_dir_b,
        metadata_b,
        effective_abstraction_hash(run_dir_b, metadata_b, None),
        at_b,
        play_only=True,
    )
    rung_a = at_a if at_a is not None else checkpoint_iteration_of(run_dir_a)
    rung_b = at_b if at_b is not None else checkpoint_iteration_of(run_dir_b)
    result = play_blueprint_match(solver_a, solver_b, num_deals=num_deals, seed=seed)
    results = {
        "run_a": metadata_a.run_id,
        "run_b": metadata_b.run_id,
        "a_mbb_per_hand": result.a_mbb_per_hand,
        "se_mbb": result.se_mbb,
        "confidence_95_mbb": result.confidence_95_mbb,
        "p_value": result.p_value,
        "num_deals": result.num_deals,
        "num_hands": result.num_hands,
        "infosets_a": storage_a.num_infosets(),
        "infosets_b": storage_b.num_infosets(),
        "seed": seed,
        # Everything about B. The row is filed under A, so the record's own
        # columns describe one half of a pairwise measurement; a chip edge is
        # uninterpretable later without which rung, which abstraction and which
        # betting tree the other half brought.
        "rung_b": rung_b,
        "card_abstraction_hash_b": metadata_b.card_abstraction_hash,
        "action_config_hash_b": metadata_b.action_config_hash,
        "config_name_b": metadata_b.config_name,
        "git_commit_b": metadata_b.git_commit,
        "pair_samples_mbb": result.pair_samples_mbb,
    }
    return EvaluationOutput(
        infosets=storage_a.num_infosets(),
        results=results,
        checkpoint_iteration=rung_a,
    )
