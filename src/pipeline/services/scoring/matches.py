"""Head-to-head gates: run against run, and blueprint against deployed resolver."""

from __future__ import annotations

import dataclasses
import functools
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from src.engine.solver.policy.threshold import apply_policy_threshold
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

if TYPE_CHECKING:
    from pathlib import Path

    from src.engine.solver.mccfr.static_solver import StaticTreeSolver
    from src.engine.solver.storage.static_array import StaticArrayStorage
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
    min_street: str | None = None,
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
        "min_street": min_street,
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
        # The other seat's, so the per-aggression rate below can be checked for
        # seat symmetry rather than assumed to have it.
        "blueprint_decisions": result.blueprint_decisions,
        "resolver_fallbacks": result.resolver_fallbacks,
        # WHY the resolver was retired, which this gate could not previously see:
        # not the chip edge (nothing, at any deal count anyone runs) but DOUBLE
        # the stack-off rate on the same deals, which an elimination format
        # punishes far harder than mbb/hand shows. Per-aggression rates over
        # thousands of actions, so they resolve where the chip edge cannot.
        "resolver_stack_off_rate": result.resolver_stack_off_rate,
        "blueprint_stack_off_rate": result.blueprint_stack_off_rate,
        "resolver_stack_offs": result.resolver_stack_offs,
        "blueprint_stack_offs": result.blueprint_stack_offs,
        "resolver_aggressions": result.resolver_aggressions,
        "blueprint_aggressions": result.blueprint_aggressions,
        # 1.0 is uniform over the whole menu. The diagnosed failure: at the
        # fielded budget the root row measured 0.997 -- a coin flip over
        # {check, three sizes, all-in}, which ships stacks at random.
        "resolver_root_entropy": result.resolver_root_entropy,
        "played_root_entropy": result.played_root_entropy,
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


def _resolver_fields(telemetry: Any | None) -> dict[str, Any]:
    """The deployed side's stack-off and root-entropy counts, or nothing.

    Empty when side A played the bare table, so a plain match keeps the exact
    result shape -- and the tier digest -- it already had.
    """
    if telemetry is None:
        return {}
    decisions = telemetry.entropy_decisions
    return {
        "resolver_decisions": telemetry.decisions,
        "blueprint_decisions": telemetry.blueprint_decisions,
        "resolver_fallbacks": telemetry.fallbacks,
        "resolver_stack_offs": telemetry.resolver_stack_offs,
        "blueprint_stack_offs": telemetry.blueprint_stack_offs,
        "resolver_aggressions": telemetry.resolver_aggressions,
        "blueprint_aggressions": telemetry.blueprint_aggressions,
        "resolver_stack_off_rate": (
            telemetry.resolver_stack_offs / telemetry.resolver_aggressions
            if telemetry.resolver_aggressions
            else 0.0
        ),
        "blueprint_stack_off_rate": (
            telemetry.blueprint_stack_offs / telemetry.blueprint_aggressions
            if telemetry.blueprint_aggressions
            else 0.0
        ),
        "resolver_root_entropy": telemetry.resolver_entropy_sum / decisions if decisions else 0.0,
        "played_root_entropy": telemetry.played_entropy_sum / decisions if decisions else 0.0,
    }


def _threshold_side(
    storage: StaticArrayStorage, solver: StaticTreeSolver, threshold: float, side: str
) -> int:
    """Purify one side of a match in place, as the seat does at load. Rows changed."""
    if threshold <= 0.0:
        return 0
    changed, trained = apply_policy_threshold(storage, solver.tree, threshold)
    logger.info(
        "match side %s thresholded at %.3f: %d of %d trained rows changed",
        side,
        threshold,
        changed,
        trained,
    )
    return changed


def evaluate_blueprint_match(
    run_dir_a: Path,
    run_dir_b: Path,
    *,
    num_deals: int = 2000,
    seed: int = 1,
    at_a: int | None = None,
    at_b: int | None = None,
    policy_threshold_a: float = 0.0,
    policy_threshold_b: float = 0.0,
    resolver_a: bool = False,
    # Applied to side A's config when it plays deployed. Pinning the iteration
    # count is what makes two arms comparable at all: the resolver is otherwise
    # WALL-CLOCK budgeted, so a busier box does fewer iterations and lands on a
    # different strategy. `resolver_match` documents the same requirement.
    resolver_max_iterations: int | None = None,
    resolver_root_prior_weight: float | None = None,
    # The weight the deployed row puts on the resolver. ZERO is the CONTROL for an
    # off-menu opponent: side A still reaches the blueprint through the resolver's
    # lookup path -- including the shadow that translates an off-tree state -- but
    # plays the blueprint's own decisions, so any gap is the resolver's DECISIONS
    # rather than the path around them. MEASURED 09-30: a bare side A cannot be the
    # control, because it cannot play at all. Against `gears@4600` it raised
    # `KeyError: State is off-tree ... 'c-x-x-b0.25-r0.40'` on all four attempts --
    # `TreePolicySource` refuses an off-tree state by design, since a uniform
    # fallback there would average a missing strategy into a plausible score.
    resolver_blend_alpha: float | None = None,
    # The earliest street the deployed side's resolver may act on. `river` is the
    # only arm that tests SEARCH rather than the leaf guess: MEASURED 09-30, a
    # river subgame at max_depth 6 has 53 leaves and ZERO truncated, so it is the
    # whole remaining game solved exactly -- while flop and turn leave 13 of 53 to
    # an assumption that is identical at depth 6 and depth 10.
    resolver_min_street: str | None = None,
    record_source: RecordSource | None = None,
) -> EvaluationOutput:
    """Head-to-head match between two runs' blueprints on duplicate deals.

    Positive ``a_mbb_per_hand`` means run A's blueprint wins chips off run B's.
    Each blueprint is pinned to the card abstraction its run trained against.

    Both rungs are addressable because two arms are only comparable at matched
    TRAINING BUDGET: without ``at_b`` an arm that happened to be continued
    further plays its whole extra ladder, which is a different arm.

    The two policy thresholds exist because the SEAT purifies what it serves
    (0.20 per rung) while every gate to date matched raw tables, so the fielded
    configuration had never been measured on chips. Set per side, they turn a
    self-match into a single-knob A/B over serving: the sampling streams are
    shared, so identical tables give provably all-zero pairs and the only
    divergence is where the cut changed a decision.

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
    # Side A's resolver knobs, overridden HERE rather than at train time: how a
    # checkpoint is PLAYED is not a property of how it was trained. A no-op when
    # side A plays the bare table, which never reads them.
    config_a = _with_resolver_overrides(
        metadata_a.config,
        min_street=resolver_min_street,
        leaf_continuation_fraction=None,
        max_iterations=resolver_max_iterations,
        root_prior_weight=resolver_root_prior_weight,
        leaf_rollouts=None,
        blend_alpha=resolver_blend_alpha,
    )
    solver_a, storage_a, _ = build_blueprint_for(
        run_dir_a,
        dataclasses.replace(metadata_a, config=config_a),
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
    # A self-match loads the same checkpoint twice, and thresholding is in place:
    # were the two tables one array, cutting A would cut B and the match would
    # report a provable zero as "the threshold costs nothing".
    if np.shares_memory(storage_a.strategy_sum, storage_b.strategy_sum):
        raise ValueError(
            "The two blueprints share one strategy table, so a per-side policy "
            "threshold would apply to both and the match would measure nothing."
        )
    changed_a = _threshold_side(storage_a, solver_a, policy_threshold_a, "A")
    changed_b = _threshold_side(storage_b, solver_b, policy_threshold_b, "B")

    rung_a = at_a if at_a is not None else checkpoint_iteration_of(run_dir_a)
    rung_b = at_b if at_b is not None else checkpoint_iteration_of(run_dir_b)
    result = play_blueprint_match(
        solver_a, solver_b, num_deals=num_deals, seed=seed, resolver_a=resolver_a
    )
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
        "policy_threshold_a": policy_threshold_a,
        "policy_threshold_b": policy_threshold_b,
        # How much of each table the cut actually moved. A threshold that
        # changed no row is a no-op arm, and the chip edge would be an exact
        # zero for a reason that has nothing to do with the strategy.
        "rows_thresholded_a": changed_a,
        "rows_thresholded_b": changed_b,
        # Pairs where the two sides actually played differently. Duplicate deals
        # cancel exactly while the tables agree, so on a self-match under one
        # threshold this is the SAMPLE SIZE of the knob: an edge of zero over
        # zero divergent pairs means the cut never fired, not that it is free.
        "divergent_pairs": sum(1 for sample in result.pair_samples_mbb if sample != 0.0),
        # WHICH strategy side A played, and under which knobs. `opponent` is lbr's
        # existing vocabulary and already an unconditional tier knob, so reusing the
        # name is what makes a deployed arm refuse to pair with a bare one.
        "opponent": "deployed" if resolver_a else "blueprint",
        **(
            {
                "resolver_max_iterations": config_a.resolver.max_iterations,
                "root_prior_weight": config_a.resolver.root_prior_weight,
                "resolver_blend_alpha": config_a.resolver.policy_blend_alpha,
                "resolver_min_street": config_a.resolver.min_street,
            }
            if resolver_a
            else {}
        ),
        **_resolver_fields(result.telemetry),
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
