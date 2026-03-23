"""Scoring a trained run.

Each estimator gets one entrypoint returning an :class:`EvaluationOutput`;
:func:`evaluate_and_record` is the orchestrator every transport calls, so
method dispatch, payload shape, and knob-tier derivation live here once.

The estimator labels are deliberately long: they travel into the ledger, and a
number whose instrument is only identified as "lbr" is a number nobody can
audit two months later.
"""

import functools
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.engine.solver.policy_source import ScorableBlueprint
from src.engine.solver.protocols import Blueprint
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.blueprint_match import play_blueprint_match
from src.pipeline.evaluation.hunl_local_best_response import (
    LBRConfig,
    LBRResult,
    compute_lbr_exploitability,
    dominant_terminal,
)
from src.pipeline.evaluation.public_tree_br import PublicBRConfig, compute_public_tree_br
from src.pipeline.evaluation.resolver_match import play_resolver_match
from src.pipeline.evaluation.statistics import variance_decomposition
from src.pipeline.services.runs import checkpoint_iteration_of, load_run_metadata
from src.pipeline.training.components import (
    build_evaluation_solver,
    build_static_evaluation_solver,
    evaluate_solver_exploitability,
)
from src.pipeline.training.run_tracker import RunMetadata
from src.shared.config import Config
from src.shared.units import pair_mean_mbb

logger = logging.getLogger(__name__)

# The two backends need different loaders (hashed key vs (node_id, bucket)), but
# everything above `policy_source_for` is shared, so the split stops here.
STATIC_MANIFEST = "STATIC_CHECKPOINT.json"


def is_static_run(run_dir: Path) -> bool:
    """Whether ``run_dir`` holds a static-tree checkpoint.

    Detected from the artifact rather than a flag on the run: a flag can be
    absent on runs written before it existed, while the manifest is what the
    loader actually needs.
    """
    return (run_dir / STATIC_MANIFEST).exists()


def build_blueprint_for(
    run_dir: Path,
    metadata: RunMetadata,
    abstraction_hash: str | None,
    at_iteration: int | None,
):
    """Load a scoreable blueprint from either backend."""
    if is_static_run(run_dir):
        return build_static_evaluation_solver(
            metadata.config,
            checkpoint_dir=run_dir,
            abstraction_hash=abstraction_hash,
            at_iteration=at_iteration,
        )
    return build_evaluation_solver(
        metadata.config,
        checkpoint_dir=run_dir,
        abstraction_hash=abstraction_hash,
        at_iteration=at_iteration,
    )


# Local Best Response: a rigorous lower bound on exploitability (LBR <= exact BR,
# validated on Kuhn/Leduc). This is the trustworthy default metric.
LBR_ESTIMATOR_LABEL = "local_best_response (rigorous lower bound on exploitability)"


# The legacy `evaluate_run` metric is a one-ply rollout that both understates the
# structure it explores AND is upward-biased by a recursive max over noisy MC
# estimates — it is not a valid bound in either direction. Kept as an explicit
# opt-in for diagnostics/comparison only; do NOT treat it as exploitability.
ROLLOUT_ESTIMATOR_LABEL = "rollout_one_ply (uninformative; not a valid bound — diagnostic only)"


# Exact best response on a deterministic sampled public tree: zero evaluation
# variance, exactly paired across checkpoints under one board plan. The absolute
# value is exploitability of the board-restricted game, not full HUNL — compare
# within a tier, don't quote as a bound on the full game.
EXACT_BR_ESTIMATOR_LABEL = "public_tree_exact_br (deterministic exact BR on sampled public tree)"


BLUEPRINT_MATCH_ESTIMATOR_LABEL = (
    "blueprint_match (duplicate-deal head-to-head chip edge; abstraction-safe, not a bound)"
)


@dataclass(frozen=True)
class EvaluationOutput:
    """Container for run evaluation output."""

    infosets: int
    results: dict[str, Any]
    # Which checkpoint was actually scored, from the manifest committed atomically
    # with the arrays. Without it a stale read -- evaluating a checkpoint written
    # before a still-running leg's newer one -- is indistinguishable from a real
    # result, which is exactly how a 10M-iteration checkpoint was once silently
    # reported as the score of a 16M-iteration run. None only for pre-manifest runs.
    checkpoint_iteration: int | None = None


@dataclass(frozen=True)
class RolloutParams:
    """Settings for the legacy one-ply rollout estimator (diagnostic opt-in only)."""

    num_samples: int = 500
    num_rollouts: int = 50
    use_average_strategy: bool = True
    seed: int | None = None


def _effective_abstraction_hash(
    run_dir: Path, metadata: RunMetadata, abstraction_hash: str | None
) -> str:
    """The abstraction hash an eval must pin to, refusing unpinnable runs."""
    effective = abstraction_hash or metadata.card_abstraction_hash
    if effective is None:
        raise ValueError(
            f"Run '{run_dir.name}' does not record which card abstraction it was trained "
            "against, so it cannot be evaluated faithfully: resolving by config name alone "
            "would silently rebucket the checkpoint under whatever abstraction that name "
            "now points at, yielding plausible but invalid numbers.\n"
            "Pass abstraction_hash explicitly if you know it (see the abstraction's "
            "metadata.json 'config_hash')."
        )
    return effective


def _load_blueprint(
    config: Config,
    checkpoint_dir: Path,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
) -> Blueprint:
    """Build a fresh evaluation blueprint (solver) from a checkpoint.

    Used as a picklable factory (via ``functools.partial``) so parallel-LBR worker
    processes each construct their own solver — the solver holds a non-picklable
    Cython member and cannot be sent across a process boundary.
    """
    solver, _ = build_evaluation_solver(
        config,
        checkpoint_dir=checkpoint_dir,
        abstraction_hash=abstraction_hash,
        at_iteration=at_iteration,
    )
    return solver


def _load_static_blueprint(
    config: Config,
    checkpoint_dir: Path,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
) -> ScorableBlueprint:
    """Picklable factory for a static blueprint, for parallel scoring workers."""
    solver, _ = build_static_evaluation_solver(
        config,
        checkpoint_dir=checkpoint_dir,
        abstraction_hash=abstraction_hash,
        at_iteration=at_iteration,
    )
    return solver


def _lbr_results_dict(result: LBRResult, big_blind: int) -> dict[str, Any]:
    """Map an LBRResult into the portable results dict.

    Per-hand records, ready-made paired samples, and the base seed travel with
    the aggregate so a later paired (common-random-numbers) comparison against
    another run — or an offline variance decomposition — never requires
    re-running the eval or re-deriving the sample definition.
    """
    samples_mbb = [pair_mean_mbb(o0.value, o1.value, big_blind) for o0, o1 in result.hand_outcomes]
    groups = [dominant_terminal(o0.terminal, o1.terminal) for o0, o1 in result.hand_outcomes]
    return {
        "exploitability_mbb": result.exploitability_mbb,
        "exploitability_bb": result.exploitability_bb,
        "std_error_mbb": result.std_error_mbb,
        "confidence_95_mbb": result.confidence_95_mbb,
        "lbr_utility_p0": result.lbr_utility_p0,
        "lbr_utility_p1": result.lbr_utility_p1,
        "num_hands": result.num_hands,
        "base_seed": result.base_seed,
        "big_blind": big_blind,
        "pair_samples_mbb": samples_mbb,
        "hand_records": [
            {
                "u0": o0.value,
                "u1": o1.value,
                "terminal_p0": o0.terminal,
                "terminal_p1": o1.terminal,
                "pot_p0": o0.pot,
                "pot_p1": o1.pot,
            }
            for o0, o1 in result.hand_outcomes
        ],
        "variance_decomposition": (
            variance_decomposition(samples_mbb, groups) if samples_mbb else None
        ),
    }


def evaluate_run_lbr(
    run_dir: Path,
    config: LBRConfig | None = None,
    *,
    resolver_iterations: int = 64,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
) -> EvaluationOutput:
    """Evaluate a run's exploitability via Local Best Response (trustworthy default).

    LBR is a rigorous lower bound on true exploitability (LBR <= exact BR, validated
    on Kuhn/Leduc). Every eval knob — hand count, scorer, opponent model, off-tree
    menu, parallelism — travels in ``config`` (:class:`LBRConfig`), so transports
    construct one object instead of relisting knobs; see the LBRConfig field docs
    for the semantics and comparison-tier rules of each knob.

    Only two knobs stay outside ``config`` because they are resolved against the
    run itself: ``abstraction_hash`` (pin the card abstraction; defaults to the
    run's recorded hash) and ``resolver_iterations``. For ``config.opponent ==
    "deployed"`` the resolver settings come from the run's own config with
    ``max_iterations`` pinned to ``resolver_iterations`` — iteration-pinned (not
    wall-clock) so the measured strategy is machine-independent and CRN pairing
    stays valid.

    The results dict carries per-hand records plus the base seed; evaluate two runs
    with the same explicit ``config.seed`` and feed the per-hand samples to
    :func:`~src.pipeline.evaluation.statistics.compare_paired_samples` for a paired
    comparison that resolves far smaller gaps than two independent intervals.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid configuration or checkpoint state.
    """
    config = config or LBRConfig()
    metadata = load_run_metadata(run_dir)
    effective_hash = _effective_abstraction_hash(run_dir, metadata, abstraction_hash)
    solver, storage = build_blueprint_for(run_dir, metadata, effective_hash, at_iteration)
    # For parallel LBR each worker rebuilds its own solver from the checkpoint (the
    # solver is not picklable across processes); the factory captures only picklable
    # args (config + checkpoint dir). Loader chosen per backend, matching the
    # blueprint above -- a static run loaded by the key-addressed loader dies on a
    # missing checkpoint.zarr, which is what LBR-on-static did before this.
    loader = _load_static_blueprint if is_static_run(run_dir) else _load_blueprint
    factory = (
        functools.partial(loader, metadata.config, run_dir, effective_hash, at_iteration)
        if config.num_workers > 1
        else None
    )
    if config.opponent == "deployed":
        config = replace(
            config,
            resolver=metadata.config.resolver.model_copy(
                update={"max_iterations": resolver_iterations}
            ),
        )
    result = compute_lbr_exploitability(solver, config, blueprint_factory=factory)
    results = _lbr_results_dict(result, big_blind=metadata.config.game.big_blind)
    results["opponent_model"] = config.opponent
    results["scorer"] = config.scorer
    if config.scorer == "lookahead":
        results["lookahead_depth"] = config.lookahead_depth
        results["lookahead_top_k"] = config.lookahead_top_k
    if config.resolver is not None:
        results["resolver_iterations"] = config.resolver.max_iterations
        results["resolver_blend_alpha"] = config.resolver.policy_blend_alpha
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir, at_iteration),
    )


def evaluate_run_exact_br(
    run_dir: Path,
    config: PublicBRConfig | None = None,
    *,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
) -> EvaluationOutput:
    """Exact best response on the sampled public tree (deterministic point value).

    Zero evaluation variance: the same checkpoint under the same
    :class:`PublicBRConfig` always scores identically, so two checkpoints in one
    tier are exactly paired — a difference is pure signal, with no hand budget
    or p-value involved. The value is the exploitability of the board-sampled
    restricted game (see :mod:`~src.pipeline.evaluation.public_tree_br`), not of
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
    loader = _load_static_blueprint if is_static_run(run_dir) else _load_blueprint
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


def evaluate_run_resolver_gate(
    run_dir: Path,
    *,
    num_deals: int = 1000,
    time_budget_ms: int = 100,
    seed: int = 1,
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
    metadata = load_run_metadata(run_dir)
    solver, storage = build_evaluation_solver(metadata.config, checkpoint_dir=run_dir)
    result = play_resolver_match(
        solver,
        num_deals=num_deals,
        time_budget_ms=time_budget_ms,
        seed=seed,
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
        "seed": seed,
        "pair_samples_mbb": result.pair_samples_mbb,
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
) -> EvaluationOutput:
    """Head-to-head match between two runs' blueprints on duplicate deals.

    Positive ``a_mbb_per_hand`` means run A's blueprint wins chips off run B's.
    Each blueprint is pinned to the card abstraction its run trained against.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid checkpoint state, or mismatched game configurations.
    """
    metadata_a = load_run_metadata(run_dir_a)
    metadata_b = load_run_metadata(run_dir_b)
    if metadata_a.config.game != metadata_b.config.game:
        raise ValueError(
            f"Game configs differ between runs ({metadata_a.config.game} vs "
            f"{metadata_b.config.game}); a chip match would be meaningless."
        )

    solver_a, storage_a = build_evaluation_solver(
        metadata_a.config,
        checkpoint_dir=run_dir_a,
        abstraction_hash=metadata_a.card_abstraction_hash,
    )
    solver_b, storage_b = build_evaluation_solver(
        metadata_b.config,
        checkpoint_dir=run_dir_b,
        abstraction_hash=metadata_b.card_abstraction_hash,
    )
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
        "pair_samples_mbb": result.pair_samples_mbb,
    }
    return EvaluationOutput(
        infosets=storage_a.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir_a),
    )


def record_blueprint_match(
    run_dir_a: Path,
    run_dir_b: Path,
    *,
    num_deals: int = 2000,
    seed: int = 1,
) -> dict[str, Any]:
    """Run a head-to-head match and persist a durable, self-describing payload.

    A blueprint match is inherently PAIRWISE, so it does not fit the single-run eval
    ledger (``evaluate_and_record``); instead the full result is written non-clobbering
    under run A's dir with BOTH runs' provenance embedded — the card-abstraction hashes
    especially, since a chip edge is uninterpretable later without recording which two
    abstractions were compared. The payload (not any ledger row) is the durable record,
    so a client guillotine after the match commits loses nothing.

    Returns the payload; the caller (Modal) commits the Volume so it survives the
    container.
    """
    metadata_a = load_run_metadata(run_dir_a)
    metadata_b = load_run_metadata(run_dir_b)
    out = evaluate_blueprint_match(run_dir_a, run_dir_b, num_deals=num_deals, seed=seed)

    def _provenance(meta: RunMetadata) -> dict[str, Any]:
        return {
            "run_id": meta.run_id,
            "git_commit": meta.git_commit,
            "git_dirty": meta.git_dirty,
            "config_name": meta.config_name,
            "card_abstraction_hash": meta.card_abstraction_hash,
            "action_config_hash": meta.action_config_hash,
            "representation_version": meta.representation_version,
        }

    payload: dict[str, Any] = {
        "op": "blueprint_match",
        "run_a": metadata_a.run_id,
        "run_b": metadata_b.run_id,
        "estimator": BLUEPRINT_MATCH_ESTIMATOR_LABEL,
        "provenance_a": _provenance(metadata_a),
        "provenance_b": _provenance(metadata_b),
        "checkpoint_iteration_a": checkpoint_iteration_of(run_dir_a),
        "checkpoint_iteration_b": checkpoint_iteration_of(run_dir_b),
        "infosets": out.infosets,
        "results": out.results,
    }
    knobs = {"run_b": metadata_b.run_id, "num_deals": num_deals, "base_seed": seed}
    try:
        result_path = eval_ledger.write_payload(run_dir_a, payload, eval_ledger.eval_slug(knobs))
        payload["result_path"] = str(result_path)
    except OSError as exc:  # recording is a research convenience; never fail the match
        logger.warning("Blueprint-match payload not written: %s", exc)
    return payload


def evaluate_run_rollout(
    run_dir: Path,
    params: RolloutParams | None = None,
) -> EvaluationOutput:
    """Evaluate a run with the legacy one-ply rollout estimator (diagnostic opt-in only).

    NOT a valid exploitability bound (see ``ROLLOUT_ESTIMATOR_LABEL``); prefer
    :func:`evaluate_run_lbr`. Kept for comparison/diagnostics.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid configuration or checkpoint state.
    """
    params = params or RolloutParams()
    metadata = load_run_metadata(run_dir)
    config = metadata.config

    solver, storage = build_evaluation_solver(
        config,
        checkpoint_dir=run_dir,
    )
    results = evaluate_solver_exploitability(
        solver,
        num_samples=params.num_samples,
        use_average_strategy=params.use_average_strategy,
        num_rollouts_per_infoset=params.num_rollouts,
        seed=params.seed,
    )
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir),
    )


def evaluate_and_record(
    run_dir: Path,
    *,
    method: str = "lbr",
    lbr: LBRConfig | None = None,
    rollout: RolloutParams | None = None,
    exact_br: PublicBRConfig | None = None,
    resolver_iterations: int = 64,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
    ledger_path: Path = eval_ledger.DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Evaluate a run and persist the result to the eval ledger (best-effort).

    The single evaluate orchestrator shared by every transport (headless CLI,
    Modal): method dispatch, payload shape, knob-tier derivation, and the
    best-effort ledger recording live here once, so a cloud eval and a local
    eval cannot drift in what they run or record.

    Returns the portable evaluate payload; when recording succeeded it carries
    ``ledger_result_path``. Recording failures print a warning but never fail
    the evaluation itself — the ledger is a research convenience.

    ``at_iteration`` scores a retained ladder rung rather than the published
    snapshot; each rung records its own ``checkpoint_iteration``, so a run's
    convergence curve arrives as ordinary ledger rows.
    """
    if method == "rollout":
        if at_iteration is not None:
            # Refuse rather than silently score the published snapshot and label the
            # row with a rung that was never loaded.
            raise ValueError(
                "--at is not supported for method 'rollout' (a diagnostic estimator, "
                "not a curve tool); use exact_br for within-run convergence curves."
            )
        params = rollout or RolloutParams()
        out = evaluate_run_rollout(run_dir, params)
        estimator = ROLLOUT_ESTIMATOR_LABEL
        knobs = eval_ledger.build_rollout_knobs_from_params(
            samples=params.num_samples,
            rollouts=params.num_rollouts,
            use_current=not params.use_average_strategy,
            base_seed=out.results.get("base_seed", params.seed),
        )
    elif method == "exact_br":
        br_config = exact_br or PublicBRConfig()
        out = evaluate_run_exact_br(
            run_dir, br_config, abstraction_hash=abstraction_hash, at_iteration=at_iteration
        )
        estimator = EXACT_BR_ESTIMATOR_LABEL
        knobs = eval_ledger.build_exact_br_knobs_from_params(
            num_flops=br_config.num_flops,
            num_turns=br_config.num_turns,
            num_rivers=br_config.num_rivers,
            board_seed=br_config.board_seed,
        )
    else:  # "lbr" (default, trustworthy)
        config = lbr or LBRConfig()
        out = evaluate_run_lbr(
            run_dir,
            config,
            resolver_iterations=resolver_iterations,
            abstraction_hash=abstraction_hash,
            at_iteration=at_iteration,
        )
        estimator = LBR_ESTIMATOR_LABEL
        knobs = eval_ledger.build_lbr_knobs(config, out.results)
    payload: dict[str, Any] = {
        "op": "evaluate",
        "run_id": run_dir.name,
        "method": method,
        "estimator": estimator,
        "infosets": out.infosets,
        "checkpoint_iteration": out.checkpoint_iteration,
        "results": out.results,
    }
    try:
        metadata = load_run_metadata(run_dir)
        result_path, _ = eval_ledger.record_evaluation(
            run_dir=run_dir,
            payload=payload,
            provenance=eval_ledger.RunProvenance(
                run_id=metadata.run_id,
                git_commit=metadata.git_commit,
                git_dirty=metadata.git_dirty,
                config_name=metadata.config_name,
                card_abstraction_hash=metadata.card_abstraction_hash,
                action_config_hash=metadata.action_config_hash,
                representation_version=metadata.representation_version,
                experiment_id=metadata.experiment_id,
                arm=metadata.arm,
                parent_run_id=metadata.parent_run_id,
                config_hash=metadata.config_hash,
            ),
            method=method,
            estimator=estimator,
            knobs=knobs,
            ledger_path=ledger_path,
        )
        payload["ledger_result_path"] = str(result_path)
        logger.info(f"  Ledger:        appended to {ledger_path} (payload: {result_path})")
    except Exception as exc:  # recording must never break the eval
        logger.warning(f"  Ledger:        skipped ({type(exc).__name__}: {exc})")
    return payload
