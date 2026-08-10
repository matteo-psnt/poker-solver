"""Local Best Response -- the trustworthy default metric."""

import functools
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.pipeline.evaluation.estimators.lbr.hunl_local_best_response import (
    LBRConfig,
    LBRResult,
    compute_lbr_exploitability,
    dominant_terminal,
)
from src.pipeline.evaluation.statistics import variance_decomposition
from src.pipeline.evaluation.units import pair_mean_mbb
from src.pipeline.services.runs import checkpoint_iteration_of, load_run_metadata
from src.pipeline.services.scoring._shared import (
    EvaluationOutput,
    _effective_abstraction_hash,
    _load_blueprint,
    build_blueprint_for,
)

logger = logging.getLogger(__name__)

LBR_ESTIMATOR_LABEL = "local_best_response (rigorous lower bound on exploitability)"


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
    loader = _load_blueprint
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
