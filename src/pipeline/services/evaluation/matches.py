"""Head-to-head gates: run against run, and blueprint against deployed resolver."""

import logging
from pathlib import Path
from typing import Any

from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.blueprint_match import play_blueprint_match
from src.pipeline.evaluation.resolver_match import play_resolver_match
from src.pipeline.services.evaluation._shared import (
    EvaluationOutput,
)
from src.pipeline.services.runs import checkpoint_iteration_of, load_run_metadata
from src.pipeline.training.components import build_static_evaluation_solver
from src.pipeline.training.run_tracker import RunMetadata

logger = logging.getLogger(__name__)

BLUEPRINT_MATCH_ESTIMATOR_LABEL = (
    "blueprint_match (duplicate-deal head-to-head chip edge; abstraction-safe, not a bound)"
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
    solver, storage = build_static_evaluation_solver(metadata.config, checkpoint_dir=run_dir)
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

    solver_a, storage_a = build_static_evaluation_solver(
        metadata_a.config,
        checkpoint_dir=run_dir_a,
        abstraction_hash=metadata_a.card_abstraction_hash,
    )
    solver_b, storage_b = build_static_evaluation_solver(
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
