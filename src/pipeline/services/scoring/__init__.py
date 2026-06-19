"""Scoring a trained run.

One module per estimator; :func:`evaluate_and_record` is the orchestrator every
transport calls, so method dispatch, payload shape and knob-tier derivation live
in exactly one place.

The estimator labels are deliberately long: they travel into the ledger, and a
number whose instrument is only identified as "lbr" is a number nobody can audit
two months later."""

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.estimators.lbr.hunl_local_best_response import (
    LBRConfig,
)
from src.pipeline.evaluation.estimators.public_tree_br import PublicBRConfig
from src.pipeline.services.runs import load_run_metadata
from src.pipeline.services.scoring._shared import (
    EvaluationOutput,
    build_blueprint_for,
)
from src.pipeline.services.scoring.exact_br import (
    EXACT_BR_ESTIMATOR_LABEL,
    evaluate_run_exact_br,
)
from src.pipeline.services.scoring.lbr import (
    LBR_ESTIMATOR_LABEL,
    evaluate_run_lbr,
)
from src.pipeline.services.scoring.matches import (
    BLUEPRINT_MATCH_ESTIMATOR_LABEL,
    RESOLVER_GATE_ESTIMATOR_LABEL,
    evaluate_blueprint_match,
    evaluate_run_resolver_gate,
    record_blueprint_match,
)
from src.shared import records

PROGRESS_ARTIFACT = "evaluate-progress.json"

logger = logging.getLogger(__name__)


class EvaluationPayload(BaseModel):
    """What one evaluation of one checkpoint measured.

    NODE-ONLY: `score` is the console's door, and the durable record is the
    per-run document this writes, not this payload. `ledger_result_path` is
    absent when recording failed -- which is deliberate and must stay possible,
    because a failed WRITE must never lose the measurement that was made.
    """

    op: Literal["evaluate"] = "evaluate"
    run_id: str
    method: str
    estimator: str
    infosets: int
    checkpoint_iteration: int | None = None
    results: dict[str, Any] = Field(default_factory=dict)
    ledger_result_path: str | None = None


def evaluate_and_record(
    run_dir: Path,
    *,
    method: str = "lbr",
    lbr: LBRConfig | None = None,
    exact_br: PublicBRConfig | None = None,
    resolver_iterations: int = 64,
    resolver_gate_deals: int = 1000,
    resolver_gate_workers: int = 1,
    leaf_continuation_fraction: float | None = None,
    resolver_max_iterations: int | None = None,
    resolver_allin_runouts: int = 1,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
    progress_file: Path | None = None,
    policy_profile: bool = False,
) -> EvaluationPayload:
    """Evaluate a run and persist the result to the eval ledger (best-effort).

    The single evaluate orchestrator every transport calls: method dispatch,
    payload shape, knob-tier derivation and best-effort ledger recording live
    here once, so a cloud eval and a local eval cannot drift.

    Returns the portable evaluate payload; when recording succeeded it carries
    ``ledger_result_path``. Recording failures print a warning but never fail
    the evaluation itself — the ledger is a research convenience.

    ``at_iteration`` scores a retained ladder rung rather than the published
    snapshot; each rung records its own ``checkpoint_iteration``, so a run's
    convergence curve arrives as ordinary ledger rows.
    """
    if method == "exact_br":
        br_config = exact_br or PublicBRConfig()
        out = evaluate_run_exact_br(
            run_dir,
            br_config,
            abstraction_hash=abstraction_hash,
            at_iteration=at_iteration,
            # Flop branches, the outermost thing the walk counts: four walks of
            # `--br-flops` each, so the default 8 gives 32 steps. The bar it
            # replaced said `0 / 1 rungs`, which read 0% for the whole ~10
            # minutes and made a long score look exactly like a hung one.
            on_branch=records.progress_writer(progress_file, records.REGISTRY[PROGRESS_ARTIFACT]),
            policy_profile=policy_profile,
        )
        estimator = EXACT_BR_ESTIMATOR_LABEL
        knobs = eval_ledger.build_exact_br_knobs_from_params(
            num_flops=br_config.num_flops,
            num_turns=br_config.num_turns,
            num_rivers=br_config.num_rivers,
            board_seed=br_config.board_seed,
            in_abstraction=br_config.in_abstraction,
            policy_threshold=br_config.policy_threshold,
            purify=br_config.purify,
        )
    elif method == "resolver_match":
        # Not an exploitability estimate at all: a head-to-head chip edge of
        # blueprint+resolver over the bare blueprint, on duplicate deals. It
        # carries no `exploitability_mbb`, which is why `evaluate.render` picks
        # its branch off the method rather than reaching into `results`.
        out = evaluate_run_resolver_gate(
            run_dir,
            num_deals=resolver_gate_deals,
            leaf_continuation_fraction=leaf_continuation_fraction,
            max_iterations=resolver_max_iterations,
            workers=resolver_gate_workers,
            allin_runouts=resolver_allin_runouts,
        )
        estimator = RESOLVER_GATE_ESTIMATOR_LABEL
        knobs = eval_ledger.build_resolver_match_knobs(out.results)
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
    payload = EvaluationPayload(
        run_id=run_dir.name,
        method=method,
        estimator=estimator,
        infosets=out.infosets,
        checkpoint_iteration=out.checkpoint_iteration,
        results=out.results,
    )
    try:
        metadata = load_run_metadata(run_dir)
        result_path, _ = eval_ledger.record_evaluation(
            run_dir=run_dir,
            payload=payload.model_dump(),
            provenance=eval_ledger.RunProvenance(
                run_id=metadata.run_id,
                git_commit=metadata.git_commit,
                git_dirty=metadata.git_dirty,
                git_branch=metadata.git_branch,
                code_snapshot=metadata.code_snapshot,
                config_name=metadata.config_name,
                card_abstraction_hash=metadata.card_abstraction_hash,
                action_config_hash=metadata.action_config_hash,
                experiment_id=metadata.experiment_id,
                arm=metadata.arm,
                parent_run_id=metadata.parent_run_id,
                config_hash=metadata.config_hash,
            ),
            method=method,
            estimator=estimator,
            knobs=knobs,
        )
        payload.ledger_result_path = str(result_path)
        logger.info(f"  Recorded:      {result_path}")
    except Exception as exc:  # recording must never break the eval  # noqa: BLE001 -- recording must never break the eval it records
        logger.warning(f"  Ledger:        skipped ({type(exc).__name__}: {exc})")
    return payload


__all__ = (
    "BLUEPRINT_MATCH_ESTIMATOR_LABEL",
    "EXACT_BR_ESTIMATOR_LABEL",
    "LBR_ESTIMATOR_LABEL",
    "RESOLVER_GATE_ESTIMATOR_LABEL",
    "EvaluationOutput",
    "build_blueprint_for",
    "evaluate_and_record",
    "evaluate_blueprint_match",
    "evaluate_run_exact_br",
    "evaluate_run_lbr",
    "evaluate_run_resolver_gate",
    "record_blueprint_match",
)
