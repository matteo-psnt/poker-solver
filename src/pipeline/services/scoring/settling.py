"""Settling: how much a run's average strategy still moves between two rungs.

Not an exploitability estimate and deliberately not shaped like one -- it needs
no best response and no board plan, so it can be read at every retained rung for
the price of two checkpoint loads. Its job is to answer "has this stopped
changing", which exploitability cannot, and to be read BESIDE exploitability,
which is the only thing that says whether what it settled on is any good.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.pipeline.blueprint.preflop import preflop_chart
from src.pipeline.evaluation.estimators.settling import SettlingResult, measure_settling
from src.pipeline.services.runs import load_run_metadata
from src.pipeline.services.scoring._shared import (
    EvaluationOutput,
    build_blueprint_for,
    effective_abstraction_hash,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.shared.ports.record import RecordSource

SETTLING_ESTIMATOR_LABEL = (
    "settling (per-street total-variation drift of the average strategy between two "
    "rungs, reach-weighted; NOT an exploitability bound)"
)


def evaluate_run_settling(
    run_dir: Path,
    *,
    from_iteration: int,
    at_iteration: int | None = None,
    abstraction_hash: str | None = None,
    record_source: RecordSource | None = None,
) -> EvaluationOutput:
    """Drift between two rungs of one run, per street.

    Both rungs are loaded ``play_only``: the measurement reads the average
    strategy and the reach behind it, never regrets, and two tables are resident
    at once where every other estimator holds one.

    Raises:
        ValueError: If ``from_iteration`` is not below the rung being measured.
    """
    metadata = load_run_metadata(run_dir, record_source)
    effective = effective_abstraction_hash(run_dir, metadata, abstraction_hash)
    if at_iteration is not None and from_iteration >= at_iteration:
        raise ValueError(
            f"--settle-from must name an EARLIER rung than the one measured: "
            f"{from_iteration} is not below {at_iteration}."
        )

    _early_solver, early, _ = build_blueprint_for(
        run_dir, metadata, effective, from_iteration, play_only=True
    )
    late_solver, late, _ = build_blueprint_for(
        run_dir, metadata, effective, at_iteration, play_only=True
    )
    measured: SettlingResult = measure_settling(
        early,
        late,
        from_iteration=from_iteration,
        at_iteration=at_iteration if at_iteration is not None else -1,
    )
    results = measured.as_dict()
    # The GAP is the tier: drift over 400 rungs and drift over 2,000 rungs are
    # not two readings of one thing, and the pair is what a curve is plotted on.
    results["rung_gap"] = at_iteration - from_iteration if at_iteration is not None else None
    # The opening spot class by class, so a ladder of settle rows is also the
    # trajectory of the one node a person reads -- 169 rows against 45M.
    opening = preflop_chart(late_solver)
    results["opening"] = {
        "actions": list(opening.actions),
        "classes": {label: list(row.strategy) for label, row in sorted(opening.classes.items())},
    }
    return EvaluationOutput(
        infosets=late.num_infosets(),
        results=results,
        checkpoint_iteration=at_iteration,
    )
