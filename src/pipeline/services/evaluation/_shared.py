"""Pieces every estimator needs: loading a scoreable blueprint, and the payload type."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.policy_source import ScorableBlueprint
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.pipeline.training.components import build_static_evaluation_solver
from src.pipeline.training.run_tracker import RunMetadata
from src.shared.config import Config

logger = logging.getLogger(__name__)


def build_blueprint_for(
    run_dir: Path,
    metadata: RunMetadata,
    abstraction_hash: str | None,
    at_iteration: int | None,
) -> tuple[StaticTreeSolver, StaticArrayStorage]:
    """Load a scoreable blueprint from a run's static checkpoint."""
    return build_static_evaluation_solver(
        metadata.config,
        checkpoint_dir=run_dir,
        abstraction_hash=abstraction_hash,
        at_iteration=at_iteration,
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
) -> ScorableBlueprint:
    """Build a fresh evaluation blueprint (solver) from a checkpoint.

    Used as a picklable factory (via ``functools.partial``) so parallel scoring
    worker processes each construct their own solver — the solver holds a
    non-picklable Cython member and cannot be sent across a process boundary.
    """
    solver, _ = build_static_evaluation_solver(
        config,
        checkpoint_dir=checkpoint_dir,
        abstraction_hash=abstraction_hash,
        at_iteration=at_iteration,
    )
    return solver
