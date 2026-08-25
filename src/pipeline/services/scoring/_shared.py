"""Pieces every estimator needs: loading a scoreable blueprint, and the payload type.

Private to this package -- the leading underscore is on the MODULE, so nothing
inside it needs one too. When the members carried it as well, two siblings
imported four underscore-prefixed names across a module boundary, which reads as
reaching into another file's privates rather than using the package's own seam.
"""

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.policy.source import ScorableBlueprint
from src.engine.solver.storage.policy_assembly import PolicyIterate
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.pipeline.blueprint.construction import build_static_evaluation_solver
from src.pipeline.services.runs import load_run_metadata
from src.pipeline.training.run_tracker import RunMetadata
from src.shared.config import Config

logger = logging.getLogger(__name__)


def build_blueprint_for(
    run_dir: Path,
    metadata: RunMetadata,
    abstraction_hash: str | None,
    at_iteration: int | None,
    policy_iterate: PolicyIterate = "average",
    avg_window_from: int | None = None,
    avg_gamma: float | None = None,
    mix: tuple[Path | None, int | None, float] = (None, None, 0.5),
) -> tuple[StaticTreeSolver, StaticArrayStorage, dict[str, Any]]:
    """Load a scoreable blueprint from a run's static checkpoint."""
    return build_static_evaluation_solver(
        metadata.config,
        checkpoint_dir=run_dir,
        abstraction_hash=abstraction_hash,
        at_iteration=at_iteration,
        policy_iterate=policy_iterate,
        avg_window_from=avg_window_from,
        avg_gamma=avg_gamma,
        mix_run=mix[0],
        mix_at=mix[1],
        mix_weight=mix[2],
    )


@dataclass(frozen=True)
class EvaluationOutput:
    """Container for run evaluation output."""

    infosets: int
    results: dict[str, Any]
    # Which checkpoint was actually scored, from the manifest committed atomically
    # with the arrays. Without it a stale read -- evaluating a checkpoint written
    # before a still-running task's newer one -- is indistinguishable from a real
    # result, which is exactly how a 10M-iteration checkpoint was once silently
    # reported as the score of a 16M-iteration run. None only for pre-manifest runs.
    checkpoint_iteration: int | None = None
    # The BETTING TREE this eval walked, as `BettingTree.fingerprint`. Eval-time,
    # unlike the run's `action_config_hash`, which is the tree it was TRAINED on:
    # a rules change moves this and leaves that alone, so without it two rows at
    # matched knobs from either side of the limp fix are indistinguishable.
    tree_fingerprint: str | None = None


def effective_abstraction_hash(
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


def load_blueprint(
    config: Config,
    checkpoint_dir: Path,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
    policy_iterate: PolicyIterate = "average",
    avg_window_from: int | None = None,
    avg_gamma: float | None = None,
    mix: tuple[Path | None, int | None, float] = (None, None, 0.5),
) -> ScorableBlueprint:
    """Build a fresh evaluation blueprint (solver) from a checkpoint.

    Used as a picklable factory (via ``functools.partial``) so parallel scoring
    worker processes each construct their own solver — the solver holds a
    non-picklable Cython member and cannot be sent across a process boundary.
    """
    solver, _, _ = build_static_evaluation_solver(
        config,
        checkpoint_dir=checkpoint_dir,
        abstraction_hash=abstraction_hash,
        at_iteration=at_iteration,
        policy_iterate=policy_iterate,
        avg_window_from=avg_window_from,
        avg_gamma=avg_gamma,
        mix_run=mix[0],
        mix_at=mix[1],
        mix_weight=mix[2],
    )
    return solver


@dataclass(frozen=True)
class PreparedBlueprint:
    """What an estimator needs before it can score.

    ``policy_record`` is empty unless the checkpoint was reassembled into a
    non-default strategy (the current iterate, a windowed average); it carries
    the fields that keep such a row from pairing with a plain-average one.
    """

    metadata: RunMetadata
    solver: StaticTreeSolver
    storage: StaticArrayStorage
    factory: Callable[[], ScorableBlueprint] | None
    policy_record: dict[str, Any]


def prepare_blueprint(
    run_dir: Path,
    abstraction_hash: str | None,
    at_iteration: int | None,
    num_workers: int,
    policy_iterate: PolicyIterate = "average",
    avg_window_from: int | None = None,
    avg_gamma: float | None = None,
    mix: tuple[Path | None, int | None, float] = (None, None, 0.5),
) -> PreparedBlueprint:
    """Everything an estimator needs before it can score: metadata, blueprint, factory.

    Both estimators opened with the same five statements, and both carried a
    comment apologising for it ("Same factory shape parallel LBR uses" /
    "matching the blueprint above"). They have to agree: a factory built from a
    different ``effective_hash`` than the in-process solver would have workers
    scoring a differently-bucketed blueprint from the coordinator, and the
    result would look like an ordinary number.

    The factory is None below two workers -- there is no subprocess to rebuild
    anything -- and otherwise captures only picklable arguments.
    """
    metadata = load_run_metadata(run_dir)
    effective_hash = effective_abstraction_hash(run_dir, metadata, abstraction_hash)
    solver, storage, policy_record = build_blueprint_for(
        run_dir,
        metadata,
        effective_hash,
        at_iteration,
        policy_iterate,
        avg_window_from,
        avg_gamma,
        mix,
    )
    factory = (
        functools.partial(
            load_blueprint,
            metadata.config,
            run_dir,
            effective_hash,
            at_iteration,
            policy_iterate,
            avg_window_from,
            avg_gamma,
            mix,
        )
        if num_workers > 1
        else None
    )
    return PreparedBlueprint(metadata, solver, storage, factory, policy_record)
