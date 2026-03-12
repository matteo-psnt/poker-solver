"""Starting and continuing training runs.

The single train/resume orchestrator shared by every transport (headless CLI,
Modal, the Azure leg wrapper), so a cloud run and a local run cannot drift.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline.services.runs import load_run_metadata
from src.pipeline.training.abstraction_resolver import AbstractionHashMismatchError
from src.pipeline.training.run_tracker import ExperimentTag
from src.pipeline.training.trainer import TrainingSession
from src.shared.config import Config
from src.shared.config_loader import load_training_config


@dataclass(frozen=True)
class TrainingOutput:
    """Machine-readable summary of a completed training run.

    ``run_id`` is a portable identifier (the run directory's name relative to
    ``runs_dir``), never an absolute path, so a follow-up evaluate/resume call
    can locate the run regardless of where a volume is mounted.
    """

    run_id: str
    runs_dir: str
    config_name: str
    iterations: int
    num_infosets: int
    runtime_seconds: float
    iterations_per_second: float
    storage_capacity: int
    status: str


@dataclass(frozen=True)
class ResumeOutput:
    """Machine-readable summary of a resume leg.

    ``no_op`` marks a leg that found the checkpoint already at or past its target
    and changed nothing — what a retried attempt sees.
    """

    run_id: str
    resumed_from_iteration: int
    target_iteration: int
    iterations: int
    num_infosets: int
    status: str
    no_op: bool


def create_training_session(
    config: Config, experiment: ExperimentTag | None = None
) -> TrainingSession:
    """Create a new training session."""
    return TrainingSession(config, experiment=experiment)


def create_resumed_session(
    run_dir: Path, capacity_override: int | None = None
) -> tuple[TrainingSession, int]:
    """Create a resumed session and return it with latest completed iteration."""
    metadata = load_run_metadata(run_dir)
    latest_iteration = metadata.iterations
    session = TrainingSession.resume(run_dir, capacity_override=capacity_override)
    return session, latest_iteration


def run_training(
    session: TrainingSession,
    *,
    num_workers: int | None = None,
    num_iterations: int | None = None,
) -> None:
    """Execute training for an existing session."""
    session.train(
        num_workers=num_workers,
        num_iterations=num_iterations,
    )


def train(
    config_name: str,
    *,
    num_workers: int | None = None,
    num_iterations: int | None = None,
    seed: int | None = None,
    config_overrides: dict[str, Any] | None = None,
    experiment: ExperimentTag | None = None,
) -> TrainingOutput:
    """Run a full training session from a named config and return a portable summary.

    This is the headless, non-interactive training entrypoint used by scripts and
    cloud (Modal) execution. It loads the config, verifies the card abstraction is
    present, trains, and returns a :class:`TrainingOutput` — no stdout parsing required.

    Args:
        config_name: Stem of a config under ``config/training`` (e.g. ``"quick_test"``).
        num_workers: Parallel worker count; defaults to all available CPUs when ``None``.
        num_iterations: Overrides the config's iteration count when provided.
        seed: Overrides ``system.seed`` for reproducibility when provided.
        config_overrides: Extra nested config overrides (``__`` separator), e.g.
            ``{"storage__initial_capacity": 8_000_000}`` for calibration sweeps.
        experiment: Experiment/arm/parent this run belongs to, recorded in run
            metadata so arms can later be grouped and attributed against controls.

    Raises:
        FileNotFoundError: If the card abstraction for the config is missing (precompute it).
        ValueError: If the card abstraction is stale (config hash mismatch — recompute it).
    """
    overrides: dict[str, Any] = dict(config_overrides or {})
    if seed is not None:
        overrides["system__seed"] = seed
    config = load_training_config(config_name, **overrides)

    # TrainingSession.__init__ builds the card abstraction before creating the run
    # directory and cleans up on failure, so we surface its errors here with an
    # actionable message rather than pre-loading the (large) abstraction pickle twice.
    try:
        session = create_training_session(config, experiment=experiment)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Card abstraction '{config.card_abstraction.config}' for training config "
            f"'{config_name}' is missing. Precompute it before training. ({e})"
        ) from e
    except AbstractionHashMismatchError as e:
        raise AbstractionHashMismatchError(
            f"Card abstraction '{config.card_abstraction.config}' for training config "
            f"'{config_name}' is stale (config hash mismatch). Recompute it. ({e})"
        ) from e

    run_training(session, num_workers=num_workers, num_iterations=num_iterations)

    metadata = load_run_metadata(session.run_dir)
    ips = metadata.iterations / metadata.runtime_seconds if metadata.runtime_seconds > 0 else 0.0
    return TrainingOutput(
        run_id=metadata.run_id,
        runs_dir=config.training.runs_dir,
        config_name=metadata.config_name,
        iterations=metadata.iterations,
        num_infosets=metadata.num_infosets,
        runtime_seconds=metadata.runtime_seconds,
        iterations_per_second=ips,
        storage_capacity=metadata.storage_capacity,
        status=metadata.status,
    )


def resume(
    run_dir: Path,
    to_iteration: int,
    *,
    num_workers: int | None = None,
    capacity_override: int | None = None,
) -> ResumeOutput:
    """Resume an existing run and train up to an ABSOLUTE iteration target.

    The single resume orchestrator shared by every transport (headless CLI, Modal),
    so a cloud resume and a local resume cannot drift.

    Absolute, not "train N more": a scheduler-retried attempt re-reads a *newer*
    checkpoint, so a relative target compounds — a leg aimed at 25.5M retried after
    committing 21.5M would chase 30.6M.

    Args:
        run_dir: Directory of the run to resume.
        to_iteration: Absolute iteration to train up to; at or below the committed
            checkpoint this is a no-op.
        num_workers: Parallel worker count; defaults to all available CPUs when ``None``.
        capacity_override: Pre-allocate shared storage above the checkpoint's capacity
            so the leg never has to resize mid-run.
    """
    session, resumed_from = create_resumed_session(run_dir, capacity_override=capacity_override)

    remaining = to_iteration - resumed_from
    if remaining > 0:
        run_training(session, num_workers=num_workers, num_iterations=remaining)
    else:
        # Nothing forks here, so the bootstrap shared memory training would have
        # released during the worker handoff would leak for the process lifetime.
        session.release_bootstrap_storage()
        # `create_resumed_session` already called `mark_resumed`, which set the run
        # to "running" and opened an attempt. On this branch nothing runs and
        # nothing else closes it, so the run would be left looking live with a
        # dangling attempt -- the exact shape `mark_resumed` treats as evidence of
        # a death. Under a scheduler this is the COMMON case (every retry past the
        # target lands here), so it must be closed, not tolerated.
        if session.run_tracker is not None:
            session.run_tracker.mark_completed()

    metadata = load_run_metadata(run_dir)
    return ResumeOutput(
        run_id=metadata.run_id,
        resumed_from_iteration=resumed_from,
        target_iteration=to_iteration,
        iterations=metadata.iterations,
        num_infosets=metadata.num_infosets,
        status=metadata.status,
        no_op=remaining <= 0,
    )
