"""Service-layer APIs for training and evaluation orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline.training.components import (
    build_evaluation_solver,
    evaluate_solver_exploitability,
)
from src.pipeline.training.run_tracker import RunMetadata, RunTracker
from src.pipeline.training.trainer import TrainingSession
from src.shared.config import Config
from src.shared.config_loader import load_training_config

# The current `evaluate_run` metric is the legacy one-ply rollout estimator, which
# understates true exploitability (see the eval-harness effort). Callers that surface
# the number to humans or an optimization loop must label it as such and NOT treat it
# as a trustworthy best-response value until the LBR/exact harness replaces it.
ROLLOUT_ESTIMATOR_LABEL = "rollout_one_ply (uninformative lower bound; not a true best response)"


@dataclass(frozen=True)
class EvaluationOutput:
    """Container for run evaluation output."""

    infosets: int
    results: dict[str, Any]


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


def list_runs(runs_dir: Path) -> list[str]:
    """List available training runs in the provided base directory."""
    return RunTracker.list_runs(runs_dir)


def load_run_metadata(run_dir: Path) -> RunMetadata:
    """Load run metadata from an existing run directory."""
    tracker = RunTracker.load(run_dir)
    return tracker.metadata


def create_training_session(config: Config) -> TrainingSession:
    """Create a new training session."""
    return TrainingSession(config)


def create_resumed_session(run_dir: Path) -> tuple[TrainingSession, int]:
    """Create a resumed session and return it with latest completed iteration."""
    metadata = load_run_metadata(run_dir)
    latest_iteration = metadata.iterations
    return TrainingSession.resume(run_dir), latest_iteration


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


def start_training(config: Config, num_workers: int) -> TrainingSession:
    """Start a new training session and run it."""
    session = create_training_session(config)
    run_training(session, num_workers=num_workers)
    return session


def train(
    config_name: str,
    *,
    num_workers: int | None = None,
    num_iterations: int | None = None,
    seed: int | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> TrainingOutput:
    """Run a full training session from a named config and return a portable summary.

    This is the headless, non-interactive training entrypoint used by scripts, the
    agent-driven improvement loop, and cloud (Modal) execution. It loads the config,
    verifies the card abstraction is present, trains, and returns a
    :class:`TrainingOutput` — no stdout parsing required.

    Args:
        config_name: Stem of a config under ``config/training`` (e.g. ``"quick_test"``).
        num_workers: Parallel worker count; defaults to all available CPUs when ``None``.
        num_iterations: Overrides the config's iteration count when provided.
        seed: Overrides ``system.seed`` for reproducibility when provided.
        config_overrides: Extra nested config overrides (``__`` separator), e.g.
            ``{"storage__initial_capacity": 8_000_000}`` for calibration sweeps.

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
        session = create_training_session(config)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Card abstraction '{config.card_abstraction.config}' for training config "
            f"'{config_name}' is missing. Precompute it before training. ({e})"
        ) from e
    except ValueError as e:
        if "hash" in str(e).lower():
            raise ValueError(
                f"Card abstraction '{config.card_abstraction.config}' for training config "
                f"'{config_name}' is stale (config hash mismatch). Recompute it. ({e})"
            ) from e
        raise

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


def resume_training(run_dir: Path, additional_iterations: int) -> tuple[TrainingSession, int]:
    """Resume training from run_dir and execute additional_iterations."""
    session, latest_iteration = create_resumed_session(run_dir)
    run_training(session, num_iterations=additional_iterations)
    return session, latest_iteration


def evaluate_run(
    run_dir: Path,
    num_samples: int,
    num_rollouts: int,
    use_average_strategy: bool,
    seed: int | None,
) -> EvaluationOutput:
    """
    Evaluate a run and return exploitability metrics.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid configuration or checkpoint state.
    """
    metadata = load_run_metadata(run_dir)
    config = metadata.config

    solver, storage = build_evaluation_solver(
        config,
        checkpoint_dir=run_dir,
    )
    results = evaluate_solver_exploitability(
        solver,
        num_samples=num_samples,
        use_average_strategy=use_average_strategy,
        num_rollouts_per_infoset=num_rollouts,
        seed=seed,
    )
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
    )
