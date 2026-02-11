"""Service-layer APIs for training and evaluation orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.exploitability import compute_exploitability
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.in_memory import InMemoryStorage
from src.training.components import (
    build_action_abstraction,
    build_card_abstraction,
    build_solver,
)
from src.training.run_tracker import RunMetadata, RunTracker
from src.training.trainer import TrainingSession
from src.utils.config import Config


@dataclass(frozen=True)
class EvaluationOutput:
    """Container for run evaluation output."""

    infosets: int
    results: dict[str, Any]


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

    storage = InMemoryStorage(checkpoint_dir=run_dir)
    action_abstraction = build_action_abstraction(config)
    card_abstraction = build_card_abstraction(
        config,
        prompt_user=False,
        auto_compute=False,
    )
    solver = build_solver(config, action_abstraction, card_abstraction, storage)
    assert isinstance(solver, MCCFRSolver), f"Expected MCCFRSolver, got {type(solver)}"

    results = compute_exploitability(
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
