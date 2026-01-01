"""
Shared builder functions for training components.

Provides centralized, reusable functions for building solver components
(abstractions, storage, solver) from configuration. Used by TrainingSession
to eliminate code duplication.
"""

from pathlib import Path
from typing import Any

from src.actions.action_model import ActionModel
from src.bucketing.base import BucketingStrategy
from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.evaluation.exploitability import compute_exploitability
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.base import Storage
from src.solver.storage.in_memory import InMemoryStorage
from src.solver.storage.shared_array import SharedArrayStorage
from src.training.abstraction_resolver import ComboAbstractionResolver
from src.training.run_tracker import RunMetadata
from src.utils.config import Config


def build_action_model(config: Config) -> ActionModel:
    """
    Build action model from config.

    Args:
        config: Configuration object

    Returns:
        ActionModel instance
    """
    action_config = config.action_model
    big_blind = config.game.big_blind
    return ActionModel(
        action_config,
        big_blind=big_blind,
        max_raises_per_street=config.resolver.max_raises_per_street,
    )


def build_card_abstraction(
    config: Config,
    abstractions_dir: Path | None = None,
) -> BucketingStrategy:
    """
    Build card abstraction from config.

    Uses combo-level abstraction with suit isomorphism for correct postflop bucketing.

    Args:
        config: Configuration object
        abstractions_dir: Optional directory containing precomputed abstractions

    Returns:
        BucketingStrategy instance (PostflopBucketer)

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If abstraction file doesn't exist
    """
    resolver = ComboAbstractionResolver(
        abstractions_dir=abstractions_dir,
        loader=PostflopPrecomputer.load,
    )
    return resolver.load(
        abstraction_path=config.card_abstraction.abstraction_path,
        abstraction_config=config.card_abstraction.config,
    )


def build_storage(
    config: Config,
    run_dir: Path | None = None,
    run_metadata: RunMetadata | None = None,
) -> Storage:
    """
    Build storage backend for training (always returns SharedArrayStorage).

    This function is ONLY for use by TrainingSession during training initialization.
    For read-only access to checkpoints (charts, analysis, debugging), use
    InMemoryStorage directly.

    Creates a single-worker SharedArrayStorage instance. The actual training will
    create the full multi-worker storage when parallel training starts.

    Args:
        config: Configuration object
        run_dir: Optional run directory (required if checkpointing is enabled)
        run_metadata: Optional run metadata for resume capacity

    Returns:
        SharedArrayStorage instance for coordinator/single-worker use

    Raises:
        ValueError: If checkpointing is enabled but run_dir is not provided

    Note:
        This storage instance is primarily for the solver's interface. Actual
        parallel training creates its own multi-worker shared memory pools.
    """
    checkpoint_enabled = config.storage.checkpoint_enabled

    if checkpoint_enabled and run_dir is None:
        raise ValueError("run_dir is required when checkpoint_enabled is true")

    # Create a minimal storage instance (actual parallel training uses its own)
    # Using run_dir.name as session_id to avoid conflicts between runs
    session_id = run_dir.name if run_dir else "default"

    # Determine initial capacity: use run metadata if resuming, else config
    initial_capacity = config.storage.initial_capacity
    if run_metadata:
        initial_capacity = run_metadata.resolve_initial_capacity(initial_capacity)

    return SharedArrayStorage(
        num_workers=1,
        worker_id=0,
        session_id=session_id,
        initial_capacity=initial_capacity,
        max_actions=config.storage.max_actions,
        is_coordinator=True,
        checkpoint_dir=run_dir if checkpoint_enabled else None,
    )


def build_solver(
    config: Config,
    action_model: ActionModel,
    card_abstraction: BucketingStrategy,
    storage: Storage,
) -> MCCFRSolver:
    """
    Build solver from config and components.

    Args:
        config: Configuration object
        action_model: Pre-built action model
        card_abstraction: Pre-built card abstraction
        storage: Pre-built storage backend

    Returns:
        MCCFRSolver instance

    Raises:
        ValueError: If solver configuration is invalid
    """

    return MCCFRSolver(
        action_model=action_model,
        card_abstraction=card_abstraction,
        storage=storage,
        config=config,
    )


def build_evaluation_solver(
    config: Config,
    *,
    checkpoint_dir: Path,
    abstractions_dir: Path | None = None,
) -> tuple[MCCFRSolver, InMemoryStorage]:
    """Build solver and storage for read-only checkpoint evaluation."""
    storage = InMemoryStorage(checkpoint_dir=checkpoint_dir)
    action_model = build_action_model(config)
    card_abstraction = build_card_abstraction(
        config,
        abstractions_dir=abstractions_dir,
    )
    solver = build_solver(config, action_model, card_abstraction, storage)
    return solver, storage


def evaluate_solver_exploitability(
    solver: MCCFRSolver,
    *,
    num_samples: int,
    num_rollouts_per_infoset: int,
    use_average_strategy: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute exploitability for a solver instance with a shared evaluation path."""
    if not isinstance(solver, MCCFRSolver):
        raise TypeError("Exploitability computation requires MCCFRSolver")
    return compute_exploitability(
        solver,
        num_samples=num_samples,
        use_average_strategy=use_average_strategy,
        num_rollouts_per_infoset=num_rollouts_per_infoset,
        seed=seed,
    )
