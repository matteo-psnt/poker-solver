"""
Shared builder functions for training components.

Provides centralized, reusable functions for building solver components
(abstractions, storage, solver) from configuration. Used by TrainingSession
to eliminate code duplication.
"""

from pathlib import Path
from typing import Any

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.base import Storage
from src.engine.solver.storage.in_memory import InMemoryStorage
from src.engine.solver.storage.shared_array import SharedArrayStorage
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint
from src.pipeline.abstraction.base import BucketingStrategy
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer
from src.pipeline.evaluation.exploitability import compute_exploitability
from src.pipeline.training.abstraction_resolver import ComboAbstractionResolver
from src.pipeline.training.run_tracker import RunMetadata
from src.shared.config import Config


def build_card_abstraction(
    config: Config,
    abstractions_dir: Path | None = None,
    abstraction_hash: str | None = None,
) -> BucketingStrategy:
    """
    Build card abstraction from config.

    Uses combo-level abstraction with suit isomorphism for correct postflop bucketing.

    Args:
        config: Configuration object
        abstractions_dir: Optional directory containing precomputed abstractions
        abstraction_hash: Optional exact abstraction config hash to pin resolution to.
            Required to faithfully evaluate a checkpoint whose abstraction has since
            been recomputed under the same config name.

    Returns:
        BucketingStrategy instance (DenseBucketer)

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If abstraction file doesn't exist
    """
    resolver = ComboAbstractionResolver(
        abstractions_dir=abstractions_dir,
        loader=PostflopPrecomputer.load,
    )
    return resolver.load(
        abstraction_config=config.card_abstraction.config,
        abstraction_hash=abstraction_hash,
    )


def resolve_card_abstraction_hash(
    config: Config,
    abstractions_dir: Path | None = None,
) -> str | None:
    """Config hash of the abstraction ``config`` currently resolves to.

    Recorded on a run so evaluation can pin the exact abstraction it trained against.
    """
    resolver = ComboAbstractionResolver(
        abstractions_dir=abstractions_dir,
        loader=PostflopPrecomputer.load,
    )
    return resolver.resolved_hash(abstraction_config=config.card_abstraction.config)


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
        # Never eager-load the checkpoint into this bootstrap storage. At
        # num_workers=1 that load is single-threaded over the entire key table
        # (>15 min at ~11M infosets) and is then discarded by
        # release_bootstrap_storage before training -- the actual workers each
        # load their own 1/N shard in parallel. Resume verifies the checkpoint
        # is non-empty via the cheap key-table row count instead.
        load_checkpoint_on_init=False,
        zarr_compression_level=config.storage.zarr_compression_level,
        zarr_chunk_size=config.storage.zarr_chunk_size,
        checkpoint_retain_every=config.storage.checkpoint_retain_every,
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
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
) -> tuple[MCCFRSolver, InMemoryStorage]:
    """Build solver and storage for read-only checkpoint evaluation.

    ``abstraction_hash`` pins the card abstraction to the one the checkpoint was
    trained against; without it the config name resolves against current code.
    ``at_iteration`` scores a retained ladder rung instead of the published
    snapshot (the within-run convergence curve).
    """
    storage = InMemoryStorage(checkpoint_dir=checkpoint_dir, at_iteration=at_iteration)
    action_model = ActionModel(config)
    card_abstraction = build_card_abstraction(
        config,
        abstractions_dir=abstractions_dir,
        abstraction_hash=abstraction_hash,
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


def build_static_evaluation_solver(
    config: Config,
    *,
    checkpoint_dir: Path,
    abstractions_dir: Path | None = None,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
) -> tuple[StaticTreeSolver, StaticArrayStorage]:
    """Build a read-only blueprint over a STATIC checkpoint.

    The static and dynamic backends are not interchangeable -- one is addressed
    by a hashed key, the other by ``(node_id, bucket)`` -- so evaluation cannot
    share a loader. It shares the seam below it instead: both produce something
    satisfying ``ScorableBlueprint``, and ``policy_source_for`` dispatches on the
    concrete storage. The scoring engines never learn which they were handed.

    ``session_id=None`` allocates process-local arrays rather than shared memory:
    an evaluation is a single read-only process, and taking a named segment would
    collide with a training run of the same id.
    """
    action_model = ActionModel(config)
    card_abstraction = build_card_abstraction(
        config,
        abstractions_dir=abstractions_dir,
        abstraction_hash=abstraction_hash,
    )
    rules = GameRules(config.game.small_blind, config.game.big_blind)
    tree = build_betting_tree(
        rules, action_model, card_abstraction, starting_stack=config.game.starting_stack
    )
    storage = StaticArrayStorage(tree)
    # Verifies the tree fingerprint, so a checkpoint written against a different
    # tree is refused rather than silently reinterpreted row-for-row.
    load_checkpoint(storage, checkpoint_dir, at_iteration=at_iteration)
    solver = StaticTreeSolver(action_model, card_abstraction, storage, config, tree=tree)
    return solver, storage
