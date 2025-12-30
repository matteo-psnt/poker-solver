"""
Shared builder functions for training components.

Provides centralized, reusable functions for building solver components
(abstractions, storage, solver) from configuration. Used by TrainingSession
to eliminate code duplication.
"""

import json
from pathlib import Path

from src.actions.betting_actions import BettingActions
from src.bucketing.base import BucketingStrategy
from src.bucketing.config import PrecomputeConfig
from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.base import Storage
from src.solver.storage.shared_array import SharedArrayStorage
from src.training.run_metadata import RunMetadata
from src.utils.config import Config


def build_action_abstraction(config: Config) -> BettingActions:
    """
    Build action abstraction from config.

    Args:
        config: Configuration object

    Returns:
        BettingActions instance
    """
    action_config = config.action_abstraction
    big_blind = config.game.big_blind
    return BettingActions(action_config, big_blind=big_blind)


def build_card_abstraction(
    config: Config, prompt_user: bool = False, auto_compute: bool = False
) -> BucketingStrategy:
    """
    Build card abstraction from config.

    Uses combo-level abstraction with suit isomorphism for correct postflop bucketing.

    Args:
        config: Configuration object
        prompt_user: Whether to prompt user if abstraction not found (default: False for tests)
        auto_compute: Whether to auto-compute if abstraction not found

    Returns:
        BucketingStrategy instance (PostflopBucketer)

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If abstraction file doesn't exist
    """
    # Get the abstraction path/config
    abstraction_path = config.card_abstraction.abstraction_path
    abstraction_config = config.card_abstraction.config

    if abstraction_path:
        # Direct path provided
        path_obj = Path(abstraction_path)
        if not path_obj.exists():
            raise FileNotFoundError(
                f"Combo abstraction file not found: {path_obj}\n"
                "Please run 'Precompute Combo Abstraction' from the CLI first."
            )
        return PostflopPrecomputer.load(path_obj)

    elif abstraction_config:
        # Config name provided - look for matching abstraction
        base_path = Path("data/combo_abstraction")

        if not base_path.exists():
            raise FileNotFoundError(
                f"No combo abstractions found (directory doesn't exist: {base_path}).\n"
                f"Please run 'Precompute Combo Abstraction' from the CLI with config '{abstraction_config}'.yaml first."
            )

        expected_config = PrecomputeConfig.from_yaml(abstraction_config)
        expected_hash = expected_config.get_config_hash()

        # Find abstraction matching this config name
        matching = []
        if base_path.exists():
            for path in base_path.iterdir():
                if not path.is_dir():
                    continue

                metadata_file = path / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    # Check if config_name matches
                    saved_config_name = metadata.get("config", {}).get("config_name")
                    if saved_config_name == abstraction_config:
                        matching.append(path)
                except Exception:
                    continue

        if not matching:
            raise FileNotFoundError(
                f"No combo abstraction found for config '{abstraction_config}'.\n"
                f"Please run 'Precompute Combo Abstraction' from the CLI first with {abstraction_config}.yaml."
            )

        # Use most recent if multiple matches
        most_recent = max(matching, key=lambda p: p.stat().st_mtime)

        # Verify config hash matches
        metadata_file = most_recent / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        saved_hash = metadata.get("config", {}).get("config_hash")

        if saved_hash and saved_hash != expected_hash:
            raise ValueError(
                f"Card abstraction config hash mismatch for '{abstraction_config}':\n"
                f"  Expected: {expected_hash}\n"
                f"  Saved:    {saved_hash}\n"
                f"The saved abstraction was computed with different parameters.\n"
                f"Please re-run 'Precompute Combo Abstraction' with the current config."
            )
        return PostflopPrecomputer.load(most_recent)

    else:
        raise ValueError(
            "card_abstraction requires either 'config' or 'abstraction_path'.\n"
            "Example: config: default"
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
    action_abstraction: BettingActions,
    card_abstraction: BucketingStrategy,
    storage: Storage,
) -> MCCFRSolver:
    """
    Build solver from config and components.

    Args:
        config: Configuration object
        action_abstraction: Pre-built action abstraction
        card_abstraction: Pre-built card abstraction
        storage: Pre-built storage backend

    Returns:
        MCCFRSolver instance

    Raises:
        ValueError: If solver configuration is invalid
    """

    return MCCFRSolver(
        action_abstraction=action_abstraction,
        card_abstraction=card_abstraction,
        storage=storage,
        config=config,
    )
