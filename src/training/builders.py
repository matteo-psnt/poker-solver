"""
Shared builder functions for training components.

Provides centralized, reusable functions for building solver components
(abstractions, storage, solver) from configuration. Used by both Trainer
and ParallelTrainer to eliminate code duplication.
"""

import json
from pathlib import Path
from typing import Optional

from src.abstraction.core.action_abstraction import ActionAbstraction
from src.abstraction.core.card_abstraction import CardAbstraction
from src.solver.base import BaseSolver
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import DiskBackedStorage, InMemoryStorage, Storage
from src.utils.config import Config


def build_action_abstraction(config: Config) -> ActionAbstraction:
    """
    Build action abstraction from config.

    Args:
        config: Configuration object

    Returns:
        ActionAbstraction instance
    """
    action_config = config.get_section("action_abstraction")
    game_config = config.get_section("game")
    big_blind = game_config.get("big_blind", 2)
    return ActionAbstraction(action_config, big_blind=big_blind)


def build_card_abstraction(
    config: Config, prompt_user: bool = False, auto_compute: bool = False
) -> CardAbstraction:
    """
    Build card abstraction from config.

    Uses combo-level abstraction with suit isomorphism for correct postflop bucketing.

    Args:
        config: Configuration object
        prompt_user: Whether to prompt user if abstraction not found (default: False for tests)
        auto_compute: Whether to auto-compute if abstraction not found

    Returns:
        CardAbstraction instance (ComboAbstraction)

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If abstraction file doesn't exist
    """
    card_config = config.get_section("card_abstraction")

    # Get the abstraction path/config
    abstraction_path = card_config.get("abstraction_path")
    abstraction_config = card_config.get("config")

    if abstraction_path:
        # Direct path provided
        abstraction_path = Path(abstraction_path)
        if not abstraction_path.exists():
            raise FileNotFoundError(
                f"Combo abstraction file not found: {abstraction_path}\n"
                "Please run 'Precompute Combo Abstraction' from the CLI first."
            )
        from src.abstraction.isomorphism.precompute import ComboPrecomputer

        return ComboPrecomputer.load(abstraction_path)

    elif abstraction_config:
        # Config name provided - look for matching abstraction
        base_path = Path("data/combo_abstraction")

        if not base_path.exists():
            raise FileNotFoundError(
                f"No combo abstractions found (directory doesn't exist: {base_path}).\n"
                f"Please run 'Precompute Combo Abstraction' from the CLI with config '{abstraction_config}'.yaml first."
            )

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
        from src.abstraction.isomorphism.precompute import ComboPrecomputer

        return ComboPrecomputer.load(most_recent)

    else:
        raise ValueError(
            "card_abstraction requires either 'config' or 'abstraction_path'.\n"
            "Example: config: default"
        )


def build_storage(config: Config, run_dir: Optional[Path] = None) -> Storage:
    """
    Build storage backend from config.

    Args:
        config: Configuration object
        run_dir: Optional run directory (required for disk storage)

    Returns:
        Storage instance (InMemoryStorage or DiskBackedStorage)

    Raises:
        ValueError: If storage backend is unknown or run_dir missing for disk
    """
    storage_config = config.get_section("storage")
    backend = storage_config.get("backend")

    if backend == "memory":
        return InMemoryStorage()
    elif backend == "disk":
        if run_dir is None:
            raise ValueError("run_dir is required for disk storage backend")

        cache_size = storage_config.get("cache_size", 100000)
        flush_frequency = storage_config.get("flush_frequency", 1000)

        return DiskBackedStorage(
            checkpoint_dir=run_dir,
            cache_size=cache_size,
            flush_frequency=flush_frequency,
        )
    else:
        raise ValueError(f"Unknown storage backend: {backend}")


def build_solver(
    config: Config,
    action_abstraction: ActionAbstraction,
    card_abstraction: CardAbstraction,
    storage: Storage,
) -> BaseSolver:
    """
    Build solver from config and components.

    Args:
        config: Configuration object
        action_abstraction: Pre-built action abstraction
        card_abstraction: Pre-built card abstraction
        storage: Pre-built storage backend

    Returns:
        BaseSolver instance (typically MCCFRSolver)

    Raises:
        ValueError: If solver type is unknown
    """
    solver_config = config.get_section("solver")
    game_config = config.get_section("game")
    system_config = config.get_section("system")

    # Merge configs for solver
    merged_config = {**game_config, **system_config}

    solver_type = solver_config.get("type", "mccfr")

    if solver_type == "mccfr":
        return MCCFRSolver(
            action_abstraction=action_abstraction,
            card_abstraction=card_abstraction,
            storage=storage,
            config=merged_config,
        )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
