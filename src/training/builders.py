"""
Shared builder functions for training components.

Provides centralized, reusable functions for building solver components
(abstractions, storage, solver) from configuration. Used by both Trainer
and ParallelTrainer to eliminate code duplication.
"""

from pathlib import Path
from typing import Optional

from src.abstraction.core.action_abstraction import ActionAbstraction
from src.abstraction.core.card_abstraction import CardAbstraction
from src.abstraction.equity.equity_bucketing import EquityBucketing
from src.abstraction.equity.manager import EquityBucketManager
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

    Supports two modes:
    1. Config-based (recommended): Uses EquityBucketManager to find/load abstraction
    2. Direct path (legacy): Loads from explicit file path

    Args:
        config: Configuration object
        prompt_user: Whether to prompt user if abstraction not found (default: False for tests)
        auto_compute: Whether to auto-compute if abstraction not found

    Returns:
        CardAbstraction instance (typically EquityBucketing)

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If bucketing file doesn't exist
    """
    card_config = config.get_section("card_abstraction")
    abstraction_type = card_config.get("type", "equity_bucketing")

    if abstraction_type == "equity_bucketing":
        # Check for direct file path FIRST (higher priority, for testing/override)
        bucketing_path = card_config.get("bucketing_path")
        if bucketing_path:
            bucketing_path = Path(bucketing_path)
            if not bucketing_path.exists():
                raise FileNotFoundError(
                    f"Equity bucketing file not found: {bucketing_path}\n"
                    "Please run 'Precompute Equity Buckets' from the CLI first."
                )
            return EquityBucketing.load(bucketing_path)

        # NEW: Config-based system (uses EquityBucketManager)
        abstraction_config = card_config.get("config")
        if abstraction_config:
            manager = EquityBucketManager()
            manager_path = manager.find_or_compute(
                config_name=abstraction_config,
                auto_compute=auto_compute,
                prompt_user=prompt_user,
            )
            return EquityBucketing.load(manager_path)

        # Neither config nor path provided
        raise ValueError(
            "equity_bucketing requires either 'config' or 'bucketing_path'.\n"
            "Recommended: Use 'config: production' to reference an equity bucket config."
        )
    else:
        raise ValueError(
            f"Unknown card abstraction type: {abstraction_type}\n"
            "Only 'equity_bucketing' is supported."
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
    backend = storage_config.get("backend", "memory")

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
