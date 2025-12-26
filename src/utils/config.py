"""
Configuration schema - Single source of truth.

Defaults are defined as dataclass field defaults. YAML files provide overrides.
No duplication, no sync required, no wrapper classes.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration with defaults."""

    num_iterations: int = 100_000
    checkpoint_frequency: int = 2_000
    log_frequency: int = 1_000
    iterations_per_worker: int = 500
    verbose: bool = True
    runs_dir: str = "data/runs"


@dataclass(frozen=True)
class StorageConfig:
    """Storage configuration with defaults."""

    max_infosets: int = 2_000_000
    max_actions: int = 10
    checkpoint_enabled: bool = True


@dataclass(frozen=True)
class SystemConfig:
    """System configuration with defaults."""

    seed: Optional[int] = None  # None means random
    config_name: str = "default"
    log_level: str = "INFO"


@dataclass(frozen=True)
class GameConfig:
    """Game configuration with defaults."""

    starting_stack: int = 200
    small_blind: int = 1
    big_blind: int = 2


@dataclass(frozen=True)
class ActionAbstractionConfig:
    """Action abstraction configuration with defaults."""

    max_raises_per_street: int = 4
    all_in_spr_threshold: float = 2.0


@dataclass(frozen=True)
class SolverConfig:
    """Solver configuration with defaults."""

    type: str = "mccfr"
    sampling_method: str = "outcome"
    cfr_plus: bool = True
    linear_cfr: bool = True


@dataclass(frozen=True)
class Config:
    """
    Complete solver configuration.

    All defaults are defined here. YAML files provide overrides via merge.
    """

    training: TrainingConfig = TrainingConfig()
    storage: StorageConfig = StorageConfig()
    system: SystemConfig = SystemConfig()
    game: GameConfig = GameConfig()
    action_abstraction: ActionAbstractionConfig = ActionAbstractionConfig()
    solver: SolverConfig = SolverConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (for serialization, logging, etc.)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create Config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance with values from dict
        """
        # Import here to avoid circular dependency
        from src.utils.config_loader import _merge_config

        # Start with defaults, merge in provided values
        base = cls()
        return _merge_config(base, config_dict)
