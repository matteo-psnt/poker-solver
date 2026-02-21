"""
Configuration schema - Single source of truth.

Defaults are defined as dataclass field defaults. YAML files provide overrides.
No duplication, no sync required, no wrapper classes.
"""

import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any


def _contains_legacy_action_abstraction(data: Any) -> bool:
    """Return True if a nested dict contains removed action_abstraction keys."""
    if isinstance(data, dict):
        if "action_abstraction" in data:
            return True
        return any(_contains_legacy_action_abstraction(v) for v in data.values())
    if isinstance(data, list):
        return any(_contains_legacy_action_abstraction(v) for v in data)
    return False


def _validate_no_legacy_action_abstraction(data: Any) -> None:
    if _contains_legacy_action_abstraction(data):
        raise ValueError(
            "Legacy key 'action_abstraction' is no longer supported. "
            "Use top-level 'action_model' and 'resolver' configuration sections."
        )


@dataclass(frozen=True)
class CardAbstractionConfig:
    """Card abstraction configuration with defaults."""

    config: str | None = "default"
    abstraction_path: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration with defaults."""

    num_iterations: int = 100_000
    checkpoint_frequency: int = 50_000
    iterations_per_worker: int = 1_000
    verbose: bool = True
    runs_dir: str = "data/runs"


@dataclass(frozen=True)
class StorageConfig:
    """Storage configuration with defaults."""

    initial_capacity: int = 2_000_000
    max_actions: int = 10
    checkpoint_enabled: bool = True
    zarr_compression_level: int = 3
    zarr_chunk_size: int = 10_000


@dataclass(frozen=True)
class SystemConfig:
    """System configuration with defaults."""

    seed: int | None = None  # None means random
    config_name: str = "default"
    log_level: str = "INFO"


@dataclass(frozen=True)
class GameConfig:
    """Game configuration with defaults."""

    starting_stack: int = 200
    small_blind: int = 1
    big_blind: int = 2


@dataclass(frozen=True)
class ActionModelConfig:
    """Action model configuration with node-template and SPR-aware defaults."""

    version: int = 1
    preflop_templates: dict[str, list[float | str]] = field(
        default_factory=lambda: {
            "sb_first_in": ["fold", "call", 2.5, 3.5],
            "bb_vs_limp": ["check", 4.0],
            "bb_vs_open": ["fold", "call"],
            "sb_vs_3bet": ["fold", "call"],
            "bb_vs_4bet": ["fold", "call", "jam"],
            "sb_vs_5bet": ["fold", "call"],
        }
    )
    postflop_templates: dict[str, list[float | str]] = field(
        default_factory=lambda: {
            "first_aggressive": [0.33, 1.0, "jam_low_spr"],
            "facing_bet": ["min_raise", "jam"],
            "after_one_raise": ["pot_raise", "jam"],
            "after_two_raises": ["jam"],
        }
    )
    spr_buckets: list[float] = field(default_factory=lambda: [2.0, 6.0])
    raise_count_rules: dict[str, str] = field(
        default_factory=lambda: {
            "facing_1": "facing_bet",
            "facing_2": "after_one_raise",
            "facing_3_plus": "after_two_raises",
        }
    )
    off_tree_mapping: str = "nearest"

    @property
    def preflop_raises(self) -> list[float]:
        """Compatibility accessor for legacy code paths."""
        values = self.preflop_templates.get("sb_first_in", [])
        return [float(v) for v in values if isinstance(v, (int, float))]

    @property
    def postflop(self) -> dict[str, list[float]]:
        """Compatibility accessor for legacy street-based templates."""
        first = self.postflop_templates.get("first_aggressive", [])
        fractions = [float(v) for v in first if isinstance(v, (int, float))]
        return {
            "flop": fractions or [0.33, 0.75, 1.25],
            "turn": fractions or [0.5, 1.0, 1.5],
            "river": fractions or [0.5, 1.0, 2.0],
        }

    @property
    def all_in_spr_threshold(self) -> float:
        if not self.spr_buckets:
            return 2.0
        return float(self.spr_buckets[0])

    @property
    def max_raises_per_street(self) -> int:
        return 4


@dataclass(frozen=True)
class ActionAbstractionConfig:
    """Legacy street-based action abstraction config (compatibility only)."""

    max_raises_per_street: int = 4
    all_in_spr_threshold: float = 2.0
    preflop_raises: list[float] = field(default_factory=lambda: [2.5, 3.5, 5.0])
    postflop: dict[str, list[float]] = field(
        default_factory=lambda: {
            "flop": [0.33, 0.66, 1.25],
            "turn": [0.50, 1.0, 1.5],
            "river": [0.50, 1.0, 2.0],
        }
    )


@dataclass(frozen=True)
class ResolverConfig:
    """Runtime subgame resolver configuration."""

    enabled: bool = True
    time_budget_ms: int = 300
    max_depth: int = 2
    max_raises_per_street: int = 2
    leaf_value_mode: str = "blueprint_rollout"


@dataclass(frozen=True)
class SolverConfig:
    """Solver configuration with defaults."""

    sampling_method: str = "external"
    cfr_plus: bool = True
    linear_cfr: bool = True

    # DCFR (Discounted CFR) parameters
    enable_dcfr: bool = False
    dcfr_alpha: float = 1.5  # Positive regret discount exponent
    dcfr_beta: float = 0.0  # Negative regret discount exponent
    dcfr_gamma: float = 2.0  # Strategy discount exponent

    # Regret-based pruning parameters
    enable_pruning: bool = False
    pruning_threshold: float = 300.0  # Absolute regret threshold for pruning
    prune_start_iteration: int = 100  # Don't prune until this iteration
    prune_reactivate_frequency: int = 100  # Re-enable all actions every N iterations


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
    action_model: ActionModelConfig = ActionModelConfig()
    resolver: ResolverConfig = ResolverConfig()
    solver: SolverConfig = SolverConfig()
    card_abstraction: CardAbstractionConfig = CardAbstractionConfig()

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary (for serialization, logging, etc.)."""
        return asdict(self)

    @classmethod
    def default(cls) -> "Config":
        """Return a Config populated with default values."""
        return cls()

    def merge(self, overrides: dict[str, Any]) -> "Config":
        """Return a new Config with the provided overrides merged in."""
        return merge_dataclass_config(self, overrides)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create Config from dictionary values merged over defaults."""
        return merge_dataclass_config(cls(), config_dict)

    @property
    def action_abstraction(self) -> ActionModelConfig:
        """Compatibility alias for migrated call sites."""
        return self.action_model


def merge_dataclass_config(base: Any, overrides: dict[str, Any], *, strict: bool = False) -> Any:
    """
    Recursively merge dictionary overrides into a dataclass instance.

    Returns a new instance of the same type as base with overrides applied.
    """
    if not is_dataclass(base):
        raise TypeError(f"base must be a dataclass, got {type(base)}")

    if not overrides:
        return base

    kwargs: dict[str, Any] = {}
    valid_field_names = {field_info.name for field_info in fields(base)}

    for field_info in fields(base):
        field_name = field_info.name
        current_value = getattr(base, field_name)

        if field_name not in overrides:
            continue

        override_value = overrides[field_name]
        if is_dataclass(current_value) and isinstance(override_value, dict):
            kwargs[field_name] = merge_dataclass_config(
                current_value,
                override_value,
                strict=strict,
            )
        else:
            kwargs[field_name] = override_value

    unknown_keys = set(overrides.keys()) - valid_field_names
    if unknown_keys:
        dataclass_name = type(base).__name__
        message = (
            f"Unknown keys in config for {dataclass_name}: {sorted(unknown_keys)}. "
            f"Valid keys are: {sorted(valid_field_names)}"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    return replace(base, **kwargs) if kwargs else base
