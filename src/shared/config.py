"""
Configuration schema — single source of truth.

Defaults are defined as Pydantic field defaults. YAML files provide overrides only.
Validation constraints live here, next to each field — nowhere else.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.shared.dicts import deep_merge_dicts

# ---------------------------------------------------------------------------
# Shared type aliases for common constraints
# ---------------------------------------------------------------------------

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]


# ---------------------------------------------------------------------------
# Base model: all config classes inherit this
# ---------------------------------------------------------------------------


class StrictFrozenModel(BaseModel):
    """Base for all config models: immutable, extra keys forbidden."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class CardAbstractionConfig(StrictFrozenModel):
    """Card abstraction configuration."""

    config: str = Field(default="default")


class TrainingConfig(StrictFrozenModel):
    """Training loop configuration."""

    num_iterations: PositiveInt = Field(default=100_000)
    checkpoint_frequency: PositiveInt = Field(default=50_000)
    iterations_per_worker: PositiveInt = Field(default=1_000)
    verbose: bool = Field(default=True)
    runs_dir: str = Field(default="data/runs")


class StorageConfig(StrictFrozenModel):
    """Storage and checkpoint configuration."""

    initial_capacity: PositiveInt = Field(default=2_000_000)
    max_actions: PositiveInt = Field(default=10)
    checkpoint_enabled: bool = Field(default=True)
    zarr_compression_level: Annotated[int, Field(ge=1, le=9)] = Field(default=1)
    zarr_chunk_size: PositiveInt = Field(default=50_000)


class SystemConfig(StrictFrozenModel):
    """System-level configuration."""

    seed: int | None = Field(default=None)
    config_name: str = Field(default="default")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


class GameConfig(StrictFrozenModel):
    """Game configuration."""

    starting_stack: PositiveInt = Field(default=200)
    small_blind: PositiveInt = Field(default=1)
    big_blind: PositiveInt = Field(default=2)

    @model_validator(mode="after")
    def blinds_are_consistent(self) -> "GameConfig":
        if self.big_blind <= self.small_blind:
            raise ValueError(
                f"big_blind ({self.big_blind}) must be greater than small_blind ({self.small_blind})"
            )
        return self


class ActionModelConfig(StrictFrozenModel):
    """Action model configuration with node-template and SPR-aware defaults."""

    version: PositiveInt = Field(default=1)
    preflop_templates: dict[str, list[float | str]] = Field(
        default_factory=lambda: {
            "sb_first_in": ["fold", "call", 2.5, 3.5, 5.0],
            "bb_vs_limp": ["check", 4.0],
            "bb_vs_open": ["fold", "call"],
            "sb_vs_3bet": ["fold", "call"],
            "bb_vs_4bet": ["fold", "call", "jam"],
            "sb_vs_5bet": ["fold", "call"],
        }
    )
    postflop_templates: dict[str, list[float | str]] = Field(
        default_factory=lambda: {
            "first_aggressive": [0.33, 0.66, 1.25],
            "facing_bet": ["min_raise", "jam"],
            "after_one_raise": ["pot_raise", "jam"],
            "after_two_raises": ["jam"],
        }
    )
    jam_spr_threshold: NonNegFloat = Field(default=2.0)
    raise_count_rules: dict[str, str] = Field(
        default_factory=lambda: {
            "facing_1": "facing_bet",
            "facing_2": "after_one_raise",
            "facing_3_plus": "after_two_raises",
        }
    )
    off_tree_mapping: Literal["nearest", "probabilistic"] = Field(default="probabilistic")

    @model_validator(mode="after")
    def validate_templates_and_rules(self) -> "ActionModelConfig":
        required_preflop_templates = {
            "sb_first_in",
            "bb_vs_limp",
            "bb_vs_open",
            "sb_vs_3bet",
            "bb_vs_4bet",
            "sb_vs_5bet",
        }
        required_postflop_templates = {
            "first_aggressive",
            "facing_bet",
            "after_one_raise",
            "after_two_raises",
        }
        required_raise_rule_keys = {
            "facing_1",
            "facing_2",
            "facing_3_plus",
        }

        missing_preflop = required_preflop_templates - set(self.preflop_templates.keys())
        if missing_preflop:
            raise ValueError(f"preflop_templates missing required keys: {sorted(missing_preflop)}")

        missing_postflop = required_postflop_templates - set(self.postflop_templates.keys())
        if missing_postflop:
            raise ValueError(
                f"postflop_templates missing required keys: {sorted(missing_postflop)}"
            )

        missing_rule_keys = required_raise_rule_keys - set(self.raise_count_rules.keys())
        if missing_rule_keys:
            raise ValueError(
                f"raise_count_rules missing required keys: {sorted(missing_rule_keys)}"
            )

        extra_rule_keys = set(self.raise_count_rules.keys()) - required_raise_rule_keys
        if extra_rule_keys:
            raise ValueError(f"raise_count_rules has unknown keys: {sorted(extra_rule_keys)}")

        for key, template_name in self.raise_count_rules.items():
            if template_name not in self.postflop_templates:
                raise ValueError(
                    f"raise_count_rules[{key}] points to unknown postflop template: {template_name}"
                )

        for template_name, tokens in self.preflop_templates.items():
            if not tokens:
                raise ValueError(f"preflop_templates[{template_name}] must not be empty")
            for token in tokens:
                if isinstance(token, (int, float)):
                    if float(token) <= 0:
                        raise ValueError(
                            f"preflop_templates[{template_name}] numeric sizes must be > 0"
                        )
                    continue
                if not isinstance(token, str):
                    raise ValueError(
                        f"preflop_templates[{template_name}] contains unsupported token type: "
                        f"{type(token).__name__}"
                    )

                normalized = token.strip().lower()
                if normalized in {"fold", "call", "check", "limp", "jam", "allin", "all_in"}:
                    continue
                if normalized.endswith("x_open"):
                    multiplier = normalized[: -len("x_open")]
                    try:
                        value = float(multiplier)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid preflop token '{token}' in preflop_templates[{template_name}]"
                        ) from exc
                    if value <= 0:
                        raise ValueError(
                            f"Invalid preflop token '{token}' in preflop_templates[{template_name}]"
                        )
                    continue
                if normalized.endswith("x_last"):
                    multiplier = normalized[: -len("x_last")]
                    try:
                        value = float(multiplier)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid preflop token '{token}' in preflop_templates[{template_name}]"
                        ) from exc
                    if value <= 0:
                        raise ValueError(
                            f"Invalid preflop token '{token}' in preflop_templates[{template_name}]"
                        )
                    continue

                raise ValueError(
                    f"Invalid preflop token '{token}' in preflop_templates[{template_name}]"
                )

        for template_name, tokens in self.postflop_templates.items():
            if not tokens:
                raise ValueError(f"postflop_templates[{template_name}] must not be empty")
            for token in tokens:
                if isinstance(token, (int, float)):
                    if float(token) <= 0:
                        raise ValueError(
                            f"postflop_templates[{template_name}] numeric sizes must be > 0"
                        )
                    continue
                if not isinstance(token, str):
                    raise ValueError(
                        f"postflop_templates[{template_name}] contains unsupported token type: "
                        f"{type(token).__name__}"
                    )
                normalized = token.strip().lower()
                if normalized not in {
                    "min_raise",
                    "pot_raise",
                    "jam",
                    "allin",
                    "all_in",
                    "jam_low_spr",
                }:
                    raise ValueError(
                        f"Invalid postflop token '{token}' in postflop_templates[{template_name}]"
                    )

        return self


class ResolverConfig(StrictFrozenModel):
    """Runtime subgame resolver configuration."""

    enabled: bool = Field(default=True)
    time_budget_ms: PositiveInt = Field(default=300)
    max_depth: PositiveInt = Field(default=2)
    max_raises_per_street: PositiveInt = Field(default=2)
    leaf_rollouts: PositiveInt = Field(default=8)
    leaf_use_average_strategy: bool = Field(default=True)
    policy_blend_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.35)
    min_strategy_prob: NonNegFloat = Field(default=1e-6)


class SolverConfig(StrictFrozenModel):
    """Solver algorithm configuration."""

    sampling_method: Literal["external", "outcome"] = Field(default="external")
    cfr_plus: bool = Field(default=True)
    iteration_weighting: Literal["none", "linear", "dcfr"] = Field(default="linear")

    # DCFR parameters — only used when iteration_weighting == "dcfr"
    dcfr_alpha: PositiveFloat = Field(default=1.5)
    dcfr_beta: NonNegFloat = Field(default=0.0)
    dcfr_gamma: PositiveFloat = Field(default=2.0)

    # Regret-based pruning
    enable_pruning: bool = Field(default=False)
    pruning_threshold: NonNegFloat = Field(default=300.0)
    prune_start_iteration: PositiveInt = Field(default=100)
    prune_reactivate_frequency: PositiveInt = Field(default=100)

    @model_validator(mode="after")
    def pruning_requires_external_sampling(self) -> "SolverConfig":
        if self.enable_pruning and self.sampling_method != "external":
            raise ValueError(
                "enable_pruning=True requires sampling_method='external'. "
                "Pruning is incompatible with outcome sampling."
            )
        return self


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class Config(StrictFrozenModel):
    """
    Complete solver configuration.

    All defaults are defined here in Python. YAML files provide only overrides.
    """

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    action_model: ActionModelConfig = Field(default_factory=ActionModelConfig)
    resolver: ResolverConfig = Field(default_factory=ResolverConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    card_abstraction: CardAbstractionConfig = Field(default_factory=CardAbstractionConfig)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dict (for JSON, logging, etc.)."""
        return self.model_dump()

    @classmethod
    def default(cls) -> "Config":
        """Return a Config populated with all defaults."""
        return cls()

    def merge(self, overrides: dict[str, Any]) -> "Config":
        """Return a new Config with the provided overrides merged in."""
        merged = deep_merge_dicts(self.model_dump(), overrides)
        return Config.model_validate(merged)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create Config from a dict merged over defaults."""
        merged = deep_merge_dicts(cls().model_dump(), config_dict)
        return cls.model_validate(merged)
