"""
Configuration schema — single source of truth.

Defaults are defined as Pydantic field defaults. YAML files provide overrides only.
Validation constraints live here, next to each field — nowhere else.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal

import xxhash
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.shared.action_tokens import JAM_TOKENS, PASSIVE_TOKENS, parse_multiplier_token
from src.shared.dicts import deep_merge_dicts

logger = logging.getLogger(__name__)

# Where runs live when nothing says otherwise. Backs both this module's
# ``TrainingConfig.runs_dir`` and the ``--runs-dir`` flag on every command that
# takes one -- ten of which used to spell it as a bare literal, while their
# sibling ``--ledger`` correctly pointed at ``DEFAULT_LEDGER_PATH``.
DEFAULT_RUNS_DIR = "data/runs"

# ---------------------------------------------------------------------------
# Shared type aliases for common constraints
# ---------------------------------------------------------------------------

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]
NonNegInt = Annotated[int, Field(ge=0)]


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
    runs_dir: str = Field(default=DEFAULT_RUNS_DIR)


class StorageConfig(StrictFrozenModel):
    """Storage and checkpoint configuration."""

    initial_capacity: PositiveInt = Field(default=2_000_000)
    max_actions: PositiveInt = Field(default=10)
    checkpoint_enabled: bool = Field(default=True)
    # Cap on the fraction of wall-clock spent checkpointing. Back-pressure defers a
    # checkpoint until (1-f)/f times the previous checkpoint's cost has elapsed, so
    # checkpointing self-limits to ~f of compute at any scale (0.1 = at most ~10%).
    max_checkpoint_overhead: Annotated[float, Field(gt=0.0, lt=1.0)] = Field(default=0.1)
    zarr_compression_level: Annotated[int, Field(ge=1, le=9)] = Field(default=1)
    zarr_chunk_size: PositiveInt = Field(default=50_000)
    # Spare one checkpoint per this many iterations from pruning, so the run ends
    # holding a ladder of snapshots instead of only its last one (0 = keep only the
    # last). Costs a full copy of the table per retained point, so keep it a large
    # multiple of checkpoint_frequency: below that every checkpoint lands in its own
    # band and nothing is ever pruned.
    checkpoint_retain_every: NonNegInt = Field(default=0)
    # Cap on each worker's cache of infoset ids owned by OTHER workers (0 =
    # unbounded, the historical behaviour). Unbounded, this cache converges on
    # the whole tree in every worker, so total memory scales O(N x workers)
    # rather than O(N) -- measured at 315 B/entry, that is 37 GB across 16
    # workers at 7M infosets, and it killed three legs on a 32 GB node. A miss
    # costs one dropped update and an id request, so the cap trades memory for
    # sample efficiency, and the cliff is STEEP once it binds. Measured on
    # quick_test at 4 workers, forcing eviction:
    #
    #   unbounded    drop 0.387  evictions 0        2489 it/s
    #   cap 10,000   drop 0.584  evictions 201,549  1137 it/s
    #
    # i.e. an undersized cap costs +51% drop rate and -54% throughput. So size it
    # to sit ABOVE the working set and treat it as an OOM backstop, not a tuning
    # knob: at ~315 B/entry, 4M entries is ~1.3 GB per worker (~10 GB across 8).
    # `remote_cache_evictions` in the metrics row is the signal -- non-zero and
    # rising means it is binding and you are paying the cliff above.
    remote_key_cache_size: NonNegInt = Field(default=4_000_000)


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
    def blinds_are_consistent(self) -> GameConfig:
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
    def validate_templates_and_rules(self) -> ActionModelConfig:
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

                if token.strip().lower() in PASSIVE_TOKENS | JAM_TOKENS:
                    continue
                try:
                    multiplier = parse_multiplier_token(token)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid preflop token '{token}' in preflop_templates[{template_name}]"
                    ) from exc
                if multiplier is None or multiplier <= 0:
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
    # Board runouts sampled for range-vs-range leaf valuation in the subgame CFR
    # (exact single board when the root is already on the river).
    leaf_rollouts: PositiveInt = Field(default=8)
    policy_blend_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.35)
    min_strategy_prob: NonNegFloat = Field(default=1e-6)
    # Fixed CFR iteration count. None means budget-driven (wall clock). Setting it
    # makes resolves machine/load-independent — a determinism knob for
    # reproducible experiments and tests.
    max_iterations: PositiveInt | None = Field(default=None)


class SolverConfig(StrictFrozenModel):
    """Solver algorithm configuration."""

    cfr_plus: bool = Field(default=False)
    iteration_weighting: Literal["none", "linear", "dcfr"] = Field(default="linear")

    # DCFR parameters — only used when iteration_weighting == "dcfr"
    dcfr_alpha: PositiveFloat = Field(default=1.5)
    dcfr_beta: NonNegFloat = Field(default=0.0)
    dcfr_gamma: PositiveFloat = Field(default=2.0)

    # Regret-based pruning (Brown & Sandholm 2019). The pruned set is derived
    # live from the persistent regrets each visit (see InfoSet.pruned_mask), so
    # the knob needs no stored state.
    #
    # Default off, and measured before trusting: a single-worker 60k-iter
    # production smoke found it throughput-NEUTRAL (~1.0x it/s; it skipped ~4.5%
    # of subtrees but external-sampling's per-node cost is already low, so the
    # saving ~ the masking overhead). Its effect on the average-strategy /
    # exploitability is NOT validated — enabling it changes the algorithm and
    # needs a paired-LBR gate, not just the it/s wash. Do not expect the
    # literature speedup here.
    enable_pruning: bool = Field(default=False)
    pruning_threshold: NonNegFloat = Field(default=300.0)
    prune_start_iteration: PositiveInt = Field(default=100)
    prune_reactivate_frequency: PositiveInt = Field(default=100)


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

    def content_hash(self) -> str:
        """Stable hash of the fully-resolved config.

        Experiment identity: ``config_name`` comes from ``system.config_name``
        inside the YAML, so two runs differing only by overrides record the same
        name and are otherwise indistinguishable. This is the field that tells
        them apart. Same canonicalization as ``ActionModel.get_config_hash`` so
        there is one hashing convention in the repo, not two.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return xxhash.xxh64(payload.encode()).hexdigest()

    @classmethod
    def default(cls) -> Config:
        """Return a Config populated with all defaults."""
        return cls()

    def merge(self, overrides: dict[str, Any]) -> Config:
        """Return a new Config with the provided overrides merged in."""
        merged = deep_merge_dicts(self.model_dump(), overrides)
        return Config.model_validate(merged)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Config:
        """Create Config from a dict merged over defaults.

        Strict: unknown keys raise (``extra="forbid"``), so typos in
        hand-authored YAML are caught. For historical run snapshots that may
        predate schema changes, use :meth:`from_persisted_dict` instead.
        """
        merged = deep_merge_dicts(cls().model_dump(), config_dict)
        return cls.model_validate(merged)

    @classmethod
    def from_persisted_dict(cls, config_dict: dict[str, Any]) -> Config:
        """Create Config from a previously saved run snapshot, tolerating drift.

        A persisted run config is machine-generated history, not user input, so
        strict validation only makes old runs unloadable when the schema evolves.
        This prunes fields that no longer exist in the schema (logging what was
        dropped) and then validates the remainder normally. Removed fields relate
        to training/abstraction setup and do not affect evaluating an already
        trained strategy.
        """
        pruned, dropped = _prune_to_schema(config_dict, cls)
        if dropped:
            logger.warning(
                "Ignoring %d legacy config field(s) from persisted run: %s",
                len(dropped),
                ", ".join(sorted(dropped)),
            )
        return cls.from_dict(pruned)


def _prune_to_schema(data: dict[str, Any], model_cls: type[BaseModel]) -> tuple[dict, list[str]]:
    """Recursively drop keys absent from ``model_cls``'s schema.

    Returns the pruned dict and the dotted paths of the dropped keys.
    """
    fields = model_cls.model_fields
    pruned: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in data.items():
        field = fields.get(key)
        if field is None:
            dropped.append(key)
            continue
        annotation = field.annotation
        if (
            isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
            and isinstance(value, dict)
        ):
            sub_pruned, sub_dropped = _prune_to_schema(value, annotation)
            pruned[key] = sub_pruned
            dropped.extend(f"{key}.{path}" for path in sub_dropped)
        else:
            pruned[key] = value
    return pruned, dropped
