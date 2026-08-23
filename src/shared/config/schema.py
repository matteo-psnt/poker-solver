"""
Configuration schema — single source of truth.

Defaults are defined as Pydantic field defaults. YAML files provide overrides only.
Validation constraints live here, next to each field — nowhere else.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal

import xxhash
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.shared.action_tokens import JAM_TOKENS, PASSIVE_TOKENS, parse_multiplier_token
from src.shared.config.merge import deep_merge_dicts

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    verbose: bool = Field(default=True)
    runs_dir: str = Field(default=DEFAULT_RUNS_DIR)


class StorageConfig(StrictFrozenModel):
    """Storage and checkpoint configuration."""

    initial_capacity: PositiveInt = Field(default=2_000_000)
    max_actions: PositiveInt = Field(default=10)
    zarr_compression_level: Annotated[int, Field(ge=1, le=9)] = Field(default=1)
    # Spare one checkpoint per this many iterations from pruning, so the run ends
    # holding a ladder of snapshots instead of only its last one (0 = keep only the
    # last). Costs a full copy of the table per retained point, so keep it a large
    # multiple of checkpoint_frequency: below that every checkpoint lands in its own
    # band and nothing is ever pruned.
    checkpoint_retain_every: NonNegInt = Field(default=0)


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


REQUIRED_PREFLOP_TEMPLATES = frozenset(
    {
        "sb_first_in",
        "bb_vs_limp",
        "sb_vs_limp_raise",
        "bb_vs_open",
        "sb_vs_3bet",
        "bb_vs_4bet",
        "sb_vs_5bet",
    }
)
REQUIRED_POSTFLOP_TEMPLATES = frozenset(
    {"first_aggressive", "facing_bet", "after_one_raise", "after_two_raises"}
)
REQUIRED_RAISE_RULE_KEYS = frozenset({"facing_1", "facing_2", "facing_3_plus"})
POSTFLOP_TOKENS = frozenset({"min_raise", "pot_raise", "jam", "allin", "all_in", "jam_low_spr"})


def _require_keys(present: Iterable[str], required: frozenset[str], field: str) -> None:
    """Refuse a mapping that omits any key the action model cannot run without."""
    missing = required - set(present)
    if missing:
        raise ValueError(f"{field} missing required keys: {sorted(missing)}")


def _size_or_word(token: float | str, field: str, template_name: str) -> str | None:
    """Check the token grammar both streets share, and normalise the word form.

    Returns:
        The stripped, lowercased word for the caller to check against its own
        street's vocabulary — or None when the token was a numeric size, which
        needs nothing further.

    Raises:
        ValueError: the size was not positive, or the token was neither a
            number nor a string.
    """
    if isinstance(token, (int, float)):
        if float(token) <= 0:
            raise ValueError(f"{field}[{template_name}] numeric sizes must be > 0")
        return None
    if not isinstance(token, str):
        raise ValueError(
            f"{field}[{template_name}] contains unsupported token type: {type(token).__name__}"
        )
    return token.strip().lower()


class ActionModelConfig(StrictFrozenModel):
    """Action model configuration with node-template and SPR-aware defaults."""

    version: PositiveInt = Field(default=1)
    preflop_templates: dict[str, list[float | str]] = Field(
        default_factory=lambda: {
            "sb_first_in": ["fold", "call", 2.5, 3.5, 5.0],
            # Numeric preflop sizes are the chips committed by THIS action in
            # big blinds: `bb_vs_limp: 2.0` is a raise to 3bb total over the
            # posted blind, as `sb_first_in: 2.5` is an open to 3bb total.
            "bb_vs_limp": ["check", 2.0, 3.0],
            "sb_vs_limp_raise": ["fold", "call", "2.3x_last", "jam"],
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
        """Refuse an action model that cannot be built into a tree.

        Order:
            Structural failures are reported before grammatical ones. A config
            missing ``bb_vs_open`` entirely and a config whose ``bb_vs_open``
            holds a bad token are different mistakes, and hearing about the
            token first sends the reader to the wrong line.
        """
        _require_keys(self.preflop_templates, REQUIRED_PREFLOP_TEMPLATES, "preflop_templates")
        _require_keys(self.postflop_templates, REQUIRED_POSTFLOP_TEMPLATES, "postflop_templates")
        _require_keys(self.raise_count_rules, REQUIRED_RAISE_RULE_KEYS, "raise_count_rules")

        extra_rule_keys = set(self.raise_count_rules) - REQUIRED_RAISE_RULE_KEYS
        if extra_rule_keys:
            raise ValueError(f"raise_count_rules has unknown keys: {sorted(extra_rule_keys)}")

        self._validate_rules_point_at_real_templates()
        self._validate_preflop_tokens()
        self._validate_postflop_tokens()
        return self

    def _validate_rules_point_at_real_templates(self) -> None:
        """Every raise-count rule must name a postflop template that exists."""
        for key, template_name in self.raise_count_rules.items():
            if template_name not in self.postflop_templates:
                raise ValueError(
                    f"raise_count_rules[{key}] points to unknown postflop template: {template_name}"
                )

    def _validate_preflop_tokens(self) -> None:
        """Preflop takes passive words, jams, or an open-ended raise multiplier."""
        for template_name, tokens in self.preflop_templates.items():
            if not tokens:
                raise ValueError(f"preflop_templates[{template_name}] must not be empty")
            for token in tokens:
                word = _size_or_word(token, "preflop_templates", template_name)
                if word is None or word in PASSIVE_TOKENS | JAM_TOKENS:
                    continue
                invalid = f"Invalid preflop token '{token}' in preflop_templates[{template_name}]"
                try:
                    multiplier = parse_multiplier_token(word)
                except ValueError as exc:
                    raise ValueError(invalid) from exc
                if multiplier is None or multiplier <= 0:
                    raise ValueError(invalid)

    def _validate_postflop_tokens(self) -> None:
        """Postflop takes a closed vocabulary, unlike preflop's multipliers."""
        for template_name, tokens in self.postflop_templates.items():
            if not tokens:
                raise ValueError(f"postflop_templates[{template_name}] must not be empty")
            for token in tokens:
                word = _size_or_word(token, "postflop_templates", template_name)
                if word is not None and word not in POSTFLOP_TOKENS:
                    raise ValueError(
                        f"Invalid postflop token '{token}' in postflop_templates[{template_name}]"
                    )


class ResolverConfig(StrictFrozenModel):
    """Runtime subgame resolver configuration."""

    enabled: bool = Field(default=True)
    time_budget_ms: PositiveInt = Field(default=300)
    # Where a river subgame is fully expanded. At 2 it was half-truncated, and
    # its error ROSE with more compute -- see test_lookahead_depth.
    max_depth: PositiveInt = Field(default=6)
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
    # A SENSITIVITY KNOB, not a strategy choice, and 0.0 is the shipped
    # behaviour. What each player is assumed to commit before showdown at a
    # depth-limit leaf, as a fraction of the pot there — see
    # `subgame_cfr.Continuation`. Zero is check-down, which is EXACT on the
    # river and an approximation only on flop/turn.
    #
    # It exists to answer "does the leaf valuation move the resolver's decisions
    # at all" before anyone builds the real thing, which is an opponent CHOICE
    # among continuations expressed as a decision node in the local tree. A
    # single fixed continuation is NOT that and may well score WORSE than
    # check-down: it is a different biased guess with no choice attached. Read
    # the MAGNITUDE of the change, never its sign.
    leaf_continuation_fraction: NonNegFloat = Field(default=0.0)


class SolverConfig(StrictFrozenModel):
    """Solver algorithm configuration."""

    cfr_plus: bool = Field(default=False)
    iteration_weighting: Literal["none", "linear", "dcfr"] = Field(default="linear")

    # DCFR parameters — only used when iteration_weighting == "dcfr"
    dcfr_alpha: PositiveFloat = Field(default=1.5)
    dcfr_beta: NonNegFloat = Field(default=0.0)
    dcfr_gamma: PositiveFloat = Field(default=2.0)

    # A MEASUREMENT SEAM, not a strategy choice. `tree` walks the betting
    # tree's edge table; `state` rebuilds a GameState per edge. The two are
    # required to produce bit-identical arrays and an identical random stream
    # (`test_tree_traversal_equivalence`), so this changes throughput and
    # nothing else — which is the only reason it is selectable at all.
    #
    # It exists because throughput here is box-variance-dominated: a 13x swing
    # was once measured on identical code across nodes, so the ONLY trustworthy
    # comparison is two arms back to back on one machine. Without a switch
    # that has to be two code snapshots, which differ in more than the thing
    # being measured.
    traversal: Literal["compiled", "tree", "state"] = Field(default="compiled")


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
