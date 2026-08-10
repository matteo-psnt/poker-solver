from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.game.state import Street
from src.shared import repo
from src.shared.config.loader import load_yaml

PositiveInt = Annotated[int, Field(gt=0)]


class StrictFrozenModel(BaseModel):
    """Base model for immutable abstraction config with strict key validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class StreetBucketConfig(StrictFrozenModel):
    """Per-street integer values for flop, turn, and river."""

    flop: PositiveInt
    turn: PositiveInt
    river: PositiveInt

    def as_street_dict(self) -> dict[Street, int]:
        return {
            Street.FLOP: self.flop,
            Street.TURN: self.turn,
            Street.RIVER: self.river,
        }


class PrecomputeConfig(StrictFrozenModel):
    """Configuration for full-coverage postflop bucket precomputation."""

    buckets: StreetBucketConfig = Field(
        default_factory=lambda: StreetBucketConfig(flop=50, turn=100, river=200)
    )
    flop_runouts: PositiveInt | None = None
    equity_histogram_bins: PositiveInt = 8

    # River bucketing feature.
    #   scalar_equity  equity vs a UNIFORM opponent range (the original)
    #   ochs           Opponent Cluster Hand Strength: a vector of win rates
    #                  against clustered opponent holdings (Johanson et al.,
    #                  AAMAS 2013)
    # Scalar equity cannot express *which* part of the opponent's range a hand
    # beats, so a bluff-catcher and a weak made hand with equal equity share a
    # bucket while wanting opposite strategies. Measured on the production
    # abstraction, scalar equity is saturated at 600 buckets (variance explained
    # 0.999999) -- the feature, not the bucket count, is the binding limit.
    river_feature: Literal["scalar_equity", "ochs"] = "scalar_equity"
    ochs_clusters: PositiveInt = 8
    num_workers: PositiveInt | None = None
    seed: int = 42
    kmeans_max_iter: PositiveInt = 300
    kmeans_n_init: PositiveInt = 10
    config_name: str | None = None

    @property
    def num_buckets(self) -> dict[Street, int]:
        """Bucket counts keyed by street enum."""
        return self.buckets.as_street_dict()

    @classmethod
    def from_yaml(cls, config_name: str) -> PrecomputeConfig:
        """Load precompute config from ``config/abstraction/<name>.yaml``."""
        config_path = repo.ROOT / "config" / "abstraction" / f"{config_name}.yaml"
        yaml_data = load_yaml(config_path)
        yaml_data["config_name"] = config_name
        return cls.model_validate(yaml_data)

    @classmethod
    def default(cls) -> PrecomputeConfig:
        """Return defaults-only config."""
        return cls()

    def get_config_hash(self) -> str:
        """
        Compute a stable hash for abstraction compatibility checks.

        Excludes non-abstraction identity fields like ``config_name``.
        """
        config_dict = {
            "buckets": self.buckets.model_dump(),
            "flop_runouts": self.flop_runouts,
            "equity_histogram_bins": self.equity_histogram_bins,
        }
        # Only perturb the hash when a non-default river feature is in play, so
        # every abstraction built before OCHS existed keeps its identity and the
        # provenance checks guarding evaluation still recognise it.
        if self.river_feature != "scalar_equity":
            config_dict["river_feature"] = self.river_feature
            config_dict["ochs_clusters"] = self.ochs_clusters
        stable_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(stable_json.encode()).hexdigest()[:16]
