from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.game.state import Street
from src.shared.config_loader import _load_yaml

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
    """Configuration for board clustering and postflop bucket precomputation."""

    board_clusters: StreetBucketConfig = Field(
        default_factory=lambda: StreetBucketConfig(flop=8, turn=8, river=8)
    )
    buckets: StreetBucketConfig = Field(
        default_factory=lambda: StreetBucketConfig(flop=50, turn=100, river=200)
    )
    representatives_per_cluster: PositiveInt = 3
    representative_selection: Literal["closest", "diverse", "random"] = "closest"
    equity_samples: PositiveInt = 1000
    num_workers: PositiveInt | None = None
    seed: int = 42
    kmeans_max_iter: PositiveInt = 300
    kmeans_n_init: PositiveInt = 10
    config_name: str | None = None

    @property
    def num_board_clusters(self) -> dict[Street, int]:
        """Board cluster counts keyed by street enum."""
        return self.board_clusters.as_street_dict()

    @property
    def num_buckets(self) -> dict[Street, int]:
        """Bucket counts keyed by street enum."""
        return self.buckets.as_street_dict()

    @classmethod
    def from_yaml(cls, config_name: str) -> "PrecomputeConfig":
        """Load precompute config from ``config/abstraction/<name>.yaml``."""
        repo_root = Path(__file__).resolve().parents[3]
        config_path = repo_root / "config" / "abstraction" / f"{config_name}.yaml"
        yaml_data = _load_yaml(config_path)
        yaml_data["config_name"] = config_name
        return cls.model_validate(yaml_data)

    @classmethod
    def default(cls) -> "PrecomputeConfig":
        """Return defaults-only config."""
        return cls()

    def get_config_hash(self) -> str:
        """
        Compute a stable hash for abstraction compatibility checks.

        Excludes non-abstraction identity fields like ``config_name``.
        """
        config_dict = {
            "board_clusters": self.board_clusters.model_dump(),
            "buckets": self.buckets.model_dump(),
            "representatives_per_cluster": self.representatives_per_cluster,
            "representative_selection": self.representative_selection,
            "equity_samples": self.equity_samples,
        }
        stable_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(stable_json.encode()).hexdigest()[:16]
