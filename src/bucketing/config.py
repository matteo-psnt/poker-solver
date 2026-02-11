"""
Abstraction precomputation configuration - Single source of truth.

Uses the same pattern as main Config - defaults in dataclass, YAML contains only overrides.
"""

import hashlib
import json
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

from src.game.state import Street
from src.utils.config_loader import _load_yaml, _merge_config


@dataclass(init=False)
class PrecomputeConfig:
    """Abstraction precomputation configuration with defaults."""

    # Board clustering (public-state abstraction)
    board_clusters_flop: int = 8
    board_clusters_turn: int = 8
    board_clusters_river: int = 8

    # Hand bucketing (hand abstraction)
    buckets_flop: int = 50
    buckets_turn: int = 100
    buckets_river: int = 200

    # Representatives per cluster (for equity computation)
    representatives_per_cluster: int = 3

    # Representative selection strategy: closest | diverse | random
    representative_selection: str = "closest"

    # Monte Carlo samples for equity calculation
    equity_samples: int = 1000

    # Parallel workers (None = CPU count)
    num_workers: int | None = None

    # Random seed for reproducibility
    seed: int = 42

    # K-means settings
    kmeans_max_iter: int = 300
    kmeans_n_init: int = 10

    # Config name (for matching during training)
    config_name: str | None = None

    def __init__(self, **kwargs: Any):
        """
        Initialize config with support for dict-style arguments.

        Converts num_board_clusters and num_buckets dicts to individual fields.
        """
        # Handle num_board_clusters dict -> individual fields
        if "num_board_clusters" in kwargs:
            clusters = kwargs.pop("num_board_clusters")
            if isinstance(clusters, dict):
                kwargs.setdefault("board_clusters_flop", clusters.get(Street.FLOP, 8))
                kwargs.setdefault("board_clusters_turn", clusters.get(Street.TURN, 8))
                kwargs.setdefault("board_clusters_river", clusters.get(Street.RIVER, 8))

        # Handle num_buckets dict -> individual fields
        if "num_buckets" in kwargs:
            buckets = kwargs.pop("num_buckets")
            if isinstance(buckets, dict):
                kwargs.setdefault("buckets_flop", buckets.get(Street.FLOP, 50))
                kwargs.setdefault("buckets_turn", buckets.get(Street.TURN, 100))
                kwargs.setdefault("buckets_river", buckets.get(Street.RIVER, 200))

        # Set fields, using defaults for missing values
        for field in fields(self.__class__):
            if field.name in kwargs:
                value = kwargs.pop(field.name)
            elif field.default is not MISSING:
                value = field.default
            elif field.default_factory is not MISSING:
                value = field.default_factory()
            else:
                raise TypeError(f"Missing required argument: {field.name}")
            object.__setattr__(self, field.name, value)

        # Warn about unexpected kwargs
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

    @property
    def num_board_clusters(self) -> dict[Street, int]:
        """Get board clusters as Street dict."""
        return self.to_street_dict("board_clusters")

    @property
    def num_buckets(self) -> dict[Street, int]:
        """Get buckets as Street dict."""
        return self.to_street_dict("buckets")

    def to_street_dict(self, field_prefix: str) -> dict[Street, int]:
        """
        Helper to convert flop/turn/river fields to Street dict.

        Args:
            field_prefix: Prefix like "board_clusters" or "buckets"

        Returns:
            Dictionary mapping Street enum to values

        Examples:
            >>> config.to_street_dict("board_clusters")
            {Street.FLOP: 8, Street.TURN: 8, Street.RIVER: 8}

            >>> config.to_street_dict("buckets")
            {Street.FLOP: 50, Street.TURN: 100, Street.RIVER: 200}
        """
        return {
            Street.FLOP: getattr(self, f"{field_prefix}_flop"),
            Street.TURN: getattr(self, f"{field_prefix}_turn"),
            Street.RIVER: getattr(self, f"{field_prefix}_river"),
        }

    @classmethod
    def from_yaml(cls, config_name: str) -> "PrecomputeConfig":
        """
        Load configuration from YAML file.

        Uses shared merge logic - YAML contains only overrides, not full defaults.

        Args:
            config_name: Name of config file (without .yaml extension)
                        e.g., 'quick_test', 'default', 'production'

        Returns:
            PrecomputeConfig instance with YAML overrides applied
        """
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "abstraction" / f"{config_name}.yaml"
        )

        # Load YAML (only contains overrides!)
        yaml_data = _load_yaml(config_path)

        # Flatten nested structures (board_clusters: {flop:, turn:, river:})
        if "board_clusters" in yaml_data and isinstance(yaml_data["board_clusters"], dict):
            clusters = yaml_data.pop("board_clusters")
            yaml_data["board_clusters_flop"] = clusters.get("flop")
            yaml_data["board_clusters_turn"] = clusters.get("turn")
            yaml_data["board_clusters_river"] = clusters.get("river")

        if "buckets" in yaml_data and isinstance(yaml_data["buckets"], dict):
            buckets = yaml_data.pop("buckets")
            yaml_data["buckets_flop"] = buckets.get("flop")
            yaml_data["buckets_turn"] = buckets.get("turn")
            yaml_data["buckets_river"] = buckets.get("river")

        # Auto-set config_name from filename
        yaml_data["config_name"] = config_name

        # Merge into defaults using shared logic
        return _merge_config(cls(), yaml_data)

    @classmethod
    def default(cls) -> "PrecomputeConfig":
        """Return default configuration (just dataclass defaults)."""
        return cls()

    def get_config_hash(self) -> str:
        """
        Compute a stable hash of the abstraction configuration.

        This hash is used to verify that a loaded abstraction matches the expected config.
        If you load an abstraction with a different config hash, the bucketing may be
        invalid because it was computed with different parameters.

        Returns:
            16-character hex string representing the config hash
        """
        # Create a stable dict representation of the config
        # Exclude config_name since it's just a label, not a parameter
        config_dict = {
            "board_clusters_flop": self.board_clusters_flop,
            "board_clusters_turn": self.board_clusters_turn,
            "board_clusters_river": self.board_clusters_river,
            "buckets_flop": self.buckets_flop,
            "buckets_turn": self.buckets_turn,
            "buckets_river": self.buckets_river,
            "representatives_per_cluster": self.representatives_per_cluster,
            "representative_selection": self.representative_selection,
            "equity_samples": self.equity_samples,
        }

        # Sort keys for stability
        stable_json = json.dumps(config_dict, sort_keys=True)

        # Return first 16 chars of hex digest
        return hashlib.sha256(stable_json.encode()).hexdigest()[:16]
