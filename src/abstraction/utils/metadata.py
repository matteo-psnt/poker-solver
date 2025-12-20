"""
Metadata dataclasses for card abstractions.

Provides core metadata structures for storing abstraction configuration
and quality metrics.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


def compute_config_hash(metadata: "EquityBucketMetadata") -> str:
    """
    Compute a short hash of the abstraction configuration.

    This hash is used to detect duplicate configurations and generate
    unique directory names.

    Args:
        metadata: Abstraction metadata

    Returns:
        6-character hex hash of the configuration
    """
    # Extract configuration parameters that define uniqueness
    config = {
        "abstraction_type": metadata.abstraction_type,
        "num_buckets": metadata.num_buckets,
        "num_board_clusters": metadata.num_board_clusters,
        "num_equity_samples": metadata.num_equity_samples,
        "num_samples_per_cluster": metadata.num_samples_per_cluster,
        "seed": metadata.seed,
    }

    # Create deterministic JSON string
    config_str = json.dumps(config, sort_keys=True)

    # Compute hash
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:6]


def generate_abstraction_name(metadata: "EquityBucketMetadata") -> str:
    """
    Generate standardized name from configuration.

    Format: buckets-F{flop}T{turn}R{river}-C{flop}C{turn}C{river}-s{samples}-{hash}

    Example: buckets-F50T100R200-C200C500C1000-s1000-a3f4b2

    Args:
        metadata: Abstraction metadata

    Returns:
        Generated name string
    """
    # Bucket counts
    f_buckets = metadata.num_buckets.get("FLOP", 0)
    t_buckets = metadata.num_buckets.get("TURN", 0)
    r_buckets = metadata.num_buckets.get("RIVER", 0)

    # Cluster counts
    f_clusters = metadata.num_board_clusters.get("FLOP", 0)
    t_clusters = metadata.num_board_clusters.get("TURN", 0)
    r_clusters = metadata.num_board_clusters.get("RIVER", 0)

    # Samples
    samples = metadata.num_equity_samples

    # Hash
    config_hash = compute_config_hash(metadata)

    return f"buckets-F{f_buckets}T{t_buckets}R{r_buckets}-C{f_clusters}C{t_clusters}C{r_clusters}-s{samples}-{config_hash}"


@dataclass
class EquityBucketMetadata:
    """Metadata for equity bucket configuration."""

    # Identification
    name: str
    created_at: str
    abstraction_type: str  # "equity_bucketing"

    # Configuration
    num_buckets: Dict[str, int]  # Street name -> num buckets
    num_board_clusters: Dict[str, int]  # Street name -> num clusters
    num_equity_samples: int
    num_samples_per_cluster: int
    seed: int

    # Quality metrics (optional)
    empty_clusters: Optional[Dict[str, int]] = None
    conflict_defaults: Optional[Dict[str, int]] = None

    # Aliases (optional user-defined names)
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EquityBucketMetadata":
        """
        Create from dictionary with backward compatibility.

        Handles old metadata formats by removing deprecated fields.
        """
        # Remove deprecated fields if present
        deprecated_fields = [
            "num_boards_sampled",
            "computation_time_seconds",
            "num_workers",
            "file_size_kb",
        ]
        for field_name in deprecated_fields:
            data.pop(field_name, None)

        # Handle missing aliases field for backward compatibility
        if "aliases" not in data:
            data["aliases"] = []

        return cls(**data)

    def get_config_hash(self) -> str:
        """Get the configuration hash for this abstraction."""
        return compute_config_hash(self)

    def generate_name(self) -> str:
        """Generate standardized name from configuration."""
        return generate_abstraction_name(self)

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"Abstraction: {self.name}",
            f"Type: {self.abstraction_type}",
            f"Created: {self.created_at}",
            f"Buckets: FLOP={self.num_buckets.get('FLOP', 'N/A')}, "
            f"TURN={self.num_buckets.get('TURN', 'N/A')}, "
            f"RIVER={self.num_buckets.get('RIVER', 'N/A')}",
            f"Board Clusters: FLOP={self.num_board_clusters.get('FLOP', 'N/A')}, "
            f"TURN={self.num_board_clusters.get('TURN', 'N/A')}, "
            f"RIVER={self.num_board_clusters.get('RIVER', 'N/A')}",
            f"MC Samples: {self.num_equity_samples}",
            f"Seed: {self.seed}",
        ]

        if self.aliases:
            lines.append(f"Aliases: {', '.join(self.aliases)}")

        return "\n".join(lines)


@dataclass
class BucketEntry:
    """Entry in the equity bucket registry."""

    name: str
    config_hash: str
    path: str
    created_at: str
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BucketEntry":
        """Create from dictionary."""
        return cls(**data)
