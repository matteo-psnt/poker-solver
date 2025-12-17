"""
Metadata management for card abstractions.

Stores abstractions in organized directories with metadata for easy selection.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AbstractionMetadata:
    """Metadata for a card abstraction."""

    # Identification
    name: str
    created_at: str
    abstraction_type: str  # "equity_bucketing" or "rank_based"

    # Configuration
    num_buckets: Dict[str, int]  # Street name -> num buckets
    num_board_clusters: Dict[str, int]  # Street name -> num clusters
    num_equity_samples: int
    num_samples_per_cluster: int

    # Board sampling
    num_boards_sampled: Dict[str, int]  # Street name -> num boards

    # Computation info
    computation_time_seconds: float
    num_workers: int
    seed: int

    # Quality metrics (optional)
    empty_clusters: Optional[Dict[str, int]] = None
    conflict_defaults: Optional[Dict[str, int]] = None

    # File info
    file_size_kb: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AbstractionMetadata":
        """Create from dictionary."""
        return cls(**data)

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
            f"Computation Time: {self.computation_time_seconds / 60:.1f} minutes",
            f"Workers: {self.num_workers}",
        ]

        if self.file_size_kb:
            lines.append(f"File Size: {self.file_size_kb:.1f} KB")

        return "\n".join(lines)


class AbstractionManager:
    """Manages card abstraction storage and retrieval."""

    def __init__(self, base_dir: Path = None):
        """
        Initialize abstraction manager.

        Args:
            base_dir: Base directory for abstractions
                     (default: data/abstractions)
        """
        if base_dir is None:
            base_dir = Path("data/abstractions")

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_abstraction(
        self,
        bucketing_file: Path,
        metadata: AbstractionMetadata,
        name: Optional[str] = None,
    ) -> Path:
        """
        Save abstraction with metadata to organized directory.

        Args:
            bucketing_file: Path to the .pkl file
            metadata: Metadata object
            name: Optional custom name (default: use metadata.name)

        Returns:
            Path to the abstraction directory
        """
        if name:
            metadata.name = name

        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{metadata.name}_{timestamp}"
        abstraction_dir = self.base_dir / dir_name
        abstraction_dir.mkdir(parents=True, exist_ok=True)

        # Copy bucketing file
        dest_file = abstraction_dir / "bucketing.pkl"
        shutil.copy(bucketing_file, dest_file)

        # Add file size to metadata
        metadata.file_size_kb = dest_file.stat().st_size / 1024

        # Save metadata
        metadata_file = abstraction_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Create README
        readme_file = abstraction_dir / "README.md"
        with open(readme_file, "w") as f:
            f.write(f"# {metadata.name}\n\n")
            f.write("## Configuration\n\n")
            f.write(f"```\n{metadata}\n```\n\n")
            f.write("## Usage\n\n")
            f.write("```python\n")
            f.write("from src.abstraction.equity_bucketing import EquityBucketing\n\n")
            f.write(f"bucketing = EquityBucketing.load('{dest_file}')\n")
            f.write("```\n")

        return abstraction_dir

    def list_abstractions(self) -> List[tuple]:
        """
        List all available abstractions.

        Returns:
            List of (name, path, metadata) tuples
        """
        abstractions = []

        for item in self.base_dir.iterdir():
            if not item.is_dir():
                continue

            metadata_file = item / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, "r") as f:
                    metadata_dict = json.load(f)
                metadata = AbstractionMetadata.from_dict(metadata_dict)
                abstractions.append((metadata.name, item, metadata))
            except Exception as e:
                print(f"Warning: Failed to load metadata from {item}: {e}")
                continue

        # Sort by creation time (newest first)
        abstractions.sort(key=lambda x: x[2].created_at, reverse=True)

        return abstractions

    def get_abstraction(self, name: str) -> Optional[Path]:
        """
        Get abstraction by name (returns most recent if multiple matches).

        Args:
            name: Abstraction name

        Returns:
            Path to bucketing.pkl file, or None if not found
        """
        abstractions = self.list_abstractions()

        for abs_name, abs_path, metadata in abstractions:
            if abs_name == name or abs_path.name.startswith(name):
                bucketing_file = abs_path / "bucketing.pkl"
                if bucketing_file.exists():
                    return bucketing_file

        return None

    def delete_abstraction(self, name: str) -> bool:
        """
        Delete an abstraction.

        Args:
            name: Abstraction name or directory name

        Returns:
            True if deleted, False if not found
        """
        abstractions = self.list_abstractions()

        for abs_name, abs_path, metadata in abstractions:
            if abs_name == name or abs_path.name.startswith(name):
                shutil.rmtree(abs_path)
                return True

        return False

    def print_summary(self):
        """Print summary of all abstractions."""
        abstractions = self.list_abstractions()

        if not abstractions:
            print("No abstractions found.")
            return

        print(f"\nAvailable Abstractions ({len(abstractions)}):")
        print("=" * 80)

        for name, path, metadata in abstractions:
            print(f"\n{name} ({path.name})")
            print("-" * 80)
            print(f"  Type: {metadata.abstraction_type}")
            print(f"  Created: {metadata.created_at}")
            print(
                f"  Buckets: F={metadata.num_buckets.get('FLOP', 'N/A')}, "
                f"T={metadata.num_buckets.get('TURN', 'N/A')}, "
                f"R={metadata.num_buckets.get('RIVER', 'N/A')}"
            )
            print(
                f"  Computation: {metadata.computation_time_seconds / 60:.1f} min "
                f"({metadata.num_workers} workers)"
            )

            if metadata.file_size_kb:
                print(f"  Size: {metadata.file_size_kb:.1f} KB")

            bucketing_file = path / "bucketing.pkl"
            print(f"  File: {bucketing_file}")
