"""
High-level management of card abstractions.

Provides abstraction storage, retrieval, and automatic computation
with registry integration.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.abstraction.utils.metadata import (
    BucketEntry,
    EquityBucketMetadata,
    compute_config_hash,
    generate_abstraction_name,
)


def _find_abstraction_file(directory: Path) -> Optional[Path]:
    """
    Helper to find abstraction file with fallback to old naming.

    Args:
        directory: Abstraction directory

    Returns:
        Path to abstraction file or None
    """
    # Try new filename first
    abstraction_file = directory / "abstraction.pkl"
    if abstraction_file.exists():
        return abstraction_file

    # Fallback to legacy filename
    old_file = directory / "bucketing.pkl"
    if old_file.exists():
        return old_file

    return None


class _EquityBucketRegistry:
    """Internal registry for tracking equity buckets by config hash and aliases."""

    def __init__(self, registry_file: Path):
        """
        Initialize registry.

        Args:
            registry_file: Path to registry JSON file
        """
        self.registry_file = Path(registry_file)
        self.entries: Dict[str, BucketEntry] = {}
        self._alias_map: Dict[str, str] = {}  # alias -> name

        # Load existing registry if it exists
        if self.registry_file.exists():
            self.load()

    def load(self):
        """Load registry from disk."""
        try:
            with open(self.registry_file, "r") as f:
                data = json.load(f)

            self.entries = {}
            self._alias_map = {}

            for name, entry_data in data.get("entries", {}).items():
                entry = BucketEntry.from_dict(entry_data)
                self.entries[name] = entry

                # Build alias map
                for alias in entry.aliases:
                    self._alias_map[alias] = name

        except Exception as e:
            print(f"Warning: Failed to load registry: {e}")
            self.entries = {}
            self._alias_map = {}

    def save(self):
        """Save registry to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entries": {name: entry.to_dict() for name, entry in self.entries.items()},
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        config_hash: str,
        path: Path,
        aliases: Optional[List[str]] = None,
    ):
        """
        Register a new abstraction.

        Args:
            name: Abstraction name
            config_hash: Configuration hash
            path: Path to abstraction directory
            aliases: Optional aliases
        """
        aliases = aliases or []

        entry = BucketEntry(
            name=name,
            config_hash=config_hash,
            path=str(path),
            created_at=datetime.now().isoformat(),
            aliases=aliases,
        )

        self.entries[name] = entry

        # Update alias map
        for alias in aliases:
            self._alias_map[alias] = name

        self.save()

    def get(self, name_or_alias: str) -> Optional[BucketEntry]:
        """
        Get registry entry by name or alias.

        Args:
            name_or_alias: Abstraction name or alias

        Returns:
            Registry entry or None
        """
        # Try direct name lookup
        if name_or_alias in self.entries:
            return self.entries[name_or_alias]

        # Try alias lookup
        if name_or_alias in self._alias_map:
            actual_name = self._alias_map[name_or_alias]
            return self.entries.get(actual_name)

        return None

    def find_by_config_hash(self, config_hash: str) -> Optional[BucketEntry]:
        """
        Find abstraction by config hash.

        Args:
            config_hash: Configuration hash to search for

        Returns:
            Registry entry or None
        """
        for entry in self.entries.values():
            if entry.config_hash == config_hash:
                return entry
        return None

    def add_alias(self, name: str, alias: str):
        """
        Add an alias to an existing abstraction.

        Args:
            name: Abstraction name
            alias: Alias to add
        """
        if name not in self.entries:
            raise ValueError(f"Abstraction {name} not found in registry")

        entry = self.entries[name]
        if alias not in entry.aliases:
            entry.aliases.append(alias)
            self._alias_map[alias] = name
            self.save()

    def remove_alias(self, alias: str):
        """
        Remove an alias.

        Args:
            alias: Alias to remove
        """
        if alias in self._alias_map:
            name = self._alias_map[alias]
            entry = self.entries[name]
            entry.aliases.remove(alias)
            del self._alias_map[alias]
            self.save()

    def list_all(self) -> List[BucketEntry]:
        """List all registered abstractions."""
        return list(self.entries.values())


class EquityBucketManager:
    """Manages equity bucket storage and retrieval with registry."""

    def __init__(self, base_dir: Path = None):
        """
        Initialize abstraction manager.

        Args:
            base_dir: Base directory for equity buckets
                     (default: data/equity_buckets)
        """
        if base_dir is None:
            base_dir = Path("data/equity_buckets")

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize registry
        self.registry = _EquityBucketRegistry(self.base_dir / ".registry.json")

    def save_abstraction(
        self,
        bucketing_file: Path,
        metadata: EquityBucketMetadata,
        aliases: Optional[List[str]] = None,
        auto_name: bool = True,
    ) -> Path:
        """
        Save abstraction with metadata using config-hash based naming.

        Args:
            bucketing_file: Path to the .pkl file
            metadata: Metadata object
            aliases: Optional list of user-defined aliases
            auto_name: If True, generate name from config (recommended)

        Returns:
            Path to the abstraction directory
        """
        aliases = aliases or []

        # Generate name from configuration if requested
        if auto_name:
            metadata.name = generate_abstraction_name(metadata)

        # Check if this config already exists
        config_hash = compute_config_hash(metadata)
        existing = self.registry.find_by_config_hash(config_hash)

        if existing:
            print(f"Warning: Abstraction with same config already exists: {existing.name}")
            print(f"Reusing existing abstraction at {existing.path}")

            # Add new aliases if provided
            for alias in aliases:
                if alias not in existing.aliases:
                    self.registry.add_alias(existing.name, alias)

            return Path(existing.path)

        # Create directory
        abstraction_dir = self.base_dir / metadata.name
        abstraction_dir.mkdir(parents=True, exist_ok=True)

        # Copy bucketing file (renamed to abstraction.pkl)
        dest_file = abstraction_dir / "abstraction.pkl"
        shutil.copy(bucketing_file, dest_file)

        # Add aliases to metadata
        metadata.aliases = aliases

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
            f.write("from src.abstraction.equity.equity_bucketing import EquityBucketing\n\n")
            f.write(f"bucketing = EquityBucketing.load('{dest_file}')\n")
            f.write("```\n")

        # Register in registry
        self.registry.register(
            name=metadata.name,
            config_hash=config_hash,
            path=abstraction_dir,
            aliases=aliases,
        )

        return abstraction_dir

    def list_abstractions(self) -> List[tuple]:
        """
        List all available abstractions.

        Returns:
            List of (name, path, metadata) tuples
        """
        abstractions = []

        for item in self.base_dir.iterdir():
            if not item.is_dir() or item.name.startswith("."):
                continue

            metadata_file = item / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, "r") as f:
                    metadata_dict = json.load(f)
                metadata = EquityBucketMetadata.from_dict(metadata_dict)
                abstractions.append((metadata.name, item, metadata))
            except Exception as e:
                print(f"Warning: Failed to load metadata from {item}: {e}")
                continue

        # Sort by creation time (newest first)
        abstractions.sort(key=lambda x: x[2].created_at, reverse=True)

        return abstractions

    def get_abstraction(self, name_or_alias: str) -> Optional[Path]:
        """
        Get equity buckets by name or alias.

        Args:
            name_or_alias: Equity bucket set name or alias

        Returns:
            Path to abstraction.pkl file, or None if not found
        """
        # Try registry lookup first (faster)
        entry = self.registry.get(name_or_alias)
        if entry:
            abstraction_file = _find_abstraction_file(Path(entry.path))
            if abstraction_file:
                return abstraction_file

        # Fallback: scan directories
        abstractions = self.list_abstractions()

        for abs_name, abs_path, metadata in abstractions:
            if abs_name == name_or_alias or name_or_alias in metadata.aliases:
                abstraction_file = _find_abstraction_file(abs_path)
                if abstraction_file:
                    return abstraction_file

        return None

    def add_alias(self, name_or_alias: str, new_alias: str):
        """
        Add an alias to an abstraction.

        Args:
            name_or_alias: Existing name or alias
            new_alias: New alias to add
        """
        # Find the abstraction
        entry = self.registry.get(name_or_alias)
        if not entry:
            raise ValueError(f"Abstraction '{name_or_alias}' not found")

        # Add alias to registry
        self.registry.add_alias(entry.name, new_alias)

        # Update metadata file
        metadata_file = Path(entry.path) / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata_dict = json.load(f)

            metadata = EquityBucketMetadata.from_dict(metadata_dict)
            if new_alias not in metadata.aliases:
                metadata.aliases.append(new_alias)

            with open(metadata_file, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

    def delete_abstraction(self, name_or_alias: str) -> bool:
        """
        Delete an abstraction.

        Args:
            name_or_alias: Abstraction name or alias

        Returns:
            True if deleted, False if not found
        """
        # Try registry lookup
        entry = self.registry.get(name_or_alias)
        if entry:
            abs_path = Path(entry.path)
            if abs_path.exists():
                shutil.rmtree(abs_path)

            # Remove from registry
            del self.registry.entries[entry.name]
            for alias in entry.aliases:
                if alias in self.registry._alias_map:
                    del self.registry._alias_map[alias]
            self.registry.save()

            return True

        # Fallback: scan directories
        abstractions = self.list_abstractions()

        for abs_name, abs_path, metadata in abstractions:
            if abs_name == name_or_alias or abs_path.name.startswith(name_or_alias):
                shutil.rmtree(abs_path)
                return True

        return False

    def print_summary(self):
        """Print summary of all equity bucket sets."""
        abstractions = self.list_abstractions()

        if not abstractions:
            print("No equity buckets found.")
            return

        print(f"\nAvailable Equity Buckets ({len(abstractions)}):")
        print("=" * 80)

        for name, path, metadata in abstractions:
            print(f"\n{name}")
            if metadata.aliases:
                print(f"  Aliases: {', '.join(metadata.aliases)}")
            print("-" * 80)
            print(f"  Type: {metadata.abstraction_type}")
            print(f"  Created: {metadata.created_at}")
            print(
                f"  Buckets: F={metadata.num_buckets.get('FLOP', 'N/A')}, "
                f"T={metadata.num_buckets.get('TURN', 'N/A')}, "
                f"R={metadata.num_buckets.get('RIVER', 'N/A')}"
            )

            # Find abstraction file
            abstraction_file = _find_abstraction_file(path)
            if abstraction_file:
                file_size_kb = abstraction_file.stat().st_size / 1024
                print(f"  Size: {file_size_kb:.1f} KB")
                print(f"  File: {abstraction_file}")

    def find_or_compute(
        self,
        config_name: str,
        auto_compute: bool = False,
        prompt_user: bool = True,
    ) -> Path:
        """
        Find equity buckets by config name, optionally computing if not found.

        Args:
            config_name: Name of the equity bucket config (e.g., "production")
            auto_compute: If True, automatically compute if not found
            prompt_user: If True, prompt user before computing

        Returns:
            Path to the abstraction.pkl file

        Raises:
            FileNotFoundError: If equity buckets not found and auto_compute=False
        """
        # Load the equity bucket config
        config_dict = load_abstraction_config(config_name)

        # Create metadata from config to compute hash
        from src.abstraction.equity.precompute import PrecomputeConfig

        precompute_config = PrecomputeConfig.from_dict(config_dict)

        # Create metadata for hash computation
        metadata = EquityBucketMetadata(
            name="",  # Will be generated
            created_at=datetime.now().isoformat(),
            abstraction_type="equity_bucketing",
            num_buckets=config_dict["num_buckets"],
            num_board_clusters=config_dict["num_board_clusters"],
            num_equity_samples=config_dict["num_equity_samples"],
            num_samples_per_cluster=config_dict["num_samples_per_cluster"],
            seed=config_dict.get("seed", 42),
        )

        # Check if already computed (by config hash)
        config_hash = compute_config_hash(metadata)
        existing = self.registry.find_by_config_hash(config_hash)

        if existing:
            # Found! Return the path
            abstraction_file = _find_abstraction_file(Path(existing.path))
            if abstraction_file:
                return abstraction_file

        # Not found - check if we should compute
        if not auto_compute and not prompt_user:
            raise FileNotFoundError(
                f"Abstraction '{config_name}' not precomputed.\n"
                f"Run: python scripts/cli.py precompute {config_name}"
            )

        # Prompt user if requested
        if prompt_user and not auto_compute:
            print(f"\nAbstraction '{config_name}' is not precomputed.")
            print(f"Config: {precompute_config}")
            response = input("\nCompute now? This may take a while. [y/N]: ")

            if response.strip().lower() not in ["y", "yes"]:
                raise FileNotFoundError(
                    f"Abstraction '{config_name}' not precomputed and user declined computation."
                )

        # Compute the abstraction
        print(f"\nComputing abstraction '{config_name}'...")
        from src.abstraction.equity.precompute import precompute_equity_bucketing

        precompute_equity_bucketing(
            config=precompute_config,
            save_with_metadata=True,
        )

        # Find it again (should exist now)
        existing = self.registry.find_by_config_hash(config_hash)
        if existing:
            abstraction_file = _find_abstraction_file(Path(existing.path))
            if abstraction_file:
                return abstraction_file

        # Fallback: try to find by aliases
        abstraction = self.get_abstraction(config_name)
        if abstraction:
            return abstraction

        raise RuntimeError(
            f"Failed to compute or find abstraction '{config_name}'. This should not happen."
        )


def load_abstraction_config(name: str) -> Dict:
    """
    Load equity bucket configuration from YAML file.

    Args:
        name: Name of the equity bucket config (e.g., "production")

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If config file not found
    """
    # Look for config file
    config_file = Path(f"config/equity_buckets/{name}.yaml")

    if not config_file.exists():
        raise FileNotFoundError(
            f"Equity bucket config '{name}' not found.\n"
            f"Expected: {config_file}\n"
            f"Available configs: {list_abstraction_configs()}"
        )

    # Load YAML
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Ensure it has a name
    if "name" not in config:
        config["name"] = name

    # Ensure config_name is set for precompute
    config["config_name"] = name

    return config


def list_abstraction_configs() -> List[str]:
    """
    List all available equity bucket configuration files.

    Returns:
        List of config names
    """
    config_dir = Path("config/equity_buckets")
    if not config_dir.exists():
        return []

    configs = []
    for config_file in config_dir.glob("*.yaml"):
        configs.append(config_file.stem)

    return sorted(configs)
