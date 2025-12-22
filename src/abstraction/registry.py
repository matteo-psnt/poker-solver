"""
Registry for tracking card abstractions.

Provides persistent storage of abstraction metadata with lookup by name,
alias, or configuration hash.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.abstraction.metadata import RegistryEntry


class AbstractionRegistry:
    """Registry for tracking abstractions by config hash and aliases."""

    def __init__(self, registry_file: Path):
        """
        Initialize registry.

        Args:
            registry_file: Path to registry JSON file
        """
        self.registry_file = Path(registry_file)
        self.entries: Dict[str, RegistryEntry] = {}
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
                entry = RegistryEntry.from_dict(entry_data)
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

        entry = RegistryEntry(
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

    def get(self, name_or_alias: str) -> Optional[RegistryEntry]:
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

    def find_by_config_hash(self, config_hash: str) -> Optional[RegistryEntry]:
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

    def list_all(self) -> List[RegistryEntry]:
        """List all registered abstractions."""
        return list(self.entries.values())
