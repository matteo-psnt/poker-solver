"""
Storage systems for CFR information sets.

Provides both in-memory and disk-backed storage with LRU caching
for efficient MCCFR training at scale.
"""

import json
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set

import h5py

from src.abstraction.infoset import InfoSet, InfoSetKey
from src.game.actions import Action


class Storage(ABC):
    """Abstract base class for infoset storage."""

    @abstractmethod
    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """Get existing infoset or create new one."""
        pass

    @abstractmethod
    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        """Get existing infoset or None."""
        pass

    @abstractmethod
    def has_infoset(self, key: InfoSetKey) -> bool:
        """Check if infoset exists."""
        pass

    @abstractmethod
    def num_infosets(self) -> int:
        """Get total number of stored infosets."""
        pass

    @abstractmethod
    def flush(self):
        """Flush any pending writes to storage."""
        pass

    @abstractmethod
    def checkpoint(self, iteration: int):
        """Save a checkpoint at given iteration."""
        pass


class InMemoryStorage(Storage):
    """
    Simple in-memory dictionary storage for infosets.

    Fast but limited by RAM. Suitable for:
    - Testing and development
    - Small games
    - Initial iterations before switching to disk storage
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self.infosets: Dict[InfoSetKey, InfoSet] = {}

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """Get existing infoset or create new one."""
        if key not in self.infosets:
            self.infosets[key] = InfoSet(key, legal_actions)
        return self.infosets[key]

    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        """Get existing infoset or None."""
        return self.infosets.get(key)

    def has_infoset(self, key: InfoSetKey) -> bool:
        """Check if infoset exists."""
        return key in self.infosets

    def num_infosets(self) -> int:
        """Get total number of stored infosets."""
        return len(self.infosets)

    def flush(self):
        """No-op for in-memory storage."""
        pass

    def checkpoint(self, iteration: int):
        """No-op for in-memory storage (could save to disk if needed)."""
        pass

    def clear(self):
        """Clear all infosets."""
        self.infosets.clear()

    def __str__(self) -> str:
        return f"InMemoryStorage(num_infosets={self.num_infosets()})"


class DiskBackedStorage(Storage):
    """
    Disk-backed storage with LRU caching for large-scale training.

    Uses:
    - LRU cache for frequently accessed infosets (in RAM)
    - HDF5 files for persistent storage (on disk)
    - Lazy writing (dirty tracking)

    Designed to handle 100K-1M+ infosets across 10M+ iterations.
    """

    def __init__(self, checkpoint_dir: Path, cache_size: int = 100000, flush_frequency: int = 1000):
        """
        Initialize disk-backed storage.

        Args:
            checkpoint_dir: Directory for checkpoint files
            cache_size: Maximum infosets to keep in LRU cache
            flush_frequency: Flush dirty infosets every N accesses
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.cache_size = cache_size
        self.flush_frequency = flush_frequency

        # LRU cache (OrderedDict maintains insertion order)
        self.cache: OrderedDict[InfoSetKey, InfoSet] = OrderedDict()

        # Track modified infosets
        self.dirty_keys: Set[InfoSetKey] = set()

        # Infoset key to integer ID mapping (for HDF5 indexing)
        self.key_to_id: Dict[InfoSetKey, int] = {}
        self.id_to_key: Dict[int, InfoSetKey] = {}
        self.next_id = 0

        # Action mapping (store legal actions separately)
        self.infoset_actions: Dict[InfoSetKey, List[Action]] = {}

        # Access counter for periodic flushing
        self.access_count = 0

        # Load existing data if present
        self._load_metadata()

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """Get existing infoset or create new one."""
        self.access_count += 1

        # Periodic flush
        if self.access_count % self.flush_frequency == 0:
            self.flush()

        # Check cache first
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]

        # Try to load from disk
        infoset = self._load_from_disk(key, legal_actions)

        if infoset is None:
            # Create new infoset
            infoset = InfoSet(key, legal_actions)
            self._assign_id(key, legal_actions)

        # Add to cache
        self._add_to_cache(key, infoset)

        return infoset

    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        """Get existing infoset or None."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        # Try to load from disk
        if key in self.key_to_id:
            legal_actions = self.infoset_actions.get(key, [])
            infoset = self._load_from_disk(key, legal_actions)
            if infoset is not None:
                # Add loaded infoset to cache (without marking as dirty)
                self.cache[key] = infoset
                self.cache.move_to_end(key)

                # Evict LRU if cache is full
                if len(self.cache) > self.cache_size:
                    oldest_key, oldest_infoset = self.cache.popitem(last=False)
                    if oldest_key in self.dirty_keys:
                        self._write_to_disk(oldest_key, oldest_infoset)
                        self.dirty_keys.discard(oldest_key)

            return infoset

        return None

    def has_infoset(self, key: InfoSetKey) -> bool:
        """Check if infoset exists."""
        return key in self.cache or key in self.key_to_id

    def mark_dirty(self, key: InfoSetKey):
        """Mark infoset as modified (needs to be written to disk)."""
        if key in self.cache:
            self.dirty_keys.add(key)

    def num_infosets(self) -> int:
        """Get total number of stored infosets."""
        return len(self.key_to_id)

    def flush(self):
        """Write dirty infosets and metadata to disk."""
        if not self.dirty_keys:
            return

        # Write all dirty infosets
        for key in self.dirty_keys:
            if key in self.cache:
                self._write_to_disk(key, self.cache[key])

        self.dirty_keys.clear()

        # Also save metadata so storage can be reloaded
        self._save_metadata(iteration=0)  # Use dummy iteration for flush

    def checkpoint(self, iteration: int):
        """Save complete checkpoint."""
        print(f"Checkpointing at iteration {iteration}...")

        # Flush all dirty data
        self.flush()

        # Save metadata
        self._save_metadata(iteration)

        print(f"Checkpoint saved: {self.num_infosets()} infosets")

    def _add_to_cache(self, key: InfoSetKey, infoset: InfoSet):
        """Add infoset to cache, evicting LRU if full."""
        # Add to cache
        self.cache[key] = infoset
        self.cache.move_to_end(key)

        # Mark as dirty (needs to be written)
        self.dirty_keys.add(key)

        # Evict LRU if cache is full
        if len(self.cache) > self.cache_size:
            # Remove oldest item
            oldest_key, oldest_infoset = self.cache.popitem(last=False)

            # Write to disk if dirty
            if oldest_key in self.dirty_keys:
                self._write_to_disk(oldest_key, oldest_infoset)
                self.dirty_keys.discard(oldest_key)

    def _assign_id(self, key: InfoSetKey, legal_actions: List[Action]):
        """Assign integer ID to new infoset."""
        if key not in self.key_to_id:
            infoset_id = self.next_id
            self.key_to_id[key] = infoset_id
            self.id_to_key[infoset_id] = key
            self.infoset_actions[key] = legal_actions
            self.next_id += 1

    def _load_from_disk(self, key: InfoSetKey, legal_actions: List[Action]) -> Optional[InfoSet]:
        """Load infoset from HDF5 files."""
        if key not in self.key_to_id:
            return None

        infoset_id = self.key_to_id[key]

        # Load regrets and strategy sums
        regret_file = self.checkpoint_dir / "regrets.h5"
        strategy_file = self.checkpoint_dir / "strategies.h5"

        if not regret_file.exists() or not strategy_file.exists():
            return None

        try:
            with h5py.File(regret_file, "r") as f:
                if str(infoset_id) not in f:
                    return None
                regrets = f[str(infoset_id)][:]

            with h5py.File(strategy_file, "r") as f:
                if str(infoset_id) not in f:
                    return None
                strategy_sum = f[str(infoset_id)][:]

            # Load reach_count and cumulative_utility (if they exist)
            reach_count = 0
            cumulative_utility = 0.0

            reach_file = self.checkpoint_dir / "reach_counts.h5"
            if reach_file.exists():
                try:
                    with h5py.File(reach_file, "r") as f:
                        if str(infoset_id) in f:
                            reach_count = int(f[str(infoset_id)][()])
                except Exception:
                    pass

            utility_file = self.checkpoint_dir / "utilities.h5"
            if utility_file.exists():
                try:
                    with h5py.File(utility_file, "r") as f:
                        if str(infoset_id) in f:
                            cumulative_utility = float(f[str(infoset_id)][()])
                except Exception:
                    pass

            # Create infoset with stored legal_actions (not current ones)
            # This ensures regrets/strategy_sum sizes match
            stored_actions = self.infoset_actions.get(key, legal_actions)
            infoset = InfoSet(key, stored_actions)
            infoset.regrets = regrets
            infoset.strategy_sum = strategy_sum
            infoset.reach_count = reach_count
            infoset.cumulative_utility = cumulative_utility

            return infoset

        except Exception as e:
            print(f"Warning: Failed to load infoset {infoset_id}: {e}")
            return None

    def _write_to_disk(self, key: InfoSetKey, infoset: InfoSet):
        """Write infoset to HDF5 files."""
        if key not in self.key_to_id:
            self._assign_id(key, infoset.legal_actions)

        infoset_id = self.key_to_id[key]

        # Write regrets
        regret_file = self.checkpoint_dir / "regrets.h5"
        with h5py.File(regret_file, "a") as f:
            dataset_name = str(infoset_id)
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(
                dataset_name, data=infoset.regrets, compression="gzip", compression_opts=1
            )

        # Write strategy sum
        strategy_file = self.checkpoint_dir / "strategies.h5"
        with h5py.File(strategy_file, "a") as f:
            dataset_name = str(infoset_id)
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(
                dataset_name, data=infoset.strategy_sum, compression="gzip", compression_opts=1
            )

        # Write reach count
        reach_file = self.checkpoint_dir / "reach_counts.h5"
        with h5py.File(reach_file, "a") as f:
            dataset_name = str(infoset_id)
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=infoset.reach_count)

        # Write cumulative utility
        utility_file = self.checkpoint_dir / "utilities.h5"
        with h5py.File(utility_file, "a") as f:
            dataset_name = str(infoset_id)
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=infoset.cumulative_utility)

    def _save_metadata(self, iteration: int):
        """Save metadata (key mappings, iteration, etc.)."""
        metadata = {
            "iteration": iteration,
            "num_infosets": self.num_infosets(),
            "next_id": self.next_id,
        }

        # Save metadata
        metadata_file = self.checkpoint_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save key mappings
        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        with open(key_mapping_file, "wb") as f:
            pickle.dump(
                {
                    "key_to_id": self.key_to_id,
                    "id_to_key": self.id_to_key,
                    "infoset_actions": self.infoset_actions,
                },
                f,
            )

    def _load_metadata(self):
        """Load metadata from disk if exists."""
        metadata_file = self.checkpoint_dir / "metadata.json"
        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"

        if not metadata_file.exists() or not key_mapping_file.exists():
            return

        try:
            # Load metadata
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                self.next_id = metadata.get("next_id", 0)

            # Load key mappings
            with open(key_mapping_file, "rb") as f:
                data = pickle.load(f)
                self.key_to_id = data["key_to_id"]
                self.id_to_key = data["id_to_key"]
                self.infoset_actions = data["infoset_actions"]

            print(f"Loaded checkpoint: {self.num_infosets()} infosets")

        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")

    def __str__(self) -> str:
        return (
            f"DiskBackedStorage(num_infosets={self.num_infosets()}, "
            f"cache_size={len(self.cache)}, dirty={len(self.dirty_keys)})"
        )
