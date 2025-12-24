"""
Storage systems for CFR information sets.

Provides both in-memory and disk-backed storage with LRU caching
for efficient MCCFR training at scale.
"""

import json
import logging
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import h5py
import numpy as np

from src.bucketing.utils.infoset import InfoSet, InfoSetKey
from src.game.actions import Action

# Setup logger
logger = logging.getLogger(__name__)


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
    def mark_dirty(self, key: InfoSetKey):
        """Mark infoset as modified (needs to be persisted)."""
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

    def mark_dirty(self, key: InfoSetKey):
        """No-op for in-memory storage (always in sync)."""
        pass

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

    IMPORTANT CONTRACT:
    ==================
    After mutating an infoset's regrets or strategy_sum, you MUST call:

        storage.mark_dirty(infoset_key)

    Failure to mark dirty will cause updates to be lost when the infoset
    is evicted from cache. This is a critical correctness requirement.

    The solver (mccfr.py) is responsible for calling mark_dirty after
    all mutations. Do not mutate infosets outside the solver.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        cache_size: int = 100000,
        flush_frequency: int = 1000,
        enable_dirty_checks: bool = False,
    ):
        """
        Initialize disk-backed storage.

        Args:
            checkpoint_dir: Directory for checkpoint files
            cache_size: Maximum infosets to keep in LRU cache
            flush_frequency: Flush dirty infosets every N accesses
            enable_dirty_checks: Enable debug checks for missing mark_dirty calls
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.cache_size = cache_size
        self.flush_frequency = flush_frequency
        self.enable_dirty_checks = enable_dirty_checks

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

        # Debug mode: snapshot infoset state to detect unmarked mutations
        if self.enable_dirty_checks:
            self.infoset_snapshots: Dict[InfoSetKey, tuple] = {}

        # Load existing data if present
        self._load_metadata()

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """
        Get existing infoset or create new one.

        IMPORTANT: After mutating the returned infoset, you MUST call mark_dirty(key).
        """
        self.access_count += 1

        # Periodic flush
        if self.access_count % self.flush_frequency == 0:
            self.flush()

        # Check cache first
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            infoset = self.cache[key]

            # Debug mode: snapshot state before returning (to detect unmarked mutations)
            if self.enable_dirty_checks and key not in self.dirty_keys:
                self._snapshot_infoset(key, infoset)

            return infoset

        # Try to load from disk
        infoset = self._load_from_disk(key, legal_actions)

        if infoset is None:
            # Create new infoset
            infoset = InfoSet(key, legal_actions)
            self._assign_id(key, legal_actions)

        # Add to cache
        self._add_to_cache(key, infoset)

        # Debug mode: snapshot new infoset state
        if self.enable_dirty_checks:
            self._snapshot_infoset(key, infoset)

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

                    # Debug mode: check for unmarked mutations before eviction
                    if self.enable_dirty_checks:
                        if self._check_unmarked_mutation(oldest_key, oldest_infoset):
                            logger.warning(
                                f"DIRTY TRACKING BUG: Infoset {oldest_key} was mutated "
                                "but mark_dirty() was not called. Updates will be lost! "
                                "This is a critical correctness bug."
                            )

                    if oldest_key in self.dirty_keys:
                        self._write_to_disk(oldest_key, oldest_infoset)
                        self.dirty_keys.discard(oldest_key)

                    # Clear snapshot after eviction
                    if self.enable_dirty_checks and oldest_key in self.infoset_snapshots:
                        del self.infoset_snapshots[oldest_key]

            return infoset

        return None

    def has_infoset(self, key: InfoSetKey) -> bool:
        """Check if infoset exists."""
        return key in self.cache or key in self.key_to_id

    def mark_dirty(self, key: InfoSetKey):
        """Mark infoset as modified (needs to be written to disk)."""
        if key in self.cache:
            self.dirty_keys.add(key)

            # Debug mode: clear snapshot since we've acknowledged the mutation
            if self.enable_dirty_checks and key in self.infoset_snapshots:
                del self.infoset_snapshots[key]

    def _snapshot_infoset(self, key: InfoSetKey, infoset: InfoSet):
        """
        Take snapshot of infoset state for debug checks.

        Stores a copy of regrets and strategy_sum to detect unmarked mutations.
        """
        self.infoset_snapshots[key] = (
            infoset.regrets.copy(),
            infoset.strategy_sum.copy(),
        )

    def _check_unmarked_mutation(self, key: InfoSetKey, infoset: InfoSet) -> bool:
        """
        Check if infoset was mutated without calling mark_dirty.

        Returns True if unmarked mutation detected, False otherwise.
        """
        if key not in self.infoset_snapshots:
            return False

        snapshot_regrets, snapshot_strategies = self.infoset_snapshots[key]

        # Check if arrays were modified
        regrets_changed = not np.array_equal(infoset.regrets, snapshot_regrets)
        strategies_changed = not np.array_equal(infoset.strategy_sum, snapshot_strategies)

        return regrets_changed or strategies_changed

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
        logger.info(f"Checkpointing at iteration {iteration}...")

        # Flush all dirty data
        self.flush()

        # Save metadata
        self._save_metadata(iteration)

        logger.info(f"Checkpoint saved: {self.num_infosets()} infosets")

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

            # Debug mode: check for unmarked mutations before eviction
            if self.enable_dirty_checks:
                if self._check_unmarked_mutation(oldest_key, oldest_infoset):
                    logger.warning(
                        f"DIRTY TRACKING BUG: Infoset {oldest_key} was mutated "
                        "but mark_dirty() was not called. Updates will be lost! "
                        "This is a critical correctness bug."
                    )

            # Write to disk if dirty
            if oldest_key in self.dirty_keys:
                self._write_to_disk(oldest_key, oldest_infoset)
                self.dirty_keys.discard(oldest_key)

            # Clear snapshot after eviction
            if self.enable_dirty_checks and oldest_key in self.infoset_snapshots:
                del self.infoset_snapshots[oldest_key]

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

            # Create infoset with stored legal_actions (not current ones)
            # This ensures regrets/strategy_sum sizes match
            stored_actions = self.infoset_actions.get(key, legal_actions)
            infoset = InfoSet(key, stored_actions)
            infoset.regrets = regrets
            infoset.strategy_sum = strategy_sum

            return infoset

        except Exception as e:
            logger.warning(f"Failed to load infoset {infoset_id}: {e}")
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

    def _save_metadata(self, iteration: int):
        """
        Save storage metadata (key mappings only).

        Note:
            Run-level metadata (iteration, stats, etc.) is now handled
            by CheckpointManager. This only saves internal storage state.

        Args:
            iteration: Current iteration (unused, kept for compatibility)
        """
        # Save key mappings (the only state we need to persist)
        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        with open(key_mapping_file, "wb") as f:
            pickle.dump(
                {
                    "key_to_id": self.key_to_id,
                    "id_to_key": self.id_to_key,
                    "infoset_actions": self.infoset_actions,
                    "next_id": self.next_id,
                },
                f,
            )

        logger.debug(
            f"Saved storage metadata: {self.num_infosets()} infosets, next_id={self.next_id}"
        )

    def _load_metadata(self):
        """
        Load storage metadata from disk if exists.

        Supports both old format (metadata.json + key_mapping.pkl)
        and new format (key_mapping.pkl only).
        """
        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"

        if not key_mapping_file.exists():
            logger.debug("No existing checkpoint found")
            return

        try:
            # Load key mappings
            with open(key_mapping_file, "rb") as f:
                data = pickle.load(f)
                self.key_to_id = data["key_to_id"]
                self.id_to_key = data["id_to_key"]
                self.infoset_actions = data["infoset_actions"]

                # next_id might be in key_mapping.pkl (new format) or metadata.json (old format)
                if "next_id" in data:
                    self.next_id = data["next_id"]
                else:
                    # Try to load from old metadata.json
                    metadata_file = self.checkpoint_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r", encoding="utf-8") as f:  # type: ignore[assignment]
                            metadata: Any = json.load(f)
                            self.next_id = metadata.get("next_id", len(self.key_to_id))
                    else:
                        # Fall back to inferring from key_to_id
                        self.next_id = len(self.key_to_id)

            logger.info(f"Loaded storage: {self.num_infosets()} infosets, next_id={self.next_id}")

        except Exception as e:
            logger.warning(f"Failed to load storage metadata: {e}")

    def __str__(self) -> str:
        return (
            f"DiskBackedStorage(num_infosets={self.num_infosets()}, "
            f"cache_size={len(self.cache)}, dirty={len(self.dirty_keys)})"
        )
