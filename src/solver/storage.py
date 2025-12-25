"""
Storage systems for CFR information sets.

Provides in-memory storage with optional disk checkpointing
for efficient MCCFR training at scale.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from src.bucketing.utils.infoset import InfoSet, InfoSetKey
from src.game.actions import Action, fold

# Setup logger
logger = logging.getLogger(__name__)


class Storage(ABC):
    """Abstract base class for infoset storage."""

    # Optional attribute expected by tests and tooling; implementations may set this
    checkpoint_dir: Optional[Path] = None

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
    In-memory dictionary storage for infosets with optional disk checkpointing.

    All operations are in-memory (fast), with periodic saves to disk for
    persistence. This is the recommended approach for training:
    - All training operations use RAM (no I/O overhead)
    - Checkpoint to disk every N iterations to avoid losing progress
    - Can resume from disk checkpoints

    Suitable for:
    - Active training (up to ~1M infosets depending on RAM)
    - Fast iteration with periodic persistence
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize in-memory storage.

        Args:
            checkpoint_dir: Optional directory for saving checkpoints.
                           If provided, enables checkpoint() to save to disk.
        """
        self.infosets: Dict[InfoSetKey, InfoSet] = {}
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # For disk checkpointing, we need key<->id mapping
        self.key_to_id: Dict[InfoSetKey, int] = {}
        self.id_to_key: Dict[int, InfoSetKey] = {}
        self.next_id = 0

        # Load existing checkpoint if present
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            self._load_from_checkpoint()

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """Get existing infoset or create new one."""
        if key not in self.infosets:
            self.infosets[key] = InfoSet(key, legal_actions)
            # Assign ID for checkpoint serialization
            if key not in self.key_to_id:
                self.key_to_id[key] = self.next_id
                self.id_to_key[self.next_id] = key
                self.next_id += 1
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
        """No-op for in-memory storage (no cache to flush)."""
        pass

    def checkpoint(self, iteration: int):
        """
        Save current state to disk checkpoint.

        If checkpoint_dir was provided, saves all infosets to HDF5 files.
        Otherwise, this is a no-op.

        Args:
            iteration: Current iteration number (for logging)
        """
        if not self.checkpoint_dir:
            return

        self._save_to_disk()

    def clear(self):
        """Clear all infosets."""
        self.infosets.clear()
        self.key_to_id.clear()
        self.id_to_key.clear()
        self.next_id = 0

    def _load_from_checkpoint(self):
        """
        Load infosets from disk checkpoint using optimized matrix format.

        Reads large compressed matrices and reconstructs individual infosets.
        """
        if not self.checkpoint_dir:
            return
        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        regrets_file = self.checkpoint_dir / "regrets.h5"
        strategies_file = self.checkpoint_dir / "strategies.h5"

        if not all(f.exists() for f in [key_mapping_file, regrets_file, strategies_file]):
            return  # No checkpoint to load

        # Load key mapping
        with open(key_mapping_file, "rb") as f:
            mapping_data = pickle.load(f)
            self.key_to_id = mapping_data["key_to_id"]
            self.id_to_key = mapping_data["id_to_key"]
            self.next_id = mapping_data["next_id"]

        # Load regrets and strategies from HDF5 (optimized matrix format only)
        with (
            h5py.File(regrets_file, "r") as regrets_h5,
            h5py.File(strategies_file, "r") as strategies_h5,
        ):
            # Load matrices
            all_regrets = regrets_h5["regrets"][:]
            action_counts = regrets_h5["action_counts"][:]
            all_strategies = strategies_h5["strategies"][:]

            # Reconstruct infosets from matrices
            for infoset_id, key in self.id_to_key.items():
                n_actions = action_counts[infoset_id]

                # Extract this infoset's data
                regrets = all_regrets[infoset_id, :n_actions].copy()
                strategies = all_strategies[infoset_id, :n_actions].copy()

                # Create infoset with placeholder legal actions (solver will set real actions)
                legal_actions = [fold() for _ in range(n_actions)]
                infoset = InfoSet(key, legal_actions)
                infoset.regrets = regrets
                infoset.strategy_sum = strategies

                self.infosets[key] = infoset

        logger.info(f"Loaded {len(self.infosets)} infosets from checkpoint")

    def _save_to_disk(self):
        """
        Save all infosets to disk using optimized matrix format.

        Instead of creating individual datasets per infoset (slow),
        we stack all data into large matrices and save with compression.
        This provides 10-50x speedup for checkpoint saves.

        Thread-safe: Creates snapshots of data to avoid issues with
        concurrent modifications during async checkpointing.
        """
        if not self.checkpoint_dir:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create snapshots to avoid "dictionary changed size" errors
        # when running in background thread (async checkpointing)
        key_to_id_snapshot = dict(self.key_to_id)
        id_to_key_snapshot = dict(self.id_to_key)
        next_id_snapshot = self.next_id

        # Save key mapping
        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        with open(key_mapping_file, "wb") as f:
            pickle.dump(
                {
                    "key_to_id": key_to_id_snapshot,
                    "id_to_key": id_to_key_snapshot,
                    "next_id": next_id_snapshot,
                },
                f,
            )

        num_infosets = len(id_to_key_snapshot)
        if num_infosets == 0:
            return

        # Snapshot infoset data (copy arrays to avoid concurrent modifications)
        infoset_data = []
        for infoset_id in range(num_infosets):
            if infoset_id not in id_to_key_snapshot:
                continue
            key = id_to_key_snapshot[infoset_id]
            if key not in self.infosets:
                continue
            infoset = self.infosets[key]
            infoset_data.append(
                {
                    "id": infoset_id,
                    "regrets": infoset.regrets.copy(),
                    "strategies": infoset.strategy_sum.copy(),
                }
            )

        if not infoset_data:
            return

        # Determine max actions across all infosets
        max_actions = max(len(item["regrets"]) for item in infoset_data)

        # Pre-allocate arrays
        all_regrets = np.zeros((num_infosets, max_actions), dtype=np.float32)
        all_strategies = np.zeros((num_infosets, max_actions), dtype=np.float32)
        action_counts = np.zeros(num_infosets, dtype=np.int32)

        # Fill arrays from snapshot
        for item in infoset_data:
            infoset_id = item["id"]
            regrets = item["regrets"]
            strategies = item["strategies"]
            n_actions = len(regrets)

            all_regrets[infoset_id, :n_actions] = regrets
            all_strategies[infoset_id, :n_actions] = strategies
            action_counts[infoset_id] = n_actions

        # Save to HDF5 with compression
        regrets_file = self.checkpoint_dir / "regrets.h5"
        strategies_file = self.checkpoint_dir / "strategies.h5"

        with h5py.File(regrets_file, "w") as regrets_h5:
            regrets_h5.create_dataset(
                "regrets",
                data=all_regrets,
                compression="gzip",
                compression_opts=4,  # Compression level 1-9 (4 is good balance)
                chunks=True,  # Enable chunking for better compression
            )
            regrets_h5.create_dataset("action_counts", data=action_counts)

        with h5py.File(strategies_file, "w") as strategies_h5:
            strategies_h5.create_dataset(
                "strategies",
                data=all_strategies,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )

    def __str__(self) -> str:
        checkpoint_info = f", checkpoint_dir={self.checkpoint_dir}" if self.checkpoint_dir else ""
        return f"InMemoryStorage(num_infosets={self.num_infosets()}{checkpoint_info})"
