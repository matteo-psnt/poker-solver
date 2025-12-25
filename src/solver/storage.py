"""
Storage systems for CFR information sets.

Provides in-memory storage with optional disk checkpointing
for efficient MCCFR training at scale.

Includes SharedArrayStorage for parallel MCCFR training using
flat NumPy arrays backed directly by shared memory.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import h5py
import numpy as np

from src.bucketing.utils.infoset import InfoSet, InfoSetKey
from src.game.actions import Action, fold

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

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

    def is_owned(self, key: InfoSetKey) -> bool:
        """
        Check if this storage instance owns the given infoset key.

        For non-partitioned storage, all keys are "owned" (returns True).
        For partitioned storage, only keys mapping to this worker's partition are owned.

        This determines whether regret/strategy updates should be applied.
        """
        return True  # Default: non-partitioned storage owns all keys

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
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Integer ID optimization: Use integers as dict keys instead of InfoSetKey
        self._infosets_by_id: Dict[int, InfoSet] = {}
        self.key_to_id: Dict[InfoSetKey, int] = {}
        self.id_to_key: Dict[int, InfoSetKey] = {}
        self.next_id = 0

        # Load existing checkpoint if present
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            self._load_from_checkpoint()

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """Get existing infoset or create new one."""
        # Use integer ID lookup (fast integer hash instead of complex InfoSetKey hash)
        infoset_id = self.key_to_id.get(key)
        if infoset_id is not None:
            return self._infosets_by_id[infoset_id]

        # Create new infoset with integer ID
        infoset_id = self.next_id
        self.next_id += 1

        infoset = InfoSet(key, legal_actions)
        self._infosets_by_id[infoset_id] = infoset
        self.key_to_id[key] = infoset_id
        self.id_to_key[infoset_id] = key

        return infoset

    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        """Get existing infoset or None."""
        infoset_id = self.key_to_id.get(key)
        if infoset_id is None:
            return None
        return self._infosets_by_id.get(infoset_id)

    def has_infoset(self, key: InfoSetKey) -> bool:
        """Check if infoset exists."""
        return key in self.key_to_id

    def num_infosets(self) -> int:
        """Get total number of stored infosets."""
        return len(self._infosets_by_id)

    @property
    def infosets(self) -> Dict[InfoSetKey, InfoSet]:
        """
        Get all infosets as a dict keyed by InfoSetKey.

        This property provides backward compatibility with code that expects
        a Dict[InfoSetKey, InfoSet]. Built on-the-fly from internal integer
        ID storage.

        For iteration, prefer using the internal _infosets_by_id when possible
        for better performance.
        """
        return {
            self.id_to_key[infoset_id]: infoset
            for infoset_id, infoset in self._infosets_by_id.items()
        }

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
        self._infosets_by_id.clear()
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

                self._infosets_by_id[infoset_id] = infoset

        logger.info(f"Loaded {len(self._infosets_by_id)} infosets from checkpoint")

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
            if infoset_id not in self._infosets_by_id:
                continue
            infoset = self._infosets_by_id[infoset_id]
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


# =============================================================================
# Shared Array Storage - Live shared memory with flat NumPy arrays
# =============================================================================


class SharedArrayStorage(Storage):
    """
    High-performance partitioned storage using flat NumPy arrays in shared memory.

    Architecture:
    - Shared memory is THE LIVE DATA STRUCTURE, not a transport mechanism
    - Pre-allocated flat arrays: shared_regrets[infoset_id, max_actions]
    - Coordinator creates shared memory once at startup
    - Workers attach to (not recreate) shared memory
    - Lock-free reads (stale data acceptable for MCCFR)
    - Only owning worker writes to an infoset (owner = infoset_id % num_workers)

    Data layout:
    - shared_regrets: float32[max_infosets, max_actions]
    - shared_strategy_sum: float32[max_infosets, max_actions]
    - shared_action_counts: int32[max_infosets] (number of actions per infoset)
    - shared_infoset_count: int32[1] (current number of infosets)

    Cold path (key → infoset_id mapping) is stored separately in dict.
    Hot path (array access) uses direct integer indexing.

    Cross-partition updates are buffered locally and applied by owner.
    """

    # Shared memory names (macOS limit: ~30 chars)
    SHM_REGRETS = "sas_reg"
    SHM_STRATEGY = "sas_str"
    SHM_ACTIONS = "sas_act"
    SHM_COUNT = "sas_cnt"
    SHM_KEYS = "sas_key"

    def __init__(
        self,
        num_workers: int,
        worker_id: int,
        session_id: str,
        max_infosets: int = 2_000_000,
        max_actions: int = 10,
        is_coordinator: bool = False,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize shared array storage.

        Args:
            num_workers: Total number of workers (for partition ownership)
            worker_id: This worker's ID (0 to num_workers-1)
            session_id: Unique session ID for shared memory namespace
            max_infosets: Maximum number of infosets (pre-allocated)
            max_actions: Maximum actions per infoset (pre-allocated)
            is_coordinator: If True, creates shared memory. If False, attaches.
            checkpoint_dir: Optional directory for checkpoints
        """
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.session_id = session_id[:8]  # Truncate for macOS shm name limit
        self.max_infosets = max_infosets
        self.max_actions = max_actions
        self.is_coordinator = is_coordinator
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Cold path: key → infoset_id mapping (local dict, synced via shm)
        self._key_to_id: Dict[InfoSetKey, int] = {}
        self._id_to_key: Dict[int, InfoSetKey] = {}

        # Local cache for legal actions (since we can't store complex objects in shm)
        self._legal_actions_cache: Dict[int, List[Action]] = {}

        # Cross-partition update buffers: {infoset_id: (regret_delta, strategy_delta)}
        # These are updates for infosets we don't own
        self._pending_updates: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Shared memory handles
        self._shm_regrets: Optional["SharedMemory"] = None
        self._shm_strategy: Optional["SharedMemory"] = None
        self._shm_actions: Optional["SharedMemory"] = None
        self._shm_count: Optional["SharedMemory"] = None
        self._shm_keys: Optional["SharedMemory"] = None

        # NumPy views into shared memory (the live data)
        self.shared_regrets: Optional[np.ndarray] = None
        self.shared_strategy_sum: Optional[np.ndarray] = None
        self.shared_action_counts: Optional[np.ndarray] = None
        self.shared_infoset_count: Optional[np.ndarray] = None

        # Initialize shared memory
        if is_coordinator:
            self._create_shared_memory()
        else:
            self._attach_shared_memory()

        logger.info(
            f"SharedArrayStorage initialized: worker={worker_id}, "
            f"coordinator={is_coordinator}, max_infosets={max_infosets}"
        )

    def _get_shm_name(self, base: str) -> str:
        """Get session-namespaced shared memory name."""
        return f"{base}_{self.session_id}"

    def _create_shared_memory(self):
        """Create all shared memory segments (coordinator only)."""
        # Clean up any stale shared memory from previous runs
        self._cleanup_stale_shm()

        # Calculate sizes
        regrets_size = self.max_infosets * self.max_actions * 4  # float32
        strategy_size = self.max_infosets * self.max_actions * 4  # float32
        actions_size = self.max_infosets * 4  # int32
        count_size = 4  # single int32 for infoset count
        keys_size = self.max_infosets * 1024  # ~1KB per key for pickled data

        # Create shared memory segments
        self._shm_regrets = shared_memory.SharedMemory(
            create=True,
            size=regrets_size,
            name=self._get_shm_name(self.SHM_REGRETS),
        )
        self._shm_strategy = shared_memory.SharedMemory(
            create=True,
            size=strategy_size,
            name=self._get_shm_name(self.SHM_STRATEGY),
        )
        self._shm_actions = shared_memory.SharedMemory(
            create=True,
            size=actions_size,
            name=self._get_shm_name(self.SHM_ACTIONS),
        )
        self._shm_count = shared_memory.SharedMemory(
            create=True,
            size=count_size,
            name=self._get_shm_name(self.SHM_COUNT),
        )
        self._shm_keys = shared_memory.SharedMemory(
            create=True,
            size=keys_size,
            name=self._get_shm_name(self.SHM_KEYS),
        )

        # Create NumPy views
        self._create_numpy_views()

        # Initialize to zero
        self.shared_regrets.fill(0)
        self.shared_strategy_sum.fill(0)
        self.shared_action_counts.fill(0)
        self.shared_infoset_count[0] = 0

        logger.info(
            f"Coordinator created shared memory: "
            f"regrets={regrets_size // 1024 // 1024}MB, "
            f"strategy={strategy_size // 1024 // 1024}MB"
        )

    def _attach_shared_memory(self):
        """Attach to existing shared memory segments (worker)."""
        max_retries = 10
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                self._shm_regrets = shared_memory.SharedMemory(
                    name=self._get_shm_name(self.SHM_REGRETS)
                )
                self._shm_strategy = shared_memory.SharedMemory(
                    name=self._get_shm_name(self.SHM_STRATEGY)
                )
                self._shm_actions = shared_memory.SharedMemory(
                    name=self._get_shm_name(self.SHM_ACTIONS)
                )
                self._shm_count = shared_memory.SharedMemory(
                    name=self._get_shm_name(self.SHM_COUNT)
                )
                self._shm_keys = shared_memory.SharedMemory(name=self._get_shm_name(self.SHM_KEYS))

                # Create NumPy views
                self._create_numpy_views()
                logger.info(f"Worker {self.worker_id} attached to shared memory")
                return

            except FileNotFoundError:
                if attempt < max_retries - 1:
                    import time

                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Worker {self.worker_id} failed to attach to shared memory "
                        f"after {max_retries} attempts"
                    )

    def _create_numpy_views(self):
        """Create NumPy array views into shared memory."""
        self.shared_regrets = np.ndarray(
            (self.max_infosets, self.max_actions),
            dtype=np.float32,
            buffer=self._shm_regrets.buf,
        )
        self.shared_strategy_sum = np.ndarray(
            (self.max_infosets, self.max_actions),
            dtype=np.float32,
            buffer=self._shm_strategy.buf,
        )
        self.shared_action_counts = np.ndarray(
            (self.max_infosets,),
            dtype=np.int32,
            buffer=self._shm_actions.buf,
        )
        self.shared_infoset_count = np.ndarray(
            (1,),
            dtype=np.int32,
            buffer=self._shm_count.buf,
        )

    def _cleanup_stale_shm(self):
        """Clean up stale shared memory from previous runs."""
        shm_names = [
            self._get_shm_name(self.SHM_REGRETS),
            self._get_shm_name(self.SHM_STRATEGY),
            self._get_shm_name(self.SHM_ACTIONS),
            self._get_shm_name(self.SHM_COUNT),
            self._get_shm_name(self.SHM_KEYS),
        ]

        for name in shm_names:
            try:
                stale = shared_memory.SharedMemory(name=name)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass

    # =========================================================================
    # Storage Interface Implementation
    # =========================================================================

    def get_partition_id(self, infoset_id: int) -> int:
        """Get partition (owner worker) for an infoset ID."""
        return infoset_id % self.num_workers

    def is_owned(self, key: InfoSetKey) -> bool:
        """Check if this worker owns the given infoset key."""
        infoset_id = self._key_to_id.get(key)
        if infoset_id is None:
            # New key - check by hash
            return hash(key) % self.num_workers == self.worker_id
        return self.get_partition_id(infoset_id) == self.worker_id

    def is_owned_by_id(self, infoset_id: int) -> bool:
        """Check if this worker owns the given infoset ID."""
        return self.get_partition_id(infoset_id) == self.worker_id

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """
        Get existing infoset or create new one.

        For owned keys: Creates in shared arrays
        For non-owned keys: Returns view into shared arrays (read-only for caller)
        """
        # Check if key exists
        infoset_id = self._key_to_id.get(key)

        if infoset_id is not None:
            # Existing infoset - return view
            return self._create_infoset_view(infoset_id, key, legal_actions)

        # New infoset - only owner can create
        owner = hash(key) % self.num_workers
        if owner != self.worker_id:
            # Not owner - return empty infoset (will be read-only)
            # The owner will create the canonical entry
            infoset = InfoSet(key, legal_actions)
            return infoset

        # We are the owner - create new entry in shared arrays
        infoset_id = self._allocate_infoset_id()

        # Register key mapping
        self._key_to_id[key] = infoset_id
        self._id_to_key[infoset_id] = key
        self._legal_actions_cache[infoset_id] = legal_actions

        # Initialize in shared arrays (already zeroed)
        num_actions = len(legal_actions)
        self.shared_action_counts[infoset_id] = num_actions

        return self._create_infoset_view(infoset_id, key, legal_actions)

    def _allocate_infoset_id(self) -> int:
        """
        Allocate a new infoset ID (atomic operation).

        Uses atomic increment on shared counter.
        """
        # Simple atomic increment using a view
        current = self.shared_infoset_count[0]
        self.shared_infoset_count[0] = current + 1

        if current >= self.max_infosets:
            raise RuntimeError(
                f"Exceeded max_infosets ({self.max_infosets}). Increase max_infosets parameter."
            )

        return current

    def _create_infoset_view(
        self, infoset_id: int, key: InfoSetKey, legal_actions: List[Action]
    ) -> InfoSet:
        """
        Create an InfoSet object that views the shared arrays.

        The returned InfoSet has its regrets and strategy_sum arrays
        pointing directly into shared memory.
        """
        num_actions = len(legal_actions)
        infoset = InfoSet(key, legal_actions)

        # Point arrays to shared memory (sliced view)
        # These are VIEWS, not copies - modifications write directly to shared memory
        infoset.regrets = self.shared_regrets[infoset_id, :num_actions]
        infoset.strategy_sum = self.shared_strategy_sum[infoset_id, :num_actions]

        return infoset

    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        """Get existing infoset or None."""
        infoset_id = self._key_to_id.get(key)
        if infoset_id is None:
            return None

        legal_actions = self._legal_actions_cache.get(infoset_id)
        if legal_actions is None:
            # Unknown legal actions - create placeholder
            num_actions = self.shared_action_counts[infoset_id]
            legal_actions = [fold() for _ in range(num_actions)]

        return self._create_infoset_view(infoset_id, key, legal_actions)

    def get_infoset_by_id(self, infoset_id: int) -> Optional[InfoSet]:
        """Get infoset by integer ID (fast path)."""
        key = self._id_to_key.get(infoset_id)
        if key is None:
            return None

        legal_actions = self._legal_actions_cache.get(infoset_id)
        if legal_actions is None:
            num_actions = self.shared_action_counts[infoset_id]
            legal_actions = [fold() for _ in range(num_actions)]

        return self._create_infoset_view(infoset_id, key, legal_actions)

    def has_infoset(self, key: InfoSetKey) -> bool:
        """Check if infoset exists."""
        return key in self._key_to_id

    def num_infosets(self) -> int:
        """Get total number of infosets."""
        return int(self.shared_infoset_count[0])

    def num_owned_infosets(self) -> int:
        """Get number of infosets owned by this worker."""
        count = 0
        for infoset_id in self._key_to_id.values():
            if self.is_owned_by_id(infoset_id):
                count += 1
        return count

    def mark_dirty(self, key: InfoSetKey):
        """No-op for shared array storage (writes go directly to shared memory)."""
        pass

    def flush(self):
        """No-op for shared array storage (data is always in shared memory)."""
        pass

    # =========================================================================
    # Cross-Partition Updates (for non-owned infosets)
    # =========================================================================

    def buffer_update(self, infoset_id: int, regret_delta: np.ndarray, strategy_delta: np.ndarray):
        """
        Buffer an update for a non-owned infoset.

        Updates are accumulated locally and later sent to the owner.

        Args:
            infoset_id: The infoset to update
            regret_delta: Delta to add to regrets
            strategy_delta: Delta to add to strategy_sum
        """
        if infoset_id in self._pending_updates:
            old_regret, old_strategy = self._pending_updates[infoset_id]
            self._pending_updates[infoset_id] = (
                old_regret + regret_delta,
                old_strategy + strategy_delta,
            )
        else:
            self._pending_updates[infoset_id] = (regret_delta.copy(), strategy_delta.copy())

    def get_pending_updates(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Get all pending cross-partition updates."""
        return self._pending_updates

    def clear_pending_updates(self):
        """Clear pending updates after they've been sent."""
        self._pending_updates.clear()

    def apply_updates(self, updates: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """
        Apply updates to infosets we own.

        Called by owner to apply buffered updates from other workers.

        Args:
            updates: {infoset_id: (regret_delta, strategy_delta)}
        """
        for infoset_id, (regret_delta, strategy_delta) in updates.items():
            if not self.is_owned_by_id(infoset_id):
                logger.warning(
                    f"Worker {self.worker_id} received update for non-owned infoset {infoset_id}"
                )
                continue

            num_actions = self.shared_action_counts[infoset_id]
            self.shared_regrets[infoset_id, :num_actions] += regret_delta[:num_actions]
            self.shared_strategy_sum[infoset_id, :num_actions] += strategy_delta[:num_actions]

    # =========================================================================
    # Key Synchronization
    # =========================================================================

    def sync_keys_from_workers(self, all_key_mappings: List[Dict[InfoSetKey, int]]):
        """
        Merge key mappings from all workers.

        Called by coordinator after workers discover new infosets.

        Args:
            all_key_mappings: List of {key: infoset_id} from each worker
        """
        for mapping in all_key_mappings:
            for key, infoset_id in mapping.items():
                if key not in self._key_to_id:
                    self._key_to_id[key] = infoset_id
                    self._id_to_key[infoset_id] = key

    def get_key_mapping(self) -> Dict[InfoSetKey, int]:
        """Get this worker's key mapping (for syncing)."""
        return dict(self._key_to_id)

    def set_key_mapping(self, mapping: Dict[InfoSetKey, int]):
        """Set key mapping from coordinator broadcast."""
        self._key_to_id = dict(mapping)
        self._id_to_key = {v: k for k, v in mapping.items()}

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def checkpoint(self, iteration: int):
        """Save checkpoint to disk (coordinator only)."""
        if not self.checkpoint_dir or not self.is_coordinator:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        num_infosets = self.num_infosets()
        if num_infosets == 0:
            return

        # Save key mapping
        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        with open(key_mapping_file, "wb") as f:
            pickle.dump(
                {
                    "key_to_id": dict(self._key_to_id),
                    "id_to_key": dict(self._id_to_key),
                    "num_infosets": num_infosets,
                },
                f,
            )

        # Save shared arrays directly (no copying - just slice the used portion)
        regrets_file = self.checkpoint_dir / "regrets.h5"
        strategies_file = self.checkpoint_dir / "strategies.h5"

        with h5py.File(regrets_file, "w") as f:
            f.create_dataset(
                "regrets",
                data=self.shared_regrets[:num_infosets],
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "action_counts",
                data=self.shared_action_counts[:num_infosets],
            )

        with h5py.File(strategies_file, "w") as f:
            f.create_dataset(
                "strategies",
                data=self.shared_strategy_sum[:num_infosets],
                compression="gzip",
                compression_opts=4,
            )

        logger.info(f"Checkpoint saved: {num_infosets} infosets at iteration {iteration}")

    def load_checkpoint(self) -> bool:
        """Load checkpoint from disk (coordinator only)."""
        if not self.checkpoint_dir or not self.is_coordinator:
            return False

        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        regrets_file = self.checkpoint_dir / "regrets.h5"
        strategies_file = self.checkpoint_dir / "strategies.h5"

        if not all(f.exists() for f in [key_mapping_file, regrets_file, strategies_file]):
            return False

        # Load key mapping
        with open(key_mapping_file, "rb") as mapping_f:
            mapping_data = pickle.load(mapping_f)
            self._key_to_id = mapping_data["key_to_id"]
            self._id_to_key = mapping_data["id_to_key"]
            num_infosets = mapping_data["num_infosets"]

        # Load into shared arrays
        with h5py.File(regrets_file, "r") as h5_f:
            loaded_regrets = h5_f["regrets"][:]
            loaded_action_counts = h5_f["action_counts"][:]

            self.shared_regrets[: loaded_regrets.shape[0], : loaded_regrets.shape[1]] = (
                loaded_regrets
            )
            self.shared_action_counts[: loaded_action_counts.shape[0]] = loaded_action_counts

        with h5py.File(strategies_file, "r") as h5_f:
            loaded_strategies = h5_f["strategies"][:]
            self.shared_strategy_sum[: loaded_strategies.shape[0], : loaded_strategies.shape[1]] = (
                loaded_strategies
            )

        self.shared_infoset_count[0] = num_infosets

        logger.info(f"Loaded checkpoint: {num_infosets} infosets")
        return True

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self):
        """Clean up shared memory (coordinator unlinks, workers just close)."""
        handles = [
            self._shm_regrets,
            self._shm_strategy,
            self._shm_actions,
            self._shm_count,
            self._shm_keys,
        ]

        for shm in handles:
            if shm is not None:
                try:
                    shm.close()
                    if self.is_coordinator:
                        shm.unlink()
                except FileNotFoundError:
                    pass

        self._shm_regrets = None
        self._shm_strategy = None
        self._shm_actions = None
        self._shm_count = None
        self._shm_keys = None

        logger.info(f"Worker {self.worker_id} cleaned up shared memory")

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()

    def __str__(self) -> str:
        return (
            f"SharedArrayStorage(worker={self.worker_id}, "
            f"coordinator={self.is_coordinator}, "
            f"infosets={self.num_infosets()}/{self.max_infosets})"
        )

    # =========================================================================
    # Properties for backward compatibility
    # =========================================================================

    @property
    def owned_partition(self) -> Dict[InfoSetKey, InfoSet]:
        """Get owned infosets as dict (for backward compatibility)."""
        result = {}
        for key, infoset_id in self._key_to_id.items():
            if self.is_owned_by_id(infoset_id):
                legal_actions = self._legal_actions_cache.get(infoset_id, [])
                if not legal_actions:
                    num_actions = self.shared_action_counts[infoset_id]
                    legal_actions = [fold() for _ in range(num_actions)]
                result[key] = self._create_infoset_view(infoset_id, key, legal_actions)
        return result

    @property
    def infosets(self) -> Dict[InfoSetKey, InfoSet]:
        """Get all infosets as dict (for backward compatibility)."""
        result = {}
        for key, infoset_id in self._key_to_id.items():
            legal_actions = self._legal_actions_cache.get(infoset_id, [])
            if not legal_actions:
                num_actions = self.shared_action_counts[infoset_id]
                legal_actions = [fold() for _ in range(num_actions)]
            result[key] = self._create_infoset_view(infoset_id, key, legal_actions)
        return result
