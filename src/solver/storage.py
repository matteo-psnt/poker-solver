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
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import xxhash

from src.bucketing.utils.infoset import InfoSet, InfoSetKey
from src.game.actions import Action, fold

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

# Setup logger
logger = logging.getLogger(__name__)


class Storage(ABC):
    """Abstract base class for infoset storage."""

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
        """
        return True

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

        self._infosets_by_id: Dict[int, InfoSet] = {}
        self.key_to_id: Dict[InfoSetKey, int] = {}
        self.id_to_key: Dict[int, InfoSetKey] = {}
        self.next_id = 0

        if self.checkpoint_dir and self.checkpoint_dir.exists():
            self._load_from_checkpoint()

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """Get existing infoset or create new one."""
        infoset_id = self.key_to_id.get(key)
        if infoset_id is not None:
            return self._infosets_by_id[infoset_id]

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
        """Get all infosets as a dict keyed by InfoSetKey."""
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
        """Save current state to disk checkpoint."""
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
        """Load infosets from disk checkpoint."""
        if not self.checkpoint_dir:
            return
        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        regrets_file = self.checkpoint_dir / "regrets.h5"
        strategies_file = self.checkpoint_dir / "strategies.h5"

        if not all(f.exists() for f in [key_mapping_file, regrets_file, strategies_file]):
            return

        with open(key_mapping_file, "rb") as f:
            mapping_data = pickle.load(f)

            # Support both InMemoryStorage and SharedArrayStorage checkpoint formats
            if "key_to_id" in mapping_data:
                # InMemoryStorage format
                self.key_to_id = mapping_data["key_to_id"]
                self.id_to_key = mapping_data["id_to_key"]
                self.next_id = mapping_data["next_id"]
            elif "owned_keys" in mapping_data:
                # SharedArrayStorage format (parallel training)
                self.key_to_id = mapping_data["owned_keys"]
                self.id_to_key = {v: k for k, v in self.key_to_id.items()}
                self.next_id = mapping_data.get("max_id", len(self.key_to_id))
            else:
                raise ValueError("Invalid checkpoint format: missing key mappings")

        with (
            h5py.File(regrets_file, "r") as regrets_h5,
            h5py.File(strategies_file, "r") as strategies_h5,
        ):
            all_regrets = regrets_h5["regrets"][:]
            action_counts = regrets_h5["action_counts"][:]
            all_strategies = strategies_h5["strategies"][:]

            for infoset_id, key in self.id_to_key.items():
                n_actions = action_counts[infoset_id]
                regrets = all_regrets[infoset_id, :n_actions].copy()
                strategies = all_strategies[infoset_id, :n_actions].copy()

                legal_actions = [fold() for _ in range(n_actions)]
                infoset = InfoSet(key, legal_actions)
                infoset.regrets = regrets
                infoset.strategy_sum = strategies

                self._infosets_by_id[infoset_id] = infoset

        logger.info(f"Loaded {len(self._infosets_by_id)} infosets from checkpoint")

    def _save_to_disk(self):
        """Save all infosets to disk using optimized matrix format."""
        if not self.checkpoint_dir:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        key_to_id_snapshot = dict(self.key_to_id)
        id_to_key_snapshot = dict(self.id_to_key)
        next_id_snapshot = self.next_id

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

        max_actions = max(len(item["regrets"]) for item in infoset_data)

        all_regrets = np.zeros((num_infosets, max_actions), dtype=np.float32)
        all_strategies = np.zeros((num_infosets, max_actions), dtype=np.float32)
        action_counts = np.zeros(num_infosets, dtype=np.int32)

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

    OWNERSHIP MODEL:
    - Ownership is determined by stable hash: owner(key) = xxhash(key) % num_workers
    - Only the owning worker may create key→ID mappings
    - Only the owning worker may write to an infoset's regrets/strategy
    - Non-owners get read-only views into shared memory

    ID ALLOCATION:
    - Each worker has an exclusive ID range: [worker_id * range_size, (worker_id+1) * range_size)
    - ID 0 is reserved as the "unknown" region for non-owner reads of undiscovered keys
    - No shared counter, no races

    DATA LAYOUT:
    - shared_regrets: float32[max_infosets, max_actions] - live data
    - shared_strategy_sum: float32[max_infosets, max_actions] - live data
    - shared_action_counts: int32[max_infosets] - number of actions per infoset

    CONSISTENCY:
    - Reads are lock-free and may be stale (acceptable for MCCFR)
    - Writes are owner-only (no locks needed)
    - Cross-partition updates are buffered and routed to owners
    """

    # Shared memory names (macOS limit: ~30 chars)
    SHM_REGRETS = "sas_reg"
    SHM_STRATEGY = "sas_str"
    SHM_ACTIONS = "sas_act"

    # Reserved ID for unknown infosets (non-owners reading undiscovered keys)
    UNKNOWN_ID = 0

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

        # =====================================================================
        # Per-worker ID allocation (race-free)
        # ID 0 is reserved for "unknown" region, so usable range starts at 1
        # =====================================================================
        usable_slots = max_infosets - 1  # Reserve slot 0
        slots_per_worker = usable_slots // num_workers
        self.id_range_start = 1 + worker_id * slots_per_worker
        self.id_range_end = 1 + (worker_id + 1) * slots_per_worker
        self.next_local_id = self.id_range_start

        # =====================================================================
        # Key→ID mappings (owner-local, no global sync)
        # =====================================================================
        # Keys we own: authoritative mapping
        self._owned_keys: Dict[InfoSetKey, int] = {}
        # Keys we've learned about from ID requests (cached, read-only)
        self._remote_keys: Dict[InfoSetKey, int] = {}

        # Legal actions cache (can't store complex objects in shm)
        self._legal_actions_cache: Dict[int, List[Action]] = {}

        # Pending ID requests to send to owners (batched)
        self._pending_id_requests: Dict[int, Set[InfoSetKey]] = {
            i: set() for i in range(num_workers)
        }

        # Cross-partition update buffers: {infoset_id: (regret_delta, strategy_delta)}
        self._pending_updates: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Shared memory handles
        self._shm_regrets: Optional["SharedMemory"] = None
        self._shm_strategy: Optional["SharedMemory"] = None
        self._shm_actions: Optional["SharedMemory"] = None

        # NumPy views into shared memory (the live data)
        self.shared_regrets: Optional[np.ndarray] = None
        self.shared_strategy_sum: Optional[np.ndarray] = None
        self.shared_action_counts: Optional[np.ndarray] = None

        # Initialize shared memory
        if is_coordinator:
            self._create_shared_memory()
        else:
            self._attach_shared_memory()

        logger.info(
            f"SharedArrayStorage initialized: worker={worker_id}, "
            f"coordinator={is_coordinator}, id_range=[{self.id_range_start}, {self.id_range_end})"
        )

    # =========================================================================
    # Stable Hashing (replaces Python hash())
    # =========================================================================

    def _stable_hash(self, key: InfoSetKey) -> int:
        """
        Compute stable hash of InfoSetKey using xxhash.

        Python's built-in hash() is randomized per process, which breaks
        ownership consistency across workers. xxhash provides a stable,
        fast, cross-process hash.

        Args:
            key: InfoSetKey to hash

        Returns:
            64-bit integer hash
        """
        # Build deterministic byte representation (no pickle randomization)
        parts = [
            str(key.player_position).encode(),
            key.street.name.encode(),
            key.betting_sequence.encode(),
            (key.preflop_hand or "").encode(),
            str(key.postflop_bucket if key.postflop_bucket is not None else -1).encode(),
            str(key.spr_bucket).encode(),
        ]
        key_bytes = b"|".join(parts)
        return xxhash.xxh64(key_bytes).intdigest()

    def get_owner(self, key: InfoSetKey) -> int:
        """
        Determine which worker owns this key.

        Ownership is determined by stable hash of the key.
        This is consistent across all processes.

        Args:
            key: InfoSetKey to check

        Returns:
            Worker ID that owns this key
        """
        return self._stable_hash(key) % self.num_workers

    def is_owned(self, key: InfoSetKey) -> bool:
        """Check if this worker owns the given infoset key."""
        return self.get_owner(key) == self.worker_id

    def is_owned_by_id(self, infoset_id: int) -> bool:
        """
        Check if this worker owns the given infoset ID.

        Ownership by ID is determined by which worker's range contains the ID.
        """
        if infoset_id == self.UNKNOWN_ID:
            return False  # Reserved region has no owner
        return self.id_range_start <= infoset_id < self.id_range_end

    # =========================================================================
    # Shared Memory Management
    # =========================================================================

    def _get_shm_name(self, base: str) -> str:
        """Get session-namespaced shared memory name."""
        return f"{base}_{self.session_id}"

    def _create_shared_memory(self):
        """Create all shared memory segments (coordinator only)."""
        self._cleanup_stale_shm()

        # Calculate sizes
        regrets_size = self.max_infosets * self.max_actions * 4  # float32
        strategy_size = self.max_infosets * self.max_actions * 4  # float32
        actions_size = self.max_infosets * 4  # int32

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

        # Create NumPy views
        self._create_numpy_views()

        # Initialize to zero (including reserved UNKNOWN_ID slot)
        self.shared_regrets.fill(0)
        self.shared_strategy_sum.fill(0)
        self.shared_action_counts.fill(0)

        logger.info(
            f"Coordinator created shared memory: "
            f"regrets={regrets_size // 1024 // 1024}MB, "
            f"strategy={strategy_size // 1024 // 1024}MB"
        )

    def _attach_shared_memory(self):
        """Attach to existing shared memory segments (worker)."""
        import time

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

                self._create_numpy_views()
                logger.info(f"Worker {self.worker_id} attached to shared memory")
                return

            except FileNotFoundError:
                if attempt < max_retries - 1:
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

    def _cleanup_stale_shm(self):
        """Clean up stale shared memory from previous runs."""
        shm_names = [
            self._get_shm_name(self.SHM_REGRETS),
            self._get_shm_name(self.SHM_STRATEGY),
            self._get_shm_name(self.SHM_ACTIONS),
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

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """
        Get existing infoset or create new one.

        OWNERSHIP RULES:
        - Only the owner may create the key→ID mapping
        - Only the owner may write to the infoset's arrays
        - Non-owners receive a view into shared memory (may be UNKNOWN_ID if key undiscovered)

        Args:
            key: InfoSetKey to look up or create
            legal_actions: Legal actions at this infoset

        Returns:
            InfoSet with arrays backed by shared memory
        """
        owner = self.get_owner(key)

        if owner == self.worker_id:
            # =================================================================
            # OWNER PATH: We own this key
            # =================================================================
            if key in self._owned_keys:
                # Known key - return view
                infoset_id = self._owned_keys[key]
            else:
                # New key - allocate ID from our exclusive range
                infoset_id = self._allocate_id()
                self._owned_keys[key] = infoset_id
                self.shared_action_counts[infoset_id] = len(legal_actions)
                self._legal_actions_cache[infoset_id] = legal_actions

            return self._create_infoset_view(infoset_id, key, legal_actions)

        else:
            # =================================================================
            # NON-OWNER PATH: Another worker owns this key
            # =================================================================
            if key in self._remote_keys:
                # We've learned this key's ID (from previous batch sync)
                infoset_id = self._remote_keys[key]
                return self._create_infoset_view(infoset_id, key, legal_actions)
            else:
                # Unknown key - buffer request, return view into UNKNOWN_ID region
                # The UNKNOWN_ID region contains zeros (uniform strategy)
                # This is acceptable: non-owners don't update, staleness is OK
                self._pending_id_requests[owner].add(key)
                return self._create_infoset_view(self.UNKNOWN_ID, key, legal_actions)

    def _allocate_id(self) -> int:
        """
        Allocate ID from this worker's exclusive range (race-free).

        Each worker has a non-overlapping range:
        - Worker 0: [1, 1 + slots_per_worker)
        - Worker 1: [1 + slots_per_worker, 1 + 2*slots_per_worker)
        - etc.

        This eliminates all races - no shared counter needed.

        Returns:
            Allocated infoset ID

        Raises:
            RuntimeError if worker's ID range is exhausted
        """
        if self.next_local_id >= self.id_range_end:
            raise RuntimeError(
                f"Worker {self.worker_id} exhausted ID range "
                f"[{self.id_range_start}, {self.id_range_end}). "
                f"Increase max_infosets or reduce worker count."
            )
        infoset_id = self.next_local_id
        self.next_local_id += 1
        return infoset_id

    def _create_infoset_view(
        self, infoset_id: int, key: InfoSetKey, legal_actions: List[Action]
    ) -> InfoSet:
        """
        Create an InfoSet with arrays viewing shared memory.

        The returned InfoSet has regrets and strategy_sum pointing directly
        into the shared memory arrays. Modifications by the owner are
        immediately visible to all workers (lock-free reads).

        Args:
            infoset_id: ID in shared arrays (or UNKNOWN_ID for unknown keys)
            key: InfoSetKey
            legal_actions: Legal actions at this infoset

        Returns:
            InfoSet backed by shared memory (read-only if UNKNOWN_ID)
        """
        num_actions = len(legal_actions)
        infoset = InfoSet(key, legal_actions)

        # Point arrays to shared memory (sliced view)
        regrets_view = self.shared_regrets[infoset_id, :num_actions]
        strategy_view = self.shared_strategy_sum[infoset_id, :num_actions]

        # CRITICAL SAFETY: Make UNKNOWN_ID views read-only to prevent
        # accidental writes corrupting the placeholder region.
        # Any attempt to write will raise ValueError.
        if infoset_id == self.UNKNOWN_ID:
            regrets_view = regrets_view.copy()  # Copy to avoid modifying shared flag
            regrets_view.setflags(write=False)
            strategy_view = strategy_view.copy()
            strategy_view.setflags(write=False)

        infoset.regrets = regrets_view
        infoset.strategy_sum = strategy_view

        return infoset

    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        """Get existing infoset or None."""
        owner = self.get_owner(key)

        if owner == self.worker_id:
            infoset_id = self._owned_keys.get(key)
        else:
            infoset_id = self._remote_keys.get(key)

        if infoset_id is None:
            return None

        legal_actions = self._legal_actions_cache.get(infoset_id)
        if legal_actions is None:
            num_actions = self.shared_action_counts[infoset_id]
            legal_actions = [fold() for _ in range(num_actions)]

        return self._create_infoset_view(infoset_id, key, legal_actions)

    def has_infoset(self, key: InfoSetKey) -> bool:
        """Check if infoset exists (in owned or remote cache)."""
        owner = self.get_owner(key)
        if owner == self.worker_id:
            return key in self._owned_keys
        else:
            return key in self._remote_keys

    def num_infosets(self) -> int:
        """Get total number of infosets allocated by this worker."""
        return self.next_local_id - self.id_range_start

    def num_owned_infosets(self) -> int:
        """Get number of infosets owned by this worker."""
        return len(self._owned_keys)

    def mark_dirty(self, key: InfoSetKey):
        """No-op for shared array storage (writes go directly to shared memory)."""
        pass

    def flush(self):
        """No-op for shared array storage (data is always in shared memory)."""
        pass

    # =========================================================================
    # ID Request/Response (Batched, Async)
    # =========================================================================

    def get_pending_id_requests(self) -> Dict[int, Set[InfoSetKey]]:
        """
        Get pending ID requests to send to owners.

        Returns:
            Dict mapping owner_id → set of keys to request
        """
        return self._pending_id_requests

    def clear_pending_id_requests(self):
        """Clear pending ID requests after they've been sent."""
        for owner_id in self._pending_id_requests:
            self._pending_id_requests[owner_id].clear()

    def respond_to_id_requests(self, requested_keys: Set[InfoSetKey]) -> Dict[InfoSetKey, int]:
        """
        Respond to ID requests from other workers.

        Called by owner to provide IDs for keys they own.

        Args:
            requested_keys: Keys that other workers are asking about

        Returns:
            Dict mapping key → infoset_id for keys we own and have allocated
        """
        responses = {}
        for key in requested_keys:
            if key in self._owned_keys:
                responses[key] = self._owned_keys[key]
        return responses

    def receive_id_responses(self, responses: Dict[InfoSetKey, int]):
        """
        Receive ID responses from owners.

        Updates remote key cache with learned IDs.

        Args:
            responses: Dict mapping key → infoset_id from owner
        """
        self._remote_keys.update(responses)

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
            if num_actions > 0:
                self.shared_regrets[infoset_id, :num_actions] += regret_delta[:num_actions]
                self.shared_strategy_sum[infoset_id, :num_actions] += strategy_delta[:num_actions]

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def checkpoint(self, iteration: int):
        """
        Save checkpoint to disk (coordinator only).

        After key collection from workers, _owned_keys contains ALL keys
        from all workers, spanning multiple ID ranges. We save:
        1. Complete key→ID mapping
        2. All used rows from shared arrays (determined by max ID)
        """
        if not self.checkpoint_dir or not self.is_coordinator:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # After key collection, _owned_keys contains all keys from all workers
        num_keys = len(self._owned_keys)
        if num_keys == 0:
            return

        # Find the range of IDs that need to be saved
        # After key collection, IDs span multiple worker ranges
        all_ids = list(self._owned_keys.values())
        max_id = max(all_ids) + 1  # +1 because we slice [:max_id]

        # Save key mapping
        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        with open(key_mapping_file, "wb") as f:
            pickle.dump(
                {
                    "owned_keys": dict(self._owned_keys),
                    "num_workers": self.num_workers,
                    "max_id": max_id,
                },
                f,
            )

        # Save shared arrays (all used rows up to max_id)
        regrets_file = self.checkpoint_dir / "regrets.h5"
        strategies_file = self.checkpoint_dir / "strategies.h5"

        with h5py.File(regrets_file, "w") as f:
            f.create_dataset(
                "regrets",
                data=self.shared_regrets[:max_id],
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "action_counts",
                data=self.shared_action_counts[:max_id],
            )

        with h5py.File(strategies_file, "w") as f:
            f.create_dataset(
                "strategies",
                data=self.shared_strategy_sum[:max_id],
                compression="gzip",
                compression_opts=4,
            )

        logger.info(f"Checkpoint saved: {num_keys} infosets at iteration {iteration}")

    def load_checkpoint(self) -> bool:
        """Load checkpoint from disk (coordinator only)."""
        if not self.checkpoint_dir or not self.is_coordinator:
            return False

        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        regrets_file = self.checkpoint_dir / "regrets.h5"
        strategies_file = self.checkpoint_dir / "strategies.h5"

        if not all(f.exists() for f in [key_mapping_file, regrets_file, strategies_file]):
            return False

        with open(key_mapping_file, "rb") as mapping_f:
            mapping_data = pickle.load(mapping_f)
            self._owned_keys = mapping_data["owned_keys"]
            saved_start = mapping_data["id_range_start"]
            self.next_local_id = mapping_data["next_local_id"]

        with h5py.File(regrets_file, "r") as h5_f:
            loaded_regrets = h5_f["regrets"][:]
            loaded_action_counts = h5_f["action_counts"][:]

            num_loaded = loaded_regrets.shape[0]
            self.shared_regrets[saved_start : saved_start + num_loaded] = loaded_regrets
            self.shared_action_counts[saved_start : saved_start + num_loaded] = loaded_action_counts

        with h5py.File(strategies_file, "r") as h5_f:
            loaded_strategies = h5_f["strategies"][:]
            num_loaded = loaded_strategies.shape[0]
            self.shared_strategy_sum[saved_start : saved_start + num_loaded] = loaded_strategies

        logger.info(f"Loaded checkpoint: {len(self._owned_keys)} owned infosets")
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

        logger.info(f"Worker {self.worker_id} cleaned up shared memory")

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()

    def __str__(self) -> str:
        return (
            f"SharedArrayStorage(worker={self.worker_id}, "
            f"coordinator={self.is_coordinator}, "
            f"owned={self.num_owned_infosets()}, "
            f"id_range=[{self.id_range_start}, {self.id_range_end}))"
        )

    # =========================================================================
    # Properties for backward compatibility
    # =========================================================================

    @property
    def owned_partition(self) -> Dict[InfoSetKey, InfoSet]:
        """Get owned infosets as dict (for backward compatibility)."""
        result = {}
        for key, infoset_id in self._owned_keys.items():
            legal_actions = self._legal_actions_cache.get(infoset_id, [])
            if not legal_actions:
                num_actions = self.shared_action_counts[infoset_id]
                legal_actions = [fold() for _ in range(num_actions)]
            result[key] = self._create_infoset_view(infoset_id, key, legal_actions)
        return result

    @property
    def infosets(self) -> Dict[InfoSetKey, InfoSet]:
        """Get all known infosets as dict (owned + remote)."""
        result = {}
        # Owned keys
        for key, infoset_id in self._owned_keys.items():
            legal_actions = self._legal_actions_cache.get(infoset_id, [])
            if not legal_actions:
                num_actions = self.shared_action_counts[infoset_id]
                legal_actions = [fold() for _ in range(num_actions)]
            result[key] = self._create_infoset_view(infoset_id, key, legal_actions)
        # Remote keys
        for key, infoset_id in self._remote_keys.items():
            legal_actions = self._legal_actions_cache.get(infoset_id, [])
            if not legal_actions:
                num_actions = self.shared_action_counts[infoset_id]
                legal_actions = [fold() for _ in range(num_actions)]
            result[key] = self._create_infoset_view(infoset_id, key, legal_actions)
        return result
