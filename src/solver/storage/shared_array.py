import pickle
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING

import numcodecs
import numpy as np
import xxhash
import zarr

from src.bucketing.utils.infoset import InfoSet, InfoSetKey
from src.game.actions import Action, fold
from src.solver.storage.base import Storage
from src.solver.storage.helpers import (
    CheckpointPaths,
    _validate_action_signatures,
    build_legal_actions,
    get_missing_checkpoint_files,
    load_action_signatures,
    load_checkpoint_data,
    load_key_mapping,
)

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory
    from multiprocessing.synchronize import Event as EventType


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
    - shared_regrets: float32[initial_capacity, max_actions] - live data
    - shared_strategy_sum: float64[initial_capacity, max_actions] - live data
    - shared_action_counts: int32[initial_capacity] - number of actions per infoset

    CONSISTENCY:
    - Reads are lock-free and may be stale (acceptable for MCCFR)
    - Writes are owner-only (no locks needed)
    - Cross-partition updates are buffered and routed to owners

    DYNAMIC RESIZING:
    - Storage automatically expands when reaching capacity threshold (85%)
    - Growth factor of 2x ensures amortized O(1) allocations
    - Stop-the-world resize: coordinator pauses workers, creates new arrays, copies data
    - Resized capacity persists across checkpoints
    """

    # Shared memory names (macOS limit: ~30 chars)
    SHM_REGRETS = "sas_reg"
    SHM_STRATEGY = "sas_str"
    SHM_ACTIONS = "sas_act"
    SHM_REACH = "sas_reach"
    SHM_UTILITY = "sas_util"

    # Reserved ID for unknown infosets (non-owners reading undiscovered keys)
    UNKNOWN_ID = 0

    # Dynamic resizing constants
    CAPACITY_THRESHOLD = 0.85  # Trigger resize at 85% capacity
    GROWTH_FACTOR = 2.0  # Double capacity on resize

    def __init__(
        self,
        num_workers: int,
        worker_id: int,
        session_id: str,
        initial_capacity: int = 2_000_000,
        max_actions: int = 10,
        is_coordinator: bool = False,
        checkpoint_dir: Path | None = None,
        ready_event: "EventType | None" = None,
        load_checkpoint_on_init: bool = True,
        zarr_compression_level: int = 3,
        zarr_chunk_size: int = 10_000,
    ):
        """
        Initialize shared array storage.

        Args:
            num_workers: Total number of workers (for partition ownership)
            worker_id: This worker's ID (0 to num_workers-1)
            session_id: Unique session ID for shared memory namespace
            initial_capacity: Infoset capacity (grows automatically via resize)
            max_actions: Maximum actions per infoset (pre-allocated)
            is_coordinator: If True, creates shared memory. If False, attaches.
            checkpoint_dir: Optional directory for checkpoints
            ready_event: Optional multiprocessing.Event for synchronization.
                        If provided:
                        - Coordinator sets it after creating shared memory
                        - Workers wait for it before attempting to attach
                        This eliminates race conditions during parallel startup.
            load_checkpoint_on_init: If True, load checkpoint data during initialization.
            zarr_compression_level: ZStd compression level for Zarr checkpoints (1-9, default 3)
            zarr_chunk_size: Number of infosets per chunk in Zarr format (default 10000)
        """
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.session_id = session_id[:8]  # Truncate for macOS shm name limit
        self.capacity = initial_capacity  # Current capacity (grows via resize)
        self.max_actions = max_actions
        self.is_coordinator = is_coordinator
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.ready_event = ready_event
        self.base_capacity = initial_capacity  # Original base capacity
        self.base_slots_per_worker = (initial_capacity - 1) // num_workers
        self._extra_regions: list[tuple[int, int, int, int]] = []
        self._extra_allocations: list[dict[str, int]] = []

        # Zarr checkpoint configuration
        self.zarr_compression_level = zarr_compression_level
        self.zarr_chunk_size = zarr_chunk_size

        # =====================================================================
        # Per-worker ID allocation (race-free)
        # ID 0 is reserved for "unknown" region, so usable range starts at 1
        # =====================================================================
        usable_slots = initial_capacity - 1  # Reserve slot 0
        slots_per_worker = usable_slots // num_workers
        self.id_range_start = 1 + worker_id * slots_per_worker
        self.id_range_end = 1 + (worker_id + 1) * slots_per_worker
        self.next_local_id = self.id_range_start

        # =====================================================================
        # Key→ID mappings (owner-local, no global sync)
        # =====================================================================
        # Keys we own: authoritative mapping
        self._owned_keys: dict[InfoSetKey, int] = {}
        # Keys we've learned about from ID requests (cached, read-only)
        self._remote_keys: dict[InfoSetKey, int] = {}

        # Legal actions cache (can't store complex objects in shm)
        self._legal_actions_cache: dict[int, list[Action]] = {}

        # Pending ID requests to send to owners (batched)
        self._pending_id_requests: dict[int, set[InfoSetKey]] = {
            i: set() for i in range(num_workers)
        }

        # Cross-partition update buffers: {infoset_id: (regret_delta, strategy_delta)}
        self._pending_updates: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Shared memory handles
        self._shm_regrets: "SharedMemory | None" = None
        self._shm_strategy: "SharedMemory | None" = None
        self._shm_actions: "SharedMemory | None" = None
        self._shm_reach: "SharedMemory | None" = None
        self._shm_utility: "SharedMemory | None" = None

        # NumPy views into shared memory (the live data)
        self.shared_regrets = np.empty((0, self.max_actions), dtype=np.float64)
        self.shared_strategy_sum = np.empty((0, self.max_actions), dtype=np.float64)
        self.shared_action_counts = np.empty((0,), dtype=np.int32)
        self.shared_reach_counts = np.empty((0,), dtype=np.int64)
        self.shared_cumulative_utility = np.empty((0,), dtype=np.float64)

        # Initialize shared memory
        if is_coordinator:
            self._create_shared_memory()
        else:
            self._attach_shared_memory()

        # Load checkpoint if available (each worker loads its owned keys)
        if checkpoint_dir and load_checkpoint_on_init:
            self.load_checkpoint()

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
        if self.id_range_start <= infoset_id < self.id_range_end:
            return True
        return any(alloc["start"] <= infoset_id < alloc["end"] for alloc in self._extra_allocations)

    def get_owner_by_id(self, infoset_id: int) -> int | None:
        """
        Determine owner worker for a given infoset ID.

        Uses initial base ranges plus any appended resize regions.
        """
        if infoset_id == self.UNKNOWN_ID:
            return None

        # Base ranges (fixed at initialization)
        base_end = 1 + self.base_slots_per_worker * self.num_workers
        if 1 <= infoset_id < base_end:
            return (infoset_id - 1) // self.base_slots_per_worker

        # Extra regions (appended on resize)
        for extra_start, extra_total, base, remainder in self._extra_regions:
            extra_end = extra_start + extra_total
            if extra_start <= infoset_id < extra_end:
                offset = infoset_id - extra_start
                # Distribute remainder to lower worker IDs
                if base == 0:
                    return offset if offset < remainder else None
                threshold = (base + 1) * remainder
                if offset < threshold:
                    return offset // (base + 1)
                return remainder + (offset - threshold) // base

        return None

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
        regrets_size = self.capacity * self.max_actions * np.dtype(np.float64).itemsize
        strategy_size = self.capacity * self.max_actions * np.dtype(np.float64).itemsize
        actions_size = self.capacity * np.dtype(np.int32).itemsize
        reach_size = self.capacity * np.dtype(np.int64).itemsize
        utility_size = self.capacity * np.dtype(np.float64).itemsize

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
        self._shm_reach = shared_memory.SharedMemory(
            create=True,
            size=reach_size,
            name=self._get_shm_name(self.SHM_REACH),
        )
        self._shm_utility = shared_memory.SharedMemory(
            create=True,
            size=utility_size,
            name=self._get_shm_name(self.SHM_UTILITY),
        )

        # Create NumPy views
        self._create_numpy_views()

        # Initialize to zero (including reserved UNKNOWN_ID slot)
        self.shared_regrets.fill(0)
        self.shared_strategy_sum.fill(0)
        self.shared_action_counts.fill(0)
        self.shared_reach_counts.fill(0)
        self.shared_cumulative_utility.fill(0)

        print(
            "Master created shared memory: "
            f"regrets={regrets_size // 1024 // 1024}MB, "
            f"strategy={strategy_size // 1024 // 1024}MB"
        )

        # Signal workers that shared memory is ready
        if self.ready_event is not None:
            self.ready_event.set()

    def _attach_shared_memory(self):
        """
        Attach to existing shared memory segments (worker).

        If ready_event is provided, waits for it before attempting to attach,
        eliminating race conditions. Otherwise falls back to retry-with-sleep.
        """
        # Wait for coordinator signal if event is provided (preferred approach)
        if self.ready_event is not None:
            wait_timeout = 30.0  # seconds
            if not self.ready_event.wait(timeout=wait_timeout):
                raise RuntimeError(
                    f"Worker {self.worker_id} timed out waiting for coordinator "
                    f"to create shared memory (waited {wait_timeout}s)"
                )

        # Attach to shared memory (minimal retry for OS propagation delay)
        max_retries = 5
        retry_delay = 0.1

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
                self._shm_reach = shared_memory.SharedMemory(
                    name=self._get_shm_name(self.SHM_REACH)
                )
                self._shm_utility = shared_memory.SharedMemory(
                    name=self._get_shm_name(self.SHM_UTILITY)
                )

                self._create_numpy_views()
                return

            except FileNotFoundError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Worker {self.worker_id} failed to attach to shared memory "
                        f"after {max_retries} attempts. "
                        f"Ensure coordinator creates memory before workers start."
                    )

    def _create_numpy_views(self):
        """Create NumPy array views into shared memory."""
        if (
            self._shm_regrets is None
            or self._shm_strategy is None
            or self._shm_actions is None
            or self._shm_reach is None
            or self._shm_utility is None
        ):
            raise RuntimeError("Shared memory buffers are not initialized")

        self.shared_regrets = np.ndarray(
            (self.capacity, self.max_actions),
            dtype=np.float64,
            buffer=self._shm_regrets.buf,
        )
        self.shared_strategy_sum = np.ndarray(
            (self.capacity, self.max_actions),
            dtype=np.float64,
            buffer=self._shm_strategy.buf,
        )
        self.shared_action_counts = np.ndarray(
            (self.capacity,),
            dtype=np.int32,
            buffer=self._shm_actions.buf,
        )
        self.shared_reach_counts = np.ndarray(
            (self.capacity,),
            dtype=np.int64,
            buffer=self._shm_reach.buf,
        )
        self.shared_cumulative_utility = np.ndarray(
            (self.capacity,),
            dtype=np.float64,
            buffer=self._shm_utility.buf,
        )

    def _cleanup_stale_shm(self):
        """Clean up stale shared memory from previous runs."""
        shm_names = [
            self._get_shm_name(self.SHM_REGRETS),
            self._get_shm_name(self.SHM_STRATEGY),
            self._get_shm_name(self.SHM_ACTIONS),
            self._get_shm_name(self.SHM_REACH),
            self._get_shm_name(self.SHM_UTILITY),
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

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: list[Action]) -> InfoSet:
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
            RuntimeError if worker's ID range is exhausted and resize not possible
        """
        if self.next_local_id < self.id_range_end:
            infoset_id = self.next_local_id
            self.next_local_id += 1
            return infoset_id

        for alloc in self._extra_allocations:
            if alloc["next"] < alloc["end"]:
                infoset_id = alloc["next"]
                alloc["next"] += 1
                return infoset_id

        raise RuntimeError(
            f"Worker {self.worker_id} exhausted ID ranges "
            f"[{self.id_range_start}, {self.id_range_end}) and extras. "
            f"Storage resize required - coordinator should trigger resize."
        )

    # =========================================================================
    # Capacity Monitoring and Dynamic Resizing
    # =========================================================================

    def get_capacity_usage(self) -> float:
        """
        Get fraction of this worker's ID range that is used.

        Returns:
            Float between 0.0 and 1.0 representing capacity usage
        """
        base_size = self.id_range_end - self.id_range_start
        extra_size = sum(alloc["end"] - alloc["start"] for alloc in self._extra_allocations)
        total_size = base_size + extra_size
        if total_size == 0:
            return 1.0
        base_used = max(0, min(self.next_local_id, self.id_range_end) - self.id_range_start)
        extra_used = sum(alloc["next"] - alloc["start"] for alloc in self._extra_allocations)
        return (base_used + extra_used) / total_size

    def needs_resize(self) -> bool:
        """
        Check if storage needs to be resized.

        Returns:
            True if capacity usage >= CAPACITY_THRESHOLD (85%)
        """
        return self.get_capacity_usage() >= self.CAPACITY_THRESHOLD

    def get_resize_stats(self) -> dict[str, int | float]:
        """
        Get statistics for resize decision.

        Returns:
            Dict with current capacity info
        """
        range_size = self.id_range_end - self.id_range_start
        extra_size = sum(alloc["end"] - alloc["start"] for alloc in self._extra_allocations)
        used = max(0, min(self.next_local_id, self.id_range_end) - self.id_range_start)
        extra_used = sum(alloc["next"] - alloc["start"] for alloc in self._extra_allocations)
        return {
            "worker_id": self.worker_id,
            "id_range_start": self.id_range_start,
            "id_range_end": self.id_range_end,
            "next_local_id": self.next_local_id,
            "range_size": range_size,
            "extra_size": extra_size,
            "used": used + extra_used,
            "capacity_usage": self.get_capacity_usage(),
            "initial_capacity": self.capacity,
        }

    def resize(self, new_capacity: int) -> None:
        """
        Resize storage to new capacity (coordinator only).

        This is a stop-the-world operation:
        1. Create new larger shared memory segments
        2. Copy existing data to new arrays
        3. Update ID ranges for all workers
        4. Clean up old shared memory

        Args:
            new_capacity: New infoset capacity

        Raises:
            RuntimeError: If called by non-coordinator or new size is smaller
        """
        if not self.is_coordinator:
            raise RuntimeError("Only coordinator can resize storage")

        if new_capacity <= self.capacity:
            raise RuntimeError(
                f"New size {new_capacity} must be larger than current {self.capacity}"
            )

        old_capacity = self.capacity
        old_regrets = self.shared_regrets
        old_strategy = self.shared_strategy_sum
        old_action_counts = self.shared_action_counts
        old_reach_counts = self.shared_reach_counts
        old_cumulative_utility = self.shared_cumulative_utility

        print(
            f"Resizing storage: {old_capacity:,} -> {new_capacity:,} infosets "
            f"(growth factor: {new_capacity / old_capacity:.1f}x)"
        )

        # Store old shared memory handles for cleanup
        old_shm_regrets = self._shm_regrets
        old_shm_strategy = self._shm_strategy
        old_shm_actions = self._shm_actions
        old_shm_reach = self._shm_reach
        old_shm_utility = self._shm_utility

        # Update initial_capacity before creating new shared memory
        old_capacity = self.capacity
        self.capacity = new_capacity

        # Create new session_id for resized shared memory
        # Append a counter to ensure unique names
        import uuid

        self.session_id = uuid.uuid4().hex[:8]

        # Create new shared memory segments
        self._create_shared_memory()

        # Copy existing data to new arrays
        # Note: Only copy up to old_capacity rows
        self.shared_regrets[:old_capacity, :] = old_regrets[:, :]
        self.shared_strategy_sum[:old_capacity, :] = old_strategy[:, :]
        self.shared_action_counts[:old_capacity] = old_action_counts[:]
        self.shared_reach_counts[:old_capacity] = old_reach_counts[:]
        self.shared_cumulative_utility[:old_capacity] = old_cumulative_utility[:]

        # Append a new extra region for allocations to avoid overlapping ranges.
        self._add_extra_region(old_capacity, new_capacity)

        # Clean up old shared memory
        try:
            for shm in (
                old_shm_regrets,
                old_shm_strategy,
                old_shm_actions,
                old_shm_reach,
                old_shm_utility,
            ):
                if shm is None:
                    continue
                shm.close()
                shm.unlink()
        except Exception as e:
            print(f"Warning: Error cleaning up old shared memory: {e}")

        print(f"Resize complete: new capacity {new_capacity:,}, new session_id={self.session_id}")

    def reattach_after_resize(
        self,
        new_session_id: str,
        new_capacity: int,
        preserved_keys: dict["InfoSetKey", int],
        preserved_next_id: int,
    ) -> None:
        """
        Reattach to resized shared memory (worker only).

        Called by workers after coordinator has resized storage.

        IMPORTANT: Data positions (IDs) are preserved during resize.
        The data is copied to the same positions in the new larger arrays.
        id_range_start stays the same, only id_range_end is extended.

        Args:
            new_session_id: New session ID for shared memory names
            new_capacity: New maximum infosets capacity
            preserved_keys: Worker's owned keys to preserve
            preserved_next_id: Worker's next_local_id to restore (absolute position)
        """
        # Close old shared memory handles
        if self._shm_regrets:
            self._shm_regrets.close()
        if self._shm_strategy:
            self._shm_strategy.close()
        if self._shm_actions:
            self._shm_actions.close()
        if self._shm_reach:
            self._shm_reach.close()
        if self._shm_utility:
            self._shm_utility.close()

        old_capacity = self.capacity

        # Update session and capacity
        self.session_id = new_session_id
        self.capacity = new_capacity

        # next_local_id stays at same absolute position - data is not moved
        self.next_local_id = preserved_next_id

        # Restore owned keys - IDs point to same absolute positions
        self._owned_keys = preserved_keys

        # Attach to new shared memory
        self._attach_shared_memory()

        # Append a new extra region for allocations
        self._add_extra_region(old_capacity, new_capacity)

        print(
            f"Worker {self.worker_id} reattached after resize: "
            f"session={new_session_id}, initial_capacity={new_capacity:,}, "
            f"id_range=[{self.id_range_start}, {self.id_range_end}), "
            f"next_id={self.next_local_id}"
        )

    def _create_infoset_view(
        self, infoset_id: int, key: InfoSetKey, legal_actions: list[Action]
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
        read_only_stats = False
        if infoset_id == self.UNKNOWN_ID:
            regrets_view = regrets_view.copy()  # Copy to avoid modifying shared flag
            regrets_view.setflags(write=False)
            strategy_view = strategy_view.copy()
            strategy_view.setflags(write=False)
            read_only_stats = True

        infoset.regrets = regrets_view
        infoset.strategy_sum = strategy_view
        infoset.attach_stats_views(
            self.shared_reach_counts,
            self.shared_cumulative_utility,
            infoset_id,
            read_only=read_only_stats,
        )
        infoset.sync_stats_to_storage(
            self.shared_reach_counts[infoset_id],
            self.shared_cumulative_utility[infoset_id],
        )

        return infoset

    def _add_extra_region(self, extra_start: int, extra_end: int) -> None:
        """
        Register a new resize region and allocate this worker's slice.
        """
        total = extra_end - extra_start
        if total <= 0:
            return

        base = total // self.num_workers
        remainder = total % self.num_workers
        self._extra_regions.append((extra_start, total, base, remainder))

        if base == 0 and self.worker_id >= remainder:
            return

        start = extra_start + self.worker_id * base + min(self.worker_id, remainder)
        end = start + base + (1 if self.worker_id < remainder else 0)
        if start >= end:
            return

        self._extra_allocations.append({"start": start, "end": end, "next": start})

    def get_infoset(self, key: InfoSetKey) -> InfoSet | None:
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

    def num_infosets(self) -> int:
        """Get total number of infosets allocated by this worker."""
        base_used = max(0, min(self.next_local_id, self.id_range_end) - self.id_range_start)
        extra_used = sum(alloc["next"] - alloc["start"] for alloc in self._extra_allocations)
        return base_used + extra_used

    def iter_infosets(self):
        for key, infoset_id in self._owned_keys.items():
            legal_actions = self._legal_actions_cache.get(infoset_id)
            if legal_actions is None:
                num_actions = self.shared_action_counts[infoset_id]
                legal_actions = [fold() for _ in range(num_actions)]
            yield self._create_infoset_view(infoset_id, key, legal_actions)

    def num_owned_infosets(self) -> int:
        """Get number of infosets owned by this worker."""
        return len(self._owned_keys)

    # =========================================================================
    # ID Request/Response (Batched, Async)
    # =========================================================================

    def get_pending_id_requests(self) -> dict[int, set[InfoSetKey]]:
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

    def respond_to_id_requests(self, requested_keys: set[InfoSetKey]) -> dict[InfoSetKey, int]:
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

    def receive_id_responses(self, responses: dict[InfoSetKey, int]):
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

    def get_pending_updates(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Get all pending cross-partition updates."""
        return self._pending_updates

    def clear_pending_updates(self):
        """Clear pending updates after they've been sent."""
        self._pending_updates.clear()

    def apply_updates(self, updates: dict[int, tuple[np.ndarray, np.ndarray]]):
        """
        Apply updates to infosets we own.

        Called by owner to apply buffered updates from other workers.

        Args:
            updates: {infoset_id: (regret_delta, strategy_delta)}
        """
        for infoset_id, (regret_delta, strategy_delta) in updates.items():
            if not self.is_owned_by_id(infoset_id):
                print(
                    f"Warning: Worker {self.worker_id} received update for non-owned infoset {infoset_id}"
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

        Uses Zarr format with ZStd compression for fast I/O and chunked storage.
        After key collection from workers, saves complete key→ID mapping and
        densified arrays to avoid sparse ID gaps.
        """
        if not self.checkpoint_dir or not self.is_coordinator:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # After key collection, _owned_keys contains all keys from all workers
        num_keys = len(self._owned_keys)
        if num_keys == 0:
            return

        items = sorted(self._owned_keys.items(), key=lambda item: item[1])
        dense_ids = {key: idx for idx, (key, _) in enumerate(items)}

        # Save key mapping
        paths = CheckpointPaths.from_dir(self.checkpoint_dir)
        with open(paths.key_mapping, "wb") as f:
            pickle.dump(
                {
                    "owned_keys": dense_ids,
                },
                f,
            )

        # Vectorized array extraction (much faster than row-by-row loop)
        # Extract old IDs in order, then use fancy indexing for bulk copy
        old_ids = np.array([old_id for (_, old_id) in items], dtype=np.int32)

        # Single vectorized operations instead of Python loop
        regrets_dense = self.shared_regrets[old_ids, :].copy()
        strategies_dense = self.shared_strategy_sum[old_ids, :].copy()
        action_counts_dense = self.shared_action_counts[old_ids].copy()
        reach_counts_dense = self.shared_reach_counts[old_ids].copy()
        cumulative_utility_dense = self.shared_cumulative_utility[old_ids].copy()

        # Use DirectoryStore for fast parallel I/O
        store = zarr.DirectoryStore(paths.checkpoint_zarr)
        root = zarr.open(store, mode="w")

        compressor = numcodecs.Blosc(
            cname="zstd",
            clevel=self.zarr_compression_level,
            shuffle=numcodecs.Blosc.BITSHUFFLE,
        )

        chunk_size = self.zarr_chunk_size

        root.create_dataset(
            "regrets",
            data=regrets_dense,
            chunks=(chunk_size, self.max_actions),
            compressor=compressor,
            dtype=np.float64,
        )
        root.create_dataset(
            "strategies",
            data=strategies_dense,
            chunks=(chunk_size, self.max_actions),
            compressor=compressor,
            dtype=np.float64,
        )
        root.create_dataset(
            "action_counts",
            data=action_counts_dense,
            chunks=(chunk_size,),
            compressor=compressor,
            dtype=np.int32,
        )
        root.create_dataset(
            "reach_counts",
            data=reach_counts_dense,
            chunks=(chunk_size,),
            compressor=compressor,
            dtype=np.int64,
        )
        root.create_dataset(
            "cumulative_utility",
            data=cumulative_utility_dense,
            chunks=(chunk_size,),
            compressor=compressor,
            dtype=np.float64,
        )

        # Store metadata
        root.attrs["iteration"] = iteration
        root.attrs["num_infosets"] = len(items)
        root.attrs["max_actions"] = self.max_actions
        root.attrs["timestamp"] = time.time()
        root.attrs["format_version"] = "1.0"

        # Save action signatures
        action_sigs = {}
        for new_id, (_, old_id) in enumerate(items):
            actions = self._legal_actions_cache.get(old_id)
            if actions is None:
                continue
            action_sigs[new_id] = [(action.type.name, action.amount) for action in actions]

        with open(paths.action_signatures, "wb") as f:
            pickle.dump(action_sigs, f)

        _validate_action_signatures(
            action_counts_dense,
            action_sigs,
            f"SharedArrayStorage.checkpoint(iter={iteration})",
        )

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint from disk.

        Optimized version:
        1. Pre-filter keys by ownership (vectorized hash computation)
        2. Load only the data we need (via array indexing)
        3. Bulk copy using NumPy fancy indexing

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.checkpoint_dir:
            return False

        missing_files = get_missing_checkpoint_files(self.checkpoint_dir)
        if missing_files:
            return False

        # Load metadata first (cheap)
        paths = CheckpointPaths.from_dir(self.checkpoint_dir)
        mapping_data = load_key_mapping(paths)
        saved_owned_keys = mapping_data["owned_keys"]
        saved_action_sigs = load_action_signatures(paths)

        if not saved_owned_keys:
            return True

        # Pre-filter keys by ownership (vectorized)
        # Build list of (key, old_id) for keys this worker owns
        my_keys = []
        my_old_ids = []
        for key, old_id in saved_owned_keys.items():
            if self.get_owner(key) == self.worker_id:
                my_keys.append(key)
                my_old_ids.append(old_id)

        if not my_keys:
            print(f"Worker {self.worker_id} owns 0/{len(saved_owned_keys)} keys from checkpoint")
            return True

        # Convert to numpy array for vectorized indexing
        my_old_ids_array = np.array(my_old_ids, dtype=np.int32)

        # Load checkpoint arrays (full load, but only once)
        data = load_checkpoint_data(
            self.checkpoint_dir, context="SharedArrayStorage.load_checkpoint"
        )

        if data.max_actions != self.max_actions:
            raise ValueError(
                f"Checkpoint max_actions mismatch: {data.max_actions} vs {self.max_actions}"
            )

        if self.capacity < data.max_id + 1:
            raise ValueError(f"Storage capacity too small: {self.capacity} vs {data.max_id + 1}")

        # Extract only the rows we need (vectorized fancy indexing)
        my_regrets = data.arrays["regrets"][my_old_ids_array, :]
        my_strategies = data.arrays["strategies"][my_old_ids_array, :]
        my_action_counts = data.arrays["action_counts"][my_old_ids_array]
        my_reach_counts = data.arrays["reach_counts"][my_old_ids_array]
        my_utility = data.arrays["cumulative_utility"][my_old_ids_array]

        # Allocate new IDs for all keys at once
        new_ids = []
        for key in my_keys:
            new_id = self._allocate_id()
            self._owned_keys[key] = new_id
            new_ids.append(new_id)

        new_ids_array = np.array(new_ids, dtype=np.int32)

        # Bulk copy using vectorized operations
        self.shared_action_counts[new_ids_array] = my_action_counts
        self.shared_reach_counts[new_ids_array] = my_reach_counts
        self.shared_cumulative_utility[new_ids_array] = my_utility

        # Bulk copy regrets/strategies (full width, mask handled by action_counts)
        # This is faster than element-by-element even if we copy some zeros
        self.shared_regrets[new_ids_array, :] = my_regrets
        self.shared_strategy_sum[new_ids_array, :] = my_strategies

        # Reconstruct legal actions
        for idx, (new_id, old_id) in enumerate(zip(new_ids, my_old_ids)):
            legal_actions = build_legal_actions(
                saved_action_sigs, old_id, "SharedArrayStorage.load_checkpoint"
            )
            self._legal_actions_cache[new_id] = legal_actions

        print(
            f"Worker {self.worker_id} loaded {len(my_keys)}/{len(saved_owned_keys)} "
            f"infosets from checkpoint"
        )
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
            self._shm_reach,
            self._shm_utility,
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
        self._shm_reach = None
        self._shm_utility = None

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
