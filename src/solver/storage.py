"""
Storage systems for CFR information sets.

Provides in-memory storage with optional disk checkpointing
for efficient MCCFR training at scale.

Includes SharedArrayStorage for parallel MCCFR training using
flat NumPy arrays backed directly by shared memory.
"""

import logging
import pickle
import time
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import xxhash

from src.bucketing.utils.infoset import InfoSet, InfoSetKey
from src.game.actions import Action, ActionType, fold

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory
    from multiprocessing.synchronize import Event as EventType

# Setup logger
logger = logging.getLogger(__name__)

CHECKPOINT_REQUIRED_FILES = (
    "regrets.npy",
    "strategies.npy",
    "action_counts.npy",
    "key_mapping.pkl",
    "action_signatures.pkl",
)


def get_missing_checkpoint_files(checkpoint_dir: Path) -> list[str]:
    """Return list of missing checkpoint files in the given directory."""
    return [name for name in CHECKPOINT_REQUIRED_FILES if not (checkpoint_dir / name).exists()]


def _reconstruct_action(action_type_name: str, amount: int) -> Action:
    """
    Reconstruct Action object from serialized signature.

    Args:
        action_type_name: Name of ActionType enum (e.g., "FOLD", "CALL", "RAISE")
        amount: Action amount

    Returns:
        Reconstructed Action object

    Raises:
        ValueError: If action_type_name is invalid
    """
    try:
        action_type = ActionType[action_type_name]
    except KeyError:
        raise ValueError(f"Invalid action type name: {action_type_name}")

    return Action(type=action_type, amount=amount)


def _validate_action_signatures(
    action_counts: np.ndarray, action_sigs: dict[int, list[tuple[str, int]]], context: str
) -> None:
    """
    Ensure action signatures align with stored action counts and IDs are unique.

    Raises:
        ValueError with context if mismatches are detected.
    """
    # ID uniqueness check
    ids = list(action_sigs.keys())
    unique = len(set(ids))
    if unique != len(ids):
        raise ValueError(
            f"{context}: duplicate infoset IDs in action_signatures "
            f"({len(ids) - unique} duplicates)"
        )

    # Length alignment check
    mismatches: list[tuple[int, int, int | None, str]] = []
    for infoset_id, sigs in action_sigs.items():
        if infoset_id >= len(action_counts):
            mismatches.append((infoset_id, len(sigs), None, "id_out_of_range"))
            continue
        n_actions = int(action_counts[infoset_id])
        if len(sigs) != n_actions:
            mismatches.append((infoset_id, len(sigs), n_actions, "len_mismatch"))

    if mismatches:
        # Show a concise preview in the exception message
        preview = "; ".join(
            f"id {mid}: sig_len {slen} vs count {cnt}" for mid, slen, cnt, _ in mismatches[:5]
        )
        raise ValueError(
            f"{context}: {len(mismatches)} action signature/count mismatches. Examples: {preview}"
        )


class Storage(ABC):
    """Abstract base class for infoset storage."""

    checkpoint_dir: Optional[Path] = None

    @abstractmethod
    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """Get existing infoset or create new one."""
        pass

    @abstractmethod
    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        """Get existing infoset or None if not found."""
        pass

    @abstractmethod
    def num_infosets(self) -> int:
        """Get total number of stored infosets."""
        pass

    def is_owned(self, key: InfoSetKey) -> bool:
        """
        Check if this storage instance owns the given infoset key.

        For non-partitioned storage, all keys are "owned" (returns True).
        For partitioned storage, only keys mapping to this worker's partition are owned.
        """
        return True

    @abstractmethod
    def checkpoint(self, iteration: int):
        """Save a checkpoint at given iteration."""
        pass


class InMemoryStorage(Storage):
    """
    Read-only storage for loading checkpoints (charts, analysis, debugging).

    This is a lightweight storage class for tools that need to load and examine
    solver strategies without modifying them. For training, use SharedArrayStorage.

    NOT FOR TRAINING - use SharedArrayStorage for all training operations.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize read-only storage from checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint files to load
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self._infosets_by_id: Dict[int, InfoSet] = {}
        self.key_to_id: Dict[InfoSetKey, int] = {}
        self.id_to_key: Dict[int, InfoSetKey] = {}
        self.next_id = 0

        if self.checkpoint_dir and self.checkpoint_dir.exists():
            self._load_from_checkpoint()

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: List[Action]) -> InfoSet:
        """Get existing infoset (read-only - does not create new ones)."""
        infoset_id = self.key_to_id.get(key)
        if infoset_id is not None:
            return self._infosets_by_id[infoset_id]

        # For read-only storage, return None or raise error
        raise ValueError(
            f"InMemoryStorage is read-only. Cannot create infoset for key {key}. "
            "Use SharedArrayStorage for training."
        )

    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        """Get existing infoset or None."""
        infoset_id = self.key_to_id.get(key)
        if infoset_id is None:
            return None
        return self._infosets_by_id.get(infoset_id)

    def get_strategy(
        self,
        key: InfoSetKey,
        legal_actions: Optional[List[Action]] = None,
        fallback: Literal["uniform", "error", "none"] = "uniform",
        use_average: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Unified strategy retrieval with consistent fallback behavior.

        This method provides a single, consistent interface for retrieving strategies
        across the codebase, replacing ad-hoc patterns in mccfr.py, exploitability.py,
        head_to_head.py, and chart_handler.py.

        Args:
            key: InfoSetKey identifier for the infoset
            legal_actions: If provided, filter and remap strategy to these actions.
                          If None, return full strategy over all stored actions.
            fallback: How to handle missing infosets:
                     - "uniform": Return uniform distribution over legal_actions (default)
                     - "error": Raise ValueError
                     - "none": Return None
            use_average: If True, use average strategy (converged Nash).
                        If False, use current strategy (regret matching).

        Returns:
            Normalized float64 strategy array summing to 1.0, or None if fallback="none"
            and infoset not found.

        Raises:
            ValueError: If fallback="error" and infoset not found

        Examples:
            # Get average strategy for all actions
            strategy = storage.get_strategy(key)

            # Get current strategy for specific legal actions
            strategy = storage.get_strategy(
                key, legal_actions=[fold(), call()], use_average=False
            )

            # Strict mode: return None if missing
            strategy = storage.get_strategy(key, fallback="none")
            if strategy is None:
                # Handle missing infoset
                pass
        """
        infoset = self.get_infoset(key)

        if infoset is None:
            # Handle missing infoset based on fallback mode
            if fallback == "error":
                raise ValueError(f"Infoset not found: {key}")
            elif fallback == "none":
                return None
            else:  # fallback == "uniform"
                if legal_actions is None:
                    return None
                return np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)

        # Infoset found - get strategy
        if legal_actions is not None:
            # Filter to specific legal actions
            strategy, _ = infoset.get_strategy_safe(legal_actions, use_average)
            return strategy

        # Return full strategy over all stored actions
        return infoset.get_filtered_strategy(use_average=use_average)

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

    def checkpoint(self, iteration: int):
        """Read-only storage cannot checkpoint."""
        raise NotImplementedError(
            "InMemoryStorage is read-only. Use SharedArrayStorage for training."
        )

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
        missing_files = get_missing_checkpoint_files(self.checkpoint_dir)
        if missing_files:
            raise ValueError(
                f"Checkpoint is incomplete. Missing files: {missing_files}\n"
                f"Required: {list(CHECKPOINT_REQUIRED_FILES)}"
            )

        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        regrets_file = self.checkpoint_dir / "regrets.npy"
        strategies_file = self.checkpoint_dir / "strategies.npy"
        action_counts_file = self.checkpoint_dir / "action_counts.npy"

        with open(key_mapping_file, "rb") as f:
            mapping_data = pickle.load(f)

            # Support both old InMemoryStorage format and new SharedArrayStorage format
            if "key_to_id" in mapping_data:
                # Old InMemoryStorage format
                self.key_to_id = mapping_data["key_to_id"]
                self.id_to_key = mapping_data["id_to_key"]
                self.next_id = mapping_data["next_id"]
            elif "owned_keys" in mapping_data:
                # SharedArrayStorage format (from parallel training)
                self.key_to_id = mapping_data["owned_keys"]
                self.id_to_key = {v: k for k, v in self.key_to_id.items()}
                self.next_id = mapping_data.get("max_id", len(self.key_to_id))
            else:
                raise ValueError("Invalid checkpoint format: missing key mappings")

        # Load action signatures
        action_sigs_file = self.checkpoint_dir / "action_signatures.pkl"
        with open(action_sigs_file, "rb") as f:
            saved_action_sigs = pickle.load(f)
        logger.info(f"Loaded {len(saved_action_sigs)} action signatures from checkpoint")

        all_regrets = np.load(regrets_file, mmap_mode="r")
        action_counts = np.load(action_counts_file, mmap_mode="r")
        all_strategies = np.load(strategies_file, mmap_mode="r")

        # Validate alignment before constructing infosets
        _validate_action_signatures(
            action_counts, saved_action_sigs, "InMemoryStorage checkpoint load"
        )

        for infoset_id, key in self.id_to_key.items():
            n_actions = action_counts[infoset_id]
            regrets = all_regrets[infoset_id, :n_actions].copy()
            strategies = all_strategies[infoset_id, :n_actions].copy()

            # Reconstruct legal actions from signatures if available
            if infoset_id in saved_action_sigs:
                action_sigs = saved_action_sigs[infoset_id]
                legal_actions = [
                    _reconstruct_action(action_type_name, amount)
                    for action_type_name, amount in action_sigs
                ]
            else:
                raise ValueError(
                    f"Missing action signatures for infoset ID {infoset_id} in checkpoint. "
                    "Cannot reconstruct legal actions."
                )

            infoset = InfoSet(key, legal_actions)
            infoset.regrets = regrets
            infoset.strategy_sum = strategies

            self._infosets_by_id[infoset_id] = infoset

        logger.info(f"Loaded {len(self._infosets_by_id)} infosets from checkpoint")

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
        max_infosets: int = 2_000_000,
        max_actions: int = 10,
        is_coordinator: bool = False,
        checkpoint_dir: Optional[Path] = None,
        action_config_hash: Optional[str] = None,
        ready_event: Optional["EventType"] = None,
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
            action_config_hash: Optional hash of action abstraction config.
                               Used to detect config changes when loading checkpoints.
                               Generate with BettingActions.get_config_hash().
            ready_event: Optional multiprocessing.Event for synchronization.
                        If provided:
                        - Coordinator sets it after creating shared memory
                        - Workers wait for it before attempting to attach
                        This eliminates race conditions during parallel startup.
        """
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.session_id = session_id[:8]  # Truncate for macOS shm name limit
        self.max_infosets = max_infosets
        self.max_actions = max_actions
        self.is_coordinator = is_coordinator
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.action_config_hash = action_config_hash
        self.ready_event = ready_event
        self.base_max_infosets = max_infosets
        self.base_slots_per_worker = (max_infosets - 1) // num_workers
        self._extra_regions: list[tuple[int, int, int, int]] = []
        self._extra_allocations: list[dict[str, int]] = []

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

        # Load checkpoint if available
        if checkpoint_dir:
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

    def verify_ownership_invariants(self) -> Dict[str, Any]:
        """
        Verify ownership invariants for debugging and testing.

        This method checks that:
        1. All owned keys map to IDs within valid ranges
        2. No ID is assigned to multiple keys
        3. All allocated IDs have corresponding action counts > 0
        4. ID ranges don't overlap

        Returns:
            Dict with verification results:
            - 'valid': bool - True if all invariants hold
            - 'errors': List[str] - List of error messages
            - 'stats': Dict - Statistics about ownership

        Note: This is a debugging tool and should not be called in hot paths.
        """
        errors = []
        stats = {
            "owned_keys": len(self._owned_keys),
            "remote_keys": len(self._remote_keys),
            "base_range_used": max(
                0, min(self.next_local_id, self.id_range_end) - self.id_range_start
            ),
            "base_range_size": self.id_range_end - self.id_range_start,
            "extra_allocations": len(self._extra_allocations),
        }

        # Check 1: All owned keys map to IDs we actually own
        for key, infoset_id in self._owned_keys.items():
            if not self.is_owned_by_id(infoset_id):
                errors.append(f"Owned key {key} maps to ID {infoset_id} which is not in our ranges")

        # Check 2: No duplicate IDs (each ID maps to at most one key)
        id_to_keys: Dict[int, List[InfoSetKey]] = {}
        for key, infoset_id in self._owned_keys.items():
            if infoset_id not in id_to_keys:
                id_to_keys[infoset_id] = []
            id_to_keys[infoset_id].append(key)

        for infoset_id, keys in id_to_keys.items():
            if len(keys) > 1:
                errors.append(f"ID {infoset_id} is assigned to multiple keys: {keys[:3]}...")

        # Check 3: All used IDs in base range have action counts
        for infoset_id in range(self.id_range_start, self.next_local_id):
            if self.shared_action_counts[infoset_id] == 0:
                # Check if this ID is actually used (has a key)
                if infoset_id in id_to_keys:
                    errors.append(f"ID {infoset_id} has key but action_count=0")

        # Check 4: Extra allocation ranges don't overlap with base or each other
        all_ranges = [(self.id_range_start, self.id_range_end, "base")]
        for i, alloc in enumerate(self._extra_allocations):
            all_ranges.append((alloc["start"], alloc["end"], f"extra_{i}"))

        for i, (start1, end1, name1) in enumerate(all_ranges):
            for j, (start2, end2, name2) in enumerate(all_ranges):
                if i >= j:
                    continue
                # Check for overlap
                if start1 < end2 and start2 < end1:
                    errors.append(
                        f"Range overlap: {name1}=[{start1},{end1}) and {name2}=[{start2},{end2})"
                    )

        # Check 5: next_local_id is within valid range
        if self.next_local_id < self.id_range_start:
            errors.append(
                f"next_local_id ({self.next_local_id}) < id_range_start ({self.id_range_start})"
            )
        if self.next_local_id > self.id_range_end:
            # This could be valid if we've exhausted base range and have extra allocations
            if not self._extra_allocations:
                errors.append(
                    f"next_local_id ({self.next_local_id}) > id_range_end ({self.id_range_end}) "
                    f"but no extra allocations"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "stats": stats,
        }

    def get_owner_by_id(self, infoset_id: int) -> Optional[int]:
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

        # Signal workers that shared memory is ready
        if self.ready_event is not None:
            self.ready_event.set()
            logger.debug("Coordinator signaled ready_event")

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
            logger.debug(f"Worker {self.worker_id} received ready signal")

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

                self._create_numpy_views()
                logger.info(f"Worker {self.worker_id} attached to shared memory")
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

    def get_resize_stats(self) -> Dict[str, int]:
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
            "capacity_usage": self.get_capacity_usage(),  # type: ignore
            "max_infosets": self.max_infosets,
        }

    def resize(self, new_max_infosets: int) -> None:
        """
        Resize storage to new capacity (coordinator only).

        This is a stop-the-world operation:
        1. Create new larger shared memory segments
        2. Copy existing data to new arrays
        3. Update ID ranges for all workers
        4. Clean up old shared memory

        Args:
            new_max_infosets: New maximum number of infosets

        Raises:
            RuntimeError: If called by non-coordinator or new size is smaller
        """
        if not self.is_coordinator:
            raise RuntimeError("Only coordinator can resize storage")

        if new_max_infosets <= self.max_infosets:
            raise RuntimeError(
                f"New size {new_max_infosets} must be larger than current {self.max_infosets}"
            )

        old_max_infosets = self.max_infosets
        old_regrets = self.shared_regrets
        old_strategy = self.shared_strategy_sum
        old_action_counts = self.shared_action_counts

        logger.info(
            f"Resizing storage: {old_max_infosets:,} -> {new_max_infosets:,} infosets "
            f"(growth factor: {new_max_infosets / old_max_infosets:.1f}x)"
        )

        # Store old shared memory handles for cleanup
        old_shm_regrets = self._shm_regrets
        old_shm_strategy = self._shm_strategy
        old_shm_actions = self._shm_actions

        # Update max_infosets before creating new shared memory
        old_max_infosets = self.max_infosets
        self.max_infosets = new_max_infosets

        # Create new session_id for resized shared memory
        # Append a counter to ensure unique names
        import uuid

        self.session_id = uuid.uuid4().hex[:8]

        # Create new shared memory segments
        self._create_shared_memory()

        # Copy existing data to new arrays
        # Note: Only copy up to old_max_infosets rows
        self.shared_regrets[:old_max_infosets, :] = old_regrets[:, :]
        self.shared_strategy_sum[:old_max_infosets, :] = old_strategy[:, :]
        self.shared_action_counts[:old_max_infosets] = old_action_counts[:]

        # Append a new extra region for allocations to avoid overlapping ranges.
        self._add_extra_region(old_max_infosets, new_max_infosets)

        # Clean up old shared memory
        try:
            old_shm_regrets.close()
            old_shm_regrets.unlink()
            old_shm_strategy.close()
            old_shm_strategy.unlink()
            old_shm_actions.close()
            old_shm_actions.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up old shared memory: {e}")

        logger.info(
            f"Resize complete: new capacity {new_max_infosets:,}, new session_id={self.session_id}"
        )

    def reattach_after_resize(
        self,
        new_session_id: str,
        new_max_infosets: int,
        preserved_keys: Dict["InfoSetKey", int],
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
            new_max_infosets: New maximum infosets capacity
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

        old_max_infosets = self.max_infosets

        # Update session and capacity
        self.session_id = new_session_id
        self.max_infosets = new_max_infosets

        # next_local_id stays at same absolute position - data is not moved
        self.next_local_id = preserved_next_id

        # Restore owned keys - IDs point to same absolute positions
        self._owned_keys = preserved_keys

        # Attach to new shared memory
        self._attach_shared_memory()

        # Append a new extra region for allocations
        self._add_extra_region(old_max_infosets, new_max_infosets)

        logger.info(
            f"Worker {self.worker_id} reattached after resize: "
            f"session={new_session_id}, max_infosets={new_max_infosets:,}, "
            f"id_range=[{self.id_range_start}, {self.id_range_end}), "
            f"next_id={self.next_local_id}"
        )

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

    def num_infosets(self) -> int:
        """Get total number of infosets allocated by this worker."""
        base_used = max(0, min(self.next_local_id, self.id_range_end) - self.id_range_start)
        extra_used = sum(alloc["next"] - alloc["start"] for alloc in self._extra_allocations)
        return base_used + extra_used

    def num_owned_infosets(self) -> int:
        """Get number of infosets owned by this worker."""
        return len(self._owned_keys)

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
                    "max_infosets": self.max_infosets,  # Persist resized capacity
                    "action_config_hash": self.action_config_hash,  # For config validation
                },
                f,
            )

        # Save shared arrays (all used rows up to max_id)
        regrets_file = self.checkpoint_dir / "regrets.npy"
        strategies_file = self.checkpoint_dir / "strategies.npy"
        action_counts_file = self.checkpoint_dir / "action_counts.npy"

        regrets_mm = np.lib.format.open_memmap(
            regrets_file,
            mode="w+",
            dtype=self.shared_regrets.dtype,
            shape=(max_id, self.shared_regrets.shape[1]),
        )
        regrets_mm[:] = self.shared_regrets[:max_id, :]
        del regrets_mm

        strategies_mm = np.lib.format.open_memmap(
            strategies_file,
            mode="w+",
            dtype=self.shared_strategy_sum.dtype,
            shape=(max_id, self.shared_strategy_sum.shape[1]),
        )
        strategies_mm[:] = self.shared_strategy_sum[:max_id, :]
        del strategies_mm

        np.save(action_counts_file, self.shared_action_counts[:max_id])

        # Save action signatures (action type and amount for each infoset)
        # This allows reconstructing actual Action objects on load
        action_sigs_file = self.checkpoint_dir / "action_signatures.pkl"
        action_sigs = {}
        for infoset_id, actions in self._legal_actions_cache.items():
            # Store (action_type_name, amount) tuples
            # Use .name instead of .value for readability and enum reordering safety
            action_sigs[infoset_id] = [(action.type.name, action.amount) for action in actions]

        with open(action_sigs_file, "wb") as f:
            pickle.dump(action_sigs, f)

        # Validate before declaring checkpoint done (fail fast on corruption)
        _validate_action_signatures(
            self.shared_action_counts[:max_id],
            action_sigs,
            f"SharedArrayStorage.checkpoint(iter={iteration})",
        )

        logger.info(
            f"Checkpoint saved: {num_keys} infosets at iteration {iteration}, "
            f"{len(action_sigs)} action signatures"
        )

    @staticmethod
    def get_checkpoint_info(checkpoint_dir: Path) -> Optional[Dict]:
        """
        Get information about a checkpoint without loading it.

        Useful for determining max_infosets before creating storage.

        Args:
            checkpoint_dir: Directory containing checkpoint files

        Returns:
            Dict with checkpoint info, or None if no valid checkpoint
        """
        key_mapping_file = checkpoint_dir / "key_mapping.pkl"
        if not key_mapping_file.exists():
            return None

        with open(key_mapping_file, "rb") as f:
            mapping_data = pickle.load(f)

        return {
            "num_infosets": len(mapping_data.get("owned_keys", {})),
            "max_infosets": mapping_data.get("max_infosets"),  # May be None for old checkpoints
            "num_workers": mapping_data.get("num_workers"),
            "max_id": mapping_data.get("max_id"),
            "action_config_hash": mapping_data.get(
                "action_config_hash"
            ),  # May be None for old checkpoints
        }

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint from disk.

        Supports loading checkpoints created with different worker configurations.
        Each worker loads only the keys it owns based on current hash-based partitioning.
        For sequential mode (num_workers=1), the single worker loads all keys.

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.checkpoint_dir:
            return False

        missing_files = get_missing_checkpoint_files(self.checkpoint_dir)
        if missing_files:
            return False

        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        regrets_file = self.checkpoint_dir / "regrets.npy"
        strategies_file = self.checkpoint_dir / "strategies.npy"
        action_counts_file = self.checkpoint_dir / "action_counts.npy"

        # Load saved checkpoint data
        with open(key_mapping_file, "rb") as f:
            mapping_data = pickle.load(f)
            saved_owned_keys = mapping_data["owned_keys"]
            saved_max_infosets = mapping_data.get("max_infosets")
            saved_config_hash = mapping_data.get("action_config_hash")

            # Log if checkpoint was from a resized storage
            if saved_max_infosets and saved_max_infosets != self.max_infosets:
                logger.info(
                    f"Checkpoint max_infosets ({saved_max_infosets:,}) differs from "
                    f"current ({self.max_infosets:,})"
                )

            # Validate action config hash if both are available
            if saved_config_hash and self.action_config_hash:
                if saved_config_hash != self.action_config_hash:
                    logger.warning(
                        f"ACTION CONFIG MISMATCH DETECTED!\n"
                        f"  Checkpoint config hash: {saved_config_hash}\n"
                        f"  Current config hash:    {self.action_config_hash}\n"
                        f"  Strategies may be INVALID if action abstraction changed.\n"
                        f"  Consider starting fresh training or using the original config."
                    )
            elif saved_config_hash and not self.action_config_hash:
                logger.info(
                    f"Checkpoint has config hash ({saved_config_hash}) but current "
                    f"storage was not initialized with one. Skipping validation."
                )
            elif not saved_config_hash and self.action_config_hash:
                logger.info(
                    f"Checkpoint does not have config hash (old format). "
                    f"Current config hash: {self.action_config_hash}"
                )

        saved_regrets = np.load(regrets_file, mmap_mode="r")
        saved_action_counts = np.load(action_counts_file, mmap_mode="r")
        saved_strategies = np.load(strategies_file, mmap_mode="r")

        # Load action signatures
        action_sigs_file = self.checkpoint_dir / "action_signatures.pkl"
        with open(action_sigs_file, "rb") as f:
            saved_action_sigs = pickle.load(f)
        logger.info(f"Loaded {len(saved_action_sigs)} action signatures from checkpoint")

        # Validate alignment (fail fast if counts/signatures diverge)
        _validate_action_signatures(
            saved_action_counts,
            saved_action_sigs,
            "SharedArrayStorage.load_checkpoint",
        )

        # Re-partition keys based on current worker configuration
        # Each key is assigned to a worker using hash-based partitioning
        loaded_count = 0
        for key, old_id in list(saved_owned_keys.items()):
            # Determine which worker should own this key
            owner_id = self.get_owner(key)

            # Only load keys that belong to this worker
            if owner_id != self.worker_id:
                continue

            # Allocate new ID in this worker's range
            new_id = self._allocate_id()
            self._owned_keys[key] = new_id

            # Copy data from old checkpoint position to new position
            n_actions = saved_action_counts[old_id]
            self.shared_action_counts[new_id] = n_actions
            self.shared_regrets[new_id, :n_actions] = saved_regrets[old_id, :n_actions]
            self.shared_strategy_sum[new_id, :n_actions] = saved_strategies[old_id, :n_actions]

            # Reconstruct legal actions from signatures
            if old_id not in saved_action_sigs:
                raise ValueError(
                    f"Missing action signatures for infoset {key} (id {old_id}). "
                    "Cannot reconstruct legal actions."
                )

            action_sigs = saved_action_sigs[old_id]
            legal_actions = [
                _reconstruct_action(action_type_name, amount)
                for action_type_name, amount in action_sigs
            ]
            self._legal_actions_cache[new_id] = legal_actions

            loaded_count += 1

        logger.info(
            f"Worker {self.worker_id} loaded {loaded_count}/{len(saved_owned_keys)} "
            f"infosets from checkpoint (worker owns {loaded_count} keys)"
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
