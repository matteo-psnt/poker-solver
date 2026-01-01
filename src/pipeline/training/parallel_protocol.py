"""Protocol definitions for parallel MCCFR worker orchestration."""

from enum import Enum


class JobType(Enum):
    """Job types for parallel training."""

    RUN_ITERATIONS = "run_iterations"
    EXCHANGE_IDS = "exchange_ids"  # Batched ID request/response between workers
    APPLY_UPDATES = "apply_updates"  # Apply cross-partition updates
    COLLECT_KEYS = "collect_keys"  # Collect owned keys for checkpointing
    RESIZE_STORAGE = "resize_storage"  # Trigger storage resize (stop-the-world)
    SHUTDOWN = "shutdown"
