"""Cross-worker ID/update synchronization helpers for SharedArrayStorage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.bucketing.utils.infoset import InfoSetKey
    from src.solver.storage.shared_array.storage import SharedArrayStorage


def respond_to_id_requests(
    storage: SharedArrayStorage, requested_keys: set[InfoSetKey]
) -> dict[InfoSetKey, int]:
    """Respond to ID requests from other workers."""
    responses = {}
    for key in requested_keys:
        infoset_id = storage.state.owned_keys.get(key)
        if infoset_id is not None:
            responses[key] = infoset_id
    return responses


def buffer_update(
    storage: SharedArrayStorage,
    infoset_id: int,
    regret_delta: np.ndarray,
    strategy_delta: np.ndarray,
) -> None:
    """Buffer a cross-partition update for later delivery to owner."""
    if infoset_id == storage.UNKNOWN_ID:
        return

    expected_actions: int | None = None
    if 0 <= infoset_id < len(storage.shared_action_counts):
        num_actions = int(storage.shared_action_counts[infoset_id])
        if num_actions > 0:
            expected_actions = num_actions

    storage.update_queue.buffer(
        infoset_id,
        regret_delta,
        strategy_delta,
        expected_actions=expected_actions,
    )


def apply_updates(
    storage: SharedArrayStorage, updates: dict[int, tuple[np.ndarray, np.ndarray]]
) -> None:
    """Apply updates to infosets owned by this worker."""
    for infoset_id, (regret_delta, strategy_delta) in updates.items():
        if not storage.is_owned_by_id(infoset_id):
            owner_id = storage.get_owner_by_id(infoset_id)
            raise RuntimeError(
                f"Worker {storage.worker_id} received update for infoset {infoset_id} "
                f"owned by worker {owner_id}"
            )

        num_actions = storage.shared_action_counts[infoset_id]
        if num_actions <= 0:
            continue
        if regret_delta.ndim != 1 or strategy_delta.ndim != 1:
            raise ValueError("Incoming updates must use 1-D arrays")
        if regret_delta.shape != strategy_delta.shape:
            raise ValueError(
                f"Incoming update shape mismatch: {regret_delta.shape} vs {strategy_delta.shape}"
            )
        if len(regret_delta) < num_actions:
            raise ValueError(
                f"Incoming update for infoset {infoset_id} has {len(regret_delta)} actions, "
                f"expected at least {num_actions}"
            )

        storage.shared_regrets[infoset_id, :num_actions] += regret_delta[:num_actions]
        storage.shared_strategy_sum[infoset_id, :num_actions] += strategy_delta[:num_actions]
