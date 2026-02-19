"""Cross-worker ID/update synchronization helpers for SharedArrayStorage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.bucketing.utils.infoset import InfoSetKey

if TYPE_CHECKING:
    from src.solver.storage.shared_array.storage import SharedArrayStorage


def get_pending_id_requests(storage: SharedArrayStorage) -> dict[int, set[InfoSetKey]]:
    """Get pending ID requests to send to owners."""
    return storage.state.pending_id_requests


def clear_pending_id_requests(storage: SharedArrayStorage) -> None:
    """Clear pending ID requests after they are sent."""
    for owner_id in storage.state.pending_id_requests:
        storage.state.pending_id_requests[owner_id].clear()


def respond_to_id_requests(
    storage: SharedArrayStorage, requested_keys: set[InfoSetKey]
) -> dict[InfoSetKey, int]:
    """Respond to ID requests from other workers."""
    responses = {}
    for key in requested_keys:
        if key in storage.state.owned_keys:
            responses[key] = storage.state.owned_keys[key]
    return responses


def receive_id_responses(storage: SharedArrayStorage, responses: dict[InfoSetKey, int]) -> None:
    """Update remote key cache from owner responses."""
    storage.state.remote_keys.update(responses)


def buffer_update(
    storage: SharedArrayStorage,
    infoset_id: int,
    regret_delta: np.ndarray,
    strategy_delta: np.ndarray,
) -> None:
    """Buffer a cross-partition update for later delivery to owner."""
    if infoset_id in storage.state.pending_updates:
        old_regret, old_strategy = storage.state.pending_updates[infoset_id]
        storage.state.pending_updates[infoset_id] = (
            old_regret + regret_delta,
            old_strategy + strategy_delta,
        )
    else:
        storage.state.pending_updates[infoset_id] = (regret_delta.copy(), strategy_delta.copy())


def get_pending_updates(
    storage: SharedArrayStorage,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Get pending cross-partition updates."""
    return storage.state.pending_updates


def clear_pending_updates(storage: SharedArrayStorage) -> None:
    """Clear pending cross-partition updates after they are sent."""
    storage.state.pending_updates.clear()


def apply_updates(
    storage: SharedArrayStorage, updates: dict[int, tuple[np.ndarray, np.ndarray]]
) -> None:
    """Apply updates to infosets owned by this worker."""
    for infoset_id, (regret_delta, strategy_delta) in updates.items():
        if not storage.is_owned_by_id(infoset_id):
            print(
                f"Warning: Worker {storage.worker_id} received update for non-owned infoset "
                f"{infoset_id}"
            )
            continue

        num_actions = storage.shared_action_counts[infoset_id]
        if num_actions > 0:
            storage.shared_regrets[infoset_id, :num_actions] += regret_delta[:num_actions]
            storage.shared_strategy_sum[infoset_id, :num_actions] += strategy_delta[:num_actions]
