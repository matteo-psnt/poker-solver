"""Cross-worker ID synchronization helpers for SharedArrayStorage."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.solver.infoset import InfoSetKey
    from src.engine.solver.storage.shared_array.storage import SharedArrayStorage


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
