"""Cross-worker ID synchronization helpers for SharedArrayStorage."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.solver.infoset import InfoSetKey
    from src.engine.solver.storage.shared_array.storage import SharedArrayStorage


def respond_to_id_requests(
    storage: SharedArrayStorage, requested_keys: set[InfoSetKey], requester: int
) -> dict[InfoSetKey, int]:
    """Respond to ID requests from another worker.

    Keys this worker owns but has not allocated yet are remembered in
    ``unanswered_id_requests`` and answered proactively when the allocation
    happens — the requester never has to retry to learn the ID.
    """
    responses = {}
    for key in requested_keys:
        infoset_id = storage.state.owned_keys.get(key)
        if infoset_id is not None:
            responses[key] = infoset_id
        else:
            storage.state.unanswered_id_requests.setdefault(key, set()).add(requester)
    return responses


def rearm_unresolved_id_requests(storage: SharedArrayStorage) -> int:
    """Move still-unanswered requested keys back to the pending sets.

    A request goes unanswered when the owning worker has not allocated the key
    yet (it only responds for keys it owns *and* has visited). Called at batch
    boundaries so those keys are retried once per batch — the owner has likely
    allocated them by then — without the per-flush re-send storm that gating on
    ``requested_id_keys`` exists to prevent.

    Returns:
        Number of keys re-armed.
    """
    unresolved = storage.state.requested_id_keys
    if not unresolved:
        return 0

    count = 0
    for key in unresolved:
        if key in storage.state.remote_keys:
            continue
        storage.state.pending_id_requests[storage.get_owner(key)].add(key)
        count += 1
    unresolved.clear()
    return count
