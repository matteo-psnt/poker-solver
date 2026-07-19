"""Infoset access/allocation helpers for SharedArrayStorage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.core.game.actions import Action, fold
from src.engine.solver.infoset import InfoSet, InfoSetKey

if TYPE_CHECKING:
    from src.engine.solver.storage.shared_array.storage import SharedArrayStorage


def get_or_create_infoset(
    storage: SharedArrayStorage, key: InfoSetKey, legal_actions: Sequence[Action]
) -> InfoSet:
    """Get existing infoset or create new one."""
    state = storage.state

    # Known keys resolve by dict lookup alone; the owner hash (xxhash over
    # encoded key bytes) is only paid on the first-ever visit to a key.
    infoset_id = state.owned_keys.get(key)
    if infoset_id is None:
        infoset_id = state.remote_keys.get(key)
    if infoset_id is not None:
        return create_infoset_view(storage, infoset_id, key, legal_actions)

    owner = storage.get_owner(key)

    if owner == storage.worker_id:
        num_actions = len(legal_actions)
        if num_actions > storage.max_actions:
            raise ValueError(
                f"Infoset has {num_actions} actions, exceeding max_actions={storage.max_actions}. "
                "Increase max_actions in storage config or reduce action abstraction complexity."
            )

        infoset_id = allocate_id(storage)
        state.owned_keys[key] = infoset_id
        state.unshipped_keys.append((key, infoset_id))
        storage.shared_action_counts[infoset_id] = num_actions
        state.legal_actions_cache[infoset_id] = legal_actions

        # Answer workers that requested this key before it existed.
        waiting = state.unanswered_id_requests.pop(key, None)
        if waiting is not None:
            for requester in waiting:
                state.pending_late_responses.setdefault(requester, {})[key] = infoset_id
        return create_infoset_view(storage, infoset_id, key, legal_actions)

    if key not in state.requested_id_keys:
        state.pending_id_requests[owner].add(key)
    return create_infoset_view(storage, storage.UNKNOWN_ID, key, legal_actions)


def allocate_id(storage: SharedArrayStorage) -> int:
    """Allocate ID from this worker's exclusive ranges."""
    if storage.state.next_local_id < storage.id_range_end:
        infoset_id = storage.state.next_local_id
        storage.state.next_local_id += 1
        return infoset_id

    for alloc in storage.state.extra_allocations:
        if alloc.next < alloc.end:
            infoset_id = alloc.next
            alloc.next += 1
            return infoset_id

    raise RuntimeError(
        f"Worker {storage.worker_id} exhausted ID ranges "
        f"[{storage.id_range_start}, {storage.id_range_end}) and extras. "
        "Storage resize required - coordinator should trigger resize."
    )


def create_infoset_view(
    storage: SharedArrayStorage, infoset_id: int, key: InfoSetKey, legal_actions: Sequence[Action]
) -> InfoSet:
    """Create an infoset backed by shared-memory views."""
    num_actions = len(legal_actions)
    infoset = InfoSet(key, legal_actions, allocate_arrays=False)

    regrets_view = storage.shared_regrets[infoset_id, :num_actions]
    strategy_view = storage.shared_strategy_sum[infoset_id, :num_actions]

    read_only_stats = False
    if infoset_id == storage.UNKNOWN_ID:
        regrets_view = regrets_view.copy()
        regrets_view.setflags(write=False)
        strategy_view = strategy_view.copy()
        strategy_view.setflags(write=False)
        read_only_stats = True
        infoset.writable = False

    infoset.regrets = regrets_view
    infoset.strategy_sum = strategy_view
    infoset.attach_stats_views(
        storage.shared_reach_counts,
        storage.shared_cumulative_utility,
        infoset_id,
        read_only=read_only_stats,
    )
    infoset.sync_stats_to_storage(
        storage.shared_reach_counts[infoset_id],
        storage.shared_cumulative_utility[infoset_id],
    )

    return infoset


def get_infoset(storage: SharedArrayStorage, key: InfoSetKey) -> InfoSet | None:
    """Get existing infoset or None."""
    owner = storage.get_owner(key)

    if owner == storage.worker_id:
        infoset_id = storage.state.owned_keys.get(key)
    else:
        infoset_id = storage.state.remote_keys.get(key)

    if infoset_id is None:
        return None

    legal_actions = storage.state.legal_actions_cache.get(infoset_id)
    if legal_actions is None:
        # Every get_infoset consumer matches stored strategy mass to actions BY
        # IDENTITY (policy lookup, range inference, exploitability). Fabricating
        # placeholder actions here would silently misattribute that mass, so a
        # miss — impossible on checkpoint-loaded storage, whose cache is fully
        # populated on load — must fail loudly instead.
        raise RuntimeError(
            f"No legal actions cached for infoset {infoset_id} ({key}); "
            "strategy lookup on this storage would misattribute probability mass."
        )

    return create_infoset_view(storage, infoset_id, key, legal_actions)


def num_infosets(storage: SharedArrayStorage) -> int:
    """Get total number of infosets allocated by this worker."""
    base_used = max(
        0,
        min(storage.state.next_local_id, storage.id_range_end) - storage.id_range_start,
    )
    extra_used = sum(alloc.used for alloc in storage.state.extra_allocations)
    return base_used + extra_used


def iter_infosets(storage: SharedArrayStorage):
    """Iterate infosets owned by this worker."""
    for key, infoset_id in storage.state.owned_keys.items():
        legal_actions = storage.state.legal_actions_cache.get(infoset_id)
        if legal_actions is None:
            # Unlike get_infoset, iteration consumers (quality metrics,
            # exploitability sweeps) read only the numeric arrays, where just the
            # action COUNT matters — placeholder identities are never consulted.
            num_actions = storage.shared_action_counts[infoset_id]
            legal_actions = [fold() for _ in range(num_actions)]
        yield create_infoset_view(storage, infoset_id, key, legal_actions)


def num_owned_infosets(storage: SharedArrayStorage) -> int:
    """Get number of infosets owned by this worker."""
    return len(storage.state.owned_keys)
