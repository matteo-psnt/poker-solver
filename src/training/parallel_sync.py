"""Queue synchronization helpers for parallel worker communication."""

from __future__ import annotations

import multiprocessing as mp
import queue
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.solver.storage.shared_array import SharedArrayStorage


def _process_incoming_updates(update_queue: mp.Queue, storage: "SharedArrayStorage") -> int:
    """
    Process incoming cross-partition updates from other workers.

    Returns number of updates processed.
    """
    count = 0
    while True:
        try:
            update_batch = update_queue.get_nowait()
            storage.apply_updates(update_batch)
            count += len(update_batch)
        except queue.Empty:
            break
    return count


def _process_id_requests(
    worker_id: int,
    request_queue: mp.Queue,
    response_queues: list[mp.Queue],
    storage: "SharedArrayStorage",
) -> int:
    """
    Process incoming ID requests and send responses.

    Only responds for keys we own and have allocated.
    """
    count = 0
    while True:
        try:
            request = request_queue.get_nowait()
            requester = request["requester"]
            keys = request["keys"]

            # Respond with IDs for keys we own
            responses = storage.respond_to_id_requests(keys)

            if responses:
                try:
                    response_queues[requester].put(responses, timeout=1.0)
                    count += len(responses)
                except queue.Full:
                    print(
                        f"[Worker {worker_id}] Warning: ID response queue for worker "
                        f"{requester} is full; dropping {len(responses)} ids",
                        file=sys.stderr,
                        flush=True,
                    )

        except queue.Empty:
            break
    return count


def _process_id_responses(response_queue: mp.Queue, storage: "SharedArrayStorage") -> int:
    """
    Process incoming ID responses and update remote key cache.
    """
    count = 0
    while True:
        try:
            responses = response_queue.get_nowait()
            storage.receive_id_responses(responses)
            count += len(responses)
        except queue.Empty:
            break
    return count


def _send_updates_to_owners(
    sender_id: int,
    num_workers: int,
    updates: dict[int, tuple[np.ndarray, np.ndarray]],
    update_queues: list[mp.Queue],
    storage: "SharedArrayStorage",
) -> None:
    """
    Send cross-partition updates to their respective owners.

    Ownership is determined by ID range, not modulo.
    """
    # Group updates by owner
    updates_by_owner: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        i: {} for i in range(num_workers)
    }

    for infoset_id, (regret_delta, strategy_delta) in updates.items():
        owner_id = storage.get_owner_by_id(infoset_id)
        if owner_id is None:
            continue
        if owner_id != sender_id:
            updates_by_owner[owner_id][infoset_id] = (regret_delta, strategy_delta)

    # Send to each owner's queue
    for owner_id, owner_updates in updates_by_owner.items():
        if owner_updates:
            try:
                update_queues[owner_id].put(owner_updates, timeout=5.0)
            except queue.Full:
                print(
                    f"[Worker {sender_id}] Warning: Update queue for worker {owner_id} is full",
                    flush=True,
                )


def _process_all_messages(
    worker_id: int,
    update_queue: mp.Queue,
    id_request_queue: mp.Queue,
    id_response_queue: mp.Queue,
    id_response_queues: list[mp.Queue],
    storage: "SharedArrayStorage",
) -> dict[str, int]:
    """
    Process all pending messages from all queues.

    Returns a dict with counts of processed items by type.
    """
    return {
        "updates": _process_incoming_updates(update_queue, storage),
        "requests": _process_id_requests(worker_id, id_request_queue, id_response_queues, storage),
        "responses": _process_id_responses(id_response_queue, storage),
    }
