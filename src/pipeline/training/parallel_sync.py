"""Queue synchronization helpers for parallel worker communication."""

from __future__ import annotations

import queue
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as mp

    from src.engine.solver.storage.shared_array import SharedArrayStorage


def _send_pending_id_requests(
    worker_id: int,
    id_request_queues: list[mp.Queue],
    storage: SharedArrayStorage,
) -> int:
    """
    Flush accumulated ID requests to their owning workers.

    Keys stay pending when an owner's queue is full, so they are retried on the
    next flush instead of being dropped.
    """
    sent = 0
    for owner_id, keys in storage.state.pending_id_requests.items():
        if not keys or owner_id == worker_id:
            continue
        # Snapshot keys so the multiprocessing pickler doesn't see mutations.
        keys_snapshot = tuple(keys)
        try:
            id_request_queues[owner_id].put_nowait({"requester": worker_id, "keys": keys_snapshot})
        except queue.Full:
            continue
        keys.clear()
        sent += len(keys_snapshot)
    return sent


def _process_id_requests(
    worker_id: int,
    request_queue: mp.Queue,
    response_queues: list[mp.Queue],
    storage: SharedArrayStorage,
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


def _process_id_responses(response_queue: mp.Queue, storage: SharedArrayStorage) -> int:
    """
    Process incoming ID responses and update remote key cache.
    """
    count = 0
    while True:
        try:
            responses = response_queue.get_nowait()
            storage.state.remote_keys.update(responses)
            count += len(responses)
        except queue.Empty:
            break
    return count


def _process_all_messages(
    worker_id: int,
    id_request_queue: mp.Queue,
    id_response_queue: mp.Queue,
    id_response_queues: list[mp.Queue],
    storage: SharedArrayStorage,
) -> dict[str, int]:
    """
    Process all pending messages from all queues.

    Returns a dict with counts of processed items by type.
    """
    return {
        "requests": _process_id_requests(worker_id, id_request_queue, id_response_queues, storage),
        "responses": _process_id_responses(id_response_queue, storage),
    }
