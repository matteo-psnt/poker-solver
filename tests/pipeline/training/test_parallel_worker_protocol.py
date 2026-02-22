"""Tests for parallel worker protocol and queue sync helpers."""

import queue
from types import SimpleNamespace

import numpy as np

from src.pipeline.training.parallel_protocol import JobType
from src.pipeline.training.parallel_sync import _process_all_messages, _send_updates_to_owners


class _DummyStorage:
    def __init__(self):
        self.applied_updates = []
        self.state = SimpleNamespace(remote_keys={})

    def apply_updates(self, updates):
        self.applied_updates.append(updates)

    def respond_to_id_requests(self, keys):
        return {key: idx for idx, key in enumerate(keys)}

    def get_owner_by_id(self, infoset_id: int):
        if infoset_id in (1, 2):
            return 0
        if infoset_id in (3, 4):
            return 1
        return None


def test_job_type_values_are_stable():
    assert JobType.RUN_ITERATIONS.value == "run_iterations"
    assert JobType.EXCHANGE_IDS.value == "exchange_ids"
    assert JobType.APPLY_UPDATES.value == "apply_updates"
    assert JobType.COLLECT_KEYS.value == "collect_keys"
    assert JobType.RESIZE_STORAGE.value == "resize_storage"
    assert JobType.SHUTDOWN.value == "shutdown"


def test_process_all_messages_drains_all_queues():
    storage = _DummyStorage()
    updates_q = queue.Queue()
    requests_q = queue.Queue()
    responses_q = queue.Queue()
    response_queues = [queue.Queue(), queue.Queue()]

    updates_q.put({1: (np.array([1.0]), np.array([2.0]))})
    requests_q.put({"requester": 1, "keys": ("k1", "k2")})
    responses_q.put({"remote_key": 9})

    stats = _process_all_messages(
        worker_id=0,
        update_queue=updates_q,  # type: ignore[arg-type]
        id_request_queue=requests_q,  # type: ignore[arg-type]
        id_response_queue=responses_q,  # type: ignore[arg-type]
        id_response_queues=response_queues,  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
    )

    assert stats == {"updates": 1, "requests": 2, "responses": 1}
    assert len(storage.applied_updates) == 1
    assert storage.state.remote_keys["remote_key"] == 9
    assert response_queues[1].qsize() == 1


def test_send_updates_to_owners_routes_cross_partition_only():
    storage = _DummyStorage()
    queues = [queue.Queue(), queue.Queue()]

    updates = {
        1: (np.array([1.0]), np.array([1.0])),  # owned by sender 0 -> local
        3: (np.array([2.0]), np.array([2.0])),  # owned by worker 1 -> sent
    }

    _send_updates_to_owners(
        sender_id=0,
        num_workers=2,
        updates=updates,
        update_queues=queues,  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
    )

    assert queues[0].qsize() == 0
    assert queues[1].qsize() == 1
