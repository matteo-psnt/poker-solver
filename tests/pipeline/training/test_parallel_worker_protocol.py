"""Tests for parallel worker protocol and queue sync helpers."""

import queue
from types import SimpleNamespace

from src.pipeline.training.parallel_protocol import JobType
from src.pipeline.training.parallel_sync import (
    _process_all_messages,
    _send_pending_id_requests,
)


class _DummyStorage:
    def __init__(self):
        self.state = SimpleNamespace(
            remote_keys={},
            pending_id_requests={0: set(), 1: set()},
            requested_id_keys=set(),
        )

    def respond_to_id_requests(self, keys):
        return {key: idx for idx, key in enumerate(keys)}


def test_job_type_values_are_stable():
    assert JobType.RUN_ITERATIONS.value == "run_iterations"
    assert JobType.EXCHANGE_IDS.value == "exchange_ids"
    assert JobType.COLLECT_KEYS.value == "collect_keys"
    assert JobType.RESIZE_STORAGE.value == "resize_storage"
    assert JobType.SHUTDOWN.value == "shutdown"


def test_process_all_messages_drains_all_queues():
    storage = _DummyStorage()
    requests_q = queue.Queue()
    responses_q = queue.Queue()
    response_queues = [queue.Queue(), queue.Queue()]

    requests_q.put({"requester": 1, "keys": ("k1", "k2")})
    responses_q.put({"remote_key": 9})

    stats = _process_all_messages(
        worker_id=0,
        id_request_queue=requests_q,  # type: ignore[arg-type]
        id_response_queue=responses_q,  # type: ignore[arg-type]
        id_response_queues=response_queues,  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
    )

    assert stats == {"requests": 2, "responses": 1}
    assert storage.state.remote_keys["remote_key"] == 9
    assert response_queues[1].qsize() == 1


def test_process_id_responses_clears_in_flight_tracking():
    storage = _DummyStorage()
    storage.state.requested_id_keys = {"remote_key", "still_waiting"}
    responses_q = queue.Queue()
    responses_q.put({"remote_key": 9})

    _process_all_messages(
        worker_id=0,
        id_request_queue=queue.Queue(),  # type: ignore[arg-type]
        id_response_queue=responses_q,  # type: ignore[arg-type]
        id_response_queues=[queue.Queue(), queue.Queue()],  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
    )

    assert storage.state.requested_id_keys == {"still_waiting"}


def test_send_pending_id_requests_flushes_cross_worker_only():
    storage = _DummyStorage()
    storage.state.pending_id_requests[0] = {"own_key"}
    storage.state.pending_id_requests[1] = {"k1", "k2"}
    request_queues = [queue.Queue(), queue.Queue()]

    sent = _send_pending_id_requests(
        worker_id=0,
        id_request_queues=request_queues,  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
    )

    assert sent == 2
    assert request_queues[0].qsize() == 0
    message = request_queues[1].get_nowait()
    assert message["requester"] == 0
    assert sorted(message["keys"]) == ["k1", "k2"]
    # Flushed keys are cleared so they are not re-sent; own-partition keys remain.
    assert storage.state.pending_id_requests[1] == set()
    assert storage.state.pending_id_requests[0] == {"own_key"}
    # Sent keys become in-flight so visits cannot re-queue them until rearm.
    assert storage.state.requested_id_keys == {"k1", "k2"}


def test_send_pending_id_requests_retries_when_queue_full():
    storage = _DummyStorage()
    storage.state.pending_id_requests[1] = {"k1"}
    full_queue = queue.Queue(maxsize=1)
    full_queue.put("occupied")

    sent = _send_pending_id_requests(
        worker_id=0,
        id_request_queues=[queue.Queue(), full_queue],  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
    )

    assert sent == 0
    # Keys stay pending for the next flush instead of being dropped.
    assert storage.state.pending_id_requests[1] == {"k1"}
