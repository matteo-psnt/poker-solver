"""Compatibility facade for parallel training modules."""

from src.training.parallel_manager import SharedArrayWorkerManager
from src.training.parallel_protocol import JobType
from src.training.parallel_sync import (
    _process_all_messages,
    _process_id_requests,
    _process_id_responses,
    _process_incoming_updates,
    _send_updates_to_owners,
)
from src.training.parallel_worker import _worker_loop

__all__ = [
    "JobType",
    "SharedArrayWorkerManager",
    "_worker_loop",
    "_process_all_messages",
    "_process_id_requests",
    "_process_id_responses",
    "_process_incoming_updates",
    "_send_updates_to_owners",
]
