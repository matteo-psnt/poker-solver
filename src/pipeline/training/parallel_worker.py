"""Worker loop implementation for parallel MCCFR training."""

from __future__ import annotations

import multiprocessing as mp
import pickle
import queue
import random
import signal
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.storage.shared_array import SharedArrayStorage
from src.pipeline.training.parallel_protocol import JobType
from src.pipeline.training.parallel_sync import (
    _process_all_messages,
    _send_pending_id_requests,
)
from src.shared.config import Config


def _compute_seed(base_seed: int, worker_id: int, batch_id: int = 0) -> int:
    """
    Derive the RNG stream for one worker's batch.

    Every ``(worker_id, batch_id)`` pair must map to a distinct, uncorrelated
    stream: workers sample chance nodes and actions off the global RNG, so two
    pairs sharing a seed would silently draw correlated MCCFR samples. Mixing the
    coordinates by hand does not give that — ``SeedSequence`` is what hashes them
    into independent streams.

    ``batch_id`` is the batch's ABSOLUTE starting iteration (see
    ``TrainingBatchCoordinator.run_batch``), never a 0-based loop counter. A
    per-leg counter would restart at 0 on resume and replay the deal stream,
    biasing the average strategy toward the re-sampled deals.
    """
    return int(np.random.SeedSequence([base_seed, worker_id, batch_id]).generate_state(1)[0])


def _worker_loop(
    worker_id: int,
    num_workers: int,
    session_id: str,
    config: Config,
    serialized_action_model: bytes,
    serialized_card_abstraction: bytes,
    base_seed: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    id_request_queues: list[mp.Queue],  # One queue per worker for ID requests
    id_response_queues: list[mp.Queue],  # One queue per worker for ID responses
    initial_capacity: int,
    max_actions: int,
    checkpoint_dir: str | None = None,
    ready_event=None,  # mp.Event | None
) -> None:
    """
    Worker process for parallel MCCFR training with shared array storage.

    OWNERSHIP MODEL:
    - Ownership determined by stable hash: owner(key) = xxhash(key) % num_workers
    - Only owner allocates IDs and creates key→ID mappings
    - Every worker writes regret/strategy updates directly to shared memory
      (lock-free, Hogwild-style) for every infoset whose global ID it knows
    - Non-owners get a read-only view into the UNKNOWN_ID region until the ID
      is learned (zeros = uniform strategy; updates are skipped)
    - Non-owners request IDs from owners (batched, async)

    Args:
        worker_id: This worker's ID (0 to num_workers-1)
        num_workers: Total number of workers
        session_id: Unique session ID for shared memory namespace
        config: Configuration object
        serialized_action_model: Pickled ActionModel
        serialized_card_abstraction: Pickled BucketingStrategy
        base_seed: Base random seed
        job_queue: Queue to receive jobs from coordinator
        result_queue: Queue to return results to coordinator
        id_request_queues: Per-worker queues for ID requests
        id_response_queues: Per-worker queues for ID responses
        initial_capacity: Initial capacity for infoset storage
        max_actions: Maximum actions per infoset
        checkpoint_dir: Optional checkpoint directory
    """
    try:
        # Let the coordinator handle Ctrl+C; workers should exit via shutdown.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        print(
            f"[Worker {worker_id}] Initializing (attaching to shared memory)...",
            file=sys.stderr,
            flush=True,
        )

        # Deserialize abstractions
        action_model = pickle.loads(serialized_action_model)
        card_abstraction = pickle.loads(serialized_card_abstraction)

        # Create storage (attach to existing shared memory)
        # ready_event ensures we wait for coordinator to create memory first
        storage = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=worker_id,
            session_id=session_id,
            initial_capacity=initial_capacity,
            max_actions=max_actions,
            is_coordinator=False,  # Worker attaches, doesn't create
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            ready_event=ready_event,
            zarr_compression_level=config.storage.zarr_compression_level,
            zarr_chunk_size=config.storage.zarr_chunk_size,
        )

        # Create solver config with worker-specific seed
        # Each worker needs a unique seed to explore different parts of the game tree
        worker_seed = _compute_seed(base_seed, worker_id)
        worker_config = config.merge({"system": {"seed": worker_seed}})

        # Create solver with shared array storage
        solver = MCCFRSolver(
            action_model=action_model,
            card_abstraction=card_abstraction,
            storage=storage,
            config=worker_config,
        )

        print(
            f"[Worker {worker_id}] Ready (id_range=[{storage.id_range_start}, {storage.id_range_end}))",
            file=sys.stderr,
            flush=True,
        )

        # Worker loop
        batch_count = 0
        while True:
            # Process all pending messages and flush outbound ID requests
            _process_all_messages(
                worker_id,
                id_request_queues[worker_id],
                id_response_queues[worker_id],
                id_response_queues,
                storage,
            )
            _send_pending_id_requests(worker_id, id_request_queues, storage)

            try:
                job = job_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            job_type = JobType(job["type"])

            # Targeted jobs share one queue; requeue those addressed to another
            # worker. Without this, one worker can consume another's job — for
            # RESIZE_STORAGE that leaves the starved worker attached to unlinked
            # memory, for EXCHANGE_IDS its pending ID requests never flush, and
            # for RUN_ITERATIONS the same (worker_id, batch_id) seed runs twice.
            target_worker = job.get("target_worker")
            if target_worker is not None and target_worker != worker_id:
                job_queue.put(job)
                continue

            if job_type == JobType.SHUTDOWN:
                break

            elif job_type == JobType.EXCHANGE_IDS:
                storage.rearm_unresolved_id_requests()
                _send_pending_id_requests(worker_id, id_request_queues, storage)

                # Process any incoming requests/responses
                _process_all_messages(
                    worker_id,
                    id_request_queues[worker_id],
                    id_response_queues[worker_id],
                    id_response_queues,
                    storage,
                )

                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "type": "exchange_ids_ack",
                        "num_owned": storage.num_owned_infosets(),
                    }
                )

            elif job_type == JobType.COLLECT_KEYS:
                # Collect owned keys for coordinator checkpointing
                owned_keys = dict(storage.state.owned_keys)
                legal_actions_cache = dict(storage.state.legal_actions_cache)

                # Defensive: ensure this worker hasn't assigned duplicate IDs
                ids = list(owned_keys.values())
                if len(set(ids)) != len(ids):
                    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
                    raise RuntimeError(
                        f"Worker {worker_id} has duplicate infoset IDs in owned_keys; "
                        f"examples: {dup_ids[:5]}"
                    )

                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "type": "keys_collected",
                        "owned_keys": owned_keys,
                        "legal_actions_cache": legal_actions_cache,
                        "id_range_start": storage.id_range_start,
                        "id_range_end": storage.id_range_end,
                        "next_local_id": storage.state.next_local_id,
                    }
                )

            elif job_type == JobType.RESIZE_STORAGE:
                # Stop-the-world storage resize
                # Coordinator has already resized; workers need to reattach
                new_session_id = job["new_session_id"]
                new_capacity = job["new_capacity"]

                print(
                    f"[Worker {worker_id}] Reattaching after resize "
                    f"(new_max={new_capacity:,}, session={new_session_id})",
                    file=sys.stderr,
                    flush=True,
                )

                # Preserve owned keys and next_local_id before reattach
                preserved_keys = dict(storage.state.owned_keys)
                preserved_next_id = storage.state.next_local_id

                # Reattach to new shared memory
                storage.reattach_after_resize(
                    new_session_id=new_session_id,
                    new_capacity=new_capacity,
                    preserved_keys=preserved_keys,
                    preserved_next_id=preserved_next_id,
                )

                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "type": "resize_ack",
                        "new_capacity": new_capacity,
                        "new_id_range": (storage.id_range_start, storage.id_range_end),
                    }
                )

            elif job_type == JobType.RUN_ITERATIONS:
                num_iterations = job["num_iterations"]
                batch_id = job.get("batch_id", batch_count)
                iteration_offset = job.get("iteration_offset", 0)

                # Update solver state
                solver.iteration = iteration_offset

                # Update seed for this batch
                batch_seed = _compute_seed(base_seed, worker_id, batch_id)
                random.seed(batch_seed)
                np.random.seed(batch_seed)

                utilities = []
                iter_start = time.time()

                try:
                    for i in range(num_iterations):
                        # Periodically process incoming messages and flush
                        # outbound ID requests, so remote infoset IDs propagate
                        # within a batch instead of only at batch boundaries.
                        if i % 50 == 0:
                            _process_all_messages(
                                worker_id,
                                id_request_queues[worker_id],
                                id_response_queues[worker_id],
                                id_response_queues,
                                storage,
                            )
                            _send_pending_id_requests(worker_id, id_request_queues, storage)

                        util = solver.train_iteration()
                        utilities.append(util)

                    iter_time = time.time() - iter_start

                    result_queue.put(
                        {
                            "worker_id": worker_id,
                            "batch_id": batch_id,
                            "type": "iterations_done",
                            "utilities": utilities,
                            "num_owned_infosets": storage.num_owned_infosets(),
                            "capacity_usage": storage.get_capacity_usage(),
                            "iter_time": iter_time,
                        }
                    )

                except Exception as e:
                    print(
                        f"[Worker {worker_id}] Error: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exc()
                    result_queue.put(
                        {
                            "worker_id": worker_id,
                            "batch_id": batch_id,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )

                batch_count += 1

        # Discard undelivered cross-worker messages: peers are shutting down too,
        # so nobody drains these queues, and a feeder thread with buffered data
        # blocks process exit until the coordinator's join timeout force-kills us
        # (worker_count x 10s of pure shutdown wall-clock). The result queue is
        # left alone so genuine results/errors always flush.
        for q in (job_queue, *id_request_queues, *id_response_queues):
            q.cancel_join_thread()

        # Cleanup (just close handles, don't unlink - coordinator does that)
        storage.cleanup()

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        try:
            result_queue.put({"worker_id": worker_id, "error": str(e)}, timeout=5)
        except Exception:
            pass
