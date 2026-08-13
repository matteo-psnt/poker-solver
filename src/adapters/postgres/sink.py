"""The Postgres implementation of `RecordSink`.

Two paths on purpose, and the split is the whole point. `opened` and `claim` go
straight to the database and are allowed to fail loudly. `emit` goes onto a
bounded queue drained by one background thread, and drops rather than waits.

Nothing here is imported by `pipeline`: the trainer holds a `RecordSink`, and
`commands` decides this is the one. `the_work_does_not_know_its_adapters`
enforces that rather than trusting it.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import update as sa_update
from sqlalchemy.dialects.postgresql import insert

from src.adapters.postgres import models

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)


class Transactional(Protocol):
    """What this sink needs of an engine, which is very little.

    Structural rather than `sqlalchemy.Engine` so the failure tests can inject
    one that hangs or explodes. Demanding the concrete class would have made the
    only interesting cases -- a stalled writer, an unreachable database --
    untestable without a real outage.
    """

    def begin(self) -> Any: ...


# Deep enough that a minute of network trouble costs nothing at a few events a
# second, shallow enough that it cannot grow into the memory a 32M-row table
# needs. Measured for scale: even naive per-row inserts run at ~10,000/s, so
# this is sized against an OUTAGE, not against throughput.
QUEUE_DEPTH = 10_000

# One round trip per batch rather than per event. `executemany` measured at
# 40,118 rows/s against 10,468 for single inserts -- but the reason to batch is
# not speed, it is holding one connection briefly instead of constantly.
BATCH = 200

_SENTINEL = object()


class PostgresSink:
    """A `RecordSink` backed by Postgres. Start it, use it, close it."""

    def __init__(self, engine: Transactional, *, queue_depth: int = QUEUE_DEPTH) -> None:
        self._engine = engine
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_depth)
        self._dropped = 0
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._drain, name="record-sink", daemon=True)
        self._worker.start()

    # -- the blocking half ------------------------------------------------
    def opened(self, run_id: str, body: Mapping[str, Any]) -> None:
        """Synchronous, and raising is correct: see the port's docstring."""
        with self._engine.begin() as connection:
            connection.execute(
                insert(models.Run)
                .values(_run_values(run_id, body))
                .on_conflict_do_update(
                    index_elements=["run_id"],
                    set_={"status": "running", "config": dict(body.get("config") or {})},
                )
            )
        self._put(run_id, "created", body, blocking=True)

    def closed(self, run_id: str, status: str, body: Mapping[str, Any]) -> None:
        """Fold the terminal status onto the run row, synchronously."""
        with self._engine.begin() as connection:
            connection.execute(
                sa_update(models.Run)
                .where(models.Run.run_id == run_id)
                .values(status=status, completed_at=body.get("completed_at"))
            )
        self._put(run_id, "status", body, blocking=True)

    def claim(self, run_id: str, iteration: int, uri: str) -> None:
        """Claim a rung, and make it THE current one.

        Two statements, and the first is not optional: `is_current` is a PARTIAL
        UNIQUE index over one row per run, so claiming a second rung while the
        first still holds the flag violates it. `claim` is allowed to raise, so
        that killed the run at its second checkpoint -- measured, and invisible
        until then because a run with one rung works.

        The clear must come FIRST. Inserting and then clearing would either trip
        the same index or, if the insert lost its flag to `DO NOTHING`, leave the
        run pointing at a stale rung -- which is worse than an error, because a
        loader resolves it and starts from the wrong place.
        """
        with self._engine.begin() as connection:
            connection.execute(
                sa_update(models.Checkpoint)
                .where(models.Checkpoint.run_id == run_id, models.Checkpoint.is_current)
                .values(is_current=False)
            )
            connection.execute(
                insert(models.Checkpoint)
                .values(
                    run_id=run_id,
                    iteration=iteration,
                    blob_uri=uri,
                    fingerprint="",
                    is_current=True,
                )
                .on_conflict_do_update(
                    index_elements=["run_id", "iteration"],
                    # `is_current` too: re-claiming an EARLIER rung is how a
                    # resume points the run back at where it restarts from.
                    set_={"blob_uri": uri, "is_current": True},
                )
            )

    # -- the lossy half ---------------------------------------------------
    def emit(self, run_id: str, event: str, body: Mapping[str, Any]) -> None:
        """Never blocks, never raises. A full queue drops and counts."""
        self._put(run_id, event, body, blocking=False)

    def flush(self, timeout: float) -> bool:
        """Drain, and report whether anything was lost along the way."""
        deadline = timeout
        try:
            self._queue.join()
        except Exception:  # noqa: BLE001 -- flush reports, it does not fail a run
            log.warning("record sink did not drain within %.1fs", deadline)
        with self._lock:
            return self._dropped == 0

    def close(self, timeout: float = 5.0) -> bool:
        drained = self.flush(timeout)
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=timeout)
        return drained

    # -- internals --------------------------------------------------------
    def _put(self, run_id: str, event: str, body: Mapping[str, Any], *, blocking: bool) -> None:
        row = {
            "run_id": run_id,
            "attempt": int(body.get("attempt") or body.get("index") or 0),
            "event": event,
            "at": body.get("ts") or datetime.now(UTC).isoformat(),
            "body": dict(body),
            # Client-assigned and random rather than a counter: two processes
            # write one run's events -- the node wrapper and the trainer -- and
            # a monotone `seq` would collide between them, which the unique
            # index would then reject and this sink would swallow.
            "event_uuid": uuid.uuid4(),
        }
        try:
            if blocking:
                self._queue.put(row)
            else:
                self._queue.put_nowait(row)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            # Once per outage, not once per event: a full queue means the
            # database is unreachable, and logging 10,000 lines about it is its
            # own failure.
            if self._dropped == 1:
                log.warning("record sink queue full; dropping events until it drains")

    def _drain(self) -> None:
        batch: list[dict[str, Any]] = []
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                self._write(batch)
                self._queue.task_done()
                return
            batch.append(item)
            done = 1
            while len(batch) < BATCH:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is _SENTINEL:
                    self._write(batch)
                    self._queue.task_done()
                    for _ in range(done):
                        self._queue.task_done()
                    return
                batch.append(nxt)
                done += 1
            self._write(batch)
            batch = []
            for _ in range(done):
                self._queue.task_done()

    def _write(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            with self._engine.begin() as connection:
                connection.execute(insert(models.RunEvent).values(batch).on_conflict_do_nothing())
                for run_id, values in _folded(batch).items():
                    connection.execute(
                        sa_update(models.Run).where(models.Run.run_id == run_id).values(**values)
                    )
        except Exception:
            with self._lock:
                self._dropped += len(batch)
            log.warning("record sink lost %d events", len(batch), exc_info=True)


def _folded(batch: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The counters a batch of progress events moves, per run.

    Folded HERE, in the same transaction as the events, because `runs` is a
    cache of the fold and a cache nothing maintains is just a stale number: a
    completed 200,000-iteration run read as 0 iterations in 0 seconds while its
    own events said otherwise.

    Best-effort on purpose, unlike `status`. These ride the lossy path, so a
    dropped batch leaves the counter behind until the next one -- which is the
    right trade for a progress reading and the wrong one for a terminal state.
    MAX rather than last-wins, so an out-of-order batch cannot walk it back.
    """
    folded: dict[str, dict[str, Any]] = {}
    for row in batch:
        body = row.get("body") or {}
        iterations = body.get("iterations")
        if iterations is None:
            continue
        current = folded.setdefault(row["run_id"], {})
        current["iterations"] = max(int(iterations), int(current.get("iterations", 0)))
        runtime = body.get("attempt_runtime_seconds") or body.get("runtime_seconds")
        if runtime is not None:
            current["runtime_seconds"] = max(
                float(runtime), float(current.get("runtime_seconds", 0.0))
            )
        infosets = body.get("num_infosets")
        if infosets is not None:
            current["num_infosets"] = int(infosets)
    return folded


def _run_values(run_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """The `runs` row a `created` event describes."""
    return {
        "run_id": run_id,
        "config_name": str(body.get("config_name") or ""),
        "kernel": body.get("kernel"),
        "arm": body.get("arm"),
        "experiment_id": body.get("experiment_id"),
        "parent_run_id": body.get("parent_run_id"),
        "action_config_hash": body.get("action_config_hash"),
        "card_abstraction_hash": body.get("card_abstraction_hash"),
        "config_hash": body.get("config_hash"),
        "config": dict(body.get("config") or {}),
        "git_commit": body.get("git_commit"),
        "git_dirty": body.get("git_dirty"),
        "git_branch": body.get("git_branch"),
        "code_snapshot": body.get("code_snapshot"),
        "storage_capacity": body.get("storage_capacity"),
        "started_at": body.get("started_at") or datetime.now(UTC).isoformat(),
        "status": "running",
    }
