"""Where a run's record goes, declared without saying what implements it.

`RunTracker` writes through this. The concrete sink -- Postgres today -- is
constructed at the composition root and injected, so `pipeline` never learns
what a database is and the layering contract holds without an exemption.

THE ASYMMETRY IS THE DESIGN. `emit` must not block and must not raise; `claim`
may do both. That is not fussiness about latency: it is the difference between
an event whose loss costs a progress bar and one whose loss costs a run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


class RecordSink(Protocol):
    """One run's record, as the trainer produces it."""

    def opened(self, run_id: str, body: Mapping[str, Any]) -> None:
        """Record that a run EXISTS, with the config that identifies it.

        Blocks, and is allowed to raise. This is the one event whose loss is
        unrecoverable: it carries the resolved config, the abstraction hash and
        the kernel, and a resume reads all three to decide it may continue.
        Without it a later task mints fresh metadata over a live ladder, trains
        from zero and appends mixed-lineage rungs -- so a run that cannot record
        its own existence must fail here rather than proceed quietly.
        """

    def emit(self, run_id: str, event: str, body: Mapping[str, Any]) -> None:
        """Record one ordinary event -- progress, an attempt boundary, a status.

        MUST NOT BLOCK and MUST NOT RAISE. A sink that stalls puts a network
        round trip inside the training loop; one that raises turns a transport
        fault into a dead task. Dropping is the correct failure: these are
        telemetry, and the trainer's job is to keep training.
        """

    def claim(self, run_id: str, iteration: int, uri: str) -> None:
        """Claim that a rung exists at `uri`.

        May block. Losing it strands bytes nothing references -- recoverable,
        because a prefix listing of the object store is ground truth and a
        reconcile can rebuild the row. That is the whole reason one rung is one
        atomically-committed object: the store's own commit IS the claim, and
        this row is a cache of it.
        """

    def flush(self, timeout: float) -> bool:
        """Drain before exit. False when rows were dropped.

        Returns rather than raises, so the caller can SAY SO in the run log
        instead of reporting a clean finish over a lossy one.
        """
