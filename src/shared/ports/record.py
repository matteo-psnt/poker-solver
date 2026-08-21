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

    def closed(self, run_id: str, status: str, body: Mapping[str, Any]) -> None:
        """Record that a run REACHED a terminal state.

        Blocks, and pairs with `opened`. It could ride the lossy path -- the
        event itself is already in the log -- but the folded `runs` row is what
        every listing reads and what decides whether a ladder may be pruned, so
        a dropped status leaves a finished run advertising itself as training.
        That is the zombie this migration exists partly to stop creating.
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


class EvalSink(Protocol):
    """One evaluation's result, as the scoring service produces it.

    Its own protocol rather than a method on `RecordSink`, because it has a
    different producer and a different life: an eval is one row written once by
    a service that has no run to open, close or flush.

    BEST EFFORT, and unlike `opened` it must not raise. The document is already
    on the share by the time this is called, and the share is the source of
    truth -- so a failure here costs a row the importer can rebuild, while
    raising would throw away hours of finished evaluation over a transport
    fault. `record_evaluation` already refuses to fail an eval it records; this
    holds the same line one layer down.
    """

    def scored(
        self, eval_id: str, run_id: str, document: Mapping[str, Any], tier_digest: str
    ) -> None:
        """Store one evaluation.

        `run_id` is the RUN DIRECTORY'S name, which is what the importer reads
        and therefore the answer both writers must agree on -- not the document's
        own field, which is written from a different source.

        `tier_digest` is computed by the CALLER, from
        `pipeline.evaluation.ledger.tiers` -- the one implementation of which
        knobs make two evals comparable. A sink cannot derive it: `adapters` may
        not import `pipeline`, and a second derivation would pair rows that must
        not be compared.
        """


class RecordSource(Protocol):
    """Where a run's record is READ from, declared without saying what holds it.

    The mirror of `RecordSink`, and it exists for one caller: a RESUME. The
    tracker folds a run's events to decide whether this task may continue --
    the config it was trained under, the abstraction it is pinned to, the
    kernel. Without that it mints fresh metadata over a live ladder and trains
    from zero, which is the failure `opened` calls unrecoverable.

    One method, because everything else folds from events. `RunMetadata` already
    knows how; what it lacked was anywhere but a file to fold from.
    """

    def events(self, run_id: str) -> list[Mapping[str, Any]]:
        """Every event of one run, oldest first.

        The order is the fold's order. `gseq` is arrival, and two processes
        write one run's events, so the reader sorts by what the events THEMSELVES
        say rather than by when they landed.
        """
