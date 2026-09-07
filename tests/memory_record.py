"""An in-memory `RecordSink` + `RecordSource`, for tests that need a record.

`run.jsonl` is no longer written, so a tracker with no sink records NOTHING --
deliberately, because a half-written run whose events go to a log nothing reads
is worse than one that says it has no record. Tests that used to assert against
the file assert against this instead, which exercises the port pair the
production path uses rather than a format nothing reads any more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class MemoryRecord:
    """Both halves of the record, held in a list."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []
        self.claims: list[tuple[str, int, str]] = []

    # -- the write side ---------------------------------------------------
    def opened(self, run_id: str, body: Any) -> None:
        self._add(run_id, "created", body)

    def closed(self, run_id: str, status: str, body: Any) -> None:
        self._add(run_id, "status", body)

    def emit(self, run_id: str, event: str, body: Any) -> None:
        self._add(run_id, event, body)

    def claim(self, run_id: str, iteration: int, uri: str) -> None:
        self.claims.append((run_id, iteration, uri))

    def flush(self, timeout: float) -> bool:
        return True

    # -- the read side ----------------------------------------------------
    def events(self, run_id: str) -> list[Mapping[str, Any]]:
        """As the adapter hands them over: the body WITH `event` merged in.

        That merge is the shape a resume needs and the one a real source has to
        reproduce -- it is a column in the database and a key in the fold. The
        body is otherwise handed back UNTOUCHED, `run_id` key included: the
        adapter's `run_id` COLUMN is an index, not a replacement for what the
        body carries, and `created` carries its own.
        """
        return [dict(body) for owner, body in self.rows if owner == run_id]

    def _add(self, run_id: str, event: str, body: Any) -> None:
        self.rows.append((run_id, {**dict(body), "event": event}))
