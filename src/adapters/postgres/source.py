"""The Postgres implementation of `RecordSource`.

Reads only. Its whole job is to hand a resume the events it used to fold out of
`run.jsonl`, so the tracker can decide whether this task may continue without
the file existing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.adapters.postgres import queries

if TYPE_CHECKING:
    from collections.abc import Mapping


class PostgresRecordSource:
    """A `RecordSource` backed by Postgres."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def events(self, run_id: str) -> list[Mapping[str, Any]]:
        """Every event of one run, oldest first.

        The stored body IS the line the log holds, so what the fold receives
        here is what it received from the file -- same keys, same absences.
        """
        return list(queries.run_event_bodies(self._engine, run_id))
