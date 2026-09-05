"""Writing the legacy `run.jsonl` that some tests still need to READ.

Production stopped writing this file when the record became Postgres, and the
writer was deleted with it. The READ path stays, because the share still holds
these files for every run published before the flip -- so the fixtures that
exercise it need a writer, and it belongs here rather than in `src/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.shared import records, run_events

if TYPE_CHECKING:
    import os


def append_event(run_dir: str | os.PathLike[str], event: str, **fields: Any) -> None:
    """Append one row to a legacy run log, as the retired writer did."""
    records.append_log(
        run_events.log_path(run_dir),
        {run_events.EVENT_KEY: event, **fields},
        records.REGISTRY[run_events.RUN_LOG_FILENAME],
    )
