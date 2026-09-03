"""What a resume is handed must be the LINE the log holds, not the row.

`event` is a column in the database and a key in `run.jsonl`, and the fold looks
it up by key. Handing over the body as stored gives a run with no `created`
event in it -- and a resume that cannot find one raises rather than continuing.

Measured on a node, which is the only place it showed: every unit test fed the
source events read straight off the FILE, where the key is already there.

    ValueError: run log has no `created` event
"""

from __future__ import annotations

from typing import Any

from src.adapters.postgres import queries


class _Rows:
    """An engine returning rows the way `run_events` stores them."""

    def __init__(self, rows: list[tuple[str, dict[str, Any] | None]]) -> None:
        self._rows = rows

    def connect(self) -> _Rows:
        return self

    def execution_options(self, **_options: Any) -> _Rows:
        return self

    def __enter__(self) -> _Rows:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(
        self, _statement: Any, _params: Any = None
    ) -> list[tuple[str, dict[str, Any] | None]]:
        return self._rows


def test_the_event_column_comes_back_as_a_key():
    engine = _Rows([("created", {"ts": "t", "config_name": "quick_test"})])
    (event,) = queries.run_event_bodies(engine, "run-a")
    assert event["event"] == "created"
    assert event["config_name"] == "quick_test"


def test_a_body_that_already_carries_it_is_not_broken():
    """A body imported from the file keeps its own `event`; the column agrees."""
    engine = _Rows([("status", {"event": "status", "status": "completed"})])
    (event,) = queries.run_event_bodies(engine, "run-a")
    assert event["event"] == "status"


def test_a_null_body_still_yields_the_kind():
    """The column is what a fold keys on; a row with no body is still an event
    of some kind, and dropping it would silently shorten the run's history."""
    engine = _Rows([("created", None)])
    (event,) = queries.run_event_bodies(engine, "run-a")
    assert event == {"event": "created"}


def test_the_fold_can_find_created_in_what_comes_back():
    """The end-to-end claim, without a database: what the source hands over is
    foldable."""
    from src.pipeline.training.run_tracker.metadata import RunMetadata

    engine = _Rows(
        [
            (
                "created",
                {
                    "ts": "2026-09-01T00:00:00+00:00",
                    "config_name": "quick_test",
                    "config": {"system": {"config_name": "quick_test"}},
                    "action_config_hash": "abc123",
                },
            ),
        ]
    )
    metadata = RunMetadata.from_events(queries.run_event_bodies(engine, "run-a"))
    assert metadata.config_name == "quick_test"
