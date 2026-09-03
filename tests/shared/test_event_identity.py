"""One event has one id, wherever it is written.

The sink assigned `uuid4()` and the importer a positional `uuid5`, so they could
never agree: an event written live and then re-imported landed TWICE. Measured
on one task -- 6 events on the share, 11 in the database -- and 83 surplus rows
across the record before the identity was shared.

Counts hid it. `backfill-record --verify` reported "database ahead (a live run)",
which is the healthy direction and was also, here, duplication.
"""

from __future__ import annotations

from src.shared import run_events


def test_the_same_event_has_the_same_id():
    body = {"ts": "2026-09-01T00:00:00+00:00", "iterations": 5}
    assert run_events.event_identity("run-a", "progress", body) == (
        run_events.event_identity("run-a", "progress", dict(body))
    )


def test_the_file_and_the_sink_agree():
    """The file's line carries `schema_version` and `event`; the sink's body
    carries neither. An id that depended on either would disagree across the two
    writers for that reason alone -- which is exactly what duplicated rows."""
    from_sink = {"ts": "t", "iterations": 5}
    from_file = {"event": "progress", "schema_version": 3, "ts": "t", "iterations": 5}
    assert run_events.event_identity("run-a", "progress", from_sink) == (
        run_events.event_identity("run-a", "progress", from_file)
    )


def test_different_content_is_a_different_event():
    a = run_events.event_identity("run-a", "progress", {"ts": "t", "iterations": 5})
    b = run_events.event_identity("run-a", "progress", {"ts": "t", "iterations": 6})
    assert a != b


def test_different_runs_do_not_collide():
    body = {"ts": "t", "iterations": 5}
    assert run_events.event_identity("run-a", "progress", body) != (
        run_events.event_identity("run-b", "progress", body)
    )


def test_different_kinds_do_not_collide():
    body = {"ts": "t"}
    assert run_events.event_identity("run-a", "progress", body) != (
        run_events.event_identity("run-a", "checkpoint", body)
    )


def test_two_attempts_are_distinct():
    """The positional id this replaces existed to keep these apart. They differ
    in `index` and `ts`, so content-addressing keeps them apart too -- verified
    against the real record, where the only rows that collapsed were identical
    to the microsecond in every field."""
    a = {"ts": "2026-08-02T20:51:29+00:00", "index": 1, "status": "died"}
    b = {"ts": "2026-08-02T20:57:00+00:00", "index": 3, "status": "failed"}
    assert run_events.event_identity("run-a", "attempt_ended", a) != (
        run_events.event_identity("run-a", "attempt_ended", b)
    )


def test_it_is_stable_across_processes():
    """A fixed namespace, so the id does not depend on when or where it was
    computed -- the sink on a node and the importer on a laptop must land on the
    same value."""
    assert str(run_events.event_identity("run-a", "created", {"ts": "t"})) == (
        "c296f3ce-634d-5ca0-93fd-76e8c3751d50"
    )
