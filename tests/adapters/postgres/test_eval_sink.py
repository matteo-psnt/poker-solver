"""The eval sink's policy, INVERTED with the retirement of the share's copy.

It used to swallow everything, and the justification was explicit: the document
was already on the share, so a lost row cost an import. Now the row IS the
evaluation. An eval that ran for hours and recorded nothing is worse than one
that says so, so it retries and then raises.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.adapters.postgres import evals

DOCUMENT = {
    "run_id": "run-a",
    "method": "exact_br",
    "estimator": "exact_br",
    "checkpoint_iteration": 150_000_000,
    "timestamp": "2026-09-01T00:00:00+00:00",
    "knobs": {"base_seed": 7, "num_hands": 4000},
    "results": {"exploitability_mbb": 854.0, "std_error_mbb": 12.5, "num_hands": 4000},
}


class _Exploding:
    def begin(self) -> Any:
        raise RuntimeError("database is gone")


class _Recording:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    def begin(self) -> _Recording:
        return self

    def __enter__(self) -> _Recording:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: Any) -> None:
        self.statements.append(statement)


def test_an_unreachable_database_raises(monkeypatch):
    """The row IS the evaluation now. Silence would mean hours of compute with
    no record and nothing saying so."""
    monkeypatch.setattr(evals, "BACKOFF_SECONDS", 0)
    with pytest.raises(RuntimeError, match="database is gone"):
        evals.PostgresEvalSink(_Exploding()).scored("e-1", "run-a", DOCUMENT, "digest")


def test_a_blip_is_retried_before_it_is_believed(monkeypatch):
    """A dropped connection is not a lost evaluation. The write is idempotent --
    `eval_id` is unique by construction and the insert ignores a conflict -- so
    retrying after an ambiguous failure cannot double-write."""
    monkeypatch.setattr(evals, "BACKOFF_SECONDS", 0)

    class _FlakyThenFine(_Recording):
        def __init__(self) -> None:
            super().__init__()
            self.tries = 0

        def begin(self):
            self.tries += 1
            if self.tries == 1:
                raise RuntimeError("connection reset")
            return self

    engine = _FlakyThenFine()
    evals.PostgresEvalSink(engine).scored("e-1", "run-a", DOCUMENT, "digest")
    assert engine.tries == 2
    assert len(engine.statements) == 1


def test_a_reachable_database_gets_one_statement():
    engine = _Recording()
    evals.PostgresEvalSink(engine).scored("e-1", "run-a", DOCUMENT, "digest")
    assert len(engine.statements) == 1


class TestTheRowIsBuiltONCE:
    """`eval_values` is the only place a document becomes columns.

    It began as the rule two writers had to share. The importer is gone and
    these hold for a stronger reason: the sink is the ONLY record of an eval, so
    a column read wrong here is not a divergence anyone can compare away -- it
    is the measurement, stored wrong, with nothing left to check it against.
    """

    def test_the_seed_comes_from_base_seed(self):
        """`board_seed` is not where the seed lives, and reading the wrong one
        gives every eval a null seed without failing."""
        assert evals.eval_values("e-1", "run-a", DOCUMENT, "d")["base_seed"] == 7

    def test_the_digest_is_passed_in_not_derived(self):
        """`adapters` may not import `pipeline`, and a second derivation of the
        tier would pair rows that must not be compared -- a five-column version
        reported -100.0 mbb where the truth was -60.0."""
        assert evals.eval_values("e-1", "run-a", DOCUMENT, "carried")["tier_digest"] == "carried"

    def test_the_whole_document_is_kept_as_the_payload(self):
        assert evals.eval_values("e-1", "run-a", DOCUMENT, "d")["payload"] == DOCUMENT

    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            ("run_id", "run-a"),
            ("method", "exact_br"),
            ("checkpoint_iteration", 150_000_000),
            ("exploitability_mbb", 854.0),
            ("num_hands", 4000),
            ("recorded_at", "2026-09-01T00:00:00+00:00"),
        ],
    )
    def test_every_column_the_readers_need(self, column, expected):
        assert evals.eval_values("e-1", "run-a", DOCUMENT, "d")[column] == expected

    def test_method_falls_back_to_estimator(self):
        """Older documents carry only `estimator`, and a blank method sorts an
        eval into a tier of its own."""
        document = {**DOCUMENT, "method": None}
        assert evals.eval_values("e-1", "run-a", document, "d")["method"] == "exact_br"

    def test_the_run_id_is_the_directorys_not_the_documents(self):
        """The importer reads it from the directory the document sits in, so the
        live writer must too. Taking it from the payload let the two writers
        disagree about which run an eval belongs to -- invisible while they
        happen to match, which they do on all 2,139 rows today."""
        document = {**DOCUMENT, "run_id": "some-other-run"}
        assert evals.eval_values("e-1", "run-a", document, "d")["run_id"] == "run-a"


def test_the_document_the_sink_receives_is_the_one_the_share_holds():
    """`write_snapshot` stamps on the way out and returns nothing, so the
    UNSTAMPED dict was what every caller got -- and the sink stored it, leaving
    37 evals whose database row had no `schema_version` while the file it
    mirrors did. One eval, two answers, and only a full-content comparison
    finds it: the counts agreed at 2,238 both ways.
    """
    from src.shared import records

    stamped = records.stamp(dict(DOCUMENT), records.REGISTRY["evals/*.json"])
    assert stamped["schema_version"] == records.REGISTRY["evals/*.json"].version
    # What `record_evaluation` now returns, and therefore what is stored.
    assert evals.eval_values("e-1", "run-a", stamped, "d")["payload"]["schema_version"] == (
        records.REGISTRY["evals/*.json"].version
    )
