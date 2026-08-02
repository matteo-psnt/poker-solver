"""The one substrate every recorded artifact is written through."""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from src.shared import records

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def _read(path) -> dict:
    """read_snapshot returns None for a torn file; these cases wrote a good one."""
    payload = records.read_snapshot(path)
    assert payload is not None
    return payload


SNAPSHOT = records.REGISTRY["baseline.json"]
SHARE_SNAPSHOT = records.REGISTRY["legs/*.start.json"]
LOG = records.REGISTRY["run.jsonl"]


class TestSnapshot:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "a.json"
        records.write_snapshot(path, {"run_id": "r"}, SNAPSHOT)
        assert _read(path)["run_id"] == "r"

    def test_stamps_the_registry_version_first(self, tmp_path):
        path = tmp_path / "a.json"
        records.write_snapshot(path, {"run_id": "r"}, SNAPSHOT)
        assert next(iter(json.loads(path.read_text()))) == "schema_version"
        assert records.version_of(_read(path)) == SNAPSHOT.version

    def test_replaces_rather_than_appends(self, tmp_path):
        path = tmp_path / "a.json"
        records.write_snapshot(path, {"n": 1}, SNAPSHOT)
        records.write_snapshot(path, {"n": 2}, SNAPSHOT)
        assert _read(path)["n"] == 2

    def test_leaves_no_temporary_file_behind(self, tmp_path):
        records.write_snapshot(tmp_path / "a.json", {"n": 1}, SNAPSHOT)
        assert not list(tmp_path.glob("*.tmp"))

    def test_an_absent_file_reads_as_none(self, tmp_path):
        assert records.read_snapshot(tmp_path / "absent.json") is None

    def test_a_torn_file_reads_as_none_rather_than_raising(self, tmp_path):
        """The expected residue of a kill under the pre-atomic writers."""
        path = tmp_path / "a.json"
        path.write_text('{"run_id": "r"')
        assert records.read_snapshot(path) is None

    def test_a_non_object_document_reads_as_none(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_text("[1, 2, 3]")
        assert records.read_snapshot(path) is None

    def test_creates_missing_parents(self, tmp_path):
        records.write_snapshot(tmp_path / "deep" / "a.json", {"n": 1}, SNAPSHOT)
        assert (tmp_path / "deep" / "a.json").is_file()


class TestAtomicityFollowsTheDestination:
    """Not a preference: SMB has no atomic rename, local disk does."""

    def test_local_is_atomic(self):
        assert records.REGISTRY["run.jsonl"].atomic
        assert records.REGISTRY["baseline.json"].atomic

    def test_scope_has_exactly_the_two_values_that_change_behaviour(self):
        """A third value would describe where a file lives without changing how
        it is written, and nothing would read it."""
        assert {a.scope for a in records.REGISTRY.values()} == {"local", "share"}

    def test_share_scope_is_not(self):
        assert not records.REGISTRY["legs/*.start.json"].atomic

    def test_a_share_snapshot_is_written_directly(self, tmp_path):
        """No temp file: a rename it cannot rely on would be worse than none."""
        records.write_snapshot(tmp_path / "leg.json", {"task_id": "t"}, SHARE_SNAPSHOT)
        assert (tmp_path / "leg.json").is_file()
        assert not list(tmp_path.glob("*.tmp"))


class TestLog:
    def test_rows_accumulate(self, tmp_path):
        path = tmp_path / "a.jsonl"
        records.append_log(path, {"n": 1}, LOG)
        records.append_log(path, {"n": 2}, LOG)
        assert [r["n"] for r in records.read_log(path)] == [1, 2]

    def test_every_row_carries_the_version(self, tmp_path):
        path = tmp_path / "a.jsonl"
        records.append_log(path, {"n": 1}, LOG)
        assert records.version_of(records.read_log(path)[0]) == LOG.version

    def test_a_torn_final_line_does_not_lose_the_history(self, tmp_path):
        path = tmp_path / "a.jsonl"
        records.append_log(path, {"n": 1}, LOG)
        with path.open("a") as handle:
            handle.write('{"n": 2')
        assert [r["n"] for r in records.read_log(path)] == [1]

    def test_a_skipped_line_can_be_reported(self, tmp_path):
        """So a caller with a repair path of its own can name it."""
        path = tmp_path / "a.jsonl"
        records.append_log(path, {"n": 1}, LOG)
        with path.open("a") as handle:
            handle.write('{"n": 2')

        seen: list[int] = []
        records.read_log(path, on_bad_line=seen.append)
        assert seen == [2]

    def test_append_propagates_io_failure(self, tmp_path):
        """Callers have different, individually correct policies; choosing one
        here would break the other."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        with pytest.raises(NotADirectoryError):
            records.append_log(blocker / "sub" / "a.jsonl", {"n": 1}, LOG)

    def test_a_missing_log_reads_as_empty(self, tmp_path):
        assert records.read_log(tmp_path / "absent.jsonl") == []

    def test_version_span_reports_a_mixed_file(self, tmp_path):
        path = tmp_path / "a.jsonl"
        path.write_text(json.dumps({"n": 0}) + "\n")  # pre-versioning row
        records.append_log(path, {"n": 1}, LOG)
        assert records.version_span(records.read_log(path)) == (0, LOG.version)


class TestTheRegistryIsAuthoritative:
    """ "What does this project store" must be answerable by reading one list."""

    def test_every_artifact_declares_what_it_is_for(self):
        for artifact in records.REGISTRY.values():
            assert artifact.what, f"{artifact.name} has no description"
            assert artifact.kind in ("snapshot", "log")
            assert artifact.scope in ("local", "share")

    def test_keys_match_their_artifact_names(self):
        for key, artifact in records.REGISTRY.items():
            assert key == artifact.name

    def test_no_module_writes_a_json_artifact_outside_the_substrate(self):
        """The drift this replaced: six writers, six sets of decisions.

        A new `write_text(json.dumps(...))` is a seventh convention, and the
        whole point is that there are two.
        """
        offenders = []
        pattern = re.compile(r"write_text\(\s*json\.dumps|json\.dump\(")
        for path in SRC.rglob("*.py"):
            if path.name == "records.py":
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(SRC)}:{number}")
        assert not offenders, "these write JSON artifacts outside src.shared.records: " + ", ".join(
            offenders
        )

    def test_every_registered_artifact_is_actually_referenced(self):
        """A registry entry nothing writes is a claim the tree does not honour."""
        sources = "\n".join(p.read_text() for p in SRC.rglob("*.py"))
        for name in records.REGISTRY:
            # Glob entries are referenced by their suffix, not their whole name.
            needle = name.split("*")[-1] if "*" in name else name
            assert needle in sources, f"{name} is registered but never referenced"
