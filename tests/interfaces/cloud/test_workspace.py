"""Answering a question against the published record, without keeping a copy."""

from __future__ import annotations

import json

import pytest

from src.interfaces.cloud import share, workspace
from src.interfaces.errors import CommandError


class _FakeShare:
    """A share as a dict of path -> bytes.

    Stands in for ShareServiceClient at the two seams workspace uses --
    ``list_entries``/``walk_files`` to discover and ``download_file`` to pull --
    so the materialiser is tested without an Azure account.
    """

    def __init__(self, files: dict[str, str]):
        self.files = files
        self.written: dict[str, str] = {}


@pytest.fixture
def fake(monkeypatch):
    store = _FakeShare(
        {
            "archive/run-a/run.jsonl": json.dumps({"event": "created", "run_id": "run-a"}) + "\n",
            "archive/run-a/STATIC_CHECKPOINT.json": json.dumps({"iteration": 1000}),
            "archive/run-a/evals/slug1.json": json.dumps({"run_id": "run-a"}),
            "archive/run-a/static-1000.zarr/0.0": "BULK",
            "archive/run-b/run.jsonl": json.dumps({"event": "created", "run_id": "run-b"}) + "\n",
        }
    )

    def walk_files(service, share_name, path, *, skip_dir=None):
        prefix = f"{path}/"
        found = []
        for p in service.files:
            if not p.startswith(prefix):
                continue
            parts = p[len(prefix) :].split("/")
            if skip_dir is not None and any(skip_dir(part) for part in parts[:-1]):
                continue
            found.append(p)
        return found

    def list_entries(service, share_name, path):
        names = set()
        prefix = f"{path}/"
        for p in service.files:
            if p.startswith(prefix):
                rest = p[len(prefix) :]
                names.add((rest.split("/")[0], "/" in rest))
        return [share.ShareEntry(name=n, is_directory=d, size=0) for n, d in sorted(names)]

    def download_file(service, share_name, path, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(service.files[path])

    def read_text(service, share_name, path):
        return service.files.get(path)

    def write_text(service, share_name, path, body):
        service.written[path] = body
        service.files[path] = body

    monkeypatch.setattr(share, "walk_files", walk_files)
    monkeypatch.setattr(share, "list_entries", list_entries)
    monkeypatch.setattr(share, "download_file", download_file)
    monkeypatch.setattr(share, "read_text", read_text)
    monkeypatch.setattr(share, "write_text", write_text)
    return store


class TestPullMetadata:
    def test_pulls_the_json_record(self, fake, tmp_path):
        workspace.pull_metadata(fake, "s", tmp_path)
        assert (tmp_path / "run-a" / "run.jsonl").is_file()
        assert (tmp_path / "run-a" / "evals" / "slug1.json").is_file()
        assert (tmp_path / "run-b" / "run.jsonl").is_file()

    def test_never_pulls_checkpoint_data(self, fake, tmp_path):
        """~540 MB of zarr chunks that no reading command opens."""
        workspace.pull_metadata(fake, "s", tmp_path)
        assert not list(tmp_path.rglob("*.zarr*"))

    def test_one_run_pulls_only_that_run(self, fake, tmp_path):
        workspace.pull_metadata(fake, "s", tmp_path, run="run-a")
        assert (tmp_path / "run-a").is_dir()
        assert not (tmp_path / "run-b").exists()

    def test_an_unpublished_run_says_what_is_published(self, fake, tmp_path):
        with pytest.raises(CommandError, match="run-a"):
            workspace.pull_metadata(fake, "s", tmp_path, run="run-nope")

    def test_the_local_tree_mirrors_the_published_one(self, fake, tmp_path):
        """The readers are ordinary local-path code; the layout must match."""
        workspace.pull_metadata(fake, "s", tmp_path)
        loaded = json.loads((tmp_path / "run-a" / "run.jsonl").read_text())
        assert loaded["run_id"] == "run-a"


class TestBaseline:
    def test_round_trips_through_the_share(self, fake):
        workspace.write_baseline(fake, "s", json.dumps({"run_id": "run-a"}))
        body = workspace.read_baseline(fake, "s")
        assert body is not None
        assert json.loads(body)["run_id"] == "run-a"

    def test_absent_baseline_reads_as_none(self, fake):
        assert workspace.read_baseline(fake, "s") is None

    def test_it_lands_at_the_share_root_beside_the_archive(self, fake):
        workspace.write_baseline(fake, "s", "{}")
        assert workspace.BASELINE_NAME in fake.written
        assert "/" not in workspace.BASELINE_NAME


class TestSourceSeam:
    """There is only one source now: the published record."""

    def test_share_derives_the_index_rather_than_reading_a_shared_file(self, tmp_path):
        """A second writable file on a share with no atomic append is the
        contention the per-run records exist to remove."""
        import argparse

        from src.interfaces.cli.commands import _base

        args = argparse.Namespace(source="share", runs_dir="unused", ledger="unused.jsonl")
        derived = _base.ledger_for(args, tmp_path)
        assert derived.parent == tmp_path, "derived inside the materialised tree"
        assert derived.is_file(), "rebuild_ledger ran"


class TestTheWalkIsPruned:
    """Filtering checkpoint data out AFTER the walk still paid for the walk.

    A run's .zarr snapshots hold thousands of chunk files and listing them is a
    round trip per directory -- measured at 167s to pull 146 small JSON files.
    """

    def test_it_never_lists_inside_a_snapshot_directory(self, fake, tmp_path, monkeypatch):
        listed: list[str] = []
        original = share.list_entries

        def spy(service, share_name, path):
            listed.append(path)
            return original(service, share_name, path)

        monkeypatch.setattr(share, "list_entries", spy)
        workspace.pull_metadata(fake, "s", tmp_path)

        assert not [p for p in listed if ".zarr" in p], f"descended into a snapshot: {listed}"
